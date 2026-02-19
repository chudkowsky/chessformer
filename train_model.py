# Import required libraries
from chessformer import ChessTransformer, ChessTransformerV2
from chess_loader import get_dataloader, get_dataloader_v2
from grok_tracker import GrokTracker, gradfilter_ema
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import time
from datetime import timedelta

# Phase 2 diffusion hyperparameters
D_DIT = 256
DIT_NHEAD = 4
DIT_D_HID = 512
DIT_NLAYERS = 6
DIFF_T = 20
DIFF_HORIZON = 4

# Configuration and hyperparameters
single_run = True
elo = 2000
max_elo = 2000
START_MODEL = None
END_MODEL = f'{elo}_elo_pos_engine'
FULL_SET = True
NUM_POS = 1e6  # Dataset size
dropout = 0.1  # Dropout probability
num_epochs = 10
INCREMENTS = num_epochs
GAMMA = .9
SCALE = 1
# Changed: 512→1024 to better utilize GPU VRAM (54% → ~80%) with 25M param model
batch_size = int(512 / SCALE)
LR = {.5: 5e-4, 1: 1e-4, 2: 1e-5}
LR = LR[SCALE] if SCALE in LR else 1e-5
CLIP = .1
# orignal 512
d_model = 512
d_hid = d_model * 2
nhead = 8
nlayers = 12
inp_ntoken = 13
out_ntoken = 2
start_time = time.time()

# Check for GPU availability
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

if SCALE != 1:
    factor = d_hid / d_model
    d_model = int(d_model * SCALE)
    d_hid = int(d_model * factor)


def _compute_loss_v1(model, boards, target, loss_fn):
    output = model(boards)
    return loss_fn(output, target)


def _compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target):
    policy_logits, _promo_logits, wdl_pred, _ply_pred = model(boards, features)
    B = boards.shape[0]
    policy_loss = F.cross_entropy(policy_logits.reshape(B, -1), from_sq * 64 + to_sq)
    wdl_loss = -(wdl_target * torch.log(wdl_pred + 1e-8)).sum(dim=-1).mean()
    return policy_loss + 0.5 * wdl_loss


def train(
    model: str = "",
    patience: int = None,
    model_version: str = "v1",
    use_grokfast: bool = False,
    grokfast_alpha: float = 0.98,
    grokfast_lamb: float = 2.0,
    grok_log: str | None = None,
):
    # Model loading or initialization
    # Changed: weights_only=False + map_location for cross-device resume, supports both paths and filenames
    if model:
        model_path = model if '/' in model else f'models/{model}'
        if model_version == 'v2':
            checkpoint = torch.load(model_path, weights_only=False, map_location=device)
            m = ChessTransformerV2(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout)
            m.load_state_dict(checkpoint['state_dict'])
            model = m.to(device)
        else:
            model = torch.load(model_path, weights_only=False, map_location=device).to(device)
        print(f'Resuming from: {model_path}')
    else:
        if model_version == 'v2':
            model = ChessTransformerV2(d_model=d_model, nhead=nhead, d_hid=d_hid, nlayers=nlayers, dropout=dropout).to(device)
        else:
            model = ChessTransformer(inp_ntoken, out_ntoken, d_model, nhead, d_hid, nlayers, dropout=dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {END_MODEL} ({model_version}) | Dataset: {DATASET}')
    print(f'Parameters: {num_params:,} | Device: {device}')
    if use_grokfast:
        print(f'Grokfast: ON (alpha={grokfast_alpha}, lamb={grokfast_lamb})')
    print()

    # Grokking tracker + Grokfast state
    grok_log_path = grok_log or f'grok_{END_MODEL}.log'
    tracker = GrokTracker(model, log_path=grok_log_path)
    ema_grads = None  # Grokfast EMA state (initialized on first backward)

    # Loss Function and Optimizer setup
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(num_epochs / INCREMENTS), gamma=GAMMA)
    # Changed: AMP (mixed precision) only on CUDA - not supported on MPS/CPU
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # Data loaders for training and testing
    # Changed: num_workers=4 for parallel data loading (separate processes, no data race)
    if model_version == 'v2':
        dataloader, testloader = get_dataloader_v2(DATASET, batch_size=batch_size, num_workers=4, num_pos=NUM_POS)
    else:
        dataloader, testloader = get_dataloader(DATASET, batch_size=batch_size, num_workers=4, num_pos=NUM_POS)

    # Changed: patience tracks epochs without test loss improvement (early stopping)
    no_improve = 0

    # Training loop
    best_loss = None
    best_test_loss = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch, batch_data in enumerate(dataloader):
            # Changed: mixed precision - forward pass in float16 for ~2x speedup (CUDA only)
            if use_amp:
                with autocast("cuda"):
                    if model_version == 'v2':
                        boards, features, from_sq, to_sq, promo, wdl_target = [x.to(device) for x in batch_data]
                        loss = _compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target)
                    else:
                        boards, target = batch_data[0].to(device), batch_data[1].to(device)
                        loss = _compute_loss_v1(model, boards, target, loss_fn)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if use_grokfast:
                    ema_grads = gradfilter_ema(model, ema_grads, grokfast_alpha, grokfast_lamb)
                if CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                if model_version == 'v2':
                    boards, features, from_sq, to_sq, promo, wdl_target = [x.to(device) for x in batch_data]
                    loss = _compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target)
                else:
                    boards, target = batch_data[0].to(device), batch_data[1].to(device)
                    loss = _compute_loss_v1(model, boards, target, loss_fn)
                optimizer.zero_grad()
                loss.backward()
                if use_grokfast:
                    ema_grads = gradfilter_ema(model, ema_grads, grokfast_alpha, grokfast_lamb)
                if CLIP:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                optimizer.step()
            total_loss += loss.item()

            # Progress output
            log_interval = max(1, len(dataloader) // 10)
            if (batch + 1) % log_interval == 0:
                avg_loss = total_loss / (batch + 1)
                elapsed = str(timedelta(seconds=int(time.time() - start_time)))
                print(
                    f'Epoch {epoch}: Batch {batch + 1}/{len(dataloader)} '
                    f'({100 * (batch + 1) / len(dataloader):.0f}%) '
                    f'| Loss: {avg_loss:.4f} | {elapsed}')

        scheduler.step()
        avg_train_loss = total_loss / len(dataloader)
        if best_loss is None or avg_train_loss < best_loss:
            best_loss = avg_train_loss

        # Evaluation on test dataset
        model.eval()
        tot_test_loss = 0
        with torch.no_grad():
            for batch_data in testloader:
                if model_version == 'v2':
                    boards, features, from_sq, to_sq, promo, wdl_target = [x.to(device) for x in batch_data]
                    loss = _compute_loss_v2(model, boards, features, from_sq, to_sq, wdl_target)
                else:
                    boards, target = batch_data[0].to(device), batch_data[1].to(device)
                    loss = _compute_loss_v1(model, boards, target, loss_fn)
                tot_test_loss += loss.item()

        avg_test_loss = tot_test_loss / len(testloader)
        print(f'Epoch {epoch} done | Train loss: {avg_train_loss:.4f} | Test loss: {avg_test_loss:.4f}')
        tracker.log_epoch(epoch, avg_train_loss, avg_test_loss)

        # Save the best model
        if best_test_loss is None or tot_test_loss < best_test_loss:
            best_test_loss = tot_test_loss
            no_improve = 0
            if epoch >= min(num_epochs - 2, 3):
                if model_version == 'v2':
                    torch.save({
                        'version': 'v2',
                        'state_dict': model.state_dict(),
                        'config': {'d_model': d_model, 'nhead': nhead, 'd_hid': d_hid, 'nlayers': nlayers},
                    }, f'models/{END_MODEL}_v2.pth')
                else:
                    torch.save(model, f'models/{END_MODEL}.pth')
                print(f'  -> Saved model (best test loss)')
        else:
            no_improve += 1

        # Changed: early stopping - stop if test loss hasn't improved for `patience` epochs
        if patience and no_improve >= patience:
            print(f'\nEarly stopping: no improvement for {patience} epochs')
            break

    print(f'\nBest Training Loss: {best_loss:.4f}')
    print(f'Best Testing Loss: {best_test_loss / len(testloader):.4f}')
    tracker.close()


def _compute_diffusion_loss(model, ns, cur_board, cur_feat, fut_board, fut_feat):
    """Compute diffusion training loss (noise prediction MSE).

    1. Encode current + future boards with frozen backbone → latents
    2. Project future latent to d_dit → x_0
    3. Sample random timestep, add noise → x_t
    4. DiT predicts noise from (x_t, t, backbone_latent)
    5. Return MSE(predicted_noise, actual_noise)
    """
    # Encode with frozen backbone
    with torch.no_grad():
        backbone_latent = model.encode(cur_board, cur_feat)
        future_latent = model.encode(fut_board, fut_feat)

    # Project future latent to d_dit → x_0
    x_0 = model.latent_to_dit(future_latent)

    # Diffusion forward process
    B = cur_board.shape[0]
    t = torch.randint(0, ns.T, (B,), device=cur_board.device)
    noise = torch.randn_like(x_0)
    x_t = ns.q_sample(x_0, t, noise)

    # Predict noise
    predicted_noise = model.diffusion(x_t, t, backbone_latent)

    return F.mse_loss(predicted_noise, noise)


def train_diffusion(
    backbone_model: str,
    pgn_path: str,
    patience: int | None = None,
):
    """Phase 2: Train diffusion model on trajectory data.

    Loads a pre-trained V2 backbone, attaches a DiT diffusion model,
    freezes backbone weights, and trains the diffusion components to
    predict noise on future latent states.

    Args:
        backbone_model: Path to pre-trained V2 model checkpoint.
        pgn_path: Path to PGN file for extracting trajectories.
        patience: Early stopping patience (None = disabled).
    """
    from diffusion_model import ChessDiT
    from noise_schedule import CosineNoiseSchedule
    from trajectory_loader import get_trajectory_dataloader

    # 1. Load pre-trained V2 backbone
    model_path = backbone_model if '/' in backbone_model else f'models/{backbone_model}'
    checkpoint = torch.load(model_path, weights_only=False, map_location=device)
    cfg = checkpoint['config']
    model = ChessTransformerV2(**cfg, dropout=0.0)
    model.load_state_dict(checkpoint['state_dict'])
    print(f'Loaded backbone: {model_path}')

    # 2. Create DiT + noise schedule
    dit = ChessDiT(
        d_dit=D_DIT, d_model=cfg['d_model'],
        nhead=DIT_NHEAD, d_hid=DIT_D_HID, nlayers=DIT_NLAYERS, T=DIFF_T,
    )
    ns = CosineNoiseSchedule(T=DIFF_T)

    # 3. Attach diffusion to V2 model
    model.attach_diffusion(dit, ns, D_DIT)
    model = model.to(device)
    ns.to(device)

    # 4. Freeze backbone, train only diffusion components
    diffusion_keywords = {'diffusion', 'latent_to_dit', 'dit_to_latent'}
    for name, p in model.named_parameters():
        if not any(k in name for k in diffusion_keywords):
            p.requires_grad = False

    trainable = [p for p in model.parameters() if p.requires_grad]
    num_trainable = sum(p.numel() for p in trainable)
    num_total = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {num_total:,} total, {num_trainable:,} trainable (diffusion)')
    print(f'DiT config: d_dit={D_DIT}, layers={DIT_NLAYERS}, T={DIFF_T}')
    print(f'Device: {device}\n')

    # 5. Optimizer + scheduler
    optimizer = torch.optim.AdamW(trainable, lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, num_epochs // INCREMENTS), gamma=GAMMA,
    )
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    # 6. Load trajectory data
    print(f'Loading trajectories from: {pgn_path} (horizon={DIFF_HORIZON})')
    train_loader, test_loader = get_trajectory_dataloader(
        pgn_path, horizon=DIFF_HORIZON, batch_size=batch_size,
        max_trajectories=int(NUM_POS),
    )
    print(f'Train batches: {len(train_loader)}, Test batches: {len(test_loader)}\n')

    # 7. Training loop
    best_test_loss = None
    no_improve = 0

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0

        for batch_idx, (cur_board, cur_feat, fut_board, fut_feat) in enumerate(train_loader):
            cur_board = cur_board.to(device)
            cur_feat = cur_feat.to(device)
            fut_board = fut_board.to(device)
            fut_feat = fut_feat.to(device)

            if use_amp:
                with autocast("cuda"):
                    loss = _compute_diffusion_loss(
                        model, ns, cur_board, cur_feat, fut_board, fut_feat,
                    )
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if CLIP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable, CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = _compute_diffusion_loss(
                    model, ns, cur_board, cur_feat, fut_board, fut_feat,
                )
                optimizer.zero_grad()
                loss.backward()
                if CLIP:
                    torch.nn.utils.clip_grad_norm_(trainable, CLIP)
                optimizer.step()

            total_loss += loss.item()

            log_interval = max(1, len(train_loader) // 10)
            if (batch_idx + 1) % log_interval == 0:
                avg = total_loss / (batch_idx + 1)
                elapsed = str(timedelta(seconds=int(time.time() - start_time)))
                print(
                    f'Epoch {epoch}: Batch {batch_idx + 1}/{len(train_loader)} '
                    f'({100 * (batch_idx + 1) / len(train_loader):.0f}%) '
                    f'| Diff loss: {avg:.6f} | {elapsed}'
                )

        scheduler.step()
        avg_train = total_loss / len(train_loader)

        # Evaluation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for cur_board, cur_feat, fut_board, fut_feat in test_loader:
                cur_board = cur_board.to(device)
                cur_feat = cur_feat.to(device)
                fut_board = fut_board.to(device)
                fut_feat = fut_feat.to(device)
                loss = _compute_diffusion_loss(
                    model, ns, cur_board, cur_feat, fut_board, fut_feat,
                )
                test_loss += loss.item()

        avg_test = test_loss / len(test_loader)
        print(f'Epoch {epoch} | Train diff: {avg_train:.6f} | Test diff: {avg_test:.6f}')

        # Save best model
        if best_test_loss is None or test_loss < best_test_loss:
            best_test_loss = test_loss
            no_improve = 0
            if epoch >= min(num_epochs - 2, 3):
                torch.save({
                    'version': 'v2+diff',
                    'state_dict': model.state_dict(),
                    'config': cfg,
                    'diffusion_config': {
                        'd_dit': D_DIT, 'nhead': DIT_NHEAD,
                        'd_hid': DIT_D_HID, 'nlayers': DIT_NLAYERS, 'T': DIFF_T,
                    },
                }, f'models/{END_MODEL}_v2_diff.pth')
                print(f'  -> Saved model (best test diff loss)')
        else:
            no_improve += 1

        if patience and no_improve >= patience:
            print(f'\nEarly stopping: no improvement for {patience} epochs')
            break

    print(f'\nBest Test Diffusion Loss: {best_test_loss / len(test_loader):.6f}')


if __name__ == '__main__':
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('elo', nargs='?', type=int, default=elo, help='ELO rating filter')
    # Changed: --device flag to override auto-detection (auto/cuda/mps/cpu)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to train on (default: auto-detect)')
    # Changed: --dataset flag to override default dataset path
    parser.add_argument('--dataset', type=str, default=None,
                        help='Path to dataset file (default: full_datasets/elo_{elo}_pos.txt)')
    # Changed: --num-pos flag to override NUM_POS
    parser.add_argument('--num-pos', type=float, default=NUM_POS,
                        help=f'Number of positions to load (default: {NUM_POS:.0f})')
    # Changed: --resume loads existing model and continues training (fine-tuning)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to existing model to continue training (e.g. models/2000_elo_pos_engine.pth)')
    # Changed: --lr overrides learning rate (useful for fine-tuning with lower LR)
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate override (default: auto from SCALE)')
    # Changed: --epochs overrides num_epochs
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'Number of epochs (default: {num_epochs})')
    # Changed: --patience for early stopping - stops if test loss doesn't improve for N epochs
    parser.add_argument('--patience', type=int, default=None,
                        help='Early stopping: stop after N epochs without test loss improvement')
    # Changed: --batch-size to adjust for different VRAM sizes (1024 for 16GB, 512 for 12GB)
    parser.add_argument('--batch-size', type=int, default=None,
                        help=f'Batch size (default: {batch_size}, lower for less VRAM)')
    parser.add_argument('--model-version', type=str, default='v2', choices=['v1', 'v2'],
                        help='Model architecture version (default: v2)')
    parser.add_argument('--grokfast', action='store_true',
                        help='Enable Grokfast EMA gradient filter (accelerates grokking)')
    parser.add_argument('--grokfast-alpha', type=float, default=0.98,
                        help='Grokfast EMA decay (default: 0.98)')
    parser.add_argument('--grokfast-lamb', type=float, default=2.0,
                        help='Grokfast amplification factor (default: 2.0)')
    parser.add_argument('--grok-log', type=str, default=None,
                        help='Path to grokking metrics log file (default: grok_{elo}.log)')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Training phase: 1=supervised policy, 2=diffusion (default: 1)')
    parser.add_argument('--backbone-model', type=str, default=None,
                        help='Phase 2: path to pre-trained V2 model for backbone')
    parser.add_argument('--pgn', type=str, default=None,
                        help='Phase 2: path to PGN file for trajectory extraction')
    parser.add_argument('--horizon', type=int, default=DIFF_HORIZON,
                        help=f'Phase 2: trajectory horizon in half-moves (default: {DIFF_HORIZON})')
    args = parser.parse_args()
    elo = args.elo
    max_elo = elo
    NUM_POS = args.num_pos
    if args.device != 'auto':
        device = torch.device(args.device)
    if args.lr is not None:
        LR = args.lr
    if args.epochs is not None:
        num_epochs = args.epochs
        INCREMENTS = num_epochs
    if args.batch_size is not None:
        batch_size = args.batch_size
    DIFF_HORIZON = args.horizon
    if single_run:
        elo = max_elo

    if args.phase == 2:
        # Phase 2: diffusion training
        if not args.backbone_model:
            print('Error: --backbone-model is required for --phase 2')
            sys.exit(1)
        if not args.pgn:
            print('Error: --pgn is required for --phase 2')
            sys.exit(1)
        END_MODEL = f'{elo}_elo_pos_engine'
        train_diffusion(
            backbone_model=args.backbone_model,
            pgn_path=args.pgn,
            patience=args.patience,
        )
    else:
        # Phase 1: supervised training
        for i in range(elo, max_elo + 1, 200):
            if args.dataset:
                DATASET = args.dataset
            else:
                DATASET = f"full_datasets/elo_{i}_pos.txt" if FULL_SET else f"sub_datasets/elo_{i}_pos.txt"
            # Changed: --resume flag takes priority over START_MODEL logic
            START_MODEL = args.resume if args.resume else ""
            END_MODEL = f'{i}_elo_pos_engine'
            train(
                model=START_MODEL, patience=args.patience,
                model_version=args.model_version,
                use_grokfast=args.grokfast,
                grokfast_alpha=args.grokfast_alpha,
                grokfast_lamb=args.grokfast_lamb,
                grok_log=args.grok_log,
            )
