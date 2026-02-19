# Import required libraries
from chessformer import ChessTransformer, ChessTransformerV2
from chess_loader import get_dataloader, get_dataloader_v2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
import time
from datetime import timedelta

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


def train(model: str = "", patience: int = None, model_version: str = "v1"):
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
    print(f'Parameters: {num_params:,} | Device: {device}\n')

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
                if CLIP:
                    scaler.unscale_(optimizer)
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
    parser.add_argument('--model-version', type=str, default='v1', choices=['v1', 'v2'],
                        help='Model architecture version (default: v1)')
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
    if single_run:
        elo = max_elo
    for i in range(elo, max_elo + 1, 200):
        if args.dataset:
            DATASET = args.dataset
        else:
            DATASET = f"full_datasets/elo_{i}_pos.txt" if FULL_SET else f"sub_datasets/elo_{i}_pos.txt"
        # Changed: --resume flag takes priority over START_MODEL logic
        START_MODEL = args.resume if args.resume else ""
        END_MODEL = f'{i}_elo_pos_engine'
        train(model=START_MODEL, patience=args.patience, model_version=args.model_version)
