# Import required libraries
from chessformer import ChessTransformer
from chess_loader import get_dataloader
import torch
import torch.nn as nn
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
# Changed: 128→512 to better utilize GPU VRAM (33% → ~80%) and reduce batches per epoch
batch_size = int(512 / SCALE)
LR = {.5: 5e-4, 1: 1e-4, 2: 1e-5}
LR = LR[SCALE] if SCALE in LR else 1e-5
CLIP = .1
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


def train(model: str = ""):
    # Model loading or initialization
    if model:
        model = torch.load(f'models/{START_MODEL}')
        print(f'Using Pretrained Model: {START_MODEL}')
    else:
        model = ChessTransformer(inp_ntoken, out_ntoken, d_model, nhead, d_hid, nlayers, dropout=dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f'Creating New Model: {END_MODEL}_best_whole.pth\nDataset: {DATASET}')
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
    dataloader, testloader = get_dataloader(DATASET, batch_size=batch_size, num_workers=4, num_pos=NUM_POS)

    # Training loop
    best_loss = None
    best_test_loss = None
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0
        for batch, (boards, target) in enumerate(dataloader):
            boards, target = boards.to(device), target.to(device)
            # Changed: mixed precision - forward pass in float16 for ~2x speedup (CUDA only)
            if use_amp:
                with autocast("cuda"):
                    output = model(boards)
                    loss = loss_fn(output, target)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                if CLIP:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
                scaler.step(optimizer)
                scaler.update()
            else:
                output = model(boards)
                loss = loss_fn(output, target)
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
            for batch in testloader:
                boards, target = batch[0].to(device), batch[1].to(device)
                output = model(boards)
                loss = loss_fn(output, target)
                tot_test_loss += loss.item()

        avg_test_loss = tot_test_loss / len(testloader)
        print(f'Epoch {epoch} done | Train loss: {avg_train_loss:.4f} | Test loss: {avg_test_loss:.4f}')

        # Save the best model
        if best_test_loss is None or tot_test_loss < best_test_loss:
            best_test_loss = tot_test_loss
            if epoch >= min(num_epochs - 2, 3):
                torch.save(model, f'models/{END_MODEL}.pth')
                print(f'  -> Saved model (best test loss)')

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
    args = parser.parse_args()
    elo = args.elo
    max_elo = elo
    NUM_POS = args.num_pos
    if args.device != 'auto':
        device = torch.device(args.device)
    if single_run:
        elo = max_elo
    new_model = True
    for i in range(elo, max_elo + 1, 200):
        if args.dataset:
            DATASET = args.dataset
        else:
            DATASET = f"full_datasets/elo_{i}_pos.txt" if FULL_SET else f"sub_datasets/elo_{i}_pos.txt"
        START_MODEL = f'{i - 200}_elo_pos_engine_best_whole.pth' if not new_model else ""
        END_MODEL = f'{i}_elo_pos_engine'
        train(model=START_MODEL)
