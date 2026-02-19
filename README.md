# Chessformer

Chess AI built on a Transformer encoder. Looks at the board once, predicts the best move — no search tree.

## Quick start

```bash
git clone https://github.com/chudkowsky/chessformer.git
cd chessformer
uv run python play_gui.py
```

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.10+, and [Git LFS](https://git-lfs.com/) (for model weights). Install LFS with `git lfs install` before cloning. `uv run` auto-installs all dependencies on first run.
>
> **AMD GPU (ROCm):** Install custom torch wheels before running: `uv pip install wheelies/*.whl --force-reinstall`

## Train your own model

**1. Get data** from https://database.lichess.org/ (~29GB compressed / ~200GB decompressed per month)

**2. Filter by ELO + convert** — keeps games where both players are above given ELO:

```bash
# Option A: stream directly (low disk usage)
curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst \
  | zstd -d \
  | uv run python pgn_to_training_data.py /dev/stdin full_datasets/elo_2000_pos.txt 2000 1000000

# Option B: download first (needs ~230GB disk, faster)
wget https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst
zstd -d lichess_db_standard_rated_2026-01.pgn.zst
uv run python pgn_to_training_data.py lichess_db_standard_rated_2026-01.pgn full_datasets/elo_2000_pos.txt 2000 1000000
```

Arguments: `<pgn_file> <output_file> <min_elo> [max_positions]`

**3. Train:**

```bash
# Train from scratch (batch-size 512 is safe for 12GB+ VRAM)
uv run python train_model.py 2000 --dataset full_datasets/elo_2000_pos.txt --num-pos 5e6 --batch-size 512 --epochs 100 --patience 5

# Fine-tune existing model with lower learning rate
uv run python train_model.py 2000 --resume models/2000_elo_pos_engine.pth --dataset full_datasets/elo_2000_pos.txt --num-pos 5e6 --batch-size 512 --lr 1e-5 --epochs 50 --patience 5

# Mac / CPU
uv run python train_model.py 2000 --dataset full_datasets/elo_2000_pos.txt --num-pos 1e6 --device mps     # Mac
uv run python train_model.py 2000 --dataset full_datasets/elo_2000_pos.txt --num-pos 1e6 --device cpu     # CPU
```

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `full_datasets/elo_{elo}_pos.txt` | Path to training data file |
| `--num-pos` | `1000000` | Number of positions to load |
| `--device` | `auto` | `auto`, `cuda`, `mps`, or `cpu` |
| `--resume` | — | Path to existing model to continue training |
| `--lr` | `1e-4` | Learning rate (lower for fine-tuning, e.g. `1e-5`) |
| `--epochs` | `10` | Number of training epochs |
| `--patience` | — | Early stopping: stop after N epochs without improvement |
| `--batch-size` | `1024` | Batch size (lower for less VRAM, e.g. `512` for 12GB) |

> **AMD GPU (ROCm):** Prefix with `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` to enable Flash Attention.

**4. Play:** `uv run python play_gui.py` — trained model appears in model selection automatically.

## Architecture

64 board squares → piece/file/rank embeddings → 12-layer Transformer encoder (8 heads, d_model=512) → per-square (from_score, to_score) → highest-ranked legal move is played.

~25M parameters | batch size 1024 | AdamW (LR 1e-4) | AMP on CUDA/ROCm

## Game modes

| Mode | Description |
|---|---|
| Play White/Black | Human vs AI |
| AI vs AI | Model plays both sides |
| vs Stockfish | Model vs Stockfish (optional, for move quality analysis) |

## File reference

| File | Purpose |
|---|---|
| `chessformer.py` | Model architecture |
| `train_model.py` | Training loop |
| `pgn_to_training_data.py` | PGN → training data (ELO filter) |
| `play_gui.py` | Pygame GUI |
| `play_against.py` | CLI game runner |
| `models/` | Trained weights (Git LFS) |

[MIT License](https://github.com/ncylich/chessformer/blob/main/LICENSE)
