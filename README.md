# Chessformer

Chess AI built on a Transformer encoder. Looks at the board once, predicts the best move — no search tree.

## Quick start

```bash
git clone https://github.com/chudkowsky/chessformer.git
cd chessformer
uv run python play_gui.py
```

> Requires [uv](https://docs.astral.sh/uv/getting-started/installation/), Python 3.12+, and [Git LFS](https://git-lfs.com/) (for model weights). Install LFS with `git lfs install` before cloning. `uv run` auto-installs all dependencies on first run.
>
> **AMD GPU (ROCm):** Place ROCm wheels in `wheelies/` and create a `uv.toml` — see [ROCm setup](#rocm-setup).

## Training

### 1. Prepare data

```bash
# Download + convert Lichess games (stream, low disk usage)
curl -s https://database.lichess.org/standard/lichess_db_standard_rated_2026-01.pgn.zst \
  | zstd -d \
  | uv run python pgn_to_training_data.py /dev/stdin full_datasets/elo_2000_pos.txt 2000 5000000
```

`pgn_to_training_data.py` args: `<pgn_file> <output_file> <min_elo> [max_positions]`

### 2. Train on data (supervised)

```bash
uv run python train_model.py 2500 --dataset full_datasets/elite_2500_v2_pos.txt
```

Best model saved automatically to `models/{elo}_elo_pos_engine_v2.pth`.

| Argument | Required | Default | Description |
|---|---|---|---|
| `elo` | **yes** | — | Target Elo label (used in output filename) |
| `--dataset` | **yes** | — | Path to training data file |
| `--num-pos` | | `1e6` | Number of positions to load |
| `--epochs` | | `10` | Number of training epochs |
| `--patience` | | off | Early stopping after N epochs without improvement |
| `--batch-size` | | `512` | Batch size (lower for less VRAM) |
| `--lr` | | `1e-4` | Learning rate (`1e-5` for fine-tuning) |
| `--resume` | | — | Path to checkpoint to continue training |
| `--grokfast` | | off | Enable Grokfast EMA gradient filter |
| `--device` | | `auto` | `auto` / `cuda` / `mps` / `cpu` |

### 3. Self-play (improves model by playing against itself)

```bash
uv run python selfplay_loop.py --model latest
```

`--model latest` automatically picks the newest model from `models/`. Best model saved back to `models/` at end.

| Argument | Required | Default | Description |
|---|---|---|---|
| `--model` | **yes** | — | Path to V2 checkpoint, or `latest` |
| `--generations` | | `20` | Number of generate-train cycles |
| `--games-per-gen` | | `200` | Games per generation |
| `--epochs-per-gen` | | `3` | Training epochs per generation |
| `--eval-games` | | `100` | Evaluation games vs baseline (0 = skip) |
| `--mcts-sims` | | `0` | MCTS simulations/move (0 = raw policy) |
| `--cpuct` | | `1.25` | MCTS exploration constant |
| `--buffer-size` | | `5` | Generations in replay buffer |
| `--device` | | `auto` | `auto` / `cuda` / `mps` / `cpu` |

### 4. Self-play with supervised data mix (recommended)

Mixes self-play games with supervised data to prevent forgetting.

```bash
uv run python selfplay_loop.py --model latest \
  --mix-supervised full_datasets/elite_2500_v2_pos.txt
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `--mix-supervised` | **yes** | — | Path to supervised dataset |
| `--mix-ratio` | | `0.5` | Fraction of supervised data in mix |

All other arguments from self-play table above also apply.

### 5. Self-play with MCTS (highest quality, slower)

```bash
uv run python selfplay_loop.py --model latest \
  --mix-supervised full_datasets/elite_2500_v2_pos.txt \
  --mcts-sims 25
```

~25 forward passes/move instead of 1. Expect ~5s/game instead of ~0.3s.

### ROCm setup

For AMD GPUs, create a `uv.toml` in the project root (gitignored):

```toml
find-links = ["wheelies"]
override-dependencies = ["torch==2.9.1+rocm7.2.0.lw.git7e1940d4"]
```

Then prefix commands with `PYTHONUNBUFFERED=1` for live output. `uv run` will use ROCm wheels automatically.

## Play

```bash
uv run python play_gui.py
```

Trained models in `models/` appear in model selection automatically.

| Mode | Description |
|---|---|
| Play White/Black | Human vs AI |
| AI vs AI | Model plays both sides |
| vs Stockfish | Model vs Stockfish (move quality analysis) |

## Architecture

### V2 (current)
64 squares → piece/file/rank embeddings + 14 auxiliary features → 12-layer Transformer encoder (8 heads, d_model=512, Shaw RPE + Smolgen attention bias) → source-destination policy head (64x64 bilinear) + WDLP value head (win/draw/loss + ply).

~42M parameters | batch size 512 | AdamW (LR 1e-4) | AMP on CUDA/ROCm | Grokfast optional

### V1 (legacy)
64 squares → embeddings → 12-layer Transformer → per-square (from_score, to_score). ~25M parameters.

## File reference

| File | Purpose |
|---|---|
| `chessformer.py` | V1 + V2 model architecture |
| `attention.py` | Shaw RPE, Smolgen, Transformer block |
| `train_model.py` | Supervised training loop |
| `selfplay_loop.py` | Self-play: game generation + gated training |
| `mcts.py` | Monte Carlo Tree Search (AlphaZero-style) |
| `model_utils.py` | Model loading, device detection, preprocessing, loss |
| `policy.py` | Move selection from model logits |
| `grok_tracker.py` | Grokking detection + Grokfast EMA filter |
| `diffusion_model.py` | ChessDiT (AdaLN-Zero denoising transformer) |
| `noise_schedule.py` | Cosine noise schedule (DDPM) |
| `trajectory_loader.py` | Trajectory pair extraction from PGN |
| `chess_loader.py` | V1/V2 data loaders |
| `pgn_to_training_data.py` | PGN → training data (ELO filter + game result) |
| `play_gui.py` | Pygame GUI |
| `models/` | Trained weights (Git LFS) |
| `tests/` | 143 tests (pytest) |

[MIT License](https://github.com/ncylich/chessformer/blob/main/LICENSE)
