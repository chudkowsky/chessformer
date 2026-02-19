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

**3. Get elite data** (optional — higher quality than standard Lichess DB):

```bash
# Download 6 months of 2500+ Elo games from Lichess Elite Database
for m in 06 07 08 09 10 11; do
  wget "https://database.nikonoel.fr/lichess_elite_2025-${m}.zip"
done
# Unzip, concatenate, convert
unzip "lichess_elite_2025-*.zip"
cat lichess_elite_2025-*.pgn > elite.pgn
uv run python pgn_to_training_data.py elite.pgn full_datasets/elite_2500_v2_pos.txt 2300
```

**4. Train:**

```bash
# === Phase 1: V2 backbone from scratch (AMD GPU, ROCm) ===
PYTHONUNBUFFERED=1 uv run --no-sync python train_model.py 2500 \
  --dataset full_datasets/elite_2500_v2_pos.txt \
  --num-pos 2e6 --epochs 10 --patience 3 --grokfast

# Resume training from saved checkpoint + Grokfast
PYTHONUNBUFFERED=1 uv run --no-sync python train_model.py 2500 \
  --dataset full_datasets/elite_2500_v2_pos.txt \
  --resume models/2500_elo_pos_engine_v2.pth \
  --num-pos 5e6 --epochs 20 --patience 5 --grokfast

# === Phase 2: diffusion training (requires trained V2 backbone) ===
PYTHONUNBUFFERED=1 uv run --no-sync python train_model.py 2500 \
  --phase 2 \
  --backbone-model models/2500_elo_pos_engine_v2.pth \
  --pgn full_datasets/lichess_elite_2025_jun_nov.pgn \
  --epochs 10 --patience 3

# === Phase 3: self-play training (requires trained V2 backbone) ===
PYTHONUNBUFFERED=1 uv run --no-sync python selfplay_loop.py \
  --model models/2500_elo_pos_engine_v2.pth \
  --generations 10 --games-per-gen 100 --epochs-per-gen 2

# Self-play with supervised data mix (recommended — prevents forgetting)
PYTHONUNBUFFERED=1 uv run --no-sync python selfplay_loop.py \
  --model models/2500_elo_pos_engine_v2.pth \
  --generations 10 --games-per-gen 100 --epochs-per-gen 2 \
  --mix-supervised full_datasets/elite_2500_v2_pos.txt --mix-ratio 0.3 \
  --eval-games 20

# Self-play via train_model.py
PYTHONUNBUFFERED=1 uv run --no-sync python train_model.py 2500 --phase 3 \
  --backbone-model models/2500_elo_pos_engine_v2.pth \
  --generations 10 --games-per-gen 100

# === Non-ROCm (standard CUDA / Mac / CPU) ===
uv run python train_model.py 2000 --dataset full_datasets/elo_2000_pos.txt --num-pos 5e6 --epochs 100 --patience 5
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
| `--batch-size` | `512` | Batch size (lower for less VRAM) |
| `--model-version` | `v2` | `v1` (legacy) or `v2` (Shaw RPE + Smolgen + WDLP) |
| `--grokfast` | off | Enable Grokfast EMA gradient filter (accelerates grokking) |
| `--grokfast-alpha` | `0.98` | EMA decay — higher = captures slower patterns (don't change) |
| `--grokfast-lamb` | `2.0` | Amplification factor — higher = stronger push (don't change) |
| `--grok-log` | `grok_{elo}.log` | Path to grokking metrics log |
| `--phase` | `1` | `1` = supervised policy, `2` = diffusion, `3` = self-play |
| `--backbone-model` | — | Phase 2/3: path to pre-trained V2 checkpoint |
| `--pgn` | — | Phase 2: PGN file for trajectory extraction |
| `--horizon` | `4` | Phase 2: trajectory horizon in half-moves |
| `--generations` | `10` | Phase 3: number of generate-train cycles |
| `--games-per-gen` | `100` | Phase 3: games to generate per generation |
| `--epochs-per-gen` | `2` | Phase 3: training epochs per generation |
| `--temp-schedule` | `1.5:10,1.0:25,0.3:999` | Phase 3: temperature schedule (`temp:ply,...`) |
| `--max-moves` | `200` | Phase 3: max half-moves per game |
| `--buffer-size` | `3` | Phase 3: recent generations in replay buffer |
| `--mix-supervised` | — | Phase 3: path to supervised dataset for mixing |
| `--mix-ratio` | `0.3` | Phase 3: fraction of supervised data in mix |
| `--eval-games` | `0` | Phase 3: evaluation games per generation (0 = skip) |

> **AMD GPU (ROCm):** Use `uv run --no-sync` to preserve ROCm wheels. Prefix with `PYTHONUNBUFFERED=1` for live log output. Optionally add `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` for Flash Attention.

**4. Play:** `uv run python play_gui.py` — trained model appears in model selection automatically.

## Architecture

### V2 (current)
64 squares → piece/file/rank embeddings + 14 auxiliary features → 12-layer Transformer encoder (8 heads, d_model=512, Shaw RPE + Smolgen attention bias) → source-destination policy head (64x64 bilinear) + WDLP value head (win/draw/loss + ply).

~42M parameters | batch size 512 | AdamW (LR 1e-4) | AMP on CUDA/ROCm | Grokfast optional

### V1 (legacy)
64 squares → embeddings → 12-layer Transformer → per-square (from_score, to_score). ~25M parameters.

## Game modes

| Mode | Description |
|---|---|
| Play White/Black | Human vs AI |
| AI vs AI | Model plays both sides |
| vs Stockfish | Model vs Stockfish (optional, for move quality analysis) |

## File reference

| File | Purpose |
|---|---|
| `chessformer.py` | V1 + V2 model architecture |
| `attention.py` | Shaw RPE, Smolgen, Transformer block |
| `train_model.py` | Training loop (Phase 1 supervised + Phase 2 diffusion + Phase 3 self-play) |
| `selfplay_loop.py` | Self-play game generation + training loop |
| `grok_tracker.py` | Grokking detection + Grokfast EMA filter |
| `diffusion_model.py` | ChessDiT (AdaLN-Zero denoising transformer) |
| `noise_schedule.py` | Cosine noise schedule (DDPM) |
| `trajectory_loader.py` | Trajectory pair extraction from PGN |
| `chess_loader.py` | V1/V2 data loaders |
| `pgn_to_training_data.py` | PGN → training data (ELO filter + game result) |
| `play_gui.py` | Pygame GUI |
| `models/` | Trained weights (Git LFS) |
| `tests/` | 100 tests (pytest) |

[MIT License](https://github.com/ncylich/chessformer/blob/main/LICENSE)
