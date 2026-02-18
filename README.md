# Chessformer

A chess-playing AI built on a Transformer encoder, trained on ~2000 ELO Lichess games.
The model learns to score board positions and select legal moves without any search tree.

---

## How it works

### Model architecture (`chessformer.py`)

The board is represented as a flat sequence of **64 tokens**, one per square (a1→h8),
where each token is a piece-type index (0 = empty, 1–6 = white pieces, 7–12 = black pieces).

The board is always shown from the **moving side's perspective** — when it is Black's turn
the board string is flipped and piece colours are swapped before encoding, so the model
always "sees" itself as White.

Three embeddings are summed per square:
- **Piece embedding** — 13-class lookup, `d_model = 512`
- **File (x) embedding** — 8-class lookup
- **Rank (y) embedding** — 8-class lookup

The combined embedding is passed through a **12-layer Transformer encoder** (8 heads,
FFN dim 1024, dropout 0.1).

The output is a `[batch, 64, 2]` tensor. The two output values per square are treated as
a **(from_score, to_score)** pair. The move `i → j` is ranked by `from_score[i] + to_score[j]`.
All legal moves are enumerated and the highest-ranked legal move is played.

Two post-processing rules are applied at inference:
- **Repetition avoidance** — if the top move would allow a threefold-repetition claim,
  the next best legal move is chosen instead.
- **Checkmate priority** — any move that immediately checkmates the opponent is played
  regardless of model score.

### Training (`train_model.py`)

| Hyperparameter | Value |
|---|---|
| Architecture | `ChessTransformer` |
| `d_model` | 512 |
| Heads | 8 |
| Layers | 12 |
| FFN dim | 1024 |
| Dropout | 0.1 |
| Loss | CrossEntropyLoss |
| Batch size | 128 |
| Dataset | ~1 million positions, ELO 2000 filter |
| Optimiser | Adam with LR scheduling (γ = 0.9) |
| Gradient clipping | 0.1 |

Training data comes from Lichess PGN dumps, filtered to games where both players are
rated ~2000. Positions are stored with the board from the moving player's POV.

---

## Available models

Trained weights live in the `models/` directory.
Both the GUI and CLI pick them up automatically — no hardcoded path needed.

| File | Description |
|---|---|
| `2000_elo_pos_engine.pth` | Main checkpoint, 2000 ELO target |
| `2000_elo_pos_engine_best_test_whole.pth` | Checkpoint with best validation loss on the full test set |

To add a new model just drop a `.pth` file into `models/` — it will appear in the
selection screen automatically.

---

## Requirements

**Python 3.10+** is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
numpy==1.26.3
python-chess==1.999
pygame
```

GPU acceleration is used automatically when available (CUDA → MPS → CPU fallback).

### Optional: Stockfish

Stockfish is used for **move quality analysis** (centipawn loss, accuracy %).
Without it the game still works — move rating is simply disabled.

- The repo may contain a pre-packaged `stockfish-ubuntu-x86-64-avx2.tar` that is
  extracted automatically on first run.
- Or supply your own binary — both scripts ask for the path at startup.

---

## Running

### GUI (`play_gui.py`)

```bash
python play_gui.py
```

**Startup flow:**

1. Enter Stockfish binary path (or press Enter to skip).
2. Enter AI vs AI move delay in seconds (default 1).
3. **Model selection screen** — click any model listed from `models/`.
4. **Mode selection** — choose one of four modes.

**Game modes:**

| Button | Description |
|---|---|
| Play White | You play White; the AI plays Black. |
| Play Black | You play Black; the AI plays White (board flipped). |
| AI vs AI | The loaded model plays both sides with a configurable delay. |
| vs Stockfish | The model faces Stockfish; choose which colour the model plays. |

**GUI features:**

- **Move quality bar** (left panel) — each move is scored by Stockfish (depth 12) and
  shown as a colour-coded bar: green = excellent, yellow = inaccuracy, red = blunder.
- **Legal move dots** — clicking a piece highlights its legal destinations.
- **Check highlight** — the king square turns red when in check.
- **Last-move highlight** — the from/to squares of the previous move are tinted.
- **End-of-game overlay** — when the game ends a Stockfish analysis summary appears on
  screen showing counts of Excellent / Good / Inaccuracy / Mistake / Blunder and
  estimated accuracy % for each player.
- Summary is also printed to the terminal.

### CLI (`play_against.py`)

```bash
python play_against.py
```

Prompts for Stockfish path then offers three modes:

| Mode | Description |
|---|---|
| `1` Human vs AI | Enter moves in SAN (`e4`, `Nf3`) or UCI (`e2e4`) format. |
| `2` AI vs AI | Both sides played by the model; configurable per-move delay. |
| `3` Model vs Stockfish | Model faces Stockfish (requires Stockfish). Choose model colour. |

Each move is rated by Stockfish in real time if available.
A summary table is printed at the end (or on Ctrl+C).

---

## File reference

| File | Purpose |
|---|---|
| `chessformer.py` | `ChessTransformer` model + `PositionalEncoding` |
| `transformer.py` | Compatibility shim so older `.pth` files unpickle correctly |
| `chess_moves_to_input_data.py` | Board-to-string conversion, perspective flipping, dataset preprocessing |
| `chess_loader.py` | `ChessDataset` / `get_dataloader` for training |
| `train_model.py` | Training loop, hyperparameters, checkpoint saving |
| `inference_test.py` | `preprocess()` and `postprocess_valid()` used by both runners |
| `play_against.py` | CLI game runner |
| `play_gui.py` | Pygame GUI game runner |
| `policy.py` | Auxiliary policy utilities |
| `models/` | Trained `.pth` weight files |
| `stockfish/` | Auto-extracted Stockfish binary (created on first run) |
