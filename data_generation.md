# Chess Training Data Generation — Theoretical Analysis & Implementation Plan

## Status

| Phase | Description | Status |
| --- | --- | --- |
| 1a | `openings.py` — curated opening FENs | ✅ Complete |
| 1b | `generate_selfplay_data.py` — sequential generation | ✅ Complete |
| 2 | Parallel generation via `multiprocessing.Pool` | ✅ Complete |
| 3 | GPU-accelerated encoding / lc0 integration | Planned |

---

## Executive Summary

This document outlines a strategy for generating high-quality, diverse chess training data using Stockfish self-play. The core challenge is that **two deterministic engines playing from the same position always produce the same game** — so we need principled randomness injection. The data is saved in PGN format and later converted to the existing training format.

---

## 1. Theoretical Motivation

### Why Self-Play Beats Lichess Data

| Property | Lichess Data | Stockfish Self-Play |
| --- | --- | --- |
| Move quality | Mixed (humans blunder) | Controllable |
| Position diversity | Very high | Needs engineering |
| ELO distribution | Wide | Fully controlled |
| Annotation | None (raw moves) | Can embed evaluations |
| Data volume | Limited by database | Unlimited |
| Perspective | Human style | Engine style |

Lichess data at 2000+ ELO still contains significant human error patterns, opening prep bias (players repeating memorized lines), and time-pressure blunders. Stockfish self-play at calibrated strength gives **precise, reproducible quality** and **no noise from time pressure or psychology**.

The key insight from AlphaZero (DeepMind, 2017): a model trained exclusively on self-play data dramatically outperformed one trained on human games, even when the human dataset was vastly larger.

### The Determinism Problem

Stockfish is a deterministic engine: given the same position and the same parameters, it always plays the same move. Two games started from the initial position with identical settings will be **byte-for-byte identical**. This makes naive self-play useless — you would generate millions of copies of the same game.

Solution: inject randomness at multiple levels to ensure diverse, non-repeating games.

---

## 2. Randomness Injection Strategies

### 2.1 Opening Diversification (Most Important)

The opening phase determines the entire character of the game — if all games start the same, they all diverge at the same point and may converge to similar middlegame structures.

#### Strategy A — Random Legal Moves for First N Plies

Apply N (e.g., 4–8) uniformly random legal moves before handing control to Stockfish. This scatters games across a huge variety of positions quickly.

- Pros: Extremely simple, maximum variety
- Cons: May produce strategically nonsensical positions that never appear in real play

#### Strategy B — Weighted Random from Stockfish MultiPV (Recommended)

Use Stockfish's `MultiPV` mode to compute the top-K moves, then sample one move proportionally to their evaluation scores (temperature sampling). This produces varied but *plausible* positions.

```text
scores = [eval_1, eval_2, ..., eval_k]  # centipawns
probs  = softmax(scores / T)            # temperature T controls spread
move   = random.choices(top_k_moves, weights=probs)
```

With temperature T:

- T → 0: Always play best move (deterministic)
- T = 50 cp: Strong preference for top moves, some variety
- T = 100 cp: Moderate randomness, all top-3 moves are plausible
- T → ∞: Uniform random (chaotic)

For opening phase (moves 1–15): use T = 80–120 cp.
For middlegame (moves 15+): hand control to Stockfish deterministically.

#### Strategy C — Curated Opening Book

Maintain a list of starting FENs or move sequences (e.g., 500 common ECO openings) and pick one randomly per game.

- Pros: Realistic positions, covers all major opening systems
- Cons: Requires curation; still deterministic after the book ends

#### Strategy D — Polyglot Opening Book

Use an existing Polyglot `.bin` opening book (e.g., `gm2001.bin`, `komodo.bin`). Follow the book for opening moves, weighted by book move frequency, then switch to Stockfish.

#### Recommended Approach: Combine B + C

- Start from a random opening FEN (from curated list of ~300 openings)
- Apply temperature sampling for moves 1–12 (relative to that opening)
- Switch to pure Stockfish for the rest of the game

---

### 2.2 Engine Strength Variation

Playing only at maximum strength (depth 20+) produces perfect play that's less instructive for the model — it won't see the kinds of positions that arise at 2000 ELO. Mix strength levels:

#### Stockfish Skill Level (0–20)

UCI parameter: `setoption name Skill Level value X`

Approximate ELO mapping:

| Skill Level | Approx ELO |
| --- | --- |
| 0 | ~1350 |
| 5 | ~1600 |
| 10 | ~1800 |
| 15 | ~2100 |
| 18 | ~2600 |
| 20 | ~3500+ |

Internally, Skill Level works by: (a) limiting the depth, and (b) occasionally playing a random move weighted by evaluation.

#### Direct ELO Limiting (More Precise)

```text
setoption name UCI_LimitStrength value true
setoption name UCI_Elo value 2000
```

UCI_Elo range: 1320–3190. This is the most principled way to target a specific strength.

#### Depth Variation

`go depth X` — search to fixed depth instead of time limit.

- Depth 5–8: Fast, ~1800 ELO equivalent
- Depth 10–12: Medium, ~2200 ELO
- Depth 15: Strong, ~2600 ELO
- Depth 20+: Near-maximum, 3000+ ELO

#### Recommended Strength Distribution for Training Data

| Category | % of games | Config |
| --- | --- | --- |
| Elite | 20% | UCI_Elo=3000, depth=18 |
| Strong | 35% | UCI_Elo=2400, depth=14 |
| Club | 30% | UCI_Elo=2000, depth=10 |
| Intermediate | 15% | UCI_Elo=1600, depth=7 |

This distribution mirrors real Lichess ratings and ensures the model learns patterns at multiple strength levels.

---

### 2.3 Move Time and Node Variation

Instead of fixed depth, use `go movetime X` (milliseconds) or `go nodes X`. This produces natural time-pressure variation:

- Fast games (100ms/move): More tactical errors, similar to blitz
- Slow games (2000ms/move): Very accurate, near-optimal
- Node limits: More CPU-consistent across machines

---

### 2.4 Asymmetric Games

Play White and Black at different strengths:

- Strong White vs. Weak Black: Model learns to press advantages, convert wins
- Weak White vs. Strong Black: Model learns defense, resource saving
- Equal strength: Most common, balanced games

This produces a dataset with varied game outcomes (wins, draws, losses) rather than all games being drawn (which pure max-strength Stockfish games often are).

---

## 3. Data Quality for Transformer Training

### What Transformers Learn Best From

1. **Move prediction**: Learn the mapping `position → best_move`. Need high-quality moves.
2. **Position understanding**: The board representation must cover all common position types.
3. **Diversity**: The model must see varied pawn structures, piece configurations, and endgame types.

### Key Data Quality Metrics

- **Position uniqueness**: What fraction of positions in the dataset are unique? Target >90%.
- **Phase coverage**: Ratio of opening/middlegame/endgame positions. Lichess data is opening-heavy; self-play can be tuned.
- **Centipawn variance**: A dataset of only equal positions (±50 cp) doesn't teach the model to handle decisive advantages. Mix in won/lost positions.
- **Move diversity per position**: Ensure the dataset doesn't repeat the same move from similar positions too often.

### Evaluation Annotations

Unlike raw Lichess PGN, self-play allows embedding Stockfish evaluations with every move. This unlocks potential future training signals:

```pgn
1. e4 {+0.35/14} e5 {+0.15/14} 2. Nf3 {+0.42/14} ...
```

Even if the current model ignores evaluations, recording them now means the data can be reused for future value-head training (as in AlphaZero).

---

## 4. PGN Format Specification

### Standard PGN Headers

```pgn
[Event "SF-SelfPlay"]
[Site "local"]
[Date "2026.02.18"]
[Round "1"]
[White "Stockfish_2000"]
[Black "Stockfish_1600"]
[Result "1-0"]
[WhiteElo "2000"]
[BlackElo "1600"]
[Opening "Sicilian Defense"]
[ECO "B20"]
[TimeControl "-"]
[Termination "Normal"]
[SFDepthWhite "10"]
[SFDepthBlack "7"]
```

Custom headers `SFDepthWhite`, `SFDepthBlack` track engine settings for reproducibility.

### Optional Evaluation Annotations

```pgn
1. e4 { [%eval 0.35] } 1... c5 { [%eval 0.28] } 2. Nf3 { [%eval 0.41] }
```

This is the standard Lichess PGN annotation format (compatible with `pgn_to_training_data.py`).

---

## 5. Implementation

### Phase 1a — Opening Book (`openings.py`) ✅

File: `openings.py`

Contains ~250 curated ECO openings as `(eco_code, name, fen, moves_uci)` tuples covering all major opening systems: Open Games, Sicilian, French, Caro-Kann, Queen's Gambit, King's Indian, English, Nimzo-Indian, and more.

```python
def load_openings() -> list[tuple]:
    """Return list of (eco, name, fen, moves_uci) for all openings."""
    return OPENINGS
```

### Phase 1b — Sequential Generator (`generate_selfplay_data.py`) ✅

File: `generate_selfplay_data.py`

Core components:

- `GameConfig` dataclass — all parameters for one game
- `temperature_sample(multipv_info, temperature_cp)` — softmax sampling over top-K moves
- `generate_game(sf_path, config, game_id)` — produces one `chess.pgn.Game`
- `sample_game_config(openings)` — randomly samples a `GameConfig`
- `main()` — sequential CLI entry point

```bash
python generate_selfplay_data.py \
    --sf-path ./stockfish/stockfish-ubuntu-x86-64-avx2 \
    --output selfplay_data/run_001.pgn \
    --num-games 500 \
    --seed 42
```

### Phase 2 — Parallel Generator ✅

Same file `generate_selfplay_data.py`, add `--workers N` flag.

Uses `multiprocessing.Pool` with `imap_unordered` — each worker owns an independent Stockfish process. Expected speedup: ~linear with core count.

```bash
python generate_selfplay_data.py \
    --sf-path ./stockfish/stockfish-ubuntu-x86-64-avx2 \
    --output selfplay_data/parallel_run.pgn \
    --num-games 5000 \
    --workers 8 \
    --seed 42
```

### Phase 3 — GPU Parallelisation (Planned)

GPU acceleration for chess self-play is unconventional — Stockfish is CPU-only. The GPU is useful for:

1. **Neural network inference in the loop**: Mix our model's moves with Stockfish for curriculum learning; GPU inference can be batched.
2. **Leela Chess Zero (lc0)**: GPU-accelerated neural engine as generator, produces more human-like patterns.
3. **CUDA-accelerated game encoding**: Convert PGN → training tensors on GPU rather than CPU.

lc0 natively supports temperature sampling:

```text
setoption name Temperature value 0.8
setoption name TempDecayMoves value 30
```

---

## 6. Integration with Existing Pipeline

### PGN → Training Data

`pgn_to_training_data.py` was extended to skip ELO/time-control filters for self-play games (detected via `Event` header starting with `"SF-SelfPlay"`).

### Full Data Pipeline

```text
generate_selfplay_data.py
    --num-games 10000
    --output selfplay_data/raw.pgn
         ↓
pgn_to_training_data.py
    selfplay_data/raw.pgn
    full_datasets/selfplay_pos.txt
    0                    (min_elo=0, no filter for self-play)
         ↓
train_model.py
    --dataset full_datasets/selfplay_pos.txt
    --model models/selfplay_v1.pth
```

---

## 7. Expected Data Volume & Quality

### Games per Hour Estimates (CPU)

| Config | Games/hr (1 core) | Games/hr (8 cores) |
| --- | --- | --- |
| Depth 7, fast openings | ~180 | ~1400 |
| Depth 10, mixed | ~80 | ~620 |
| Depth 14, annotated | ~25 | ~190 |
| Depth 18, elite | ~8 | ~60 |

### Positions per Game (Average)

Average game length: 40–80 moves → 80–160 positions per game (both sides).

- 1,000 games ≈ 100,000 positions
- 10,000 games ≈ 1,000,000 positions (matches current Lichess dataset size)
- 100,000 games ≈ 10,000,000 positions

### Data Quality Comparison

| Metric | Lichess 2000+ | SF Self-Play (2000 ELO) | SF Self-Play (mixed) |
| --- | --- | --- | --- |
| Avg centipawn loss | ~50 | ~10 | ~25 |
| Position uniqueness | ~98% | ~95% | ~97% |
| Phase coverage | Opening-heavy | Balanced | Balanced |
| Move consistency | Human-like | Engine-like | Engine-like |

---

## 8. Key Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| Games too similar (low diversity) | Combine opening book + temperature sampling |
| Stockfish process leaks | Always use `engine.quit()` in try/finally blocks |
| PGN parse errors | Validate with `python-chess` before writing |
| Drawn games dominate at high ELO | Use asymmetric strengths; target 40-30-30 W/D/L split |
| CPU bottleneck | Use multiprocessing (Phase 2); prioritize fast depth settings |
| Training worse than Lichess data | Mix self-play + Lichess data; A/B test model quality |

---

## 9. File Structure

```text
chessformer/
├── generate_selfplay_data.py    # Main generator (Phase 1 + 2)
├── openings.py                  # Opening book (~250 ECO openings)
├── pgn_to_training_data.py      # Extended to handle self-play PGN
├── selfplay_data/               # Output directory (gitignored)
│   ├── run_001.pgn
│   └── ...
└── full_datasets/
    ├── elo_2000_pos.txt          # Existing Lichess data
    └── selfplay_pos.txt          # Self-play converted data
```

---

Document version: 2.0 — Feb 2026
