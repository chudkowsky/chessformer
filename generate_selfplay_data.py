"""
Stockfish self-play data generator.

Generates diverse chess games between Stockfish instances at varying strength
levels and saves them in PGN format for use as training data.

Randomness is injected via:
  1. Opening book — games start from ~250 curated ECO opening positions.
  2. Temperature sampling — first temp_cutoff_ply moves are sampled from
     Stockfish MultiPV results via softmax, producing varied but plausible play.
  3. Asymmetric strength — White and Black are assigned different ELO targets
     and search depths per game, producing a mix of outcomes.

Usage:
    # Sequential (Phase 1)
    python generate_selfplay_data.py \\
        --sf-path ./stockfish/stockfish-ubuntu-x86-64-avx2 \\
        --output selfplay_data/run_001.pgn \\
        --num-games 500 --seed 42

    # Parallel (Phase 2)
    python generate_selfplay_data.py \\
        --sf-path ./stockfish/stockfish-ubuntu-x86-64-avx2 \\
        --output selfplay_data/run_parallel.pgn \\
        --num-games 5000 --workers 8 --seed 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import date
from multiprocessing import Pool

import chess
import chess.engine
import chess.pgn
import numpy as np

from openings import load_openings, sample_opening

# ---------------------------------------------------------------------------
# Strength tiers: (weight, white_elo, black_elo, white_depth, black_depth)
# ---------------------------------------------------------------------------
_STRENGTH_TIERS = [
    (0.15, 3000, 3000, 14, 14),   # Elite vs Elite
    (0.10, 2800, 2400, 13, 11),   # Asymmetric strong (W > B)
    (0.10, 2400, 2800, 11, 13),   # Asymmetric strong (B > W)
    (0.20, 2200, 2200, 11, 11),   # Strong club
    (0.20, 2000, 2000,  9,  9),   # Club
    (0.10, 2000, 1600,  9,  7),   # Intermediate asymmetric
    (0.10, 1600, 2000,  7,  9),   # Intermediate asymmetric (flipped)
    (0.05, 1600, 1600,  7,  7),   # Intermediate
]
_TIER_WEIGHTS = [t[0] for t in _STRENGTH_TIERS]

# UCI_Elo is clamped to Stockfish's supported range
_UCI_ELO_MIN = 1320
_UCI_ELO_MAX = 3190


@dataclass
class GameConfig:
    """All parameters needed to generate one self-play game."""
    white_elo: int
    black_elo: int
    white_depth: int
    black_depth: int
    opening_eco: str
    opening_name: str
    opening_fen: str
    opening_moves: list[str]
    temperature: float       # centipawns; controls opening move diversity
    temp_cutoff_ply: int     # stop temperature sampling after this many plies
    annotate_evals: bool
    game_id: int
    seed: int


def _clamp_elo(elo: int) -> int:
    return max(_UCI_ELO_MIN, min(_UCI_ELO_MAX, elo))


def sample_game_config(
    openings: list,
    game_id: int,
    seed: int,
    annotate_evals: bool = True,
) -> GameConfig:
    """Randomly sample configuration for one game."""
    rng = random.Random(seed)
    tier = rng.choices(_STRENGTH_TIERS, weights=_TIER_WEIGHTS, k=1)[0]
    _, welo, belo, wdepth, bdepth = tier
    eco, name, fen, moves = rng.choice(openings)
    temperature = rng.uniform(60.0, 150.0)
    temp_cutoff = rng.randint(8, 20)
    return GameConfig(
        white_elo=welo,
        black_elo=belo,
        white_depth=wdepth,
        black_depth=bdepth,
        opening_eco=eco,
        opening_name=name,
        opening_fen=fen,
        opening_moves=list(moves),
        temperature=temperature,
        temp_cutoff_ply=temp_cutoff,
        annotate_evals=annotate_evals,
        game_id=game_id,
        seed=seed,
    )


def temperature_sample(
    multipv_info: list,
    temperature_cp: float,
    board: chess.Board,
    rng: random.Random,
) -> chess.Move:
    """
    Sample a move from MultiPV engine results using softmax temperature.

    Args:
        multipv_info:   List of info dicts from engine.analyse(..., multipv=N).
        temperature_cp: Temperature in centipawns. Higher = more random.
        board:          Current board (used as fallback for legal moves).
        rng:            Seeded RNG for reproducibility.

    Returns:
        A chess.Move sampled proportionally to exp(score / temperature_cp).
    """
    moves: list[chess.Move] = []
    raw_scores: list[float] = []

    for info in multipv_info:
        if "pv" not in info or not info["pv"]:
            continue
        move = info["pv"][0]
        score_obj = info["score"].relative
        # Convert mate scores: mate in N → large finite value
        cp = score_obj.score(mate_score=2000)
        if cp is None:
            continue
        moves.append(move)
        raw_scores.append(float(cp) / temperature_cp)

    if not moves:
        legal = list(board.legal_moves)
        return rng.choice(legal)

    # Numerically stable softmax
    arr = np.array(raw_scores)
    arr -= arr.max()
    weights = np.exp(arr)
    weights /= weights.sum()

    idx = np.random.choice(len(moves), p=weights)
    return moves[idx]


def generate_game(sf_path: str, config: GameConfig) -> chess.pgn.Game:
    """
    Generate one complete Stockfish self-play game.

    Returns a chess.pgn.Game object ready for PGN serialisation.
    Raises on engine errors; caller should catch and log.
    """
    rng = random.Random(config.seed)
    np.random.seed(config.seed)

    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    try:
        board = chess.Board(config.opening_fen)

        # Apply opening moves
        for uci in config.opening_moves:
            mv = chess.Move.from_uci(uci)
            if mv not in board.legal_moves:
                break  # opening line invalid for this position; stop early
            board.push(mv)

        # Build PGN game object and replay the opening into it
        game = chess.pgn.Game()
        game.setup(chess.Board(config.opening_fen))
        node: chess.pgn.GameNode = game
        for mv in board.move_stack:
            node = node.add_variation(mv)

        # PGN headers
        today = date.today().strftime("%Y.%m.%d")
        game.headers["Event"] = "SF-SelfPlay"
        game.headers["Site"] = "local"
        game.headers["Date"] = today
        game.headers["Round"] = str(config.game_id + 1)
        game.headers["White"] = f"Stockfish_{config.white_elo}"
        game.headers["Black"] = f"Stockfish_{config.black_elo}"
        game.headers["Result"] = "*"
        game.headers["WhiteElo"] = str(config.white_elo)
        game.headers["BlackElo"] = str(config.black_elo)
        game.headers["ECO"] = config.opening_eco
        game.headers["Opening"] = config.opening_name
        game.headers["TimeControl"] = "-"
        game.headers["SFDepthWhite"] = str(config.white_depth)
        game.headers["SFDepthBlack"] = str(config.black_depth)

        ply = board.ply()

        while not board.is_game_over(claim_draw=True):
            is_white = board.turn == chess.WHITE
            depth = config.white_depth if is_white else config.black_depth
            elo   = config.white_elo   if is_white else config.black_elo

            clamped_elo = _clamp_elo(elo)
            engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": clamped_elo,
            })

            eval_score: int | None = None

            if ply < config.temp_cutoff_ply and config.temperature > 0:
                # Temperature sampling phase for opening diversity
                multipv = min(5, board.legal_moves.count())
                if multipv < 1:
                    break
                info_list = engine.analyse(
                    board,
                    chess.engine.Limit(depth=depth),
                    multipv=multipv,
                )
                move = temperature_sample(info_list, config.temperature, board, rng)
            else:
                # Pure engine play
                if config.annotate_evals:
                    info = engine.analyse(board, chess.engine.Limit(depth=depth))
                    move = info["pv"][0] if info.get("pv") else None
                    eval_score = info["score"].white().score(mate_score=10000)
                else:
                    result = engine.play(board, chess.engine.Limit(depth=depth))
                    move = result.move

            if move is None or move not in board.legal_moves:
                # Fallback: let engine pick freely
                result = engine.play(board, chess.engine.Limit(depth=depth))
                move = result.move

            node = node.add_variation(move)
            if eval_score is not None:
                node.comment = f"[%eval {eval_score / 100:.2f}]"

            board.push(move)
            ply += 1

        # Finalise result
        outcome = board.outcome(claim_draw=True)
        result_str = outcome.result() if outcome else "*"
        game.headers["Result"] = result_str
        game.headers["Termination"] = (
            outcome.termination.name.replace("_", " ").title()
            if outcome else "Unknown"
        )

        return game

    finally:
        engine.quit()


# ---------------------------------------------------------------------------
# Worker entry point (used by multiprocessing.Pool)
# ---------------------------------------------------------------------------

def _worker(args: tuple) -> str | None:
    """Generate one game and return its PGN string, or None on failure."""
    sf_path, config = args
    try:
        game = generate_game(sf_path, config)
        return str(game)
    except Exception as exc:  # noqa: BLE001
        print(f"[worker] game {config.game_id} failed: {exc}", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Stockfish self-play PGN training data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sf-path",
        default="./stockfish/stockfish-ubuntu-x86-64-avx2",
        help="Path to Stockfish binary.",
    )
    parser.add_argument(
        "--output",
        default="selfplay_data/selfplay.pgn",
        help="Output PGN file path.",
    )
    parser.add_argument(
        "--num-games", type=int, default=100,
        help="Number of games to generate.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (each game gets seed = base + game_id).",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of parallel worker processes (1 = sequential).",
    )
    parser.add_argument(
        "--no-evals", action="store_true",
        help="Skip %eval annotations (faster generation).",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.sf_path):
        print(f"[error] Stockfish binary not found: {args.sf_path}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    openings = load_openings()
    configs = [
        sample_game_config(
            openings,
            game_id=i,
            seed=args.seed + i,
            annotate_evals=not args.no_evals,
        )
        for i in range(args.num_games)
    ]

    t_start = time.monotonic()
    done = 0
    failed = 0

    with open(args.output, "w") as out_file:
        if args.workers <= 1:
            # --- Sequential ---
            for cfg in configs:
                pgn_str = _worker((args.sf_path, cfg))
                if pgn_str:
                    out_file.write(pgn_str + "\n\n")
                    out_file.flush()
                    done += 1
                else:
                    failed += 1
                elapsed = time.monotonic() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(
                    f"\r[{done+failed}/{args.num_games}] "
                    f"done={done} failed={failed} "
                    f"rate={rate:.1f} games/s",
                    end="",
                    flush=True,
                )
        else:
            # --- Parallel ---
            work_items = [(args.sf_path, cfg) for cfg in configs]
            with Pool(processes=args.workers) as pool:
                for pgn_str in pool.imap_unordered(_worker, work_items):
                    if pgn_str:
                        out_file.write(pgn_str + "\n\n")
                        out_file.flush()
                        done += 1
                    else:
                        failed += 1
                    elapsed = time.monotonic() - t_start
                    rate = done / elapsed if elapsed > 0 else 0
                    print(
                        f"\r[{done+failed}/{args.num_games}] "
                        f"done={done} failed={failed} "
                        f"rate={rate:.1f} games/s",
                        end="",
                        flush=True,
                    )

    elapsed = time.monotonic() - t_start
    print(
        f"\n\nFinished: {done} games saved to {args.output} "
        f"({failed} failed) in {elapsed:.1f}s "
        f"({done/elapsed:.1f} games/s)"
    )


if __name__ == "__main__":
    main()
