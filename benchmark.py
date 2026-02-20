"""Benchmark suite for ChessTransformerV2 models.

Measures model strength via:
- Model vs Stockfish at configurable skill levels
- Model vs Model head-to-head comparison
- Approximate Elo estimation from winrates

Usage:
    # Model vs Stockfish (skill 5, 30 games)
    uv run benchmark.py vs-stockfish models/2500_elo_pos_engine_v2.pth --skill 5 --games 30

    # Model vs Model (50 games)
    uv run benchmark.py vs-model models/new.pth models/old.pth --games 50

    # Full benchmark (multiple Stockfish levels)
    uv run benchmark.py full models/2500_elo_pos_engine_v2.pth --games 20
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass

import chess
import chess.engine
import torch

from chessformer import ChessTransformerV2
from model_utils import detect_device, load_model, preprocess_board
from openings import sample_opening
from policy import greedy_move_v2

import random


# --- Data types ---


@dataclass(frozen=True)
class MatchResult:
    """Result of a multi-game match."""

    wins: int
    draws: int
    losses: int
    label_a: str
    label_b: str

    @property
    def total(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        """Score from player A's perspective (1.0 = all wins, 0.0 = all losses)."""
        return (self.wins + 0.5 * self.draws) / self.total if self.total > 0 else 0.5

    @property
    def elo_diff(self) -> float:
        """Approximate Elo difference (A minus B)."""
        s = max(0.001, min(0.999, self.score))
        return -400 * math.log10(1 / s - 1)

    def summary(self) -> str:
        return (
            f"{self.label_a} vs {self.label_b}: "
            f"+{self.wins} ={self.draws} -{self.losses} "
            f"({self.score:.1%} winrate, {self.elo_diff:+.0f} Elo)"
        )


# --- Engine helpers ---


def _find_stockfish() -> str:
    """Find Stockfish binary, checking common locations."""
    import shutil
    from pathlib import Path

    candidates = [
        Path("stockfish/stockfish-ubuntu-x86-64-avx2"),
        Path("stockfish/stockfish"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)

    found = shutil.which("stockfish")
    if found:
        return found

    raise FileNotFoundError(
        "Stockfish not found. Place it in stockfish/ or ensure it's in PATH."
    )


def _make_engine(sf_path: str, skill: int, depth: int) -> chess.engine.SimpleEngine:
    """Create a Stockfish engine with given skill level."""
    engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    engine.configure({"Skill Level": skill})
    return engine


# --- Core benchmark functions ---


def play_model_vs_stockfish_game(
    model: ChessTransformerV2,
    device: torch.device,
    engine: chess.engine.SimpleEngine,
    sf_depth: int,
    model_is_white: bool,
    game_seed: int,
) -> str:
    """Play one game, return '1-0', '0-1', or '1/2-1/2'."""
    rng = random.Random(game_seed)
    opening = sample_opening(rng)

    board = chess.Board()
    for uci in opening[3]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)

    with torch.no_grad():
        while not board.is_game_over(claim_draw=True) and board.ply() < 200:
            is_white_turn = board.turn == chess.WHITE

            if is_white_turn == model_is_white:
                board_t, feat_t = preprocess_board(board, device)
                policy, promo, _wdl, _ply = model(board_t, feat_t)
                move = greedy_move_v2(board, policy[0], promo[0])
            else:
                result = engine.play(board, chess.engine.Limit(depth=sf_depth))
                move = result.move

            board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def play_model_vs_model_game(
    model_a: ChessTransformerV2,
    model_b: ChessTransformerV2,
    device: torch.device,
    a_is_white: bool,
    game_seed: int,
) -> str:
    """Play one game between two models, return result string."""
    rng = random.Random(game_seed)
    opening = sample_opening(rng)

    board = chess.Board()
    for uci in opening[3]:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        board.push(move)

    with torch.no_grad():
        while not board.is_game_over(claim_draw=True) and board.ply() < 200:
            is_white_turn = board.turn == chess.WHITE
            active_model = model_a if is_white_turn == a_is_white else model_b

            board_t, feat_t = preprocess_board(board, device)
            policy, promo, _wdl, _ply = active_model(board_t, feat_t)
            move = greedy_move_v2(board, policy[0], promo[0])
            board.push(move)

    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return "1/2-1/2"
    return "1-0" if outcome.winner == chess.WHITE else "0-1"


def benchmark_vs_stockfish(
    model: ChessTransformerV2,
    device: torch.device,
    sf_path: str,
    skill: int,
    num_games: int,
    sf_depth: int = 10,
) -> MatchResult:
    """Run a match: model vs Stockfish at given skill level."""
    engine = _make_engine(sf_path, skill, sf_depth)

    wins, draws, losses = 0, 0, 0

    try:
        for i in range(num_games):
            model_is_white = i % 2 == 0
            result = play_model_vs_stockfish_game(
                model, device, engine, sf_depth, model_is_white, game_seed=i
            )

            if result == "1/2-1/2":
                draws += 1
            elif (result == "1-0") == model_is_white:
                wins += 1
            else:
                losses += 1

            if (i + 1) % 10 == 0 or i + 1 == num_games:
                total = i + 1
                score = (wins + 0.5 * draws) / total
                print(f"  [{total}/{num_games}] +{wins} ={draws} -{losses} ({score:.1%})")
    finally:
        engine.quit()

    return MatchResult(
        wins=wins,
        draws=draws,
        losses=losses,
        label_a="Model",
        label_b=f"Stockfish (skill {skill}, depth {sf_depth})",
    )


def benchmark_vs_model(
    model_a: ChessTransformerV2,
    model_b: ChessTransformerV2,
    device: torch.device,
    num_games: int,
    label_a: str = "Model A",
    label_b: str = "Model B",
) -> MatchResult:
    """Run a match between two models."""
    model_a.eval()
    model_b.eval()
    wins, draws, losses = 0, 0, 0

    for i in range(num_games):
        a_is_white = i % 2 == 0
        result = play_model_vs_model_game(
            model_a, model_b, device, a_is_white, game_seed=i
        )

        if result == "1/2-1/2":
            draws += 1
        elif (result == "1-0") == a_is_white:
            wins += 1
        else:
            losses += 1

        if (i + 1) % 10 == 0 or i + 1 == num_games:
            total = i + 1
            score = (wins + 0.5 * draws) / total
            print(f"  [{total}/{num_games}] +{wins} ={draws} -{losses} ({score:.1%})")

    return MatchResult(
        wins=wins, draws=draws, losses=losses,
        label_a=label_a, label_b=label_b,
    )


def full_benchmark(
    model: ChessTransformerV2,
    device: torch.device,
    sf_path: str,
    num_games: int,
    skills: tuple[int, ...] = (1, 3, 5, 7, 10),
    sf_depth: int = 10,
) -> list[MatchResult]:
    """Run model against multiple Stockfish skill levels."""
    results = []
    for skill in skills:
        print(f"\n--- Stockfish Skill {skill} ({num_games} games) ---")
        result = benchmark_vs_stockfish(
            model, device, sf_path, skill, num_games, sf_depth
        )
        results.append(result)
        print(f"  {result.summary()}")
    return results


# --- CLI ---


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark ChessTransformerV2 models"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # vs-stockfish
    p_sf = subparsers.add_parser("vs-stockfish", help="Model vs Stockfish")
    p_sf.add_argument("model", help="Path to V2 model checkpoint")
    p_sf.add_argument("--skill", type=int, default=5, help="Stockfish skill level (0-20)")
    p_sf.add_argument("--depth", type=int, default=10, help="Stockfish search depth")
    p_sf.add_argument("--games", type=int, default=30, help="Number of games")
    p_sf.add_argument("--sf-path", default=None, help="Path to Stockfish binary")
    p_sf.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    # vs-model
    p_mm = subparsers.add_parser("vs-model", help="Model vs Model")
    p_mm.add_argument("model_a", help="Path to first model")
    p_mm.add_argument("model_b", help="Path to second model")
    p_mm.add_argument("--games", type=int, default=50, help="Number of games")
    p_mm.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    # full
    p_full = subparsers.add_parser("full", help="Full benchmark (multiple SF levels)")
    p_full.add_argument("model", help="Path to V2 model checkpoint")
    p_full.add_argument("--games", type=int, default=20, help="Games per skill level")
    p_full.add_argument("--depth", type=int, default=10, help="Stockfish search depth")
    p_full.add_argument("--skills", default="1,3,5,7,10",
                        help="Comma-separated skill levels (default: 1,3,5,7,10)")
    p_full.add_argument("--sf-path", default=None, help="Path to Stockfish binary")
    p_full.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])

    args = parser.parse_args()
    device = detect_device(args.device)

    if args.command == "vs-stockfish":
        sf_path = args.sf_path or _find_stockfish()
        model, version, _ = load_model(args.model, device)
        model.eval()
        print(f"Model: {args.model} ({version})")
        print(f"Stockfish: skill {args.skill}, depth {args.depth}")
        print(f"Device: {device}")
        print(f"Games: {args.games}\n")

        start = time.time()
        result = benchmark_vs_stockfish(
            model, device, sf_path, args.skill, args.games, args.depth
        )
        elapsed = time.time() - start
        print(f"\n{result.summary()}")
        print(f"Time: {elapsed:.0f}s ({elapsed / args.games:.1f}s/game)")

    elif args.command == "vs-model":
        model_a, ver_a, _ = load_model(args.model_a, device)
        model_b, ver_b, _ = load_model(args.model_b, device)
        model_a.eval()
        model_b.eval()
        label_a = args.model_a.split("/")[-1]
        label_b = args.model_b.split("/")[-1]
        print(f"Model A: {args.model_a} ({ver_a})")
        print(f"Model B: {args.model_b} ({ver_b})")
        print(f"Device: {device}")
        print(f"Games: {args.games}\n")

        start = time.time()
        result = benchmark_vs_model(
            model_a, model_b, device, args.games, label_a, label_b
        )
        elapsed = time.time() - start
        print(f"\n{result.summary()}")
        print(f"Time: {elapsed:.0f}s ({elapsed / args.games:.1f}s/game)")

    elif args.command == "full":
        sf_path = args.sf_path or _find_stockfish()
        model, version, _ = load_model(args.model, device)
        model.eval()
        skills = tuple(int(s) for s in args.skills.split(","))
        print(f"Model: {args.model} ({version})")
        print(f"Stockfish levels: {skills}")
        print(f"Device: {device}")
        print(f"Games per level: {args.games}")

        start = time.time()
        results = full_benchmark(
            model, device, sf_path, args.games, skills, args.depth
        )
        elapsed = time.time() - start

        print(f"\n{'='*60}")
        print("  BENCHMARK RESULTS")
        print(f"{'='*60}")
        for r in results:
            print(f"  {r.summary()}")
        print(f"{'='*60}")
        print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
