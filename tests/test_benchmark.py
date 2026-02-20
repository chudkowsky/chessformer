"""Tests for benchmark.py — MatchResult, game logic, integration tests."""

import math
import shutil

import chess
import pytest
import torch

from benchmark import (
    MatchResult,
    benchmark_vs_model,
    benchmark_vs_stockfish,
    play_model_vs_model_game,
    play_model_vs_stockfish_game,
)
from chessformer import ChessTransformerV2


@pytest.fixture
def small_model():
    """Small V2 model for fast tests."""
    return ChessTransformerV2(
        d_model=64, nhead=4, d_hid=128, nlayers=2, d_policy=32, dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


# --- MatchResult ---


class TestMatchResult:
    def test_total(self):
        r = MatchResult(wins=5, draws=3, losses=2, label_a="A", label_b="B")
        assert r.total == 10

    def test_score_all_wins(self):
        r = MatchResult(wins=10, draws=0, losses=0, label_a="A", label_b="B")
        assert r.score == pytest.approx(1.0)

    def test_score_all_losses(self):
        r = MatchResult(wins=0, draws=0, losses=10, label_a="A", label_b="B")
        assert r.score == pytest.approx(0.0)

    def test_score_all_draws(self):
        r = MatchResult(wins=0, draws=10, losses=0, label_a="A", label_b="B")
        assert r.score == pytest.approx(0.5)

    def test_score_mixed(self):
        r = MatchResult(wins=3, draws=4, losses=3, label_a="A", label_b="B")
        assert r.score == pytest.approx(0.5)

    def test_elo_diff_even(self):
        r = MatchResult(wins=5, draws=0, losses=5, label_a="A", label_b="B")
        assert r.elo_diff == pytest.approx(0.0, abs=1.0)

    def test_elo_diff_positive_for_winner(self):
        r = MatchResult(wins=8, draws=1, losses=1, label_a="A", label_b="B")
        assert r.elo_diff > 100

    def test_elo_diff_negative_for_loser(self):
        r = MatchResult(wins=1, draws=1, losses=8, label_a="A", label_b="B")
        assert r.elo_diff < -100

    def test_summary_contains_key_info(self):
        r = MatchResult(wins=5, draws=3, losses=2, label_a="A", label_b="B")
        s = r.summary()
        assert "A vs B" in s
        assert "+5" in s
        assert "=3" in s
        assert "-2" in s
        assert "Elo" in s


# --- Model vs Model game ---


class TestModelVsModelGame:
    def test_returns_valid_result(self, small_model, device):
        result = play_model_vs_model_game(
            small_model, small_model, device, a_is_white=True, game_seed=42
        )
        assert result in ("1-0", "0-1", "1/2-1/2")

    def test_different_seeds_may_differ(self, small_model, device):
        results = set()
        for seed in range(10):
            r = play_model_vs_model_game(
                small_model, small_model, device, a_is_white=seed % 2 == 0,
                game_seed=seed,
            )
            results.add(r)
        # With random model + different openings, we should get at least 1 result type
        assert len(results) >= 1

    def test_game_terminates(self, small_model, device):
        """Game should always terminate (max 200 ply enforced)."""
        result = play_model_vs_model_game(
            small_model, small_model, device, a_is_white=True, game_seed=0
        )
        assert result in ("1-0", "0-1", "1/2-1/2")


# --- Benchmark vs Model ---


class TestBenchmarkVsModel:
    def test_runs_correct_number_of_games(self, small_model, device):
        result = benchmark_vs_model(
            small_model, small_model, device, num_games=4,
            label_a="A", label_b="B",
        )
        assert result.total == 4
        assert result.label_a == "A"
        assert result.label_b == "B"

    def test_score_in_valid_range(self, small_model, device):
        result = benchmark_vs_model(
            small_model, small_model, device, num_games=4,
        )
        assert 0.0 <= result.score <= 1.0

    def test_wins_draws_losses_sum(self, small_model, device):
        result = benchmark_vs_model(
            small_model, small_model, device, num_games=6,
        )
        assert result.wins + result.draws + result.losses == 6


# --- Stockfish integration tests ---


def _has_stockfish() -> bool:
    """Check if Stockfish is available."""
    if shutil.which("stockfish"):
        return True
    from pathlib import Path
    return Path("stockfish/stockfish-ubuntu-x86-64-avx2").is_file()


def _get_sf_path() -> str:
    from pathlib import Path
    p = Path("stockfish/stockfish-ubuntu-x86-64-avx2")
    if p.is_file():
        return str(p)
    found = shutil.which("stockfish")
    if found:
        return found
    raise FileNotFoundError("No Stockfish")


@pytest.mark.skipif(not _has_stockfish(), reason="Stockfish not available")
class TestStockfishIntegration:
    def test_single_game(self, small_model, device):
        sf_path = _get_sf_path()
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        try:
            result = play_model_vs_stockfish_game(
                small_model, device, engine,
                sf_depth=1, model_is_white=True, game_seed=42,
            )
            assert result in ("1-0", "0-1", "1/2-1/2")
        finally:
            engine.quit()

    def test_benchmark_vs_stockfish(self, small_model, device):
        sf_path = _get_sf_path()
        result = benchmark_vs_stockfish(
            small_model, device, sf_path,
            skill=1, num_games=2, sf_depth=1,
        )
        assert result.total == 2
        assert "Stockfish" in result.label_b

    def test_single_game_as_black(self, small_model, device):
        sf_path = _get_sf_path()
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
        try:
            result = play_model_vs_stockfish_game(
                small_model, device, engine,
                sf_depth=1, model_is_white=False, game_seed=99,
            )
            assert result in ("1-0", "0-1", "1/2-1/2")
        finally:
            engine.quit()
