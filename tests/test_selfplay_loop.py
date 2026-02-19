"""Tests for selfplay_loop.py — game generation, data format, training integration."""

import chess
import random
from unittest.mock import patch, MagicMock
import torch
import pytest

from chessformer import ChessTransformerV2
from selfplay_loop import (
    SelfPlayConfig,
    preprocess_board,
    generate_game,
    mcts_sample_move,
    parse_temp_schedule,
    get_temperature,
)
from mcts import MCTS
from policy import sample_move_v2, greedy_move_v2


@pytest.fixture
def small_model():
    """Small V2 model for fast tests."""
    return ChessTransformerV2(
        d_model=64, nhead=4, d_hid=128, nlayers=2, d_policy=32, dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def default_config():
    return SelfPlayConfig(
        model_path="test.pth",
        output_dir="/tmp/test_selfplay",
        generations=1,
        games_per_gen=1,
        epochs_per_gen=1,
        batch_size=32,
        lr=1e-4,
        temp_schedule=[(1.0, 999)],
        max_moves=50,
        resign_threshold=0.99,
        resign_count=100,
        draw_threshold=0.99,
        draw_count=100,
        buffer_size=1,
        use_diffusion=False,
        mix_supervised=None,
        mix_ratio=0.0,
        eval_games=0,
        device="cpu",
        mcts_sims=0,
        cpuct=1.25,
    )


# --- Temp schedule ---


class TestTempSchedule:
    def test_parse_simple(self):
        result = parse_temp_schedule("1.5:10,1.0:25,0.3:999")
        assert result == [(1.5, 10), (1.0, 25), (0.3, 999)]

    def test_parse_single(self):
        result = parse_temp_schedule("1.0:999")
        assert result == [(1.0, 999)]

    def test_get_temperature_first_range(self):
        schedule = [(1.5, 10), (1.0, 25), (0.3, 999)]
        assert get_temperature(schedule, 0) == 1.5
        assert get_temperature(schedule, 9) == 1.5

    def test_get_temperature_middle_range(self):
        schedule = [(1.5, 10), (1.0, 25), (0.3, 999)]
        assert get_temperature(schedule, 10) == 1.0
        assert get_temperature(schedule, 24) == 1.0

    def test_get_temperature_last_range(self):
        schedule = [(1.5, 10), (1.0, 25), (0.3, 999)]
        assert get_temperature(schedule, 25) == 0.3
        assert get_temperature(schedule, 100) == 0.3


# --- Preprocess ---


class TestPreprocessBoard:
    def test_output_shapes(self, device):
        board = chess.Board()
        board_t, feat_t = preprocess_board(board, device)
        assert board_t.shape == (1, 64)
        assert feat_t.shape == (1, 14)
        assert board_t.dtype == torch.long
        assert feat_t.dtype == torch.float32

    def test_after_move(self, device):
        board = chess.Board()
        board.push_san("e4")
        board_t, feat_t = preprocess_board(board, device)
        assert board_t.shape == (1, 64)
        assert feat_t.shape == (1, 14)


# --- sample_move_v2 ---


class TestSampleMoveV2:
    def test_returns_legal_move(self, small_model, device):
        board = chess.Board()
        small_model.eval()
        board_t, feat_t = preprocess_board(board, device)
        with torch.no_grad():
            policy, promo, _, _ = small_model(board_t, feat_t)
        move, log_prob = sample_move_v2(board, policy[0], promo[0], temperature=1.0)
        assert move in board.legal_moves

    def test_greedy_at_zero_temp(self, small_model, device):
        board = chess.Board()
        small_model.eval()
        board_t, feat_t = preprocess_board(board, device)
        with torch.no_grad():
            policy, promo, _, _ = small_model(board_t, feat_t)
        # Greedy should be deterministic
        moves = [
            sample_move_v2(board, policy[0], promo[0], temperature=0.0)[0]
            for _ in range(5)
        ]
        assert all(m == moves[0] for m in moves)

    def test_greedy_matches_greedy_move_v2(self, small_model, device):
        board = chess.Board()
        small_model.eval()
        board_t, feat_t = preprocess_board(board, device)
        with torch.no_grad():
            policy, promo, _, _ = small_model(board_t, feat_t)
        greedy = greedy_move_v2(board, policy[0], promo[0])
        sampled, _ = sample_move_v2(board, policy[0], promo[0], temperature=0.0)
        assert sampled == greedy

    def test_log_prob_is_scalar(self, small_model, device):
        board = chess.Board()
        small_model.eval()
        board_t, feat_t = preprocess_board(board, device)
        with torch.no_grad():
            policy, promo, _, _ = small_model(board_t, feat_t)
        _, log_prob = sample_move_v2(board, policy[0], promo[0])
        assert log_prob.dim() == 0
        assert log_prob.item() <= 0.0


# --- Game generation ---


class TestGenerateGame:
    def test_produces_training_lines(self, small_model, device, default_config):
        from openings import sample_opening

        opening = sample_opening(random.Random(42))
        lines = generate_game(
            model=small_model,
            device=device,
            opening=opening,
            config=default_config,
            rng=random.Random(42),
        )
        assert len(lines) > 0

    def test_training_line_format(self, small_model, device, default_config):
        from openings import sample_opening

        opening = sample_opening(random.Random(42))
        lines = generate_game(
            model=small_model,
            device=device,
            opening=opening,
            config=default_config,
            rng=random.Random(42),
        )
        for line in lines:
            parts = line.split()
            assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {line}"
            board_str, uci, result = parts
            assert len(board_str) == 64
            assert all(c in ".PNBRQKpnbrqk" for c in board_str)
            result_f = float(result)
            assert result_f in (0.0, 0.5, 1.0)

    def test_max_moves_respected(self, small_model, device):
        from openings import sample_opening

        config = SelfPlayConfig(
            model_path="test.pth",
            output_dir="/tmp/test_selfplay",
            generations=1,
            games_per_gen=1,
            epochs_per_gen=1,
            batch_size=32,
            lr=1e-4,
            temp_schedule=[(1.0, 999)],
            max_moves=15,
            resign_threshold=0.99,
            resign_count=100,
            draw_threshold=0.99,
            draw_count=100,
            buffer_size=1,
            use_diffusion=False,
            mix_supervised=None,
            mix_ratio=0.0,
            eval_games=0,
            device="cpu",
            mcts_sims=0,
            cpuct=1.25,
        )
        opening = sample_opening(random.Random(42))
        lines = generate_game(
            model=small_model,
            device=device,
            opening=opening,
            config=config,
            rng=random.Random(42),
        )
        # max_moves=15 means max 15 total ply including opening
        assert len(lines) <= 15


# --- Config ---


class TestMctsSampleMove:
    """Tests for mcts_sample_move() helper."""

    @pytest.fixture
    def sample_policy(self):
        return {
            chess.Move.from_uci("e2e4"): 0.6,
            chess.Move.from_uci("d2d4"): 0.3,
            chess.Move.from_uci("g1f3"): 0.1,
        }

    def test_greedy_picks_highest(self, sample_policy):
        move = mcts_sample_move(sample_policy, temperature=0.0, rng=random.Random(42))
        assert move == chess.Move.from_uci("e2e4")

    def test_greedy_deterministic(self, sample_policy):
        moves = [
            mcts_sample_move(sample_policy, temperature=0.0, rng=random.Random(i))
            for i in range(10)
        ]
        assert all(m == moves[0] for m in moves)

    def test_sampling_returns_valid_move(self, sample_policy):
        for seed in range(20):
            move = mcts_sample_move(sample_policy, temperature=1.0, rng=random.Random(seed))
            assert move in sample_policy

    def test_single_move_returns_it(self):
        policy = {chess.Move.from_uci("a2a3"): 1.0}
        move = mcts_sample_move(policy, temperature=1.0, rng=random.Random(0))
        assert move == chess.Move.from_uci("a2a3")

    def test_high_temp_explores_more(self, sample_policy):
        """High temperature should produce more variety than greedy."""
        moves_seen = set()
        for seed in range(50):
            move = mcts_sample_move(sample_policy, temperature=2.0, rng=random.Random(seed))
            moves_seen.add(move)
        assert len(moves_seen) >= 2

    def test_with_real_mcts_output(self, small_model, device):
        """mcts_sample_move works with actual MCTS search output."""
        board = chess.Board()
        mcts = MCTS(small_model, device, num_simulations=10)
        visit_policy, _wdl = mcts.search(board)
        move = mcts_sample_move(visit_policy, temperature=1.0, rng=random.Random(42))
        assert move in board.legal_moves


# --- MCTS + generate_game integration ---


@pytest.fixture
def mcts_config():
    return SelfPlayConfig(
        model_path="test.pth",
        output_dir="/tmp/test_selfplay_mcts",
        generations=1,
        games_per_gen=1,
        epochs_per_gen=1,
        batch_size=32,
        lr=1e-4,
        temp_schedule=[(1.0, 999)],
        max_moves=30,
        resign_threshold=0.99,
        resign_count=100,
        draw_threshold=0.99,
        draw_count=100,
        buffer_size=1,
        use_diffusion=False,
        mix_supervised=None,
        mix_ratio=0.0,
        eval_games=0,
        device="cpu",
        mcts_sims=5,
        cpuct=1.25,
    )


class TestGenerateGameMCTS:
    """generate_game() with MCTS enabled — verify MCTS is actually used."""

    def test_mcts_search_is_called(self, small_model, device, mcts_config):
        """Verify MCTS.search() is actually invoked during game generation."""
        from openings import sample_opening

        opening = sample_opening(random.Random(42))
        original_search = MCTS.search
        search_calls: list[chess.Board] = []

        def tracking_search(self, board):
            search_calls.append(board)
            return original_search(self, board)

        with patch.object(MCTS, "search", tracking_search):
            generate_game(
                model=small_model,
                device=device,
                opening=opening,
                config=mcts_config,
                rng=random.Random(42),
            )
        assert len(search_calls) > 0, "MCTS.search() was never called"

    def test_sample_move_v2_not_called(self, small_model, device, mcts_config):
        """When MCTS is on, raw sample_move_v2 should NOT be used."""
        from openings import sample_opening

        opening = sample_opening(random.Random(42))
        with patch("selfplay_loop.sample_move_v2") as mock_sample:
            generate_game(
                model=small_model,
                device=device,
                opening=opening,
                config=mcts_config,
                rng=random.Random(42),
            )
            mock_sample.assert_not_called()

    def test_training_line_format_with_mcts(self, small_model, device, mcts_config):
        """MCTS game produces valid training lines."""
        from openings import sample_opening

        opening = sample_opening(random.Random(42))
        lines = generate_game(
            model=small_model,
            device=device,
            opening=opening,
            config=mcts_config,
            rng=random.Random(42),
        )
        assert len(lines) > 0
        for line in lines:
            parts = line.split()
            assert len(parts) == 3, f"Expected 3 parts, got {len(parts)}: {line}"
            board_str, uci, result = parts
            assert len(board_str) == 64
            assert all(c in ".PNBRQKpnbrqk" for c in board_str)
            result_f = float(result)
            assert result_f in (0.0, 0.5, 1.0)

    def test_non_mcts_uses_sample_move_v2(self, small_model, device, default_config):
        """When mcts_sims=0, sample_move_v2 IS called (not MCTS)."""
        from openings import sample_opening

        opening = sample_opening(random.Random(42))
        with patch("selfplay_loop.sample_move_v2", wraps=sample_move_v2) as mock_sample:
            generate_game(
                model=small_model,
                device=device,
                opening=opening,
                config=default_config,
                rng=random.Random(42),
            )
            assert mock_sample.call_count > 0, "sample_move_v2 was never called"


# --- Config ---


class TestSelfPlayConfig:
    def test_frozen(self, default_config):
        with pytest.raises(AttributeError):
            default_config.generations = 5

    def test_mcts_fields_present(self, default_config):
        assert default_config.mcts_sims == 0
        assert default_config.cpuct == 1.25
