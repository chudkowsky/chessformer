"""Tests for mcts.py — MCTS search, node operations, integration with model."""

import chess
import random
import torch
import pytest

from chessformer import ChessTransformerV2
from mcts import MCTSNode, MCTS


@pytest.fixture
def small_model():
    """Small V2 model for fast tests."""
    return ChessTransformerV2(
        d_model=64, nhead=4, d_hid=128, nlayers=2, d_policy=32, dropout=0.0,
    )


@pytest.fixture
def device():
    return torch.device("cpu")


class TestMCTSNode:
    def test_q_value_zero_visits(self):
        node = MCTSNode(board=chess.Board())
        assert node.q_value() == 0.0

    def test_q_value_after_updates(self):
        node = MCTSNode(board=chess.Board(), visit_count=4, value_sum=2.0)
        assert node.q_value() == pytest.approx(0.5)

    def test_is_expanded_empty(self):
        node = MCTSNode(board=chess.Board())
        assert not node.is_expanded()

    def test_is_expanded_with_children(self):
        parent = MCTSNode(board=chess.Board())
        child = MCTSNode(board=chess.Board(), parent=parent)
        parent.children[chess.Move.from_uci("e2e4")] = child
        assert parent.is_expanded()


class TestMCTSSearch:
    def test_returns_legal_moves(self, small_model, device):
        mcts = MCTS(small_model, device, num_simulations=10)
        board = chess.Board()
        policy, wdl = mcts.search(board)
        legal = set(board.legal_moves)
        for move in policy:
            assert move in legal

    def test_visit_counts_sum_to_one(self, small_model, device):
        mcts = MCTS(small_model, device, num_simulations=20)
        board = chess.Board()
        policy, _wdl = mcts.search(board)
        assert sum(policy.values()) == pytest.approx(1.0, abs=0.01)

    def test_nonempty_from_starting_position(self, small_model, device):
        mcts = MCTS(small_model, device, num_simulations=5)
        policy, _wdl = mcts.search(chess.Board())
        assert len(policy) > 0

    def test_empty_for_checkmate(self, small_model, device):
        """Scholar's mate position — no legal moves."""
        board = chess.Board()
        for uci in ["f2f3", "e7e5", "g2g4", "d8h4"]:
            board.push(chess.Move.from_uci(uci))
        assert board.is_checkmate()
        mcts = MCTS(small_model, device, num_simulations=5)
        policy, _wdl = mcts.search(board)
        assert len(policy) == 0

    def test_wdl_is_tuple_of_three(self, small_model, device):
        mcts = MCTS(small_model, device, num_simulations=5)
        _policy, wdl = mcts.search(chess.Board())
        assert len(wdl) == 3
        assert sum(wdl) == pytest.approx(1.0, abs=0.1)

    def test_more_sims_produces_valid_policy(self, small_model, device):
        """More simulations still produce a valid, non-empty policy."""
        board = chess.Board()
        policy, _ = MCTS(small_model, device, num_simulations=50).search(board)
        assert len(policy) > 0
        assert sum(policy.values()) == pytest.approx(1.0, abs=0.01)
