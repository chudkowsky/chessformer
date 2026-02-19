"""Tests for trajectory_loader.py — TrajectoryDataset, extract_trajectories."""

import torch
import pytest
from trajectory_loader import TrajectoryDataset


@pytest.fixture
def sample_pairs():
    """Two trajectory pairs with known board strings."""
    # Simple 64-char board strings (all dots = empty board)
    current1 = "RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr"
    future1 = "RNBQKBNRPPPP.PPP............P...................pppppppprnbqkbnr"
    current2 = "RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr"
    future2 = "RNBQKBNRPPPPPPPP................n...............pppp.ppprnbqkb.r"
    return [(current1, future1), (current2, future2)]


class TestTrajectoryDataset:
    def test_length(self, sample_pairs):
        ds = TrajectoryDataset(sample_pairs)
        assert len(ds) == 2

    def test_output_shapes(self, sample_pairs):
        ds = TrajectoryDataset(sample_pairs)
        cur_board, cur_feat, fut_board, fut_feat = ds[0]
        assert cur_board.shape == (64,)
        assert cur_feat.shape == (14,)
        assert fut_board.shape == (64,)
        assert fut_feat.shape == (14,)

    def test_board_dtype(self, sample_pairs):
        ds = TrajectoryDataset(sample_pairs)
        cur_board, _, fut_board, _ = ds[0]
        assert cur_board.dtype == torch.long
        assert fut_board.dtype == torch.long

    def test_features_dtype(self, sample_pairs):
        ds = TrajectoryDataset(sample_pairs)
        _, cur_feat, _, fut_feat = ds[0]
        assert cur_feat.dtype == torch.float32
        assert fut_feat.dtype == torch.float32

    def test_different_pairs_different_futures(self, sample_pairs):
        """Different trajectory pairs should have different future boards."""
        ds = TrajectoryDataset(sample_pairs)
        _, _, fut1, _ = ds[0]
        _, _, fut2, _ = ds[1]
        assert not torch.equal(fut1, fut2)

    def test_board_values_in_range(self, sample_pairs):
        ds = TrajectoryDataset(sample_pairs)
        cur_board, _, fut_board, _ = ds[0]
        assert (cur_board >= 0).all() and (cur_board <= 12).all()
        assert (fut_board >= 0).all() and (fut_board <= 12).all()

    def test_empty_dataset(self):
        ds = TrajectoryDataset([])
        assert len(ds) == 0
