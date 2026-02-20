"""Tests for trajectory_loader.py — TrajectoryDataset, extract_trajectories."""

import torch
import pytest
from chess_loader import PIECE_TO_INDEX, compute_features
from trajectory_loader import TrajectoryDataset


# Starting position and after 1.e4 (from white's perspective)
STARTING_BOARD = "RNBQKBNRPPPPPPPP................................pppppppprnbqkbnr"
AFTER_E4 = "RNBQKBNRPPPP.PPP............P...................pppppppprnbqkbnr"
# After 1...e5 (from black's perspective — board is flipped)
AFTER_E4_E5 = "RNBQKBNRPPPP.PPP.............p..................pppp.ppprnbqkbnr"


@pytest.fixture
def sample_pairs():
    """Two trajectory pairs with known board strings."""
    return [(STARTING_BOARD, AFTER_E4), (STARTING_BOARD, AFTER_E4_E5)]


class TestTrajectoryDataset:
    def test_length(self, sample_pairs):
        """Dataset length matches number of trajectory pairs."""
        ds = TrajectoryDataset(sample_pairs)
        assert len(ds) == 2

    def test_empty_dataset(self):
        """Empty input → empty dataset."""
        ds = TrajectoryDataset([])
        assert len(ds) == 0

    def test_output_shapes_and_types(self, sample_pairs):
        """Each item: (cur_board[64], cur_feat[14], fut_board[64], fut_feat[14])."""
        ds = TrajectoryDataset(sample_pairs)
        cur_board, cur_feat, fut_board, fut_feat = ds[0]
        assert cur_board.shape == (64,) and cur_board.dtype == torch.long
        assert cur_feat.shape == (14,) and cur_feat.dtype == torch.float32
        assert fut_board.shape == (64,) and fut_board.dtype == torch.long
        assert fut_feat.shape == (14,) and fut_feat.dtype == torch.float32

    def test_board_values_in_piece_index_range(self, sample_pairs):
        """Board values must be valid PIECE_TO_INDEX indices [0, 12]."""
        ds = TrajectoryDataset(sample_pairs)
        for i in range(len(ds)):
            cur_board, _, fut_board, _ = ds[i]
            assert (cur_board >= 0).all() and (cur_board <= 12).all(), (
                f"Pair {i}: current board has values outside [0, 12]"
            )
            assert (fut_board >= 0).all() and (fut_board <= 12).all(), (
                f"Pair {i}: future board has values outside [0, 12]"
            )

    def test_current_and_future_differ(self, sample_pairs):
        """Current and future boards in a pair should be different positions."""
        ds = TrajectoryDataset(sample_pairs)
        cur_board, _, fut_board, _ = ds[0]
        assert not torch.equal(cur_board, fut_board), (
            "Current and future boards should differ (a move was played)"
        )

    def test_known_encoding_starting_position(self):
        """Starting position encoding matches PIECE_TO_INDEX manually."""
        ds = TrajectoryDataset([(STARTING_BOARD, AFTER_E4)])
        cur_board, _, _, _ = ds[0]
        # First square is 'R' = 4, second 'N' = 2, etc.
        expected_first_8 = [
            PIECE_TO_INDEX[c] for c in "RNBQKBNR"
        ]
        assert cur_board[:8].tolist() == expected_first_8

    def test_known_encoding_future_has_moved_pawn(self):
        """After e4, pawn should have moved from rank 2 to rank 4."""
        ds = TrajectoryDataset([(STARTING_BOARD, AFTER_E4)])
        _, _, fut_board, _ = ds[0]
        # In AFTER_E4, index 12 (e2) is '.' = 0, index 28 (e4) is 'P' = 1
        assert fut_board[12].item() == PIECE_TO_INDEX['.']
        assert fut_board[28].item() == PIECE_TO_INDEX['P']

    def test_features_reflect_material(self):
        """Material balance feature should match board content."""
        ds = TrajectoryDataset([(STARTING_BOARD, AFTER_E4)])
        _, cur_feat, _, fut_feat = ds[0]
        # Starting position: material balance = 0 (equal pieces)
        expected_material = compute_features(STARTING_BOARD)[0]
        assert cur_feat[0].item() == pytest.approx(expected_material)
        # After e4, material is still equal (no captures)
        assert fut_feat[0].item() == pytest.approx(0.0)

    def test_different_pairs_have_different_futures(self, sample_pairs):
        """Different trajectory pairs should have different future boards."""
        ds = TrajectoryDataset(sample_pairs)
        _, _, fut1, _ = ds[0]
        _, _, fut2, _ = ds[1]
        assert not torch.equal(fut1, fut2)

    def test_all_64_squares_are_valid_pieces(self):
        """Every square in encoded board maps to a valid piece character."""
        valid_indices = set(PIECE_TO_INDEX.values())
        ds = TrajectoryDataset([(STARTING_BOARD, AFTER_E4)])
        cur_board, _, fut_board, _ = ds[0]
        for sq in range(64):
            assert cur_board[sq].item() in valid_indices
            assert fut_board[sq].item() in valid_indices
