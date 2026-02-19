"""Tests for model_utils — device detection, preprocessing, loss computation."""

import chess
import torch
import pytest
from chess_loader import PIECE_TO_INDEX
from chessformer import ChessTransformerV2
from model_utils import (
    compute_loss_v2,
    detect_device,
    load_model,
    preprocess_board,
    preprocess_board_v1,
)


@pytest.fixture
def small_model():
    """Small V2 model for fast tests."""
    return ChessTransformerV2(
        d_model=64, nhead=4, d_hid=128, nlayers=2, d_policy=32, dropout=0.0,
    )


class TestDetectDevice:
    def test_cpu_explicit(self):
        """'cpu' → torch.device('cpu')."""
        assert detect_device("cpu") == torch.device("cpu")

    def test_auto_returns_device(self):
        """'auto' resolves to a valid torch.device."""
        d = detect_device("auto")
        assert isinstance(d, torch.device)

    def test_invalid_device_raises(self):
        """Invalid device string → RuntimeError."""
        with pytest.raises(RuntimeError):
            detect_device("nonexistent_device_xyz")


class TestPreprocessBoard:
    def test_v2_shapes_and_types(self):
        """V2: board[1,64] long, features[1,14] float."""
        board = chess.Board()
        board_t, feat_t = preprocess_board(board, torch.device("cpu"))
        assert board_t.shape == (1, 64)
        assert feat_t.shape == (1, 14)
        assert board_t.dtype == torch.long
        assert feat_t.dtype == torch.float32

    def test_v2_board_values_are_valid_piece_indices(self):
        """All values should be in [0, 12] (PIECE_TO_INDEX range)."""
        board = chess.Board()
        board_t, _ = preprocess_board(board, torch.device("cpu"))
        assert (board_t >= 0).all() and (board_t <= 12).all()

    def test_v2_starting_position_has_correct_pieces(self):
        """Starting position: first 8 squares = rank 8 = opponent back rank."""
        board = chess.Board()
        board_t, _ = preprocess_board(board, torch.device("cpu"))
        # Board display: rank 8 (top) first → rnbqkbnr (opponent pieces, lowercase)
        expected_rank8 = [PIECE_TO_INDEX[c] for c in "rnbqkbnr"]
        assert board_t[0, :8].tolist() == expected_rank8

    def test_v2_material_balance_zero_at_start(self):
        """Starting position has equal material → feature[0] ≈ 0."""
        board = chess.Board()
        _, feat_t = preprocess_board(board, torch.device("cpu"))
        assert feat_t[0, 0].item() == pytest.approx(0.0)

    def test_v2_different_for_black(self):
        """After e2e4, board from black's perspective should be flipped."""
        board = chess.Board()
        white_board, _ = preprocess_board(board, torch.device("cpu"))
        board.push_san("e4")
        black_board, _ = preprocess_board(board, torch.device("cpu"))
        assert not torch.equal(white_board, black_board)

    def test_v1_shape_and_values(self):
        """V1: board[1,64] long, values in [0, 12]."""
        board = chess.Board()
        board_t = preprocess_board_v1(board, torch.device("cpu"))
        assert board_t.shape == (1, 64)
        assert board_t.dtype == torch.long
        assert (board_t >= 0).all() and (board_t <= 12).all()

    def test_v1_different_for_black(self):
        """V1 board flips for black's turn."""
        board = chess.Board()
        white_board = preprocess_board_v1(board, torch.device("cpu"))
        board.push_san("e4")
        black_board = preprocess_board_v1(board, torch.device("cpu"))
        assert not torch.equal(white_board, black_board)


class TestLoadModel:
    def test_v2_roundtrip(self, small_model, tmp_path):
        """Save and reload V2 checkpoint — weights should match."""
        cfg = {"d_model": 64, "nhead": 4, "d_hid": 128, "nlayers": 2, "d_policy": 32}
        path = tmp_path / "test_v2.pth"
        torch.save(
            {"version": "v2", "state_dict": small_model.state_dict(), "config": cfg},
            path,
        )
        loaded, version, loaded_cfg = load_model(str(path), torch.device("cpu"))
        assert version == "v2"
        assert loaded_cfg == cfg
        assert isinstance(loaded, ChessTransformerV2)
        # Verify weights actually match
        for (n1, p1), (n2, p2) in zip(
            small_model.named_parameters(), loaded.named_parameters()
        ):
            assert torch.equal(p1.cpu(), p2.cpu()), f"Weight mismatch in {n1}"

    def test_v2_produces_same_output(self, small_model, tmp_path):
        """Loaded model should produce identical output for same input."""
        cfg = {"d_model": 64, "nhead": 4, "d_hid": 128, "nlayers": 2, "d_policy": 32}
        path = tmp_path / "test_v2.pth"
        torch.save(
            {"version": "v2", "state_dict": small_model.state_dict(), "config": cfg},
            path,
        )
        loaded, _, _ = load_model(str(path), torch.device("cpu"))

        board = torch.randint(0, 13, (1, 64))
        feat = torch.randn(1, 14)
        small_model.eval()
        loaded.eval()
        with torch.no_grad():
            p1, _, w1, _ = small_model(board, feat)
            p2, _, w2, _ = loaded(board, feat)
        assert torch.equal(p1, p2)
        assert torch.equal(w1, w2)

    def test_invalid_checkpoint_raises(self, tmp_path):
        """Non-model file → ValueError."""
        path = tmp_path / "bad.pth"
        torch.save("not a model", path)
        with pytest.raises(ValueError, match="Cannot load model"):
            load_model(str(path), torch.device("cpu"))


class TestComputeLossV2:
    def test_loss_is_positive_scalar(self, small_model):
        """Loss is a scalar > 0 (policy CE + WDL CE)."""
        B = 4
        boards = torch.randint(0, 13, (B, 64))
        features = torch.randn(B, 14)
        from_sq = torch.randint(0, 64, (B,))
        to_sq = torch.randint(0, 64, (B,))
        wdl = torch.softmax(torch.randn(B, 3), dim=-1)

        loss = compute_loss_v2(small_model, boards, features, from_sq, to_sq, wdl)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_backward_updates_parameters(self, small_model):
        """Loss.backward() produces gradients (promo/ply heads excluded — unused by loss)."""
        B = 4
        boards = torch.randint(0, 13, (B, 64))
        features = torch.randn(B, 14)
        from_sq = torch.randint(0, 64, (B,))
        to_sq = torch.randint(0, 64, (B,))
        wdl = torch.softmax(torch.randn(B, 3), dim=-1)

        loss = compute_loss_v2(small_model, boards, features, from_sq, to_sq, wdl)
        loss.backward()

        # Params that SHOULD get gradient (everything except promo/ply heads)
        no_grad = [
            n for n, p in small_model.named_parameters()
            if (p.grad is None or p.grad.abs().sum() == 0)
            and "promo" not in n and "ply" not in n
        ]
        assert len(no_grad) == 0, f"Unexpected params without gradient: {no_grad}"

        # Promo/ply heads should NOT get gradient (compute_loss_v2 ignores them)
        promo_ply_with_grad = [
            n for n, p in small_model.named_parameters()
            if ("promo" in n or "ply" in n)
            and p.grad is not None and p.grad.abs().sum() > 0
        ]
        assert len(promo_ply_with_grad) == 0, (
            f"Promo/ply params shouldn't get gradient: {promo_ply_with_grad}"
        )

    def test_loss_decreases_with_training_step(self, small_model):
        """One optimizer step should reduce the loss."""
        B = 8
        boards = torch.randint(0, 13, (B, 64))
        features = torch.randn(B, 14)
        from_sq = torch.randint(0, 64, (B,))
        to_sq = torch.randint(0, 64, (B,))
        wdl = torch.softmax(torch.randn(B, 3), dim=-1)

        loss_before = compute_loss_v2(
            small_model, boards, features, from_sq, to_sq, wdl
        ).item()

        optimizer = torch.optim.SGD(small_model.parameters(), lr=0.01)
        for _ in range(5):
            optimizer.zero_grad()
            loss = compute_loss_v2(
                small_model, boards, features, from_sq, to_sq, wdl
            )
            loss.backward()
            optimizer.step()

        loss_after = compute_loss_v2(
            small_model, boards, features, from_sq, to_sq, wdl
        ).item()
        assert loss_after < loss_before, (
            f"Loss should decrease: {loss_before:.4f} → {loss_after:.4f}"
        )
