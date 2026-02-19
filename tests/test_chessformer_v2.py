"""Tests for ChessTransformerV2 — forward pass, shapes, gradients, heads."""

import torch
import pytest
from chessformer import ChessTransformerV2


@pytest.fixture
def model():
    """Small V2 model for fast tests."""
    return ChessTransformerV2(
        d_model=64, nhead=4, d_hid=128, nlayers=2, d_policy=32, dropout=0.0,
    )


@pytest.fixture
def batch():
    """Fake batch: (board[B,64], features[B,14])."""
    B = 3
    board = torch.randint(0, 13, (B, 64))
    features = torch.randn(B, 14)
    return board, features


class TestForwardPass:
    def test_output_shapes(self, model, batch):
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        B = board.shape[0]
        assert policy.shape == (B, 64, 64)
        assert promo.shape == (B, 64, 4)
        assert wdl.shape == (B, 3)
        assert ply.shape == (B, 1)

    def test_without_features(self, model, batch):
        """Model works without auxiliary features (features=None)."""
        board, _ = batch
        policy, promo, wdl, ply = model(board, features=None)
        assert policy.shape == (board.shape[0], 64, 64)

    def test_wdl_sums_to_one(self, model, batch):
        board, features = batch
        _, _, wdl, _ = model(board, features)
        sums = wdl.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_ply_non_negative(self, model, batch):
        board, features = batch
        _, _, _, ply = model(board, features)
        assert (ply >= 0).all()

    def test_no_nan(self, model, batch):
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        for name, t in [("policy", policy), ("promo", promo), ("wdl", wdl), ("ply", ply)]:
            assert not torch.isnan(t).any(), f"NaN in {name}"


class TestGradients:
    def test_backward_pass(self, model, batch):
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        loss = policy.sum() + promo.sum() + wdl.sum() + ply.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_all_params_receive_gradient(self, model, batch):
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        # Use non-trivial WDL loss (wdl.sum() has zero grad because softmax sums to 1)
        target_wdl = torch.tensor([[1.0, 0.0, 0.0]] * board.shape[0])
        wdl_loss = -(target_wdl * torch.log(wdl + 1e-8)).sum()
        loss = policy.sum() + promo.sum() + wdl_loss + ply.sum()
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"


class TestDeterminism:
    def test_eval_mode_deterministic(self, model, batch):
        model.eval()
        board, features = batch
        with torch.no_grad():
            p1, pr1, w1, pl1 = model(board, features)
            p2, pr2, w2, pl2 = model(board, features)
        assert torch.equal(p1, p2)
        assert torch.equal(w1, w2)


class TestDifferentInputs:
    def test_different_boards_different_output(self, model):
        """Two different board positions should produce different policy logits."""
        model.eval()
        b1 = torch.randint(0, 13, (1, 64))
        b2 = torch.randint(0, 13, (1, 64))
        f = torch.randn(1, 14)
        with torch.no_grad():
            p1, _, _, _ = model(b1, f)
            p2, _, _, _ = model(b2, f)
        assert not torch.allclose(p1, p2, atol=1e-5)


class TestDataLoader:
    def test_chess_loader_v2_integration(self):
        """ChessDatasetV2 returns tensors with correct shapes."""
        from chess_loader import compute_features, result_to_wdl, ChessDatasetV2

        boards = [[0] * 64, [1] * 64]
        moves = [(0, 8, -1), (32, 40, 0)]
        features = [compute_features("." * 64), compute_features("P" * 64)]
        wdl = [result_to_wdl(1.0), result_to_wdl(0.0)]

        ds = ChessDatasetV2(boards, moves, features, wdl)
        assert len(ds) == 2
        board, feat, from_sq, to_sq, promo, wdl_t = ds[0]
        assert board.shape == (64,)
        assert feat.shape == (14,)
        assert wdl_t.shape == (3,)
