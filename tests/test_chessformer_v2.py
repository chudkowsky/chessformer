"""Tests for ChessTransformerV2 — forward pass, shapes, gradients, heads."""

import torch
from torch import nn
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


class TestEncode:
    def test_encode_shape(self, model, batch):
        """encode() returns [B, 64, d_model] latent."""
        board, features = batch
        latent = model.encode(board, features)
        assert latent.shape == (board.shape[0], 64, model.d_model)

    def test_encode_without_features(self, model, batch):
        board, _ = batch
        latent = model.encode(board, features=None)
        assert latent.shape == (board.shape[0], 64, model.d_model)

    def test_encode_matches_forward(self, model, batch):
        """encode() output is the same latent used by forward() heads."""
        model.eval()
        board, features = batch
        with torch.no_grad():
            latent = model.encode(board, features)
            policy_from_encode, _ = model.policy_head(latent)
            policy_from_forward, _, _, _ = model(board, features)
        assert torch.equal(policy_from_encode, policy_from_forward)

    def test_encode_gradient_flows(self, model, batch):
        board, features = batch
        latent = model.encode(board, features)
        loss = latent.sum()
        loss.backward()
        assert model.embedding.weight.grad is not None
        assert model.embedding.weight.grad.abs().sum() > 0


class TestDiffusionAttachment:
    def test_attach_diffusion(self, model):
        """attach_diffusion() creates projection layers."""
        from diffusion_model import ChessDiT
        from noise_schedule import CosineNoiseSchedule

        d_dit = 32
        dit = ChessDiT(
            d_dit=d_dit, d_model=model.d_model,
            nhead=4, d_hid=64, nlayers=1, T=5,
        )
        ns = CosineNoiseSchedule(T=5)
        model.attach_diffusion(dit, ns, d_dit)

        assert hasattr(model, "latent_to_dit")
        assert hasattr(model, "dit_to_latent")
        assert model.latent_to_dit.in_features == model.d_model
        assert model.latent_to_dit.out_features == d_dit

    def test_forward_without_diffusion_unchanged(self, model, batch):
        """use_diffusion=True without attached diffusion = normal forward."""
        model.eval()
        board, features = batch
        with torch.no_grad():
            p1, _, _, _ = model(board, features, use_diffusion=False)
            p2, _, _, _ = model(board, features, use_diffusion=True)
        assert torch.equal(p1, p2)

    def test_diffusion_augment_changes_output(self, model, batch):
        """With diffusion attached, use_diffusion=True changes the output.

        dit_to_latent is zero-initialized (so diffusion starts as no-op).
        We set it to non-zero manually to simulate a trained state.
        """
        from diffusion_model import ChessDiT
        from noise_schedule import CosineNoiseSchedule

        d_dit = 32
        dit = ChessDiT(
            d_dit=d_dit, d_model=model.d_model,
            nhead=4, d_hid=64, nlayers=1, T=3,
        )
        ns = CosineNoiseSchedule(T=3)
        model.attach_diffusion(dit, ns, d_dit)

        # Break zero-init so diffusion output actually affects latent
        nn.init.normal_(model.dit_to_latent.weight, std=0.1)

        model.eval()
        board, features = batch
        with torch.no_grad():
            p_no_diff, _, _, _ = model(board, features, use_diffusion=False)
            p_with_diff, _, _, _ = model(board, features, use_diffusion=True)
        # Diffusion adds context → outputs differ
        assert not torch.equal(p_no_diff, p_with_diff)


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
