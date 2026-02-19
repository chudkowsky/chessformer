"""Tests for ChessTransformerV2 — forward pass, gradients, heads, diffusion."""

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
    """V2 forward: board + features → policy, promo, wdl, ply."""

    def test_output_shapes(self, model, batch):
        """policy[B,64,64], promo[B,64,4], wdl[B,3], ply[B,1]."""
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        B = board.shape[0]
        assert policy.shape == (B, 64, 64)
        assert promo.shape == (B, 64, 4)
        assert wdl.shape == (B, 3)
        assert ply.shape == (B, 1)

    def test_features_affect_output(self, model, batch):
        """Auxiliary features should change policy (not just board alone)."""
        board, features = batch
        model.eval()
        with torch.no_grad():
            p_with_feat, _, _, _ = model(board, features)
            p_no_feat, _, _, _ = model(board, features=None)
        assert not torch.allclose(p_with_feat, p_no_feat, atol=1e-5), (
            "Features should affect policy output"
        )

    def test_wdl_is_valid_distribution(self, model, batch):
        """WDL sums to 1, all values in [0, 1]."""
        board, features = batch
        _, _, wdl, _ = model(board, features)
        sums = wdl.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert (wdl >= 0).all() and (wdl <= 1).all()

    def test_ply_non_negative(self, model, batch):
        """Ply prediction should be non-negative."""
        board, features = batch
        _, _, _, ply = model(board, features)
        assert (ply >= 0).all()

    def test_no_nan_in_outputs(self, model, batch):
        """No NaN in any output tensor."""
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        for name, t in [("policy", policy), ("promo", promo), ("wdl", wdl), ("ply", ply)]:
            assert not torch.isnan(t).any(), f"NaN in {name}"


class TestGradients:
    """Gradient flow through all model parameters."""

    def test_all_params_receive_nonzero_gradient(self, model, batch):
        """Every parameter gets non-zero gradient with proper loss."""
        board, features = batch
        policy, promo, wdl, ply = model(board, features)
        # wdl.sum() has zero grad (softmax sums to 1), so use CE loss
        target_wdl = torch.tensor([[1.0, 0.0, 0.0]] * board.shape[0])
        wdl_loss = -(target_wdl * torch.log(wdl + 1e-8)).sum()
        loss = policy.sum() + promo.sum() + wdl_loss + ply.sum()
        loss.backward()
        no_grad = [n for n, p in model.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"


class TestDeterminism:
    """Eval mode should be deterministic (no dropout)."""

    def test_eval_mode_deterministic(self, model, batch):
        """Same input → same output in eval mode."""
        model.eval()
        board, features = batch
        with torch.no_grad():
            p1, pr1, w1, pl1 = model(board, features)
            p2, pr2, w2, pl2 = model(board, features)
        assert torch.equal(p1, p2)
        assert torch.equal(w1, w2)


class TestDifferentInputs:
    """Model should discriminate between different positions."""

    def test_different_boards_different_policy(self, model):
        """Two different board positions → different policy logits."""
        model.eval()
        b1 = torch.randint(0, 13, (1, 64))
        b2 = torch.randint(0, 13, (1, 64))
        f = torch.randn(1, 14)
        with torch.no_grad():
            p1, _, _, _ = model(b1, f)
            p2, _, _, _ = model(b2, f)
        assert not torch.allclose(p1, p2, atol=1e-5)


class TestEncode:
    """encode() returns backbone latent used by all output heads."""

    def test_encode_shape(self, model, batch):
        """Latent shape: [B, 64, d_model]."""
        board, features = batch
        latent = model.encode(board, features)
        assert latent.shape == (board.shape[0], 64, model.d_model)

    def test_encode_consistent_with_forward(self, model, batch):
        """encode() latent → policy_head gives same result as forward()."""
        model.eval()
        board, features = batch
        with torch.no_grad():
            latent = model.encode(board, features)
            policy_from_encode, _ = model.policy_head(latent)
            policy_from_forward, _, _, _ = model(board, features)
        assert torch.equal(policy_from_encode, policy_from_forward)

    def test_encode_gradient_flows_to_embedding(self, model, batch):
        """Gradients from latent reach the input embedding layer."""
        board, features = batch
        latent = model.encode(board, features)
        latent.sum().backward()
        assert model.embedding.weight.grad is not None
        assert model.embedding.weight.grad.abs().sum() > 0


class TestDiffusionAttachment:
    """Diffusion DiT attachment to backbone."""

    def test_attach_creates_projection_layers(self, model):
        """attach_diffusion() creates latent_to_dit and dit_to_latent."""
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

    def test_no_diffusion_attached_same_output(self, model, batch):
        """use_diffusion=True without attached DiT = normal forward."""
        model.eval()
        board, features = batch
        with torch.no_grad():
            p1, _, _, _ = model(board, features, use_diffusion=False)
            p2, _, _, _ = model(board, features, use_diffusion=True)
        assert torch.equal(p1, p2)

    def test_diffusion_changes_output(self, model, batch):
        """With DiT attached and non-zero weights, output changes."""
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
        assert not torch.equal(p_no_diff, p_with_diff)


class TestDataLoader:
    """ChessDatasetV2 integration: tensors have correct shapes and values."""

    def test_chess_loader_v2_shapes_and_wdl(self):
        """Dataset returns correct tensor shapes; WDL is one-hot."""
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
        # result=1.0 → win → WDL = (1, 0, 0)
        assert wdl_t.tolist() == [1.0, 0.0, 0.0]
