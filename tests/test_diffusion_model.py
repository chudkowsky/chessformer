"""Tests for diffusion_model.py — ChessDiT, DiTBlock, TimestepEmbedding."""

import torch
import pytest
from diffusion_model import ChessDiT, DiTBlock, TimestepEmbedding


@pytest.fixture
def dit():
    """Small DiT for fast tests."""
    return ChessDiT(
        d_dit=64, d_model=128, nhead=4, d_hid=128, nlayers=2, T=20,
    )


@pytest.fixture
def inputs():
    B = 3
    return {
        "x_t": torch.randn(B, 64, 64),          # noisy latent
        "t": torch.randint(0, 20, (B,)),          # timesteps
        "backbone_latent": torch.randn(B, 64, 128),  # from policy backbone
    }


class TestChessDiT:
    def test_output_shape(self, dit, inputs):
        eps = dit(**inputs)
        assert eps.shape == inputs["x_t"].shape

    def test_different_timesteps_after_training(self, dit, inputs):
        """After a few training steps, different timesteps produce different outputs.

        AdaLN-Zero has a multi-step gradient unblocking chain:
        - Step 1: only final_proj gets gradient (everything else zero-init)
        - Step 2: final_adaLN gets gradient (final_proj now non-zero)
        After step 2, final_adaLN is non-zero → conditioning (incl. timestep)
        affects the output through shift/scale modulation.
        """
        optimizer = torch.optim.SGD(dit.parameters(), lr=0.1)
        target = torch.randn_like(inputs["x_t"])
        for _ in range(5):
            optimizer.zero_grad()
            eps = dit(**inputs)
            loss = (eps - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        dit.eval()
        with torch.no_grad():
            inputs1 = {**inputs, "t": torch.tensor([0, 0, 0])}
            inputs2 = {**inputs, "t": torch.tensor([19, 19, 19])}
            eps1 = dit(**inputs1)
            eps2 = dit(**inputs2)
        assert not torch.allclose(eps1, eps2, atol=1e-5)

    def test_final_proj_receives_gradient(self, dit, inputs):
        """Zero-init: gradient reaches final_proj (which starts learning first)."""
        eps = dit(**inputs)
        loss = eps.sum()
        loss.backward()
        assert dit.final_proj.weight.grad is not None
        assert dit.final_proj.weight.grad.abs().sum() > 0

    def test_all_params_receive_gradient_after_training(self, dit, inputs):
        """After several steps, gradient flows to all layers.

        AdaLN-Zero unblocking chain needs 3 steps minimum:
        - Step 1: final_proj learns (zero → non-zero)
        - Step 2: final_adaLN + block adaLN_modulations learn (gradient
          now flows back through non-zero final_proj)
        - Step 3: block internals (qkv, ffn) + cond_proj + time_embed learn
          (gradient flows through non-zero adaLN gates)
        """
        optimizer = torch.optim.SGD(dit.parameters(), lr=0.1)
        target = torch.randn_like(inputs["x_t"])
        for _ in range(5):
            optimizer.zero_grad()
            eps = dit(**inputs)
            loss = (eps - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        eps = dit(**inputs)
        loss = (eps - target).pow(2).sum()
        loss.backward()
        no_grad = [
            n for n, p in dit.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0
        ]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"

    def test_no_nan(self, dit, inputs):
        eps = dit(**inputs)
        assert not torch.isnan(eps).any()

    def test_zero_init_starts_near_zero(self, dit, inputs):
        """With zero-init, initial output should be near zero."""
        dit.eval()
        with torch.no_grad():
            eps = dit(**inputs)
        # Output should be small (zero-init final projection)
        assert eps.abs().mean() < 0.1


class TestDiTBlock:
    def test_output_shape(self):
        block = DiTBlock(d_dit=64, num_heads=4, d_hid=128)
        x = torch.randn(2, 64, 64)
        c = torch.randn(2, 64)
        out = block(x, c)
        assert out.shape == x.shape

    def test_zero_init_identity(self):
        """With zero-init gates, block should act as identity initially."""
        block = DiTBlock(d_dit=64, num_heads=4, d_hid=128)
        x = torch.randn(1, 64, 64)
        c = torch.zeros(1, 64)
        out = block(x, c)
        # Gates are zero → output should be very close to input
        assert torch.allclose(out, x, atol=1e-5)


class TestTimestepEmbedding:
    def test_output_shape(self):
        embed = TimestepEmbedding(T=20, d_dit=64)
        t = torch.tensor([0, 10, 20])
        out = embed(t)
        assert out.shape == (3, 64)

    def test_different_timesteps(self):
        embed = TimestepEmbedding(T=20, d_dit=64)
        out = embed(torch.tensor([0, 10, 20]))
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])
