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
        """epsilon shape matches input x_t shape."""
        eps = dit(**inputs)
        assert eps.shape == inputs["x_t"].shape

    def test_zero_init_output_is_near_zero(self, dit, inputs):
        """AdaLN-Zero: final_proj is zero-init, so initial eps should be ~0."""
        dit.eval()
        with torch.no_grad():
            eps = dit(**inputs)
        # final_proj.weight and bias are zeros → output should be exactly zero
        assert eps.abs().max() < 1e-6, f"Expected near-zero output, got max={eps.abs().max():.6f}"

    def test_zero_init_weights_are_actually_zero(self, dit):
        """Verify zero-init layers have exactly zero weights at construction."""
        assert dit.final_proj.weight.abs().max() == 0.0
        assert dit.final_proj.bias.abs().max() == 0.0
        # adaLN modulation in each DiTBlock should also be zero
        for i, layer in enumerate(dit.layers):
            w = layer.adaLN_modulation[1].weight
            assert w.abs().max() == 0.0, f"DiTBlock {i} adaLN not zero-init"

    def test_conditioning_affects_output(self, dit, inputs):
        """Different backbone_latent should produce different epsilon after training."""
        optimizer = torch.optim.SGD(dit.parameters(), lr=0.1)
        target = torch.randn_like(inputs["x_t"])
        for _ in range(5):
            optimizer.zero_grad()
            loss = (dit(**inputs) - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        dit.eval()
        with torch.no_grad():
            eps1 = dit(inputs["x_t"], inputs["t"], torch.randn(3, 64, 128))
            eps2 = dit(inputs["x_t"], inputs["t"], torch.randn(3, 64, 128))
        assert not torch.allclose(eps1, eps2, atol=1e-5), (
            "Different conditioning should produce different output after training"
        )

    def test_different_timesteps_after_training(self, dit, inputs):
        """After training, different timesteps produce different outputs.

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
            loss = (dit(**inputs) - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        dit.eval()
        with torch.no_grad():
            inputs1 = {**inputs, "t": torch.tensor([0, 0, 0])}
            inputs2 = {**inputs, "t": torch.tensor([19, 19, 19])}
            eps1 = dit(**inputs1)
            eps2 = dit(**inputs2)
        assert not torch.allclose(eps1, eps2, atol=1e-5)

    def test_all_params_receive_gradient_after_training(self, dit, inputs):
        """After several steps, gradient flows to all layers.

        AdaLN-Zero unblocking chain needs 3 steps minimum:
        - Step 1: final_proj learns (zero → non-zero)
        - Step 2: final_adaLN + block adaLN_modulations learn
        - Step 3: block internals (qkv, ffn) + cond_proj + time_embed learn
        """
        optimizer = torch.optim.SGD(dit.parameters(), lr=0.1)
        target = torch.randn_like(inputs["x_t"])
        for _ in range(5):
            optimizer.zero_grad()
            loss = (dit(**inputs) - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        loss = (dit(**inputs) - target).pow(2).sum()
        loss.backward()
        no_grad = [
            n for n, p in dit.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0
        ]
        assert len(no_grad) == 0, f"Params without gradient: {no_grad}"

    def test_no_nan(self, dit, inputs):
        """No NaN in output (numerical stability)."""
        eps = dit(**inputs)
        assert not torch.isnan(eps).any()


class TestDiTBlock:
    def test_output_shape(self):
        """Output shape matches input shape."""
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
        assert torch.allclose(out, x, atol=1e-5)

    def test_nonzero_conditioning_still_identity(self):
        """Even with nonzero conditioning, zero-init gates keep block as identity."""
        block = DiTBlock(d_dit=64, num_heads=4, d_hid=128)
        x = torch.randn(1, 64, 64)
        c = torch.randn(1, 64)  # nonzero conditioning
        out = block(x, c)
        # Gates start at zero → gate1 * attn = 0, gate2 * ffn = 0
        assert torch.allclose(out, x, atol=1e-5)

    def test_trained_block_is_not_identity(self):
        """After training, block should no longer be identity."""
        block = DiTBlock(d_dit=64, num_heads=4, d_hid=128)
        x = torch.randn(2, 64, 64)
        c = torch.randn(2, 64)
        optimizer = torch.optim.SGD(block.parameters(), lr=0.1)
        target = torch.randn_like(x)
        for _ in range(10):
            optimizer.zero_grad()
            loss = (block(x, c) - target).pow(2).sum()
            loss.backward()
            optimizer.step()

        block.eval()
        with torch.no_grad():
            out = block(x, c)
        assert not torch.allclose(out, x, atol=1e-3), (
            "Block should differ from identity after training"
        )


class TestTimestepEmbedding:
    def test_output_shape(self):
        """[B] int timesteps → [B, d_dit] float embedding."""
        embed = TimestepEmbedding(T=20, d_dit=64)
        t = torch.tensor([0, 10, 20])
        out = embed(t)
        assert out.shape == (3, 64)

    def test_different_timesteps_produce_different_embeddings(self):
        embed = TimestepEmbedding(T=20, d_dit=64)
        out = embed(torch.tensor([0, 10, 20]))
        assert not torch.allclose(out[0], out[1])
        assert not torch.allclose(out[1], out[2])
        assert not torch.allclose(out[0], out[2])

    def test_same_timestep_same_embedding(self):
        """Deterministic: same t → same embedding."""
        embed = TimestepEmbedding(T=20, d_dit=64)
        embed.eval()
        out = embed(torch.tensor([5, 5, 5]))
        assert torch.equal(out[0], out[1])
        assert torch.equal(out[1], out[2])

    def test_embedding_is_nonzero(self):
        """Timestep embeddings should not be zero vectors."""
        embed = TimestepEmbedding(T=20, d_dit=64)
        out = embed(torch.tensor([0, 10, 20]))
        for i in range(3):
            assert out[i].abs().sum() > 0, f"Embedding for t={[0,10,20][i]} is zero"
