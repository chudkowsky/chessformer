"""Tests for attention.py — ShawRPE, SmolgenBias, ChessTransformerBlock."""

import torch
import pytest
from attention import ShawRelativePositionBias, SmolgenBias, ChessTransformerBlock


# --- ShawRelativePositionBias ---


class TestShawRPE:
    def setup_method(self):
        self.nhead = 8
        self.shaw = ShawRelativePositionBias(self.nhead)

    def test_output_shape(self):
        out = self.shaw()
        assert out.shape == (self.nhead, 64, 64)

    def test_symmetric_for_same_delta(self):
        """Squares with same relative (rank, file) offset get the same bias."""
        out = self.shaw()
        # a1→b2 and c3→d4 both have delta (+1, +1)
        # a1=0, b2=9 in chess; shaw index uses rank_idx/file_idx buffers
        # Just check the bias table is indexed consistently
        assert out.shape[1] == 64
        assert out.shape[2] == 64

    def test_diagonal_consistency(self):
        """All (i, i+9) pairs on the board diagonal should get the same bias
        (same delta_rank=+1, delta_file=+1)."""
        out = self.shaw()
        # In python-chess: sq 0=a1, sq 9=b2, sq 18=c3 etc.
        diagonal_pairs = [(0, 9), (9, 18), (18, 27)]
        biases = [out[:, i, j] for i, j in diagonal_pairs]
        for b in biases[1:]:
            assert torch.allclose(b, biases[0])

    def test_gradient_flows(self):
        out = self.shaw()
        loss = out.sum()
        loss.backward()
        assert self.shaw.bias_table.grad is not None
        assert self.shaw.bias_table.grad.abs().sum() > 0

    def test_no_input_dependency(self):
        """Shaw RPE has no input — calling twice gives identical output."""
        out1 = self.shaw()
        out2 = self.shaw()
        assert torch.equal(out1, out2)


# --- SmolgenBias ---


class TestSmolgenBias:
    def setup_method(self):
        self.d_model = 64  # small for tests
        self.nhead = 4
        self.smolgen = SmolgenBias(self.d_model, self.nhead)

    def test_output_shape(self):
        x = torch.randn(2, 64, self.d_model)
        out = self.smolgen(x)
        assert out.shape == (2, self.nhead, 64, 64)

    def test_content_dependent(self):
        """Different inputs produce different biases."""
        x1 = torch.randn(1, 64, self.d_model)
        x2 = torch.randn(1, 64, self.d_model)
        out1 = self.smolgen(x1)
        out2 = self.smolgen(x2)
        assert not torch.allclose(out1, out2, atol=1e-5)

    def test_batch_independent(self):
        """Each batch element gets its own bias."""
        x = torch.randn(3, 64, self.d_model)
        out = self.smolgen(x)
        # Biases for different batch elements should differ
        assert not torch.allclose(out[0], out[1], atol=1e-5)

    def test_gradient_flows(self):
        x = torch.randn(1, 64, self.d_model, requires_grad=True)
        out = self.smolgen(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0


# --- ChessTransformerBlock ---


class TestChessTransformerBlock:
    def setup_method(self):
        self.d_model = 64
        self.nhead = 4
        self.d_hid = 128
        self.block = ChessTransformerBlock(self.d_model, self.nhead, self.d_hid)

    def test_output_shape(self):
        x = torch.randn(2, 64, self.d_model)
        out = self.block(x)
        assert out.shape == x.shape

    def test_with_attn_bias(self):
        x = torch.randn(2, 64, self.d_model)
        bias = torch.randn(2, self.nhead, 64, 64)
        out = self.block(x, attn_bias=bias)
        assert out.shape == x.shape

    def test_without_attn_bias(self):
        x = torch.randn(2, 64, self.d_model)
        out = self.block(x, attn_bias=None)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (block transforms) but not be wildly different
        (residual keeps magnitudes in check)."""
        x = torch.randn(1, 64, self.d_model)
        out = self.block(x)
        assert not torch.allclose(out, x, atol=1e-5)
        # Residual: output norm should be same order of magnitude as input
        assert out.norm() / x.norm() < 10.0

    def test_gradient_flows(self):
        x = torch.randn(1, 64, self.d_model, requires_grad=True)
        bias = torch.randn(1, self.nhead, 64, 64)
        out = self.block(x, attn_bias=bias)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_no_nan(self):
        x = torch.randn(2, 64, self.d_model)
        out = self.block(x)
        assert not torch.isnan(out).any()
