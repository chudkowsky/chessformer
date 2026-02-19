"""
attention.py — Custom attention components for ChessTransformer V2.

Three building blocks:
- ShawRelativePositionBias: learned bias per (delta_rank, delta_file) pair
- SmolgenBias: content-dependent attention bias from full board state
- ChessTransformerBlock: pre-norm transformer block with combined biases
"""

import chess
import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ShawRelativePositionBias(nn.Module):
    """Topology-aware position bias for the 8x8 chess board.

    Learns a separate attention bias for each (delta_rank, delta_file) pair
    between squares. Deltas range from -7 to +7, giving a (15, 15) table
    per attention head. This directly models chess geometry: diagonals for
    bishops, L-shapes for knights, files for rooks — unlike sinusoidal PE
    which treats the board as a 1D sequence.

    Shaw et al. "Self-Attention with Relative Position Representations" (2018).
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.bias_table = nn.Parameter(torch.zeros(num_heads, 15, 15))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        # Precompute (rank, file) for each of the 64 squares using python-chess
        coords = torch.tensor(
            [(chess.square_rank(sq), chess.square_file(sq)) for sq in range(64)]
        )
        # Pairwise differences shifted to [0, 14] range
        rank_diff = coords[:, 0].unsqueeze(1) - coords[:, 0].unsqueeze(0) + 7
        file_diff = coords[:, 1].unsqueeze(1) - coords[:, 1].unsqueeze(0) + 7
        self.register_buffer("rank_idx", rank_diff.long())
        self.register_buffer("file_idx", file_diff.long())

    def forward(self) -> Tensor:
        """Returns position bias [num_heads, 64, 64]."""
        return self.bias_table[:, self.rank_idx, self.file_idx]


class SmolgenBias(nn.Module):
    """Content-dependent dynamic attention bias (simplified BT4 smolgen).

    Compresses the full 64-token board representation into a small vector,
    then projects to per-head 64x64 attention bias matrices. This lets the
    attention pattern adapt to the specific position — e.g. suppressing
    long-range connections in closed positions, amplifying them on open
    diagonals.

    BT4 (Lc0) uses a similar mechanism at 8M params; this factored version
    achieves the same expressivity with ~0.5M params.
    """

    def __init__(
        self, d_model: int, num_heads: int, compress_dim: int = 256
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.compress = nn.Sequential(
            nn.Linear(64 * d_model, compress_dim),
            nn.Mish(),
        )
        self.project = nn.Linear(compress_dim, num_heads * 64 * 64)

    def forward(self, x: Tensor) -> Tensor:
        """[B, 64, d_model] → [B, num_heads, 64, 64]."""
        B = x.shape[0]
        flat = x.reshape(B, -1)
        compressed = self.compress(flat)
        biases = self.project(compressed)
        return biases.reshape(B, self.num_heads, 64, 64)


class ChessTransformerBlock(nn.Module):
    """Pre-norm transformer block with additive attention biases.

    Pre-norm (LN before attention/FFN) is more stable during training than
    post-norm — used by GPT-2+, LLaMA, and all modern transformers.
    Accepts combined Shaw PE + smolgen biases added to attention logits.
    """

    def __init__(self, d_model: int, num_heads: int, d_hid: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hid),
            nn.Mish(),
            nn.Linear(d_hid, d_model),
        )

    def forward(self, x: Tensor, attn_bias: Tensor | None = None) -> Tensor:
        """
        Args:
            x: [B, 64, d_model]
            attn_bias: [B, num_heads, 64, 64] combined Shaw + smolgen bias
        """
        B, S, D = x.shape

        # Pre-norm self-attention
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each [B, S, num_heads, head_dim]
        q = q.transpose(1, 2)  # [B, num_heads, S, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention with bias
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + h

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x
