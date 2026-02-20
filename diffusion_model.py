"""diffusion_model.py — DiT (Diffusion Transformer) for chess latent space.

Denoises "imagined" future board states in the latent space of the policy
backbone. Conditioned on the current position's latent representation and
the diffusion timestep.

Architecture:
    - Input: noisy latent x_t [B, 64, d_dit]
    - Conditioning: backbone latent [B, 64, d_model] → mean-pooled → projected
    - Timestep: Embedding(T, d_dit) → MLP
    - DiT blocks with AdaLN-Zero (adaptive LayerNorm + zero-init gates)
    - Output: predicted noise epsilon [B, 64, d_dit]

AdaLN-Zero trick: gate parameters initialize to 0, so each block starts
as identity. This means the model begins training as a simple pass-through,
then gradually learns to denoise — much more stable than random init.

Reference: Peebles & Xie "Scalable Diffusion Models with Transformers" (2023).
"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class TimestepEmbedding(nn.Module):
    """Sinusoidal + MLP timestep embedding.

    Maps scalar timestep t → d_dit vector. Uses sinusoidal encoding
    (same as positional encoding in original Transformer) followed by
    a 2-layer MLP to project to the right dimension.
    """

    def __init__(self, T: int, d_dit: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(T + 1, d_dit)
        self.mlp = nn.Sequential(
            nn.Linear(d_dit, d_dit * 4),
            nn.Mish(),
            nn.Linear(d_dit * 4, d_dit),
        )

    def forward(self, t: Tensor) -> Tensor:
        """[B] int → [B, d_dit] float."""
        return self.mlp(self.embed(t))


class DiTBlock(nn.Module):
    """Transformer block with AdaLN-Zero conditioning.

    Standard transformer block (self-attention + FFN) but LayerNorm parameters
    are generated from a conditioning vector c. Additionally, each sub-block
    has a learned gate that starts at 0 (zero-init), making the block act as
    identity at initialization.

    Conditioning c produces 6 vectors via a single linear projection:
        shift1, scale1, gate1 (for attention)
        shift2, scale2, gate2 (for FFN)

    Then:
        h = LN(x) * (1 + scale1) + shift1
        h = self_attention(h)
        x = x + gate1 * h
        h = LN(x) * (1 + scale2) + shift2
        h = ffn(h)
        x = x + gate2 * h
    """

    def __init__(self, d_dit: int, num_heads: int, d_hid: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_dit // num_heads

        # AdaLN modulation: c → 6 * d_dit params
        self.adaLN_modulation = nn.Sequential(
            nn.Mish(),
            nn.Linear(d_dit, 6 * d_dit),
        )
        # Zero-init the final linear so gates start at 0
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

        self.norm1 = nn.LayerNorm(d_dit, elementwise_affine=False)
        self.qkv = nn.Linear(d_dit, 3 * d_dit)
        self.out_proj = nn.Linear(d_dit, d_dit)

        self.norm2 = nn.LayerNorm(d_dit, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_dit, d_hid),
            nn.Mish(),
            nn.Linear(d_hid, d_dit),
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: [B, 64, d_dit] — noisy latent tokens
            c: [B, d_dit]     — conditioning vector (timestep + position)
        """
        B, S, D = x.shape

        # Generate modulation params from conditioning
        mod = self.adaLN_modulation(c).unsqueeze(1)  # [B, 1, 6*d_dit]
        shift1, scale1, gate1, shift2, scale2, gate2 = mod.chunk(6, dim=-1)

        # Modulated self-attention
        h = self.norm1(x) * (1 + scale1) + shift1
        qkv = self.qkv(h).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        h = (attn @ v).transpose(1, 2).reshape(B, S, D)
        h = self.out_proj(h)
        x = x + gate1 * h

        # Modulated FFN
        h = self.norm2(x) * (1 + scale2) + shift2
        h = self.ffn(h)
        x = x + gate2 * h

        return x


class ChessDiT(nn.Module):
    """Diffusion Transformer for chess latent-space denoising.

    Takes a noisy latent state x_t and predicts the noise epsilon that was
    added, conditioned on the current position's latent representation and
    the diffusion timestep.

    Config: d_dit=256, nhead=4, nlayers=6, T=20 → ~10M params
    """

    def __init__(
        self,
        d_dit: int = 256,
        d_model: int = 512,
        nhead: int = 4,
        d_hid: int = 512,
        nlayers: int = 6,
        T: int = 20,
    ) -> None:
        super().__init__()
        self.d_dit = d_dit

        # Timestep embedding
        self.time_embed = TimestepEmbedding(T, d_dit)

        # Condition projection: backbone latent → d_dit
        # Mean-pools 64 tokens to a single vector, then projects
        self.cond_proj = nn.Sequential(
            nn.Linear(d_model, d_dit),
            nn.Mish(),
            nn.Linear(d_dit, d_dit),
        )

        # DiT blocks
        self.layers = nn.ModuleList(
            [DiTBlock(d_dit, nhead, d_hid) for _ in range(nlayers)]
        )

        # Final layer: LN + linear, zero-initialized for stable start
        self.final_norm = nn.LayerNorm(d_dit, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.Mish(), nn.Linear(d_dit, 2 * d_dit))
        nn.init.zeros_(self.final_adaLN[1].weight)
        nn.init.zeros_(self.final_adaLN[1].bias)
        self.final_proj = nn.Linear(d_dit, d_dit)
        # Zero-init final projection so initial output is zero (pure residual)
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        backbone_latent: Tensor,
    ) -> Tensor:
        """Predict noise epsilon from noisy latent.

        Args:
            x_t:             [B, 64, d_dit] — noisy latent state
            t:               [B] int        — diffusion timestep
            backbone_latent: [B, 64, d_model] — current position encoding from backbone

        Returns:
            epsilon: [B, 64, d_dit] — predicted noise
        """
        # Combine timestep + condition into a single vector
        time_emb = self.time_embed(t)                              # [B, d_dit]
        cond = self.cond_proj(backbone_latent.mean(dim=1))         # [B, d_dit]
        c = time_emb + cond                                        # [B, d_dit]

        # Pass through DiT blocks
        x = x_t
        for layer in self.layers:
            x = layer(x, c)

        # Final layer with AdaLN
        mod = self.final_adaLN(c).unsqueeze(1)           # [B, 1, 2*d_dit]
        shift, scale = mod.chunk(2, dim=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        epsilon = self.final_proj(x)

        return epsilon
