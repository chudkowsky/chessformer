"""noise_schedule.py — Cosine noise schedule for continuous DDPM.

Implements the forward diffusion process: gradually adding Gaussian noise
to clean latent states over T timesteps. The reverse process (denoising)
is handled by the DiT model in diffusion_model.py.

Key formula:
    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise

where alpha_bar_t follows a cosine curve from ~1 (clean) to ~0 (pure noise).

Reference: Nichol & Dhariwal "Improved DDPM" (2021), Section 3.2.
"""

import math
import torch
from torch import Tensor


class CosineNoiseSchedule:
    """Cosine noise schedule with precomputed diffusion coefficients.

    Args:
        T: number of diffusion timesteps (20 for chess, per DiffuSearch finding)
        s: small offset to prevent alpha_bar from being exactly 1 at t=0
    """

    def __init__(self, T: int = 20, s: float = 0.008) -> None:
        self.T = T

        # alpha_bar: cumulative signal retention, shape [T+1]
        # alpha_bar[0] ≈ 1 (clean), alpha_bar[T] ≈ 0 (pure noise)
        steps = torch.arange(T + 1, dtype=torch.float64)
        f_t = torch.cos(((steps / T) + s) / (1 + s) * (math.pi / 2)) ** 2
        alpha_bar = f_t / f_t[0]
        # Clamp to avoid numerical issues at boundaries
        alpha_bar = alpha_bar.clamp(min=1e-5, max=1.0 - 1e-5)

        self.alpha_bar = alpha_bar.float()
        # sqrt values used in q_sample
        self.sqrt_alpha_bar = self.alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - self.alpha_bar).sqrt()

        # Per-step alpha and beta (needed for reverse process / denoising step)
        # alpha_t = alpha_bar_t / alpha_bar_{t-1}
        # beta_t = 1 - alpha_t
        alpha = torch.cat([
            torch.tensor([1.0]),
            self.alpha_bar[1:] / self.alpha_bar[:-1],
        ])
        alpha = alpha.clamp(min=1e-5, max=1.0)
        self.beta = (1.0 - alpha).float()
        self.alpha = alpha.float()

        # Posterior variance for denoising: beta_tilde_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        alpha_bar_prev = torch.cat([torch.tensor([1.0]), self.alpha_bar[:-1]])
        self.posterior_variance = (
            self.beta * (1.0 - alpha_bar_prev) / (1.0 - self.alpha_bar)
        ).clamp(min=1e-20)

    def q_sample(self, x_0: Tensor, t: Tensor, noise: Tensor | None = None) -> Tensor:
        """Forward process: add noise to clean data.

        Args:
            x_0:   clean data, any shape [B, ...]
            t:     timestep indices, shape [B], values in [0, T]
            noise: optional pre-sampled Gaussian noise, same shape as x_0

        Returns:
            x_t: noisy data at timestep t, same shape as x_0
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Gather coefficients for each batch element, reshape for broadcasting
        sqrt_ab = self.sqrt_alpha_bar[t]
        sqrt_1_ab = self.sqrt_one_minus_alpha_bar[t]

        # Reshape to broadcast: [B] -> [B, 1, 1, ...] matching x_0 dims
        while sqrt_ab.dim() < x_0.dim():
            sqrt_ab = sqrt_ab.unsqueeze(-1)
            sqrt_1_ab = sqrt_1_ab.unsqueeze(-1)

        return sqrt_ab * x_0 + sqrt_1_ab * noise

    def to(self, device: torch.device) -> "CosineNoiseSchedule":
        """Move all precomputed tensors to a device."""
        self.alpha_bar = self.alpha_bar.to(device)
        self.sqrt_alpha_bar = self.sqrt_alpha_bar.to(device)
        self.sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar.to(device)
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
        return self
