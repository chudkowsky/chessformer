"""Grokking detection and Grokfast acceleration for training.

GrokTracker monitors weight norms, gradient norms, effective rank of
activations, and train/test loss gap to detect the grokking transition
(delayed generalization after memorization).

gradfilter_ema implements the Grokfast EMA gradient filter (Lee et al., 2024)
which amplifies slow-varying gradient components, achieving >50x speedup
of grokking with negligible computational overhead.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GrokTracker:
    """Tracks grokking-related metrics during training.

    Monitors:
    - Weight L2 norms (total) — rising = memorization, falling = generalization
    - Gradient L2 norms — spikes often precede grokking transition
    - Effective rank of activations (via SVD entropy) — drop = found structure
    - Train/test loss gap — sustained gap + sudden test improvement = grokking
    - Grokking onset detection — sustained weight norm decrease
    """

    def __init__(self, model: nn.Module, log_path: str | None = None) -> None:
        self.model = model
        self.weight_norm_history: list[float] = []
        self.grad_norm_history: list[float] = []
        self.train_loss_history: list[float] = []
        self.test_loss_history: list[float] = []
        self.log_path = log_path
        self._log_file = open(log_path, "w") if log_path else None

    def compute_weight_norms(self) -> float:
        """Total L2 norm of all parameters."""
        total_sq = sum(
            p.data.norm(2).item() ** 2 for p in self.model.parameters()
        )
        norm = total_sq**0.5
        self.weight_norm_history.append(norm)
        return norm

    def compute_gradient_norms(self) -> float:
        """Total L2 norm of all gradients (call after backward)."""
        total_sq = sum(
            p.grad.data.norm(2).item() ** 2
            for p in self.model.parameters()
            if p.grad is not None
        )
        norm = total_sq**0.5
        self.grad_norm_history.append(norm)
        return norm

    def compute_effective_rank(self, activations: Tensor) -> float:
        """Effective dimensionality via SVD entropy.

        Low rank = model found structured, low-dimensional representation.
        High rank = representations are spread/random (memorization).

        Args:
            activations: Tensor of shape [..., d_model] from a hidden layer.

        Returns:
            Effective rank (1.0 = degenerate, d_model = fully random).
        """
        flat = activations.reshape(-1, activations.shape[-1]).float()
        svs = torch.linalg.svdvals(flat)
        p = svs / svs.sum()
        p = p[p > 1e-12]
        entropy = -(p * p.log()).sum()
        return torch.exp(entropy).item()

    def check_grokking_onset(self, window: int = 50) -> bool:
        """Detect sustained weight norm decrease (cleanup phase signal).

        Returns True if average weight norm over the last `window` entries
        is <95% of the preceding window — indicates the model is pruning
        memorization circuits.
        """
        if len(self.weight_norm_history) < 2 * window:
            return False
        recent = sum(self.weight_norm_history[-window:]) / window
        earlier = sum(self.weight_norm_history[-2 * window : -window]) / window
        return recent < 0.95 * earlier

    def log_epoch(
        self, epoch: int, train_loss: float, test_loss: float
    ) -> dict[str, float]:
        """Log epoch-level metrics and return them as a dict."""
        self.train_loss_history.append(train_loss)
        self.test_loss_history.append(test_loss)
        w_norm = self.compute_weight_norms()
        gap = test_loss - train_loss  # positive = overfitting
        grok = self.check_grokking_onset()

        line = (
            f"[grok] Epoch {epoch} | "
            f"W_norm: {w_norm:.2f} | "
            f"Train: {train_loss:.4f} | Test: {test_loss:.4f} | "
            f"Gap: {gap:+.4f} | "
            f"Grokking: {'>>> YES <<<' if grok else 'no'}"
        )
        print(line)
        if self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()

        return {
            "weight_norm": w_norm,
            "loss_gap": gap,
            "grokking_onset": grok,
        }

    def log_batch(self, step: int) -> dict[str, float]:
        """Log batch-level metrics (lightweight — weight + grad norms only).

        Call after backward() but before optimizer.step().
        """
        w_norm = self.compute_weight_norms()
        g_norm = self.compute_gradient_norms()
        return {"weight_norm": w_norm, "gradient_norm": g_norm}

    def close(self) -> None:
        if self._log_file:
            self._log_file.close()
            self._log_file = None


def gradfilter_ema(
    model: nn.Module,
    grads: dict[str, Tensor] | None = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> dict[str, Tensor]:
    """Grokfast EMA gradient filter (Lee et al., 2024).

    Amplifies slow-varying gradient components by maintaining an EMA
    of gradients and adding the amplified EMA back to current gradients.
    Achieves >50x speedup of grokking with negligible overhead.

    Call after loss.backward() and before optimizer.step():
        loss.backward()
        ema_grads = gradfilter_ema(model, ema_grads)
        optimizer.step()

    Args:
        model: Model whose gradients to filter.
        grads: Previous EMA state (None on first call — initializes).
        alpha: EMA decay factor (0.98 = slow, captures long-term patterns).
        lamb: Amplification factor (2.0 = double the slow components).

    Returns:
        Updated EMA state dict (pass to next call).
    """
    if grads is None:
        return {
            n: p.grad.data.detach().clone()
            for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None
        }
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data += grads[n] * lamb
    return grads
