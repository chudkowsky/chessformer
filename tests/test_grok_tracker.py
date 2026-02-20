"""Tests for GrokTracker and Grokfast EMA gradient filter."""

import pytest
import torch
from torch import nn

from grok_tracker import GrokTracker, gradfilter_ema


# --- GrokTracker ---


class TestWeightNorms:
    def test_positive_for_initialized_model(self):
        model = nn.Linear(10, 5)
        tracker = GrokTracker(model)
        norm = tracker.compute_weight_norms()
        assert norm > 0

    def test_appends_to_history(self):
        model = nn.Linear(10, 5)
        tracker = GrokTracker(model)
        tracker.compute_weight_norms()
        tracker.compute_weight_norms()
        assert len(tracker.weight_norm_history) == 2

    def test_zero_for_zero_weights(self):
        model = nn.Linear(10, 5, bias=False)
        nn.init.zeros_(model.weight)
        tracker = GrokTracker(model)
        assert tracker.compute_weight_norms() == 0.0


class TestGradientNorms:
    def test_positive_after_backward(self):
        model = nn.Linear(10, 5)
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        tracker = GrokTracker(model)
        assert tracker.compute_gradient_norms() > 0

    def test_zero_before_backward(self):
        model = nn.Linear(10, 5)
        tracker = GrokTracker(model)
        assert tracker.compute_gradient_norms() == 0.0

    def test_appends_to_history(self):
        model = nn.Linear(10, 5)
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        tracker = GrokTracker(model)
        tracker.compute_gradient_norms()
        assert len(tracker.grad_norm_history) == 1


class TestEffectiveRank:
    def test_full_rank_activations(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        act = torch.randn(32, 16)
        rank = tracker.compute_effective_rank(act)
        # Random matrix has high effective rank (close to min(rows, cols))
        assert rank > 5.0

    def test_rank_one_activations(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        # All columns identical → rank 1
        col = torch.randn(32, 1)
        act = col.expand(32, 16)
        rank = tracker.compute_effective_rank(act)
        assert rank < 1.5  # close to 1

    def test_low_rank_lower_than_full(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        full = torch.randn(32, 16)
        low = torch.randn(32, 1).expand(32, 16)
        assert tracker.compute_effective_rank(low) < tracker.compute_effective_rank(full)

    def test_3d_input(self):
        """Works with [B, seq, d_model] shaped activations."""
        tracker = GrokTracker(nn.Linear(1, 1))
        act = torch.randn(4, 64, 32)  # batch=4, seq=64, d_model=32
        rank = tracker.compute_effective_rank(act)
        assert rank > 1.0


class TestGrokkingOnset:
    def test_not_enough_data(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        tracker.weight_norm_history = [100.0] * 5
        assert not tracker.check_grokking_onset(window=5)

    def test_detected_on_decrease(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        # Norms were 100, then dropped to 80 → 20% decrease > 5% threshold
        tracker.weight_norm_history = [100.0] * 10 + [80.0] * 10
        assert tracker.check_grokking_onset(window=10)

    def test_not_detected_when_stable(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        tracker.weight_norm_history = [100.0] * 20
        assert not tracker.check_grokking_onset(window=10)

    def test_not_detected_on_increase(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        tracker.weight_norm_history = [80.0] * 10 + [100.0] * 10
        assert not tracker.check_grokking_onset(window=10)


class TestLogEpoch:
    def test_returns_metrics(self):
        model = nn.Linear(10, 5)
        tracker = GrokTracker(model)
        metrics = tracker.log_epoch(1, train_loss=2.5, test_loss=3.0)
        assert "weight_norm" in metrics
        assert "loss_gap" in metrics
        assert "grokking_onset" in metrics

    def test_loss_gap_positive_means_overfitting(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        metrics = tracker.log_epoch(1, train_loss=1.0, test_loss=2.0)
        # test > train → overfitting → positive gap
        assert metrics["loss_gap"] == pytest.approx(1.0)

    def test_tracks_history(self):
        tracker = GrokTracker(nn.Linear(1, 1))
        tracker.log_epoch(1, 2.5, 3.0)
        tracker.log_epoch(2, 2.0, 2.8)
        assert len(tracker.train_loss_history) == 2
        assert len(tracker.test_loss_history) == 2

    def test_writes_to_file(self, tmp_path):
        log_file = tmp_path / "grok.log"
        tracker = GrokTracker(nn.Linear(1, 1), log_path=str(log_file))
        tracker.log_epoch(1, 2.5, 3.0)
        tracker.close()
        content = log_file.read_text()
        assert "Epoch 1" in content
        assert "W_norm" in content


class TestLogBatch:
    def test_returns_both_norms(self):
        model = nn.Linear(10, 5)
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        tracker = GrokTracker(model)
        metrics = tracker.log_batch(step=0)
        assert "weight_norm" in metrics
        assert "gradient_norm" in metrics
        assert metrics["weight_norm"] > 0
        assert metrics["gradient_norm"] > 0


# --- Grokfast ---


class TestGrokfast:
    def test_first_call_initializes(self):
        model = nn.Linear(10, 5)
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        grads = gradfilter_ema(model, None)
        assert isinstance(grads, dict)
        assert len(grads) > 0

    def test_amplifies_gradients(self):
        model = nn.Linear(10, 5, bias=False)
        torch.manual_seed(42)
        x = torch.randn(4, 10)

        # Step 1: initialize EMA
        loss = model(x).sum()
        loss.backward()
        ema = gradfilter_ema(model, None, alpha=0.5, lamb=1.0)

        # Step 2: apply filter — gradient should be modified
        model.zero_grad()
        loss = model(x).sum()
        loss.backward()
        orig_grad = model.weight.grad.data.clone()
        ema = gradfilter_ema(model, ema, alpha=0.5, lamb=1.0)
        # Gradient = original + lamb * ema → different from original
        assert not torch.allclose(model.weight.grad.data, orig_grad)

    def test_no_effect_with_zero_lamb(self):
        model = nn.Linear(10, 5, bias=False)
        x = torch.randn(4, 10)

        loss = model(x).sum()
        loss.backward()
        ema = gradfilter_ema(model, None, lamb=0.0)

        model.zero_grad()
        loss = model(x).sum()
        loss.backward()
        orig_grad = model.weight.grad.data.clone()
        gradfilter_ema(model, ema, lamb=0.0)
        assert torch.allclose(model.weight.grad.data, orig_grad)

    def test_ema_converges(self):
        """With constant gradients, EMA converges to that gradient."""
        model = nn.Linear(10, 5, bias=False)
        x = torch.randn(4, 10)
        ema = None
        for _ in range(100):
            model.zero_grad()
            loss = model(x).sum()
            loss.backward()
            ema = gradfilter_ema(model, ema, alpha=0.9, lamb=0.0)

        # After many steps with lamb=0, EMA should equal current grad
        current_grad = model.weight.grad.data
        assert torch.allclose(ema["weight"], current_grad, atol=1e-3)

    def test_skips_frozen_params(self):
        model = nn.Linear(10, 5)
        model.bias.requires_grad = False
        loss = model(torch.randn(2, 10)).sum()
        loss.backward()
        grads = gradfilter_ema(model, None)
        assert all("bias" not in k for k in grads)
