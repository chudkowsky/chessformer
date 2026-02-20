"""Tests for noise_schedule.py — CosineNoiseSchedule."""

import torch
import pytest
from noise_schedule import CosineNoiseSchedule


@pytest.fixture
def schedule():
    return CosineNoiseSchedule(T=20)


class TestAlphaBar:
    def test_monotonically_decreasing(self, schedule):
        """alpha_bar should decrease: more noise at higher timesteps."""
        diffs = schedule.alpha_bar[1:] - schedule.alpha_bar[:-1]
        assert (diffs <= 0).all()

    def test_bounds(self, schedule):
        """alpha_bar should be in (0, 1)."""
        assert (schedule.alpha_bar > 0).all()
        assert (schedule.alpha_bar < 1).all()

    def test_near_one_at_start(self, schedule):
        """alpha_bar[0] should be close to 1 (almost no noise)."""
        assert schedule.alpha_bar[0] > 0.99

    def test_near_zero_at_end(self, schedule):
        """alpha_bar[T] should be close to 0 (almost pure noise)."""
        assert schedule.alpha_bar[-1] < 0.05

    def test_length(self, schedule):
        """alpha_bar has T+1 entries (t=0 to t=T inclusive)."""
        assert len(schedule.alpha_bar) == 21


class TestQSample:
    def test_output_shape(self, schedule):
        x_0 = torch.randn(4, 64, 256)
        t = torch.randint(0, 20, (4,))
        x_t = schedule.q_sample(x_0, t)
        assert x_t.shape == x_0.shape

    def test_t0_close_to_clean(self, schedule):
        """At t=0, x_t should be very close to x_0 (minimal noise)."""
        x_0 = torch.randn(2, 64, 256)
        t = torch.zeros(2, dtype=torch.long)
        x_t = schedule.q_sample(x_0, t)
        assert torch.allclose(x_t, x_0, atol=0.1)

    def test_tT_close_to_noise(self, schedule):
        """At t=T, x_t should have very little signal from x_0."""
        x_0 = torch.ones(2, 64, 256) * 10.0  # large signal
        t = torch.full((2,), schedule.T, dtype=torch.long)
        noise = torch.randn_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise=noise)
        # Signal should be almost gone — x_t should be close to noise
        signal_ratio = (x_t - noise).abs().mean() / noise.abs().mean()
        assert signal_ratio < 0.3

    def test_different_t_different_noise(self, schedule):
        """Different timesteps should produce different noise levels."""
        x_0 = torch.randn(1, 64, 256)
        noise = torch.randn_like(x_0)
        x_t5 = schedule.q_sample(x_0, torch.tensor([5]), noise=noise)
        x_t15 = schedule.q_sample(x_0, torch.tensor([15]), noise=noise)
        assert not torch.allclose(x_t5, x_t15)

    def test_custom_noise(self, schedule):
        """Providing noise=zeros should give x_t = sqrt(alpha_bar) * x_0."""
        x_0 = torch.randn(2, 64, 256)
        t = torch.tensor([10, 10])
        noise = torch.zeros_like(x_0)
        x_t = schedule.q_sample(x_0, t, noise=noise)
        expected = schedule.sqrt_alpha_bar[10] * x_0
        assert torch.allclose(x_t, expected)


class TestDeviceTransfer:
    def test_to_cpu(self, schedule):
        schedule.to(torch.device("cpu"))
        assert schedule.alpha_bar.device == torch.device("cpu")

    def test_q_sample_after_move(self, schedule):
        schedule.to(torch.device("cpu"))
        x_0 = torch.randn(2, 64, 256)
        t = torch.tensor([5, 10])
        x_t = schedule.q_sample(x_0, t)
        assert x_t.shape == x_0.shape
