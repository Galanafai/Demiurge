"""Unit tests for v-prediction conventions.

These would have caught v5's output explosion bug immediately.
Each test verifies a mathematical identity that MUST hold for v-prediction
to work correctly.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from model.vpred_conventions import (
    compute_v_target,
    eps_to_x0,
    q_sample,
    v_to_eps,
    v_to_x0,
    x0_to_v,
)


def test_v_round_trip_via_x0() -> None:
    """v -> x_0 -> v should be identity."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    alpha_bar = torch.tensor(0.7)

    v = compute_v_target(x_0, eps, alpha_bar)
    x_t = q_sample(x_0, eps, alpha_bar)
    x_0_recovered = v_to_x0(x_t, v, alpha_bar)

    max_diff = (x_0 - x_0_recovered).abs().max().item()
    assert max_diff < 1e-4, f"v -> x_0 round trip failed: max diff={max_diff}"


def test_v_round_trip_via_eps() -> None:
    """v -> eps -> v should be identity."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    alpha_bar = torch.tensor(0.7)

    v = compute_v_target(x_0, eps, alpha_bar)
    x_t = q_sample(x_0, eps, alpha_bar)
    eps_recovered = v_to_eps(x_t, v, alpha_bar)

    max_diff = (eps - eps_recovered).abs().max().item()
    assert max_diff < 1e-4, f"v -> eps round trip failed: max diff={max_diff}"


def test_q_sample_matches_formula() -> None:
    """q_sample should match sqrt(ab)*x0 + sqrt(1-ab)*eps exactly."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)

    for ab in [0.001, 0.1, 0.5, 0.9, 0.999]:
        alpha_bar = torch.tensor(ab)
        x_t = q_sample(x_0, eps, alpha_bar)
        expected = alpha_bar.sqrt() * x_0 + (1 - alpha_bar).sqrt() * eps
        assert torch.allclose(x_t, expected, atol=1e-6), f"q_sample mismatch at ab={ab}"


def test_v_eps_x0_consistency() -> None:
    """Recovered x_0 and eps must satisfy q_sample."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    alpha_bar = torch.tensor(0.5)

    v = compute_v_target(x_0, eps, alpha_bar)
    x_t = q_sample(x_0, eps, alpha_bar)

    x_0_hat = v_to_x0(x_t, v, alpha_bar)
    eps_hat = v_to_eps(x_t, v, alpha_bar)
    x_t_recovered = q_sample(x_0_hat, eps_hat, alpha_bar)

    max_diff = (x_t - x_t_recovered).abs().max().item()
    assert max_diff < 1e-3, f"x_t consistency failed: max diff={max_diff}"


def test_eps_prediction_round_trip() -> None:
    """Epsilon prediction round trip: x_0 -> x_t -> x_0."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    alpha_bar = torch.tensor(0.5)

    x_t = q_sample(x_0, eps, alpha_bar)
    x_0_recovered = eps_to_x0(x_t, eps, alpha_bar)

    max_diff = (x_0 - x_0_recovered).abs().max().item()
    assert max_diff < 1e-4, f"eps prediction round trip failed: max diff={max_diff}"


def test_v_target_at_zero_snr() -> None:
    """At alpha_bar=0 (zero terminal SNR), v should equal -x_0."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    alpha_bar = torch.tensor(0.0)

    v = compute_v_target(x_0, eps, alpha_bar)
    expected = -x_0  # sqrt(0)*eps - sqrt(1)*x_0 = -x_0

    assert torch.allclose(v, expected, atol=1e-6), \
        f"v at ab=0 should be -x_0; max diff={(v - expected).abs().max():.6f}"


def test_v_target_at_alpha_bar_one() -> None:
    """At alpha_bar=1 (no noise), v should equal eps."""
    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    alpha_bar = torch.tensor(1.0)

    v = compute_v_target(x_0, eps, alpha_bar)
    expected = eps  # sqrt(1)*eps - sqrt(0)*x_0 = eps

    assert torch.allclose(v, expected, atol=1e-6)


def test_v_distribution_is_unit_normal() -> None:
    """If x_0,eps ~ N(0,1) then v ~ N(0,1) for any alpha_bar."""
    torch.manual_seed(42)
    x_0 = torch.randn(2000, 12, 13)
    eps = torch.randn(2000, 12, 13)

    for ab in [0.1, 0.5, 0.9]:
        alpha_bar = torch.tensor(ab)
        v = compute_v_target(x_0, eps, alpha_bar)

        assert abs(v.mean().item()) < 0.05, f"v mean at ab={ab}: {v.mean():.4f}"
        assert abs(v.std().item() - 1.0) < 0.05, \
            f"v std at ab={ab}: {v.std():.4f} (expect 1.0)"


def test_schedule_compute_v_target_matches_canonical() -> None:
    """schedule.py compute_v_target must match vpred_conventions canonical."""
    from model.schedule import CosineSchedule

    schedule = CosineSchedule(T=1000, zero_terminal_snr=True)

    if not hasattr(schedule, "compute_v_target"):
        pytest.skip("schedule.compute_v_target not implemented")

    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    t = torch.tensor([500] * 4)

    v_schedule = schedule.compute_v_target(x_0, eps, t)

    alpha_bar = schedule.alpha_bar(t).view(-1, 1, 1)
    v_canonical = compute_v_target(x_0, eps, alpha_bar)

    max_diff = (v_schedule - v_canonical).abs().max().item()
    assert max_diff < 1e-5, (
        f"schedule.compute_v_target differs from canonical: max diff={max_diff}\n"
        "This is the v5 bug. Check sign convention in schedule.py."
    )


def test_schedule_predict_x0_from_v_matches_canonical() -> None:
    """schedule predict_x0_from_v must invert compute_v_target exactly."""
    from model.schedule import CosineSchedule

    schedule = CosineSchedule(T=1000, zero_terminal_snr=True)

    has_method = hasattr(schedule, "predict_x0_from_v") or hasattr(schedule, "v_to_x0")
    if not has_method:
        pytest.skip("schedule predict_x0_from_v not implemented")

    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    t = torch.tensor([500] * 4)
    alpha_bar = schedule.alpha_bar(t).view(-1, 1, 1)

    x_t = q_sample(x_0, eps, alpha_bar)
    v = compute_v_target(x_0, eps, alpha_bar)
    x_0_canonical = v_to_x0(x_t, v, alpha_bar)

    if hasattr(schedule, "predict_x0_from_v"):
        x_0_schedule = schedule.predict_x0_from_v(x_t, t, v)
    else:
        x_0_schedule = schedule.v_to_x0(x_t, t, v)

    max_diff = (x_0_schedule - x_0_canonical).abs().max().item()
    assert max_diff < 1e-5, (
        f"schedule.predict_x0_from_v differs from canonical: max diff={max_diff}\n"
        "This causes output explosion at sampling time."
    )
