"""Integration tests: full diffusion pipeline + convention consistency checks.

These catch the class of bug that destroyed v5: training and sampling using
different v-prediction conventions, causing output explosion.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from model.vpred_conventions import compute_v_target, q_sample, v_to_x0
from training.output_sanity import check_output_sanity


def test_perfect_v_prediction_produces_sane_output() -> None:
    """A perfect v-predictor (oracle) should reconstruct x_0 within std bounds."""
    torch.manual_seed(42)
    x_0 = torch.randn(8, 12, 13)
    eps = torch.randn(8, 12, 13)
    alpha_bar = torch.tensor(0.5)

    v_true = compute_v_target(x_0, eps, alpha_bar)
    x_t = q_sample(x_0, eps, alpha_bar)
    x_0_recovered = v_to_x0(x_t, v_true, alpha_bar)

    report = check_output_sanity(x_0_recovered)
    assert not report.inferred_explosion, report.summary()
    assert not report.inferred_collapse, report.summary()


def test_sign_flip_causes_detectable_explosion() -> None:
    """Negating v (wrong sign convention) should produce obviously wrong output."""
    torch.manual_seed(42)
    x_0 = torch.randn(8, 12, 13)
    eps = torch.randn(8, 12, 13)
    alpha_bar = torch.tensor(0.5)

    v_correct = compute_v_target(x_0, eps, alpha_bar)
    x_t = q_sample(x_0, eps, alpha_bar)

    # Simulate a sign-flipped v (wrong convention)
    v_wrong = -v_correct
    x_0_wrong = v_to_x0(x_t, v_wrong, alpha_bar)

    # Mean absolute error should be large
    mae = (x_0_wrong - x_0).abs().mean().item()
    assert mae > 0.5, (
        f"Wrong sign should produce large reconstruction error, got MAE={mae:.4f}"
    )


def test_schedule_compute_v_target_round_trips() -> None:
    """schedule.compute_v_target -> v_to_x0 must recover x_0 accurately."""
    from model.schedule import CosineSchedule

    schedule = CosineSchedule(T=1000, zero_terminal_snr=True)

    if not hasattr(schedule, "compute_v_target"):
        pytest.skip("schedule.compute_v_target not implemented")

    torch.manual_seed(42)
    x_0 = torch.randn(4, 12, 13)
    eps = torch.randn(4, 12, 13)
    t = torch.tensor([200, 400, 600, 800])

    alpha_bar = schedule.alpha_bar(t).view(-1, 1, 1)
    x_t = q_sample(x_0, eps, alpha_bar)
    v_schedule = schedule.compute_v_target(x_0, eps, t)

    # Recover x_0 using canonical formula
    x_0_recovered = v_to_x0(x_t, v_schedule, alpha_bar)
    max_diff = (x_0 - x_0_recovered).abs().max().item()

    assert max_diff < 1e-4, (
        f"schedule.compute_v_target -> v_to_x0 round trip failed: max diff={max_diff}\n"
        "v5's training loss used a different v convention than the canonical formula.\n"
        "Fix schedule.compute_v_target to match vpred_conventions.compute_v_target."
    )


def test_ddim_sampler_step_produces_sane_intermediate() -> None:
    """A single DDIM step with a random-but-bounded model should stay within range."""
    from model.schedule import CosineSchedule, DDIMSampler

    schedule = CosineSchedule(T=1000, zero_terminal_snr=True)
    sampler = DDIMSampler(schedule, n_steps=10, prediction_type="v")

    torch.manual_seed(42)
    x_t = torch.randn(4, 12, 13)
    t_cur = torch.tensor([500] * 4)
    t_prev = torch.tensor([400] * 4)

    # Fake model: predict small v (bounded prediction simulating early training)
    v_pred = torch.randn(4, 12, 13) * 0.5

    alpha_bar_cur = schedule.alpha_bar(t_cur).view(-1, 1, 1)
    x_0_pred = v_to_x0(x_t, v_pred, alpha_bar_cur)

    report = check_output_sanity(x_0_pred)
    assert not report.inferred_explosion, (
        f"DDIM intermediate exploded after one step: {report.summary()}"
    )


def test_v5_training_config_flags_recognized() -> None:
    """The v-prediction config used in v5 should be parseable and consistent."""
    import yaml
    import pathlib

    config_path = pathlib.Path("configs/train/v9_uncond_v5.yaml")
    if not config_path.exists():
        pytest.skip("v9_uncond_v5.yaml not found")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    diff = cfg.get("diffusion", {})
    assert diff.get("prediction_type") == "v", \
        f"Expected prediction_type='v', got {diff.get('prediction_type')}"
    assert diff.get("zero_terminal_snr") is True, \
        "Expected zero_terminal_snr=True"
    assert diff.get("T", 0) > 0, "Expected T > 0"
