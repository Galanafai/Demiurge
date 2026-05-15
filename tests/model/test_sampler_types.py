"""Tests for DDIMSampler.sample_with_types and SceneDenoiser.conditional_sampling_fn.

Validates the fix for the training/inference type_ids mismatch that caused
object-type collapse in v2/v3/v4. Uses a tiny model (d_model=32, 1 layer)
on CPU to keep the test suite fast.
"""
from __future__ import annotations

import torch
import pytest

from model.denoiser import DenoiserConfig, SceneDenoiser, N_MAX, N_CONT, N_TYPE
from model.schedule import CosineSchedule, DDIMSampler


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_model() -> SceneDenoiser:
    """Tiny model for fast CPU tests (d_model=32, 1 layer, 2 heads)."""
    cfg = DenoiserConfig(n_layers=1, d_model=32, n_heads=2, ffn_mult=2, dropout=0.0)
    model = SceneDenoiser(cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def sampler_5step() -> DDIMSampler:
    schedule = CosineSchedule(T=1000)
    return DDIMSampler(schedule, n_steps=5)


# ---------------------------------------------------------------------------
# Test 1: sample_with_types returns correct output shapes and types
# ---------------------------------------------------------------------------


def test_sampler_returns_type_ids(tiny_model: SceneDenoiser, sampler_5step: DDIMSampler) -> None:
    """sample_with_types must return (x_cont, type_ids) with correct shapes."""
    fn = tiny_model.conditional_sampling_fn(text_emb=None)
    x_cont, type_ids = sampler_5step.sample_with_types(
        fn, shape=(3, N_MAX, N_CONT), seed=0, device="cpu"
    )
    assert x_cont.shape == (3, N_MAX, N_CONT), f"x_cont shape {x_cont.shape}"
    assert type_ids.shape == (3, N_MAX), f"type_ids shape {type_ids.shape}"
    assert type_ids.dtype == torch.long, f"type_ids dtype {type_ids.dtype}"
    assert type_ids.min().item() >= 0
    assert type_ids.max().item() < N_TYPE


# ---------------------------------------------------------------------------
# Test 2: conditional_sampling_fn returns (eps, type_logits) with correct shapes
# ---------------------------------------------------------------------------


def test_conditional_sampling_fn_output_shapes(tiny_model: SceneDenoiser) -> None:
    """conditional_sampling_fn must return (eps, type_logits) with matching shapes."""
    fn = tiny_model.conditional_sampling_fn(text_emb=None)
    B = 2
    x_t = torch.randn(B, N_MAX, N_CONT)
    type_ids = torch.zeros(B, N_MAX, dtype=torch.long)
    t = torch.zeros(B, dtype=torch.long)
    eps, logits = fn(x_t, type_ids, t)
    assert eps.shape == (B, N_MAX, N_CONT), f"eps shape {eps.shape}"
    assert logits.shape == (B, N_MAX, N_TYPE), f"logits shape {logits.shape}"


# ---------------------------------------------------------------------------
# Test 3: sampler is deterministic with same seed
# ---------------------------------------------------------------------------


def test_sampler_deterministic_with_seed(tiny_model: SceneDenoiser, sampler_5step: DDIMSampler) -> None:
    """Two calls with the same seed must produce identical (x_cont, type_ids)."""
    fn = tiny_model.conditional_sampling_fn(text_emb=None)
    shape = (2, N_MAX, N_CONT)
    x1, t1 = sampler_5step.sample_with_types(fn, shape, seed=7, device="cpu")
    x2, t2 = sampler_5step.sample_with_types(fn, shape, seed=7, device="cpu")
    assert torch.allclose(x1, x2, atol=1e-6), "x_cont not deterministic across seeds"
    assert (t1 == t2).all(), "type_ids not deterministic across seeds"


# ---------------------------------------------------------------------------
# Test 4: different seeds produce different outputs
# ---------------------------------------------------------------------------


def test_sampler_different_seeds_differ(tiny_model: SceneDenoiser, sampler_5step: DDIMSampler) -> None:
    """Different seeds must produce different outputs (with overwhelming probability)."""
    fn = tiny_model.conditional_sampling_fn(text_emb=None)
    shape = (4, N_MAX, N_CONT)
    x1, _ = sampler_5step.sample_with_types(fn, shape, seed=0, device="cpu")
    x2, _ = sampler_5step.sample_with_types(fn, shape, seed=1, device="cpu")
    assert not torch.allclose(x1, x2, atol=1e-4), "Different seeds produced identical x_cont"


# ---------------------------------------------------------------------------
# Test 5: type diversity -- uniform init must not collapse to all-same type
# ---------------------------------------------------------------------------


def test_sampler_type_distribution_diverse(tiny_model: SceneDenoiser) -> None:
    """With uniform type_init, a batch of 20 scenes must produce >1 distinct type.

    A freshly-initialised (untrained) model with random weights will produce
    nearly-uniform type logits -- this test verifies the argmax chain is
    exercised and not trivially short-circuited.
    """
    schedule = CosineSchedule(T=1000)
    sampler = DDIMSampler(schedule, n_steps=3)
    fn = tiny_model.conditional_sampling_fn(text_emb=None)
    _, type_ids = sampler.sample_with_types(
        fn, shape=(20, N_MAX, N_CONT), seed=42, device="cpu", type_init="uniform"
    )
    n_unique_types = type_ids.unique().numel()
    assert n_unique_types > 1, (
        f"Expected >1 distinct type, got {n_unique_types}. "
        "This suggests the sampler argmax chain is not running."
    )


# ---------------------------------------------------------------------------
# Test 6: zeros init replicates legacy (broken) behavior -- for regression docs
# ---------------------------------------------------------------------------


def test_sampler_zeros_init_matches_legacy(
    tiny_model: SceneDenoiser, sampler_5step: DDIMSampler
) -> None:
    """type_init='zeros' must produce the same x_cont as the legacy noise_prediction_fn.

    This documents the bug: when type_ids=zeros at every step, sample_with_types
    produces the same continuous trajectory as the old broken sampler. The type_ids
    output will still differ (legacy returned zeros post-hoc; new sampler returns
    argmax of head_type applied to zero-initialized type context).
    """
    shape = (2, N_MAX, N_CONT)
    seed = 13

    # Legacy sampler: type_ids frozen at zero, uses old sample() API.
    legacy_fn = tiny_model.noise_prediction_fn(text_emb=None)
    x_legacy = sampler_5step.sample(legacy_fn, shape, seed=seed, device="cpu")

    # New sampler with zeros init: should replicate the same continuous trajectory
    # because type_ids=zeros at every step identical to legacy behavior.
    new_fn = tiny_model.conditional_sampling_fn(text_emb=None)
    x_new, _ = sampler_5step.sample_with_types(
        new_fn, shape, seed=seed, device="cpu", type_init="zeros"
    )

    assert torch.allclose(x_legacy, x_new, atol=1e-5), (
        "zeros init continuous trajectory does not match legacy sampler. "
        "The regression baseline is broken."
    )


# ---------------------------------------------------------------------------
# Test 7: invalid type_init raises ValueError
# ---------------------------------------------------------------------------


def test_sampler_invalid_type_init_raises(
    tiny_model: SceneDenoiser, sampler_5step: DDIMSampler
) -> None:
    fn = tiny_model.conditional_sampling_fn(text_emb=None)
    with pytest.raises(ValueError, match="Unknown type_init"):
        sampler_5step.sample_with_types(
            fn, (1, N_MAX, N_CONT), seed=0, device="cpu", type_init="bad_value"
        )


# ---------------------------------------------------------------------------
# Test 8: noise_prediction_fn (deprecated) still works for backward compat
# ---------------------------------------------------------------------------


def test_noise_prediction_fn_backward_compat(
    tiny_model: SceneDenoiser, sampler_5step: DDIMSampler
) -> None:
    """noise_prediction_fn must still be callable and return correct eps shape."""
    fn = tiny_model.noise_prediction_fn(text_emb=None)
    x_t = torch.randn(2, N_MAX, N_CONT)
    t = torch.zeros(2, dtype=torch.long)
    eps = fn(x_t, t, None)
    assert eps.shape == (2, N_MAX, N_CONT), f"Legacy fn eps shape {eps.shape}"
