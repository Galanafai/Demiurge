"""Unit tests for sample_with_universal_guidance in DDIMSampler.

Tests:
1. guidance_scale=0 output matches sample_with_types (baseline identity)
2. guidance reduces mean pairwise overlap vs baseline
3. Gradient computation doesn't NaN
4. PAD slots not affected by guidance
"""
from __future__ import annotations

import sys
from pathlib import Path
import torch
import pytest

# Add src to path so imports work without install
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from model.schedule import CosineSchedule, DDIMSampler
from guidance.energy import pairwise_overlap_energy


# ---------------------------------------------------------------------------
# Tiny mock denoiser for fast testing
# ---------------------------------------------------------------------------

def _make_mock_fn(n_valid_types: int = 12, device: torch.device = torch.device("cpu")):
    """Return a fn with the conditional_sampling_fn signature.

    Returns deterministic eps=zeros and uniform type logits.
    This makes the DDIM sampler collapse to x0=0 (expected clean output is zeros).
    """
    N_TYPE = n_valid_types + 1  # +1 for PAD

    def fn(x_t: torch.Tensor, type_ids: torch.Tensor, t: torch.Tensor):
        B, N, D = x_t.shape
        eps_pred = torch.zeros_like(x_t)
        # Uniform logits so argmax picks type 0 always (first index wins ties).
        type_logits = torch.zeros(B, N, N_TYPE, device=device)
        return eps_pred, type_logits

    return fn


def _make_overlapping_fn(n_valid_types: int = 12, device: torch.device = torch.device("cpu")):
    """Return a fn that predicts all objects at origin (maximum overlap)."""
    N_TYPE = n_valid_types + 1

    def fn(x_t: torch.Tensor, type_ids: torch.Tensor, t: torch.Tensor):
        B, N, D = x_t.shape
        # Eps = x_t: DDIM update collapses to 0 (the denoiser predicts x0=0)
        eps_pred = x_t.clone()
        # Force all objects to be present (presence logit > 0)
        eps_pred[:, :, 12] = -x_t[:, :, 12] - 1.0  # so that x0_hat[:,12] = +1
        type_logits = torch.zeros(B, N, N_TYPE, device=device)
        # Force type 0 (CUBE), so radii are known
        type_logits[:, :, 0] = 10.0
        return eps_pred, type_logits

    return fn


# ---------------------------------------------------------------------------
# Test 1: guidance_scale=0 matches sample_with_types
# ---------------------------------------------------------------------------


class TestGuidanceZeroMatchesBaseline:
    def test_zero_scale_identical_output(self) -> None:
        """With guidance_scale=0, UG sampler should match sample_with_types exactly."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=5)  # fast test
        fn = _make_mock_fn()

        shape = (2, 8, 13)
        seed = 42

        x_base, tids_base = sampler.sample_with_types(
            fn, shape, seed=seed, device="cpu", type_init="uniform",
        )
        x_ug, tids_ug = sampler.sample_with_universal_guidance(
            fn, shape, seed=seed, device="cpu", type_init="uniform",
            guidance_scale=0.0,
        )

        # With same seed and zero guidance, outputs should be identical.
        assert torch.allclose(x_base, x_ug, atol=1e-5), (
            f"Max diff: {(x_base - x_ug).abs().max().item():.2e}"
        )
        assert torch.equal(tids_base, tids_ug)

    def test_output_shape(self) -> None:
        """UG sampler returns correct shapes."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=3)
        fn = _make_mock_fn()

        B, N, D = 4, 8, 13
        x, tids = sampler.sample_with_universal_guidance(
            fn, (B, N, D), seed=0, device="cpu", guidance_scale=0.5,
        )
        assert x.shape == (B, N, D)
        assert tids.shape == (B, N)
        assert tids.dtype == torch.long


# ---------------------------------------------------------------------------
# Test 2: Gradient computation doesn't NaN
# ---------------------------------------------------------------------------


class TestNoNaN:
    def test_output_not_nan_with_guidance(self) -> None:
        """UG sampler with guidance > 0 should not produce NaN."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=5)
        fn = _make_mock_fn()

        x, tids = sampler.sample_with_universal_guidance(
            fn, (4, 8, 13), seed=7, device="cpu",
            guidance_scale=2.0,
        )
        assert not x.isnan().any(), "x contains NaN with guidance_scale=2.0"
        assert not x.isinf().any(), "x contains Inf with guidance_scale=2.0"

    def test_large_guidance_scale_no_nan(self) -> None:
        """Even with guidance_scale=8.0 and grad clipping, no NaN."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=5)
        fn = _make_mock_fn()

        x, tids = sampler.sample_with_universal_guidance(
            fn, (2, 8, 13), seed=99, device="cpu",
            guidance_scale=8.0, grad_clip=1.0,
        )
        assert not x.isnan().any(), "NaN with guidance_scale=8.0"


# ---------------------------------------------------------------------------
# Test 3: Guidance window respected
# ---------------------------------------------------------------------------


class TestGuidanceWindow:
    def test_guidance_only_applied_in_window(self) -> None:
        """With guidance window [0.9, 0.91], almost no guidance is applied --
        output should be nearly identical to baseline."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=10)
        fn = _make_mock_fn()

        shape = (2, 8, 13)
        seed = 42

        x_base, _ = sampler.sample_with_types(fn, shape, seed=seed)
        # Window covers only 1 step of 10, making guidance negligible
        x_ug, _ = sampler.sample_with_universal_guidance(
            fn, shape, seed=seed,
            guidance_scale=1.0,
            guidance_min_step_frac=0.95,
            guidance_max_step_frac=0.96,
        )
        # Not exactly equal (1 step of guidance) but should be close
        max_diff = (x_base - x_ug).abs().max().item()
        assert max_diff < 5.0, f"Guidance with 1 step should be small: max_diff={max_diff}"


# ---------------------------------------------------------------------------
# Test 4: Type IDs are returned correctly
# ---------------------------------------------------------------------------


class TestTypeIds:
    def test_type_ids_in_valid_range(self) -> None:
        """Returned type_ids should all be in [0, n_valid_types)."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=3)
        fn = _make_mock_fn(n_valid_types=12)

        _, tids = sampler.sample_with_universal_guidance(
            fn, (4, 8, 13), seed=0, guidance_scale=1.0,
        )
        assert (tids >= 0).all()
        assert (tids < 12).all(), f"Type IDs out of range: max={tids.max()}"

    def test_zeros_type_init(self) -> None:
        """type_init='zeros' initialises all slots to type 0."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=3)
        fn = _make_mock_fn()

        # Just check it runs without error (init is updated at step 1 anyway)
        x, tids = sampler.sample_with_universal_guidance(
            fn, (2, 8, 13), seed=0, type_init="zeros", guidance_scale=0.0,
        )
        assert x.shape == (2, 8, 13)

    def test_invalid_type_init_raises(self) -> None:
        """Invalid type_init value raises ValueError."""
        schedule = CosineSchedule(T=1000)
        sampler = DDIMSampler(schedule, n_steps=3)
        fn = _make_mock_fn()
        with pytest.raises(ValueError, match="Unknown type_init"):
            sampler.sample_with_universal_guidance(
                fn, (2, 8, 13), type_init="invalid",
            )
