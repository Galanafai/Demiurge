"""Tests for class_weights extension to src/model/loss.py.

Three tests:
1. test_class_weights_changes_loss      -- minority-class errors penalized more
2. test_class_weights_none_backward_compat -- None gives identical output
3. test_type_head_grad_norm_with_weights -- grad norm on head_type exceeds 1e-5
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.denoiser import (  # noqa: E402
    N_CONT,
    N_MAX,
    N_TYPE,
    DenoiserConfig,
    SceneDenoiser,
)
from model.loss import LossWeights, SceneDiffusionLoss  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_cfg() -> DenoiserConfig:
    return DenoiserConfig(n_layers=2, d_model=64, n_heads=4, ffn_mult=2, dropout=0.0)


def _random_batch(B: int = 4, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return dict(
        x_cont=torch.randn(B, N_MAX, N_CONT),
        type_ids=torch.randint(0, N_TYPE, (B, N_MAX)),
        t=torch.randint(0, 1000, (B,)),
        eps_xyz=torch.randn(B, N_MAX, 3),
        eps_rot6d=torch.randn(B, N_MAX, 6),
        eps_scale=torch.randn(B, N_MAX, 3),
        eps_presence=torch.randn(B, N_MAX, 1),
        presence_mask=torch.ones(B, N_MAX, dtype=torch.bool),
    )


def _forward_loss(model: SceneDenoiser, batch: dict, loss_fn: SceneDiffusionLoss):
    pred = model(batch["x_cont"], batch["type_ids"], batch["t"])
    return loss_fn(
        pred,
        batch["eps_xyz"], batch["eps_rot6d"], batch["eps_scale"],
        batch["eps_presence"], batch["type_ids"], batch["presence_mask"],
    )


# ---------------------------------------------------------------------------
# Test 1: class_weights changes the type_ce loss value
# ---------------------------------------------------------------------------


def test_class_weights_changes_loss() -> None:
    """Minority-class upweighting must produce a different (typically higher) loss.

    We construct weights that heavily upweight every class except type_id=0,
    then check that the type_ce component differs from the uniform-weight case.
    The losses can go either direction depending on the batch composition, so
    we just assert they differ (not equal) within reasonable precision.
    """
    torch.manual_seed(42)
    model = SceneDenoiser(_small_cfg())
    model.eval()

    # Batch where type_ids are a mix of 0 and non-zero.
    batch = _random_batch(B=8, seed=42)
    # Force type_ids to have some 0s and some non-zeros for meaningful test.
    batch["type_ids"] = torch.tensor([0, 1, 2, 3, 0, 4, 5, 0]).unsqueeze(1).expand(8, N_MAX).clone()

    loss_uniform = SceneDiffusionLoss(
        LossWeights(type_ce=1.0),
        class_weights=None,
    )
    # Heavily upweight non-zero types (20x), downweight type_id=0.
    cw = torch.ones(N_TYPE)
    cw[0] = 0.05  # majority class -- very low weight
    for tid in range(1, N_TYPE):
        cw[tid] = 20.0  # minority classes -- high weight

    loss_weighted = SceneDiffusionLoss(
        LossWeights(type_ce=1.0),
        class_weights=cw,
    )

    with torch.no_grad():
        out_uniform = _forward_loss(model, batch, loss_uniform)
        out_weighted = _forward_loss(model, batch, loss_weighted)

    assert abs(out_uniform.type_ce.item() - out_weighted.type_ce.item()) > 1e-6, (
        f"type_ce should differ with class_weights. "
        f"Uniform: {out_uniform.type_ce.item():.6f}, "
        f"Weighted: {out_weighted.type_ce.item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: class_weights=None preserves original behavior
# ---------------------------------------------------------------------------


def test_class_weights_none_backward_compat() -> None:
    """Passing class_weights=None must give identical output to original loss_fn."""
    torch.manual_seed(1)
    model = SceneDenoiser(_small_cfg())
    model.eval()
    batch = _random_batch(B=4, seed=1)

    # Original signature (no class_weights kwarg).
    loss_original = SceneDiffusionLoss(LossWeights())
    # New signature with explicit None.
    loss_none = SceneDiffusionLoss(LossWeights(), class_weights=None)

    with torch.no_grad():
        out_orig = _forward_loss(model, batch, loss_original)
        out_none = _forward_loss(model, batch, loss_none)

    assert abs(out_orig.total.item() - out_none.total.item()) < 1e-6, (
        f"class_weights=None should give identical total loss. "
        f"Original: {out_orig.total.item():.8f}, None: {out_none.total.item():.8f}"
    )
    assert abs(out_orig.type_ce.item() - out_none.type_ce.item()) < 1e-6, (
        f"class_weights=None should give identical type_ce. "
        f"Original: {out_orig.type_ce.item():.8f}, None: {out_none.type_ce.item():.8f}"
    )


# ---------------------------------------------------------------------------
# Test 3: gradient norm on type head with weights active
# ---------------------------------------------------------------------------


def test_type_head_grad_norm_with_weights() -> None:
    """With type_ce=1.0 and class_weights active, head_type gradient norm > 1e-5."""
    torch.manual_seed(99)
    model = SceneDenoiser(_small_cfg())
    model.train()

    cw = torch.ones(N_TYPE)
    cw[0] = 0.05
    for tid in range(1, N_TYPE):
        cw[tid] = 5.0

    loss_fn = SceneDiffusionLoss(LossWeights(type_ce=1.0), class_weights=cw)
    batch = _random_batch(B=8, seed=99)

    model.zero_grad()
    out = _forward_loss(model, batch, loss_fn)
    out.total.backward()

    head_type_grad_norm = model.head_type.weight.grad.norm().item()
    assert head_type_grad_norm > 1e-5, (
        f"head_type gradient norm should be > 1e-5 with type_ce=1.0 and class_weights. "
        f"Got {head_type_grad_norm:.2e}. Check loss computation."
    )
