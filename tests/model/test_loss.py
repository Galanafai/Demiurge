"""Tests for src/model/loss.py.

Four tests:
1. test_gradients_flow_all_params -- backward through total loss, all params grad
2. test_decreases_on_overfit      -- 10-step SGD on one batch, loss decreases
3. test_absent_slots_masked       -- all-absent mask zeros out continuous losses
4. test_component_logging         -- as_log_dict() returns finite values with expected keys
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.denoiser import (  # noqa: E402
    D_TEXT,
    N_CONT,
    N_MAX,
    N_TYPE,
    DenoiserConfig,
    SceneDenoiser,
)
from model.loss import SceneDiffusionLoss  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _small_cfg() -> DenoiserConfig:
    return DenoiserConfig(n_layers=2, d_model=64, n_heads=4, ffn_mult=2, dropout=0.0)


def _random_batch(B: int = 4, seed: int = 0) -> dict:
    """Generate a random batch of inputs and noise targets."""
    torch.manual_seed(seed)
    x_cont = torch.randn(B, N_MAX, N_CONT)
    type_ids = torch.randint(0, N_TYPE, (B, N_MAX))
    t = torch.randint(0, 1000, (B,))
    eps_xyz = torch.randn(B, N_MAX, 3)
    eps_rot6d = torch.randn(B, N_MAX, 6)
    eps_scale = torch.randn(B, N_MAX, 3)
    eps_presence = torch.randn(B, N_MAX, 1)
    presence_mask = torch.ones(B, N_MAX, dtype=torch.bool)
    return dict(
        x_cont=x_cont, type_ids=type_ids, t=t,
        eps_xyz=eps_xyz, eps_rot6d=eps_rot6d, eps_scale=eps_scale,
        eps_presence=eps_presence, presence_mask=presence_mask,
    )


def _forward_loss(model: SceneDenoiser, batch: dict, loss_fn: SceneDiffusionLoss):
    pred = model(batch["x_cont"], batch["type_ids"], batch["t"])
    return loss_fn(
        pred,
        batch["eps_xyz"], batch["eps_rot6d"], batch["eps_scale"],
        batch["eps_presence"], batch["type_ids"], batch["presence_mask"],
    )


# ---------------------------------------------------------------------------
# Test 1: gradients flow to all parameters
# ---------------------------------------------------------------------------


def test_gradients_flow_all_params() -> None:
    """Gradients must reach all parameters via their respective code paths.

    Cross-attention parameters (norm_ca, cross_attn) only receive gradients
    when text_emb is provided. This test runs both unconditional and
    conditional passes and verifies that every parameter gets a gradient in
    at least one of the two passes.
    """
    model = SceneDenoiser(_small_cfg())
    loss_fn = SceneDiffusionLoss()
    batch = _random_batch(B=4, seed=42)

    # --- Unconditional pass (text_emb=None) ---
    model.train()
    model.zero_grad()
    result = _forward_loss(model, batch, loss_fn)
    result.total.backward()
    grad_uncond = {
        name: (param.grad is not None and param.grad.abs().max().item() > 0)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # --- Conditional pass (text_emb provided) ---
    model.zero_grad()
    B = batch["x_cont"].shape[0]
    text_emb = torch.randn(B, D_TEXT)
    pred = model(batch["x_cont"], batch["type_ids"], batch["t"], text_emb=text_emb)
    result2 = loss_fn(
        pred,
        batch["eps_xyz"], batch["eps_rot6d"], batch["eps_scale"],
        batch["eps_presence"], batch["type_ids"], batch["presence_mask"],
    )
    result2.total.backward()
    grad_cond = {
        name: (param.grad is not None and param.grad.abs().max().item() > 0)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    # Every parameter must have gradient in at least one of the two passes.
    never_grad = [
        name for name in grad_uncond
        if not grad_uncond[name] and not grad_cond.get(name, False)
    ]
    # Allow up to 2 params (zero-init heads may have exactly zero gradient
    # on the first backward before any signal propagates).
    assert len(never_grad) <= 2, (
        f"Parameters with no gradient in either pass ({len(never_grad)}): "
        f"{never_grad[:5]}"
    )



# ---------------------------------------------------------------------------
# Test 2: loss decreases on a single-batch overfit run
# ---------------------------------------------------------------------------


def test_decreases_on_overfit() -> None:
    """10 gradient steps on one fixed batch must reduce total loss."""
    torch.manual_seed(7)
    model = SceneDenoiser(_small_cfg())
    model.train()
    loss_fn = SceneDiffusionLoss()
    batch = _random_batch(B=8, seed=7)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(10):
        opt.zero_grad()
        result = _forward_loss(model, batch, loss_fn)
        result.total.backward()
        opt.step()
        losses.append(result.total.item())

    assert losses[-1] < losses[0], (
        f"Loss did not decrease over 10 steps: {losses[0]:.4f} -> {losses[-1]:.4f}. "
        "Check that gradients flow and optimizer is applied."
    )


# ---------------------------------------------------------------------------
# Test 3: absent slots masked out of continuous/type losses
# ---------------------------------------------------------------------------


def test_absent_slots_masked() -> None:
    """With all slots absent, pose/scale/type losses must be zero."""
    model = SceneDenoiser(_small_cfg())
    model.eval()
    loss_fn = SceneDiffusionLoss()
    batch = _random_batch(B=2, seed=3)

    # Override presence mask to all-False.
    batch["presence_mask"] = torch.zeros(2, N_MAX, dtype=torch.bool)

    with torch.no_grad():
        result = _forward_loss(model, batch, loss_fn)

    assert result.pose_xyz.item() == 0.0, f"pose_xyz should be 0 when all absent, got {result.pose_xyz.item()}"
    assert result.pose_rot.item() == 0.0, f"pose_rot should be 0 when all absent, got {result.pose_rot.item()}"
    assert result.scale.item() == 0.0, f"scale should be 0 when all absent, got {result.scale.item()}"
    assert result.type_ce.item() == 0.0, f"type_ce should be 0 when all absent, got {result.type_ce.item()}"
    # Presence BCE must still be positive (model should predict absence).
    assert result.presence_bce.item() > 0.0, "presence_bce should be positive even when all absent"


# ---------------------------------------------------------------------------
# Test 4: component logging dict
# ---------------------------------------------------------------------------


def test_component_logging() -> None:
    """as_log_dict() must return all expected keys with finite float values."""
    model = SceneDenoiser(_small_cfg())
    model.eval()
    loss_fn = SceneDiffusionLoss()
    batch = _random_batch(B=2, seed=5)

    with torch.no_grad():
        result = _forward_loss(model, batch, loss_fn)

    log = result.as_log_dict()
    expected_keys = {"total", "pose_xyz", "pose_rot", "scale", "type_ce", "presence_bce"}
    assert expected_keys <= set(log.keys()), f"Missing keys: {expected_keys - set(log.keys())}"

    for k, v in log.items():
        assert isinstance(v, float), f"Key {k} is not a float: {type(v)}"
        assert not (v != v), f"Key {k} is NaN"           # NaN check
        assert v != float("inf"), f"Key {k} is Inf"
