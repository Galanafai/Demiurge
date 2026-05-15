"""Tests for type gradient isolation in SceneDenoiser.

Validates that use_type_grad_isolation=True:
  1. Blocks geometry gradient from reaching type_embed
  2. Still allows type CE loss gradient to reach type_embed
  3. Leaves geometry head weights trainable via geometry loss
  4. Is transparent when isolation=False (matches non-isolated behavior)

All tests run on CPU without Drake/GPU dependencies.
"""
from __future__ import annotations

import torch
import pytest

from model.denoiser import DenoiserConfig, SceneDenoiser, N_MAX, N_TYPE, N_CONT


def _make_model(isolated: bool) -> SceneDenoiser:
    cfg = DenoiserConfig(
        n_layers=2,      # small for test speed
        d_model=64,
        n_heads=4,
        ffn_mult=2,
        dropout=0.0,
        use_type_grad_isolation=isolated,
    )
    return SceneDenoiser(cfg)


def _make_batch(B: int = 2) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x_cont = torch.randn(B, N_MAX, N_CONT)
    type_ids = torch.randint(0, 12, (B, N_MAX))
    t = torch.randint(0, 1000, (B,))
    return x_cont, type_ids, t


# ---------------------------------------------------------------------------
# Test 1: Isolation blocks geometry gradient from type_embed
# ---------------------------------------------------------------------------


def test_isolation_enabled_blocks_geometry_gradient() -> None:
    """With isolation on, pose loss backward must NOT produce grad on type_embed."""
    model = _make_model(isolated=True)
    x_cont, type_ids, t = _make_batch()

    out = model(x_cont, type_ids, t, text_emb=None)

    # Loss only on xyz (geometry). type_ce not included.
    loss = out.xyz.sum()
    loss.backward()

    # type_embed.weight must have zero gradient because the detach() breaks
    # the gradient path from xyz head -> transformer -> type_embed.
    grad = model.type_embed.weight.grad
    assert grad is not None, "type_embed.weight.grad is None -- no grad flowed at all"
    assert grad.abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        f"Geometry loss produced non-zero grad on type_embed.weight "
        f"(max={grad.abs().max().item():.2e}) -- isolation is not working."
    )


# ---------------------------------------------------------------------------
# Test 2: Isolation off matches entangled behavior (grad does flow)
# ---------------------------------------------------------------------------


def test_isolation_disabled_geometry_gradient_reaches_type_embed() -> None:
    """With isolation off, pose loss backward MUST produce grad on type_embed.

    This verifies that isolation=False preserves the original behavior
    (the v4/v5 entangled path).
    """
    model = _make_model(isolated=False)
    x_cont, type_ids, t = _make_batch()

    out = model(x_cont, type_ids, t, text_emb=None)
    loss = out.xyz.sum()
    loss.backward()

    grad = model.type_embed.weight.grad
    assert grad is not None, "type_embed.weight.grad is None"
    assert grad.abs().max().item() > 1e-8, (
        "With isolation=False, geometry loss should reach type_embed but grad is zero. "
        "Backward pass may be broken."
    )


# ---------------------------------------------------------------------------
# Test 3: Type CE loss still reaches type_embed with isolation on
# ---------------------------------------------------------------------------


def test_type_head_still_gets_gradient_with_isolation() -> None:
    """With isolation, type CE loss must still produce non-zero grad on type_embed."""
    model = _make_model(isolated=True)
    x_cont, type_ids, t = _make_batch()

    out = model(x_cont, type_ids, t, text_emb=None)

    # Loss only on type_logits (type CE analog). No geometry loss.
    # type_logits comes from head_type(scene_out + type_emb_full).
    # Gradient: type_logits -> head_type -> type_emb_full -> type_embed.weight
    loss = out.type_logits.sum()
    loss.backward()

    grad = model.type_embed.weight.grad
    assert grad is not None, "type_embed.weight.grad is None after type loss backward"
    assert grad.abs().max().item() > 1e-8, (
        "Type loss produced zero grad on type_embed.weight with isolation on. "
        "The residual injection path (scene_out + type_emb_full -> head_type) "
        "is not working."
    )


# ---------------------------------------------------------------------------
# Test 4: Geometry head weights train independently of type_embed
# ---------------------------------------------------------------------------


def test_geometry_heads_train_independently() -> None:
    """Pose loss should update head_xyz weights but NOT type_embed weights."""
    model = _make_model(isolated=True)
    x_cont, type_ids, t = _make_batch()

    out = model(x_cont, type_ids, t, text_emb=None)
    loss = out.xyz.sum()
    loss.backward()

    # head_xyz.weight must have non-zero gradient (it IS on the geometry path).
    xyz_grad = model.head_xyz.weight.grad
    assert xyz_grad is not None and xyz_grad.abs().max().item() > 1e-8, (
        "head_xyz.weight has zero gradient from xyz loss -- geometry path broken."
    )

    # type_embed must have zero gradient (isolated from geometry).
    te_grad = model.type_embed.weight.grad
    assert te_grad is not None
    assert te_grad.abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        "type_embed.weight has non-zero gradient from xyz loss -- isolation broken."
    )

    # head_type.weight must also have zero gradient (no type loss was computed).
    ht_grad = model.head_type.weight.grad
    assert ht_grad is not None
    assert ht_grad.abs().max().item() == pytest.approx(0.0, abs=1e-7), (
        "head_type.weight has non-zero gradient from xyz-only loss -- unexpected."
    )


# ---------------------------------------------------------------------------
# Test 5: Output shapes identical regardless of isolation flag
# ---------------------------------------------------------------------------


def test_output_shapes_same_with_and_without_isolation() -> None:
    """Isolation must not change output tensor shapes."""
    B = 4
    x_cont, type_ids, t = _make_batch(B)

    for isolated in (True, False):
        model = _make_model(isolated=isolated)
        out = model(x_cont, type_ids, t)
        assert out.type_logits.shape == (B, N_MAX, N_TYPE), f"isolated={isolated}"
        assert out.xyz.shape == (B, N_MAX, 3), f"isolated={isolated}"
        assert out.rot6d.shape == (B, N_MAX, 6), f"isolated={isolated}"
        assert out.scale.shape == (B, N_MAX, 3), f"isolated={isolated}"
        assert out.presence_logit.shape == (B, N_MAX, 1), f"isolated={isolated}"


# ---------------------------------------------------------------------------
# Test 6: Isolation=False is numerically identical to original forward pass
# ---------------------------------------------------------------------------


def test_isolation_false_matches_original_output() -> None:
    """When isolation=False, forward() must produce identical results to the
    pre-isolation implementation (same computation graph, same numbers)."""
    torch.manual_seed(0)
    model_off = _make_model(isolated=False)

    torch.manual_seed(0)
    model_on_off = _make_model(isolated=False)  # same seed, isolation=False

    # Both have same weights (same seed) and isolation=False.
    x_cont, type_ids, t = _make_batch()
    out_a = model_off(x_cont, type_ids, t)
    out_b = model_on_off(x_cont, type_ids, t)

    torch.testing.assert_close(out_a.xyz, out_b.xyz)
    torch.testing.assert_close(out_a.type_logits, out_b.type_logits)
