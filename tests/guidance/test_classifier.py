"""Tests for src/guidance/classifier.py -- ValidityClassifier."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from guidance.classifier import ValidityClassifier  # noqa: E402
from model.denoiser import N_CONT, N_MAX, N_TYPE  # noqa: E402


def _random_batch(B: int = 4, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    return {
        "x_t": torch.randn(B, N_MAX, N_CONT),
        "type_ids": torch.randint(0, N_TYPE, (B, N_MAX)),
        "t": torch.randint(0, 1000, (B,)),
    }


# ---------------------------------------------------------------------------
# Test 1: output shape
# ---------------------------------------------------------------------------


def test_classifier_forward_shape() -> None:
    """Output must be scalar logit per sample: shape (B,)."""
    B = 8
    model = ValidityClassifier()
    batch = _random_batch(B=B, seed=0)

    with torch.no_grad():
        logit = model(batch["x_t"], batch["type_ids"], batch["t"])

    assert logit.shape == (B,), (
        f"ValidityClassifier output shape should be ({B},), got {logit.shape}"
    )
    assert logit.dtype == torch.float32, f"Output should be float32, got {logit.dtype}"


# ---------------------------------------------------------------------------
# Test 2: gradient flows through x_t
# ---------------------------------------------------------------------------


def test_classifier_gradient_flows() -> None:
    """Backward through log_prob_valid must produce grad w.r.t. x_t."""
    model = ValidityClassifier()
    model.eval()
    batch = _random_batch(B=4, seed=1)

    x_t = batch["x_t"].requires_grad_(True)
    log_p = model.log_prob_valid(x_t, batch["type_ids"], batch["t"])
    log_p.sum().backward()

    assert x_t.grad is not None, "Gradient w.r.t. x_t must be populated after backward."
    assert x_t.grad.shape == x_t.shape, (
        f"Gradient shape {x_t.grad.shape} does not match x_t shape {x_t.shape}"
    )
    assert x_t.grad.abs().max().item() > 1e-10, (
        "Gradient w.r.t. x_t is effectively zero. Check classifier forward pass."
    )


# ---------------------------------------------------------------------------
# Test 3: gradient is non-zero on continuous dims (xyz/rot6d/scale)
# ---------------------------------------------------------------------------


def test_classifier_grad_wrt_continuous_dims() -> None:
    """Gradient must be non-trivially non-zero on xyz, rot6d, scale dims [0:12]."""
    model = ValidityClassifier()
    model.eval()
    batch = _random_batch(B=4, seed=2)

    x_t = batch["x_t"].requires_grad_(True)
    log_p = model.log_prob_valid(x_t, batch["type_ids"], batch["t"])
    log_p.sum().backward()

    grad = x_t.grad
    assert grad is not None
    # Continuous dims are [0:12]. Check each sub-range.
    xyz_grad_norm = grad[:, :, 0:3].norm().item()
    rot_grad_norm = grad[:, :, 3:9].norm().item()
    scale_grad_norm = grad[:, :, 9:12].norm().item()

    assert xyz_grad_norm > 1e-8, (
        f"Gradient on xyz dims should be non-zero, got norm={xyz_grad_norm:.2e}"
    )
    assert rot_grad_norm > 1e-8, (
        f"Gradient on rot6d dims should be non-zero, got norm={rot_grad_norm:.2e}"
    )
    assert scale_grad_norm > 1e-8, (
        f"Gradient on scale dims should be non-zero, got norm={scale_grad_norm:.2e}"
    )


# ---------------------------------------------------------------------------
# Test 4: log_prob_valid returns values in (-inf, 0]
# ---------------------------------------------------------------------------


def test_log_prob_valid_range() -> None:
    """log_prob_valid = log sigmoid(logit) must be <= 0."""
    model = ValidityClassifier()
    model.eval()
    batch = _random_batch(B=16, seed=3)

    with torch.no_grad():
        log_p = model.log_prob_valid(batch["x_t"], batch["type_ids"], batch["t"])

    assert (log_p <= 0).all(), (
        f"log_prob_valid must be <= 0 (log probability). "
        f"Got max={log_p.max().item():.4f}"
    )
    assert torch.isfinite(log_p).all(), "log_prob_valid must be finite."
