"""Tests for presence loss with class balancing.

v5 had presence collapse: 27.6% active at 30k, 16.1% at 60k (worsening).
These tests verify the pos_weight calculation and BCE computation.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import torch.nn.functional as F
import pytest

from model.presence_loss import (
    PRESENCE_POS_WEIGHT,
    compute_presence_pos_weight,
    presence_bce_balanced,
)


def test_pos_weight_formula() -> None:
    """pos_weight = (1 - p) / p for occupancy p."""
    assert compute_presence_pos_weight(0.5) == 1.0
    assert compute_presence_pos_weight(0.25) == 3.0
    assert abs(compute_presence_pos_weight(0.244) - 3.098) < 0.01


def test_pos_weight_invalid_inputs() -> None:
    """Should reject occupancy values outside (0, 1)."""
    with pytest.raises(ValueError):
        compute_presence_pos_weight(0.0)
    with pytest.raises(ValueError):
        compute_presence_pos_weight(1.0)
    with pytest.raises(ValueError):
        compute_presence_pos_weight(-0.1)
    with pytest.raises(ValueError):
        compute_presence_pos_weight(1.5)


def test_balanced_bce_equals_standard_when_weight_one() -> None:
    """With pos_weight=1 (balanced data), should match standard BCE."""
    torch.manual_seed(42)
    logits = torch.randn(100, 12) * 0.5
    targets = torch.bernoulli(torch.ones(100, 12) * 0.5)

    loss_balanced = presence_bce_balanced(logits, targets, pos_weight=1.0)
    loss_standard = F.binary_cross_entropy_with_logits(logits, targets)

    assert torch.allclose(loss_balanced, loss_standard, atol=1e-5)


def test_balanced_bce_penalizes_false_negatives() -> None:
    """With pos_weight=3, false negatives cost more than false positives."""
    logits_zero = torch.zeros(100, 12)
    targets = torch.zeros(100, 12)
    targets[:, :6] = 1.0  # 50% positive

    loss_std = F.binary_cross_entropy_with_logits(logits_zero, targets)
    loss_bal = presence_bce_balanced(logits_zero, targets, pos_weight=3.0)

    assert loss_bal > loss_std, (
        f"Balanced ({loss_bal:.4f}) should exceed standard ({loss_std:.4f})"
        " when false-negative rate is high"
    )


def test_default_pos_weight_matches_training_occupancy() -> None:
    """PRESENCE_POS_WEIGHT constant should match occupancy of 0.244."""
    expected = compute_presence_pos_weight(0.244)
    assert abs(PRESENCE_POS_WEIGHT - expected) < 0.05


def test_balanced_bce_no_nan_or_inf() -> None:
    """Should produce finite loss under extreme imbalance."""
    logits = torch.randn(100, 12)
    targets = torch.zeros(100, 12)
    targets[:, :1] = 1.0  # ~8% positive

    loss = presence_bce_balanced(logits, targets, pos_weight=10.0)

    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() > 0


def test_balanced_bce_gradient_flows() -> None:
    """Gradient should flow through the balanced BCE loss."""
    logits = torch.randn(8, 12, requires_grad=True)
    targets = torch.bernoulli(torch.ones(8, 12) * 0.244)

    loss = presence_bce_balanced(logits, targets)
    loss.backward()

    assert logits.grad is not None
    assert not torch.isnan(logits.grad).any()
