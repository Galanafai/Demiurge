"""Tests for output sanity checks.

These verify that we can DETECT v5's failure modes before they destroy
a training run. If these tests cannot detect explosion/collapse, the
sanity check itself is broken.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import pytest

from training.output_sanity import (
    OutputSanityReport,
    assert_sane_output,
    check_output_sanity,
)


def test_detect_explosion() -> None:
    """Should detect output with std > 2.0 as explosion (v5's exact failure)."""
    torch.manual_seed(42)
    bad_output = torch.randn(8, 12, 13) * 5.0
    bad_output = bad_output.clamp(-9.98, 9.98)

    report = check_output_sanity(bad_output)

    assert report.inferred_explosion, \
        f"Should detect explosion, got std={report.overall_std:.2f}"
    assert report.saturation_rate > 0.05, "High saturation expected with std=5"


def test_detect_collapse() -> None:
    """Should detect output with std < 0.1 as collapse."""
    bad_output = torch.randn(8, 12, 13) * 0.01

    report = check_output_sanity(bad_output)

    assert report.inferred_collapse, \
        f"Should detect collapse, got std={report.overall_std:.4f}"


def test_healthy_output_not_flagged() -> None:
    """Should NOT flag a healthy whitened output (std~1, mean~0)."""
    torch.manual_seed(42)
    good_output = torch.randn(8, 12, 13)

    report = check_output_sanity(good_output)

    assert not report.inferred_explosion
    assert not report.inferred_collapse
    assert report.saturation_rate < 0.05


def test_assert_sane_raises_on_explosion() -> None:
    """assert_sane_output should raise with 'explosion' on exploded output."""
    bad_output = torch.randn(8, 12, 13) * 5.0

    with pytest.raises(AssertionError, match="explosion"):
        assert_sane_output(bad_output, step=12345)


def test_assert_sane_raises_on_collapse() -> None:
    """assert_sane_output should raise with 'collapse' on collapsed output."""
    bad_output = torch.randn(8, 12, 13) * 0.01

    with pytest.raises(AssertionError, match="collapse"):
        assert_sane_output(bad_output, step=12345)


def test_assert_sane_passes_on_healthy() -> None:
    """assert_sane_output should not raise on healthy unit-normal output."""
    good_output = torch.randn(8, 12, 13)

    assert_sane_output(good_output, step=0)  # must not raise


def test_presence_active_rate_50pct() -> None:
    """Should correctly report 50% presence active rate."""
    output = torch.zeros(100, 12, 13)
    # First 6 slots: above threshold; last 6: below
    output[:, :6, 12] = 2.0
    output[:, 6:, 12] = -2.0

    report = check_output_sanity(output, presence_threshold=-0.589)

    assert abs(report.presence_active_rate - 0.5) < 0.01


def test_simulates_v5_failure_exactly() -> None:
    """Reproduce v5's exact output distribution and verify we catch it."""
    # v5 step-30k stats: std=5.32, range~+-10, 27.6% presence active
    torch.manual_seed(42)
    output = torch.randn(8, 12, 13) * 5.32
    output = output.clamp(-9.98, 9.98)
    # Force presence dimension to match v5's 27.6% active
    output[:, :, 12] = torch.where(
        torch.rand(8, 12) < 0.276,
        torch.ones(8, 12),
        torch.full((8, 12), -2.0),
    )

    report = check_output_sanity(output)

    assert report.inferred_explosion, "Must detect v5's output explosion"
    assert report.saturation_rate > 0.05, "Must detect v5's clamp saturation"
