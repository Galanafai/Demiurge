"""Runtime sanity checks on model output during sampling.

v5 failed because output exploded to std=5.32 (5x too large). This is
detectable immediately at sampling time, not 60k training steps later.

These checks should run at every probe/sampling call and in the training
validation loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class OutputSanityReport:
    """Report on model output distribution."""

    overall_mean: float
    overall_std: float
    per_dim_mean: list[float]
    per_dim_std: list[float]
    saturation_rate: float       # fraction of values within 5% of clamp boundary
    inferred_explosion: bool     # std > 2.0 indicates explosion
    inferred_collapse: bool      # std < 0.1 indicates collapse
    presence_active_rate: float  # fraction of slots above presence threshold

    def summary(self) -> str:
        lines = [
            "Output sanity:",
            f"  overall std: {self.overall_std:.4f} (expect ~1.0 for whitened space)",
            f"  saturation: {self.saturation_rate * 100:.1f}% at clamp boundary",
            f"  presence active: {self.presence_active_rate * 100:.1f}%",
        ]
        if self.inferred_explosion:
            lines.append("  WARNING: OUTPUT EXPLOSION (std > 2.0)")
        if self.inferred_collapse:
            lines.append("  WARNING: OUTPUT COLLAPSE (std < 0.1)")
        if self.saturation_rate > 0.1:
            lines.append(f"  WARNING: HIGH SATURATION ({self.saturation_rate * 100:.1f}%)")
        return "\n".join(lines)


def check_output_sanity(
    x_cont: torch.Tensor,
    presence_threshold: float = -0.589,
    clamp_value: float = 10.0,
) -> OutputSanityReport:
    """Run sanity checks on a batch of generated outputs.

    Args:
        x_cont: (B, N_MAX, N_CONT) generated continuous tensor.
            Presence logit is expected at dim index 12.
        presence_threshold: logit threshold above which a slot is active.
        clamp_value: magnitude at which values are considered saturated.

    Returns:
        OutputSanityReport with full diagnostics.
    """
    flat = x_cont.detach().float().flatten()
    overall_mean = flat.mean().item()
    overall_std = flat.std().item()

    n_dims = x_cont.shape[-1]
    per_dim_mean: list[float] = []
    per_dim_std: list[float] = []
    for d in range(n_dims):
        vals = x_cont[:, :, d].detach().float()
        per_dim_mean.append(vals.mean().item())
        per_dim_std.append(vals.std().item())

    saturation_threshold = clamp_value * 0.95
    saturated = (x_cont.detach().abs() > saturation_threshold).float().mean().item()

    pres_logits = x_cont[:, :, 12].detach().float()
    presence_active = (pres_logits > presence_threshold).float().mean().item()

    return OutputSanityReport(
        overall_mean=overall_mean,
        overall_std=overall_std,
        per_dim_mean=per_dim_mean,
        per_dim_std=per_dim_std,
        saturation_rate=saturated,
        inferred_explosion=overall_std > 2.0,
        inferred_collapse=overall_std < 0.1,
        presence_active_rate=presence_active,
    )


def assert_sane_output(x_cont: torch.Tensor, step: int) -> None:
    """Hard assertion that output is sane. Raises if not.

    Use this in test code and in training validation loops.

    Raises:
        AssertionError: if the output is exploded, collapsed, or saturated.
    """
    report = check_output_sanity(x_cont)

    if report.inferred_explosion:
        raise AssertionError(
            f"Output explosion at step {step}: std={report.overall_std:.2f} > 2.0\n"
            f"This is v5's failure mode. Check v-prediction convention.\n"
            f"{report.summary()}"
        )

    if report.inferred_collapse:
        raise AssertionError(
            f"Output collapse at step {step}: std={report.overall_std:.4f} < 0.1\n"
            f"Model is producing nearly-constant output.\n"
            f"{report.summary()}"
        )

    if report.saturation_rate > 0.3:
        raise AssertionError(
            f"High saturation at step {step}: {report.saturation_rate * 100:.1f}% at clamp\n"
            f"Model output is being clipped, indicating distribution mismatch.\n"
            f"{report.summary()}"
        )
