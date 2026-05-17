"""Analytical pairwise interpenetration energy for Universal Guidance.

Implements the analytical energy function E(x_0_hat) used in the
Universal Guidance framework (Bansal et al., ICML 2023).

For each pair of present objects (i, j) in a scene:
  - Compute 3D Euclidean distance between their positions.
  - Compute combined bounding sphere radii (r_i + r_j).
  - Penetration depth = max(0, r_i + r_j - distance).
  - Energy += penetration_depth^2 (differentiable proxy for collision).

The squared penalty makes the gradient zero at non-overlapping pairs and
grows smoothly with increasing penetration, providing a strong push away
from collision states without discontinuities.

Counts each pair once (lower triangle), consistent with the symmetry of
the collision metric.
"""
from __future__ import annotations

import torch
from torch import Tensor

from .object_geometry import get_radii_tensor


def pairwise_overlap_energy(
    x_0_hat: Tensor,
    type_ids: Tensor,
    presence_threshold: float = 0.0,
) -> Tensor:
    """Compute total pairwise interpenetration energy per scene.

    Args:
        x_0_hat: Clean scene estimate. Shape: (B, N, 13).
            Channel layout: xyz(0-2), rot6d(3-8), scale(9-11), presence_logit(12).
            Only the xyz channels (0-2) are used for distance computation.
        type_ids: Object type IDs. Shape: (B, N), long, values in [0, 12].
            Type 12 (PAD) is automatically excluded via presence masking
            AND zero-radius assignment in object_geometry.
        presence_threshold: Logit threshold for presence channel (index 12).
            Objects with x_0_hat[..., 12] <= threshold are excluded.
            Default 0.0 matches the probe_vlm.py decode convention.

    Returns:
        energy: Float tensor of shape (B,). Lower energy = fewer collisions.
            Fully differentiable w.r.t. x_0_hat.

    Notes:
        - Self-pairs (i == i) are excluded via the eye mask.
        - Each pair (i, j) is counted once: the sum is divided by 2.
        - PAD slots have radius=0 in get_radii_tensor and are also excluded
          by the presence mask when their logit <= threshold.
        - The function is differentiable everywhere except at exact penetration
          depth = 0 (where the clamped gradient is zero anyway).
    """
    B, N, _D = x_0_hat.shape
    device = x_0_hat.device

    # --- Positions (xyz) ---
    positions = x_0_hat[..., :3]  # (B, N, 3), differentiable

    # --- Presence mask ---
    # presence_logit is channel 12; objects above threshold are "present".
    presence = x_0_hat[..., 12] > presence_threshold  # (B, N), bool

    # --- Bounding radii ---
    radii = get_radii_tensor(type_ids)  # (B, N), float32

    # --- Pairwise distances ---
    # diff: (B, N, N, 3); distances: (B, N, N)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    dist = diff.norm(dim=-1) + 1e-8  # avoid sqrt(0) gradient singularity

    # --- Sum of radii ---
    sum_radii = radii.unsqueeze(2) + radii.unsqueeze(1)  # (B, N, N)

    # --- Penetration depth (clamped) ---
    penetration = torch.clamp(sum_radii - dist, min=0.0)  # (B, N, N)

    # --- Pair mask: both objects present, not self-pair ---
    pair_mask = presence.unsqueeze(2) & presence.unsqueeze(1)  # (B, N, N)
    eye = torch.eye(N, device=device, dtype=torch.bool)
    pair_mask = pair_mask & ~eye.unsqueeze(0)  # exclude diagonal

    # --- Energy: sum of squared penetrations, counting each pair once ---
    energy_full = (penetration.pow(2) * pair_mask.float()).sum(dim=(1, 2))
    return energy_full / 2.0  # (B,), pairs counted twice above


