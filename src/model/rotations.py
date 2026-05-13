"""6D continuous rotation representation for diffusion model training.

Implements the representation from Zhou et al. (2019), "On the Continuity of
Rotation Representations in Neural Networks". The key insight is that the
standard quaternion and Euler angle representations have discontinuities in
their mapping from SO(3), which cause gradient issues for neural networks
trained on rotation prediction. The 6D representation (first two columns of
a rotation matrix) is continuous everywhere.

Convention used throughout this module:
  - Quaternions are stored in wxyz order (w first), matching Drake and the
    Demiurge schema (SceneTensor.poses[:, 3:7]).
  - 6D vectors are [r1x, r1y, r1z, r2x, r2y, r2z] where r1, r2 are the
    first two columns of the rotation matrix BEFORE Gram-Schmidt projection.
  - The model predicts 6D vectors; they are projected to SO(3) via
    Gram-Schmidt, then converted to quaternions for storage and Drake.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Quaternion <-> rotation matrix
# ---------------------------------------------------------------------------


def quat_wxyz_to_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion(s) in wxyz order to rotation matrix/matrices.

    Args:
        q: Tensor of shape (..., 4) with (w, x, y, z) ordering.
           Assumed to be unit quaternions; unnormalized input gives
           unnormalized rotation matrices.

    Returns:
        Rotation matrix tensor of shape (..., 3, 3).
    """
    w, x, y, z = q.unbind(dim=-1)

    # Precompute products (factor of 2 absorbed below).
    x2 = x * x
    y2 = y * y
    z2 = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = torch.stack([
        1 - 2*(y2 + z2), 2*(xy - wz),     2*(xz + wy),
        2*(xy + wz),     1 - 2*(x2 + z2), 2*(yz - wx),
        2*(xz - wy),     2*(yz + wx),     1 - 2*(x2 + y2),
    ], dim=-1)  # (..., 9)

    return R.unflatten(-1, (3, 3))  # (..., 3, 3)


def matrix_to_quat_wxyz(R: Tensor) -> Tensor:
    """Convert rotation matrix/matrices to unit quaternion(s) in wxyz order.

    Uses the Shepperd method (numerically stable branch selection).

    Args:
        R: Rotation matrix tensor of shape (..., 3, 3).

    Returns:
        Unit quaternion tensor of shape (..., 4) in wxyz order.
    """
    # Shepperd's method: pick the branch with the largest diagonal element
    # to maximise numerical stability.
    trace = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]  # (...)

    # Case 1: trace > 0 (w is largest component)
    s1 = (trace + 1.0).clamp(min=0.0).sqrt() * 2.0  # 4w
    w1 = 0.25 * s1
    x1 = (R[..., 2, 1] - R[..., 1, 2]) / s1.clamp(min=1e-8)
    y1 = (R[..., 0, 2] - R[..., 2, 0]) / s1.clamp(min=1e-8)
    z1 = (R[..., 1, 0] - R[..., 0, 1]) / s1.clamp(min=1e-8)

    # Case 2: R[0,0] is the largest diagonal element
    s2 = (1.0 + R[..., 0, 0] - R[..., 1, 1] - R[..., 2, 2]).clamp(min=0.0).sqrt() * 2.0
    w2 = (R[..., 2, 1] - R[..., 1, 2]) / s2.clamp(min=1e-8)
    x2 = 0.25 * s2
    y2 = (R[..., 0, 1] + R[..., 1, 0]) / s2.clamp(min=1e-8)
    z2 = (R[..., 0, 2] + R[..., 2, 0]) / s2.clamp(min=1e-8)

    # Case 3: R[1,1] is the largest diagonal element
    s3 = (1.0 + R[..., 1, 1] - R[..., 0, 0] - R[..., 2, 2]).clamp(min=0.0).sqrt() * 2.0
    w3 = (R[..., 0, 2] - R[..., 2, 0]) / s3.clamp(min=1e-8)
    x3 = (R[..., 0, 1] + R[..., 1, 0]) / s3.clamp(min=1e-8)
    y3 = 0.25 * s3
    z3 = (R[..., 2, 1] + R[..., 1, 2]) / s3.clamp(min=1e-8)

    # Case 4: R[2,2] is the largest diagonal element
    s4 = (1.0 + R[..., 2, 2] - R[..., 0, 0] - R[..., 1, 1]).clamp(min=0.0).sqrt() * 2.0
    w4 = (R[..., 1, 0] - R[..., 0, 1]) / s4.clamp(min=1e-8)
    x4 = (R[..., 0, 2] + R[..., 2, 0]) / s4.clamp(min=1e-8)
    y4 = (R[..., 2, 1] + R[..., 1, 2]) / s4.clamp(min=1e-8)
    z4 = 0.25 * s4

    # Select the numerically stable branch for each element.
    cond1 = trace > 0
    cond2 = (~cond1) & (R[..., 0, 0] > R[..., 1, 1]) & (R[..., 0, 0] > R[..., 2, 2])
    cond3 = (~cond1) & (~cond2) & (R[..., 1, 1] > R[..., 2, 2])

    w = torch.where(cond1, w1, torch.where(cond2, w2, torch.where(cond3, w3, w4)))
    x = torch.where(cond1, x1, torch.where(cond2, x2, torch.where(cond3, x3, x4)))
    y = torch.where(cond1, y1, torch.where(cond2, y2, torch.where(cond3, y3, y4)))
    z = torch.where(cond1, z1, torch.where(cond2, z2, torch.where(cond3, z3, z4)))

    q = torch.stack([w, x, y, z], dim=-1)
    return F.normalize(q, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Quaternion <-> 6D
# ---------------------------------------------------------------------------


def quat_wxyz_to_6d(q: Tensor) -> Tensor:
    """Convert unit quaternion(s) to 6D rotation representation.

    Converts to rotation matrix first, then extracts the first two columns,
    flattened to [r1x, r1y, r1z, r2x, r2y, r2z].

    Args:
        q: Unit quaternion tensor of shape (..., 4) in wxyz order.

    Returns:
        6D representation tensor of shape (..., 6).
    """
    R = quat_wxyz_to_matrix(q)           # (..., 3, 3)
    col1 = R[..., :, 0]                  # (..., 3) -- first column
    col2 = R[..., :, 1]                  # (..., 3) -- second column
    return torch.cat([col1, col2], dim=-1)  # (..., 6)


def rot6d_to_matrix(r: Tensor) -> Tensor:
    """Project a 6D vector to a valid rotation matrix via Gram-Schmidt.

    The first three elements are treated as an unnormalized first column;
    the next three as an unnormalized second column. The third column is
    the cross product. This mapping is continuous everywhere except when
    a1 and a2 are parallel (measure-zero event in practice).

    Args:
        r: 6D rotation tensor of shape (..., 6).

    Returns:
        Rotation matrix tensor of shape (..., 3, 3) with det=+1 and
        R^T R = I (up to floating-point precision).
    """
    a1 = r[..., :3]   # (..., 3)
    a2 = r[..., 3:]   # (..., 3)

    # Gram-Schmidt orthonormalization.
    b1 = F.normalize(a1, p=2, dim=-1, eps=1e-8)
    # Remove b1 component from a2.
    dot = (a2 * b1).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, p=2, dim=-1, eps=1e-8)
    # Third column via cross product (right-handed frame, det=+1).
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack columns -> rotation matrix.
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


def rot6d_to_quat_wxyz(r: Tensor) -> Tensor:
    """Project a 6D rotation vector to a unit quaternion in wxyz order.

    This is the composed operation used at DDIM inference time to convert
    the model's raw 6D output to a quaternion suitable for Drake and storage.

    Args:
        r: 6D rotation tensor of shape (..., 6).

    Returns:
        Unit quaternion tensor of shape (..., 4) in wxyz order.
    """
    R = rot6d_to_matrix(r)
    return matrix_to_quat_wxyz(R)
