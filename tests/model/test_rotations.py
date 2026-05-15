"""Tests for src/model/rotations.py.

Four tests:
1. test_round_trip_random       -- quat -> 6D -> matrix -> quat round-trip
2. test_projection_arbitrary    -- arbitrary 6D vector -> valid SO(3) matrix
3. test_gradient_flow           -- gradients exist through rot6d_to_quat_wxyz
4. test_gram_schmidt_degenerate -- near-parallel/zero inputs produce no NaN
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from model.rotations import (  # noqa: E402
    matrix_to_quat_wxyz,
    quat_wxyz_to_6d,
    rot6d_to_matrix,
    rot6d_to_quat_wxyz,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def random_unit_quats(n: int, seed: int = 0) -> torch.Tensor:
    """Generate n random unit quaternions in wxyz order."""
    torch.manual_seed(seed)
    q = torch.randn(n, 4)
    return torch.nn.functional.normalize(q, p=2, dim=-1)


def quats_equivalent(q1: torch.Tensor, q2: torch.Tensor, tol: float = 1e-4) -> bool:
    """Return True if q1 and q2 represent the same rotation (up to antipodal sign)."""
    dot = (q1 * q2).sum(dim=-1).abs()
    return bool((dot > 1.0 - tol).all().item())


# ---------------------------------------------------------------------------
# Test 1: round-trip quat -> 6D -> matrix -> quat
# ---------------------------------------------------------------------------


def test_round_trip_random() -> None:
    """Quaternion round-trip via 6D representation must recover original rotation."""
    q_orig = random_unit_quats(256, seed=42)
    r6d = quat_wxyz_to_6d(q_orig)           # (..., 6)
    R = rot6d_to_matrix(r6d)                 # (..., 3, 3)
    q_recovered = matrix_to_quat_wxyz(R)     # (..., 4)

    assert quats_equivalent(q_orig, q_recovered, tol=1e-4), (
        "Quaternion round-trip quat -> 6D -> matrix -> quat failed. "
        "Max absolute dot-product deviation from 1.0: "
        f"{(1.0 - (q_orig * q_recovered).sum(-1).abs()).max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 2: arbitrary 6D -> valid SO(3) matrix
# ---------------------------------------------------------------------------


def test_projection_arbitrary() -> None:
    """rot6d_to_matrix must project any 6D input to a valid SO(3) rotation.

    Validity checks:
    - det(R) = 1 (right-handed, not a reflection)
    - R^T R = I (orthonormal columns)
    """
    torch.manual_seed(7)
    r_arb = torch.randn(128, 6)   # arbitrary, not on the SO(3) manifold
    R = rot6d_to_matrix(r_arb)    # (128, 3, 3)

    # Determinant should be +1.
    dets = torch.linalg.det(R)    # (128,)
    assert (dets - 1.0).abs().max().item() < 1e-5, (
        f"rot6d_to_matrix produced det != 1. Max deviation: "
        f"{(dets - 1.0).abs().max().item():.6f}"
    )

    # Orthonormality: R^T R should be identity.
    eye = torch.eye(3).expand(128, 3, 3)
    orth_err = (R.transpose(-1, -2) @ R - eye).abs().max().item()
    assert orth_err < 1e-5, (
        f"rot6d_to_matrix produced non-orthogonal matrix. "
        f"Max R^T R - I deviation: {orth_err:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 3: gradient flow through rot6d_to_quat_wxyz
# ---------------------------------------------------------------------------


def test_gradient_flow() -> None:
    """Gradients must exist through rot6d_to_quat_wxyz for random inputs.

    This validates that the Gram-Schmidt projection and matrix-to-quat
    conversion are differentiable in the autograd graph for typical inputs.
    """
    torch.manual_seed(3)
    r = torch.randn(32, 6, requires_grad=True)
    q = rot6d_to_quat_wxyz(r)   # (32, 4)

    loss = q.sum()
    loss.backward()

    assert r.grad is not None, "No gradient flowed to 6D input"
    assert not r.grad.isnan().any(), "NaN gradients in 6D input"
    assert r.grad.abs().max().item() > 0, "All-zero gradients (degenerate case)"


# ---------------------------------------------------------------------------
# Test 4: Gram-Schmidt stability on degenerate inputs
# ---------------------------------------------------------------------------


def test_gram_schmidt_degenerate() -> None:
    """Near-zero or near-parallel input columns must not produce NaN.

    Tests three degenerate cases:
    a. Near-zero first column (close to origin).
    b. Near-zero second column.
    c. Nearly parallel columns (a2 ~ a1).
    """
    # a. Near-zero first column.
    r_a = torch.zeros(1, 6)
    r_a[0, :3] = 1e-9
    r_a[0, 3:] = torch.tensor([0.0, 1.0, 0.0])
    R_a = rot6d_to_matrix(r_a)
    assert not R_a.isnan().any(), "NaN produced with near-zero first column"

    # b. Near-zero second column.
    r_b = torch.zeros(1, 6)
    r_b[0, :3] = torch.tensor([1.0, 0.0, 0.0])
    r_b[0, 3:] = 1e-9
    R_b = rot6d_to_matrix(r_b)
    assert not R_b.isnan().any(), "NaN produced with near-zero second column"

    # c. Nearly parallel columns (a2 ~ a1).
    # When inputs are rank-1, (a2 - dot*b1) collapses to near-zero and
    # F.normalize clips via eps, producing an arbitrary non-NaN b2.
    # The contract here is NaN-free and inf-free only; det=1 is not
    # achievable from genuinely parallel input (rank-1 pair has no
    # unique orthogonal direction).
    base = torch.tensor([1.0, 0.0, 0.0])
    r_c = torch.zeros(1, 6)
    r_c[0, :3] = base
    r_c[0, 3:] = base + torch.tensor([1e-7, 0.0, 0.0])
    R_c = rot6d_to_matrix(r_c)
    assert not R_c.isnan().any(), "NaN produced with nearly parallel columns"
    assert not R_c.isinf().any(), "Inf produced with nearly parallel columns"
