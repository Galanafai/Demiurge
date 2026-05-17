"""Conservative bounding sphere radii per YCB object type.

These are deliberately larger than actual object extents to ensure no
false non-collisions. Drake validation uses tighter convex geometry, so
we overestimate here to push the gradient strongly away from collision.

Radii are the approximate half-diagonal of the object's bounding box,
chosen to ensure the sphere encloses the full YCB mesh geometry.
"""
from __future__ import annotations

import torch
from torch import Tensor

# Bounding sphere radii in metres, indexed by type_id 0-11.
# Type 12 is the PAD token -- assigned zero radius so PAD slots
# contribute nothing to pairwise energy.
BOUNDING_RADII: dict[int, float] = {
    0:  0.035,  # CUBE          -- ~7cm side, r=half-diagonal
    1:  0.040,  # SPHERE        -- ~7cm diameter
    2:  0.045,  # CYLINDER      -- ~9cm tall, 5cm diameter
    3:  0.065,  # BOX_TALL      -- cereal-box-sized, tallest object
    4:  0.055,  # BOX_FLAT      -- wide flat box
    5:  0.055,  # MUSTARD_BOTTLE -- YCB 019, ~19cm tall
    6:  0.060,  # SUGAR_BOX      -- YCB 004, ~17cm tall
    7:  0.045,  # TOMATO_SOUP_CAN -- YCB 005, ~10cm tall, 7cm dia
    8:  0.070,  # BLEACH_CLEANSER -- YCB 021, ~25cm tall (largest)
    9:  0.055,  # BANANA         -- YCB 011, ~21cm long
    10: 0.060,  # MASTER_CHEF_CAN -- YCB 002, ~14cm tall, 10cm dia
    11: 0.045,  # GELATIN_BOX    -- YCB 009, ~8cm tall
    12: 0.000,  # PAD            -- no geometry, never contributes
}

# Maximum radius across all non-PAD types (useful for sanity checks).
MAX_RADIUS: float = max(v for k, v in BOUNDING_RADII.items() if k != 12)

# Pre-built list in [0, 12] order for fast indexing.
_RADII_LIST: list[float] = [BOUNDING_RADII[i] for i in range(13)]


def get_radii_tensor(type_ids: Tensor) -> Tensor:
    """Map (B, N) type_ids to (B, N) bounding radii in metres.

    Args:
        type_ids: Long tensor of shape (B, N) with values in [0, 12].
            Values >= 13 are treated as PAD (radius 0).

    Returns:
        Float tensor of shape (B, N) with per-slot bounding sphere radii.
    """
    device = type_ids.device
    radii_lookup = torch.tensor(_RADII_LIST, device=device, dtype=torch.float32)
    # Clamp to [0, 12] so out-of-range values hit the PAD radius (index 12).
    clamped = type_ids.long().clamp(0, 12)
    return radii_lookup[clamped]
