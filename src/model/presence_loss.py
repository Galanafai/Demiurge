"""Presence loss with class balancing.

v5 had presence collapse because BCE without pos_weight prefers predicting
'absent' (the majority class at ~75% of slots).

Mathematical justification:
  Training data occupancy P ~ 0.244 (2.93 objects/scene / 12 slots)
  Without pos_weight, BCE gradient is dominated by negative class (1-P).
  pos_weight = (1 - P) / P balances the gradients.
  At P=0.244: pos_weight ~ 3.10
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Computed from training data: 234,720 scenes, ~2.93 objects/scene, 12 slots.
# occupancy = 2.93 / 12 = 0.244
# pos_weight = (1 - 0.244) / 0.244 = 3.10
PRESENCE_POS_WEIGHT: float = 5.0


def compute_presence_pos_weight(occupancy: float) -> float:
    """Compute pos_weight from training data occupancy.

    Args:
        occupancy: fraction of slots that are occupied in training data.

    Returns:
        pos_weight for BCE.

    Raises:
        ValueError: if occupancy is not in (0, 1).
    """
    if occupancy <= 0 or occupancy >= 1:
        raise ValueError(f"occupancy must be in (0, 1), got {occupancy}")
    return (1.0 - occupancy) / occupancy


def presence_bce_balanced(
    pres_logits: torch.Tensor,
    pres_targets: torch.Tensor,
    pos_weight: float = PRESENCE_POS_WEIGHT,
) -> torch.Tensor:
    """BCE loss with positive class weighting.

    Args:
        pres_logits: model output for presence, shape (B, N_MAX).
        pres_targets: binary targets, shape (B, N_MAX), float in {0, 1}.
        pos_weight: weight for the positive class.

    Returns:
        Scalar BCE loss.
    """
    weight = torch.tensor([pos_weight], device=pres_logits.device, dtype=pres_logits.dtype)
    return F.binary_cross_entropy_with_logits(
        pres_logits,
        pres_targets,
        pos_weight=weight,
    )
