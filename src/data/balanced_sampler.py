"""Class-balanced sampler for the Demiurge training dataset.

Addresses type_id majority-class collapse by weighting each training
example proportional to the mean inverse-frequency of the object types
present in that scene.

This is distinct from the existing task-template-balanced sampler:
- Template sampler: ensures equal representation of tabletop_reach /
  cluttered_pick / obstacle_avoidance templates.
- Class-balanced sampler: ensures rare object types (e.g. mustard_bottle,
  tomato_soup_can) appear as often per batch as the majority type (cube).

Both can coexist; this sampler replaces the template sampler when
``use_class_balanced_sampler: true`` is set in the training config.

Weight formula per scene:
    w_i = mean over present slots j of (1 / (freq[type_id_j] + eps))

where freq[t] = count of type t / total present-slot observations across
the full training split.  Using the mean (not max) avoids over-weighting
multi-object scenes that happen to contain one rare type.
"""
from __future__ import annotations

from collections import Counter
from typing import Any

from torch.utils.data import WeightedRandomSampler

from scene.schema import SceneTensor


def _compute_type_frequencies(
    examples: list[tuple[Any, ...]],
    indices: list[int],
) -> dict[int, float]:
    """Compute type_id frequency over the training split.

    Args:
        examples: Full list of (SceneTensor, desc, report, sdf) tuples.
        indices: Indices into examples that form the training split
            (held-out excluded).

    Returns:
        Mapping from type_id (int) to frequency in [0, 1].
    """
    counts: Counter[int] = Counter()
    for i in indices:
        scene: SceneTensor = examples[i][0]
        for slot_idx in range(len(scene.object_types)):
            if scene.presence[slot_idx]:
                counts[int(scene.object_types[slot_idx])] += 1
    total = sum(counts.values())
    if total == 0:
        raise RuntimeError(
            "ClassBalancedSampler: zero present-slot observations in training split. "
            "Check that the dataset is non-empty and SceneTensor.presence is set correctly."
        )
    return {tid: c / total for tid, c in counts.items()}


def _scene_weight(
    scene: SceneTensor,
    freq: dict[int, float],
    eps: float,
) -> float:
    """Compute per-scene sampling weight.

    Weight = mean of (1 / (freq[type_id] + eps)) over present slots.
    Scenes with all slots absent (shouldn't occur in training data but
    handled defensively) receive weight = eps / (1 + eps) ~ eps.

    Args:
        scene: SceneTensor for this example.
        freq: Type-id frequency mapping from _compute_type_frequencies.
        eps: Smoothing term to prevent division by zero for unseen types.

    Returns:
        Scalar float weight >= 0.
    """
    slot_weights: list[float] = []
    for slot_idx in range(len(scene.object_types)):
        if scene.presence[slot_idx]:
            tid = int(scene.object_types[slot_idx])
            f = freq.get(tid, 0.0)
            slot_weights.append(1.0 / (f + eps))
    if not slot_weights:
        return eps  # empty scene: minimal weight, should not appear in training
    return sum(slot_weights) / len(slot_weights)


def make_class_balanced_sampler(
    examples: list[tuple[Any, ...]],
    indices: list[int],
    eps: float = 0.01,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler balanced by object type distribution.

    Args:
        examples: Full (SceneTensor, desc, report, sdf) list.
        indices: Training-split indices into examples.
        eps: Smoothing epsilon for inverse-frequency computation.
            Default 0.01 limits maximum weight to 100x minimum.

    Returns:
        WeightedRandomSampler with replacement, sampling len(indices)
        elements per epoch.

    Raises:
        RuntimeError: If training split is empty or has no present slots.
        ValueError: If eps <= 0.
    """
    if eps <= 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if not indices:
        raise RuntimeError("make_class_balanced_sampler: training indices list is empty.")

    freq = _compute_type_frequencies(examples, indices)
    weights: list[float] = []
    for i in indices:
        scene: SceneTensor = examples[i][0]
        weights.append(_scene_weight(scene, freq, eps))

    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(indices),
        replacement=True,
        generator=None,  # seeded externally via torch.manual_seed in train.py
    )
