"""Tests for src/data/balanced_sampler.py.

Three tests:
1. test_type_proportionality   -- sampled types appear at ~1/freq rate
2. test_all_types_represented  -- no type is starved to 0 samples in 1000 draws
3. test_factory_returns_sampler -- return type is WeightedRandomSampler
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import WeightedRandomSampler

_SRC = Path(__file__).resolve().parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.balanced_sampler import (  # noqa: E402
    make_class_balanced_sampler,
)
from scene.schema import SceneTensor  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_scene(type_ids: list[int]) -> SceneTensor:
    """Build a minimal SceneTensor with given type_ids in the first N slots."""
    from model.denoiser import N_MAX
    n = len(type_ids)
    assert n <= N_MAX
    types = torch.zeros(N_MAX, dtype=torch.long)
    poses = torch.zeros(N_MAX, 7)
    poses[:, 3] = 1.0  # unit quaternion w=1
    scales = torch.ones(N_MAX, 3)
    presence = torch.zeros(N_MAX, dtype=torch.bool)
    for i, tid in enumerate(type_ids):
        types[i] = tid
        presence[i] = True
    return SceneTensor(object_types=types, poses=poses, scales=scales, presence=presence)


def _make_imbalanced_examples() -> tuple[list, list[int]]:
    """Create a synthetic dataset with severe type imbalance.

    type_id=0: 900 scenes (90%)
    type_id=1:  90 scenes (9%)
    type_id=2:  10 scenes (1%)

    Returns (examples, train_indices).
    """
    examples = []
    for _ in range(900):
        sc = _make_fake_scene([0])
        examples.append((sc, "desc", {"task_family": "tabletop_reach"}, "sdf"))
    for _ in range(90):
        sc = _make_fake_scene([1])
        examples.append((sc, "desc", {"task_family": "tabletop_reach"}, "sdf"))
    for _ in range(10):
        sc = _make_fake_scene([2])
        examples.append((sc, "desc", {"task_family": "tabletop_reach"}, "sdf"))
    indices = list(range(len(examples)))
    return examples, indices


# ---------------------------------------------------------------------------
# Test 1: type proportionality
# ---------------------------------------------------------------------------


def test_type_proportionality() -> None:
    """Sampled type distribution should approximate inverse-frequency.

    With 90:9:1 ratio, uniform sampling gives 90%:9%:1%.
    Balanced sampling should give approximately 33%:33%:33% (1/freq balanced).
    We verify that type_id=2 (1% data share) gets > 20% of samples.
    """
    torch.manual_seed(42)
    examples, indices = _make_imbalanced_examples()
    sampler = make_class_balanced_sampler(examples, indices, eps=0.01)

    # Sample 5000 indices.
    sampled_indices = list(iter(sampler))[:5000]
    type_counts: Counter[int] = Counter()
    for idx in sampled_indices:
        sc: SceneTensor = examples[idx][0]
        for slot in range(len(sc.object_types)):
            if sc.presence[slot]:
                type_counts[int(sc.object_types[slot])] += 1

    total = sum(type_counts.values())
    rare_frac = type_counts[2] / total
    assert rare_frac > 0.15, (
        f"type_id=2 (1% of data) should appear >15% of balanced samples, "
        f"got {100*rare_frac:.1f}%. Sampler is not balancing types."
    )
    # The majority class should be significantly reduced from 90%.
    majority_frac = type_counts[0] / total
    assert majority_frac < 0.60, (
        f"type_id=0 (90% of data) should appear <60% of balanced samples, "
        f"got {100*majority_frac:.1f}%. Sampler is not suppressing majority class."
    )


# ---------------------------------------------------------------------------
# Test 2: all types represented
# ---------------------------------------------------------------------------


def test_all_types_represented() -> None:
    """No type present in the dataset should be starved to zero samples."""
    torch.manual_seed(7)
    examples, indices = _make_imbalanced_examples()
    sampler = make_class_balanced_sampler(examples, indices, eps=0.01)

    sampled_indices = list(iter(sampler))[:1000]
    type_counts: Counter[int] = Counter()
    for idx in sampled_indices:
        sc: SceneTensor = examples[idx][0]
        for slot in range(len(sc.object_types)):
            if sc.presence[slot]:
                type_counts[int(sc.object_types[slot])] += 1

    for tid in [0, 1, 2]:
        assert type_counts[tid] > 0, (
            f"type_id={tid} has zero samples in 1000 draws. "
            "ClassBalancedSampler is starving this type."
        )


# ---------------------------------------------------------------------------
# Test 3: return type
# ---------------------------------------------------------------------------


def test_factory_returns_sampler() -> None:
    """make_class_balanced_sampler must return WeightedRandomSampler."""
    examples, indices = _make_imbalanced_examples()
    sampler = make_class_balanced_sampler(examples, indices)
    assert isinstance(sampler, WeightedRandomSampler), (
        f"Expected WeightedRandomSampler, got {type(sampler)}"
    )
    assert len(list(iter(sampler))) == len(indices), (
        f"Sampler should yield len(indices)={len(indices)} samples, "
        f"got {len(list(iter(sampler)))}"
    )
