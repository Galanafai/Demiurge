"""Procedural scene sampler stub for Demiurge.

Generates random SceneTensors within the workspace bounds. Each scene has a
random number of objects (1 to N_MAX), random types, positions, and orientations.
Positions are sampled uniformly within bounds, with a minimum separation
constraint to reduce trivially-interpenetrating scenes.

This is a Week 1 stub. The full sampler (Week 2+) will use rejection sampling
based on the validator to guarantee physical validity.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
from scene.vocab import ObjectTypeId


def sample_scene(
    rng: np.random.Generator,
    bounds: WorkspaceBounds | None = None,
    min_objects: int = 1,
    max_objects: int = N_MAX,
    min_separation_m: float = 0.06,
) -> SceneTensor:
    """Sample a single random SceneTensor in physical space.

    Positions are sampled by rejection: if a new object is placed within
    min_separation_m of any existing object, resample (up to 50 attempts).
    If placement fails, the scene will have fewer than n_objects present.

    All objects rest on the table (z = half-height of the primitive).

    Args:
        rng: Numpy random generator for reproducibility.
        bounds: Workspace bounds. Defaults to WorkspaceBounds.default().
        min_objects: Minimum number of objects to place.
        max_objects: Maximum number of objects to place.
        min_separation_m: Minimum centre-to-centre distance for placement.

    Returns:
        Physical-space SceneTensor with unit quaternions and identity scales.
    """
    if bounds is None:
        bounds = WorkspaceBounds.default()

    n_objects = int(rng.integers(min_objects, max_objects + 1))
    type_ids = rng.integers(0, len(ObjectTypeId), size=n_objects)

    # Pre-compute half-heights for z-placement.
    from scene.vocab import OBJECT_VOCAB

    half_heights = [
        OBJECT_VOCAB[int(tid)].canonical_half_extents_m[2] for tid in type_ids
    ]

    # Sample positions with separation constraint.
    placed_xy: list[np.ndarray] = []
    xyzs: list[np.ndarray] = []
    valid_count = 0

    for i in range(n_objects):
        hz = half_heights[i]
        placed = False
        for _ in range(50):
            x = float(rng.uniform(bounds.x_min + 0.05, bounds.x_max - 0.05))
            y = float(rng.uniform(bounds.y_min + 0.05, bounds.y_max - 0.05))
            xy = np.array([x, y])
            # Check separation from all placed objects.
            too_close = False
            for prev_xy in placed_xy:
                if np.linalg.norm(xy - prev_xy) < min_separation_m:
                    too_close = True
                    break
            if not too_close:
                placed_xy.append(xy)
                xyzs.append(np.array([x, y, hz]))
                placed = True
                valid_count += 1
                break
        if not placed:
            break

    if valid_count == 0:
        # Fallback: single object at workspace centre.
        tid = int(rng.integers(0, len(ObjectTypeId)))
        hz = OBJECT_VOCAB[tid].canonical_half_extents_m[2]
        xyzs = [np.array([0.0, 0.0, hz])]
        type_ids = np.array([tid])
        valid_count = 1

    # Build tensors.
    ot = torch.zeros(N_MAX, dtype=torch.int64)
    poses = torch.zeros(N_MAX, 7, dtype=torch.float32)
    scales = torch.ones(N_MAX, 3, dtype=torch.float32)
    pres = torch.zeros(N_MAX, dtype=torch.bool)

    for i in range(valid_count):
        ot[i] = int(type_ids[i])
        poses[i, 0] = xyzs[i][0]
        poses[i, 1] = xyzs[i][1]
        poses[i, 2] = xyzs[i][2]

        # Random yaw rotation (objects rest upright on table, only yaw varies).
        yaw = float(rng.uniform(0, 2 * math.pi))
        # wxyz quaternion for pure yaw rotation.
        poses[i, 3] = math.cos(yaw / 2)
        poses[i, 4] = 0.0
        poses[i, 5] = 0.0
        poses[i, 6] = math.sin(yaw / 2)

        pres[i] = True

    # Inactive slots get identity quaternion.
    for i in range(valid_count, N_MAX):
        poses[i, 3] = 1.0

    return SceneTensor(object_types=ot, poses=poses, scales=scales, presence=pres)


def sample_batch(
    n_scenes: int,
    seed: int = 0,
    bounds: WorkspaceBounds | None = None,
    min_objects: int = 1,
    max_objects: int = N_MAX,
) -> list[SceneTensor]:
    """Sample a batch of random scenes.

    Each scene gets a unique sub-seed derived from the base seed for
    reproducibility.

    Args:
        n_scenes: Number of scenes to generate.
        seed: Base seed for the generator.
        bounds: Workspace bounds.
        min_objects: Minimum objects per scene.
        max_objects: Maximum objects per scene.

    Returns:
        List of SceneTensor in physical space.
    """
    rng = np.random.default_rng(seed)
    return [
        sample_scene(rng, bounds=bounds, min_objects=min_objects, max_objects=max_objects)
        for _ in range(n_scenes)
    ]
