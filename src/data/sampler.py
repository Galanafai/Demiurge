"""Full procedural sampler for Demiurge dataset generation.

Implements three TaskTemplate classes covering the task families defined in
SKILL.md Layer 3. Each template samples a CandidateScene containing a
SceneTensor plus metadata needed for description generation and downstream
validation.

Templates:
  TabletopReachTemplate    -- 1-4 objects, goal above the target object.
  ClutteredPickTemplate    -- 3-6 objects in a tight cluster, goal above target.
  ObstacleAvoidanceTemplate -- 2-5 objects in a line across the workspace.

All templates are deterministic when given the same np.random.Generator.
Spatial parameters are constructor-injectable for Hydra config overrides.

This module is additive. sampler_stub.py (Week 1) is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import torch

from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
from scene.vocab import OBJECT_VOCAB

# ---------------------------------------------------------------------------
# CandidateScene
# ---------------------------------------------------------------------------


@dataclass
class CandidateScene:
    """A sampled scene plus task metadata.

    Attributes:
        scene: Physical-space SceneTensor (not yet normalized).
        task_family: Identifier for the generating template.
        target_idx: Index into scene.presence for the target object
            (0-indexed among ALL N_MAX slots, may be a padded slot --
            always check scene.presence[target_idx] is True).
        goal_xyz: IK goal position in world frame (3,).
    """

    scene: SceneTensor
    task_family: str
    target_idx: int
    goal_xyz: np.ndarray


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class TaskTemplate(Protocol):
    """Common interface for all task templates."""

    name: str

    def sample_scene(self, rng: np.random.Generator) -> CandidateScene:
        """Sample one candidate scene.

        Args:
            rng: Numpy random generator. Mutated in-place for reproducibility.

        Returns:
            CandidateScene in physical space.
        """
        ...

    def sample_task_description(
        self,
        scene: CandidateScene,
        rng: np.random.Generator,
    ) -> str:
        """Generate a natural-language task description from a scene.

        Args:
            scene: The candidate scene (from sample_scene).
            rng: Numpy random generator.

        Returns:
            Non-empty task description string.
        """
        ...


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# IK goal height above the target object centroid (must match validator constant).
_IK_GOAL_HEIGHT_M: float = 0.10

# Minimum distance between any two object centres during placement.
_DEFAULT_MIN_SEPARATION_M: float = 0.06

# Maximum rejection attempts per object placement before giving up.
_MAX_PLACEMENT_ATTEMPTS: int = 60


def _half_height(type_id: int, scale: float = 1.0) -> float:
    """Return the half-height (z-extent) for an object type at the given scale."""
    return OBJECT_VOCAB[type_id].canonical_half_extents_m[2] * scale


def _bounding_radius(type_id: int, scale: float = 1.0) -> float:
    """Return the bounding radius for an object type at the given scale."""
    return OBJECT_VOCAB[type_id].bounding_radius_m * scale


def _random_yaw_quat(rng: np.random.Generator) -> list[float]:
    """Return a wxyz quaternion for a random yaw rotation (object upright)."""
    yaw = float(rng.uniform(0.0, 2.0 * math.pi))
    return [math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]


def _build_scene_tensor(
    type_ids: list[int],
    xyzs: list[np.ndarray],
    scales: list[float],
    quats: list[list[float]],
) -> SceneTensor:
    """Pack lists into a SceneTensor. Inactive slots get identity quaternion."""
    n = len(type_ids)
    assert n <= N_MAX

    ot = torch.zeros(N_MAX, dtype=torch.int64)
    poses = torch.zeros(N_MAX, 7, dtype=torch.float32)
    sc = torch.ones(N_MAX, 3, dtype=torch.float32)
    pres = torch.zeros(N_MAX, dtype=torch.bool)

    for i in range(n):
        ot[i] = type_ids[i]
        poses[i, 0] = float(xyzs[i][0])
        poses[i, 1] = float(xyzs[i][1])
        poses[i, 2] = float(xyzs[i][2])
        q = quats[i]
        poses[i, 3] = q[0]
        poses[i, 4] = q[1]
        poses[i, 5] = q[2]
        poses[i, 6] = q[3]
        sc[i] = scales[i]
        pres[i] = True

    for i in range(n, N_MAX):
        poses[i, 3] = 1.0  # identity quaternion for inactive slots

    return SceneTensor(object_types=ot, poses=poses, scales=sc, presence=pres)


def _place_objects(
    rng: np.random.Generator,
    n_objects: int,
    type_ids: list[int],
    scales: list[float],
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    min_separation_m: float,
    margin: float = 0.05,
) -> tuple[list[np.ndarray], list[int]]:
    """Place n_objects with minimum separation constraint.

    Returns:
        (placed_xyzs, placed_indices) where placed_indices are the original
        indices of objects that were successfully placed. May be fewer than
        n_objects if the workspace is too crowded.
    """
    placed_xy: list[np.ndarray] = []
    placed_xyzs: list[np.ndarray] = []
    placed_indices: list[int] = []

    for i in range(n_objects):
        hz = _half_height(type_ids[i], scales[i])
        br = _bounding_radius(type_ids[i], scales[i])
        # Effective separation: use bounding radius so objects do not visually overlap.
        eff_sep = max(min_separation_m, br * 1.2)

        placed = False
        for _ in range(_MAX_PLACEMENT_ATTEMPTS):
            x = float(rng.uniform(x_range[0] + margin, x_range[1] - margin))
            y = float(rng.uniform(y_range[0] + margin, y_range[1] - margin))
            xy = np.array([x, y])
            if all(np.linalg.norm(xy - prev) >= eff_sep for prev in placed_xy):
                placed_xy.append(xy)
                placed_xyzs.append(np.array([x, y, hz]))
                placed_indices.append(i)
                placed = True
                break

        if not placed and i == 0:
            # Guarantee at least one object (workspace centre fallback).
            x = float(rng.uniform(x_range[0] + margin, x_range[1] - margin))
            y = float(rng.uniform(y_range[0] + margin, y_range[1] - margin))
            hz = _half_height(type_ids[i], scales[i])
            placed_xy.append(np.array([x, y]))
            placed_xyzs.append(np.array([x, y, hz]))
            placed_indices.append(i)

    return placed_xyzs, placed_indices


# ---------------------------------------------------------------------------
# Template 1: TabletopReach
# ---------------------------------------------------------------------------


class TabletopReachTemplate:
    """1-4 objects spread across the workspace. Goal above the target object.

    Designed to produce the highest acceptance rate of the three templates.
    Objects are well-separated, goal is directly above the target, and the
    workspace is uncluttered enough for BiRRT to find a path reliably.

    Parameters:
        min_objects: Minimum number of objects (default 1).
        max_objects: Maximum number of objects (default 4).
        min_separation_m: Minimum centre-to-centre separation (default 0.06m).
        bounds: Workspace bounds (default WorkspaceBounds.default()).
    """

    name: str = "tabletop_reach"

    def __init__(
        self,
        min_objects: int = 1,
        max_objects: int = 4,
        min_separation_m: float = _DEFAULT_MIN_SEPARATION_M,
        bounds: WorkspaceBounds | None = None,
    ) -> None:
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_separation_m = min_separation_m
        self.bounds = bounds or WorkspaceBounds.default()

    def sample_scene(self, rng: np.random.Generator) -> CandidateScene:
        n = int(rng.integers(self.min_objects, self.max_objects + 1))
        n_vocab = len(OBJECT_VOCAB)
        type_ids = [int(rng.integers(0, n_vocab)) for _ in range(n)]
        scales = [float(rng.uniform(0.8, 1.2)) for _ in range(n)]
        quats = [_random_yaw_quat(rng) for _ in range(n)]

        b = self.bounds
        xyzs, placed = _place_objects(
            rng, n, type_ids, scales,
            x_range=(b.x_min, b.x_max),
            y_range=(b.y_min, b.y_max),
            min_separation_m=self.min_separation_m,
        )

        # Resolve to placed objects only.
        type_ids = [type_ids[i] for i in placed]
        scales = [scales[i] for i in placed]
        quats = [quats[i] for i in placed]

        # Target: random present object.
        target_local = int(rng.integers(0, len(type_ids)))
        target_xyz = xyzs[target_local]
        goal_xyz = np.array([
            target_xyz[0],
            target_xyz[1],
            target_xyz[2] + _IK_GOAL_HEIGHT_M,
        ])

        scene = _build_scene_tensor(type_ids, xyzs, scales, quats)
        return CandidateScene(
            scene=scene,
            task_family=self.name,
            target_idx=target_local,
            goal_xyz=goal_xyz,
        )

    def sample_task_description(
        self,
        scene: CandidateScene,
        rng: np.random.Generator,
    ) -> str:
        from data.descriptions import generate_description
        return generate_description(scene, rng)


# ---------------------------------------------------------------------------
# Template 2: ClutteredPick
# ---------------------------------------------------------------------------


class ClutteredPickTemplate:
    """3-6 objects in a tight spatial cluster. Goal above the target.

    Objects are placed in a narrower sub-region of the workspace, creating
    a cluttered scene where IK and RRT must navigate around neighbours.
    This template produces the lowest acceptance rate of the three.

    Parameters:
        min_objects: Minimum number of objects (default 3).
        max_objects: Maximum number of objects (default 6).
        min_separation_m: Minimum centre-to-centre separation (default 0.08m).
            Slightly larger than tabletop_reach to ensure physical validity
            despite objects being in a smaller region.
        cluster_x: (min, max) x-range of the cluster (default (-0.20, 0.20)).
        cluster_y: (min, max) y-range of the cluster (default (0.15, 0.40)).
    """

    name: str = "cluttered_pick"

    def __init__(
        self,
        min_objects: int = 3,
        max_objects: int = 6,
        min_separation_m: float = 0.08,
        cluster_x: tuple[float, float] = (-0.20, 0.20),
        cluster_y: tuple[float, float] = (0.15, 0.40),
    ) -> None:
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.min_separation_m = min_separation_m
        self.cluster_x = cluster_x
        self.cluster_y = cluster_y

    def sample_scene(self, rng: np.random.Generator) -> CandidateScene:
        n = int(rng.integers(self.min_objects, self.max_objects + 1))
        n_vocab = len(OBJECT_VOCAB)
        type_ids = [int(rng.integers(0, n_vocab)) for _ in range(n)]
        scales = [float(rng.uniform(0.8, 1.1)) for _ in range(n)]
        quats = [_random_yaw_quat(rng) for _ in range(n)]

        xyzs, placed = _place_objects(
            rng, n, type_ids, scales,
            x_range=self.cluster_x,
            y_range=self.cluster_y,
            min_separation_m=self.min_separation_m,
            margin=0.04,
        )

        type_ids = [type_ids[i] for i in placed]
        scales = [scales[i] for i in placed]
        quats = [quats[i] for i in placed]

        # Target: random present object.
        target_local = int(rng.integers(0, len(type_ids)))
        target_xyz = xyzs[target_local]
        goal_xyz = np.array([
            target_xyz[0],
            target_xyz[1],
            target_xyz[2] + _IK_GOAL_HEIGHT_M,
        ])

        scene = _build_scene_tensor(type_ids, xyzs, scales, quats)
        return CandidateScene(
            scene=scene,
            task_family=self.name,
            target_idx=target_local,
            goal_xyz=goal_xyz,
        )

    def sample_task_description(
        self,
        scene: CandidateScene,
        rng: np.random.Generator,
    ) -> str:
        from data.descriptions import generate_description
        return generate_description(scene, rng)


# ---------------------------------------------------------------------------
# Template 3: ObstacleAvoidance
# ---------------------------------------------------------------------------


class ObstacleAvoidanceTemplate:
    """2-5 objects in a loose perpendicular line. Target on the far side.

    Objects are arranged roughly perpendicular to the y-axis, creating a
    partial barrier. The target is placed behind the barrier relative to
    the robot base (larger y). The arm must plan around the line of
    objects to reach the goal, exercising BiRRT more than tabletop_reach.

    Parameters:
        min_objects: Minimum number of objects (default 2).
        max_objects: Maximum number of objects (default 5).
        barrier_y: y-coordinate of the obstacle line (default 0.25m).
        target_y_range: (min, max) y-range for target placement behind the
            barrier (default (0.35, 0.45)).
        min_separation_m: Minimum centre-to-centre separation (default 0.10m).
            Wider spacing along the barrier leaves gaps for the arm.
    """

    name: str = "obstacle_avoidance"

    def __init__(
        self,
        min_objects: int = 2,
        max_objects: int = 5,
        barrier_y: float = 0.25,
        target_y_range: tuple[float, float] = (0.35, 0.45),
        min_separation_m: float = 0.10,
    ) -> None:
        self.min_objects = min_objects
        self.max_objects = max_objects
        self.barrier_y = barrier_y
        self.target_y_range = target_y_range
        self.min_separation_m = min_separation_m

    def sample_scene(self, rng: np.random.Generator) -> CandidateScene:
        n_vocab = len(OBJECT_VOCAB)

        # Barriers: n-1 objects along the barrier y-line.
        n_total = int(rng.integers(self.min_objects, self.max_objects + 1))
        n_barriers = max(1, n_total - 1)

        barrier_type_ids = [int(rng.integers(0, n_vocab)) for _ in range(n_barriers)]
        barrier_scales = [float(rng.uniform(0.8, 1.2)) for _ in range(n_barriers)]
        barrier_quats = [_random_yaw_quat(rng) for _ in range(n_barriers)]

        # Place barriers in a row along x, at barrier_y.
        barrier_xyzs, placed_b = _place_objects(
            rng, n_barriers, barrier_type_ids, barrier_scales,
            x_range=(-0.35, 0.35),
            y_range=(self.barrier_y - 0.04, self.barrier_y + 0.04),
            min_separation_m=self.min_separation_m,
            margin=0.0,
        )
        barrier_type_ids = [barrier_type_ids[i] for i in placed_b]
        barrier_scales = [barrier_scales[i] for i in placed_b]
        barrier_quats = [barrier_quats[i] for i in placed_b]

        # Target: one object beyond the barrier (larger y).
        tgt_type_id = int(rng.integers(0, n_vocab))
        tgt_scale = float(rng.uniform(0.8, 1.2))
        tgt_quat = _random_yaw_quat(rng)
        tgt_hz = _half_height(tgt_type_id, tgt_scale)
        tgt_x = float(rng.uniform(-0.20, 0.20))
        tgt_y = float(rng.uniform(self.target_y_range[0], self.target_y_range[1]))
        tgt_xyz = np.array([tgt_x, tgt_y, tgt_hz])

        all_type_ids = [*barrier_type_ids, tgt_type_id]
        all_xyzs = [*barrier_xyzs, tgt_xyz]
        all_scales = [*barrier_scales, tgt_scale]
        all_quats = [*barrier_quats, tgt_quat]

        target_local = len(all_type_ids) - 1  # last object is the target
        goal_xyz = np.array([
            tgt_xyz[0],
            tgt_xyz[1],
            tgt_xyz[2] + _IK_GOAL_HEIGHT_M,
        ])

        scene = _build_scene_tensor(all_type_ids, all_xyzs, all_scales, all_quats)
        return CandidateScene(
            scene=scene,
            task_family=self.name,
            target_idx=target_local,
            goal_xyz=goal_xyz,
        )

    def sample_task_description(
        self,
        scene: CandidateScene,
        rng: np.random.Generator,
    ) -> str:
        from data.descriptions import generate_description
        return generate_description(scene, rng)


# ---------------------------------------------------------------------------
# ProceduralSampler: round-robin across templates
# ---------------------------------------------------------------------------


class ProceduralSampler:
    """Samples candidate scenes from multiple templates in configurable mix.

    Args:
        templates: List of TaskTemplate instances.
        mix: Optional weight list matching templates. If None, uniform mix.
        seed: Base seed for the internal generator.
    """

    def __init__(
        self,
        templates: list[TaskTemplate] | None = None,
        mix: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        if templates is None:
            templates = [
                TabletopReachTemplate(),
                ClutteredPickTemplate(),
                ObstacleAvoidanceTemplate(),
            ]
        self.templates: list[TaskTemplate] = templates

        if mix is None:
            mix = [1.0 / len(templates)] * len(templates)
        assert len(mix) == len(templates)
        total = sum(mix)
        self._weights = np.array([w / total for w in mix], dtype=float)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> CandidateScene:
        """Sample one candidate scene from the template distribution."""
        idx = int(self._rng.choice(len(self.templates), p=self._weights))
        return self.templates[idx].sample_scene(self._rng)

    def sample_batch(self, n: int) -> list[CandidateScene]:
        """Sample n candidate scenes."""
        return [self.sample() for _ in range(n)]
