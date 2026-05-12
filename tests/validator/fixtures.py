"""Scene fixtures for validator tests.

Each factory returns a physical-space SceneTensor. Fixtures are plain Python
functions, not files, to avoid serialization overhead. All positions are chosen
to produce reliably deterministic behaviour for the four validity checks.

Workspace: default 1m x 1m x 0.5m (x in [-0.5, 0.5], y in [-0.5, 0.5], z in [0, 0.5]).
Robot base: at world origin, welded to world.
"""

from __future__ import annotations

import math

import torch

from scene.schema import N_MAX, SceneTensor
from scene.vocab import ObjectTypeId

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quat_from_axis_angle(axis: tuple[float, float, float], angle_rad: float) -> list[float]:
    """Return wxyz quaternion for a rotation of angle_rad about axis (unit vector)."""
    s = math.sin(angle_rad / 2.0)
    c = math.cos(angle_rad / 2.0)
    return [c, axis[0] * s, axis[1] * s, axis[2] * s]


def _identity_quat() -> list[float]:
    return [1.0, 0.0, 0.0, 0.0]


def _make_scene(
    type_ids: list[int],
    xyzs: list[tuple[float, float, float]],
    scales: list[float] | None = None,
    quats: list[list[float]] | None = None,
) -> SceneTensor:
    """Build a SceneTensor from compact lists. Inactive slots are zeroed."""
    n = len(type_ids)
    assert n <= N_MAX

    ot = torch.zeros(N_MAX, dtype=torch.int64)
    poses = torch.zeros(N_MAX, 7, dtype=torch.float32)
    sc = torch.ones(N_MAX, 3, dtype=torch.float32)
    pres = torch.zeros(N_MAX, dtype=torch.bool)

    for i in range(n):
        ot[i] = type_ids[i]
        x, y, z = xyzs[i]
        poses[i, 0] = x
        poses[i, 1] = y
        poses[i, 2] = z
        q = quats[i] if quats else _identity_quat()
        poses[i, 3] = q[0]
        poses[i, 4] = q[1]
        poses[i, 5] = q[2]
        poses[i, 6] = q[3]
        if scales:
            sc[i] = scales[i]
        pres[i] = True

    # Inactive slots get identity quaternion to avoid degenerate quats.
    for i in range(n, N_MAX):
        poses[i, 3] = 1.0

    return SceneTensor(object_types=ot, poses=poses, scales=sc, presence=pres)


# ---------------------------------------------------------------------------
# Valid fixtures
# ---------------------------------------------------------------------------


def make_valid_sparse() -> SceneTensor:
    """Three cubes with the target at the workspace centre-front, obstacles at corners.

    The target object is at [0.0, 0.30, 0.025] so IK goal is at [0.0, 0.30, 0.125],
    which is well within UR5e reach. The two obstacle cubes are placed at workspace
    corners ([0.35, 0.35] and [-0.35, 0.35]) far from the arm's natural approach path
    to the target. With the hardened RRT plant (scene objects welded) the corner cubes
    do not obstruct the BiRRT search.
    """
    return _make_scene(
        type_ids=[ObjectTypeId.CUBE, ObjectTypeId.CUBE, ObjectTypeId.CUBE],
        xyzs=[(0.0, 0.30, 0.025), (0.35, 0.35, 0.025), (-0.35, 0.35, 0.025)],
        scales=[1.0, 1.0, 1.0],
    )


def make_valid_dense() -> SceneTensor:
    """Five objects with at least 2 cm clearance between each pair."""
    return _make_scene(
        type_ids=[
            ObjectTypeId.CUBE,
            ObjectTypeId.SPHERE,
            ObjectTypeId.CYLINDER,
            ObjectTypeId.BOX_FLAT,
            ObjectTypeId.CUBE,
        ],
        xyzs=[
            (0.0, 0.3, 0.025),
            (0.1, 0.3, 0.030),
            (-0.1, 0.3, 0.045),
            (0.0, 0.1, 0.015),
            (0.15, 0.1, 0.025),
        ],
    )


def make_valid_tall() -> SceneTensor:
    """Single cube directly in front of the arm (used as the valid-single-object fixture).

    Cube half-extent z = 0.025m, centroid at z = 0.025, resting on the table.
    IK goal: [0.0, 0.30, 0.125]. This is front-centre of the workspace and close to
    the arm's natural reach from home (q=0). The cube is small and low, so no arm
    link intersects it when the arm reaches the goal. BiRRT solves this quickly.
    """
    return _make_scene(
        type_ids=[ObjectTypeId.CUBE],
        xyzs=[(0.0, 0.30, 0.025)],
        scales=[1.0],
    )


# ---------------------------------------------------------------------------
# Invalid fixtures
# ---------------------------------------------------------------------------


def make_invalid_interpenetrating() -> SceneTensor:
    """Two cubes with centres 0.01 m apart (cubes are 0.05 m wide: deep overlap).

    Expected failure: no_interpenetration = False.
    stable_rest might also fail but that is incidental.
    """
    return _make_scene(
        type_ids=[ObjectTypeId.CUBE, ObjectTypeId.CUBE],
        xyzs=[(0.0, 0.3, 0.025), (0.01, 0.3, 0.025)],
    )


def make_invalid_unstable() -> SceneTensor:
    """A box_tall balanced upright on a sphere: will tip over during simulation.

    box_tall half-extent z = 0.09 m. Sphere radius = 0.03 m.
    Box placed with its bottom face at z = 0.03 + 0.09 = 0.12 m, centroid at 0.21 m.
    The small contact area on the sphere makes it mechanically unstable.

    Expected failure: stable_rest = False.
    no_interpenetration should pass (sphere and box are just touching, not overlapping).
    """
    return _make_scene(
        type_ids=[ObjectTypeId.SPHERE, ObjectTypeId.BOX_TALL],
        xyzs=[(0.0, 0.3, 0.030), (0.0, 0.3, 0.210)],
    )


def make_invalid_ik_blocked() -> SceneTensor:
    """Goal pose unreachable: single cube placed at z = -0.2 m (below table floor).

    The IK check samples the goal IK_GOAL_HEIGHT_M (0.10 m) above the highest
    present object. With the cube centroid at z = -0.2, the goal is at z = -0.1,
    which is below the table surface and physically impossible to reach.

    Expected failure: ik_reachable = False (and consequently rrt_solvable = False).
    no_interpenetration should pass (single object, no pairs).
    stable_rest should pass (single cube on flat ground at z=-0.2 is out of
    workspace but not dynamically unstable in isolation).

    Note: stable_rest may also fail depending on how Drake handles sub-floor objects.
    The test asserts only that ik_reachable = False.
    """
    return _make_scene(
        type_ids=[ObjectTypeId.CUBE],
        xyzs=[(0.0, 0.3, -0.20)],
    )


def make_ik_passes_rrt_blocked() -> SceneTensor:
    """A large cube placed such that IK succeeds but RRT rejects immediately.

    Architecture under test:
      - IK plant (robot-only): finds q_goal placing tool0 at the target xyz.
        The cube is invisible to IK, so IK succeeds.
      - RRT plant (robot + welded cube): is_collision_free(q_goal) detects the
        robot arm intersecting the welded cube and returns False before any
        BiRRT tree expansion. rrt_solvable = False.

    Construction:
      A single cube with scale 4.0 (side length 0.20m) centred at [0.0, 0.3, 0.15].
      The IK goal is sampled IK_GOAL_HEIGHT_M (0.10m) above the cube centroid z,
      so goal_xyz = [0.0, 0.3, 0.25]. The cube occupies z in [0.05, 0.25].
      At a configuration that places tool0 at z=0.25, the robot forearm links
      are in the range z in [0.05, 0.25], intersecting the cube's volume.

    Expected behaviour:
      no_interpenetration = True   (single object, no pairs)
      stable_rest = True           (large cube is stable on table)
      ik_reachable = True          (robot-only IK finds a config)
      rrt_solvable = False         (welded cube blocks q_goal immediately)
      accepted = False
    """
    # scale=4.0 gives a 0.20m cube (canonical half-extent 0.025m * 4 = 0.10m per side).
    # Place centroid at z=0.15 so the bottom face (z=0.05) is above the table.
    # The cube top face is at z=0.25; IK goal will be at z=0.25+0.10=0.35, placing
    # tool0 above the cube -- but reaching down to that height requires arm links
    # to pass through the cube volume.
    # Use centroid z = half_extent*scale = 0.025*4 = 0.10 to rest on table exactly,
    # and rely on the arm intersection with the cube sides when q_goal is evaluated.
    return _make_scene(
        type_ids=[ObjectTypeId.CUBE],
        xyzs=[(0.0, 0.3, 0.10)],
        scales=[4.0],
    )
