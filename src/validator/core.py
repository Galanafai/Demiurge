"""Drake validator for Demiurge scene tensors.

Implements SceneValidator with four physical validity checks:
  1. no_interpenetration: minimum pairwise signed distance > threshold.
  2. stable_rest: forward simulation for 1.5 s, pose drift below threshold.
  3. ik_reachable: IK from home to a goal pose above the tallest present object.
  4. rrt_solvable: BiRRT from home config to IK solution within budget.

Each validate() call builds a fresh MultibodyPlant and SceneGraph to avoid
Drake context staleness (SKILL.md Drake pitfall #1). validate_batch() uses
multiprocessing.Pool; each worker creates its own plant.

Drake is the single source of truth for validity. No approximations are used
without an explicit APPROX comment.
"""

from __future__ import annotations

import hashlib
import logging
import multiprocessing as mp
import pathlib
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
from scene.vocab import OBJECT_VOCAB

logger = logging.getLogger(__name__)

# Path to the UR5e URDF bundled in the repository.
_UR5E_URDF: pathlib.Path = (
    pathlib.Path(__file__).parent.parent.parent / "assets" / "models" / "ur5e" / "ur5e.urdf"
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTERP_THRESHOLD_M: float = 1e-3
"""Minimum acceptable signed distance between any two collision bodies (1 mm)."""

POSE_DRIFT_TRANS_M: float = 0.005
"""Maximum acceptable translation drift per object after 1.5 s simulation (5 mm)."""

POSE_DRIFT_ROT_RAD: float = 0.0873
"""Maximum acceptable rotation drift per object after 1.5 s simulation (~5 degrees)."""

SIM_DURATION_S: float = 1.5
"""Duration of the forward simulation for the stable_rest check."""

IK_GOAL_HEIGHT_M: float = 0.10
"""Height above the target object centroid for the IK goal pose."""

IK_POSITION_TOL_M: float = 0.01
"""Half-width of the IK position constraint box (1 cm)."""

IK_ORIENTATION_TOL_RAD: float = 0.1
"""Orientation tolerance for the IK constraint (radians)."""

IK_MIN_DIST_M: float = 0.001
"""Minimum collision distance for the IK collision-free constraint (1 mm)."""

IK_MIN_DIST_INFLUENCE_M: float = 0.05
"""Influence distance for the IK collision-free constraint smoothing (5 cm)."""

RRT_BUDGET_S: float = 5.0
"""Default wall-clock budget for BiRRT planning."""

RRT_STEP_SIZE_RAD: float = 0.05
"""Maximum joint-space step per RRT extension (radians)."""

RRT_GOAL_BIAS: float = 0.10
"""Probability of sampling the goal directly in BiRRT."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ValidityReport:
    """Physical validity report for one scene.

    Boolean flags follow the four check names. Continuous metrics are also stored
    for downstream analysis (e.g. rejection rate calibration).

    Attributes:
        no_interpenetration: True if minimum signed distance >= INTERP_THRESHOLD_M.
        stable_rest: True if max pose drift over 1.5 s is within thresholds.
        ik_reachable: True if IK found a collision-free configuration.
        rrt_solvable: True if BiRRT found a path from home to IK solution.
        min_signed_distance_m: Minimum pairwise signed distance. Negative = overlap.
        max_pose_drift_trans_m: Maximum translation drift across all objects.
        max_pose_drift_rot_rad: Maximum rotation drift across all objects.
        ik_solution: Joint configuration found by IK, or None if IK failed.
        rrt_seed: RNG seed used for BiRRT (reproducibility).
        accepted: Conjunction of all four boolean checks.
        error: Non-empty string if an unexpected exception occurred in a check.
            A non-empty error does NOT mean a check passed; the corresponding
            boolean will be False.
        elapsed_s: Wall-clock time for the full validate() call.
    """

    no_interpenetration: bool = False
    stable_rest: bool = False
    ik_reachable: bool = False
    rrt_solvable: bool = False
    min_signed_distance_m: float = float("nan")
    max_pose_drift_trans_m: float = float("nan")
    max_pose_drift_rot_rad: float = float("nan")
    ik_solution: np.ndarray | None = None
    rrt_seed: int = 0
    accepted: bool = False
    error: str = ""
    elapsed_s: float = 0.0

    def __post_init__(self) -> None:
        self.accepted = (
            self.no_interpenetration
            and self.stable_rest
            and self.ik_reachable
            and self.rrt_solvable
        )


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


class SceneValidator:
    """Physical validity checker for SceneTensor scenes.

    Each call to validate() builds a fresh Drake plant+scene_graph pair.
    Instances are safe to use from a single process. For multiprocessing,
    use validate_batch() which creates one validator per worker.

    Args:
        workspace_bounds: The valid workspace for object placement.
        rrt_budget_s: Wall-clock time budget for BiRRT.
        urdf_path: Path to the UR5e URDF. Defaults to the bundled asset.
    """

    def __init__(
        self,
        workspace_bounds: WorkspaceBounds | None = None,
        rrt_budget_s: float = RRT_BUDGET_S,
        urdf_path: pathlib.Path | None = None,
    ) -> None:
        self._bounds = workspace_bounds or WorkspaceBounds.default()
        self._rrt_budget_s = rrt_budget_s
        self._urdf_path = urdf_path or _UR5E_URDF
        if not self._urdf_path.exists():
            raise FileNotFoundError(
                f"UR5e URDF not found at {self._urdf_path}. "
                "Ensure assets/models/ur5e/ur5e.urdf is present in the repository."
            )
        self._cache: dict[bytes, ValidityReport] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, scene: SceneTensor, rrt_seed: int = 0) -> ValidityReport:
        """Run all four validity checks on a scene.

        Results are cached by scene content hash. The cache is not shared across
        processes; each worker has its own instance.

        Args:
            scene: Physical-space SceneTensor (not normalized).
            rrt_seed: Seed for BiRRT; logged in the report for reproducibility.

        Returns:
            ValidityReport with all fields populated.
        """
        cache_key = _scene_hash(scene)
        if cache_key in self._cache:
            return self._cache[cache_key]

        t0 = time.monotonic()
        report = self._run_checks(scene, rrt_seed)
        report.elapsed_s = time.monotonic() - t0

        self._cache[cache_key] = report
        return report

    def validate_batch(
        self,
        scenes: Sequence[SceneTensor],
        seeds: Sequence[int] | None = None,
        num_workers: int = 1,
    ) -> list[ValidityReport]:
        """Validate a batch of scenes, optionally in parallel.

        Each worker process creates its own SceneValidator instance to avoid
        shared Drake context state.

        Args:
            scenes: Sequence of physical-space SceneTensors.
            seeds: RRT seeds per scene. Defaults to range(len(scenes)).
            num_workers: Number of worker processes.

        Returns:
            List of ValidityReport in the same order as scenes.
        """
        if seeds is None:
            effective_seeds = list(range(len(scenes)))
        else:
            effective_seeds = list(seeds)

        args = [
            (scene, seed, self._bounds, self._rrt_budget_s, self._urdf_path)
            for scene, seed in zip(scenes, effective_seeds)
        ]

        if num_workers <= 1:
            return [_worker_validate(a) for a in args]

        with mp.Pool(
            processes=num_workers,
            initializer=_pool_initializer,
        ) as pool:
            return pool.map(_worker_validate, args)

    # ------------------------------------------------------------------
    # Internal check orchestration
    # ------------------------------------------------------------------

    def _run_checks(self, scene: SceneTensor, rrt_seed: int) -> ValidityReport:
        """Run all checks, short-circuiting on the first failure where safe."""
        report = ValidityReport(rrt_seed=rrt_seed)

        # Check 1: interpenetration (pure geometry, no simulation).
        try:
            ok, min_dist = self._check_interpenetration(scene)
            report.no_interpenetration = ok
            report.min_signed_distance_m = min_dist
        except Exception as exc:
            report.error += f"interpenetration_error={exc!r}; "
            report.no_interpenetration = False

        if not report.no_interpenetration:
            return report  # Short-circuit: no point simulating an overlapping scene.

        # Check 2: stable rest (forward simulation).
        try:
            ok, max_trans, max_rot = self._check_stable_rest(scene)
            report.stable_rest = ok
            report.max_pose_drift_trans_m = max_trans
            report.max_pose_drift_rot_rad = max_rot
        except Exception as exc:
            report.error += f"stable_rest_error={exc!r}; "
            report.stable_rest = False

        # Check 3: IK reachability (always attempted even if stable_rest failed,
        # because the two checks are independent).
        try:
            ok, q_sol = self._check_ik_reachable(scene)
            report.ik_reachable = ok
            report.ik_solution = q_sol
        except Exception as exc:
            report.error += f"ik_error={exc!r}; "
            report.ik_reachable = False

        # Check 4: BiRRT solvability (only if IK succeeded).
        if report.ik_reachable and report.ik_solution is not None:
            try:
                ok = self._check_rrt_solvable(
                    scene, report.ik_solution, rrt_seed
                )
                report.rrt_solvable = ok
            except Exception as exc:
                report.error += f"rrt_error={exc!r}; "
                report.rrt_solvable = False

        report.accepted = (
            report.no_interpenetration
            and report.stable_rest
            and report.ik_reachable
            and report.rrt_solvable
        )
        return report

    # ------------------------------------------------------------------
    # Check 1: Interpenetration
    # ------------------------------------------------------------------

    def _check_interpenetration(self, scene: SceneTensor) -> tuple[bool, float]:
        """Return (no_overlap, min_signed_distance_m).

        Builds a fresh continuous-time plant (time_step=0.0) for geometry queries.
        Negative distance means penetration.
        """
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        _populate_plant_with_scene(plant, scene, self._bounds)
        plant.Finalize()

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        sg_context = scene_graph.GetMyContextFromRoot(context)

        query_object = scene_graph.get_query_output_port().Eval(sg_context)
        pairs = query_object.ComputeSignedDistancePairwiseClosestPoints()

        if not pairs:
            return True, float("inf")

        min_dist = min(p.distance for p in pairs)
        return min_dist >= INTERP_THRESHOLD_M, min_dist

    # ------------------------------------------------------------------
    # Check 2: Stable rest
    # ------------------------------------------------------------------

    def _check_stable_rest(
        self, scene: SceneTensor
    ) -> tuple[bool, float, float]:
        """Return (stable, max_trans_drift_m, max_rot_drift_rad).

        Builds a discrete-time plant (time_step=0.001) for simulation.
        Advances to SIM_DURATION_S and measures pose changes per object body.
        """
        from pydrake.math import RigidTransform
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.analysis import Simulator
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
        # Add the table surface so objects rest on it rather than falling.
        _add_table_to_plant(plant)
        body_indices = _populate_plant_with_scene(plant, scene, self._bounds)
        plant.Finalize()

        diagram = builder.Build()
        simulator = Simulator(diagram)
        simulator.Initialize()

        context = simulator.get_mutable_context()
        plant_context_t0 = plant.GetMyContextFromRoot(context)

        # Record initial poses.
        poses_t0: list[RigidTransform] = []
        for idx in body_indices:
            body = plant.get_body(idx)
            X = plant.EvalBodyPoseInWorld(plant_context_t0, body)
            poses_t0.append(X)

        simulator.AdvanceTo(SIM_DURATION_S)

        plant_context_tf = plant.GetMyContextFromRoot(context)
        max_trans = 0.0
        max_rot = 0.0
        for idx, X0 in zip(body_indices, poses_t0):
            body = plant.get_body(idx)
            Xf = plant.EvalBodyPoseInWorld(plant_context_tf, body)
            trans_drift = float(
                np.linalg.norm(Xf.translation() - X0.translation())
            )
            # Rotation drift: angle of relative rotation R0^T * Rf.
            dR = X0.rotation().matrix().T @ Xf.rotation().matrix()
            # Rodrigues angle = arccos((trace(R)-1)/2).
            cos_angle = (np.trace(dR) - 1.0) / 2.0
            cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
            rot_drift = float(np.arccos(cos_angle))
            max_trans = max(max_trans, trans_drift)
            max_rot = max(max_rot, rot_drift)

        stable = (
            max_trans <= POSE_DRIFT_TRANS_M and max_rot <= POSE_DRIFT_ROT_RAD
        )
        return stable, max_trans, max_rot

    # ------------------------------------------------------------------
    # Check 3: IK reachability
    # ------------------------------------------------------------------

    def _check_ik_reachable(
        self, scene: SceneTensor
    ) -> tuple[bool, np.ndarray | None]:
        """Return (success, q_solution_or_None).

        Builds a robot-only plant (no scene objects). This avoids the optimizer
        treating scene objects as free variables, which caused unit-quaternion
        constraint violations when scene object DOF were included.

        The goal pose is IK_GOAL_HEIGHT_M above the highest present object
        centroid. Only a position constraint is imposed; orientation is unconstrained
        (a reach task, not a grasp task). Scene-collision awareness is delegated
        to the BiRRT check.

        Returns the 6-DOF joint configuration (robot DOF only).
        """
        from pydrake.multibody.inverse_kinematics import InverseKinematics
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.solvers import Solve
        from pydrake.systems.framework import DiagramBuilder

        # Robot-only plant: no scene objects, no collision constraint against them.
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        _load_ur5e(plant, self._urdf_path)
        plant.Finalize()
        _filter_ur5e_self_collisions(plant, scene_graph)

        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        q0 = np.zeros(plant.num_positions())  # UR5e welded base: 6 DOF all zero.
        plant.SetPositions(plant_context, q0)

        goal_xyz = _goal_pose_above_scene(scene)

        tool0_frame = plant.GetFrameByName("tool0")
        world_frame = plant.world_frame()

        # Fresh IK per goal (SKILL.md Drake pitfall #2).
        ik = InverseKinematics(plant, plant_context)
        ik.AddPositionConstraint(
            frameB=tool0_frame,
            p_BQ=np.zeros(3),
            frameA=world_frame,
            p_AQ_lower=goal_xyz - IK_POSITION_TOL_M,
            p_AQ_upper=goal_xyz + IK_POSITION_TOL_M,
        )
        # Self-collision avoidance. The near-chain pairs are already filtered
        # by _filter_ur5e_self_collisions, so this only constrains meaningful
        # non-adjacent link pairs.
        ik.AddMinimumDistanceLowerBoundConstraint(
            IK_MIN_DIST_M, IK_MIN_DIST_INFLUENCE_M
        )

        ik.prog().SetInitialGuess(ik.q(), q0)
        result = Solve(ik.prog())

        if result.is_success():
            q_sol = result.GetSolution(ik.q())
            return True, q_sol
        return False, None

    # ------------------------------------------------------------------
    # Check 4: BiRRT
    # ------------------------------------------------------------------

    def _check_rrt_solvable(
        self,
        scene: SceneTensor,
        q_goal: np.ndarray,
        seed: int,
    ) -> bool:
        """Return True if BiRRT finds a path from the home config to q_goal.

        Uses a manual bidirectional RRT. The plant contains the UR5e robot plus
        all present scene objects welded as static obstacles at their scene poses.
        Collision queries therefore include robot-vs-object and robot-self pairs.

        The IK plant (used in _check_ik_reachable) is robot-only and has no
        knowledge of scene objects. IK can return a q_goal that collides with a
        scene object. This check catches that case: is_collision_free(q_goal) is
        evaluated first; if the goal configuration intersects any welded object the
        check returns False immediately before any tree expansion.
        """
        from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
        from pydrake.systems.framework import DiagramBuilder

        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
        _load_ur5e(plant, self._urdf_path)
        _populate_rrt_plant_with_scene(plant, scene, self._bounds)
        plant.Finalize()
        _filter_ur5e_self_collisions(plant, scene_graph)
        diagram = builder.Build()
        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        sg_context = scene_graph.GetMyContextFromRoot(context)

        rng = np.random.default_rng(seed)
        q_home = np.zeros(plant.num_positions())  # 6 DOF, all zero = robot home.

        # Joint limits from the robot plant (6 DOF).
        lower = plant.GetPositionLowerLimits()
        upper = plant.GetPositionUpperLimits()

        def is_collision_free(q: np.ndarray) -> bool:
            plant.SetPositions(plant_context, q)
            query_object = scene_graph.get_query_output_port().Eval(sg_context)
            pairs = query_object.ComputeSignedDistancePairwiseClosestPoints()
            if not pairs:
                return True
            return min(p.distance for p in pairs) >= INTERP_THRESHOLD_M

        def extend(
            tree: list[tuple[np.ndarray, int]], q_target: np.ndarray
        ) -> tuple[bool, np.ndarray]:
            """Extend the tree toward q_target. Returns (reached, q_new)."""
            nearest_idx = min(
                range(len(tree)),
                key=lambda idx: float(np.linalg.norm(tree[idx][0] - q_target)),
            )
            q_near = tree[nearest_idx][0]
            direction = q_target - q_near
            dist = float(np.linalg.norm(direction))
            if dist < 1e-8:
                return True, q_near
            step = min(RRT_STEP_SIZE_RAD, dist)
            q_new = q_near + step * direction / dist
            q_new = np.clip(q_new, lower, upper)
            if is_collision_free(q_new):
                tree.append((q_new, nearest_idx))
                reached = float(np.linalg.norm(q_new - q_target)) < RRT_STEP_SIZE_RAD
                return reached, q_new
            return False, q_near

        if not is_collision_free(q_home) or not is_collision_free(q_goal):
            return False

        tree_a: list[tuple[np.ndarray, int]] = [(q_home, -1)]
        tree_b: list[tuple[np.ndarray, int]] = [(q_goal, -1)]

        deadline = time.monotonic() + self._rrt_budget_s
        while time.monotonic() < deadline:
            if rng.random() < RRT_GOAL_BIAS:
                q_rand = tree_b[0][0]
            else:
                q_rand = rng.uniform(lower, upper)

            reached_a, q_new_a = extend(tree_a, q_rand)
            if reached_a:
                return True  # tree_a reached the sample from tree_b's root

            reached_b, _ = extend(tree_b, q_new_a)
            if reached_b:
                return True  # tree_b reached q_new_a (trees connected)

            # Swap trees every iteration (bidirectional).
            tree_a, tree_b = tree_b, tree_a

        return False  # Budget exhausted.


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _scene_hash(scene: SceneTensor) -> bytes:
    """Compute a deterministic hash of a SceneTensor for cache keying."""
    h = hashlib.sha256()
    h.update(scene.object_types.numpy().tobytes())
    h.update(scene.poses.numpy().tobytes())
    h.update(scene.scales.numpy().tobytes())
    h.update(scene.presence.numpy().tobytes())
    return h.digest()


def _load_ur5e(plant: object, urdf_path: pathlib.Path) -> None:
    """Load the UR5e URDF and weld its base to the world frame."""
    from pydrake.math import RigidTransform
    from pydrake.multibody.parsing import Parser

    parser = Parser(plant)  # type: ignore[arg-type]
    parser.AddModels(str(urdf_path))
    plant.WeldFrames(  # type: ignore[attr-defined]
        plant.world_frame(),  # type: ignore[attr-defined]
        plant.GetFrameByName("base_link"),  # type: ignore[attr-defined]
        RigidTransform(),
    )


# UR5e kinematic chain order (base to tip).
_UR5E_CHAIN: list[str] = [
    "base_link",
    "shoulder_link",
    "upper_arm_link",
    "forearm_link",
    "wrist_1_link",
    "wrist_2_link",
    "wrist_3_link",
]


def _filter_ur5e_self_collisions(plant: object, scene_graph: object) -> None:
    """Exclude collision pairs between links within 2 joints on the UR5e chain.

    Drake auto-filters adjacent (parent-child) pairs. This function additionally
    filters chain-distance-2 pairs whose primitive cylinder approximations
    overlap in common configurations.

    Must be called after plant.Finalize() but before builder.Build().
    """
    from pydrake.geometry import CollisionFilterDeclaration, GeometrySet

    cfm = scene_graph.collision_filter_manager()  # type: ignore[attr-defined]
    for i in range(len(_UR5E_CHAIN)):
        for j in range(i + 2, min(i + 3, len(_UR5E_CHAIN))):
            body_a = plant.GetBodyByName(_UR5E_CHAIN[i])  # type: ignore[attr-defined]
            body_b = plant.GetBodyByName(_UR5E_CHAIN[j])  # type: ignore[attr-defined]
            gids_a = plant.GetCollisionGeometriesForBody(body_a)  # type: ignore[attr-defined]
            gids_b = plant.GetCollisionGeometriesForBody(body_b)  # type: ignore[attr-defined]
            if gids_a and gids_b:
                cfm.Apply(
                    CollisionFilterDeclaration().ExcludeBetween(
                        GeometrySet(gids_a), GeometrySet(gids_b)
                    )
                )


_TABLE_SDF: str = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="table">
    <static>true</static>
    <link name="surface">
      <pose>0 0 -0.01 0 0 0</pose>
      <collision name="col">
        <geometry><box><size>2.0 2.0 0.02</size></box></geometry>
      </collision>
    </link>
  </model>
</sdf>"""


def _add_table_to_plant(plant: object) -> None:
    """Add a large thin static table surface to the plant (top face at z=0).

    The table is a fixed (static) body 2 m x 2 m x 0.02 m centred at z=-0.01 m
    so that the top surface lies at world z=0. Objects placed at z>=0 rest on it.
    """
    from pydrake.multibody.parsing import Parser

    Parser(plant).AddModelsFromString(_TABLE_SDF, "sdf")  # type: ignore[arg-type]


def _populate_plant_with_scene(
    plant: object,
    scene: SceneTensor,
    bounds: WorkspaceBounds,
) -> list[object]:
    """Add all present scene objects to the plant.

    Returns the list of BodyIndex objects for the added bodies so callers
    can track them for pose queries.
    """
    from pydrake.multibody.parsing import Parser

    parser = Parser(plant)  # type: ignore[arg-type]
    body_indices: list[object] = []

    for i in range(N_MAX):
        if not scene.presence[i].item():
            continue

        from scene.vocab import build_sdf

        type_id = int(scene.object_types[i].item())
        entry = OBJECT_VOCAB[type_id]
        scale = float(scene.scales[i].mean().item())
        model_name = f"obj_{i}_{entry.name}"
        sdf_str = build_sdf(entry, model_name=model_name, scale=scale)
        model_instances = parser.AddModelsFromString(sdf_str, "sdf")

        # Set the default free-body pose so Drake places the object at the
        # position and orientation specified by the SceneTensor.
        xyz = scene.poses[i, 0:3].numpy().astype(float)
        q_wxyz = scene.poses[i, 3:7].numpy().astype(float)
        from pydrake.common.eigen_geometry import Quaternion as DrakeQuaternion
        from pydrake.math import RigidTransform, RotationMatrix

        drake_quat = DrakeQuaternion(
            float(q_wxyz[0]),
            float(q_wxyz[1]),
            float(q_wxyz[2]),
            float(q_wxyz[3]),
        )
        rot = RotationMatrix(drake_quat)
        X_WB = RigidTransform(rot, xyz)

        for mi in model_instances:
            for body_idx in plant.GetBodyIndices(mi):  # type: ignore[attr-defined]
                body = plant.get_body(body_idx)  # type: ignore[attr-defined]
                if body.name() == "link":
                    plant.SetDefaultFloatingBaseBodyPose(body, X_WB)  # type: ignore[attr-defined]
                    body_indices.append(body_idx)

    return body_indices


def _populate_rrt_plant_with_scene(
    plant: object,
    scene: SceneTensor,
    bounds: WorkspaceBounds,
) -> None:
    """Weld all present scene objects into the plant as static obstacles.

    Unlike _populate_plant_with_scene (which sets a floating-body default pose
    for dynamic simulation), this function welds each object body to the world
    frame at its scene pose. Welded bodies contribute collision geometry to
    queries but add no DOF to the joint-space.

    Must be called before plant.Finalize().

    Drake API notes:
    - Bodies are resolved via plant.GetBodyIndices(model_instance), not
      GetBodyByName, to avoid ambiguity when multiple objects share the link
      name 'link'.
    - WeldFrames uses body.body_frame() as the child frame, not a named link
      frame, to be safe against SDFs with non-identity inertial origins.
    - Pre-Finalize, is_floating_base_body() cannot be called. Non-world bodies
      are identified by excluding plant.world_body().index().
    """
    from pydrake.common.eigen_geometry import Quaternion as DrakeQuaternion
    from pydrake.math import RigidTransform, RotationMatrix
    from pydrake.multibody.parsing import Parser

    from scene.vocab import build_sdf

    parser = Parser(plant)  # type: ignore[arg-type]
    world_body_idx = plant.world_body().index()  # type: ignore[attr-defined]

    for i in range(N_MAX):
        if not scene.presence[i].item():
            continue

        type_id = int(scene.object_types[i].item())
        entry = OBJECT_VOCAB[type_id]
        scale = float(scene.scales[i].mean().item())
        model_name = f"rrt_obj_{i}_{entry.name}"
        sdf_str = build_sdf(entry, model_name=model_name, scale=scale)

        xyz = scene.poses[i, 0:3].numpy().astype(float)
        q_wxyz = scene.poses[i, 3:7].numpy().astype(float)
        drake_quat = DrakeQuaternion(
            float(q_wxyz[0]),
            float(q_wxyz[1]),
            float(q_wxyz[2]),
            float(q_wxyz[3]),
        )
        X_WB = RigidTransform(RotationMatrix(drake_quat), xyz)

        model_instances = parser.AddModelsFromString(sdf_str, "sdf")
        for mi in model_instances:
            for body_idx in plant.GetBodyIndices(mi):  # type: ignore[attr-defined]
                if body_idx == world_body_idx:
                    continue
                body = plant.get_body(body_idx)  # type: ignore[attr-defined]
                # Weld via body.body_frame(), not GetFrameByName, to handle any
                # SDF where the link frame and body frame differ.
                plant.WeldFrames(  # type: ignore[attr-defined]
                    plant.world_frame(),  # type: ignore[attr-defined]
                    body.body_frame(),
                    X_WB,
                )


def _goal_pose_above_scene(scene: SceneTensor) -> np.ndarray:
    """Return an xyz goal position IK_GOAL_HEIGHT_M above the highest active object."""
    max_z = 0.0
    best_xy = np.array([0.0, 0.0])
    for i in range(N_MAX):
        if not scene.presence[i].item():
            continue
        z = float(scene.poses[i, 2].item())
        if z > max_z:
            max_z = z
            best_xy = scene.poses[i, 0:2].numpy()
    return np.array([best_xy[0], best_xy[1], max_z + IK_GOAL_HEIGHT_M])


# ---------------------------------------------------------------------------
# Multiprocessing helpers
# ---------------------------------------------------------------------------


def _pool_initializer() -> None:
    """Worker process initializer. Imports Drake so the first validate() is fast."""
    import pydrake.multibody.plant  # noqa: F401


def _worker_validate(
    args: tuple[SceneTensor, int, WorkspaceBounds, float, pathlib.Path],
) -> ValidityReport:
    """Top-level function for multiprocessing.Pool workers.

    Must be a module-level function (not a closure) to be picklable.
    """
    scene, seed, bounds, rrt_budget_s, urdf_path = args
    validator = SceneValidator(
        workspace_bounds=bounds,
        rrt_budget_s=rrt_budget_s,
        urdf_path=urdf_path,
    )
    return validator.validate(scene, rrt_seed=seed)
