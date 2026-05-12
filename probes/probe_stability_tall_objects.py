#!/usr/bin/env python3
"""Stability probe: test tall vocab entries in single-object isolation.

Replicates SceneValidator._check_stable_rest exactly (same plant, table,
time_step, sim duration, drift thresholds) but operates on a single object
at a time so we can isolate which entry is responsible for the stable_rest
regression from 95% -> 67.8%.

Entries tested (hz > 0.08m):
  - BOX_TALL      (ID 3): hz=0.090m
  - MUSTARD_BOTTLE (ID 5): hz=0.095m
  - SUGAR_BOX     (ID 6): hz=0.0875m
  - BLEACH_CLEANSER (ID 8): hz=0.125m

Each entry is tested at 3 scales (0.8, 1.0, 1.2) and 3 yaw values
(0, pi/4, pi/2) to catch scale-dependent or orientation-dependent failures.

Output: per-entry table of (scale, yaw_deg, trans_drift_mm, rot_drift_deg,
PASS/FAIL) to stdout. Total runtime ~60s.

Usage:
    uv run python probes/probe_stability_tall_objects.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from scene.schema import N_MAX, SceneTensor, WorkspaceBounds  # noqa: E402
from scene.vocab import OBJECT_VOCAB, ObjectTypeId, build_sdf  # noqa: E402
from validator.core import (  # noqa: E402
    POSE_DRIFT_ROT_RAD,
    POSE_DRIFT_TRANS_M,
    SIM_DURATION_S,
    _TABLE_SDF,
)

# Exact same constants as SceneValidator._check_stable_rest.
_TIME_STEP = 0.001

# Entries to probe: all with hz > 0.08m in the current 12-vocab.
_PROBE_ENTRIES = [
    (ObjectTypeId.BOX_TALL,        "BOX_TALL",        3),
    (ObjectTypeId.MUSTARD_BOTTLE,  "MUSTARD_BOTTLE",  5),
    (ObjectTypeId.SUGAR_BOX,       "SUGAR_BOX",       6),
    (ObjectTypeId.BLEACH_CLEANSER, "BLEACH_CLEANSER", 8),
]

_SCALES = [0.8, 1.0, 1.2]
# Yaw values: 0, 45, 90 degrees.
_YAWS_DEG = [0.0, 45.0, 90.0]


def _make_single_object_scene(type_id: int, scale: float, yaw_deg: float) -> SceneTensor:
    """Build a SceneTensor with one object centred at (0.0, 0.3, hz)."""
    entry = OBJECT_VOCAB[type_id]
    hz = entry.canonical_half_extents_m[2] * scale

    yaw = math.radians(yaw_deg)
    # Quaternion wxyz for yaw rotation about Z.
    q_w = math.cos(yaw / 2.0)
    q_z = math.sin(yaw / 2.0)

    import torch
    ot = torch.zeros(N_MAX, dtype=torch.int64)
    poses = torch.zeros(N_MAX, 7, dtype=torch.float32)
    sc = torch.ones(N_MAX, 3, dtype=torch.float32)
    pres = torch.zeros(N_MAX, dtype=torch.bool)

    ot[0] = type_id
    poses[0, 0] = 0.0   # x
    poses[0, 1] = 0.3   # y: within UR5e reachable workspace
    poses[0, 2] = hz    # z: centroid at half-height above table (z=0 surface)
    poses[0, 3] = q_w   # quaternion w
    poses[0, 4] = 0.0   # x
    poses[0, 5] = 0.0   # y
    poses[0, 6] = q_z   # z
    sc[0] = scale
    pres[0] = True

    # Identity quaternion for inactive slots.
    for i in range(1, N_MAX):
        poses[i, 3] = 1.0

    return SceneTensor(object_types=ot, poses=poses, scales=sc, presence=pres)


def _run_stable_rest(scene: SceneTensor) -> tuple[bool, float, float]:
    """Replicate SceneValidator._check_stable_rest exactly."""
    from pydrake.math import RigidTransform  # noqa: F401 (used in type annotation)
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.analysis import Simulator
    from pydrake.systems.framework import DiagramBuilder

    from validator.core import _populate_plant_with_scene

    bounds = WorkspaceBounds.default()

    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=_TIME_STEP)

    # Add table (identical to _add_table_to_plant).
    Parser(plant).AddModelsFromString(_TABLE_SDF, "sdf")

    body_indices = _populate_plant_with_scene(plant, scene, bounds)
    plant.Finalize()

    diagram = builder.Build()
    simulator = Simulator(diagram)
    simulator.Initialize()

    context = simulator.get_mutable_context()
    plant_ctx_t0 = plant.GetMyContextFromRoot(context)

    poses_t0 = []
    for idx in body_indices:
        body = plant.get_body(idx)
        X = plant.EvalBodyPoseInWorld(plant_ctx_t0, body)
        poses_t0.append(X)

    simulator.AdvanceTo(SIM_DURATION_S)

    plant_ctx_tf = plant.GetMyContextFromRoot(context)
    max_trans = 0.0
    max_rot = 0.0
    for idx, X0 in zip(body_indices, poses_t0):
        body = plant.get_body(idx)
        Xf = plant.EvalBodyPoseInWorld(plant_ctx_tf, body)
        trans_drift = float(np.linalg.norm(Xf.translation() - X0.translation()))
        dR = X0.rotation().matrix().T @ Xf.rotation().matrix()
        cos_angle = float(np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0))
        rot_drift = float(np.arccos(cos_angle))
        max_trans = max(max_trans, trans_drift)
        max_rot = max(max_rot, rot_drift)

    stable = max_trans <= POSE_DRIFT_TRANS_M and max_rot <= POSE_DRIFT_ROT_RAD
    return stable, max_trans, max_rot


def main() -> None:
    print(
        f"\nStability probe: SIM_DURATION={SIM_DURATION_S}s, "
        f"TRANS_THRESHOLD={POSE_DRIFT_TRANS_M*1000:.1f}mm, "
        f"ROT_THRESHOLD={math.degrees(POSE_DRIFT_ROT_RAD):.1f}deg\n"
    )

    header = (
        f"{'Entry':<20} {'hz_m':>6} {'scale':>6} {'yaw_deg':>8} "
        f"{'trans_mm':>10} {'rot_deg':>9}  {'Result'}"
    )
    print(header)
    print("-" * len(header))

    any_existing_fail = False
    any_new_fail = False
    results: dict[str, list[tuple[float, float, str]]] = {}

    for type_id_enum, label, type_id_int in _PROBE_ENTRIES:
        entry = OBJECT_VOCAB[type_id_int]
        hz_canonical = entry.canonical_half_extents_m[2]
        results[label] = []

        for scale in _SCALES:
            for yaw_deg in _YAWS_DEG:
                scene = _make_single_object_scene(type_id_int, scale, yaw_deg)
                try:
                    stable, trans_m, rot_rad = _run_stable_rest(scene)
                except Exception as exc:
                    print(
                        f"  {label:<18} {hz_canonical:>6.3f} {scale:>6.2f} "
                        f"{yaw_deg:>8.1f} {'ERROR':>10}  {exc!r}"
                    )
                    results[label].append((float("nan"), float("nan"), "ERROR"))
                    continue

                result_str = "PASS" if stable else "FAIL"
                trans_mm = trans_m * 1000
                rot_deg = math.degrees(rot_rad)

                print(
                    f"  {label:<18} {hz_canonical:>6.3f} {scale:>6.2f} "
                    f"{yaw_deg:>8.1f} {trans_mm:>10.3f} {rot_deg:>9.3f}  {result_str}"
                )
                results[label].append((trans_mm, rot_deg, result_str))

                if not stable:
                    is_existing = type_id_int < 8
                    if is_existing:
                        any_existing_fail = True
                    else:
                        any_new_fail = True

        print()  # blank line between entries

    # Summary.
    print("=== Summary ===\n")
    for type_id_enum, label, type_id_int in _PROBE_ENTRIES:
        entry_results = results[label]
        n_pass = sum(1 for _, _, r in entry_results if r == "PASS")
        n_fail = sum(1 for _, _, r in entry_results if r == "FAIL")
        n_err = sum(1 for _, _, r in entry_results if r == "ERROR")
        tag = "[EXISTING ID<8]" if type_id_int < 8 else "[NEW ID>=8]"
        print(f"  {label:<20} {tag}  {n_pass}/{len(entry_results)} pass  {n_fail} fail  {n_err} error")

    print()
    if any_existing_fail:
        print("HALT: At least one existing vocab entry (ID < 8) fails stable_rest in isolation.")
        print("This indicates a deeper problem predating the vocab expansion. Investigate before proceeding.")
    elif any_new_fail:
        failing = [
            label
            for type_id_enum, label, type_id_int in _PROBE_ENTRIES
            if type_id_int >= 8 and any(r == "FAIL" for _, _, r in results[label])
        ]
        print(f"New entries failing in isolation: {failing}")
        print("These will be added to the Option B drop list. Vocab size adjusted accordingly.")
    else:
        print("All entries PASS stable_rest in isolation.")
        print("The stable_rest regression is a multi-object interaction effect, not per-object instability.")


if __name__ == "__main__":
    main()
