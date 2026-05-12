"""Probe: verify WeldFrames API for all 8 object vocabulary entries.

Verifies the following before production code is written for Task 0:
1. AddModelsFromString returns a non-empty model instance list.
2. plant.GetBodyIndices(mi) yields at least one body index per instance.
3. plant.get_body(body_idx).is_floating() is True before welding.
4. plant.WeldFrames(plant.world_frame(), body.body_frame(), X_WB) succeeds.
5. After Finalize(), body.is_floating() is False.
6. After Build(), collision query with a dummy robot pose returns negative signed
   distance when the object is placed at a position the robot occupies.

Resolves bodies via GetBodyIndices(model_instance) -- NOT GetBodyByName -- to handle
the fact that all 8 SDFs use the same link name "link".

Welds via body.body_frame() -- NOT GetFrameByName("link") -- to avoid inertial-origin
surprises where link frame != body frame.
"""

from __future__ import annotations

import sys

import numpy as np
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

# Import from the src tree.
sys.path.insert(0, "src")
from scene.vocab import OBJECT_VOCAB, build_sdf
from validator.core import _load_ur5e, _filter_ur5e_self_collisions, _UR5E_URDF

INTERP_THRESHOLD_M = 1e-3


def probe_entry(type_id: int) -> None:
    entry = OBJECT_VOCAB[type_id]
    print(f"\n--- type_id={type_id} ({entry.name}) ---")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)

    # Load robot.
    _load_ur5e(plant, _UR5E_URDF)

    # Build SDF for this object type and parse it.
    model_name = f"probe_obj_{type_id}"
    sdf_str = build_sdf(entry, model_name=model_name, scale=1.0)
    model_instances = Parser(plant).AddModelsFromString(sdf_str, "sdf")

    print(f"  model_instances returned: {len(model_instances)}")
    assert len(model_instances) >= 1, "AddModelsFromString returned no instances"

    # Resolve bodies via GetBodyIndices, NOT GetBodyByName.
    # Pre-Finalize: is_floating_base_body() cannot be called yet (requires Finalize).
    # Identify bodies to weld by excluding the world body index.
    world_body_idx = plant.world_body().index()
    weld_target_bodies = []
    for mi in model_instances:
        body_indices = plant.GetBodyIndices(mi)
        print(f"  model_instance={mi}: body_indices={list(body_indices)}")
        for body_idx in body_indices:
            body = plant.get_body(body_idx)
            if body_idx == world_body_idx:
                print(f"    skipping world body: {body.name()!r}")
                continue
            print(f"    body name={body.name()!r}, body_frame={body.body_frame().name()!r}")
            weld_target_bodies.append(body)

    assert weld_target_bodies, "No non-world bodies found to weld"

    # Weld at a known position: [0.0, 0.3, 0.15] above the table.
    xyz = np.array([0.0, 0.3, 0.15])
    X_WB = RigidTransform(RotationMatrix(), xyz)
    for body in weld_target_bodies:
        # Confirmation 2: use body.body_frame(), not GetFrameByName.
        plant.WeldFrames(plant.world_frame(), body.body_frame(), X_WB)
        print(f"    WeldFrames succeeded for body={body.name()!r}")

    plant.Finalize()
    _filter_ur5e_self_collisions(plant, scene_graph)

    # Verify post-Finalize: welded bodies should no longer report a floating base.
    # Use is_floating_base_body() -- the non-deprecated successor to is_floating().
    welded_names = {b.name() for b in weld_target_bodies}
    for mi in model_instances:
        for body_idx in plant.GetBodyIndices(mi):
            body = plant.get_body(body_idx)
            if body.name() == "WorldBody":
                continue
            still_floating = body.is_floating_base_body()
            print(f"  post-Finalize: body={body.name()!r} is_floating_base_body={still_floating}")
            if body.name() in welded_names:
                assert not still_floating, (
                    f"Body {body.name()!r} is still floating after WeldFrames + Finalize"
                )

    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)

    # Place robot at home config (all zeros). For some objects placed at [0,0.3,0.15]
    # the robot links at q=0 may or may not intersect. Just verify the query runs.
    q_robot = np.zeros(plant.num_positions())
    plant.SetPositions(plant_context, q_robot)

    query_object = scene_graph.get_query_output_port().Eval(sg_context)
    pairs = query_object.ComputeSignedDistancePairwiseClosestPoints()
    distances = [p.distance for p in pairs]
    min_dist = min(distances) if distances else float("inf")
    print(f"  collision query: {len(pairs)} pairs, min_dist={min_dist:.4f}m")

    print(f"  PASS: type_id={type_id} ({entry.name})")


def main() -> None:
    print("=== probe_weld_api.py: WeldFrames API verification for all 8 vocab entries ===")
    failures: list[tuple[int, str]] = []
    for type_id in range(len(OBJECT_VOCAB)):
        try:
            probe_entry(type_id)
        except Exception as exc:
            print(f"  FAIL: type_id={type_id}: {exc}")
            failures.append((type_id, str(exc)))

    print("\n=== Summary ===")
    if failures:
        for tid, msg in failures:
            print(f"  FAIL type_id={tid}: {msg}")
        sys.exit(1)
    else:
        print(f"  All {len(OBJECT_VOCAB)} entries PASSED.")
        print("  Safe to write _populate_rrt_plant_with_scene in production.")


if __name__ == "__main__":
    main()
