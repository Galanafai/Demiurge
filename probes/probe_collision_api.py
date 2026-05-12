"""Probe: Drake 1.51.1 collision query API.

Tests ComputeSignedDistancePairwiseClosestPoints to discover exact call signatures.

Run with: uv run python probes/probe_collision_api.py
"""

import numpy as np
from pydrake.geometry import SceneGraph
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.framework import DiagramBuilder

BOX_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="test">
    <model name="box_a">
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <inertial><mass>0.5</mass>
          <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                   <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
        </inertial>
        <collision name="col"><geometry><box><size>0.05 0.05 0.05</size></box></geometry></collision>
      </link>
    </model>
    <model name="box_b">
      <pose>0.1 0 0 0 0 0</pose>
      <link name="link">
        <inertial><mass>0.5</mass>
          <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                   <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
        </inertial>
        <collision name="col"><geometry><box><size>0.05 0.05 0.05</size></box></geometry></collision>
      </link>
    </model>
  </world>
</sdf>"""

OVERLAP_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="test">
    <model name="box_a">
      <pose>0 0 0 0 0 0</pose>
      <link name="link">
        <inertial><mass>0.5</mass>
          <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                   <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
        </inertial>
        <collision name="col"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision>
      </link>
    </model>
    <model name="box_b">
      <pose>0.02 0 0 0 0 0</pose>
      <link name="link">
        <inertial><mass>0.5</mass>
          <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                   <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
        </inertial>
        <collision name="col"><geometry><box><size>0.1 0.1 0.1</size></box></geometry></collision>
      </link>
    </model>
  </world>
</sdf>"""


def build_plant_and_context(sdf: str) -> tuple[MultibodyPlant, object, SceneGraph]:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.AddModelsFromString(sdf, "sdf")
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)
    return plant, plant_context, scene_graph, sg_context  # type: ignore[return-value]


def test_signed_distance(label: str, sdf: str) -> None:
    plant, plant_ctx, scene_graph, sg_ctx = build_plant_and_context(sdf)
    query_object = scene_graph.get_query_output_port().Eval(sg_ctx)

    try:
        # Signature 1: with max_distance kwarg
        pairs = query_object.ComputeSignedDistancePairwiseClosestPoints(max_distance=10.0)
        dists = [p.distance for p in pairs]
        print(f"{label} [max_distance kwarg]: {len(pairs)} pairs, min_dist={min(dists, default=float('inf')):.4f}m")
    except Exception as e:
        print(f"{label} [max_distance kwarg]: FAILED {e}")

    try:
        # Signature 2: positional arg
        pairs2 = query_object.ComputeSignedDistancePairwiseClosestPoints(10.0)
        dists2 = [p.distance for p in pairs2]
        print(f"{label} [positional]:  {len(pairs2)} pairs, min_dist={min(dists2, default=float('inf')):.4f}m")
    except Exception as e:
        print(f"{label} [positional]:  FAILED {e}")

    try:
        # Signature 3: no args (no max distance filter)
        pairs3 = query_object.ComputeSignedDistancePairwiseClosestPoints()
        dists3 = [p.distance for p in pairs3]
        print(f"{label} [no args]:     {len(pairs3)} pairs, min_dist={min(dists3, default=float('inf')):.4f}m")
    except Exception as e:
        print(f"{label} [no args]:     FAILED {e}")


if __name__ == "__main__":
    print("=== Separated boxes (expect positive min_dist) ===")
    test_signed_distance("separated", BOX_SDF)
    print()
    print("=== Overlapping boxes (expect negative min_dist) ===")
    test_signed_distance("overlapping", OVERLAP_SDF)
