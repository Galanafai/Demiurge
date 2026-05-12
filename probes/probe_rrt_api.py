"""Probe: Drake 1.51.1 motion planning API availability.

Tests whether KinematicTrajectoryOptimization and GcsTrajectoryOptimization are
available, and probes the manual sampling approach for BiRRT.

Run with: uv run python probes/probe_rrt_api.py
"""

import numpy as np


def probe_trajectory_optimization() -> None:
    try:
        from pydrake.planning import KinematicTrajectoryOptimization
        print("KinematicTrajectoryOptimization: AVAILABLE")
    except ImportError as e:
        print(f"KinematicTrajectoryOptimization: NOT AVAILABLE ({e})")

    try:
        from pydrake.planning import GcsTrajectoryOptimization
        print("GcsTrajectoryOptimization: AVAILABLE")
    except ImportError as e:
        print(f"GcsTrajectoryOptimization: NOT AVAILABLE ({e})")

    try:
        from pydrake.planning import RobotDiagramBuilder
        print("RobotDiagramBuilder: AVAILABLE")
    except ImportError as e:
        print(f"RobotDiagramBuilder: NOT AVAILABLE ({e})")

    try:
        import pydrake.planning as planning_mod
        print(f"pydrake.planning contents: {[x for x in dir(planning_mod) if not x.startswith('_')]}")
    except ImportError as e:
        print(f"pydrake.planning: NOT AVAILABLE ({e})")


def probe_collision_checker() -> None:
    try:
        from pydrake.planning import CollisionChecker, SceneGraphCollisionChecker
        print("CollisionChecker: AVAILABLE")
        print("SceneGraphCollisionChecker: AVAILABLE")
    except ImportError as e:
        print(f"CollisionChecker/SceneGraphCollisionChecker: NOT AVAILABLE ({e})")


def probe_manual_rrt_prerequisites() -> None:
    """Check that the primitives needed for manual BiRRT are accessible."""
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
    from pydrake.systems.framework import DiagramBuilder

    SIMPLE_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="w">
    <model name="arm">
      <link name="base">
        <inertial><mass>1.0</mass>
          <inertia><ixx>0.01</ixx><ixy>0</ixy><ixz>0</ixz>
                   <iyy>0.01</iyy><iyz>0</iyz><izz>0.01</izz></inertia>
        </inertial>
      </link>
      <joint name="j1" type="revolute">
        <parent>world</parent><child>base</child>
        <axis><xyz>0 0 1</xyz><limit><lower>-3.14</lower><upper>3.14</upper></limit></axis>
      </joint>
    </model>
  </world>
</sdf>"""

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelsFromString(SIMPLE_SDF, "sdf")
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    sg_context = scene_graph.GetMyContextFromRoot(context)

    # SetPositions.
    plant.SetPositions(plant_context, np.zeros(plant.num_positions()))
    print(f"SetPositions: OK, num_positions={plant.num_positions()}")

    # Query object for collision checking.
    query_object = scene_graph.get_query_output_port().Eval(sg_context)
    print(f"QueryObject: OK, type={type(query_object).__name__}")

    # HasCollisions / signed distance for collision-free check.
    try:
        pairs = query_object.ComputeSignedDistancePairwiseClosestPoints()
        print(f"ComputeSignedDistancePairwiseClosestPoints: OK, {len(pairs)} pairs")
    except Exception as e:
        print(f"ComputeSignedDistancePairwiseClosestPoints: FAILED {e}")

    print("Manual BiRRT prerequisites: all accessible")


if __name__ == "__main__":
    print("=== Trajectory optimization APIs ===")
    probe_trajectory_optimization()
    print()
    print("=== CollisionChecker APIs ===")
    probe_collision_checker()
    print()
    print("=== Manual BiRRT prerequisites ===")
    probe_manual_rrt_prerequisites()
