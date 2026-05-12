"""Probe: Drake 1.51.1 IK and Simulator APIs.

Tests InverseKinematics constraint API and Simulator.AdvanceTo for the validator.

Run with: uv run python probes/probe_ik_sim_api.py
"""

import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.solvers import Solve
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder


# Minimal UR5e URDF is not bundled. Use a simple 2-DOF arm for probing.
SIMPLE_ARM_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="arm_world">
    <model name="simple_arm">
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
      <link name="link1">
        <pose>0 0 0.1 0 0 0</pose>
        <inertial><mass>0.5</mass>
          <inertia><ixx>0.005</ixx><ixy>0</ixy><ixz>0</ixz>
                   <iyy>0.005</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
        </inertial>
        <collision name="col"><geometry><cylinder><radius>0.02</radius><length>0.2</length></cylinder></geometry></collision>
      </link>
      <joint name="j2" type="revolute">
        <parent>base</parent><child>link1</child>
        <axis><xyz>0 1 0</xyz><limit><lower>-1.57</lower><upper>1.57</upper></limit></axis>
      </joint>
    </model>
  </world>
</sdf>"""


def probe_simulator() -> None:
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    Parser(plant).AddModelsFromString(SIMPLE_ARM_SDF, "sdf")
    plant.Finalize()
    diagram = builder.Build()

    simulator = Simulator(diagram)
    simulator.Initialize()

    context = simulator.get_mutable_context()
    print(f"Simulator: initial time={context.get_time():.3f}s")

    simulator.AdvanceTo(1.5)
    print(f"Simulator: after AdvanceTo(1.5), time={context.get_time():.3f}s")
    print("Simulator.AdvanceTo: OK")


def probe_ik_api() -> None:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    Parser(plant).AddModelsFromString(SIMPLE_ARM_SDF, "sdf")
    plant.Finalize()
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)

    ik = InverseKinematics(plant, plant_context)
    print(f"InverseKinematics constructed: num_vars={ik.prog().num_vars()}")

    # Test AddPositionConstraint.
    ee_frame = plant.GetFrameByName("link1")
    world_frame = plant.world_frame()
    target_pos = np.array([0.0, 0.0, 0.3])
    try:
        ik.AddPositionConstraint(
            frameB=ee_frame,
            p_BQ=np.zeros(3),
            frameA=world_frame,
            p_AQ_lower=target_pos - 0.05,
            p_AQ_upper=target_pos + 0.05,
        )
        print("AddPositionConstraint: OK")
    except Exception as e:
        print(f"AddPositionConstraint: FAILED {e}")

    # Test AddMinimumDistanceLowerBoundConstraint.
    try:
        ik.AddMinimumDistanceLowerBoundConstraint(0.001, 0.1)
        print("AddMinimumDistanceLowerBoundConstraint(min_dist, influence): OK")
    except Exception as e:
        print(f"AddMinimumDistanceLowerBoundConstraint: FAILED: {e}")

    # Solve.
    try:
        q0 = np.zeros(plant.num_positions())
        ik.prog().SetInitialGuess(ik.q(), q0)
        result = Solve(ik.prog())
        print(f"Solve: success={result.is_success()}, cost={result.get_optimal_cost():.4f}")
    except Exception as e:
        print(f"Solve: FAILED {e}")


if __name__ == "__main__":
    print("=== Simulator.AdvanceTo probe ===")
    probe_simulator()
    print()
    print("=== InverseKinematics probe ===")
    probe_ik_api()
