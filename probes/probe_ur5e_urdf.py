"""Probe: verify UR5e URDF loads correctly in Drake and confirm num_positions=6.

Also probes GetFrameByName for the TCP frame (tool0) used in IK.

Run with: uv run python probes/probe_ur5e_urdf.py
"""

import pathlib

import numpy as np
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

URDF_PATH = pathlib.Path(__file__).parent.parent / "assets" / "models" / "ur5e" / "ur5e.urdf"


def main() -> None:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.AddModels(str(URDF_PATH))
    plant.Finalize()

    print(f"Loaded: {URDF_PATH.name}")
    print(f"  num_positions:         {plant.num_positions()}")
    print(f"  num_velocities:        {plant.num_velocities()}")
    print(f"  num_actuators:         {plant.num_actuators()}")
    print(f"  num_bodies:            {plant.num_bodies()}")

    # Check TCP frame.
    try:
        tool0 = plant.GetFrameByName("tool0")
        print(f"  tool0 frame:           {tool0.name()}")
    except Exception as e:
        print(f"  tool0 frame:           NOT FOUND ({e})")

    # Compute FK at home position (all zeros).
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(context)
    plant.SetPositions(plant_context, np.zeros(plant.num_positions()))

    tool0_frame = plant.GetFrameByName("tool0")
    world_frame = plant.world_frame()
    X_WT = plant.CalcRelativeTransform(plant_context, world_frame, tool0_frame)
    pos = X_WT.translation()
    print(f"  TCP at home (m):       [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    print("  (expected near z=0.91 for UR5e at all-zero config)")


if __name__ == "__main__":
    main()
