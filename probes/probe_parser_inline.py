"""Probe: verify Drake Parser accepts inline SDF strings for box, sphere, cylinder primitives.

Also probes: MultibodyPlant + SceneGraph setup pattern used throughout the validator.
Run with: uv run python probes/probe_parser_inline.py
"""

from pydrake.geometry import SceneGraph
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.systems.framework import DiagramBuilder


# Minimal inline SDF for a box primitive.
BOX_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test_box">
    <link name="box_link">
      <inertial>
        <mass>0.5</mass>
        <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
      </inertial>
      <collision name="box_col">
        <geometry><box><size>0.05 0.05 0.05</size></box></geometry>
      </collision>
      <visual name="box_vis">
        <geometry><box><size>0.05 0.05 0.05</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>"""

SPHERE_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test_sphere">
    <link name="sphere_link">
      <inertial>
        <mass>0.3</mass>
        <inertia><ixx>0.0005</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>0.0005</iyy><iyz>0</iyz><izz>0.0005</izz></inertia>
      </inertial>
      <collision name="sphere_col">
        <geometry><sphere><radius>0.03</radius></sphere></geometry>
      </collision>
      <visual name="sphere_vis">
        <geometry><sphere><radius>0.03</radius></sphere></geometry>
      </visual>
    </link>
  </model>
</sdf>"""

CYLINDER_SDF = """<?xml version="1.0"?>
<sdf version="1.7">
  <model name="test_cylinder">
    <link name="cyl_link">
      <inertial>
        <mass>0.4</mass>
        <inertia><ixx>0.0008</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>0.0008</iyy><iyz>0</iyz><izz>0.0004</izz></inertia>
      </inertial>
      <collision name="cyl_col">
        <geometry><cylinder><radius>0.025</radius><length>0.1</length></cylinder></geometry>
      </collision>
      <visual name="cyl_vis">
        <geometry><cylinder><radius>0.025</radius><length>0.1</length></cylinder></geometry>
      </visual>
    </link>
  </model>
</sdf>"""


def parse_sdf(sdf_string: str, model_name: str) -> None:
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    parser.AddModelsFromString(sdf_string, "sdf")
    plant.Finalize()
    print(f"  OK   {model_name}: num_bodies={plant.num_bodies()}, num_joints={plant.num_joints()}")


def main() -> None:
    print("Testing inline SDF parsing:")
    parse_sdf(BOX_SDF, "box")
    parse_sdf(SPHERE_SDF, "sphere")
    parse_sdf(CYLINDER_SDF, "cylinder")
    print("All inline SDF round-trips passed.")


if __name__ == "__main__":
    main()
