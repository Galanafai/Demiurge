"""Probe: test multi-model SDF loading strategies in Drake 1.51.1.

Tests:
1. Calling AddModelsFromString once per model (should always work).
2. Using a <world> wrapper to load all models in one call.

Run with: uv run python probes/probe_multimodel_sdf.py
"""

from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder


BOX_INNER = """\
  <model name="box_a">
    <link name="link">
      <inertial><mass>0.5</mass>
        <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
      </inertial>
      <collision name="col"><geometry><box><size>0.05 0.05 0.05</size></box></geometry></collision>
    </link>
  </model>"""

SPHERE_INNER = """\
  <model name="sphere_b">
    <link name="link">
      <inertial><mass>0.3</mass>
        <inertia><ixx>0.001</ixx><ixy>0</ixy><ixz>0</ixz>
                 <iyy>0.001</iyy><iyz>0</iyz><izz>0.001</izz></inertia>
      </inertial>
      <collision name="col"><geometry><sphere><radius>0.03</radius></sphere></geometry></collision>
    </link>
  </model>"""

WORLD_SDF = f"""\
<?xml version="1.0"?>
<sdf version="1.7">
  <world name="scene">
{BOX_INNER}
{SPHERE_INNER}
  </world>
</sdf>"""


def test_one_per_call() -> None:
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    for inner, name in [(BOX_INNER, "box_a"), (SPHERE_INNER, "sphere_b")]:
        sdf = f'<?xml version="1.0"?>\n<sdf version="1.7">\n{inner}\n</sdf>'
        parser.AddModelsFromString(sdf, "sdf")
    plant.Finalize()
    print(f"one_per_call: num_bodies={plant.num_bodies()} (expected 3: world+2 links)")


def test_world_wrapper() -> None:
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    try:
        parser.AddModelsFromString(WORLD_SDF, "sdf")
        plant.Finalize()
        print(f"world_wrapper: num_bodies={plant.num_bodies()} (expected 3: world+2 links)")
    except Exception as e:
        print(f"world_wrapper: FAILED with {e}")


if __name__ == "__main__":
    test_one_per_call()
    test_world_wrapper()
