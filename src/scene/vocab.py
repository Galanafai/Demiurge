"""Object vocabulary for the Demiurge scene schema.

Defines OBJECT_VOCAB: a fixed 16-entry registry mapping integer type IDs to ObjectEntry
instances. All SDFs are inline geometry strings. The PyPI drake wheel does not ship YCB
mesh assets, so YCB-keyed entries (IDs 5-15) use primitive approximations with dimensions
matched to the real YCB objects. This is sufficient for Drake collision and IK checks.

Do not add entries beyond ID 15 without explicit approval (AGENTS.md: fixed vocabulary).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ObjectTypeId(IntEnum):
    """Canonical integer IDs for each object type in the vocabulary."""

    CUBE = 0
    SPHERE = 1
    CYLINDER = 2
    BOX_TALL = 3
    BOX_FLAT = 4
    MUSTARD_BOTTLE = 5   # YCB 006: approximated as cylinder r=0.03 h=0.19
    SUGAR_BOX = 6        # YCB 004: approximated as box 0.038 x 0.086 x 0.175
    TOMATO_SOUP_CAN = 7  # YCB 005: approximated as cylinder r=0.033 h=0.102
    # Week 2.5 expansion: 8 new YCB-keyed entries (inline primitives; mesh SDFs not
    # shipped in the PyPI drake wheel).
    BLEACH_CLEANSER = 8   # YCB 021: cylinder r=0.040 h=0.250
    BANANA = 9            # YCB 011: box 0.090 x 0.040 x 0.180 (oriented lengthwise)
    MASTER_CHEF_CAN = 10  # YCB 002: cylinder r=0.052 h=0.142
    GELATIN_BOX = 11      # YCB 009: box 0.056 x 0.112 x 0.166
    PUDDING_BOX = 12      # YCB 008: box 0.124 x 0.166 x 0.044
    CRACKER_BOX = 13      # YCB 003: box 0.120 x 0.316 x 0.042
    POTTED_MEAT_CAN = 14  # YCB 010: cylinder r=0.043 h=0.110
    POWER_DRILL = 15      # YCB 035: box 0.186 x 0.094 x 0.350


@dataclass(frozen=True)
class ObjectEntry:
    """Metadata for one object type in the vocabulary.

    Attributes:
        name: Human-readable identifier.
        bounding_radius_m: Radius of the bounding sphere in metres, used for
            proximity queries and scale clamping.
        canonical_half_extents_m: (x, y, z) half-extents of the tightest
            enclosing axis-aligned box at scale=1. Used to compute inertia
            tensors and to build the SDF.
        sdf_template: SDF XML string with format placeholders:
            {model_name}, {mass}, {sx}, {sy}, {sz} for box,
            {model_name}, {mass}, {radius}, {length} for cylinder,
            {model_name}, {mass}, {radius} for sphere.
        sdf_kind: One of "box", "sphere", "cylinder". Controls which template
            placeholders are active.
    """

    name: str
    bounding_radius_m: float
    canonical_half_extents_m: tuple[float, float, float]
    sdf_template: str
    sdf_kind: str


# ---------------------------------------------------------------------------
# SDF templates
# ---------------------------------------------------------------------------

_BOX_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy}</iyy><iyz>0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </collision>
      <visual name="vis">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
      </visual>
    </link>
  </model>
</sdf>"""

_SPHERE_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy}</iyy><iyz>0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </collision>
      <visual name="vis">
        <geometry><sphere><radius>{radius}</radius></sphere></geometry>
      </visual>
    </link>
  </model>
</sdf>"""

_CYLINDER_SDF = """\
<?xml version="1.0"?>
<sdf version="1.7">
  <model name="{model_name}">
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz>
          <iyy>{iyy}</iyy><iyz>0</iyz>
          <izz>{izz}</izz>
        </inertia>
      </inertial>
      <collision name="col">
        <geometry><cylinder>
          <radius>{radius}</radius><length>{length}</length>
        </cylinder></geometry>
      </collision>
      <visual name="vis">
        <geometry><cylinder>
          <radius>{radius}</radius><length>{length}</length>
        </cylinder></geometry>
      </visual>
    </link>
  </model>
</sdf>"""

# ---------------------------------------------------------------------------
# Vocabulary entries
# ---------------------------------------------------------------------------

OBJECT_VOCAB: dict[int, ObjectEntry] = {
    ObjectTypeId.CUBE: ObjectEntry(
        name="cube",
        bounding_radius_m=0.0433,  # half-diagonal of 0.05m cube
        canonical_half_extents_m=(0.025, 0.025, 0.025),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.SPHERE: ObjectEntry(
        name="sphere",
        bounding_radius_m=0.030,
        canonical_half_extents_m=(0.030, 0.030, 0.030),
        sdf_template=_SPHERE_SDF,
        sdf_kind="sphere",
    ),
    ObjectTypeId.CYLINDER: ObjectEntry(
        name="cylinder",
        bounding_radius_m=0.053,  # sqrt(r^2 + (h/2)^2) for r=0.025, h=0.09
        canonical_half_extents_m=(0.025, 0.025, 0.045),
        sdf_template=_CYLINDER_SDF,
        sdf_kind="cylinder",
    ),
    ObjectTypeId.BOX_TALL: ObjectEntry(
        name="box_tall",
        bounding_radius_m=0.098,
        canonical_half_extents_m=(0.025, 0.025, 0.090),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.BOX_FLAT: ObjectEntry(
        name="box_flat",
        bounding_radius_m=0.056,
        canonical_half_extents_m=(0.050, 0.050, 0.015),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    # YCB-keyed entries: inline primitive approximations.
    # Real YCB meshes are not shipped in the drake PyPI wheel.
    ObjectTypeId.MUSTARD_BOTTLE: ObjectEntry(
        name="mustard_bottle",
        bounding_radius_m=0.098,  # sqrt(0.03^2 + 0.095^2)
        canonical_half_extents_m=(0.030, 0.030, 0.095),
        sdf_template=_CYLINDER_SDF,
        sdf_kind="cylinder",
    ),
    ObjectTypeId.SUGAR_BOX: ObjectEntry(
        name="sugar_box",
        bounding_radius_m=0.102,  # half-diagonal of 0.038 x 0.086 x 0.175
        canonical_half_extents_m=(0.019, 0.043, 0.0875),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.TOMATO_SOUP_CAN: ObjectEntry(
        name="tomato_soup_can",
        bounding_radius_m=0.062,  # sqrt(0.033^2 + 0.051^2)
        canonical_half_extents_m=(0.033, 0.033, 0.051),
        sdf_template=_CYLINDER_SDF,
        sdf_kind="cylinder",
    ),
    # -------------------------------------------------------------------------
    # Week 2.5 additions: YCB-keyed inline primitives.
    # Dimensions sourced from YCB dataset bounding-box measurements.
    # Bounding radius = sqrt(hx^2+hy^2+hz^2) for box, sqrt(r^2+(h/2)^2) for cylinder.
    # -------------------------------------------------------------------------
    ObjectTypeId.BLEACH_CLEANSER: ObjectEntry(
        name="bleach_cleanser",
        bounding_radius_m=0.132,  # sqrt(0.040^2 + 0.125^2)
        canonical_half_extents_m=(0.040, 0.040, 0.125),
        sdf_template=_CYLINDER_SDF,
        sdf_kind="cylinder",
    ),
    ObjectTypeId.BANANA: ObjectEntry(
        name="banana",
        bounding_radius_m=0.103,  # sqrt(0.045^2 + 0.020^2 + 0.090^2) = 0.1026
        canonical_half_extents_m=(0.045, 0.020, 0.090),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.MASTER_CHEF_CAN: ObjectEntry(
        name="master_chef_can",
        bounding_radius_m=0.088,  # sqrt(0.052^2 + 0.071^2)
        canonical_half_extents_m=(0.052, 0.052, 0.071),
        sdf_template=_CYLINDER_SDF,
        sdf_kind="cylinder",
    ),
    ObjectTypeId.GELATIN_BOX: ObjectEntry(
        name="gelatin_box",
        bounding_radius_m=0.104,  # sqrt(0.028^2 + 0.056^2 + 0.083^2)
        canonical_half_extents_m=(0.028, 0.056, 0.083),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.PUDDING_BOX: ObjectEntry(
        name="pudding_box",
        bounding_radius_m=0.106,  # sqrt(0.062^2 + 0.083^2 + 0.022^2)
        canonical_half_extents_m=(0.062, 0.083, 0.022),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.CRACKER_BOX: ObjectEntry(
        name="cracker_box",
        bounding_radius_m=0.170,  # sqrt(0.060^2 + 0.158^2 + 0.021^2) = 0.1703
        canonical_half_extents_m=(0.060, 0.158, 0.021),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
    ObjectTypeId.POTTED_MEAT_CAN: ObjectEntry(
        name="potted_meat_can",
        bounding_radius_m=0.070,  # sqrt(0.043^2 + 0.055^2)
        canonical_half_extents_m=(0.043, 0.043, 0.055),
        sdf_template=_CYLINDER_SDF,
        sdf_kind="cylinder",
    ),
    ObjectTypeId.POWER_DRILL: ObjectEntry(
        name="power_drill",
        bounding_radius_m=0.204,  # sqrt(0.093^2 + 0.047^2 + 0.175^2) = 0.2037
        canonical_half_extents_m=(0.093, 0.047, 0.175),
        sdf_template=_BOX_SDF,
        sdf_kind="box",
    ),
}

assert len(OBJECT_VOCAB) == 16, "Vocabulary must have exactly 16 entries."


def build_sdf(entry: ObjectEntry, model_name: str, scale: float = 1.0, mass: float = 0.5) -> str:
    """Render an ObjectEntry SDF template at the given uniform scale and mass.

    Inertia tensors are recomputed from the scaled geometry so that Drake's
    simulator can numerically integrate the scene without poorly conditioned
    dynamics.

    Args:
        entry: The ObjectEntry from OBJECT_VOCAB.
        model_name: Unique model name used as the SDF <model name="...">.
        scale: Uniform scale factor applied to all geometry dimensions.
        mass: Mass of the rigid body in kg.

    Returns:
        Rendered SDF XML string ready for Parser.AddModelsFromString.
    """
    hx, hy, hz = (v * scale for v in entry.canonical_half_extents_m)
    kind = entry.sdf_kind

    if kind == "box":
        sx, sy, sz = 2 * hx, 2 * hy, 2 * hz
        ixx = (1.0 / 12.0) * mass * (sy**2 + sz**2)
        iyy = (1.0 / 12.0) * mass * (sx**2 + sz**2)
        izz = (1.0 / 12.0) * mass * (sx**2 + sy**2)
        return entry.sdf_template.format(
            model_name=model_name,
            mass=mass,
            sx=sx,
            sy=sy,
            sz=sz,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
        )
    elif kind == "sphere":
        r = hx  # canonical_half_extents_m are equal for sphere
        ixx = iyy = izz = (2.0 / 5.0) * mass * r**2
        return entry.sdf_template.format(
            model_name=model_name,
            mass=mass,
            radius=r,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
        )
    elif kind == "cylinder":
        r = hx  # hx == hy for cylinder
        length = 2 * hz
        ixx = iyy = (1.0 / 12.0) * mass * (3 * r**2 + length**2)
        izz = 0.5 * mass * r**2
        return entry.sdf_template.format(
            model_name=model_name,
            mass=mass,
            radius=r,
            length=length,
            ixx=ixx,
            iyy=iyy,
            izz=izz,
        )
    else:
        raise ValueError(f"Unknown sdf_kind '{kind}' for entry '{entry.name}'.")
