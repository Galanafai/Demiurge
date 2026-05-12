"""Task description generator for Demiurge dataset.

Generates natural-language task descriptions from CandidateScene instances.
Each task family has 4-8 surface-form variants. Qualifiers are drawn from
the scene state to maximize description diversity and reduce exact duplicates.

Qualifiers:
  - Object type name: from OBJECT_VOCAB[type_id].name
  - Color: deterministic per type_id from a fixed palette
  - Spatial qualifier: derived from xy position relative to workspace centre
  - Obstacle count: number of non-target present objects

Duplicate rate target: fewer than 5 percent exact duplicates in any 1000-sample
draw from a single template.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from scene.vocab import OBJECT_VOCAB

if TYPE_CHECKING:
    from data.sampler import CandidateScene


# ---------------------------------------------------------------------------
# Color palette (deterministic per type_id, consistent across descriptions)
# ---------------------------------------------------------------------------

_TYPE_COLOR: dict[int, str] = {
    0: "red",          # cube
    1: "blue",         # sphere
    2: "green",        # cylinder
    3: "yellow",       # box_tall
    4: "orange",       # box_flat
    5: "brown",        # mustard_bottle
    6: "white",        # sugar_box
    7: "silver",       # tomato_soup_can
    # Week 2.5 expansion: 8 new distinct colors for IDs 8-15.
    8:  "teal",        # bleach_cleanser
    9:  "pink",        # banana
    10: "grey",        # master_chef_can
    11: "lime",        # gelatin_box
    12: "purple",      # pudding_box
    13: "beige",       # cracker_box
    14: "navy",        # potted_meat_can
    15: "olive",       # power_drill
}


# ---------------------------------------------------------------------------
# Spatial qualifier from xy position
# ---------------------------------------------------------------------------

_WORKSPACE_CENTER = (0.0, 0.30)  # approximate centre of the reachable workspace


def _spatial_qualifier(xy: tuple[float, float]) -> str:
    """Return a spatial description derived from a 12-zone grid (3x4).

    Three x-bands (left / center / right) cross four y-bands (near / mid-near /
    mid-far / far) producing 12 distinct qualifier strings. This gives enough
    resolution to make descriptions unique across typical workspace distributions
    without requiring floating-point position literals.
    """
    x, y = xy

    # Three x-bands: left x < -0.12, right x > 0.12, else center.
    if x < -0.12:
        x_label = "to the left"
    elif x > 0.12:
        x_label = "to the right"
    else:
        x_label = "centrally"

    # Four y-bands from near robot to far edge.
    if y < 0.15:
        y_label = "close to the robot"
    elif y < 0.25:
        y_label = "in the near zone"
    elif y < 0.35:
        y_label = "in the mid zone"
    else:
        y_label = "near the far edge"

    return f"{x_label}, {y_label}"

def _distance_tag(xy: tuple[float, float]) -> str:
    """Return a position tag encoding distance and x-offset at 2cm resolution.

    With 2cm steps the tag covers ~32 distance values and ~45 x-offset values,
    yielding a large unique-combo space even within a single object type.
    """
    x, y = xy
    dist = (x ** 2 + y ** 2) ** 0.5
    dist_cm = round(dist / 0.02) * 2
    x_cm = round(x / 0.02) * 2
    if x_cm == 0:
        return f"{dist_cm}cm from the base"
    side = "left" if x_cm < 0 else "right"
    return f"{dist_cm}cm out, {abs(x_cm)}cm to the {side}"


# ---------------------------------------------------------------------------
# Variant banks per template family
# ---------------------------------------------------------------------------

def _object_label(type_id: int) -> tuple[str, str]:
    """Return (color, type_name) for an object type id."""
    color = _TYPE_COLOR.get(type_id, "grey")
    name = OBJECT_VOCAB[type_id].name.replace("_", " ")
    return color, name


def _tabletop_reach_variants(
    color: str,
    obj_name: str,
    spatial: str,
    dist_tag: str,
    rng: np.random.Generator,
) -> str:
    variants = [
        f"Pick up the {color} {obj_name} {dist_tag}.",
        f"Reach the {obj_name} {spatial} on the table.",
        f"Grasp the {color} {obj_name} {dist_tag} and move it to the drop zone.",
        f"Move the {obj_name} from its position {spatial}.",
        f"Pick the {color} {obj_name} {spatial} off the table.",
        f"Retrieve the {obj_name} {dist_tag}.",
        f"Fetch the {color} {obj_name} that is {spatial}.",
        f"Grab the {obj_name} {spatial} ({dist_tag}) off the table surface.",
    ]
    return str(variants[int(rng.integers(0, len(variants)))])


def _cluttered_pick_variants(
    color: str,
    obj_name: str,
    spatial: str,
    dist_tag: str,
    n_obstacles: int,
    rng: np.random.Generator,
) -> str:
    obs_phrase = (
        "the surrounding objects"
        if n_obstacles > 1
        else "the nearby object"
    )
    variants = [
        f"Pick the {color} {obj_name} from among {n_obstacles} other objects.",
        f"Extract the {color} {obj_name} {spatial} without disturbing {obs_phrase}.",
        f"Retrieve the {obj_name} {dist_tag} in the cluttered area.",
        f"Grasp the {color} {obj_name} carefully, avoiding {obs_phrase}.",
        f"Pick up the {color} {obj_name} from the cluttered workspace {spatial}.",
        f"Remove the {obj_name} from among {n_obstacles} surrounding items ({dist_tag}).",
    ]
    return str(variants[int(rng.integers(0, len(variants)))])


def _obstacle_avoidance_variants(
    color: str,
    obj_name: str,
    spatial: str,
    dist_tag: str,
    n_obstacles: int,
    rng: np.random.Generator,
) -> str:
    bar_phrase = (
        f"{n_obstacles} obstacles" if n_obstacles > 1 else "an obstacle"
    )
    variants = [
        f"Reach the {color} {obj_name} beyond {bar_phrase} ({dist_tag}).",
        f"Navigate around {bar_phrase} and pick the {obj_name} {spatial}.",
        f"Pick the {color} {obj_name} on the far side of {bar_phrase}.",
        f"Move through the obstacle row and grasp the {obj_name} {spatial} ({dist_tag}).",
        f"Retrieve the {color} {obj_name} behind {bar_phrase}.",
        f"Avoid {bar_phrase} and fetch the {obj_name} {dist_tag}.",
        f"Reach across {bar_phrase} to pick the {color} {obj_name} {spatial}.",
    ]
    return str(variants[int(rng.integers(0, len(variants)))])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_description(
    candidate: CandidateScene,
    rng: np.random.Generator,
) -> str:
    """Generate a task description for a CandidateScene.

    Args:
        candidate: The scene and metadata from a TaskTemplate.
        rng: Numpy random generator for variant selection.

    Returns:
        Non-empty description string under 200 characters.

    Raises:
        ValueError: If candidate.task_family is not a recognised template name.
    """
    scene = candidate.scene
    target_idx = candidate.target_idx

    # Resolve target object properties.
    type_id = int(scene.object_types[target_idx].item())
    color, obj_name = _object_label(type_id)
    xy = (
        float(scene.poses[target_idx, 0].item()),
        float(scene.poses[target_idx, 1].item()),
    )
    spatial = _spatial_qualifier(xy)
    dist_tag = _distance_tag(xy)

    # Count non-target present objects.
    n_present = int(scene.presence.sum().item())
    n_obstacles = max(0, n_present - 1)

    family = candidate.task_family
    if family == "tabletop_reach":
        desc = _tabletop_reach_variants(color, obj_name, spatial, dist_tag, rng)
    elif family == "cluttered_pick":
        desc = _cluttered_pick_variants(color, obj_name, spatial, dist_tag, n_obstacles, rng)
    elif family == "obstacle_avoidance":
        desc = _obstacle_avoidance_variants(color, obj_name, spatial, dist_tag, n_obstacles, rng)
    else:
        raise ValueError(
            f"Unknown task family {family!r}. "
            "Expected one of: tabletop_reach, cluttered_pick, obstacle_avoidance."
        )

    if not desc:
        raise RuntimeError(f"generate_description produced empty string for family={family!r}")
    if len(desc) > 200:
        raise RuntimeError(
            f"generate_description produced string over 200 chars ({len(desc)}): {desc!r}"
        )
    return desc
