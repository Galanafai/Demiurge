"""Probe: discover Drake's bundled YCB SDF resource paths for Python 3.11 / drake 1.51.1.

Run with: uv run python probes/probe_ycb_paths.py
"""

import pydrake.common
from pydrake.common import FindResourceOrThrow


def main() -> None:
    candidates = [
        # Standard drake YCB naming patterns
        "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf",
        "drake/manipulation/models/ycb/sdf/004_sugar_box.sdf",
        "drake/manipulation/models/ycb/sdf/005_tomato_soup_can.sdf",
        # Alternative paths seen in some Drake versions
        "drake/manipulation/models/ycb/006_mustard_bottle/006_mustard_bottle.sdf",
        "drake/manipulation/models/ycb/004_sugar_box/004_sugar_box.sdf",
        "drake/manipulation/models/ycb/005_tomato_soup_can/005_tomato_soup_can.sdf",
    ]
    print(f"Drake version check: pydrake loaded from {pydrake.common.__file__}")
    for path in candidates:
        try:
            resolved = FindResourceOrThrow(path)
            print(f"  FOUND  {path}")
            print(f"         -> {resolved}")
        except Exception as e:
            print(f"  MISS   {path}  ({e})")


if __name__ == "__main__":
    main()
