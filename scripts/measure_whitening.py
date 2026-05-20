"""Measure exact per-dim whitening constants from the training dataset.

Run this once before v5 training to get precise std values.
Output is written to artifacts/v5_whitening_constants.json and printed.
Copy the values into:
  - scripts/train.py  _DATA_STD_XYZ, _DATA_STD_SCALE
  - scripts/probe_drake.py  _DATA_STD_XYZ, _DATA_STD_SCALE

Usage:
    cd /workspace/Demiurge && source .venv/bin/activate
    python3 scripts/measure_whitening.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data.reader import ShardReader
from scene.schema import WorkspaceBounds

bounds = WorkspaceBounds.default()
all_xyz: list[list[float]] = []
all_scale: list[list[float]] = []

DATA_PATHS = [
    _ROOT / "data" / "v1",
    _ROOT / "data" / "v2",
]

for path in DATA_PATHS:
    if not path.exists():
        print(f"SKIP {path} (not found)")
        continue
    reader = ShardReader(str(path))
    for i, item in enumerate(reader):
        scene = item[0] if isinstance(item, tuple) else item
        s_norm = scene.normalize(bounds)
        pres = s_norm.presence
        for j in range(len(pres)):
            p = pres[j]
            is_present = bool(p) if isinstance(p, bool) else float(p) > 0.5
            if is_present:
                all_xyz.append(s_norm.poses[j, :3].tolist())
                all_scale.append(s_norm.scales[j].tolist())
        if i >= 4999:
            break
    print(f"Read {path.name}: {len(all_xyz)} objects so far")

xyz_arr   = np.array(all_xyz,   dtype=np.float32)
scale_arr = np.array(all_scale, dtype=np.float32)

xyz_mean  = xyz_arr.mean(axis=0)
xyz_std   = xyz_arr.std(axis=0)
sc_mean   = scale_arr.mean(axis=0)
sc_std    = scale_arr.std(axis=0)

print(f"\n=== Whitening constants ({len(xyz_arr)} objects) ===")
print(f"xyz_mean : {xyz_mean.tolist()}")
print(f"xyz_std  : {xyz_std.tolist()}")
print(f"scale_mean: {sc_mean.tolist()}")
print(f"scale_std : {sc_std.tolist()}")

print(f"\nKey ratio z_std/x_std = {xyz_std[2]/xyz_std[0]:.3f}  (was the covariance mismatch)")

# Verify whitening
xyz_w = (xyz_arr - xyz_mean) / xyz_std
print(f"\nPost-whitening xyz mean: {xyz_w.mean(axis=0).tolist()}")
print(f"Post-whitening xyz std : {xyz_w.std(axis=0).tolist()}")

constants = {
    "xyz_mean":   xyz_mean.tolist(),
    "xyz_std":    xyz_std.tolist(),
    "scale_mean": sc_mean.tolist(),
    "scale_std":  sc_std.tolist(),
    "n_objects":  int(len(xyz_arr)),
}
out = _ROOT / "artifacts" / "v5_whitening_constants.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(constants, indent=2))
print(f"\nSaved to {out}")

print("\n=== Paste into train.py and probe_drake.py ===")
print(f"_DATA_MEAN_XYZ   = torch.tensor({[round(v,6) for v in xyz_mean.tolist()]})")
print(f"_DATA_STD_XYZ    = torch.tensor({[round(v,6) for v in xyz_std.tolist()]})")
print(f"_DATA_MEAN_SCALE = torch.tensor({[round(v,6) for v in sc_mean.tolist()]})")
print(f"_DATA_STD_SCALE  = torch.tensor({[round(v,6) for v in sc_std.tolist()]})")
