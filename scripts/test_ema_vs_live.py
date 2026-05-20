"""Compare EMA vs live weights at peak (20k) and collapsed (100k) steps.

Determines whether v4 collapse is in live weights (architectural failure)
or EMA averaging in bad steps (recoverable without full retrain).

Usage:
    python3 scripts/test_ema_vs_live.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np

from model.denoiser import DenoiserConfig, SceneDenoiser
from model.schedule import CosineSchedule, DDIMSampler
from model.rotations import rot6d_to_quat_wxyz
from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
from validator.core import SceneValidator

DEV = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")

bounds = WorkspaceBounds.default()
validator = SceneValidator(rrt_budget_s=1.0)

# Zero-centering offsets -- must match train.py _DATA_NORM_MEAN_*
_MXY = torch.tensor([-0.049061, +0.380961, -0.753032], device=DEV)
_MSC = torch.tensor([-0.040733, -0.040733, -0.040733], device=DEV)

BASE = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "v9_uncond_v4")


def probe(path: str, n: int = 50, ema: bool = True) -> int:
    ckpt = torch.load(path, map_location=DEV, weights_only=False)
    arch = ckpt["arch"]
    cfg = DenoiserConfig(
        n_layers=arch["n_layers"],
        d_model=arch["d_model"],
        n_heads=arch["n_heads"],
        ffn_mult=arch["ffn_mult"],
        dropout=0.0,
        use_type_grad_isolation=False,
        use_slot_id_embed=arch.get("use_slot_id_embed", True),
    )
    mdl = SceneDenoiser(cfg).to(DEV)
    key = "ema_state" if ema else "model_state"
    mdl.load_state_dict(ckpt[key])
    mdl.eval()

    with torch.no_grad():
        x0, tids = DDIMSampler(CosineSchedule(1000), n_steps=50).sample_with_types(
            mdl.conditional_sampling_fn(None),
            (n, N_MAX, 13),
            seed=42,
            device=DEV,
            type_init="uniform",
        )

    xyz   = (x0[:, :, :3]   + _MXY).clamp(-1, 1)
    scale = (x0[:, :, 9:12] + _MSC).clamp(-1, 1)
    rot6d    = x0[:, :, 3:9]
    pres_bit = x0[:, :, 12]

    ok, interp, z_phys, z_raw = 0, 0, [], []
    for i in range(n):
        pres  = pres_bit[i] > -0.589
        quats = rot6d_to_quat_wxyz(rot6d[i])
        st = SceneTensor(
            object_types=tids[i].cpu(),
            poses=torch.cat([xyz[i], quats], -1).cpu(),
            scales=scale[i].cpu(),
            presence=pres.cpu(),
        ).denormalize(bounds)
        for j in range(N_MAX):
            if pres[j]:
                z_phys.append(float(st.poses[j, 2]))
                z_raw.append(float(x0[i, j, 2]))
        try:
            r = validator.validate(st, rrt_seed=i)
            if r.accepted:
                ok += 1
            if not r.no_interpenetration:
                interp += 1
        except Exception:
            pass

    tag = "EMA " if ema else "LIVE"
    z_p = np.mean(z_phys) if z_phys else 0.0
    z_r = np.mean(z_raw)  if z_raw  else 0.0
    print(
        f"  {tag}: {ok:2d}/{n} ({100 * ok / n:4.1f}%)  "
        f"interp={interp:2d}  z_phys={z_p:+.4f}m  z_norm_raw={z_r:+.4f}"
    )
    del mdl
    torch.cuda.empty_cache()
    return ok


print("\n=== Step 20k  (PEAK) ===")
ema_20  = probe(f"{BASE}/step_00020000.pt", ema=True)
live_20 = probe(f"{BASE}/step_00020000.pt", ema=False)

print("\n=== Step 40k  (LAST WITH VALIDITY) ===")
ema_40  = probe(f"{BASE}/step_00040000.pt", ema=True)
live_40 = probe(f"{BASE}/step_00040000.pt", ema=False)

print("\n=== Step 100k (FULLY COLLAPSED) ===")
ema_100  = probe(f"{BASE}/step_00100000.pt", ema=True)
live_100 = probe(f"{BASE}/step_00100000.pt", ema=False)

print("\n=== RESULTS ===")
print(f"  Step  20k:  EMA={ema_20:2d}  LIVE={live_20:2d}")
print(f"  Step  40k:  EMA={ema_40:2d}  LIVE={live_40:2d}")
print(f"  Step 100k:  EMA={ema_100:2d}  LIVE={live_100:2d}")

print("\n=== INTERPRETATION ===")
if live_100 > ema_100 + 2:
    print("CONCLUSION: EMA was dragging down live weights.")
    print("  => No v5 needed. Identify best LIVE checkpoint and use it.")
    best_live = max([(live_20, 20000), (live_40, 40000), (live_100, 100000)])
    print(f"  => Best LIVE: step {best_live[1]} ({best_live[0]}/{50} = {100*best_live[0]/50:.1f}%)")
elif live_100 == 0 and ema_100 == 0 and live_20 > 0:
    print("CONCLUSION: Model worked briefly then genuinely collapsed.")
    print("  => Training instability. LIVE weights also regressed.")
    print("  => Design v5: add LR cosine decay + reduce pose_xyz weight.")
elif live_100 == 0 and ema_100 == 0 and live_20 == 0:
    print("CONCLUSION: Model never produced valid scenes even in live weights.")
    print("  => Architectural failure. v5 needs structural changes.")
else:
    print("CONCLUSION: Mixed -- examine numbers above.")
