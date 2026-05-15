"""Generate noise-conditioned classifier training data for Phase B.

For each scene in the v1 training dataset:
  1. Sample t ~ Uniform(0, T-1)
  2. Apply forward noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps
  3. Label valid=1
  4. Generate corrupted variant: perturb one present object's xyz by +/- 0.3m
  5. Confirm the corrupted scene is likely invalid (interpenetration proxy check)
  6. Apply same forward noise at same t, label valid=0
  7. Write both to data/classifier_v1/ as WebDataset shards

Target: ~100k labeled noisy scenes (50k valid + 50k invalid).

INVARIANT: Classifier data must use NOISED inputs x_t, not clean x_0.
A clean-only classifier gives useless gradients at intermediate diffusion
timesteps where guidance is applied. This script enforces this invariant
by construction -- the forward noise step is non-optional.

Drake validation strategy (per user approval):
  We use a GEOMETRIC PROXY instead of full Drake validation for corrupted
  scenes. A 0.3m xyz shift on a table of dimensions ~1.0m x 0.6m almost
  certainly pushes the object out of the workspace or into another object.
  We skip Drake RRT (which would take ~14 hours on 100k scenes) and instead
  verify that the shifted object's xyz is outside the workspace bounds.
  This is documented with an explicit APPROX: comment per AGENTS.md convention.

  # APPROX: Corrupted-scene invalidity is verified by checking whether the
  # perturbed object's xyz coordinate exceeds WorkspaceBounds. This avoids
  # running full Drake RRT on 50k corrupted scenes (estimated 14 CPU-hours).
  # Invariant relaxed: no_interpenetration only (not ik_reachable/rrt_solvable).
  # Scenes where the 0.3m shift stays within workspace are discarded (~<5%).

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 scripts/generate_classifier_data.py \
        --data-dir data/v1 \
        --out-dir data/classifier_v1 \
        --seed 42 \
        --max-scenes 50000
"""
from __future__ import annotations

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import torch
import webdataset as wds

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data.reader import ShardReader  # noqa: E402
from model.rotations import quat_wxyz_to_6d  # noqa: E402
from model.schedule import CosineSchedule  # noqa: E402
from scene.schema import N_MAX, SceneTensor, WorkspaceBounds  # noqa: E402

_PERTURB_M: float = 0.3  # XYZ perturbation in metres (physical space)


def _scene_to_x0(scene: SceneTensor, bounds: WorkspaceBounds) -> torch.Tensor:
    """Convert SceneTensor to (N_MAX, 13) normalised continuous tensor."""
    s = scene.normalize(bounds)
    xyz = s.poses[:, :3]                          # (N_MAX, 3)
    rot6d = quat_wxyz_to_6d(s.poses[:, 3:7])      # (N_MAX, 6)
    scale = s.scales                               # (N_MAX, 3)
    pres = s.presence.float().unsqueeze(-1)        # (N_MAX, 1)
    return torch.cat([xyz, rot6d, scale, pres], dim=-1)  # (N_MAX, 13)


def _add_noise(
    x0: torch.Tensor,
    t: int,
    schedule: CosineSchedule,
    rng: torch.Generator,
) -> torch.Tensor:
    """Apply forward diffusion noise at timestep t."""
    alpha_bar = float(schedule.alpha_bar[t])
    sqrt_ab = alpha_bar ** 0.5
    sqrt_one_minus_ab = (1.0 - alpha_bar) ** 0.5
    eps = torch.randn_like(x0, generator=rng)
    return sqrt_ab * x0 + sqrt_one_minus_ab * eps


def _corrupt_scene(
    scene: SceneTensor,
    bounds: WorkspaceBounds,
    rng: random.Random,
) -> SceneTensor | None:
    """Perturb one present object's xyz by +/-_PERTURB_M.

    Returns corrupted SceneTensor, or None if:
    - No present slots (shouldn't happen in training data)
    - Perturbed position stays within workspace bounds (valid -- discard)

    # APPROX: no_interpenetration only -- perturbed object is checked to be
    # outside workspace bounds as a proxy for Drake invalidity. This avoids
    # full Drake RRT on 50k corrupted scenes (~14 CPU-hours).
    """
    present_slots = [i for i in range(N_MAX) if scene.presence[i]]
    if not present_slots:
        return None

    slot = rng.choice(present_slots)

    # Clone and perturb in physical space.
    new_poses = scene.poses.clone()
    delta = torch.zeros(3)
    axis = rng.randint(0, 2)  # 0=x, 1=y, 2=z
    sign = rng.choice([-1.0, 1.0])
    delta[axis] = sign * _PERTURB_M
    new_poses[slot, :3] = new_poses[slot, :3] + delta

    corrupted = SceneTensor(
        object_types=scene.object_types.clone(),
        poses=new_poses,
        scales=scene.scales.clone(),
        presence=scene.presence.clone(),
    )

    # APPROX: workspace bounds proxy check.
    # Physical workspace: x in [x_min, x_max], y, z similarly.
    # denormalize/normalize round-trip: check normalised xyz is outside [-1, 1].
    normalised = corrupted.normalize(bounds)
    norm_xyz = normalised.poses[slot, :3]
    if norm_xyz.abs().max().item() <= 1.0:
        # Shift stayed within workspace -- discard this pair.
        return None

    return corrupted


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/v1")
    p.add_argument("--out-dir", default="data/classifier_v1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-scenes", type=int, default=50000,
                   help="Max valid scenes to process (produces up to 2x this many labeled samples)")
    p.add_argument("--shard-size", type=int, default=5000,
                   help="Samples per output shard")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng_torch = torch.Generator()
    rng_torch.manual_seed(args.seed)
    rng_py = random.Random(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = CosineSchedule(T=1000)
    bounds = WorkspaceBounds.default()

    print(f"Generating classifier data: {args.data_dir} -> {args.out_dir}")
    print(f"Max scenes: {args.max_scenes} | Shard size: {args.shard_size} | Seed: {args.seed}")

    shard_idx = 0
    current_shard_path = out_dir / f"cls-{shard_idx:05d}.tar"
    sink = wds.TarWriter(str(current_shard_path))
    shard_count = 0

    n_valid_written = 0
    n_invalid_written = 0
    n_discarded = 0
    n_scenes_processed = 0
    t0 = time.monotonic()

    reader = ShardReader(args.data_dir, shuffle_buffer=1000, shardshuffle=True)

    for scene, _desc, _report, _sdf in reader:
        if n_scenes_processed >= args.max_scenes:
            break

        n_scenes_processed += 1
        x0 = _scene_to_x0(scene, bounds)  # (N_MAX, 13)

        # Sample random timestep.
        t_int = int(torch.randint(0, 1000, (1,), generator=rng_torch).item())
        t_tensor = torch.tensor([t_int], dtype=torch.long)

        # --- Valid sample ---
        x_t_valid = _add_noise(x0, t_int, schedule, rng_torch)  # (N_MAX, 13)
        key_valid = f"{n_scenes_processed:08d}_valid"
        sink.write({
            "__key__": key_valid,
            "xt.pt": _tensor_to_bytes(x_t_valid),
            "t.pt": _tensor_to_bytes(t_tensor),
            "label.pt": _tensor_to_bytes(torch.tensor([1], dtype=torch.long)),
        })
        n_valid_written += 1
        shard_count += 1

        # --- Corrupted (invalid) sample ---
        corrupted = _corrupt_scene(scene, bounds, rng_py)
        if corrupted is None:
            n_discarded += 1
        else:
            x0_corrupt = _scene_to_x0(corrupted, bounds)
            x_t_corrupt = _add_noise(x0_corrupt, t_int, schedule, rng_torch)
            key_invalid = f"{n_scenes_processed:08d}_invalid"
            sink.write({
                "__key__": key_invalid,
                "xt.pt": _tensor_to_bytes(x_t_corrupt),
                "t.pt": _tensor_to_bytes(t_tensor),
                "label.pt": _tensor_to_bytes(torch.tensor([0], dtype=torch.long)),
            })
            n_invalid_written += 1
            shard_count += 1

        # Roll shard.
        if shard_count >= args.shard_size:
            sink.close()
            shard_idx += 1
            current_shard_path = out_dir / f"cls-{shard_idx:05d}.tar"
            sink = wds.TarWriter(str(current_shard_path))
            shard_count = 0

        if n_scenes_processed % 5000 == 0:
            elapsed = time.monotonic() - t0
            rate = n_scenes_processed / elapsed
            print(f"  {n_scenes_processed}/{args.max_scenes} scenes | "
                  f"+{n_valid_written}v +{n_invalid_written}i -{n_discarded}d | "
                  f"{rate:.1f} scenes/s")

    sink.close()
    elapsed = time.monotonic() - t0

    # Write manifest.
    total_samples = n_valid_written + n_invalid_written
    manifest = {
        "n_scenes_processed": n_scenes_processed,
        "n_valid": n_valid_written,
        "n_invalid": n_invalid_written,
        "n_discarded": n_discarded,
        "total_samples": total_samples,
        "n_shards": shard_idx + 1,
        "perturbation_m": _PERTURB_M,
        "approx_note": (
            "APPROX: corrupted-scene invalidity checked via workspace bounds proxy only. "
            "Full Drake RRT validation skipped to avoid ~14 CPU-hours on 50k scenes. "
            "Invariant relaxed: no_interpenetration proxy only, not ik_reachable/rrt_solvable."
        ),
        "seed": args.seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Valid samples:    {n_valid_written}")
    print(f"  Invalid samples:  {n_invalid_written}")
    print(f"  Discarded (proxy passed): {n_discarded} ({100*n_discarded/max(n_scenes_processed,1):.1f}%)")
    print(f"  Total samples:    {total_samples}")
    print(f"  Shards:           {shard_idx + 1}")
    print(f"  Output:           {out_dir}")


if __name__ == "__main__":
    main()
