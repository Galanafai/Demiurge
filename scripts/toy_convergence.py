"""Toy convergence test for the Demiurge DDPM denoiser.

Gate before full GPU training. Must reach validity_rate >= 0.9 on 100
sampled scenes within 30 minutes of CPU training. If the gate fails,
halt and produce a debug artifact. Do NOT extend the budget.

Toy distribution:
  - N=2000 real validated scenes from data/v1 (first 2000 in shard order)
  - All normalisation is delegated to SceneTensor.normalize(WorkspaceBounds)
  - No synthetic sampler -- uses actual geometry, object types, and z positions
    that the Drake validator already accepted during dataset generation.

Model config: d_model=128, 3 layers, 4 heads (toy variant, ~1.3M params)
Training: AdamW lr=1e-3, 10,000 steps max, batch_size=64

Usage:
    uv run python scripts/toy_convergence.py --seed 42
    uv run python scripts/toy_convergence.py --seed 42 --n-train 2000 --max-steps 10000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from model.denoiser import DenoiserConfig, SceneDenoiser  # noqa: E402
from model.loss import LossWeights, SceneDiffusionLoss  # noqa: E402
from model.rotations import rot6d_to_quat_wxyz  # noqa: E402
from model.schedule import CosineSchedule, DDIMSampler  # noqa: E402
from scene.schema import N_MAX, SceneTensor, WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402

BUDGET_S: float = 30 * 60.0   # 30-minute hard gate
VALIDITY_THRESHOLD: float = 0.9
TOY_N_TRAIN: int = 700        # tabletop_reach/1-obj -- all available in shard 0
TOY_BATCH: int = 64
TOY_MAX_STEPS: int = 25_000   # ~28min at ~900 steps/min; leaves 2min for Drake
TOY_DDIM_STEPS: int = 50
TOY_VAL_N: int = 100
TOY_TASK_FAMILY: str = "tabletop_reach"
TOY_MAX_OBJECTS: int = 1      # 1-object scenes: simplest valid distribution


# ---------------------------------------------------------------------------
# Toy scene sampler
# ---------------------------------------------------------------------------


def _load_toy_dataset(
    n: int,
    seed: int = 0,
    bounds: WorkspaceBounds | None = None,
    task_family: str = "tabletop_reach",
    max_objects: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load n real validated scenes from data/v1 as the toy distribution.

    Filters to scenes from ``task_family`` with at most ``max_objects``
    present. This creates a homogeneous, low-diversity distribution that a
    small model can overfit in a few thousand steps.

    Returns:
        data:      (n, N_MAX, 13) continuous feature tensor in normalised space
        type_ids:  (n, N_MAX) long tensor
        presence:  (n, N_MAX) bool tensor
    """
    from data.reader import ShardReader
    from model.rotations import quat_wxyz_to_6d

    if bounds is None:
        bounds = WorkspaceBounds.default()

    data_list: list[torch.Tensor] = []
    type_list: list[torch.Tensor] = []
    pres_list: list[torch.Tensor] = []

    reader = ShardReader("data/v1")
    for scene, _desc, report, _sdf in reader:
        if len(data_list) >= n:
            break
        fam = report.get("task_family", "")
        n_obj = int(scene.presence.sum().item())
        if fam != task_family or n_obj > max_objects:
            continue
        sn = scene.normalize(bounds)
        xyz = sn.poses[:, :3]                       # (N_MAX, 3) normalised
        rot6d = quat_wxyz_to_6d(sn.poses[:, 3:7])  # (N_MAX, 6)
        scale = sn.scales                            # (N_MAX, 3) normalised
        pres_bit = sn.presence.float().unsqueeze(-1) # (N_MAX, 1)
        feat = torch.cat([xyz, rot6d, scale, pres_bit], dim=-1)  # (N_MAX, 13)
        data_list.append(feat)
        type_list.append(sn.object_types)
        pres_list.append(sn.presence)

    if len(data_list) < n:
        raise RuntimeError(
            f"Only {len(data_list)} {task_family}/{max_objects}-obj scenes "
            f"available in data/v1, requested {n}. "
            "Reduce --n-train or relax the filter."
        )

    return (
        torch.stack(data_list),   # (n, N_MAX, 13)
        torch.stack(type_list),   # (n, N_MAX)
        torch.stack(pres_list),   # (n, N_MAX) bool
    )


# ---------------------------------------------------------------------------
# Validation pass (Drake)
# ---------------------------------------------------------------------------


def _validate_samples(
    x0_batch: torch.Tensor,
    type_ids_ref: torch.Tensor,
    bounds: WorkspaceBounds,
    validator: SceneValidator,
) -> int:
    """Run Drake validity check on a batch of decoded scenes.

    Args:
        x0_batch: Tensor of shape (B, N_MAX, 13).
        type_ids_ref: Reference type_ids tensor (B, N_MAX) from the dataset,
            used to assign object types for decoding (model does not predict types).
        bounds: WorkspaceBounds for denormalisation.
        validator: SceneValidator instance.

    Returns:
        Number of accepted scenes.
    """
    accepted = 0
    B = x0_batch.shape[0]

    for i in range(B):
        x = x0_batch[i]   # (N_MAX, 13)
        xyz_n = x[:, :3]
        rot6d_pred = x[:, 3:9]
        scale_n = x[:, 9:12]
        pres_bit = x[:, 12]

        pres_mask = pres_bit > 0.0

        # Project 6D -> quaternion.
        quats = rot6d_to_quat_wxyz(rot6d_pred)         # (N_MAX, 4)
        poses_raw = torch.cat([xyz_n, quats], dim=-1)  # (N_MAX, 7)

        # Use the reference type_ids from the dataset (round-robin over val batch).
        types = type_ids_ref[i % type_ids_ref.shape[0]]

        try:
            st_norm = SceneTensor(
                object_types=types,
                poses=poses_raw.float(),
                scales=scale_n.clamp(-1.8, 1.0).float(),  # normalised range clamp
                presence=pres_mask,
            )
        except ValueError:
            continue   # malformed output; count as rejected

        st = st_norm.denormalize(bounds)
        rpt = validator.validate(st)
        if rpt.accepted:
            accepted += 1

    return accepted


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-train", type=int, default=TOY_N_TRAIN)
    p.add_argument("--max-steps", type=int, default=TOY_MAX_STEPS)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    wall_start = time.monotonic()

    print("=== Toy Convergence Test ===")
    print(f"Budget: {BUDGET_S/60:.0f} min | Threshold: {VALIDITY_THRESHOLD:.0%} | "
          f"Val scenes: {TOY_VAL_N}")

    # Toy model: d_model=128, 3 layers, 4 heads (~0.5M params)
    toy_cfg = DenoiserConfig(n_layers=3, d_model=128, n_heads=4, ffn_mult=4, dropout=0.0)
    model = SceneDenoiser(toy_cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toy model: {n_params/1e6:.3f}M parameters")

    schedule = CosineSchedule(T=1000)
    loss_fn = SceneDiffusionLoss(LossWeights(
        pose_xyz=1.0, pose_rot=1.0, scale=0.5, type_ce=0.1, presence_bce=0.05
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    bounds = WorkspaceBounds.default()

    # Load toy dataset: tabletop_reach, 1-object scenes only.
    print(f"Loading {args.n_train} real scenes from data/v1 "
          f"({TOY_TASK_FAMILY}, max {TOY_MAX_OBJECTS} objects)...")
    dataset, type_ids_full, presence_full = _load_toy_dataset(
        args.n_train, seed=args.seed, bounds=bounds,
        task_family=TOY_TASK_FAMILY, max_objects=TOY_MAX_OBJECTS,
    )
    print(f"Dataset ready: {dataset.shape}.")

    model.train()
    step = 0
    losses: list[float] = []

    print("Training...")
    while step < args.max_steps:
        # Check budget.
        elapsed = time.monotonic() - wall_start
        if elapsed >= BUDGET_S:
            print(f"\n[BUDGET EXCEEDED at step {step} / {elapsed/60:.1f} min]")
            break

        # Random mini-batch.
        idx = torch.randint(0, args.n_train, (TOY_BATCH,))
        x_cont = dataset[idx]                  # (B, N_MAX, 13)
        type_ids = type_ids_full[idx]          # (B, N_MAX)
        presence = presence_full[idx]          # (B, N_MAX)

        # Sample random timestep and add noise.
        t_idx = torch.randint(0, 1000, (TOY_BATCH,))
        eps_xyz = torch.randn(TOY_BATCH, N_MAX, 3)
        eps_rot = torch.randn(TOY_BATCH, N_MAX, 6)
        eps_scale = torch.randn(TOY_BATCH, N_MAX, 3)
        eps_pres = torch.randn(TOY_BATCH, N_MAX, 1)
        eps_full = torch.cat([eps_xyz, eps_rot, eps_scale, eps_pres], dim=-1)
        x_noisy, _ = schedule.add_noise(x_cont, t_idx, eps_full)

        optimizer.zero_grad()
        pred = model(x_noisy, type_ids, t_idx, text_emb=None)
        loss_out = loss_fn(pred, eps_xyz, eps_rot, eps_scale, eps_pres, type_ids, presence)
        loss_out.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss_out.total.item())
        step += 1

        if step % 500 == 0:
            recent_loss = sum(losses[-100:]) / 100
            elapsed = time.monotonic() - wall_start
            print(f"  step={step:5d} | loss={recent_loss:.4f} | elapsed={elapsed/60:.1f}min")

    # --- Validity check via Drake ---
    elapsed_before_val = time.monotonic() - wall_start
    remaining = BUDGET_S - elapsed_before_val
    if remaining < 60:
        print(f"\n[WARNING] Only {remaining:.0f}s left before budget; Drake validation may not complete in time]")

    print(f"\nValidating {TOY_VAL_N} sampled scenes via Drake...")
    model.eval()
    sampler = DDIMSampler(schedule, n_steps=TOY_DDIM_STEPS)
    fn = model.noise_prediction_fn(text_emb=None)
    validator = SceneValidator(rrt_budget_s=1.0)

    accepted = 0
    val_batch = 16
    seed = 1000
    remaining_val = TOY_VAL_N

    while remaining_val > 0:
        b = min(val_batch, remaining_val)
        x0 = sampler.sample(fn, (b, N_MAX, 13), seed=seed, device=torch.device("cpu"))
        seed += 1
        remaining_val -= b
        accepted += _validate_samples(x0, type_ids_full[:val_batch], bounds, validator)

    validity_rate = accepted / TOY_VAL_N
    total_elapsed = time.monotonic() - wall_start

    print("\n=== TOY CONVERGENCE RESULT ===")
    print(f"  Steps trained:   {step}")
    print(f"  Wall-clock:      {total_elapsed/60:.1f} min")
    print(f"  Validity rate:   {validity_rate:.3f} ({accepted}/{TOY_VAL_N})")
    print(f"  Threshold:       {VALIDITY_THRESHOLD:.2f}")
    status = "PASS" if validity_rate >= VALIDITY_THRESHOLD else "FAIL"
    print(f"  Status:          {status}")
    print("==============================")

    if validity_rate < VALIDITY_THRESHOLD:
        # Write debug artifact.
        _write_debug_artifact(step, total_elapsed, losses, validity_rate, accepted)
        sys.exit(1)


def _write_debug_artifact(
    step: int,
    elapsed_s: float,
    losses: list[float],
    validity_rate: float,
    accepted: int,
) -> None:
    """Write artifacts/toy_convergence_debug.md on failure."""
    out = Path("artifacts/toy_convergence_debug.md")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Compute simple loss curve stats.
    first_100 = losses[:100]
    last_100 = losses[-100:]
    avg_first = sum(first_100) / len(first_100) if first_100 else float("nan")
    avg_last = sum(last_100) / len(last_100) if last_100 else float("nan")

    content = f"""# Toy Convergence Debug Report

**Generated:** automatically on FAIL

## Summary

| Metric | Value |
|---|---|
| Steps completed | {step} |
| Wall-clock | {elapsed_s/60:.1f} min |
| Validity rate | {validity_rate:.3f} ({accepted}/{TOY_VAL_N}) |
| Threshold | {VALIDITY_THRESHOLD:.2f} |
| Status | FAIL |

## Loss Curve

| Window | Avg Loss |
|---|---|
| First 100 steps | {avg_first:.4f} |
| Last 100 steps | {avg_last:.4f} |
| Reduction | {(avg_first - avg_last)/avg_first*100:.1f}% |

## Failure Mode Hypotheses

Check the following in order before patching:

1. **Rotation projection gradient issue**: If rot6d loss is not decreasing,
   verify that gradients flow through `rot6d_to_matrix` (Gram-Schmidt).
   Run `tests/model/test_rotations.py::test_gradient_flow`.

2. **Presence head collapse**: If all sampled scenes have presence_bit < 0 for
   all slots, the presence BCE loss may be dominating and the model predicts
   all-absent. Increase `presence_bce` weight or verify BCE target is correct.

3. **Scale normalisation**: Toy scenes have scale=0 in normalised space.
   If the model is predicting large scale noise, the denormalized scene may
   place objects outside workspace bounds. Check `denormalize()` output.

4. **Validation failure mode**: Run a small manual validation to see which
   Drake checks are failing (non-interpenetration, stability, IK, RRT).
   The rrt_budget_s=1.0 in toy mode may be too tight; verify against
   the profile run budget.

## Next Steps

- Do NOT start full GPU training until toy passes.
- Do NOT extend the 30-minute budget. Diagnose first.
- Address the highest-probability failure mode above, re-run.
"""
    out.write_text(content)
    print(f"\nDebug artifact written to {out}")


if __name__ == "__main__":
    main()
