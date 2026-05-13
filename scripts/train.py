"""Training harness for the Demiurge DDPM scene denoiser.

Usage:
    uv run python scripts/train.py --config configs/train/baseline.yaml --seed 42
    uv run python scripts/train.py --config configs/train/baseline.yaml --seed 42 --max-epochs 5
    uv run python scripts/train.py --config configs/train/baseline.yaml --seed 42 \
        --override wandb.experiment=my_run training.batch_size=64
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from model.denoiser import DenoiserConfig, SceneDenoiser  # noqa: E402
from model.loss import LossWeights, SceneDiffusionLoss  # noqa: E402
from model.rotations import quat_wxyz_to_6d  # noqa: E402
from model.schedule import CosineSchedule, DDIMSampler  # noqa: E402
from scene.schema import N_MAX, WorkspaceBounds  # noqa: E402

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class ConfigMismatchError(RuntimeError):
    """Raised when a checkpoint's architecture differs from the current config."""


# ---------------------------------------------------------------------------
# YAML + CLI helpers
# ---------------------------------------------------------------------------


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> None:
    """Apply dotted-key=value overrides in-place, e.g. training.batch_size=64."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item!r}")
        key, val = item.split("=", 1)
        # Parse val: try int, float, bool, else str
        for coerce in (int, float, lambda x: {"true": True, "false": False}[x.lower()]):
            try:
                val = coerce(val)  # type: ignore[assignment]
                break
            except (ValueError, KeyError):
                pass
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = val


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-epochs", type=int, default=None,
                   help="Stop after N epochs (benchmark mode). Prints steps/sec extrapolation.")
    p.add_argument("--override", nargs="*", default=[], metavar="KEY=VAL",
                   help="Dotted key=value overrides applied after YAML load")
    return p.parse_args()


# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------


def _init_wandb(cfg: dict[str, Any], run_cfg: dict[str, Any], git_sha: str) -> Any:
    """Initialise W&B run. Raises EnvironmentError if API key is absent."""
    wcfg = cfg.get("wandb", {})
    if not wcfg.get("enabled", False):
        return None
    api_key = os.environ.get("WANDB_API_KEY", "")
    if not api_key:
        raise OSError(
            "wandb.enabled=true but WANDB_API_KEY is not set in the environment. "
            "Export WANDB_API_KEY before running, or set wandb.enabled=false. "
            "Do not use WANDB_MODE=offline for production GPU runs."
        )
    import wandb
    run = wandb.init(
        project=wcfg.get("project", "demiurge"),
        name=wcfg.get("experiment", "unnamed"),
        tags=wcfg.get("tags", []),
        config={**run_cfg, "git_sha": git_sha},
    )
    return run


def _git_sha() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _arch_fingerprint(dcfg: DenoiserConfig) -> dict[str, Any]:
    return {"n_layers": dcfg.n_layers, "d_model": dcfg.d_model,
            "n_heads": dcfg.n_heads, "ffn_mult": dcfg.ffn_mult}


def save_checkpoint(
    out_dir: Path,
    step: int,
    model: SceneDenoiser,
    ema_weights: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    dcfg: DenoiserConfig,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "ema_state": ema_weights,
        "optimizer_state": optimizer.state_dict(),
        "arch": _arch_fingerprint(dcfg),
    }
    path = out_dir / f"step_{step:08d}.pt"
    torch.save(ckpt, path)
    # Atomic symlink update for "latest".
    latest = out_dir / "latest.pt"
    tmp = out_dir / "latest.pt.tmp"
    torch.save(ckpt, tmp)
    tmp.rename(latest)


def load_checkpoint(
    latest: Path,
    model: SceneDenoiser,
    optimizer: torch.optim.Optimizer,
    dcfg: DenoiserConfig,
) -> tuple[int, dict[str, torch.Tensor]]:
    """Load checkpoint; raises ConfigMismatchError if architecture differs."""
    ckpt = torch.load(latest, weights_only=False)
    saved_arch = ckpt.get("arch", {})
    current_arch = _arch_fingerprint(dcfg)
    if saved_arch != current_arch:
        raise ConfigMismatchError(
            f"Checkpoint architecture {saved_arch} does not match current config "
            f"{current_arch}. Cannot resume -- create a new output directory or "
            "match the config to the checkpoint."
        )
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return int(ckpt["step"]), dict(ckpt["ema_state"])


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------


def _ema_init(model: SceneDenoiser) -> dict[str, torch.Tensor]:
    return {k: v.clone().float() for k, v in model.state_dict().items()}


def _ema_update(
    ema: dict[str, torch.Tensor], model: SceneDenoiser, decay: float
) -> None:
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema[k].mul_(decay).add_(v.float(), alpha=1.0 - decay)


# ---------------------------------------------------------------------------
# Dataset + collation
# ---------------------------------------------------------------------------


def _scene_to_cont(scene: Any, bounds: WorkspaceBounds) -> torch.Tensor:
    """Convert a SceneTensor to the (N_MAX, 13) continuous feature tensor.

    Layout: xyz(3) | rot6d(6) | scale(3) | presence_bit(1)
    Normalised to workspace bounds before packing.
    """
    s = scene.normalize(bounds)
    xyz = s.poses[:, :3]                          # (N_MAX, 3)
    rot6d = quat_wxyz_to_6d(s.poses[:, 3:7])      # (N_MAX, 6)
    scale = s.scales                               # (N_MAX, 3)
    pres = s.presence.float().unsqueeze(-1)        # (N_MAX, 1)
    return torch.cat([xyz, rot6d, scale, pres], dim=-1)  # (N_MAX, 13)


def _collate(
    batch: list[tuple[Any, str, dict, str]],
    bounds: WorkspaceBounds,
    text_cache: dict[str, torch.Tensor] | None,
) -> dict[str, torch.Tensor]:
    x_conts, type_ids_list, presence_list, text_embs = [], [], [], []
    for scene, desc, _report, _sdf in batch:
        x_conts.append(_scene_to_cont(scene, bounds))
        type_ids_list.append(scene.object_types)
        presence_list.append(scene.presence)
        if text_cache is not None:
            key = hashlib.sha256(desc.encode()).hexdigest()
            text_embs.append(text_cache[key])
    out: dict[str, torch.Tensor] = {
        "x_cont": torch.stack(x_conts),           # (B, N_MAX, 13)
        "type_ids": torch.stack(type_ids_list),   # (B, N_MAX)
        "presence": torch.stack(presence_list),   # (B, N_MAX)
    }
    if text_embs:
        out["text_emb"] = torch.stack(text_embs)  # (B, 384)
    return out


# ---------------------------------------------------------------------------
# Validation pass
# ---------------------------------------------------------------------------


@torch.no_grad()
def run_validation(
    model: SceneDenoiser,
    schedule: CosineSchedule,
    ddim_steps: int,
    n_scenes: int,
    bounds: WorkspaceBounds,
    device: torch.device,
    text_emb: torch.Tensor | None = None,
) -> float:
    """Sample scenes via DDIM and measure Drake validity rate.

    Returns fraction of sampled scenes accepted by SceneValidator.
    """
    from model.rotations import rot6d_to_quat_wxyz
    from scene.schema import SceneTensor
    from validator.core import SceneValidator

    model.eval()
    sampler = DDIMSampler(schedule, n_steps=ddim_steps)
    fn = model.noise_prediction_fn(text_emb)
    validator = SceneValidator(rrt_budget_s=2.0)

    accepted = 0
    batch_size = min(n_scenes, 16)
    remaining = n_scenes
    seed = 0

    while remaining > 0:
        b = min(batch_size, remaining)
        x0 = sampler.sample(fn, (b, N_MAX, 13), seed=seed, device=device)
        seed += 1
        remaining -= b

        # Decode x0 -> SceneTensor and validate.
        xyz = x0[:, :, :3]
        rot6d_pred = x0[:, :, 3:9]
        scale_pred = x0[:, :, 9:12]
        pres_bit = x0[:, :, 12]

        for i in range(b):
            pres_mask = pres_bit[i] > 0.0
            quats = rot6d_to_quat_wxyz(rot6d_pred[i])             # (N_MAX, 4)
            poses_raw = torch.cat([xyz[i], quats], dim=-1)        # (N_MAX, 7)
            types = torch.zeros(N_MAX, dtype=torch.long)
            st_norm = SceneTensor(
                object_types=types,
                poses=poses_raw,
                scales=scale_pred[i].clamp(0.5, 2.0),
                presence=pres_mask,
            )
            st = st_norm.denormalize(bounds)
            rpt = validator.validate(st)
            if rpt.accepted:
                accepted += 1

    model.train()
    return accepted / n_scenes


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = _load_yaml(args.config)
    _apply_overrides(cfg, args.override or [])

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = WorkspaceBounds.default()

    # --- Model ---
    mcfg_raw = cfg.get("model", {})
    dcfg = DenoiserConfig(
        n_layers=mcfg_raw.get("n_layers", 6),
        d_model=mcfg_raw.get("d_model", 256),
        n_heads=mcfg_raw.get("n_heads", 8),
        ffn_mult=mcfg_raw.get("ffn_mult", 4),
        dropout=mcfg_raw.get("dropout", 0.1),
    )
    model = SceneDenoiser(dcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_params/1e6:.2f}M parameters | device={device}")

    # --- Diffusion ---
    dcfg_diff = cfg.get("diffusion", {})
    T = int(dcfg_diff.get("T", 1000))
    ddim_steps = int(dcfg_diff.get("ddim_steps", 50))
    schedule = CosineSchedule(T=T)

    # --- Training config ---
    tcfg = cfg.get("training", {})
    batch_size = int(tcfg.get("batch_size", 128))
    lr = float(tcfg.get("lr", 3e-4))
    warmup_steps = int(tcfg.get("warmup_steps", 1000))
    max_steps = int(tcfg.get("max_steps", 100000))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    grad_accum = int(tcfg.get("gradient_accumulation", 1))
    ema_decay = float(tcfg.get("ema_decay", 0.9999))
    ckpt_steps = int(tcfg.get("checkpoint_steps", 5000))
    val_every = int(tcfg.get("val_every_epochs", 10))
    val_n = int(tcfg.get("val_n_scenes", 100))
    weighted_sampling = bool(tcfg.get("weighted_sampling", True))
    mixed_prec = tcfg.get("dtype", "float32")
    use_amp = mixed_prec in ("float16", "bfloat16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if mixed_prec == "bfloat16" else torch.float16

    # --- Loss ---
    lcfg = cfg.get("loss", {})
    loss_weights = LossWeights(
        pose_xyz=float(lcfg.get("pose_xyz", 1.0)),
        pose_rot=float(lcfg.get("pose_rot", 1.0)),
        scale=float(lcfg.get("scale", 0.5)),
        type_ce=float(lcfg.get("type_ce", 0.1)),
        presence_bce=float(lcfg.get("presence_bce", 0.05)),
    )
    loss_fn = SceneDiffusionLoss(loss_weights)

    # --- Dataset ---
    from data.reader import ShardReader
    dscfg = cfg.get("dataset", {})
    data_dir = dscfg.get("path", "data/v1")
    expected_hash = dscfg.get("hash", "")

    # Verify dataset hash.
    if expected_hash:
        import pathlib
        h = hashlib.sha256()
        manifest_bytes = (pathlib.Path(data_dir) / "manifest.json").read_bytes()
        h.update(manifest_bytes)
        manifest = json.loads(manifest_bytes)
        for shard in manifest.get("completed_shards", []):
            with open(pathlib.Path(data_dir) / shard, "rb") as f:
                while chunk := f.read(1 << 20):
                    h.update(chunk)
        actual_hash = h.hexdigest()
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Dataset hash mismatch.\n  expected: {expected_hash}\n  actual:   {actual_hash}\n"
                "The dataset may have been modified. Update configs/train/baseline.yaml if intentional."
            )
        print(f"Dataset hash verified: {actual_hash[:16]}...")

    # Load text embedding cache if present.
    text_cache: dict[str, torch.Tensor] | None = None
    cache_path = Path(data_dir) / "text_embeddings.pt"
    if cache_path.exists():
        text_cache = torch.load(cache_path, weights_only=True)
        print(f"Text embedding cache loaded: {len(text_cache)} entries")
    else:
        print("No text embedding cache found -- running unconditional.")

    # Build dataset as a list for weighted sampling (fits in RAM at 50k scenes).
    print("Loading dataset into memory for weighted sampling...")
    all_examples: list[tuple] = []
    template_counts: dict[str, int] = {}
    reader = ShardReader(data_dir)
    for scene, desc, report, sdf in reader:
        fam = report.get("task_family", "unknown")
        template_counts[fam] = template_counts.get(fam, 0) + 1
        all_examples.append((scene, desc, report, sdf))
    print(f"Loaded {len(all_examples)} examples. Template distribution: {template_counts}")

    # Separate held-out split: last held_out_per_template scenes per template.
    held_out_n = int(dscfg.get("held_out_per_template", 100))
    held_out_indices: set[int] = set()
    per_template: dict[str, list[int]] = {}
    for i, (_, _, report, _) in enumerate(all_examples):
        fam = report.get("task_family", "unknown")
        per_template.setdefault(fam, []).append(i)
    for fam, indices in per_template.items():
        held_out_indices.update(indices[-held_out_n:])
    train_indices = [i for i in range(len(all_examples)) if i not in held_out_indices]
    print(f"Train: {len(train_indices)} | Held-out: {len(held_out_indices)}")

    # WeightedRandomSampler weights.
    if weighted_sampling:
        train_families = [all_examples[i][2].get("task_family", "unknown") for i in train_indices]
        fam_counts = {f: train_families.count(f) for f in set(train_families)}
        weights = [1.0 / fam_counts[f] for f in train_families]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)
    else:
        sampler = None  # type: ignore[assignment]

    class _IndexDataset(torch.utils.data.Dataset):
        def __init__(self, examples: list, indices: list[int]) -> None:
            self._examples = examples
            self._indices = indices

        def __len__(self) -> int:
            return len(self._indices)

        def __getitem__(self, idx: int) -> tuple:
            return self._examples[self._indices[idx]]

    train_ds = _IndexDataset(all_examples, train_indices)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: _collate(b, bounds, text_cache),
        num_workers=int(dscfg.get("num_workers", 0)),
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )

    # --- Optimizer + LR schedule ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None  # type: ignore[attr-defined]

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine decay to 10% of peak.
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Output dir + resume ---
    out_dir = Path(cfg.get("output", {}).get("dir", "checkpoints/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    ema_weights = _ema_init(model)
    latest = out_dir / "latest.pt"
    if latest.exists():
        print(f"Resuming from {latest}")
        start_step, ema_weights = load_checkpoint(latest, model, optimizer, dcfg)
        for _ in range(start_step):
            lr_scheduler.step()
        print(f"Resumed at step {start_step}")

    # --- W&B ---
    git_sha = _git_sha()
    wandb_run = _init_wandb(cfg, {**cfg, "seed": args.seed, "git_sha": git_sha, "n_params": n_params}, git_sha)

    # --- Training loop ---
    max_epochs = args.max_epochs
    steps_per_epoch = len(loader)
    n_epochs = max_epochs if max_epochs else math.ceil((max_steps - start_step) / max(1, steps_per_epoch))

    print(f"Steps/epoch: {steps_per_epoch} | Target epochs: {n_epochs} | Start step: {start_step}")

    model.train()
    step = start_step
    benchmark_step_times: list[float] = []
    grad_accum_count = 0

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_t0 = time.monotonic()

        for batch in loader:
            if step >= max_steps:
                break

            x_cont = batch["x_cont"].to(device)
            type_ids = batch["type_ids"].to(device)
            presence = batch["presence"].to(device)
            text_emb_b = batch.get("text_emb")
            if text_emb_b is not None:
                text_emb_b = text_emb_b.to(device)

            # Sample random timesteps.
            B = x_cont.shape[0]
            t_idx = torch.randint(0, T, (B,), device=device)

            # Sample noise and compute noisy input.
            eps_xyz = torch.randn_like(x_cont[:, :, :3])
            eps_rot = torch.randn_like(x_cont[:, :, 3:9])
            eps_scale = torch.randn_like(x_cont[:, :, 9:12])
            eps_pres = torch.randn_like(x_cont[:, :, 12:13])

            eps_cont = torch.cat([eps_xyz, eps_rot, eps_scale, eps_pres], dim=-1)
            x_noisy, _ = schedule.add_noise(x_cont, t_idx, eps_cont)

            step_t0 = time.monotonic()
            import contextlib
            amp_ctx = torch.amp.autocast(device_type=device.type, dtype=amp_dtype) if use_amp else contextlib.nullcontext()

            with amp_ctx:
                pred = model(x_noisy, type_ids, t_idx, text_emb_b)
                loss_out = loss_fn(
                    pred,
                    eps_xyz, eps_rot, eps_scale, eps_pres,
                    type_ids, presence,
                )
                loss = loss_out.total / grad_accum

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            grad_accum_count += 1
            if grad_accum_count >= grad_accum:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step()
                _ema_update(ema_weights, model, ema_decay)
                grad_accum_count = 0
                step += 1

                step_elapsed = time.monotonic() - step_t0
                benchmark_step_times.append(step_elapsed)

                epoch_loss += loss_out.total.item()

                if wandb_run is not None:
                    log_dict = {f"loss/{k}": v for k, v in loss_out.as_log_dict().items()}
                    log_dict["train/grad_norm"] = float(grad_norm)
                    log_dict["train/lr"] = lr_scheduler.get_last_lr()[0]
                    log_dict["train/step"] = step
                    wandb_run.log(log_dict, step=step)

                if step % ckpt_steps == 0:
                    save_checkpoint(out_dir, step, model, ema_weights, optimizer, dcfg)
                    print(f"  [step {step}] checkpoint saved")

                if step % 100 == 0:
                    recent = benchmark_step_times[-100:]
                    sps = len(recent) / sum(recent) if recent else 0.0
                    print(f"  step={step:6d} | loss={loss_out.total.item():.4f} | {sps:.1f} steps/s")

        # --- End of epoch ---
        epoch_elapsed = time.monotonic() - epoch_t0

        # Validation pass.
        if (epoch + 1) % val_every == 0:
            vrate = run_validation(model, schedule, ddim_steps, val_n, bounds, device)
            print(f"  [epoch {epoch+1}] validity_rate={vrate:.3f}")
            if wandb_run is not None:
                wandb_run.log({"val/validity_rate": vrate, "train/step": step}, step=step)

        print(f"Epoch {epoch+1}/{n_epochs} | loss={epoch_loss/max(1,steps_per_epoch):.4f} | {epoch_elapsed:.1f}s")

    # --- Benchmark readout ---
    if max_epochs is not None and benchmark_step_times:
        sps = len(benchmark_step_times) / sum(benchmark_step_times)
        remaining_steps = max_steps - step
        projected_s = remaining_steps / max(sps, 1e-6)
        projected_h = projected_s / 3600.0
        print("\n=== 5-EPOCH BENCHMARK ===")
        print(f"  Steps completed:      {len(benchmark_step_times)}")
        print(f"  Throughput:           {sps:.2f} steps/sec")
        print(f"  Remaining steps:      {remaining_steps:,}")
        print(f"  Projected wall-clock: {projected_h:.1f} hours for {max_steps:,} total steps")
        print(f"  Projected wall-clock: {projected_s/60:.0f} min to completion")
        print("=========================")

    # Final checkpoint.
    save_checkpoint(out_dir, step, model, ema_weights, optimizer, dcfg)
    print(f"Training complete at step {step}. Checkpoint saved to {out_dir}/latest.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
