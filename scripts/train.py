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


def _init_wandb(
    cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    git_sha: str,
    resume_run_id: str | None = None,
) -> Any:
    """Initialise W&B run. Raises EnvironmentError if API key is absent.

    When resume_run_id is provided (restored from checkpoint), the run
    is resumed with resume='must' so curves are continuous across restarts.
    """
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
    if resume_run_id:
        run = wandb.init(
            project=wcfg.get("project", "demiurge"),
            id=resume_run_id,
            resume="must",
            tags=wcfg.get("tags", []),
            config={**run_cfg, "git_sha": git_sha},
        )
    else:
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
    wandb_run_id: str | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "ema_state": ema_weights,
        "optimizer_state": optimizer.state_dict(),
        "arch": _arch_fingerprint(dcfg),
        "wandb_run_id": wandb_run_id,  # persists run identity across restarts
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
) -> tuple[int, dict[str, torch.Tensor], str | None]:
    """Load checkpoint; raises ConfigMismatchError if architecture differs.

    Returns (step, ema_weights, wandb_run_id). wandb_run_id is None for
    checkpoints saved before this field was added.
    """
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    saved_arch = ckpt.get("arch", {})
    current_arch = _arch_fingerprint(dcfg)
    if saved_arch != current_arch:
        raise ConfigMismatchError(
            f"Checkpoint architecture {saved_arch} does not match current config "
            f"{current_arch}. Cannot resume -- create a new output directory or "
            "match the config to the checkpoint."
        )
    model.load_state_dict(ckpt["model_state"])  # model already on device; this broadcasts
    optimizer.load_state_dict(ckpt["optimizer_state"])
    # EMA was saved from whichever device the training run used. Move to the
    # current model device so _ema_update doesn't hit a device mismatch.
    model_device = next(model.parameters()).device
    ema = {k: v.to(model_device).float() for k, v in ckpt["ema_state"].items()}
    return int(ckpt["step"]), ema, ckpt.get("wandb_run_id")


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
            # EMA and model are both on the same device (CUDA during GPU runs).
            # The 35MB EMA copy is negligible on a 24GB card.
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
            if key not in text_cache:
                raise KeyError(
                    f"Description hash {key[:12]}... not found in text_embeddings.pt. "
                    "Re-run scripts/precompute_text_embeddings.py or check canonicalization."
                )
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
        # Clamp xyz to normalised workspace bounds [-1, 1] before decode.
        # DDIM accumulation can drift marginally outside this range even with
        # the x0_pred clamp; applying it here prevents Drake from receiving
        # physically impossible coordinates (e.g. y=-0.51m outside workspace).
        xyz = x0[:, :, :3].clamp(-1.0, 1.0)
        rot6d_pred = x0[:, :, 3:9]
        # Normalised scale range: (physical - 1.0) / 0.5, so physical [0.5, 1.5]
        # maps to normalised [-1.0, 1.0]. Clamp in normalised space.
        scale_pred = x0[:, :, 9:12].clamp(-1.0, 1.0)
        pres_bit = x0[:, :, 12]

        for i in range(b):
            pres_mask = pres_bit[i] > 0.0
            quats = rot6d_to_quat_wxyz(rot6d_pred[i])             # (N_MAX, 4)
            poses_raw = torch.cat([xyz[i], quats], dim=-1)        # (N_MAX, 7)
            types = torch.zeros(N_MAX, dtype=torch.long)
            # Drake runs on CPU; move decoded tensors from CUDA to CPU before
            # constructing SceneTensor. denormalize() uses CPU bounds tensors.
            st_norm = SceneTensor(
                object_types=types,
                poses=poses_raw.cpu(),
                scales=scale_pred[i].cpu(),
                presence=pres_mask.cpu(),
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
    # Fix DataLoader multiprocessing on PyTorch nightly + Blackwell (sm_120).
    # The default 'file_descriptor' strategy deadlocks when num_workers > 0;
    # 'file_system' avoids shared-memory fd limits and is stable on Linux.
    import torch.multiprocessing as _mp
    _mp.set_sharing_strategy('file_system')

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
    # Cosine LR warm restart: at cosine_restart_step, jump LR to cosine_restart_lr
    # then decay to 1% of that value by max_steps. Set to None to disable (v1/v2 behavior).
    cosine_restart_step: int | None = tcfg.get("cosine_restart_step", None)
    cosine_restart_lr: float = float(tcfg.get("cosine_restart_lr", 1e-4))
    # W&B log throttle: only log every N steps to reduce Python overhead at high throughput.
    log_every_steps: int = int(tcfg.get("log_every_steps", 1))
    # CFG dropout: probability of replacing text_emb with None per batch.
    # 0.0 = always conditional (conditional_v1 behavior -- caused collapse).
    # 0.15 = 15% unconditional passes, forces model to learn both paths.
    cfg_dropout = float(tcfg.get("cfg_dropout", 0.0))
    # Warm init: load model_state from a pre-trained checkpoint before training.
    # Cross-attention layers stay randomly initialized; all shared layers warm-start.
    warm_init_from: str | None = tcfg.get("warm_init_from", None)

    # --- Loss ---
    lcfg = cfg.get("loss", {})
    loss_weights = LossWeights(
        pose_xyz=float(lcfg.get("pose_xyz", 1.0)),
        pose_rot=float(lcfg.get("pose_rot", 1.0)),
        scale=float(lcfg.get("scale", 0.5)),
        type_ce=float(lcfg.get("type_ce", 0.1)),
        presence_bce=float(lcfg.get("presence_bce", 0.05)),
    )
    # class_weights: inverse-frequency per-class weights for type cross-entropy.
    # Read from training.class_weights as a list of floats in the YAML.
    # Must have exactly N_TYPE entries and must not be all-ones or contain zeros
    # (that would indicate a placeholder was left in the config).
    _cw_raw: list[float] | None = tcfg.get("class_weights", None)
    class_weights_tensor: torch.Tensor | None = None
    if _cw_raw is not None:
        from model.denoiser import N_TYPE as _N_TYPE
        if len(_cw_raw) != _N_TYPE:
            raise ValueError(
                f"training.class_weights has {len(_cw_raw)} entries but N_TYPE={_N_TYPE}. "
                "Run scripts/compute_class_distribution.py to generate the correct values."
            )
        # Assertion: reject placeholder weights (all-ones or any exact 0 or 1).
        if any(w == 0.0 or w == 1.0 for w in _cw_raw):
            raise ValueError(
                f"training.class_weights appears to be a placeholder (contains 0.0 or 1.0): "
                f"{_cw_raw}. Run scripts/compute_class_distribution.py first."
            )
        class_weights_tensor = torch.tensor(_cw_raw, dtype=torch.float32)
        print(f"Class weights loaded: {[f'{w:.3f}' for w in _cw_raw]}")
    loss_fn = SceneDiffusionLoss(loss_weights, class_weights=class_weights_tensor)

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

    # Load text embedding cache.
    # text_conditioning=true in config means this is REQUIRED -- raise if missing.
    text_conditioning = bool(cfg.get("text_conditioning", False))
    text_cache: dict[str, torch.Tensor] | None = None
    cache_path = Path(data_dir) / "text_embeddings.pt"
    if cache_path.exists():
        raw_cache = torch.load(cache_path, weights_only=True)
        # Ensure all embeddings are float32 on CPU for collation.
        text_cache = {k: v.float().cpu() for k, v in raw_cache.items()}
        print(f"Text embedding cache loaded: {len(text_cache)} entries")
    elif text_conditioning:
        raise FileNotFoundError(
            f"text_conditioning=true but no text_embeddings.pt found at {cache_path}. "
            "Run scripts/precompute_text_embeddings.py first."
        )
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
    use_class_balanced = bool(tcfg.get("use_class_balanced_sampler", False))
    if use_class_balanced:
        from data.balanced_sampler import make_class_balanced_sampler
        sampler = make_class_balanced_sampler(all_examples, train_indices, eps=0.01)
        print(f"Using ClassBalancedSampler (type-frequency balanced, eps=0.01)")
    elif weighted_sampling:
        train_families = [all_examples[i][2].get("task_family", "unknown") for i in train_indices]
        fam_counts = {f: train_families.count(f) for f in set(train_families)}
        weights = [1.0 / fam_counts[f] for f in train_families]
        sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_indices), replacement=True)
        print(f"Using template-balanced WeightedRandomSampler")
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
    num_workers = int(dscfg.get("num_workers", 0))
    pin_memory = bool(dscfg.get("pin_memory", device.type == "cuda"))
    persistent_workers = bool(dscfg.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor: int | None = int(dscfg.get("prefetch_factor", 2)) if num_workers > 0 else None

    _base_seed = args.seed

    def _worker_init(worker_id: int) -> None:  # pragma: no cover
        """Seed each DataLoader worker deterministically from the global seed."""
        import random

        import numpy as np
        worker_seed = _base_seed + worker_id
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)
        np.random.seed(worker_seed)

    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: _collate(b, bounds, text_cache),
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=_worker_init if num_workers > 0 else None,
        drop_last=True,
    )
    print(f"DataLoader: num_workers={num_workers} pin_memory={pin_memory} "
          f"persistent_workers={persistent_workers} prefetch_factor={prefetch_factor}")

    # --- Optimizer + LR schedule ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None  # type: ignore[attr-defined]

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        # Cosine warm restart: if configured, apply a second cosine segment
        # from cosine_restart_step to max_steps, starting at cosine_restart_lr.
        if cosine_restart_step is not None and step >= cosine_restart_step:
            # Scale factor relative to base lr so LambdaLR multiplier is correct.
            restart_scale = cosine_restart_lr / max(lr, 1e-12)
            seg_progress = (step - cosine_restart_step) / max(1, max_steps - cosine_restart_step)
            # Decay from restart_lr to 1% of restart_lr.
            return restart_scale * (0.01 + 0.99 * (1.0 + math.cos(math.pi * seg_progress)) / 2.0)
        # Primary cosine decay: lr -> 10% of peak over warmup_steps..cosine_restart_step (or max_steps).
        decay_end = cosine_restart_step if cosine_restart_step is not None else max_steps
        progress = (step - warmup_steps) / max(1, decay_end - warmup_steps)
        progress = min(progress, 1.0)  # clamp: don't decay past restart point
        return 0.1 + 0.9 * (1.0 + math.cos(math.pi * progress)) / 2.0

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Output dir + resume / warm-init ---
    out_dir = Path(cfg.get("output", {}).get("dir", "checkpoints/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    start_step = 0
    ema_weights = _ema_init(model)
    wandb_resume_id: str | None = None
    latest = out_dir / "latest.pt"
    if latest.exists():
        print(f"Resuming from {latest}")
        start_step, ema_weights, wandb_resume_id = load_checkpoint(latest, model, optimizer, dcfg)
        for _ in range(start_step):
            lr_scheduler.step()
        print(f"Resumed at step {start_step}"
              + (f" (W&B run {wandb_resume_id})" if wandb_resume_id else ""))
    elif warm_init_from is not None:
        # Warm-init: copy matching keys from an unconditional checkpoint.
        # Cross-attention layers (key prefix "blocks.") that don't exist in the source
        # are left randomly initialized. This gives the pose/presence/scale heads a
        # head start while the conditioning mechanism learns from scratch.
        wi_path = Path(warm_init_from)
        if not wi_path.exists():
            raise FileNotFoundError(
                f"warm_init_from={warm_init_from!r} does not exist. "
                "Download the checkpoint before starting training."
            )
        wi_ckpt = torch.load(wi_path, map_location="cpu", weights_only=False)
        src_state = wi_ckpt.get("model_state", wi_ckpt)  # handle bare state-dict
        tgt_state = model.state_dict()
        matched, skipped = 0, 0
        for k, v in src_state.items():
            if k in tgt_state and tgt_state[k].shape == v.shape:
                tgt_state[k] = v.to(tgt_state[k].dtype)
                matched += 1
            else:
                skipped += 1
        model.load_state_dict(tgt_state)
        ema_weights = _ema_init(model)  # re-seed EMA from warm weights
        print(f"Warm-init from {wi_path}: {matched} keys loaded, {skipped} skipped (shape/name mismatch)")

    # --- W&B ---
    git_sha = _git_sha()
    wandb_run = _init_wandb(
        cfg,
        {**cfg, "seed": args.seed, "git_sha": git_sha, "n_params": n_params},
        git_sha,
        resume_run_id=wandb_resume_id,
    )

    # --- CFG dropout smoke test ---
    if cfg_dropout > 0.0 and text_cache is not None:
        print(f"CFG dropout={cfg_dropout:.2f} -- running 1000-batch smoke test...")
        _cfg_rng = torch.Generator()
        _cfg_rng.manual_seed(args.seed + 99999)  # isolated generator for smoke test
        _null_count = sum(
            1 for _ in range(1000)
            if torch.rand(1, generator=_cfg_rng).item() < cfg_dropout
        )
        _actual_rate = _null_count / 1000
        _lo, _hi = 0.13, 0.17
        if not (_lo <= _actual_rate <= _hi):
            raise RuntimeError(
                f"CFG dropout smoke test FAILED: rate={_actual_rate:.3f} outside [{_lo}, {_hi}]. "
                f"Check cfg_dropout={cfg_dropout} and RNG seeding."
            )
        print(f"CFG dropout smoke test PASSED: {_null_count}/1000 null batches ({_actual_rate:.3f}) -- within [{_lo}, {_hi}]")

    # Deterministic per-step CFG RNG: seeded from global seed, advances one draw per step.
    # Pattern is fully reproducible: resume from checkpoint gives identical dropout sequence.
    _cfg_step_rng = torch.Generator()
    _cfg_step_rng.manual_seed(args.seed + 12345)

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
                # CFG dropout: replace text conditioning with None with probability
                # cfg_dropout, forcing the model to learn an unconditional fallback.
                # Uses a deterministic per-step generator so dropout patterns are
                # reproducible across runs with the same seed.
                if cfg_dropout > 0.0 and torch.rand(1, generator=_cfg_step_rng).item() < cfg_dropout:
                    text_emb_b = None

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

                if wandb_run is not None and step % log_every_steps == 0:
                    log_dict = {f"loss/{k}": v for k, v in loss_out.as_log_dict().items()}
                    log_dict["train/grad_norm"] = float(grad_norm)
                    log_dict["train/lr"] = lr_scheduler.get_last_lr()[0]
                    log_dict["train/step"] = step
                    wandb_run.log(log_dict, step=step)

                if step % ckpt_steps == 0:
                    save_checkpoint(out_dir, step, model, ema_weights, optimizer, dcfg,
                                    wandb_run_id=wandb_run.id if wandb_run else None)
                    print(f"  [step {step}] checkpoint saved")

                if step % 100 == 0:
                    recent = benchmark_step_times[-100:]
                    sps = len(recent) / sum(recent) if recent else 0.0
                    print(f"  step={step:6d} | loss={loss_out.total.item():.4f} | {sps:.1f} steps/s")

        # --- End of epoch ---
        epoch_elapsed = time.monotonic() - epoch_t0

        # Validation pass.
        if (epoch + 1) % val_every == 0 and val_n > 0:
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
    save_checkpoint(out_dir, step, model, ema_weights, optimizer, dcfg,
                    wandb_run_id=wandb_run.id if wandb_run else None)
    print(f"Training complete at step {step}. Checkpoint saved to {out_dir}/latest.pt")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
