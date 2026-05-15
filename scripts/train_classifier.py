"""Train the noise-conditioned Drake validity classifier (Phase B).

Trains ValidityClassifier on data/classifier_v1/ shards.
Accepts --config pointing at a YAML and --seed for determinism.

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 scripts/train_classifier.py \
        --config configs/classifier/classifier_v1.yaml \
        --seed 42

Halt conditions:
    - AUC < 0.85 after one epoch: raises RuntimeError with diagnostic info.
    - Shard directory missing or empty: raises FileNotFoundError.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from guidance.classifier import ValidityClassifier  # noqa: E402
from model.denoiser import N_MAX  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_shard_dir(shard_dir: Path) -> list[dict]:
    """Load all classifier shards into memory.

    Each sample: {'xt': Tensor(N_MAX, N_CONT), 't': int, 'label': int}.
    """
    import webdataset as wds

    urls = sorted(str(p) for p in shard_dir.glob("cls-*.tar"))
    if not urls:
        raise FileNotFoundError(
            f"No classifier shards found in {shard_dir}. "
            "Run scripts/generate_classifier_data.py first."
        )

    samples = []
    ds = wds.WebDataset(urls, nodesplitter=wds.shardlists.single_node_only)
    for sample in ds:
        xt = torch.load(io.BytesIO(sample["xt.pt"]), weights_only=True)
        t = int(torch.load(io.BytesIO(sample["t.pt"]), weights_only=True).item())
        label = int(torch.load(io.BytesIO(sample["label.pt"]), weights_only=True).item())
        samples.append({"xt": xt, "t": t, "label": label})
    return samples


def _build_loaders(
    samples: list[dict],
    val_frac: float,
    batch_size: int,
    seed: int,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Split samples into train/val and build DataLoaders."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    perm = torch.randperm(len(samples), generator=rng).tolist()
    n_val = max(1, int(len(samples) * val_frac))
    val_idx = set(perm[:n_val])
    train_idx = [i for i in range(len(samples)) if i not in val_idx]
    val_list = [perm[i] for i in range(n_val)]

    def _collate(batch: list[dict]) -> dict:
        xts = torch.stack([s["xt"] for s in batch])         # (B, N_MAX, 13)
        ts = torch.tensor([s["t"] for s in batch])          # (B,)
        labels = torch.tensor([s["label"] for s in batch], dtype=torch.float32)  # (B,)
        type_ids = torch.zeros(len(batch), N_MAX, dtype=torch.long)
        return {"xt": xts, "t": ts, "label": labels, "type_ids": type_ids}

    class _ListDS(torch.utils.data.Dataset):
        def __init__(self, data: list, idx: list[int]) -> None:
            self._data = data
            self._idx = idx
        def __len__(self) -> int:
            return len(self._idx)
        def __getitem__(self, i: int) -> dict:
            return self._data[self._idx[i]]

    train_ds = _ListDS(samples, train_idx)
    val_ds = _ListDS(samples, val_list)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# AUC evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_auc(
    model: ValidityClassifier,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> float:
    """Compute ROC-AUC on validation set."""
    from sklearn.metrics import roc_auc_score  # type: ignore[import]

    model.eval()
    all_logits = []
    all_labels = []
    for batch in val_loader:
        xt = batch["xt"].to(device)
        t = batch["t"].to(device)
        type_ids = batch["type_ids"].to(device)
        labels = batch["label"]

        logit = model(xt, type_ids, t)
        all_logits.append(logit.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits).float().numpy()
    labels = torch.cat(all_labels).numpy()
    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    return float(roc_auc_score(labels, probs))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.config) as f:
        cfg: dict = yaml.safe_load(f) or {}

    dcfg = cfg.get("data", {})
    shard_dir = Path(dcfg.get("shard_dir", "data/classifier_v1"))
    val_frac = float(dcfg.get("val_frac", 0.2))

    tcfg = cfg.get("training", {})
    batch_size = int(tcfg.get("batch_size", 256))
    lr = float(tcfg.get("lr", 3e-4))
    warmup_steps = int(tcfg.get("warmup_steps", 500))
    auc_threshold = float(tcfg.get("auc_threshold", 0.85))

    mcfg = cfg.get("model", {})
    d_model = int(mcfg.get("d_model", 128))
    n_layers = int(mcfg.get("n_layers", 4))
    n_heads = int(mcfg.get("n_heads", 4))

    out_dir = Path(cfg.get("output", {}).get("dir", "checkpoints/classifier_v1"))
    out_dir.mkdir(parents=True, exist_ok=True)

    wcfg = cfg.get("wandb", {})
    wandb_run = None
    if wcfg.get("enabled", False):
        api_key = os.environ.get("WANDB_API_KEY", "")
        if not api_key:
            raise OSError("wandb.enabled=true but WANDB_API_KEY not set.")
        import wandb
        wandb_run = wandb.init(
            project=wcfg.get("project", "demiurge"),
            name=wcfg.get("experiment", "classifier_v1"),
            tags=wcfg.get("tags", []),
            config={**cfg, "seed": args.seed},
        )

    print(f"Loading classifier data from {shard_dir} ...")
    samples = _load_shard_dir(shard_dir)
    print(f"Loaded {len(samples)} samples.")

    train_loader, val_loader = _build_loaders(samples, val_frac, batch_size, args.seed)
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")

    model = ValidityClassifier(d_model=d_model, n_layers=n_layers, n_heads=n_heads).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ValidityClassifier: {n_params/1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # Linear warmup scheduler.
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # --- Training loop (one epoch) ---
    model.train()
    step = 0
    t0 = time.monotonic()
    total_loss = 0.0

    for batch in train_loader:
        xt = batch["xt"].to(device)
        t_idx = batch["t"].to(device)
        type_ids = batch["type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logit = model(xt, type_ids, t_idx)
        loss = criterion(logit, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        step += 1

        if step % 100 == 0:
            avg_loss = total_loss / step
            print(f"  step={step:5d} | loss={avg_loss:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")
            if wandb_run is not None:
                wandb_run.log({"train/loss": avg_loss, "step": step})

    elapsed = time.monotonic() - t0
    print(f"\nTraining complete: {step} steps in {elapsed:.1f}s")

    # --- Evaluation ---
    auc = _compute_auc(model, val_loader, device)
    print(f"Held-out AUC: {auc:.4f} (threshold: {auc_threshold})")

    if wandb_run is not None:
        wandb_run.log({"val/auc": auc})

    if auc < auc_threshold:
        raise RuntimeError(
            f"Classifier AUC {auc:.4f} is below threshold {auc_threshold}. "
            "HALT: do not proceed to Phase B guidance sweep. "
            "Possible causes: insufficient training data, too few epochs, "
            "or corrupted data generation (check APPROX note in generate_classifier_data.py)."
        )

    # --- Save checkpoint ---
    ckpt = {
        "step": step,
        "model_state": model.state_dict(),
        "auc": auc,
        "config": cfg,
        "seed": args.seed,
    }
    ckpt_path = out_dir / "latest.pt"
    torch.save(ckpt, ckpt_path)
    print(f"Checkpoint saved to {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
