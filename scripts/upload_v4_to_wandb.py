"""Upload v4 checkpoints and sweep results to W&B as artifacts.

Usage:
    python3 scripts/upload_v4_to_wandb.py

Uploads:
  - All step_*.pt checkpoints from checkpoints/v9_uncond_v4/
  - All v4_sweep_*.json probe results from artifacts/
  - The ema_vs_live_test.log if it exists
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

import wandb

WANDB_PROJECT   = "demiurge"
WANDB_ENTITY    = "galanafai-self"
RUN_ID          = "980i4spp"          # existing v4 run from training
CKPT_DIR        = _ROOT / "checkpoints" / "v9_uncond_v4"
ARTIFACTS_DIR   = _ROOT / "artifacts"
LOGS_DIR        = _ROOT / "logs"

def main() -> None:
    print(f"Resuming W&B run {RUN_ID} ...")
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        id=RUN_ID,
        resume="must",
    )

    # 1. Checkpoints artifact
    print("\n--- Uploading checkpoints ---")
    ckpt_art = wandb.Artifact(
        name="v9_uncond_v4_checkpoints",
        type="model",
        description="All 10 step checkpoints for v9_uncond_v4 (zero-centered prior fix)",
        metadata={"train_config": "configs/train/v9_uncond_v4.yaml"},
    )
    ckpts = sorted(CKPT_DIR.glob("step_*.pt"))
    for p in ckpts:
        ckpt_art.add_file(str(p), name=p.name)
        print(f"  + {p.name}  ({p.stat().st_size / 1e9:.2f} GB)")
    run.log_artifact(ckpt_art)
    print(f"  => Uploaded {len(ckpts)} checkpoints")

    # 2. Sweep results artifact
    print("\n--- Uploading sweep probe results ---")
    sweep_art = wandb.Artifact(
        name="v9_uncond_v4_sweep",
        type="evaluation",
        description="Drake validity probe at every 10k checkpoint for v4",
    )
    sweep_files = sorted(ARTIFACTS_DIR.glob("v4_sweep_*.json"))
    for p in sweep_files:
        sweep_art.add_file(str(p), name=p.name)
    if sweep_files:
        print(f"  + {len(sweep_files)} probe JSONs")

    # Attach sweep summary as a table
    summary_rows = []
    for p in sweep_files:
        step = int(p.stem.split("_")[-1])
        try:
            d = json.loads(p.read_text())
            summary_rows.append({
                "step": step,
                "validity_pct": round(d["validity_rate"] * 100, 1),
                "n_accepted": d.get("n_accepted", 0),
                "n_scenes": d.get("n_scenes", 100),
                "interpenetration": d.get("reject_counts", {}).get("interpenetration", None),
                "rrt_failed": d.get("reject_counts", {}).get("rrt_failed", None),
            })
        except Exception as e:
            print(f"  WARN: could not parse {p.name}: {e}")

    if summary_rows:
        tbl = wandb.Table(
            columns=list(summary_rows[0].keys()),
            data=[list(r.values()) for r in summary_rows],
        )
        run.log({"v4_sweep/validity_table": tbl})
        # Also log the validity trajectory as a line chart
        for row in summary_rows:
            run.log({"v4_sweep/validity_pct": row["validity_pct"], "v4_sweep/step": row["step"]})

    run.log_artifact(sweep_art)

    # 3. EMA vs live test log (if it exists)
    ema_log = LOGS_DIR / "ema_vs_live_test.log"
    if ema_log.exists():
        print("\n--- Uploading EMA vs live test log ---")
        diag_art = wandb.Artifact(
            name="v9_uncond_v4_ema_vs_live",
            type="evaluation",
            description="EMA vs live weights diagnostic at steps 20k/40k/100k",
        )
        diag_art.add_file(str(ema_log), name=ema_log.name)
        run.log_artifact(diag_art)
        print(f"  + {ema_log.name}")

    # 4. Log sweep summary metrics directly to the run
    if summary_rows:
        peak = max(summary_rows, key=lambda r: r["validity_pct"])
        run.summary["v4_peak_validity_pct"] = peak["validity_pct"]
        run.summary["v4_peak_validity_step"] = peak["step"]
        run.summary["v4_final_validity_pct"] = summary_rows[-1]["validity_pct"]
        print(f"\nSummary: peak={peak['validity_pct']}% @ step {peak['step']}, "
              f"final={summary_rows[-1]['validity_pct']}%")

    run.finish()
    print("\nDone. Check https://wandb.ai/galanafai-self/demiurge")


if __name__ == "__main__":
    main()
