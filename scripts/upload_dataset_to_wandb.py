"""Upload Demiurge v1 dataset artifacts to W&B.

Run once from the laptop whenever the dataset changes, before spinning
up a new pod. The pod-side download is handled by scripts/pod_bootstrap.sh.

Usage:
    uv run python scripts/upload_dataset_to_wandb.py [--data-dir data/v1]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import wandb


def _upload(
    local_path: str,
    artifact_name: str,
    artifact_type: str,
    desc: str,
    *,
    entity: str,
    project: str,
) -> None:
    p = pathlib.Path(local_path)
    if not p.exists():
        raise FileNotFoundError(f"{local_path} not found -- run precompute_text_embeddings.py first")
    sz = p.stat().st_size / 1e6
    print(f"Uploading {p.name}  ({sz:.0f} MB) -> {entity}/{project}/{artifact_name}:latest ...")
    with wandb.init(
        project=project,
        entity=entity,
        job_type="dataset-upload",
        name=f"upload-{p.name}",
        settings=wandb.Settings(silent=True),
    ) as run:
        art = wandb.Artifact(artifact_name, type=artifact_type, description=desc)
        art.add_file(local_path, name=p.name)
        run.log_artifact(art, aliases=["latest"])
    print(f"  OK: {artifact_name}:latest")


def main() -> None:
    ap = argparse.ArgumentParser(description="Upload v1 dataset to W&B Artifacts")
    ap.add_argument("--data-dir", default="data/v1", help="Path to data/v1 directory")
    ap.add_argument("--entity", default="galanafai-self")
    ap.add_argument("--project", default="demiurge")
    ap.add_argument("--skip-shards", action="store_true", help="Skip tarball upload (fast re-run)")
    args = ap.parse_args()

    d = pathlib.Path(args.data_dir)
    kw = dict(entity=args.entity, project=args.project)

    if not args.skip_shards:
        shards = sorted(d.glob("*.tar"))
        if not shards:
            print(f"ERROR: no .tar shards found in {d}. Aborting.", file=sys.stderr)
            sys.exit(1)
        for shard in shards:
            stem = shard.stem.replace("-", "_")  # v1-00001 -> v1_00001
            _upload(str(shard), f"v1_shard_{stem.split('_')[-1]}",
                    "dataset", f"Demiurge v1 shard {stem}", **kw)

    _upload(str(d / "text_embeddings.pt"), "v1_text_embeddings",
            "dataset", "Demiurge v1 text embedding cache (384-dim float32)", **kw)
    _upload(str(d / "manifest.json"), "v1_manifest",
            "dataset", "Demiurge v1 dataset manifest", **kw)

    print("\nAll artifacts uploaded. Pod can now run:")
    print("  bash scripts/pod_bootstrap.sh")


if __name__ == "__main__":
    main()
