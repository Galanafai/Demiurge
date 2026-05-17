"""Build the held-out evaluation suite for downstream Demiurge evaluation.

Samples 200 unique task descriptions from data/v1, stratified by template
family, validates a reference scene for each, and persists the suite to
data/eval_suite_v1/eval_suite.jsonl.

FIXED BENCHMARK INVARIANT: If eval_suite.jsonl already exists this script
aborts immediately with a non-zero exit code. Do NOT regenerate after first
creation; it invalidates all previously reported metrics.

Usage:
    python3 scripts/build_eval_suite.py \
        --data-dir data/v1 \
        --out-dir data/eval_suite_v1 \
        --n-prompts 200 \
        --seed 7919
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.reader import ShardReader  # noqa: E402
from data.sampler import ProceduralSampler  # noqa: E402
from scene.schema import WorkspaceBounds  # noqa: E402
from validator.core import SceneValidator  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default="data/v1")
    p.add_argument("--out-dir", default="data/eval_suite_v1")
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--seed", type=int, default=7919)
    p.add_argument("--rrt-budget", type=float, default=2.0)
    return p.parse_args()


def collect_held_out(data_dir: Path, n_prompts: int, seed: int) -> list[dict]:
    """Collect unique task descriptions, stratified by template family.

    First attempts to draw from the last 100 scenes per template in data_dir
    (ShardReader). If fewer than n_prompts unique descriptions are available
    there, supplements by drawing from ProceduralSampler directly until
    n_prompts are collected.
    """
    import torch
    from data.sampler import ProceduralSampler

    per_template: dict[str, list[dict]] = defaultdict(list)
    try:
        for scene, desc, report, _ in ShardReader(data_dir):
            if not desc:
                continue
            tmpl = report.get("task_family", "default") if isinstance(report, dict) else "default"
            per_template[tmpl].append({"description": desc, "template": tmpl})
    except Exception as e:
        print(f"  ShardReader warning: {e}", file=sys.stderr)

    rng = torch.Generator()
    rng.manual_seed(seed)

    sampled: list[dict] = []
    if per_template:
        total = sum(len(v) for v in per_template.values())
        for tmpl in sorted(per_template):
            items = per_template[tmpl][-100:]  # held-out tail
            n_take = max(1, round(n_prompts * len(items) / total))
            idxs = torch.randperm(len(items), generator=rng).tolist()[:n_take]
            sampled.extend(items[i] for i in idxs)

    # Deduplicate
    seen: set[str] = set()
    deduped: list[dict] = []
    for item in sampled:
        key = hashlib.sha256(item["description"].encode()).hexdigest()[:16]
        if key not in seen:
            seen.add(key)
            item["desc_id"] = key
            deduped.append(item)

    # Supplement from ProceduralSampler if needed
    if len(deduped) < n_prompts:
        import numpy as np
        from data.descriptions import generate_description
        from data.sampler import ProceduralSampler

        proc_seed = seed ^ 0xDEAD
        ps = ProceduralSampler(seed=proc_seed)
        rng = np.random.default_rng(proc_seed)
        print(f"  ShardReader yielded {len(deduped)} - supplementing with ProceduralSampler ...")
        attempts = 0
        while len(deduped) < n_prompts and attempts < n_prompts * 20:
            attempts += 1
            try:
                cand = ps.sample()
                desc = generate_description(cand, rng)
                if not desc:
                    continue
                key = hashlib.sha256(desc.encode()).hexdigest()[:16]
                if key not in seen:
                    seen.add(key)
                    tmpl = str(getattr(cand, "task_family", "procedural"))
                    deduped.append({"description": desc, "template": tmpl, "desc_id": key})
            except Exception:
                pass

    print(f"  Total unique descriptions: {len(deduped[:n_prompts])}")
    return deduped[:n_prompts]


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    suite_path = out_dir / "eval_suite.jsonl"

    # FIXED BENCHMARK GUARD
    if suite_path.exists():
        print(
            f"ERROR: {suite_path} already exists. "
            "Do not regenerate the held-out eval suite after first creation. "
            "Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    print(f"Building held-out suite from {data_dir} ...")
    held_out = collect_held_out(data_dir, args.n_prompts, args.seed)
    print(f"  Collected {len(held_out)} unique descriptions")

    validator = SceneValidator(
        workspace_bounds=WorkspaceBounds.default(),
        rrt_budget_s=args.rrt_budget,
    )
    sampler = ProceduralSampler(seed=args.seed)
    bounds = WorkspaceBounds.default()

    print("Generating reference scenes ...")
    records: list[dict] = []
    n_failed = 0
    t0 = time.monotonic()

    for i, item in enumerate(held_out):
        desc = item["description"]
        desc_id = item["desc_id"]
        template = item["template"]

        # Sample candidates until one validates or we give up after 20 attempts.
        reference = None
        for attempt in range(20):
            candidate = sampler.sample()
            st_phys = candidate.scene.denormalize(bounds)
            report = validator.validate(st_phys)
            if report.accepted:
                reference = {
                    "ik_solution": getattr(report, "ik_solution", None),
                    "rrt_plan_length": getattr(report, "rrt_plan_length", None),
                    "scene_tensor": {
                        "object_types": candidate.scene.object_types.tolist(),
                        "poses": candidate.scene.poses.tolist(),
                        "scales": candidate.scene.scales.tolist(),
                        "presence": candidate.scene.presence.tolist(),
                    },
                }
                break

        if reference is None:
            n_failed += 1
            reference = {}  # suite entry still written; reference_scene is empty

        record = {
            "desc_id": desc_id,
            "description": desc,
            "template": template,
            "reference": reference,
        }
        records.append(record)

        if (i + 1) % 25 == 0:
            elapsed = time.monotonic() - t0
            print(f"  [{i+1}/{len(held_out)}] elapsed={elapsed:.0f}s failed={n_failed}")

    with open(suite_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    elapsed = time.monotonic() - t0
    print(f"\nSuite written to {suite_path}")
    print(f"  {len(records)} entries, {n_failed} without valid reference scene")
    print(f"  Total time: {elapsed:.0f}s")

    # Write metadata
    meta = {
        "n_prompts": len(records),
        "n_failed_reference": n_failed,
        "seed": args.seed,
        "data_dir": str(data_dir),
        "rrt_budget_s": args.rrt_budget,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
