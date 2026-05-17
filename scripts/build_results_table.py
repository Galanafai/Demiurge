"""Build results table from eval harness outputs.

Reads all artifacts/eval_v1/<sampler_name>.json files, computes mean +/- std
across seeds for each metric, and writes artifacts/results_table.md.

Usage:
    python3 scripts/build_results_table.py \
        --eval-dir artifacts/eval_v1 \
        --out artifacts/results_table.md
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-dir", default="artifacts/eval_v1")
    p.add_argument("--out", default="artifacts/results_table.md")
    return p.parse_args()


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    n = len(values)
    mu = sum(values) / n
    if n < 2:
        return mu, 0.0
    var = sum((v - mu) ** 2 for v in values) / (n - 1)
    return mu, math.sqrt(var)


METRICS = [
    ("validity_rate", "Validity (%)", 100.0),
    ("diversity",     "Diversity",    1.0),
]

# Display order for samplers
SAMPLER_ORDER = [
    "v7_uncond",
    "v7_cond_baseline",
    "v7_rejection",
    "v7_ug_best",
    "v7_ug_scale1",
]


def load_results(eval_dir: Path) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    for path in sorted(eval_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        name = data.get("sampler", path.stem)
        results[name] = data.get("runs", [])
    return results


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = load_results(eval_dir)
    if not results:
        print(f"No JSON files found in {eval_dir}")
        return

    # Compute per-sampler mean +/- std for each metric
    table: dict[str, dict[str, tuple[float, float]]] = {}
    for name, runs in results.items():
        table[name] = {}
        for key, _, scale in METRICS:
            vals = [r.get(key, 0.0) * scale for r in runs if key in r]
            table[name][key] = _mean_std(vals)

    # Find best value per metric column (highest is best for all current metrics)
    best: dict[str, float] = {}
    for key, _, _ in METRICS:
        best[key] = max(
            table[name][key][0]
            for name in table
            if key in table[name]
        )

    def fmt(mu: float, std: float, key: str, scale: float) -> str:
        """Format as 'X.XX +/- Y.YY', bolded if best."""
        if scale == 100.0:
            s = f"{mu:.1f} +/- {std:.1f}"
        else:
            s = f"{mu:.4f} +/- {std:.4f}"
        if abs(mu - best[key]) < 1e-6:
            return f"**{s}**"
        return s

    # Ordered sampler list
    ordered = [n for n in SAMPLER_ORDER if n in table]
    # Append any unexpected names not in SAMPLER_ORDER
    for n in sorted(table):
        if n not in ordered:
            ordered.append(n)

    # Build markdown
    metric_headers = " | ".join(label for _, label, _ in METRICS)
    header = f"| Sampler | {metric_headers} | Seeds |"
    sep_parts = ["---"] * (len(METRICS) + 2)
    sep = "| " + " | ".join(sep_parts) + " |"

    rows = [header, sep]
    for name in ordered:
        if name not in table:
            continue
        n_seeds = len(results[name])
        cells = []
        for key, _, scale in METRICS:
            mu, std = table[name].get(key, (0.0, 0.0))
            cells.append(fmt(mu, std, key, scale))
        rows.append(f"| `{name}` | {' | '.join(cells)} | {n_seeds} |")

    # Compose full document
    lines = [
        "# Demiurge Week 5: Evaluation Results",
        "",
        "Metrics are mean +/- std across seeds. "
        "Bold = best in column. Validity shown as percentage.",
        "",
        "\n".join(rows),
        "",
        "## Notes",
        "",
        "- `v7_uncond`: conditional_v7, CFG=0 (unconditional baseline)",
        "- `v7_cond_baseline`: conditional_v7, CFG=1.0, no Universal Guidance",
        "- `v7_rejection`: v7 conditioned, rejection sampling (generate 4x, keep valid)",
        "- `v7_ug_best`: v7 conditioned with Universal Guidance, scale=0.5 (best from sweep)",
        "- `v7_ug_scale1`: v7 conditioned with Universal Guidance, scale=1.0",
    ]

    content = "\n".join(lines) + "\n"
    with open(out_path, "w") as f:
        f.write(content)

    print(f"Results table written to {out_path}")
    print()
    # Also print to stdout for quick review
    print(content)


if __name__ == "__main__":
    main()
