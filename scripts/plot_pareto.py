"""Plot Pareto frontier from Phase B guidance sweep results.

Reads per-run JSON files from artifacts/guidance_sweep_v1/ and produces
a validity vs. diversity Pareto plot with error bars across 3 seeds.

Usage:
    python3 scripts/plot_pareto.py \\
        --sweep-dir artifacts/guidance_sweep_v1/ \\
        --output artifacts/pareto_v1.png \\
        --include-baselines
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_results(sweep_dir: Path) -> list[dict]:
    results = []
    for f in sorted(sweep_dir.glob("*.json")):
        if f.name == "sweep_summary.json":
            continue
        try:
            results.append(json.loads(f.read_text()))
        except Exception as e:
            print(f"WARNING: could not read {f}: {e}")
    return results


def _aggregate(results: list[dict]) -> list[dict]:
    """Aggregate across seeds: mean +/- std per (w, mode) config."""
    groups: dict = defaultdict(list)
    for r in results:
        key = (float(r["w"]), r["mode"])
        groups[key].append(r)

    agg = []
    for (w, mode), runs in sorted(groups.items()):
        vals = [r["validity_rate"] for r in runs]
        divs = [r["type_diversity"] for r in runs]
        t0fs = [r.get("type0_frac", float("nan")) for r in runs]
        agg.append({
            "w": w,
            "mode": mode,
            "n_seeds": len(runs),
            "validity_mean": float(np.mean(vals)),
            "validity_std": float(np.std(vals)),
            "diversity_mean": float(np.mean(divs)),
            "diversity_std": float(np.std(divs)),
            "type0_frac_mean": float(np.nanmean(t0fs)),
        })
    return agg


def _pareto_front(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of Pareto-dominant points (higher validity AND diversity is better)."""
    n = len(points)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if points[j][0] >= points[i][0] and points[j][1] >= points[i][1]:
                if points[j][0] > points[i][0] or points[j][1] > points[i][1]:
                    dominated[i] = True
                    break
    return [i for i in range(n) if not dominated[i]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", default="artifacts/guidance_sweep_v1/")
    p.add_argument("--output", default="artifacts/pareto_v1.png")
    p.add_argument("--output-pdf", default="")
    p.add_argument("--include-baselines", action="store_true")
    args = p.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available. Install with: pip install matplotlib")
        sys.exit(1)

    sweep_dir = Path(args.sweep_dir)
    if not sweep_dir.exists():
        print(f"ERROR: sweep dir not found: {sweep_dir}")
        sys.exit(1)

    results = _load_results(sweep_dir)
    if not results:
        print("ERROR: no per-run JSON files found in sweep dir.")
        sys.exit(1)

    agg = _aggregate(results)

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_xlabel("Type Diversity (mean distinct types / 12)", fontsize=12)
    ax.set_ylabel("Drake Validity Rate", fontsize=12)
    ax.set_title("Phase B: Classifier Guidance Pareto Frontier\nDemiurge v6 — conditional scene generation", fontsize=13)

    mode_colors = {"continuous": "#4C9BE8", "full": "#E86B4C"}
    mode_markers = {"continuous": "o", "full": "s"}

    points_xy = [(a["diversity_mean"], a["validity_mean"]) for a in agg]
    pareto_idx = set(_pareto_front(points_xy))

    for i, a in enumerate(agg):
        w, mode = a["w"], a["mode"]
        color = mode_colors.get(mode, "#888888")
        marker = mode_markers.get(mode, "D")
        alpha = 1.0 if i in pareto_idx else 0.55

        ax.errorbar(
            a["diversity_mean"], a["validity_mean"],
            xerr=a["diversity_std"], yerr=a["validity_std"],
            fmt=marker, color=color, markersize=10,
            alpha=alpha, capsize=3, linewidth=1.2,
            label=f"w={w} {mode}" if a["n_seeds"] >= 1 else None,
        )
        ax.annotate(
            f"w={w}",
            (a["diversity_mean"], a["validity_mean"]),
            textcoords="offset points", xytext=(5, 3),
            fontsize=7, alpha=alpha,
        )

    # Draw Pareto frontier line.
    pf_pts = sorted(
        [points_xy[i] for i in pareto_idx],
        key=lambda xy: xy[0],
    )
    if pf_pts:
        pf_x, pf_y = zip(*pf_pts)
        ax.step(pf_x, pf_y, where="post", color="black", linewidth=1.5,
                linestyle="--", alpha=0.7, label="Pareto frontier")

    # Baseline references.
    if args.include_baselines:
        baselines = [
            ("v2 (broken sampler)", 0.098, None, "#999999"),
            ("v4 (broken sampler)", 0.075, None, "#bbbbbb"),
            ("v6 (rejection, uniform)", 0.065, None, "#2ECC71"),
        ]
        for label, val, div, bc in baselines:
            if div is not None:
                ax.axhline(val, color=bc, linestyle=":", alpha=0.5, linewidth=1.0)
            ax.axhline(val, color=bc, linestyle=":", alpha=0.5, linewidth=1.0, label=label)

    # Legend.
    cont_patch = mpatches.Patch(color=mode_colors["continuous"], label="mode=continuous")
    full_patch = mpatches.Patch(color=mode_colors["full"], label="mode=full")
    ax.legend(handles=[cont_patch, full_patch], loc="lower right", fontsize=9)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.1%}"))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"Pareto plot saved to {out_path}")

    if args.output_pdf:
        fig.savefig(args.output_pdf, format="pdf", bbox_inches="tight")
        print(f"PDF saved to {args.output_pdf}")

    # Print table.
    print(f"\n{'Config':<25} {'Valid':>7} {'±':>5} {'Diversity':>10} {'±':>5} {'type0':>6} {'Pareto':>7}")
    print("-" * 75)
    for i, a in enumerate(agg):
        pareto_mark = "**" if i in pareto_idx else ""
        print(
            f"w={a['w']:.1f} {a['mode']:<12}"
            f"  {a['validity_mean']:>6.1%}"
            f"  {a['validity_std']:>4.1%}"
            f"  {a['diversity_mean']:>9.3f}"
            f"  {a['diversity_std']:>4.3f}"
            f"  {a['type0_frac_mean']:>5.1%}"
            f"  {pareto_mark:>7}"
        )


if __name__ == "__main__":
    main()
