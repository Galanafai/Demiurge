"""Pareto plot for Universal Guidance sweep.

Reads summary.jsonl from the sweep and VLM scores, produces:
- Drake validity vs diversity Pareto plot
- Annotated with guidance scale and seed error bars
- Baselines: v7 unconditional (6.5%), v7 conditioned no guidance (1.1%)

Usage:
    python3 scripts/plot_ug_pareto.py \\
        --sweep-dir artifacts/ug_sweep_v1/ \\
        --vlm-scores artifacts/ug_sweep_v1/vlm_scores.jsonl \\
        --output-png artifacts/pareto_v1.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--vlm-scores", default=None)
    p.add_argument("--output-png", required=True)
    p.add_argument("--output-pdf", default=None)
    return p.parse_args()


def load_summary(sweep_dir: Path) -> list[dict]:
    rows = []
    with open(sweep_dir / "summary.jsonl") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def load_vlm_scores(path: Path) -> dict[tuple[float, int], float]:
    """Return {(guidance_scale, seed): mean_vlm_score}."""
    scores: dict[tuple[float, int], list[float]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["guidance_scale"], rec["seed"])
            scores[key].append(rec["vlm_score"])
    return {k: sum(v) / len(v) for k, v in scores.items()}


def is_pareto_optimal(points: list[tuple[float, float]]) -> list[bool]:
    """Return boolean mask of Pareto-optimal points (maximize both axes)."""
    n = len(points)
    optimal = [True] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Point i is dominated if j >= i on both axes and > on at least one
            if (points[j][0] >= points[i][0] and points[j][1] >= points[i][1]
                    and (points[j][0] > points[i][0] or points[j][1] > points[i][1])):
                optimal[i] = False
                break
    return optimal


def main() -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.cm as cm
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available -- install with: uv add matplotlib")
        raise

    args = parse_args()
    sweep_dir = Path(args.sweep_dir)

    rows = load_summary(sweep_dir)
    vlm_scores: dict = {}
    if args.vlm_scores and Path(args.vlm_scores).exists():
        vlm_scores = load_vlm_scores(Path(args.vlm_scores))

    # Aggregate by guidance_scale (mean across seeds, std for error bars)
    by_scale: dict[float, dict] = defaultdict(lambda: {"validity": [], "diversity": [], "energy": [], "vlm": []})
    for row in rows:
        gs = row["guidance_scale"]
        by_scale[gs]["validity"].append(row["drake_validity_rate"])
        by_scale[gs]["diversity"].append(row["diversity"])
        by_scale[gs]["energy"].append(row["mean_energy"])
        key = (gs, row["seed"])
        if key in vlm_scores:
            by_scale[gs]["vlm"].append(vlm_scores[key])

    scales = sorted(by_scale.keys())
    mean_validity = [np.mean(by_scale[s]["validity"]) for s in scales]
    std_validity = [np.std(by_scale[s]["validity"]) for s in scales]
    mean_diversity = [np.mean(by_scale[s]["diversity"]) for s in scales]
    std_diversity = [np.std(by_scale[s]["diversity"]) for s in scales]
    mean_vlm = [np.mean(by_scale[s]["vlm"]) if by_scale[s]["vlm"] else None for s in scales]

    # Pareto frontier (validity, diversity)
    points = list(zip(mean_validity, mean_diversity))
    pareto_mask = is_pareto_optimal(points)

    # Color by guidance scale
    colors = cm.viridis(np.linspace(0.1, 0.9, len(scales)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Universal Guidance: Pareto Sweep (v7, cfg=1.0)", fontsize=14, fontweight="bold")

    # --- Left: Drake validity vs diversity ---
    ax = axes[0]
    for i, (s, mv, sv, md, sd, is_opt) in enumerate(
        zip(scales, mean_validity, std_validity, mean_diversity, std_diversity, pareto_mask)
    ):
        ax.errorbar(md, mv * 100, xerr=sd, yerr=sv * 100,
                    fmt="o", color=colors[i], markersize=10,
                    capsize=4, label=f"UG scale={s}")
        ax.annotate(f"s={s}", (md, mv * 100), textcoords="offset points",
                    xytext=(5, 5), fontsize=8)
        if is_opt:
            ax.scatter([md], [mv * 100], s=200, facecolors="none",
                       edgecolors="gold", linewidths=2, zorder=5)

    # Baselines
    ax.axhline(y=6.5, color="gray", linestyle="--", alpha=0.6, label="v7 unconditional (6.5%)")
    ax.axhline(y=1.1, color="red", linestyle="--", alpha=0.6, label="v7 conditioned, no guidance (1.1%)")

    ax.set_xlabel("Mean Scene Diversity (centroid distance)")
    ax.set_ylabel("Drake Validity Rate (%)")
    ax.set_title("Drake Validity vs Diversity")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # --- Right: VLM score vs validity (if available) ---
    ax2 = axes[1]
    if any(v is not None for v in mean_vlm):
        for i, (s, mv, sv, vlm) in enumerate(zip(scales, mean_validity, std_validity, mean_vlm)):
            if vlm is not None:
                ax2.errorbar(mv * 100, vlm, xerr=sv * 100,
                             fmt="o", color=colors[i], markersize=10,
                             capsize=4, label=f"UG scale={s}")
                ax2.annotate(f"s={s}", (mv * 100, vlm), textcoords="offset points",
                             xytext=(5, 5), fontsize=8)
        ax2.axhline(y=2.0, color="green", linestyle="--", alpha=0.6, label="Gate 1 threshold (2.0)")
        ax2.set_xlabel("Drake Validity Rate (%)")
        ax2.set_ylabel("VLM Mean Score (1-5)")
        ax2.set_title("VLM Score vs Drake Validity")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "VLM scores not yet available.\nRun judge_sweep_with_vlm.py first.",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=12)
        ax2.set_title("VLM Score vs Drake Validity (pending)")

    plt.tight_layout()

    out_png = Path(args.output_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved PNG: {out_png}")

    if args.output_pdf:
        plt.savefig(args.output_pdf, bbox_inches="tight")
        print(f"Saved PDF: {args.output_pdf}")

    # Print summary table
    print()
    print("=== Summary Table ===")
    print(f"{'Scale':>8}  {'Validity%':>10}  {'Diversity':>10}  {'VLM':>6}  {'Pareto':>6}")
    print("-" * 50)
    for s, mv, sd, md, vlm, is_opt in zip(
        scales, mean_validity, std_diversity, mean_diversity, mean_vlm, pareto_mask
    ):
        vlm_str = f"{vlm:.3f}" if vlm is not None else "  N/A"
        opt_str = "*" if is_opt else ""
        print(f"  {s:>6.1f}  {mv*100:>9.1f}%  {md:>10.4f}  {vlm_str:>6}  {opt_str:>6}")


if __name__ == "__main__":
    main()
