"""Compute type_id class distribution across the v1 dataset.

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 scripts/compute_class_distribution.py \
        --data-dir data/v1 --eps 0.01

Output:
    Frequency table to stdout + YAML-ready class_weights list.
    Optionally writes artifacts/class_distribution.md when --out is given.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data.reader import ShardReader  # noqa: E402
from model.denoiser import N_TYPE  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data/v1", help="Path to shard directory")
    p.add_argument("--eps", type=float, default=0.01,
                   help="Epsilon added to frequencies before inversion (prevents div by zero)")
    p.add_argument("--out", default=None,
                   help="Optional output path for markdown report (e.g. artifacts/class_distribution.md)")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    print(f"Scanning {data_dir} ...", flush=True)
    counts: Counter[int] = Counter()
    n_scenes = 0

    reader = ShardReader(data_dir, shuffle_buffer=0, shardshuffle=False)
    for scene, _desc, _report, _sdf in reader:
        n_scenes += 1
        for i in range(len(scene.object_types)):
            if scene.presence[i]:
                counts[int(scene.object_types[i])] += 1
        if n_scenes % 5000 == 0:
            print(f"  ... {n_scenes} scenes scanned", flush=True)

    total = sum(counts.values())
    print(f"\nScanned {n_scenes} scenes, {total} present-slot type observations.\n")

    # --- Frequency table ---
    print(f"{'type_id':>8}  {'count':>8}  {'freq%':>7}  {'inv_freq_raw':>14}  {'inv_freq_eps':>14}")
    print("-" * 62)

    # Ensure all N_TYPE classes appear (even if count=0).
    all_type_ids = list(range(N_TYPE))
    freqs: dict[int, float] = {}
    for tid in all_type_ids:
        c = counts.get(tid, 0)
        freq = c / max(total, 1)
        freqs[tid] = freq
        inv_raw = 1.0 / freq if freq > 0 else float("inf")
        inv_eps = 1.0 / (freq + args.eps)
        marker = "  <-- PAD" if tid == N_TYPE - 1 else ""
        print(f"{tid:>8}  {c:>8}  {100*freq:>6.2f}%  {inv_raw:>14.4f}  {inv_eps:>14.4f}{marker}")

    # --- Compute class_weights (inverse-frequency with epsilon) ---
    # Normalise so weights sum to N_TYPE (makes the CE loss scale similar to
    # uniform weighting but rebalanced across classes).
    raw_weights = [1.0 / (freqs[tid] + args.eps) for tid in all_type_ids]
    mean_w = sum(raw_weights) / len(raw_weights)
    norm_weights = [w / mean_w for w in raw_weights]

    print(f"\nInverse-frequency class_weights (eps={args.eps}, normalised to mean=1.0):")
    weights_str = ", ".join(f"{w:.6f}" for w in norm_weights)
    print(f"  [{weights_str}]")
    print()
    print("YAML snippet for configs/train/conditional_v4.yaml:")
    print("  training:")
    print(f"    class_weights: [{weights_str}]")

    # --- Optional markdown output ---
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Class Distribution: v1 Dataset",
            "",
            f"Scanned **{n_scenes}** scenes, **{total}** present-slot type observations.",
            f"Epsilon: {args.eps}.",
            "",
            "## Frequency Table",
            "",
            "| type_id | count | freq% | inv_freq (eps) | weight (norm) |",
            "|---|---|---|---|---|",
        ]
        for tid in all_type_ids:
            c = counts.get(tid, 0)
            freq = freqs[tid]
            inv_eps = 1.0 / (freq + args.eps)
            w_norm = norm_weights[tid]
            tag = " (PAD)" if tid == N_TYPE - 1 else ""
            lines.append(f"| {tid}{tag} | {c} | {100*freq:.2f}% | {inv_eps:.4f} | {w_norm:.6f} |")

        lines += [
            "",
            "## class_weights (YAML)",
            "",
            "```yaml",
            "training:",
            f"  class_weights: [{weights_str}]",
            "```",
            "",
            "Weights normalised so mean = 1.0. Cube (type_id=0) receives the lowest weight "
            "(most frequent). Rare types receive weights >> 1.0.",
        ]
        out_path.write_text("\n".join(lines) + "\n")
        print(f"\nReport written to {out_path}")


if __name__ == "__main__":
    main()
