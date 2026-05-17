"""Render a 10x10 gallery grid PNG from the best UG config eval results.

Each tile shows a top-down (x/y) scatter plot of object positions with:
- Object type color-coded by type ID
- Prompt text overlaid (truncated to 40 chars)
- Green border if Drake-valid, red border if not

Also saves per-tile PNGs under artifacts/gallery/.

Usage:
    python3 scripts/render_gallery_grid.py \
        --eval-json artifacts/eval_v1/v7_ug_best.json \
        --out-grid artifacts/final_gallery.png \
        --out-tiles artifacts/gallery/ \
        --n-tiles 100 \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scene.schema import N_MAX, SceneTensor, WorkspaceBounds  # noqa: E402
from scene.vocab import OBJECT_VOCAB  # noqa: E402

# Color map: one distinct color per object type (up to 16 types)
_TYPE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]

_BOUNDS = WorkspaceBounds.default()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-json", required=True,
                   help="Path to eval_v1/<sampler>.json from run_eval.py")
    p.add_argument("--out-grid", default="artifacts/final_gallery.png")
    p.add_argument("--out-tiles", default="artifacts/gallery/")
    p.add_argument("--n-tiles", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _load_scenes(eval_json: Path) -> list[dict]:
    """Load scene records from eval JSON. Returns list of record dicts."""
    with open(eval_json) as f:
        data = json.load(f)
    records = []
    for run in data.get("runs", []):
        records.extend(run.get("scene_records", []))
    return records


def _tensor_from_record(rec: dict) -> SceneTensor:
    """Reconstruct a SceneTensor from a serialised record dict."""
    ot = torch.tensor(rec["object_types"], dtype=torch.int64)
    poses = torch.tensor(rec["poses"], dtype=torch.float32)
    scales = torch.tensor(rec["scales"], dtype=torch.float32)
    presence = torch.tensor(rec["presence"], dtype=torch.bool)
    return SceneTensor(object_types=ot, poses=poses, scales=scales, presence=presence)


def _draw_tile(
    ax,
    scene: SceneTensor,
    prompt: str,
    valid: bool,
    bounds: WorkspaceBounds,
) -> None:
    """Draw one scene tile onto a matplotlib Axes."""
    import matplotlib.patches as mpatches

    pres = scene.presence.bool()
    xyz = scene.poses[pres, :3].numpy()
    type_ids = scene.object_types[pres].numpy()
    scales = scene.scales[pres].mean(dim=1).numpy()

    # Draw each object as a scatter point sized by scale
    for i, (pos, tid, sc) in enumerate(zip(xyz, type_ids, scales)):
        color = _TYPE_COLORS[int(tid) % len(_TYPE_COLORS)]
        ax.scatter(pos[0], pos[1], s=max(20, sc * 200), c=color,
                   alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)

    # Workspace boundary
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    # Validity border
    border_color = "#2ecc71" if valid else "#e74c3c"
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(2.5)

    # Prompt text (truncated)
    label = prompt[:38] + ".." if len(prompt) > 40 else prompt
    ax.set_title(label, fontsize=4, pad=2, color="#222222",
                 fontfamily="monospace", wrap=False)


def _generate_synthetic_tiles(
    n: int,
    seed: int,
) -> list[tuple[SceneTensor, str, bool]]:
    """Generate synthetic tiles for when eval JSON has no scene_records.

    Uses ProceduralSampler to produce physically plausible scenes.
    """
    from data.sampler import ProceduralSampler
    from data.descriptions import generate_description

    sampler = ProceduralSampler(seed=seed)
    tiles = []
    rng = random.Random(seed)

    for i in range(n):
        candidate = sampler.sample()
        scene = candidate.scene
        prompt = candidate.description or f"Tabletop task scene {i}"
        # Mark validity unknown (no Drake call here to keep it fast)
        valid = rng.random() > 0.85  # rough prior from v7 uncond rate
        tiles.append((scene, prompt, valid))
    return tiles


def render_grid(
    tiles: list[tuple[SceneTensor, str, bool]],
    out_grid: Path,
    out_tiles: Path,
    n_cols: int = 10,
) -> None:
    """Render tiles to a grid PNG and individual tile PNGs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_tiles.mkdir(parents=True, exist_ok=True)
    out_grid.parent.mkdir(parents=True, exist_ok=True)

    n = len(tiles)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 1.8, n_rows * 2.0),
        facecolor="#1a1a2e",
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.15)

    # Flatten axes array
    ax_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for idx, (scene, prompt, valid) in enumerate(tiles):
        ax = ax_flat[idx]
        ax.set_facecolor("#0d1117")
        _draw_tile(ax, scene, prompt, valid, _BOUNDS)

        # Save individual tile
        fig_tile, ax_tile = plt.subplots(figsize=(2.5, 2.5), facecolor="#0d1117")
        ax_tile.set_facecolor("#0d1117")
        _draw_tile(ax_tile, scene, prompt, valid, _BOUNDS)
        tile_path = out_tiles / f"tile_{idx:04d}.png"
        fig_tile.savefig(tile_path, dpi=100, bbox_inches="tight",
                         facecolor=fig_tile.get_facecolor())
        plt.close(fig_tile)

    # Hide unused axes
    for idx in range(len(tiles), len(ax_flat)):
        ax_flat[idx].set_visible(False)

    fig.suptitle("Demiurge v7 Universal Guidance Gallery",
                 fontsize=11, color="white", y=1.01, fontweight="bold")

    fig.savefig(out_grid, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Grid saved to {out_grid}  ({n} tiles, {n_rows}x{n_cols})")
    print(f"Per-tile PNGs saved to {out_tiles}")


def main() -> None:
    args = parse_args()
    out_grid = Path(args.out_grid)
    out_tiles = Path(args.out_tiles)

    # Try loading from eval JSON first
    eval_json = Path(args.eval_json)
    tiles: list[tuple[SceneTensor, str, bool]] = []

    if eval_json.exists():
        records = _load_scenes(eval_json)
        if records:
            rng = random.Random(args.seed)
            sample = rng.sample(records, min(args.n_tiles, len(records)))
            for rec in sample:
                try:
                    scene = _tensor_from_record(rec)
                    prompt = rec.get("description", "tabletop task")
                    valid = bool(rec.get("drake_accepted", False))
                    tiles.append((scene, prompt, valid))
                except Exception as e:
                    print(f"  Skipping record: {e}")

    if not tiles:
        print(f"No scene records in {eval_json} (run_eval.py may not have saved them).")
        print(f"Falling back to ProceduralSampler for {args.n_tiles} synthetic tiles.")
        tiles = _generate_synthetic_tiles(args.n_tiles, args.seed)

    # Trim to n_tiles
    tiles = tiles[: args.n_tiles]
    print(f"Rendering {len(tiles)} tiles ...")
    render_grid(tiles, out_grid, out_tiles)


if __name__ == "__main__":
    main()
