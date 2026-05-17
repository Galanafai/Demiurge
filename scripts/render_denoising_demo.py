"""Render a 30-second denoising demo MP4.

Runs DDIM sampling on v7 for one fixed prompt, capturing the scene state
x_cont at each of 50 steps. Renders each step as a top-down matplotlib
scatter plot showing objects drifting from noise into a coherent scene.
Encodes frames via ffmpeg to artifacts/denoising_demo.mp4 at 24fps.

Falls back gracefully if ffmpeg is unavailable: saves frame PNGs to
artifacts/denoising_frames/ instead.

Usage:
    python3 scripts/render_denoising_demo.py \
        --checkpoint checkpoints/conditional_v7/latest.pt \
        --prompt "Pick the mustard bottle avoiding the red cube" \
        --cfg-scale 1.0 \
        --ug-scale 0.5 \
        --seed 42 \
        --out artifacts/denoising_demo.mp4 \
        --fps 24
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser  # noqa: E402
from model.rotations import rot6d_to_quat_wxyz  # noqa: E402
from model.schedule import CosineSchedule, DDIMSampler  # noqa: E402
from model.text_encoder import TextEncoder  # noqa: E402
from scene.schema import SceneTensor, WorkspaceBounds  # noqa: E402

DDIM_STEPS = 50

_TYPE_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
    "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3",
]

DEMO_PROMPT = "Pick the mustard bottle avoiding the red cube"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default="checkpoints/conditional_v7/latest.pt")
    p.add_argument("--prompt", default=DEMO_PROMPT)
    p.add_argument("--cfg-scale", type=float, default=1.0)
    p.add_argument("--ug-scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="artifacts/denoising_demo.mp4")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--frames-dir", default="artifacts/denoising_frames/")
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> SceneDenoiser:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    cfg = DenoiserConfig(
        n_layers=arch["n_layers"], d_model=arch["d_model"],
        n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"], dropout=0.0,
        use_type_grad_isolation=arch.get("use_type_grad_isolation", False),
    )
    model = SceneDenoiser(cfg).to(device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    print(f"Loaded EMA from {ckpt_path.name} (step={ckpt['step']})")
    return model


def encode_prompt(prompt: str, device: torch.device) -> torch.Tensor:
    """Encode a single prompt string to a (1, D) embedding tensor."""
    encoder = TextEncoder()
    emb = encoder.encode(prompt)  # returns (D,) Tensor
    return emb.unsqueeze(0).to(device)  # (1, D)


def sample_with_capture(
    model: SceneDenoiser,
    text_emb: torch.Tensor,
    seed: int,
    device: torch.device,
    cfg_scale: float,
    ug_scale: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Run DDIM and capture x_0 estimate at every step via manual loop.

    Returns:
        frames_x: list of (N_MAX, 13) tensors, one per DDIM step
        frames_type: list of (N_MAX,) long tensors
        final_types: (N_MAX,) type tensor (final step)
    """
    schedule = CosineSchedule(T=1000)

    frames_x, frames_type = _capture_by_manual_loop(
        model, text_emb, seed, device, cfg_scale, ug_scale, schedule
    )
    final_types = frames_type[-1] if frames_type else torch.zeros(N_MAX, dtype=torch.long)
    return frames_x, frames_type, final_types


def _capture_by_manual_loop(
    model: SceneDenoiser,
    text_emb: torch.Tensor,
    seed: int,
    device: torch.device,
    cfg_scale: float,
    ug_scale: float,
    schedule: CosineSchedule,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Manual DDIM step loop that captures x_0 estimate at each step.

    Mirrors DDIMSampler.sample_with_universal_guidance internals exactly,
    adding frame capture. alpha_bar() takes a Tensor as required by the API.
    """
    from guidance.energy import pairwise_overlap_energy

    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    # DDIM timesteps: evenly spaced descending, same as DDIMSampler internals
    T = 1000
    step_indices = torch.linspace(T - 1, 0, DDIM_STEPS).long()  # descending

    x_t = torch.randn(1, N_MAX, 13, device=device, generator=generator)
    type_ids = torch.randint(0, 12, (1, N_MAX), device=device, generator=generator)

    if cfg_scale == 1.0:
        fn = model.conditional_sampling_fn(text_emb)
    else:
        fn_u = model.conditional_sampling_fn(text_emb=None)
        fn_c = model.conditional_sampling_fn(text_emb)
        _s = cfg_scale

        def fn(x_t_: torch.Tensor, tids: torch.Tensor, t: torch.Tensor):
            eu, lu = fn_u(x_t_, tids, t)
            ec, lc = fn_c(x_t_, tids, t)
            return eu + _s * (ec - eu), lc

    frames_x: list[torch.Tensor] = []
    frames_type: list[torch.Tensor] = []

    guidance_min = int(0.1 * DDIM_STEPS)
    guidance_max = int(0.9 * DDIM_STEPS)

    with torch.no_grad():
        for step_idx in range(DDIM_STEPS):
            t_val = step_indices[step_idx]
            t_batch = t_val.expand(1).to(device)  # (1,) tensor required by fn

            # Forward pass (no grad needed for eps and type logits)
            eps_pred, type_logits = fn(x_t, type_ids, t_batch)

            # Tweedie estimate of x_0 using schedule.alpha_bar(Tensor)
            ab_t = schedule.alpha_bar(t_batch)  # (1,) or scalar Tensor
            x_0_hat = (x_t - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()

            # Universal Guidance gradient (active only in middle steps)
            if ug_scale > 0.0 and guidance_min <= step_idx <= guidance_max:
                x_0_hat_g = x_0_hat.detach().requires_grad_(True)
                energy = pairwise_overlap_energy(x_0_hat_g, type_ids).sum()
                grad = torch.autograd.grad(energy, x_0_hat_g)[0]
                # Correct eps: eps_guided = eps + ug_scale * sqrt(1-ab) * grad
                eps_pred = eps_pred + ug_scale * (1.0 - ab_t).sqrt() * grad.detach()
                # Recompute x_0_hat with guided eps
                x_0_hat = (x_t - (1.0 - ab_t).sqrt() * eps_pred) / ab_t.sqrt()

            # Update discrete types via argmax
            type_ids = type_logits.argmax(dim=-1)

            # Capture this frame (detached, on CPU to save VRAM)
            frames_x.append(x_0_hat[0].detach().cpu())
            frames_type.append(type_ids[0].detach().cpu())

            # DDIM step: x_{t-1} = sqrt(ab_{t-1}) * x_0_hat + sqrt(1-ab_{t-1}) * eps
            if step_idx < DDIM_STEPS - 1:
                t_prev = step_indices[step_idx + 1]
                t_prev_batch = t_prev.expand(1).to(device)
                ab_prev = schedule.alpha_bar(t_prev_batch)
                x_t = ab_prev.sqrt() * x_0_hat + (1.0 - ab_prev).sqrt() * eps_pred
            # else: final step, x_t not needed

    return frames_x, frames_type



def render_frame(
    ax,
    x_cont: torch.Tensor,
    type_ids: torch.Tensor,
    step_idx: int,
    total_steps: int,
    prompt: str,
    bounds: WorkspaceBounds,
) -> None:
    """Draw one denoising step frame onto a matplotlib Axes."""
    ax.clear()
    ax.set_facecolor("#0d1117")

    pres_mask = x_cont[:, 12] > 0.0
    xyz = x_cont[:, :3]  # (N_MAX, 3) - normalized space

    for i in range(N_MAX):
        tid = int(type_ids[i].item())
        color = _TYPE_COLORS[tid % len(_TYPE_COLORS)]
        alpha = 0.9 if pres_mask[i] else 0.2
        size = 120 if pres_mask[i] else 30
        ax.scatter(
            xyz[i, 0].item(), xyz[i, 1].item(),
            s=size, c=color, alpha=alpha, zorder=3,
            edgecolors="white" if pres_mask[i] else "none",
            linewidths=0.5,
        )

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_edgecolor("#2c3e50")
        spine.set_linewidth(1.5)

    progress = step_idx / max(total_steps - 1, 1)
    ax.set_title(
        f"Step {step_idx + 1}/{total_steps}  |  t={1.0 - progress:.2f}",
        fontsize=9, color="#ecf0f1", pad=4,
    )

    # Prompt watermark at bottom
    label = prompt[:55] + ".." if len(prompt) > 57 else prompt
    ax.text(
        0.5, -0.08, label, transform=ax.transAxes,
        ha="center", va="top", fontsize=6, color="#7f8c8d",
        fontfamily="monospace",
    )


def save_frames(
    frames_x: list[torch.Tensor],
    frames_type: list[torch.Tensor],
    prompt: str,
    frames_dir: Path,
) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames_dir.mkdir(parents=True, exist_ok=True)
    bounds = WorkspaceBounds.default()
    paths: list[Path] = []

    fig, ax = plt.subplots(1, 1, figsize=(5, 5), facecolor="#0d1117")

    print(f"Rendering {len(frames_x)} frames ...")
    for i, (x, t) in enumerate(zip(frames_x, frames_type)):
        render_frame(ax, x, t, i, len(frames_x), prompt, bounds)
        path = frames_dir / f"frame_{i:04d}.png"
        fig.savefig(path, dpi=100, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        paths.append(path)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(frames_x)} frames rendered")

    plt.close(fig)
    return paths


def encode_mp4(frames_dir: Path, out_path: Path, fps: int) -> bool:
    """Encode frames to MP4 via ffmpeg. Returns True on success."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found - skipping MP4 encoding. Frames saved as PNGs.")
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pattern = str(frames_dir / "frame_%04d.png")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        str(out_path),
    ]
    print(f"Encoding MP4: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr[-500:]}")
        return False
    print(f"MP4 saved to {out_path}")
    return True


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Prompt: {args.prompt}")

    model = load_model(Path(args.checkpoint), device)
    text_emb = encode_prompt(args.prompt, device)

    print(f"Capturing {DDIM_STEPS} DDIM denoising frames ...")
    frames_x, frames_type, _ = sample_with_capture(
        model, text_emb, args.seed, device,
        args.cfg_scale, args.ug_scale,
    )

    print(f"Captured {len(frames_x)} frames")
    frames_dir = Path(args.frames_dir)
    save_frames(frames_x, frames_type, args.prompt, frames_dir)

    out_path = Path(args.out)
    ok = encode_mp4(frames_dir, out_path, args.fps)
    if not ok:
        print(f"Frames are at {frames_dir} - encode manually with:")
        print(f"  ffmpeg -framerate {args.fps} -i {frames_dir}/frame_%04d.png "
              f"-c:v libx264 -pix_fmt yuv420p {out_path}")


if __name__ == "__main__":
    main()
