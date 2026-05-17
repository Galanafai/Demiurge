"""Build a warm-init checkpoint for conditional_v8c by padding v7 weights.

v7 architecture:  d_model=256, n_layers=6,  n_heads=8
v8c architecture: d_model=384, n_layers=16, n_heads=6

Padding strategy (per-parameter):
  - Shared scalar/1D tensors (LayerNorm weight/bias, biases): pad with zeros
    or copy directly if shape matches.
  - 2D weight matrices (Linear.weight): pad new rows/cols with zero so the
    original d_model=256 subspace is exactly preserved.
    The output of any padded linear layer in the first forward pass is
    identical to the v7 output for the first 256 components, and zero for
    the new 128 components -- a safe, smooth continuation.
  - Embedding tables (type_embed, slot_id not present in v7): pad new rows/cols.
  - New layers (6-15): initialized by SceneDenoiser._init_weights() (AdaLN
    zero-init, presence head bias=-2.0). These are then overwritten from
    the padded checkpoint only when shapes match -- so new layers keep
    their default init, which is safe.

Output: checkpoints/conditional_v8c_warminit/init.pt
  Contains model_state (padded), ema_state (same), step=0, arch fingerprint.
  The training script warm_init_from= loads this and copies matching keys.

Usage:
    python3 -u scripts/build_warminit_v8c.py [--v7-ckpt PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from model.denoiser import DenoiserConfig, SceneDenoiser  # noqa: E402


# ---------------------------------------------------------------------------
# Padding helpers
# ---------------------------------------------------------------------------


def _pad_tensor(src: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Pad src to target_shape with zeros, preserving existing values.

    For each dimension, the src values are placed in the leading slice
    (src.shape[i] elements); the remaining (target_shape[i] - src.shape[i])
    elements are zero.

    Args:
        src: Source tensor from v7. Any dtype.
        target_shape: Desired output shape. Must have same ndim as src and
            target_shape[i] >= src.shape[i] for all i.

    Returns:
        New tensor with target_shape, zero-padded, same dtype as src.
    """
    if src.shape == target_shape:
        return src.clone()
    out = torch.zeros(target_shape, dtype=src.dtype)
    slices = tuple(slice(0, s) for s in src.shape)
    out[slices] = src
    return out


def _can_pad(src_shape: torch.Size, tgt_shape: torch.Size) -> bool:
    """Return True if src can be padded to tgt (same ndim, each dim non-decreasing)."""
    if len(src_shape) != len(tgt_shape):
        return False
    return all(s <= t for s, t in zip(src_shape, tgt_shape))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_warminit(v7_ckpt_path: Path, out_path: Path) -> None:
    print(f"Loading v7 checkpoint from {v7_ckpt_path} ...")
    v7_ckpt = torch.load(v7_ckpt_path, map_location="cpu", weights_only=False)

    # Prefer EMA weights: they are the smoothed, better-generalising version.
    if "ema_state" in v7_ckpt:
        v7_state: dict[str, torch.Tensor] = {
            k: v.float() for k, v in v7_ckpt["ema_state"].items()
        }
        print("Using EMA weights from v7 checkpoint.")
    elif "model_state" in v7_ckpt:
        v7_state = {k: v.float() for k, v in v7_ckpt["model_state"].items()}
        print("Using model_state from v7 checkpoint (no EMA found).")
    else:
        raise KeyError(
            f"v7 checkpoint at {v7_ckpt_path} has neither 'ema_state' nor "
            "'model_state'. Keys: " + str(list(v7_ckpt.keys()))
        )

    v7_params = sum(p.numel() for p in v7_state.values())
    print(f"v7 parameter count: {v7_params / 1e6:.2f}M")

    # Build v8c model to get target state dict.
    v8c_cfg = DenoiserConfig(
        n_layers=16, d_model=384, n_heads=6, ffn_mult=4,
        dropout=0.1, use_type_grad_isolation=True, use_slot_id_embed=False,
    )
    v8c_model = SceneDenoiser(v8c_cfg)
    v8c_state: dict[str, torch.Tensor] = {
        k: v.clone().float() for k, v in v8c_model.state_dict().items()
    }
    v8c_params = sum(p.numel() for p in v8c_state.values())
    print(f"v8c parameter count: {v8c_params / 1e6:.2f}M")

    # Apply padding from v7 -> v8c.
    matched_exact = 0
    matched_padded = 0
    skipped_new = 0
    skipped_shrink = 0

    for key, v8c_tensor in v8c_state.items():
        if key not in v7_state:
            # New key (e.g., new layers 6-15, slot_id_embed if added): keep v8c default.
            skipped_new += 1
            continue

        v7_tensor = v7_state[key]

        if v7_tensor.shape == v8c_tensor.shape:
            # Exact shape match: direct copy.
            v8c_state[key] = v7_tensor.clone()
            matched_exact += 1

        elif _can_pad(v7_tensor.shape, v8c_tensor.shape):
            # Paddable: v7 fits inside v8c shape.
            v8c_state[key] = _pad_tensor(v7_tensor, v8c_tensor.shape)
            matched_padded += 1

        else:
            # Shape incompatible (e.g., v7 has larger shape in some dim -- shouldn't happen
            # when going to a strictly larger model, but guard defensively).
            skipped_shrink += 1
            print(
                f"  SKIP (shrink not allowed): {key} "
                f"v7={tuple(v7_tensor.shape)} -> v8c={tuple(v8c_tensor.shape)}"
            )

    total_keys = len(v8c_state)
    print(
        f"\nKey mapping summary ({total_keys} total v8c keys):\n"
        f"  exact copy:   {matched_exact:4d}\n"
        f"  zero-padded:  {matched_padded:4d}\n"
        f"  new (default): {skipped_new:4d}\n"
        f"  skipped (shrink): {skipped_shrink:4d}"
    )

    # Sanity: count how many v8c parameters came from v7.
    v7_sourced = sum(
        v8c_state[k].numel()
        for k in v8c_state
        if k in v7_state
        and (v7_state[k].shape == v8c_state[k].shape or _can_pad(v7_state[k].shape, v8c_state[k].shape))
    )
    print(
        f"\nParameters sourced from v7: {v7_sourced / 1e6:.2f}M "
        f"/ {v8c_params / 1e6:.2f}M total "
        f"({100 * v7_sourced / max(v8c_params, 1):.1f}%)"
    )

    # Load padded state into model and verify it doesn't crash.
    v8c_model.load_state_dict({k: v for k, v in v8c_state.items()}, strict=True)
    print("load_state_dict strict=True: OK")

    # Save the padded checkpoint.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arch = {"n_layers": 16, "d_model": 384, "n_heads": 6, "ffn_mult": 4}
    ckpt = {
        "step": 0,
        "model_state": {k: v.float() for k, v in v8c_model.state_dict().items()},
        "ema_state": {k: v.float() for k, v in v8c_model.state_dict().items()},
        "arch": arch,
        "wandb_run_id": None,
        "warminit_source": str(v7_ckpt_path),
        "warminit_v7_step": v7_ckpt.get("step", "unknown"),
    }
    torch.save(ckpt, out_path)
    print(f"\nWarm-init checkpoint saved to {out_path}")
    print(f"  Size on disk: {out_path.stat().st_size / 1e6:.1f} MB")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--v7-ckpt",
        default="checkpoints/conditional_v7/latest.pt",
        help="Path to v7 checkpoint (default: checkpoints/conditional_v7/latest.pt)",
    )
    p.add_argument(
        "--out",
        default="checkpoints/conditional_v8c_warminit/init.pt",
        help="Output path for padded checkpoint (default: checkpoints/conditional_v8c_warminit/init.pt)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_warminit(Path(args.v7_ckpt), Path(args.out))
