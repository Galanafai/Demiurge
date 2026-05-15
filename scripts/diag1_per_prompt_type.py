"""Diagnostic 1: Per-prompt type distribution at cfg=1.0.

Tests whether type 5 dominates regardless of prompt (signal-blind collapse)
or if type varies with prompt content (text-type signal present).

Each row: target object type -> actual type distribution generated when
prompted specifically for that object.

Pass criteria: top generated type matches or is near target type for
at least 6/13 prompts. If all rows produce t5, the cross-attention
type signal is broken.
"""
from __future__ import annotations

import sys
from collections import Counter

import torch

sys.path.insert(0, "src")
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser
from model.schedule import CosineSchedule, DDIMSampler
from model.text_encoder import TextEncoder

PROMPTS: dict[int, tuple[str, list[str]]] = {
    0:  ("cube",             ["place a cube", "set a cube on the table", "put a cube here"]),
    1:  ("sphere",           ["place a sphere", "set a sphere here", "put a sphere"]),
    2:  ("cylinder",         ["place a cylinder", "set a cylinder here"]),
    3:  ("cracker_box",      ["place a cracker box", "set down a cracker box"]),
    4:  ("sugar_box",        ["place a sugar box", "set down a sugar box"]),
    5:  ("tomato_soup_can",  ["place a tomato soup can", "set a tomato soup can"]),
    6:  ("mustard_bottle",   ["place a mustard bottle", "set a mustard bottle"]),
    7:  ("tuna_can",         ["place a tuna can", "set a tuna can here"]),
    8:  ("gelatin_box",      ["place a gelatin box", "set a gelatin box"]),
    9:  ("banana",           ["place a banana", "put a banana here"]),
    10: ("master_chef_can",  ["place a master chef can", "set a master chef can"]),
    11: ("bleach_cleanser",  ["place a bleach cleanser", "set a bleach cleanser"]),
    12: ("mug",              ["place a mug", "set a mug on the table", "put a mug"]),
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ckpt = torch.load(
    "checkpoints/conditional_v7/latest.pt",
    map_location="cpu", weights_only=False,
)
arch = ckpt["arch"]
cfg = DenoiserConfig(
    n_layers=arch["n_layers"], d_model=arch["d_model"],
    n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"],
    dropout=0.0, use_type_grad_isolation=True,
)
model = SceneDenoiser(cfg).to(device)
model.load_state_dict(ckpt["ema_state"])
model.eval()
print(f"Loaded v7 EMA at step {ckpt['step']}")

encoder = TextEncoder(device=device, cache_path="data/v1/text_embeddings.pt")
schedule = CosineSchedule(T=1000)
sampler = DDIMSampler(schedule, n_steps=50)

n_type = 13
header = f"{'Target':>16}  " + " ".join(f"t{t:2d}" for t in range(n_type)) + "  Top"
print("\nPer-prompt type distribution (8 scenes x n_prompts per target):")
print(header)
print("-" * len(header) + "-" * 20)

for target_tid, (name, ps) in PROMPTS.items():
    embs = encoder.encode_batch(ps).to(device)   # (n_prompts, 384)
    type_counts: Counter[int] = Counter()
    total = 0
    for i, _ in enumerate(ps):
        fn = model.conditional_sampling_fn(embs[i : i + 1])
        with torch.no_grad():
            x, tids = sampler.sample_with_types(
                fn, (8, N_MAX, 13),
                seed=42 + i, device=device, type_init="uniform",
            )
        pres = x[:, :, 12] > 0.0
        for b in range(8):
            for n in range(N_MAX):
                if pres[b, n]:
                    type_counts[int(tids[b, n].item())] += 1
                    total += 1

    if total == 0:
        print(f"{name:>16}: NO OBJECTS GENERATED")
        continue

    dist = "  ".join(f"{100 * type_counts[t] / total:4.0f}" for t in range(n_type))
    top_tid, top_cnt = type_counts.most_common(1)[0]
    top_pct = 100 * top_cnt / total
    match = "MATCH" if top_tid == target_tid else "  -- "
    print(f"{name:>16}: {dist}  -> t{top_tid}={top_pct:.0f}% {match}")

print("\nDiagnostic 1 complete.")
