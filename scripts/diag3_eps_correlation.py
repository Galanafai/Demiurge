"""Diagnostic 3: Conditional vs unconditional eps correlation.

Measures the Pearson correlation between eps_cond and eps_uncond at
a fixed noise level (t=500) for several prompts.

v6 baseline: ~0.00 (completely uncorrelated -- CFG interpolation destroyed geometry)
v7 target:   >0.50 (correlated -- CFG interpolation is stable)

Also reports:
- Norm of eps_cond vs eps_uncond (should be similar magnitudes)
- Norm of (eps_cond - eps_uncond): the "guidance vector" magnitude

High correlation confirms the CFG training objective was met.
Low correlation confirms v6's uncorrelated-paths problem persists.
"""
from __future__ import annotations

import sys

import torch

sys.path.insert(0, "src")
from model.denoiser import N_MAX, DenoiserConfig, SceneDenoiser
from model.text_encoder import TextEncoder

TEST_PROMPTS = [
    "place a cube on the table",
    "retrieve the mustard bottle from the workspace",
    "arrange a tomato soup can and a sugar box",
    "pick the banana from the cluttered area",
    "set down a gelatin box at 30cm forward",
]

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
print(f"Loaded v7 EMA at step {ckpt['step']}\n")

encoder = TextEncoder(device=device, cache_path="data/v1/text_embeddings.pt")

# Fixed noise state (same for all prompts for comparability)
torch.manual_seed(42)
x_t = torch.randn(4, N_MAX, 13, device=device)
type_ids = torch.randint(0, 13, (4, N_MAX), device=device)
t_vec = torch.full((4,), 500, device=device)

header = f"  {'Prompt':>40}  {'corr':>7}  {'|eps_c|':>8}  {'|eps_u|':>8}  {'|diff|':>8}"
print(header)
print("  " + "-" * (len(header) - 2))

correlations: list[float] = []
for prompt in TEST_PROMPTS:
    text_emb = encoder.encode_batch([prompt]).to(device)  # (1, 384)
    with torch.no_grad():
        fn_c = model.conditional_sampling_fn(text_emb)
        fn_u = model.conditional_sampling_fn(None)
        eps_c, _ = fn_c(x_t, type_ids, t_vec)
        eps_u, _ = fn_u(x_t, type_ids, t_vec)

    corr = torch.corrcoef(
        torch.stack([eps_c.flatten().float(), eps_u.flatten().float()])
    )[0, 1].item()
    correlations.append(corr)
    norm_c = eps_c.norm().item()
    norm_u = eps_u.norm().item()
    norm_diff = (eps_c - eps_u).norm().item()
    print(f"  {prompt[:40]:>40}  {corr:>7.4f}  {norm_c:>8.2f}  {norm_u:>8.2f}  {norm_diff:>8.2f}")

mean_corr = sum(correlations) / len(correlations)
print(f"\n  Mean correlation across {len(correlations)} prompts: {mean_corr:.4f}")
print()
print("  Interpretation:")
if mean_corr > 0.7:
    print("  STRONG CORRELATION (>0.7): CFG training worked. Both paths agree on geometry.")
    print("  The unconditional path collapse in Drake is a SAMPLING issue, not a model issue.")
elif mean_corr > 0.4:
    print("  MODERATE CORRELATION (0.4-0.7): CFG partially fixed. More training needed.")
    print("  The uncond path has learned some geometry but diverges from cond at high noise.")
elif mean_corr > 0.1:
    print("  WEAK CORRELATION (0.1-0.4): CFG barely worked. Paths largely independent.")
else:
    print("  NO CORRELATION (<0.1): V6 problem persists. Paths uncorrelated.")
    print("  The uncond path has NOT learned to track the cond path.")
print()
print("  Reference: v6 was ~0.00. v7 minimum target > 0.50 for stable CFG sweep.")
print("\nDiagnostic 3 complete.")
