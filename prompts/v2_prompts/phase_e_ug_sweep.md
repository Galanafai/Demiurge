# Antigravity Master Prompt — Phase E: Universal Guidance Sweep on v9

You are continuing Phase E of the v9 build. Phase D trained v9 conditional
(49M, warm-init from v9 uncond, CFG dropout 0.15). Phase D probes confirmed
the model exceeds gate thresholds.

Phase E: Apply Universal Guidance to v9. Reuse Week 5 UG infrastructure.
Sweep guidance scales, produce Pareto plot v9.

# Pre-Flight

## Step 0.1: Branch + verify Phase D

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

cd /workspace/Demiurge

git checkout main
git checkout -b week6/v9-phase-e

# Verify Phase D outputs
ls -lh checkpoints/conditional_v9_cond/latest.pt
cat artifacts/v9_cond_phase_d_results.md | head -20

# Confirm v9 cond on W&B
python3 -c "
import wandb
api = wandb.Api()
art = api.artifact('galanafai-self/demiurge/conditional_v9_cond_checkpoint:latest')
print(f'v9 cond on W&B: {art.size / 1024 / 1024:.1f} MB')
"

# Verify UG infrastructure
ls -lh src/guidance/energy.py
ls -lh src/guidance/object_geometry.py
ls -lh scripts/run_universal_guidance_sweep.py

# Verify tests pass
pytest tests/guidance/ tests/model/test_universal_guidance.py -q --tb=no 2>&1 | tail -5

# Before Doing Anything Else

1. Read AGENTS.md, SKILL.md.
2. Read artifacts/v9_cond_phase_d_results.md.
3. Read src/guidance/energy.py (UG implementation).
4. Read artifacts/ug_sweep_v2/summary.jsonl (v7 UG results from Week 5).
5. State in Plan Artifact:
   - Best v9 CFG from Phase D
   - UG sweep design: 6 scales × 3 seeds × 200 prompts × 5 scenes = 18,000 scenes
   - Expected cost: $3 GPU + $2 VLM
   - Phase E success: Pareto plot produced

# Phase E Execution

## Step E.1: UG smoke test

Verify UG works on v9 (same SceneTensor format, should work without modification):

python3 -u scripts/run_universal_guidance_sweep.py \
  --model-checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --cfg-scale 1.0 \
  --guidance-scales 2.0 \
  --seeds 42 \
  --n-prompts 50 \
  --n-per-prompt 4 \
  --drake-workers 24 \
  --output-dir artifacts/v9_ug_smoke/ \
  --budget-cap 0.50 \
  2>&1 | tee logs/v9_ug_smoke.log

Expected at scale=2.0:
- If similar to v7: ~11% validity in-distribution
- If v9 better: ~15-20%
- If worse: 0-5%

If smoke shows >5%: proceed.
If 0-5%: investigate before full sweep.

## Step E.2: Full UG sweep

tmux new-session -d -s ug_sweep_v9 "
cd /workspace/Demiurge
PYTHONUNBUFFERED=1 \
PYTHONPATH=/workspace/Demiurge/src \
python3 -u scripts/run_universal_guidance_sweep.py \
  --model-checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --cfg-scale 1.0 \
  --guidance-scales 0.0 0.5 1.0 2.0 4.0 8.0 \
  --seeds 42 123 456 \
  --n-prompts 200 \
  --n-per-prompt 5 \
  --drake-workers 24 \
  --output-dir artifacts/v9_ug_sweep/ \
  --budget-cap 5 \
  2>&1 | tee logs/v9_ug_sweep.log
"

echo "Waiting 120s for sweep to start..."
sleep 120

tmux list-sessions
tail -30 logs/v9_ug_sweep.log

Expected:
- 6 scales × 3 seeds = 18 runs
- 1000 scenes per run = 18,000 scenes total
- Wall-clock: 4-5 hours
- Cost: ~$3

Monitor every 30 min:
ssh ... 'tail -30 /workspace/Demiurge/logs/v9_ug_sweep.log'

## Step E.3: VLM judging on sweep subset

After sweep completes:

python3 -u scripts/judge_sweep_with_vlm.py \
  --sweep-dir artifacts/v9_ug_sweep/ \
  --samples-per-config 100 \
  --model claude-haiku-4-5 \
  --max-tokens 500 \
  --budget-usd 3 \
  --out artifacts/v9_ug_sweep/vlm_scores.jsonl

Expected cost: 18 configs × 100 samples ≈ $1.50-2.

## Step E.4: Pareto plot v9 (with v7 overlay)

python3 -u scripts/plot_pareto.py \
  --sweep-dir artifacts/v9_ug_sweep/ \
  --vlm-scores artifacts/v9_ug_sweep/vlm_scores.jsonl \
  --output-png artifacts/pareto_v9.png \
  --output-pdf artifacts/pareto_v9.pdf \
  --include-baselines \
  --reference-sweep artifacts/ug_sweep_v2/

Plot:
- X-axis: pairwise diversity
- Y-axis: Drake validity rate
- v9 points: 6 guidance scales × 3 seeds, error bars
- v7 points (reference, lighter color): same scales
- Annotate Pareto frontier
- Baselines: random procedural (9.0%), rejection (5.5%)

## Step E.5: Compare v9 vs v7 UG results

python3 - << 'PYEOF'
import json
from pathlib import Path

# Load v9 sweep
v9_results = {}
v9_file = Path("artifacts/v9_ug_sweep/summary.jsonl")
if v9_file.exists():
    with open(v9_file) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["guidance_scale"], rec["seed"])
            v9_results[key] = rec["validity_rate"]

# Load v7 sweep (from Week 5)
v7_results = {}
v7_file = Path("artifacts/ug_sweep_v2/summary.jsonl")
if v7_file.exists():
    with open(v7_file) as f:
        for line in f:
            rec = json.loads(line)
            key = (rec["guidance_scale"], rec["seed"])
            v7_results[key] = rec["validity_rate"]

# Group by guidance scale, average across seeds
import statistics

print("=== v7 vs v9 UG Results ===")
print(f"{'Scale':<8} {'v7 mean':<12} {'v9 mean':<12} {'Delta':<10}")
for scale in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
    v7_vals = [v7_results.get((scale, s), None) for s in [42, 123, 456]]
    v7_vals = [v for v in v7_vals if v is not None]
    v9_vals = [v9_results.get((scale, s), None) for s in [42, 123, 456]]
    v9_vals = [v for v in v9_vals if v is not None]
    
    if v7_vals and v9_vals:
        v7_mean = statistics.mean(v7_vals) * 100
        v9_mean = statistics.mean(v9_vals) * 100
        delta = v9_mean - v7_mean
        print(f"{scale:<8} {v7_mean:<12.1f} {v9_mean:<12.1f} {delta:+.1f}pp")
    else:
        print(f"{scale:<8} (data missing)")

# Best across scales
v7_best = max(v7_results.values()) * 100 if v7_results else 0
v9_best = max(v9_results.values()) * 100 if v9_results else 0
print(f"\nBest v7 UG: {v7_best:.1f}%")
print(f"Best v9 UG: {v9_best:.1f}%")
print(f"Improvement: {v9_best - v7_best:+.1f}pp")
PYEOF

## Step E.6: Phase E results document

python3 - << 'PYEOF'
import json
from datetime import datetime
from pathlib import Path

# Aggregate v9 UG sweep
v9_by_scale = {}
with open("artifacts/v9_ug_sweep/summary.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        scale = rec["guidance_scale"]
        if scale not in v9_by_scale:
            v9_by_scale[scale] = []
        v9_by_scale[scale].append(rec["validity_rate"])

import statistics
v9_summary = {
    scale: {
        "mean": statistics.mean(vals) * 100,
        "std": statistics.stdev(vals) * 100 if len(vals) > 1 else 0,
        "n_seeds": len(vals),
    }
    for scale, vals in v9_by_scale.items()
}

best_scale = max(v9_summary.keys(), key=lambda k: v9_summary[k]["mean"])
best_validity = v9_summary[best_scale]["mean"]

doc = f"""# Phase E Results: Universal Guidance Sweep on v9

Date: {datetime.now().strftime('%Y-%m-%d')}
Model: conditional_v9_cond (49M, warm-init from v9 uncond)
Sweep: 6 scales × 3 seeds × 200 prompts × 5 scenes = 18,000 scenes

## UG Sweep Results (v9)

| Scale | Mean Validity | Std | N Seeds |
|---|---|---|---|
"""

for scale in sorted(v9_summary.keys()):
    s = v9_summary[scale]
    doc += f"| {scale} | {s['mean']:.1f}% | {s['std']:.1f} | {s['n_seeds']} |\n"

doc += f"\n## Best UG Configuration\n\n"
doc += f"- Best scale: {best_scale}\n"
doc += f"- Best validity: {best_validity:.1f}%\n"

doc += f"\n## Comparison to v7 (Week 5)\n\n"
doc += f"- Original v7 UG best (scale=0.5): 11.1% in-distribution\n"
doc += f"- v9 UG best (scale={best_scale}): {best_validity:.1f}%\n"

if best_validity > 11.1:
    doc += f"- IMPROVEMENT: {best_validity - 11.1:+.1f}pp over v7\n"
else:
    doc += f"- REGRESSION: {best_validity - 11.1:+.1f}pp from v7\n"

doc += f"\n## Artifacts\n\n"
doc += f"- artifacts/v9_ug_sweep/summary.jsonl\n"
doc += f"- artifacts/v9_ug_sweep/vlm_scores.jsonl\n"
doc += f"- artifacts/pareto_v9.png\n"
doc += f"- artifacts/pareto_v9.pdf\n"

doc += f"\n## Next Step\n\nPhase F: full held-out evaluation harness.\n"
doc += f"v9 + UG@{best_scale} will be one of 5 samplers in eval.\n"

with open("artifacts/v9_ug_results.md", "w") as f:
    f.write(doc)

print(doc)
PYEOF

## Step E.7: Commit

cd /workspace/Demiurge

git add artifacts/v9_ug_sweep/
git add artifacts/pareto_v9.png artifacts/pareto_v9.pdf
git add artifacts/v9_ug_results.md
git commit -m "phase E: v9 UG sweep + Pareto plot

Best UG scale: <X>, validity <Y>%
Comparison to v7 UG (11.1% in-dist): <delta>"

git push origin week6/v9-phase-e 2>&1 || echo "push failed (expected)"

# Final Status

echo ""
echo "=================================================="
echo "PHASE E COMPLETE - UNIVERSAL GUIDANCE SWEEP"
echo "=================================================="
cat artifacts/v9_ug_results.md | head -30
echo ""
echo "=================================================="

exit

# Constraints

- python3 -u
- Cache VLM responses
- DO NOT modify Phase D checkpoint
- DO NOT pause pod during sweep

# Halt Conditions

- v9 cond checkpoint missing: halt
- Smoke test shows 0% validity: halt, investigate
- Sweep produces NaN at any config: halt
- Total cost > $7: halt and surface

# Cost Estimate

- Pod time: 5 hours × $0.74/hr = ~$4
- VLM API: $1-2
- Total: ~$5-6

Begin with Step 0.1.
