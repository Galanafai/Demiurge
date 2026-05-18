# Antigravity Master Prompt — Phase F: Full Held-Out Evaluation on v9

You are continuing Phase F of the v9 build. Phases A-E complete:
- A: 234k combined dataset with locked splits
- B: v9 unconditional (49M, 200k steps)
- C: VLM gate passed
- D: v9 conditional (warm-init, CFG dropout)
- E: UG sweep on v9, Pareto plot generated

Phase F: Run Week 5 eval harness against v9. Compare to v7 directly using
same held-out benchmark.

# Pre-Flight

## Step 0.1: Branch + verify all phases

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

cd /workspace/Demiurge

git checkout main
git checkout -b week6/v9-phase-f

# Verify all v9 checkpoints
ls -lh checkpoints/conditional_v9_uncond/latest.pt
ls -lh checkpoints/conditional_v9_cond/latest.pt

# Verify W&B
python3 -c "
import wandb
api = wandb.Api()
for name in ['conditional_v9_uncond_checkpoint', 'conditional_v9_cond_checkpoint', 'conditional_v7_checkpoint']:
    try:
        art = api.artifact(f'galanafai-self/demiurge/{name}:latest')
        print(f'{name}: {art.size / 1024 / 1024:.1f} MB')
    except Exception as e:
        print(f'{name}: MISSING - {e}')
"

# Verify Phase E results
cat artifacts/v9_ug_results.md | head -20
ls -lh artifacts/v9_ug_sweep/summary.jsonl

# Verify Week 5 eval suite exists (LOCKED benchmark)
ls -lh data/eval_suite_v1/
echo "200 held-out prompts (locked from Week 5)"

# Verify eval harness scripts
ls -lh scripts/run_eval.py
ls -lh src/eval/metrics.py

# Before Doing Anything Else

1. Read AGENTS.md, SKILL.md.
2. Read artifacts/v9_ug_results.md (Phase E best UG scale).
3. Read artifacts/week5_results.md (v7 reference numbers).
4. Read scripts/run_eval.py (eval harness).
5. State in Plan Artifact:
   - Eval samplers: 5 (v9 uncond, v9 cond, v9 rejection, v9+UG@best, v9+UG@1)
   - Seeds: 3
   - Held-out prompts: 200 (locked benchmark)
   - Expected wall-clock: 3-4 hours
   - Expected cost: $2-3

# Phase F Execution

## Step F.1: Identify best v9 UG scale from Phase E

python3 - << 'PYEOF'
import json
import statistics

results = {}
with open("artifacts/v9_ug_sweep/summary.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        scale = rec["guidance_scale"]
        if scale not in results:
            results[scale] = []
        results[scale].append(rec["validity_rate"])

# Average per scale
by_scale = {s: statistics.mean(vals) for s, vals in results.items()}
best_scale = max(by_scale, key=by_scale.get)
print(f"Best UG scale: {best_scale} ({by_scale[best_scale]*100:.1f}%)")

# Save for use in eval
with open("artifacts/v9_best_ug_scale.json", "w") as f:
    json.dump({"best_scale": best_scale, "validity": by_scale[best_scale]}, f)
PYEOF

BEST_UG=$(python3 -c "import json; print(json.load(open('artifacts/v9_best_ug_scale.json'))['best_scale'])")
echo "Best v9 UG scale: $BEST_UG"

## Step F.2: Run eval harness on 5 samplers

The eval harness from Week 5 supports multiple samplers. Configure for v9.

cat > /workspace/Demiurge/configs/eval/v9_samplers.yaml << YAMLEOF
samplers:
  - name: v9_uncond
    type: unconditional
    checkpoint: checkpoints/conditional_v9_uncond/latest.pt
    head: ema
    
  - name: v9_cond_baseline
    type: conditional
    checkpoint: checkpoints/conditional_v9_cond/latest.pt
    head: ema
    cfg_scale: 1.0
    
  - name: v9_rejection
    type: rejection
    checkpoint: checkpoints/conditional_v9_cond/latest.pt
    head: ema
    cfg_scale: 1.0
    max_attempts: 50
    
  - name: v9_ug_best
    type: universal_guidance
    checkpoint: checkpoints/conditional_v9_cond/latest.pt
    head: ema
    cfg_scale: 1.0
    guidance_scale: $BEST_UG
    
  - name: v9_ug_scale1
    type: universal_guidance
    checkpoint: checkpoints/conditional_v9_cond/latest.pt
    head: ema
    cfg_scale: 1.0
    guidance_scale: 1.0

eval:
  prompts_file: data/eval_suite_v1/heldout_prompts.json
  n_scenes_per_prompt: 8
  seeds: [42, 123, 456]
  drake_workers: 24
  output_dir: artifacts/eval_v9/
  metrics:
    - raw_validity_rate
    - diversity
    - prompt_following
YAMLEOF

# Launch eval in tmux
tmux new-session -d -s eval_v9 "
cd /workspace/Demiurge
PYTHONUNBUFFERED=1 \
PYTHONPATH=/workspace/Demiurge/src \
python3 -u scripts/run_eval.py \
  --config configs/eval/v9_samplers.yaml \
  2>&1 | tee logs/eval_v9.log
"

echo "Waiting 120s for eval to start..."
sleep 120

tmux list-sessions
tail -30 logs/eval_v9.log

Expected:
- 5 samplers × 3 seeds × 200 prompts × 8 scenes = 24,000 scenes total
- Wall-clock: 3-4 hours with parallel Drake
- Cost: ~$2-3

## Step F.3: Wait for completion

Monitor progress:
ssh ... 'tail -50 /workspace/Demiurge/logs/eval_v9.log'

When eval ends:
ssh ... 'tmux list-sessions | grep eval_v9 || echo "eval done"'

## Step F.4: Build comparison table v7 vs v9

python3 - << 'PYEOF'
import json
import statistics
from pathlib import Path
from datetime import datetime

# Load v9 eval results
v9_results = {}
for sampler in ["v9_uncond", "v9_cond_baseline", "v9_rejection", "v9_ug_best", "v9_ug_scale1"]:
    path = Path(f"artifacts/eval_v9/{sampler}.json")
    if path.exists():
        with open(path) as f:
            v9_results[sampler] = json.load(f)

# Load v7 eval results (from Week 5)
v7_results = {}
for sampler in ["v7_uncond", "v7_cond_baseline", "v7_rejection", "v7_ug_best", "v7_ug_scale1"]:
    path = Path(f"artifacts/eval_v1/{sampler}.json")
    if path.exists():
        with open(path) as f:
            v7_results[sampler] = json.load(f)

# Map for comparison
sampler_pairs = [
    ("Uncond", "v7_uncond", "v9_uncond"),
    ("Cond baseline", "v7_cond_baseline", "v9_cond_baseline"),
    ("Rejection", "v7_rejection", "v9_rejection"),
    ("UG@best", "v7_ug_best", "v9_ug_best"),
    ("UG@1.0", "v7_ug_scale1", "v9_ug_scale1"),
]

doc = f"""# v7 vs v9 Comparison Table

Date: {datetime.now().strftime('%Y-%m-%d')}
Benchmark: 200 held-out prompts (locked since Week 5)
Samples per prompt: 8
Seeds: 3 per sampler

## Drake Validity Comparison

| Sampler | v7 (Week 5) | v9 (this run) | Delta |
|---|---|---|---|
"""

for label, v7_key, v9_key in sampler_pairs:
    v7 = v7_results.get(v7_key, {})
    v9 = v9_results.get(v9_key, {})
    v7_val = v7.get("validity_rate_mean", 0) * 100 if v7 else 0
    v9_val = v9.get("validity_rate_mean", 0) * 100 if v9 else 0
    delta = v9_val - v7_val
    doc += f"| {label} | {v7_val:.1f}% | {v9_val:.1f}% | {delta:+.1f}pp |\n"

doc += "\n## Diversity Comparison\n\n"
doc += "| Sampler | v7 diversity | v9 diversity | Delta |\n"
doc += "|---|---|---|---|\n"
for label, v7_key, v9_key in sampler_pairs:
    v7 = v7_results.get(v7_key, {})
    v9 = v9_results.get(v9_key, {})
    v7_div = v7.get("diversity_mean", 0)
    v9_div = v9.get("diversity_mean", 0)
    delta = v9_div - v7_div
    doc += f"| {label} | {v7_div:.3f} | {v9_div:.3f} | {delta:+.3f} |\n"

doc += "\n## Prompt Following (VLM-judged)\n\n"
doc += "| Sampler | v7 VLM mean | v9 VLM mean | Delta |\n"
doc += "|---|---|---|---|\n"
for label, v7_key, v9_key in sampler_pairs:
    v7 = v7_results.get(v7_key, {})
    v9 = v9_results.get(v9_key, {})
    v7_vlm = v7.get("vlm_mean", None)
    v9_vlm = v9.get("vlm_mean", None)
    if v7_vlm is not None and v9_vlm is not None:
        delta = v9_vlm - v7_vlm
        doc += f"| {label} | {v7_vlm:.2f} | {v9_vlm:.2f} | {delta:+.2f} |\n"

# Headline number
v9_best = max(v9.get("validity_rate_mean", 0) for v9 in v9_results.values()) * 100
v7_best = max(v7.get("validity_rate_mean", 0) for v7 in v7_results.values()) * 100
doc += f"\n## Headline Result\n\n"
doc += f"- Best v7 sampler: {v7_best:.1f}% Drake validity on held-out\n"
doc += f"- Best v9 sampler: {v9_best:.1f}% Drake validity on held-out\n"
doc += f"- Improvement: {v9_best - v7_best:+.1f}pp\n"

with open("artifacts/results_table_v9.md", "w") as f:
    f.write(doc)

print(doc)
PYEOF

## Step F.5: Update Pareto plot with held-out eval data

python3 -u scripts/plot_pareto.py \
  --sweep-dir artifacts/v9_ug_sweep/ \
  --eval-dir artifacts/eval_v9/ \
  --vlm-scores artifacts/v9_ug_sweep/vlm_scores.jsonl \
  --output-png artifacts/pareto_v9_final.png \
  --output-pdf artifacts/pareto_v9_final.pdf \
  --include-baselines \
  --include-heldout \
  --reference-sweep artifacts/ug_sweep_v2/

## Step F.6: Ablation - pick 1

Best single ablation: UG bounding radius scaling. Quick to run, interesting result.

Or: guidance schedule variants.

Or skip ablations if out of time.

If running: 
python3 -u scripts/run_ablation_ug_radius.py \
  --checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --radius-scales 0.5 1.0 1.5 2.0 \
  --guidance-scale $BEST_UG \
  --n-prompts 100 \
  --seeds 42 \
  --out artifacts/v9_ablation_ug_radius.json

## Step F.7: Commit Phase F

git add artifacts/eval_v9/
git add artifacts/results_table_v9.md
git add artifacts/pareto_v9_final.png artifacts/pareto_v9_final.pdf
git add configs/eval/v9_samplers.yaml
git commit -m "phase F: full held-out evaluation v7 vs v9

Best v9 sampler: <name> at <X>%
Best v7 sampler: <name> at <Y>%
Improvement: <delta>pp"

git push origin week6/v9-phase-f 2>&1 || echo "push failed (expected)"

# Final Status

echo ""
echo "=================================================="
echo "PHASE F COMPLETE - FULL EVALUATION"
echo "=================================================="
cat artifacts/results_table_v9.md | head -40
echo ""
echo "=================================================="

exit

# Constraints

- python3 -u
- DO NOT regenerate eval suite (locked)
- Cache VLM responses
- DO NOT pause pod during eval

# Halt Conditions

- v9 cond checkpoint missing: halt
- eval_v9/ directory missing after eval: halt
- v7 results missing (for comparison): warn but continue
- Total cost > $5: halt

# Cost Estimate

- Pod time: 4 hours × $0.74/hr = ~$3
- VLM API: $2
- Total: ~$5

Begin with Step 0.1.
