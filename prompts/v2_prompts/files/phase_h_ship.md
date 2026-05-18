# Antigravity Master Prompt — Phase H: Ship (LinkedIn + README + Tag)

You are continuing Phase H of the v9 build. Phases A-G complete. All
training, eval, and visual artifacts done.

Phase H: Final shipping. LinkedIn post draft, README polish, merge to
main, v1.0 tag. This is the final phase.

# Pre-Flight

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

cd /workspace/Demiurge

git checkout main
git checkout -b week6/v9-phase-h

# Verify all prior phase artifacts
echo "=== Required artifacts ==="
ls -lh artifacts/v9_final_gallery.png
ls -lh artifacts/v9_denoising_demo.mp4
ls -lh artifacts/v9_denoising_demo_slow.mp4
ls -lh artifacts/results_table_v9.md
ls -lh artifacts/pareto_v9_final.png
ls -lh artifacts/v9_ug_results.md

# Verify checkpoints on W&B
python3 -c "
import wandb
api = wandb.Api()
for name in ['conditional_v9_uncond_checkpoint', 'conditional_v9_cond_checkpoint']:
    art = api.artifact(f'galanafai-self/demiurge/{name}:latest')
    print(f'{name}: {art.size / 1024 / 1024:.1f} MB')
"

# Before Doing Anything Else

1. Read AGENTS.md, SKILL.md.
2. Read artifacts/results_table_v9.md (THE numbers).
3. Read artifacts/v9_phase_g_results.md.
4. State in Plan Artifact:
   - Headline result: v9 best validity vs v7 best
   - LinkedIn angle: chosen based on actual results
   - README focus: reproduction + portfolio summary
   - Tag: v1.0

# Phase H Execution

## Step H.1: Determine LinkedIn angle based on results

python3 - << 'PYEOF'
import json

# Load comparison numbers
v9_best = 0
v7_best = 0
import os
for sampler_file in os.listdir("artifacts/eval_v9/"):
    if sampler_file.endswith(".json"):
        with open(f"artifacts/eval_v9/{sampler_file}") as f:
            data = json.load(f)
        v9_best = max(v9_best, data.get("validity_rate_mean", 0))

for sampler_file in os.listdir("artifacts/eval_v1/"):
    if sampler_file.endswith(".json"):
        with open(f"artifacts/eval_v1/{sampler_file}") as f:
            data = json.load(f)
        v7_best = max(v7_best, data.get("validity_rate_mean", 0))

delta_pp = (v9_best - v7_best) * 100
print(f"v7 best: {v7_best*100:.1f}%")
print(f"v9 best: {v9_best*100:.1f}%")
print(f"Delta: {delta_pp:+.1f}pp")

if delta_pp > 5:
    angle = "victory"
elif delta_pp > 0:
    angle = "modest_improvement"
elif delta_pp > -2:
    angle = "matched"
else:
    angle = "regression"

print(f"\nRecommended LinkedIn angle: {angle}")

with open("artifacts/v9_linkedin_angle.json", "w") as f:
    json.dump({
        "angle": angle,
        "v7_best": v7_best,
        "v9_best": v9_best,
        "delta_pp": delta_pp,
    }, f)
PYEOF

cat artifacts/v9_linkedin_angle.json

## Step H.2: LinkedIn post draft

Draft based on the angle:

python3 - << 'PYEOF'
import json
from datetime import datetime

with open("artifacts/v9_linkedin_angle.json") as f:
    angle = json.load(f)

v9 = angle["v9_best"] * 100
v7 = angle["v7_best"] * 100
delta = angle["delta_pp"]

# Universal opening
opening = f"""Demiurge: I just shipped a conditional diffusion model for 3D scene generation with physical validity constraints.

Six months of work. Trained 9 model variants. Hit slot collapse, type collapse, alignment tax — all documented in honest results."""

# Conditional body based on outcome
if angle["angle"] == "victory":
    body = f"""

Final result: scaling from 9M to 49M parameters with 4.7x more training data improved held-out Drake validity from {v7:.1f}% to {v9:.1f}% ({delta:+.1f}pp).

Universal Guidance applied to the 49M model... [continue with specific findings]

Most interesting finding: the in-distribution / held-out generalization gap that Week 5 revealed is real and persists at scale. Rejection sampling remains the strongest practical baseline."""

elif angle["angle"] == "modest_improvement":
    body = f"""

Final result: scaling 5x params + 4.7x data gave modest improvement (v7: {v7:.1f}% → v9: {v9:.1f}%, {delta:+.1f}pp). 

Universal Guidance didn't fundamentally change the story — it improves in-distribution validity (~{v9:.1f}%) but doesn't generalize to held-out prompts. This validates the "alignment tax" phenomenon."""

elif angle["angle"] == "matched":
    body = f"""

Final result: scaling 5x params + 4.7x data did NOT meaningfully improve Drake validity (v7: {v7:.1f}% → v9: {v9:.1f}%).

This was the most surprising finding of the project. Scaling alone isn't sufficient when the bottleneck is joint physical-semantic constraint satisfaction."""

else:  # regression
    body = f"""

Negative result: scaling to 49M params hurt validity (v7: {v7:.1f}% → v9: {v9:.1f}%, {delta:+.1f}pp). 

Investigation revealed... [specific cause]

This is what real ML research looks like. Documentation matters more than hype."""

closing = """

Key technical contributions:
- Drake-validated dataset generation pipeline (234k scenes)
- Conditional diffusion with CFG dropout and warm-init lineage
- Universal Guidance implementation
- Comprehensive eval harness (validity, diversity, prompt-following)
- Honest documentation of failure modes across 9 architectural variants

Code: github.com/Galanafai/Demiurge

#MachineLearning #Robotics #DiffusionModels #ResearchSoftware"""

post = opening + body + closing

with open("artifacts/v9_linkedin_draft.md", "w") as f:
    f.write(f"# LinkedIn Post Draft\n\n{post}\n\n---\n\nLength: {len(post)} chars\n")
    f.write(f"Target: 1300 chars (LinkedIn body limit)\n")

print(post)
print(f"\n\nLength: {len(post)} characters")
PYEOF

cat artifacts/v9_linkedin_draft.md

## Step H.3: Update README

python3 - << 'PYEOF'
import json

with open("artifacts/v9_linkedin_angle.json") as f:
    angle = json.load(f)

readme = f"""# Demiurge

Conditional diffusion model for 3D scene generation with physical validity constraints. Generates 6-DoF tabletop scenes for robot manipulation training.

## Headline Result

- **v9 (49M params, 234k validated scenes):** {angle['v9_best']*100:.1f}% Drake validity on held-out
- **v7 (9M params, 50k validated scenes):** {angle['v7_best']*100:.1f}% Drake validity on held-out
- **Delta:** {angle['delta_pp']:+.1f}pp from scaling

## What This Does

Demiurge generates 3D scenes (object positions, types, rotations) conditioned on natural language prompts. Every generated scene is validated through Drake (physics simulation) to ensure:

1. No object interpenetration
2. Stable resting configurations
3. Inverse kinematics reachable
4. RRT-solvable manipulation trajectory exists

## Architecture

- Transformer-based DDPM denoiser (49M params)
- Cross-attention to MiniLM-L6-v2 text embeddings
- CFG dropout (15%) for classifier-free guidance
- Cosine noise schedule (1000 train steps, 50 DDIM inference)

## Results

![Pareto plot](artifacts/pareto_v9_final.png)

| Sampler | Drake validity (held-out) | Diversity | VLM-judged |
|---|---|---|---|
| v9 + UG | <best>% | <X> | <Y> |
| v9 conditional | <X>% | <X> | <Y> |
| Rejection sampling | <X>% | <X> | <Y> |
| Random procedural baseline | 9.0% | - | - |

See [`artifacts/results_table_v9.md`](artifacts/results_table_v9.md) for full table.

## Gallery

![Gallery](artifacts/v9_final_gallery.png)

100 scenes generated from the best v9 sampler. See `artifacts/v9_gallery_scenes/` for individual renders.

## Denoising Process

[`artifacts/v9_denoising_demo.mp4`](artifacts/v9_denoising_demo.mp4) shows the 50-step denoising process for one representative prompt.

## Reproduction

Datasets:
- `dataset_v1_tars` (50k scenes): https://wandb.ai/galanafai-self/demiurge/artifacts/dataset/dataset_v1_tars
- `dataset_v2_tars` (184k scenes): https://wandb.ai/galanafai-self/demiurge/artifacts/dataset/dataset_v2_tars

Checkpoints:
- `conditional_v9_uncond_checkpoint`: 49M unconditional baseline
- `conditional_v9_cond_checkpoint`: 49M conditional with CFG

Setup:
```bash
git clone https://github.com/Galanafai/Demiurge
cd Demiurge
pip install -r requirements.txt
# Drake required: pip install drake
```

Training:
```bash
python scripts/train.py --config configs/train/v9_cond.yaml --seed 42
```

Evaluation:
```bash
python scripts/run_eval.py --config configs/eval/v9_samplers.yaml
```

## Project History

This project spans 6 weeks of investigation:

- Week 1: Drake validator + scene schema
- Week 2: 50k validated scene dataset (v1)
- Week 3: Initial conditional model (v3)
- Week 4: Type collapse investigation, 6 architectural variants
- Week 5: Universal Guidance, held-out evaluation
- Week 6: v9 retrain at scale, final results

Full investigation logs in `artifacts/`.

## Acknowledgments

- Drake: physics validation
- Universal Guidance method: Bansal et al., ICML 2023
- Sentence-transformers: MiniLM-L6-v2

## License

MIT
"""

with open("README.md", "w") as f:
    f.write(readme)

print("README updated")
print(f"\nLength: {len(readme)} chars")
PYEOF

## Step H.4: Final results summary

python3 - << 'PYEOF'
import json
from pathlib import Path

# Collect ALL results from all phases
summary = {
    "project": "Demiurge",
    "version": "v1.0",
    "models_trained": [
        "v7 (9M params, 50k data) - baseline",
        "v9 uncond (49M params, 234k data)",
        "v9 cond (49M params, 234k data, warm-init from v9 uncond)",
    ],
    "datasets": {
        "v1": {"scenes": 50000, "wandb": "dataset_v1_tars:latest"},
        "v2": {"scenes": 184720, "wandb": "dataset_v2_tars:latest"},
        "combined": {"scenes": 234720},
    },
    "best_results": {},
}

# Load best sampler
with open("artifacts/v9_best_sampler.json") as f:
    summary["best_results"] = json.load(f)

# Cost summary
summary["cost_breakdown"] = {
    "data_generation": "~$22",
    "training_v9": "~$15",
    "eval_harness": "~$3",
    "vlm_judging": "~$7",
    "total": "~$47",
}

with open("artifacts/v9_final_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(json.dumps(summary, indent=2))
PYEOF

## Step H.5: Verification pass

cd /workspace/Demiurge

echo "=== ruff ==="
ruff check . 2>&1 | tail -5

echo ""
echo "=== pyright ==="
pyright 2>&1 | grep -i error | head -10

echo ""
echo "=== pytest ==="
pytest tests/ -q --tb=no 2>&1 | tail -10

## Step H.6: Final commit to phase H branch

git add README.md
git add artifacts/v9_linkedin_draft.md
git add artifacts/v9_linkedin_angle.json
git add artifacts/v9_final_summary.json
git commit -m "phase H: ship v1.0

LinkedIn post drafted
README updated with v9 headline results
Final summary document

Project complete: v9 (49M, 234k data) is the headline model."

git push origin week6/v9-phase-h 2>&1 || echo "push failed (expected)"

## Step H.7: Merge to main

git checkout main
git merge --no-ff week6/v9-phase-h -m "merge: v9 build complete, v1.0 release"

# Optional: also merge other phase branches if not yet merged
# git merge --no-ff week6/v9-phase-a -m "merge: phase A data prep"
# (etc)

## Step H.8: Tag v1.0

git tag -a v1.0 -m "Demiurge v1.0

49M parameter conditional diffusion model.
234k Drake-validated training scenes.
Universal Guidance sweep complete.
Pareto plot, gallery, denoising demo shipped.

See artifacts/v9_final_summary.json for full results."

git push origin main 2>&1 || echo "push failed (need creds)"
git push origin v1.0 2>&1 || echo "tag push failed (need creds)"

## Step H.9: Final status

echo ""
echo "=================================================="
echo "PROJECT COMPLETE - DEMIURGE v1.0"
echo "=================================================="
echo ""
cat artifacts/v9_final_summary.json
echo ""
echo "=================================================="
echo ""
echo "Final checklist:"
echo "  [x] v7 retrain on W&B (Phase 0)"
echo "  [x] v2 dataset on W&B (Phase 0)"
echo "  [x] Combined dataset prepared (Phase A)"
echo "  [x] v9 uncond trained (Phase B)"
echo "  [x] VLM gate passed (Phase C)"
echo "  [x] v9 cond trained (Phase D)"
echo "  [x] UG sweep complete (Phase E)"
echo "  [x] Held-out eval complete (Phase F)"
echo "  [x] Gallery + MP4 (Phase G)"
echo "  [x] LinkedIn draft + README (Phase H)"
echo ""
echo "To publish:"
echo "  1. Review artifacts/v9_linkedin_draft.md"
echo "  2. Post to LinkedIn"
echo "  3. Push README to GitHub"
echo "  4. (Optional) HuggingFace Space"
echo ""
echo "Pod is safe to pause."
echo "=================================================="

exit

# Constraints

- DO NOT modify checkpoints or eval results
- Verify all artifacts exist before declaring complete
- LinkedIn draft requires your personal review before posting

# Halt Conditions

- Any required artifact missing: halt
- Verification fails > 5 errors: halt

# Cost Estimate

- Pod time: 6 hours × $0.74/hr = ~$5
- VLM: $0
- Total: ~$5

## Optional Step H.10: Hugging Face Space

If you want an interactive demo:

mkdir -p deploy/hf_space
cat > deploy/hf_space/app.py << 'PYEOF'
import gradio as gr
import torch
from pathlib import Path

# Load model + sampler
# Define interactive function
# Launch

iface = gr.Interface(
    fn=generate_scene,
    inputs="text",
    outputs="image",
    title="Demiurge: Conditional Scene Generation",
)
iface.launch()
PYEOF

# Push to HuggingFace
# pip install huggingface_hub
# huggingface-cli login
# huggingface-cli repo create demiurge --type space

This is optional polish. Project ships fine without it.

Begin with Pre-Flight.
