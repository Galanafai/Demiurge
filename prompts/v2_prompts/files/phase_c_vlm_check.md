# Antigravity Master Prompt — Phase C: VLM Quality Check on v9 Uncond

You are continuing Phase C of the v9 build. Phase B trained the 49M
unconditional model on 234k scenes. Auto-uploaded to W&B as
conditional_v9_uncond_checkpoint:latest.

Phase C: Visual sanity check via VLM judging BEFORE expensive
conditional training in Phase D. This catches Drake-valid-but-
visually-wrong scenes (e.g., right geometry, wrong types).

# Pre-Flight

## Step 0.1: Branch + verify Phase B completion

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

In session:

cd /workspace/Demiurge

git checkout main
git checkout -b week6/v9-phase-c

# Verify Phase B artifacts
echo "=== Phase B verification ==="
ls -lh checkpoints/conditional_v9_uncond/latest.pt
cat logs/upload_v9_uncond_monitor.log | tail -5

# Confirm v9 uncond on W&B
python3 - << 'PYEOF'
import wandb
api = wandb.Api()
try:
    art = api.artifact("galanafai-self/demiurge/conditional_v9_uncond_checkpoint:latest")
    print(f"v9 uncond on W&B: {art.size / 1024 / 1024:.1f} MB")
    print(f"Created: {art.created_at}")
except Exception as e:
    print(f"FAIL: v9 uncond not on W&B - {e}")
    print("HALT - Phase B did not complete properly")
PYEOF

# Verify Drake probe from Phase B
ls -lh artifacts/v9_uncond_phase_b_results.json
cat artifacts/v9_uncond_phase_b_results.json | python3 -c "
import json, sys
d = json.load(sys.stdin)
v = d.get('validity_rate', 0) * 100
print(f'Phase B Drake validity: {v:.1f}%')
if v < 5.0:
    print('WARNING: Below Phase B gate threshold of 5%')
"

## Step 0.2: ANTHROPIC_API_KEY check

echo "=== VLM API key ==="
[ -n "$ANTHROPIC_API_KEY" ] && echo "ANTHROPIC_API_KEY: SET" || echo "ANTHROPIC_API_KEY: MISSING - source ~/.bashrc"

# Test API connectivity
python3 - << 'PYEOF'
import os
import anthropic
client = anthropic.Anthropic()
try:
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=20,
        messages=[{"role": "user", "content": "ping"}],
    )
    print("VLM API reachable")
except Exception as e:
    print(f"VLM API failed: {e}")
PYEOF

## Step 0.3: Verify VLM judging infrastructure exists

ls -lh scripts/judge_with_vlm.py 2>/dev/null && echo "VLM script: OK" || echo "VLM script: MISSING"
ls -lh src/eval/prompts/scene_judge.txt 2>/dev/null && echo "Judge prompt: OK" || echo "Judge prompt: MISSING"

If missing: surface and halt. Use Week 5 commits as reference for what
should exist.

# Before Doing Anything Else

1. Read AGENTS.md and SKILL.md.
2. Read artifacts/v9_uncond_phase_b_results.md.
3. Read src/eval/prompts/scene_judge.txt (judge prompt template).
4. State in Plan Artifact:
   - Phase B Drake validity: X%
   - Target VLM mean score: > 2.0
   - Sample size: 100 unconditional scenes
   - VLM model: Claude Haiku 4.5
   - Expected cost: $1-3

# Phase C Execution

## Step C.1: Generate 100 unconditional scenes for judging

mkdir -p artifacts/v9_uncond_vlm_subset

python3 -u scripts/generate_for_vlm_judging.py \
  --config configs/train/v9_uncond.yaml \
  --checkpoint checkpoints/conditional_v9_uncond/latest.pt \
  --n-scenes 100 \
  --head ema \
  --seed 42 \
  --text-mode none \
  --drake-workers 24 \
  --render \
  --output-dir artifacts/v9_uncond_vlm_subset/ \
  2>&1 | tee logs/v9_phase_c_generate.log

If generate_for_vlm_judging.py doesn't exist, adapt scripts/probe_drake.py:

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import torch
import json
import os
from pathlib import Path
from model.denoiser import SceneDenoiser
from model.schedule import DiffusionSchedule
from model.sampler import DDIMSampler
from validator.core import SceneValidator
from rendering.offscreen import render_scene_to_png  # adapt if name differs

# Load model
ckpt = torch.load("checkpoints/conditional_v9_uncond/latest.pt", map_location="cuda")
model = SceneDenoiser(d_model=384, n_layers=16, n_heads=8, ffn_multiplier=4, n_max=12, n_types=12, use_cross_attention=False).cuda()
model.load_state_dict(ckpt["ema_state"] if "ema_state" in ckpt else ckpt["model_state"])
model.eval()

# Sample scenes
sampler = DDIMSampler(model, DiffusionSchedule(), n_inference_steps=50)
validator = SceneValidator(rrt_budget_s=1.0)

out_dir = Path("artifacts/v9_uncond_vlm_subset")
out_dir.mkdir(parents=True, exist_ok=True)

n_target = 100
n_generated = 0
n_attempted = 0

records = []
while n_generated < n_target and n_attempted < n_target * 3:
    with torch.no_grad():
        scene = sampler.sample(batch_size=1, text_emb=None)[0]
    n_attempted += 1
    
    # Validate via Drake
    try:
        report = validator.validate(scene, rrt_seed=n_attempted)
        accepted = report.accepted
    except Exception:
        accepted = False
    
    # Render to PNG
    png_path = out_dir / f"scene_{n_generated:04d}.png"
    render_scene_to_png(scene, str(png_path))
    
    records.append({
        "idx": n_generated,
        "png": str(png_path),
        "drake_accepted": accepted,
        "n_objects": int(scene.presence.sum()),
        "types": [int(scene.object_types[i]) for i in range(12) if scene.presence[i]],
    })
    n_generated += 1
    print(f"Generated {n_generated}/{n_target} (attempt {n_attempted})")

with open(out_dir / "manifest.json", "w") as f:
    json.dump(records, f, indent=2)

print(f"\n100 scenes generated")
print(f"Drake-valid: {sum(r['drake_accepted'] for r in records)}/{n_generated}")
PYEOF

ls -lh artifacts/v9_uncond_vlm_subset/ | head -10

## Step C.2: VLM scoring on 100 scenes

python3 -u scripts/judge_with_vlm.py \
  --input-dir artifacts/v9_uncond_vlm_subset/ \
  --output artifacts/v9_uncond_vlm_scores.jsonl \
  --judge-prompt src/eval/prompts/scene_judge.txt \
  --model claude-haiku-4-5 \
  --max-tokens 500 \
  --budget-usd 5 \
  --cache-dir .vlm_cache/

If judge_with_vlm.py doesn't exist, build it:

python3 - << 'PYEOF'
import sys
import json
import hashlib
import base64
from pathlib import Path
import anthropic

client = anthropic.Anthropic()

with open("src/eval/prompts/scene_judge.txt") as f:
    judge_prompt = f.read()

with open("artifacts/v9_uncond_vlm_subset/manifest.json") as f:
    records = json.load(f)

cache_dir = Path(".vlm_cache")
cache_dir.mkdir(exist_ok=True)

results = []
total_cost = 0.0
for rec in records:
    png_path = rec["png"]
    
    # Cache key
    with open(png_path, "rb") as f:
        png_bytes = f.read()
    key = hashlib.sha256(png_bytes + judge_prompt.encode()).hexdigest()
    cache_file = cache_dir / f"{key}.json"
    
    if cache_file.exists():
        with open(cache_file) as f:
            response_data = json.load(f)
        print(f"  cache hit for {png_path}")
    else:
        # Call VLM
        b64 = base64.standard_b64encode(png_bytes).decode()
        msg = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": judge_prompt},
                ]
            }]
        )
        response_data = {
            "text": msg.content[0].text,
            "input_tokens": msg.usage.input_tokens,
            "output_tokens": msg.usage.output_tokens,
        }
        # Approximate cost: Haiku 4.5 input $1/1M, output $5/1M
        cost = (response_data["input_tokens"] * 1e-6) + (response_data["output_tokens"] * 5e-6)
        total_cost += cost
        with open(cache_file, "w") as f:
            json.dump(response_data, f)
        print(f"  judged {png_path}: ${cost:.4f}")
    
    # Parse score from response text
    text = response_data["text"]
    score = None
    for line in text.split("\n"):
        if "score:" in line.lower() or "rating:" in line.lower():
            for tok in line.split():
                try:
                    score = float(tok)
                    break
                except ValueError:
                    continue
        if score is not None:
            break
    
    rec_out = {**rec, "vlm_score": score, "vlm_text": text}
    results.append(rec_out)

print(f"\nTotal API cost: ${total_cost:.2f}")

# Save
with open("artifacts/v9_uncond_vlm_scores.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

# Compute stats
scores = [r["vlm_score"] for r in results if r["vlm_score"] is not None]
if scores:
    import statistics
    print(f"\nVLM scores ({len(scores)}/{len(results)} parsed):")
    print(f"  Mean: {statistics.mean(scores):.2f}")
    print(f"  Median: {statistics.median(scores):.2f}")
    print(f"  Stdev: {statistics.stdev(scores) if len(scores) > 1 else 0:.2f}")
    print(f"  Min: {min(scores)}, Max: {max(scores)}")
    
    from collections import Counter
    print(f"\nScore distribution:")
    dist = Counter(int(s) for s in scores)
    for score, count in sorted(dist.items()):
        print(f"  {score}: {count} ({100*count/len(scores):.0f}%)")
PYEOF

cat artifacts/v9_uncond_vlm_scores.jsonl | wc -l
echo "Scores saved"

## Step C.3: Score analysis

python3 - << 'PYEOF'
import json
import statistics
from collections import Counter

scores = []
with open("artifacts/v9_uncond_vlm_scores.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("vlm_score") is not None:
            scores.append({
                "score": rec["vlm_score"],
                "drake_accepted": rec.get("drake_accepted", False),
                "types": rec.get("types", []),
            })

print(f"\n=== VLM Score Analysis ===")
print(f"Total scored: {len(scores)}")

# Overall mean
all_scores = [s["score"] for s in scores]
print(f"Mean: {statistics.mean(all_scores):.2f}")
print(f"Median: {statistics.median(all_scores):.2f}")

# Score distribution
dist = Counter(int(s) for s in all_scores)
print(f"\nDistribution:")
for score in sorted(dist.keys()):
    pct = 100 * dist[score] / len(all_scores)
    bar = "*" * int(pct / 2)
    print(f"  {score}: {dist[score]:3} ({pct:5.1f}%) {bar}")

# Score x Drake validity
drake_valid = [s for s in scores if s["drake_accepted"]]
drake_invalid = [s for s in scores if not s["drake_accepted"]]
if drake_valid:
    print(f"\nDrake-valid scenes ({len(drake_valid)}):")
    print(f"  Mean VLM score: {statistics.mean([s['score'] for s in drake_valid]):.2f}")
if drake_invalid:
    print(f"\nDrake-invalid scenes ({len(drake_invalid)}):")
    print(f"  Mean VLM score: {statistics.mean([s['score'] for s in drake_invalid]):.2f}")

# Object types covered
all_types = set()
for s in scores:
    all_types.update(s["types"])
print(f"\nObject types in sample: {sorted(all_types)} ({len(all_types)}/12)")

# Save summary
summary = {
    "n_scored": len(scores),
    "mean_score": statistics.mean(all_scores),
    "median_score": statistics.median(all_scores),
    "distribution": {str(k): v for k, v in dist.items()},
    "types_covered": sorted(all_types),
    "drake_valid_mean": statistics.mean([s["score"] for s in drake_valid]) if drake_valid else None,
    "drake_invalid_mean": statistics.mean([s["score"] for s in drake_invalid]) if drake_invalid else None,
}
with open("artifacts/v9_uncond_vlm_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
PYEOF

## Step C.4: Quick gallery render

Pick top-scoring, bottom-scoring, and random samples for visual inspection.

python3 - << 'PYEOF'
import json
import shutil
from pathlib import Path

scores = []
with open("artifacts/v9_uncond_vlm_scores.jsonl") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("vlm_score") is not None:
            scores.append(rec)

# Sort by score
scores_sorted = sorted(scores, key=lambda x: x["vlm_score"], reverse=True)

gallery_dir = Path("artifacts/v9_uncond_quick_gallery")
gallery_dir.mkdir(parents=True, exist_ok=True)

# Top 10
for i, rec in enumerate(scores_sorted[:10]):
    src = Path(rec["png"])
    dst = gallery_dir / f"top_{i:02d}_score{rec['vlm_score']:.0f}.png"
    if src.exists():
        shutil.copy(src, dst)

# Bottom 10
for i, rec in enumerate(scores_sorted[-10:]):
    src = Path(rec["png"])
    dst = gallery_dir / f"bot_{i:02d}_score{rec['vlm_score']:.0f}.png"
    if src.exists():
        shutil.copy(src, dst)

# 10 random (middle)
middle = scores_sorted[len(scores_sorted)//2-5:len(scores_sorted)//2+5]
for i, rec in enumerate(middle):
    src = Path(rec["png"])
    dst = gallery_dir / f"mid_{i:02d}_score{rec['vlm_score']:.0f}.png"
    if src.exists():
        shutil.copy(src, dst)

print(f"Gallery: {len(list(gallery_dir.glob('*.png')))} images at {gallery_dir}")
PYEOF

# Combine to grid PNG
python3 - << 'PYEOF'
from PIL import Image
from pathlib import Path

gallery_dir = Path("artifacts/v9_uncond_quick_gallery")
images = sorted(gallery_dir.glob("*.png"))

if not images:
    print("No images for grid")
else:
    # 6x5 grid (30 images)
    cols, rows = 6, 5
    sample = Image.open(images[0])
    w, h = sample.size
    
    grid = Image.new("RGB", (w * cols, h * rows), "white")
    for i, img_path in enumerate(images[:cols*rows]):
        img = Image.open(img_path)
        col = i % cols
        row = i // cols
        grid.paste(img, (col * w, row * h))
    
    grid.save("artifacts/v9_uncond_quick_gallery.png")
    print(f"Grid saved: {grid.size}")
PYEOF

## Step C.5: Decision document

python3 - << 'PYEOF'
import json
from datetime import datetime

with open("artifacts/v9_uncond_vlm_summary.json") as f:
    vlm = json.load(f)
with open("artifacts/v9_uncond_phase_b_results.json") as f:
    drake = json.load(f)

drake_pct = drake.get("validity_rate", 0) * 100
vlm_mean = vlm["mean_score"]

doc = f"""# Phase C Decision: VLM Quality Check Result

Date: {datetime.now().strftime('%Y-%m-%d')}
Model: conditional_v9_uncond (49M params, 234k data, 200k steps)

## Phase B Probes

- Drake validity (uncond, EMA): {drake_pct:.1f}%
- Comparison to v7 original: 6.5% baseline
- Comparison to v7 cold-start retrain: 0.8% baseline

## Phase C VLM Scoring

- Scenes judged: {vlm['n_scored']}
- VLM model: Claude Haiku 4.5
- Mean score: {vlm_mean:.2f}
- Median score: {vlm['median_score']:.2f}
- Score distribution: {vlm['distribution']}
- Object types covered: {vlm['types_covered']} ({len(vlm['types_covered'])}/12)

## Decision Gate

Required for Phase D (conditional training):
- [ ] VLM mean > 2.0: {'PASS' if vlm_mean > 2.0 else 'FAIL'} ({vlm_mean:.2f})
- [ ] No peak at score=1: {'PASS' if vlm['distribution'].get('1', 0) < 0.5 * vlm['n_scored'] else 'FAIL'}
- [ ] Type diversity >= 10/12: {'PASS' if len(vlm['types_covered']) >= 10 else 'FAIL'}

"""

if vlm_mean > 2.0 and len(vlm["types_covered"]) >= 10:
    doc += "## Verdict: PROCEED TO PHASE D\n\n"
    doc += "All gates pass. v9 unconditional model produces visually plausible scenes "
    doc += "with adequate type diversity. Safe to invest in conditional training.\n"
else:
    doc += "## Verdict: HALT\n\n"
    doc += "One or more gates failed. Investigation needed before Phase D.\n\n"
    if vlm_mean <= 2.0:
        doc += f"- VLM mean score {vlm_mean:.2f} below threshold 2.0\n"
        doc += "  Possible causes: visual artifacts, type collapse, geometry wrong\n"
    if len(vlm["types_covered"]) < 10:
        doc += f"- Only {len(vlm['types_covered'])} object types appeared\n"
        doc += "  Likely cause: type collapse despite larger dataset\n"
        doc += "  Mitigation: add class-balanced sampler to Phase D config\n"

with open("artifacts/v9_phase_c_decision.md", "w") as f:
    f.write(doc)

print(doc)
PYEOF

cat artifacts/v9_phase_c_decision.md

## Step C.6: Commit Phase C results

cd /workspace/Demiurge

git add artifacts/v9_uncond_vlm_scores.jsonl
git add artifacts/v9_uncond_vlm_summary.json
git add artifacts/v9_uncond_quick_gallery.png
git add artifacts/v9_phase_c_decision.md
git add artifacts/v9_uncond_vlm_subset/manifest.json
git commit -m "phase C: v9 uncond VLM quality check complete

VLM Mean: <X.XX>
Type diversity: <N>/12
Verdict: <PROCEED/HALT>"

git push origin week6/v9-phase-c 2>&1 || echo "push failed (no creds, expected)"

# Final Status

echo ""
echo "=================================================="
echo "PHASE C COMPLETE - VLM QUALITY CHECK"
echo "=================================================="
cat artifacts/v9_phase_c_decision.md | head -30
echo ""
echo "=================================================="

exit

# Constraints

- python3 -u for all commands
- Cache VLM responses (reuse on retry)
- DO NOT modify Phase B checkpoint
- DO NOT spend > $5 on VLM API for this phase

# Halt Conditions

- v9 uncond checkpoint missing from W&B: halt
- Generation produces < 50 scenes: investigate
- VLM API errors > 20% of calls: halt
- Total VLM spend > $5: halt

# Cost Estimate

- Pod time: ~2 hours × $0.74/hr = ~$1.50
- VLM API (100 scenes Haiku 4.5): ~$1-3
- Total: ~$3-5

# Phase C Decision Gate (the entire point of this phase)

Required to proceed to Phase D:

- VLM mean score > 2.0
- No catastrophic distribution issue (no peak at 1)
- Type diversity >= 10/12 types appear

If pass: proceed to Phase D (cond training, ~10 hours, ~$7).
If fail: this is the cheap stop. Diagnose:
- VLM < 2.0 but Drake OK: investigate visual quality issues
- Type diversity < 10/12: add class-balanced sampler to Phase D config
- Both fail: serious architectural issue, may need to retrain at 25M

Begin with Step 0.1.
