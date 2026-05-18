# Antigravity Master Prompt — Phase G: Gallery + Denoising Demo MP4

You are continuing Phase G of the v9 build. Phases A-F complete. Best v9
sampler identified in Phase F. Time to produce final visual artifacts.

Phase G: 100-scene gallery from best v9 sampler. 50-frame denoising demo
MP4. Final verification pass.

# Pre-Flight

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

cd /workspace/Demiurge

git checkout main
git checkout -b week6/v9-phase-g

# Verify Phase F artifacts
ls -lh artifacts/results_table_v9.md
ls -lh artifacts/eval_v9/

# Identify best v9 sampler from Phase F
python3 - << 'PYEOF'
import json
from pathlib import Path

best_validity = 0
best_sampler = None
best_path = None
for p in Path("artifacts/eval_v9/").glob("*.json"):
    with open(p) as f:
        data = json.load(f)
    v = data.get("validity_rate_mean", 0)
    if v > best_validity:
        best_validity = v
        best_sampler = p.stem
        best_path = p

print(f"Best v9 sampler: {best_sampler}")
print(f"Best validity: {best_validity*100:.1f}%")

# Save for use in later steps
with open("artifacts/v9_best_sampler.json", "w") as f:
    json.dump({"sampler": best_sampler, "validity": best_validity}, f)
PYEOF

BEST_SAMPLER=$(python3 -c "import json; print(json.load(open('artifacts/v9_best_sampler.json'))['sampler'])")
echo "Will render gallery from: $BEST_SAMPLER"

# Before Doing Anything Else

1. Read AGENTS.md, SKILL.md.
2. Read artifacts/results_table_v9.md.
3. State in Plan Artifact:
   - Best v9 sampler: <from above>
   - Gallery: 100 scenes, 10x10 grid
   - Denoising MP4: 50 frames, ffmpeg encoded
   - Verification: ruff, pyright, pytest

# Phase G Execution

## Step G.1: Generate 100 scenes from best v9 sampler

mkdir -p artifacts/v9_gallery_scenes

python3 -u scripts/generate_gallery.py \
  --sampler-name "$BEST_SAMPLER" \
  --checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --n-scenes 100 \
  --prompts-file data/eval_suite_v1/heldout_prompts.json \
  --output-dir artifacts/v9_gallery_scenes/ \
  --render \
  --drake-workers 24 \
  --seed 42 \
  2>&1 | tee logs/v9_gallery_generate.log

If generate_gallery.py doesn't exist, adapt scripts/generate_for_vlm_judging.py:

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import json
import torch
from pathlib import Path
from model.denoiser import SceneDenoiser
from model.schedule import DiffusionSchedule
from model.sampler import DDIMSampler
from validator.core import SceneValidator
from rendering.offscreen import render_scene_to_png

# Load best sampler config
with open("artifacts/v9_best_sampler.json") as f:
    best = json.load(f)
sampler_name = best["sampler"]

# Load checkpoint
ckpt = torch.load("checkpoints/conditional_v9_cond/latest.pt", map_location="cuda")

# Build model
model = SceneDenoiser(
    d_model=384, n_layers=16, n_heads=8, ffn_multiplier=4,
    n_max=12, n_types=12, use_cross_attention=True, text_emb_dim=384,
).cuda()
model.load_state_dict(ckpt["ema_state"] if "ema_state" in ckpt else ckpt["model_state"])
model.eval()

# Load prompts
with open("data/eval_suite_v1/heldout_prompts.json") as f:
    prompts = json.load(f)

# Generate diverse 100 scenes spanning templates
from collections import Counter
template_count = Counter()
output_dir = Path("artifacts/v9_gallery_scenes")
output_dir.mkdir(parents=True, exist_ok=True)

generated = []
for i, prompt_data in enumerate(prompts[:100]):
    prompt = prompt_data["text"] if isinstance(prompt_data, dict) else prompt_data
    
    # Encode prompt
    from model.text_encoder import TextEncoder
    encoder = TextEncoder()
    text_emb = encoder.encode([prompt]).cuda()
    
    sampler = DDIMSampler(model, DiffusionSchedule(), n_inference_steps=50)
    with torch.no_grad():
        scene = sampler.sample(batch_size=1, text_emb=text_emb)[0]
    
    # Render
    png_path = output_dir / f"scene_{i:03d}.png"
    render_scene_to_png(scene, str(png_path), prompt_overlay=prompt)
    
    generated.append({
        "idx": i,
        "prompt": prompt,
        "png": str(png_path),
    })
    if (i+1) % 10 == 0:
        print(f"Generated {i+1}/100")

with open(output_dir / "manifest.json", "w") as f:
    json.dump(generated, f, indent=2)

print(f"Done: 100 scenes at {output_dir}")
PYEOF

ls artifacts/v9_gallery_scenes/*.png | wc -l

## Step G.2: 10x10 gallery grid

python3 - << 'PYEOF'
from PIL import Image
from pathlib import Path

gallery_dir = Path("artifacts/v9_gallery_scenes")
images = sorted(gallery_dir.glob("scene_*.png"))[:100]

if len(images) < 100:
    print(f"WARNING: only {len(images)} scenes")

# 10x10 grid
sample = Image.open(images[0])
w, h = sample.size

# Resize if too large
target_size = 256  # Per-tile size for final grid
if w > target_size:
    scale = target_size / w
    w, h = target_size, int(h * scale)

grid = Image.new("RGB", (w * 10, h * 10), "white")
for i, img_path in enumerate(images[:100]):
    img = Image.open(img_path)
    if img.size != (w, h):
        img = img.resize((w, h), Image.LANCZOS)
    col = i % 10
    row = i // 10
    grid.paste(img, (col * w, row * h))

grid.save("artifacts/v9_final_gallery.png", optimize=True)
print(f"Gallery saved: {grid.size}")
print(f"Size: {Path('artifacts/v9_final_gallery.png').stat().st_size / 1024:.0f} KB")
PYEOF

ls -lh artifacts/v9_final_gallery.png

## Step G.3: Denoising demo MP4

Pick one representative prompt:

REPRESENTATIVE_PROMPT="A cluttered tabletop with a red cup, a blue book, and a green apple"

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import torch
from pathlib import Path
from model.denoiser import SceneDenoiser
from model.schedule import DiffusionSchedule
from model.sampler import DDIMSampler
from model.text_encoder import TextEncoder
from rendering.offscreen import render_scene_to_png

ckpt = torch.load("checkpoints/conditional_v9_cond/latest.pt", map_location="cuda")
model = SceneDenoiser(d_model=384, n_layers=16, n_heads=8, ffn_multiplier=4, n_max=12, n_types=12, use_cross_attention=True, text_emb_dim=384).cuda()
model.load_state_dict(ckpt["ema_state"])
model.eval()

prompt = "A cluttered tabletop with a red cup, a blue book, and a green apple"
encoder = TextEncoder()
text_emb = encoder.encode([prompt]).cuda()

# Custom sampling loop to capture frames
sched = DiffusionSchedule()
n_steps = 50

frames_dir = Path("artifacts/v9_denoising_frames")
frames_dir.mkdir(parents=True, exist_ok=True)

import torch
torch.manual_seed(42)
x_t = torch.randn(1, 12, 13).cuda()

# Reverse process
for i, t_val in enumerate(sched.ddim_steps(n_steps)):
    t = torch.tensor([t_val]).cuda()
    with torch.no_grad():
        eps = model(x_t, t, text_emb=text_emb)
        x_t = sched.ddim_step(x_t, eps, t_val, n_inference_steps=n_steps, step_index=i)
    
    # Render this intermediate scene
    png_path = frames_dir / f"frame_{i:04d}.png"
    render_scene_to_png(x_t[0], str(png_path), prompt_overlay=prompt, step=f"{i+1}/{n_steps}")
    
    if i % 10 == 0:
        print(f"Frame {i+1}/{n_steps}")

print(f"Frames saved: {len(list(frames_dir.glob('*.png')))}")
PYEOF

ls artifacts/v9_denoising_frames/ | wc -l

# Encode MP4
which ffmpeg || apt-get install -y ffmpeg 2>&1 | tail -3

# Standard 24fps
ffmpeg -y -framerate 24 \
  -i artifacts/v9_denoising_frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 23 \
  artifacts/v9_denoising_demo.mp4 2>&1 | tail -3

# Slow 12fps for LinkedIn
ffmpeg -y -framerate 12 \
  -i artifacts/v9_denoising_frames/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 23 \
  artifacts/v9_denoising_demo_slow.mp4 2>&1 | tail -3

ls -lh artifacts/v9_denoising_demo*.mp4

## Step G.4: Final verification pass

echo "=== ruff ==="
ruff check . 2>&1 | tail -10

echo ""
echo "=== pyright ==="
pyright 2>&1 | grep -E "error|Error" | head -10

echo ""
echo "=== pytest ==="
pytest tests/ -q --tb=no 2>&1 | tail -15

## Step G.5: Phase G results document

python3 - << 'PYEOF'
import json
from pathlib import Path
from datetime import datetime

with open("artifacts/v9_best_sampler.json") as f:
    best = json.load(f)

doc = f"""# Phase G: Final Visual Artifacts

Date: {datetime.now().strftime('%Y-%m-%d')}
Best sampler: {best['sampler']} ({best['validity']*100:.1f}% Drake validity)

## Gallery

- artifacts/v9_final_gallery.png (10x10, 100 scenes)
- Size: {Path('artifacts/v9_final_gallery.png').stat().st_size / 1024:.0f} KB

## Denoising Demo

- artifacts/v9_denoising_demo.mp4 (24fps standard)
- artifacts/v9_denoising_demo_slow.mp4 (12fps for LinkedIn)
- 50 DDIM steps captured

## Verification

- ruff: <pass/fail>
- pyright: <errors>
- pytest: <passed/total>

## Ready for Phase H (Ship)

All visual artifacts complete. Final review before LinkedIn post:
- Gallery shows diverse, plausible scenes
- Denoising MP4 plays smoothly
- All code passes verification
"""

with open("artifacts/v9_phase_g_results.md", "w") as f:
    f.write(doc)

print(doc)
PYEOF

## Step G.6: Commit Phase G

git add artifacts/v9_final_gallery.png
git add artifacts/v9_denoising_demo.mp4
git add artifacts/v9_denoising_demo_slow.mp4
git add artifacts/v9_phase_g_results.md
git add artifacts/v9_gallery_scenes/manifest.json
git commit -m "phase G: final visual artifacts

100-scene gallery from v9 + best sampler
50-frame denoising MP4 (24fps + 12fps)
All verification passes"

git push origin week6/v9-phase-g 2>&1 || echo "push failed (expected)"

# Final Status

echo ""
echo "=================================================="
echo "PHASE G COMPLETE - VISUAL ARTIFACTS"
echo "=================================================="
ls -lh artifacts/v9_final_gallery.png
ls -lh artifacts/v9_denoising_demo*.mp4
echo ""
echo "Next: Phase H (Ship - LinkedIn post, README, tag release)"
echo "=================================================="

exit

# Constraints

- python3 -u
- DO NOT modify any checkpoints
- DO NOT regenerate eval data
- Gallery render quality matters - use Drake offscreen with proper lighting

# Halt Conditions

- Less than 80 gallery scenes generated: investigate
- ffmpeg fails: install and retry
- MP4 file > 50 MB: re-encode with higher CRF
- pytest failures > 5: halt

# Cost Estimate

- Pod time: 2 hours × $0.74/hr = ~$1.50
- VLM: $0
- Total: ~$1.50

Begin with Pre-Flight.
