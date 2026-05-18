# Antigravity Master Prompt — Phase D: v9 Conditional Training (49M + Text)

You are continuing Phase D of the v9 build. Phase B trained v9 unconditional
to convergence. Phase C verified VLM quality > 2.0.

Phase D: Add text conditioning with CFG dropout. Warm-init from v9 uncond.
Target: Drake cond validity > 3% (better than v7's 1.4% baseline).

# Pre-Flight

## Step 0.1: Branch + verify Phases B and C

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

cd /workspace/Demiurge

git checkout main
git checkout -b week6/v9-phase-d

# Verify Phase B uncond checkpoint
ls -lh checkpoints/conditional_v9_uncond/latest.pt

# Verify Phase C decision
cat artifacts/v9_phase_c_decision.md | head -30

# Confirm v9 uncond on W&B
python3 -c "
import wandb
api = wandb.Api()
art = api.artifact('galanafai-self/demiurge/conditional_v9_uncond_checkpoint:latest')
print(f'v9 uncond ready: {art.size / 1024 / 1024:.1f} MB')
"

## Step 0.2: GPU and disk

nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
df -h /workspace

Need: GPU memory >= 14 GB free, disk >= 10 GB free.

# Before Doing Anything Else

1. Read AGENTS.md, SKILL.md.
2. Read artifacts/v9_phase_c_decision.md (Phase C result).
3. Read configs/train/v9_uncond.yaml (architecture reference).
4. Read configs/train/conditional_v7.yaml (CFG dropout pattern reference).
5. State in Plan Artifact:
   - Architecture: same as Phase B (d_model=384, 16 layers, 49M)
   - Warm-init: from v9 uncond
   - CFG dropout: 0.15
   - LR: 1e-4 (lower than Phase B since warm-init)
   - Steps: 200,000
   - Phase D success: Drake cond > 3% AT best CFG

# Phase D Execution

## Step D.1: Conditional config

cat > /workspace/Demiurge/configs/train/v9_cond.yaml << 'YAMLEOF'
# v9 conditional training config
# 49M params, 234k data, warm-init from v9 uncond, CFG dropout

model:
  d_model: 384
  n_layers: 16
  n_heads: 8
  ffn_multiplier: 4
  n_max: 12
  n_types: 12
  use_cross_attention: true
  text_emb_dim: 384
  type_grad_isolation: true

diffusion:
  schedule: cosine
  n_train_steps: 1000
  n_inference_steps: 50

training:
  max_steps: 200000
  batch_size: 64
  lr: 1e-4
  warmup_steps: 2000
  schedule: cosine_decay
  warm_init_from: checkpoints/conditional_v9_uncond/latest.pt
  warm_init_keys: [model_state]
  cfg_dropout: 0.15
  val_every_steps: 5000
  val_n_scenes: 100
  num_workers: 8
  grad_clip_norm: 1.0

loss:
  pose_xyz: 1.0
  pose_rot: 1.0
  scale: 1.0
  type_ce: 0.1
  presence_bce: 0.5

ema:
  decay: 0.9999
  enabled: true

data:
  source: data/combined/
  manifest: shard_manifest.json
  split_indices: train_indices.json
  text_embeddings: data/combined/text_embeddings.pt

logging:
  project: demiurge
  entity: galanafai-self
  run_name: conditional_v9_cond
  log_every: 100
  checkpoint_every: 10000
  checkpoint_dir: checkpoints/conditional_v9_cond

drake_probe:
  enabled: true
  every_steps: 5000
  n_scenes: 100
  head: ema
  drake_workers: 4
  cfg_scale: 1.0
  text_mode: full-cond
YAMLEOF

## Step D.2: Build auto-upload script

cat > /workspace/Demiurge/scripts/upload_v9_cond.py << 'PYEOF'
#!/usr/bin/env python3
"""Upload v9 conditional checkpoint to W&B with verification."""
import wandb
import os
import sys
import time

CHECKPOINT = "/workspace/Demiurge/checkpoints/conditional_v9_cond/latest.pt"
ARTIFACT_NAME = "conditional_v9_cond_checkpoint"

print(f"Auto-upload v9 cond")
time.sleep(60)

if not os.path.exists(CHECKPOINT):
    print(f"FAIL: {CHECKPOINT} not found")
    sys.exit(1)

size_mb = os.path.getsize(CHECKPOINT) / 1024 / 1024
print(f"Checkpoint: {size_mb:.1f} MB")

run = wandb.init(
    project="demiurge",
    entity="galanafai-self",
    job_type="checkpoint_upload",
    name=f"v9_cond_upload_{int(time.time())}",
)

artifact = wandb.Artifact(
    ARTIFACT_NAME,
    type="model",
    description=f"v9 conditional: 49M params, warm-init from v9 uncond, CFG dropout 0.15. Trained {time.strftime('%Y-%m-%d')}.",
    metadata={
        "params": "49M",
        "warm_init": "v9_uncond",
        "cfg_dropout": 0.15,
        "max_steps": 200000,
    },
)
artifact.add_file(CHECKPOINT, name="checkpoint.pt")
run.log_artifact(artifact)
run.finish()

time.sleep(60)

api = wandb.Api()
try:
    check_art = api.artifact(f"galanafai-self/demiurge/{ARTIFACT_NAME}:latest")
    print(f"VERIFIED: {check_art.size / 1024 / 1024:.1f} MB")
    print("UPLOAD SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
    sys.exit(1)
PYEOF
chmod +x /workspace/Demiurge/scripts/upload_v9_cond.py

## Step D.3: Smoke test conditional model

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import torch
import yaml
from model.denoiser import SceneDenoiser

with open("/workspace/Demiurge/configs/train/v9_cond.yaml") as f:
    cfg = yaml.safe_load(f)

model = SceneDenoiser(
    d_model=cfg["model"]["d_model"],
    n_layers=cfg["model"]["n_layers"],
    n_heads=cfg["model"]["n_heads"],
    ffn_multiplier=cfg["model"]["ffn_multiplier"],
    n_max=cfg["model"]["n_max"],
    n_types=cfg["model"]["n_types"],
    use_cross_attention=True,
    text_emb_dim=384,
).cuda()

# Verify warm-init works
ckpt = torch.load("checkpoints/conditional_v9_uncond/latest.pt", map_location="cuda")
state = ckpt.get("model_state", ckpt)
# Warm-init may drop cross-attention layers (new in cond)
loaded, missing = model.load_state_dict(state, strict=False)
print(f"Loaded keys: {len(loaded) if hasattr(loaded, '__len__') else 'OK'}")
print(f"Missing keys (expected: cross_attn): {len(missing) if hasattr(missing, '__len__') else 0}")

# Test forward
x = torch.randn(2, 12, 13).cuda()
t = torch.randint(0, 1000, (2,)).cuda()
text_emb = torch.randn(2, 384).cuda()

with torch.no_grad():
    out_cond = model(x, t, text_emb=text_emb)
    out_uncond = model(x, t, text_emb=None)
    print(f"Cond output: {out_cond.shape}, no NaN: {not torch.isnan(out_cond).any()}")
    print(f"Uncond output: {out_uncond.shape}, no NaN: {not torch.isnan(out_uncond).any()}")
    # Verify outputs differ (cross-attention has effect)
    diff = (out_cond - out_uncond).abs().mean().item()
    print(f"Cond vs uncond diff: {diff:.4f}")
    if diff < 1e-6:
        print("WARNING: Cond and uncond outputs identical - cross-attention not connected")
    else:
        print("PASS: Cross-attention has effect")
PYEOF

## Step D.4: Launch conditional training

tmux kill-server 2>/dev/null

tmux new-session -d -s train_v9_cond "
cd /workspace/Demiurge
export WANDB_API_KEY=\$WANDB_API_KEY
PYTHONUNBUFFERED=1 \
PYTHONPATH=/workspace/Demiurge/src \
python3 -u scripts/train.py \
  --config configs/train/v9_cond.yaml \
  --seed 42 \
  2>&1 | tee logs/train_v9_cond.log
"

echo "Waiting 120s for training to start..."
sleep 120

echo "=== Active sessions ==="
tmux list-sessions

echo ""
echo "=== Training startup ==="
tail -50 logs/train_v9_cond.log

echo ""
echo "=== GPU state ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

Pass criteria:
- Training started
- No errors in first 50 steps
- GPU utilization > 70%
- W&B run started

## Step D.5: Launch auto-upload monitor

tmux new-session -d -s upload_monitor "
cd /workspace/Demiurge
export WANDB_API_KEY=\$WANDB_API_KEY

while tmux has-session -t train_v9_cond 2>/dev/null; do
  sleep 600
  echo \"[\$(date)] training still running\" >> logs/upload_v9_cond_monitor.log
done

echo \"[\$(date)] training ended, uploading\" >> logs/upload_v9_cond_monitor.log

python3 /workspace/Demiurge/scripts/upload_v9_cond.py 2>&1 | tee -a logs/upload_v9_cond.log

if [ \$? -eq 0 ]; then
  echo \"AUTO-UPLOAD SUCCESS\" >> logs/upload_v9_cond_monitor.log
else
  echo \"AUTO-UPLOAD FAILED\" >> logs/upload_v9_cond_monitor.log
fi
"

## Step D.6: Stop here, training runs autonomously

echo ""
echo "=================================================="
echo "PHASE D TRAINING LAUNCHED AUTONOMOUSLY"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Model: 49M params (warm-init from v9 uncond)"
echo "  CFG dropout: 0.15"
echo "  Steps: 200,000"
echo "  LR: 1e-4 (lower since warm-init)"
echo ""
echo "Expected:"
echo "  Wall-clock: ~10-12 hours at ~5-6 steps/sec"
echo "  Cost: ~\$8 pod time"
echo ""
echo "Drake probes during training:"
echo "  - Every 5k steps at cfg=1.0"
echo "  - 100 scenes per probe"
echo "  - Logged to W&B"
echo ""
echo "Auto-upload:"
echo "  - upload_monitor watches train_v9_cond"
echo "  - Uploads to W&B as conditional_v9_cond_checkpoint"
echo ""
echo "Do not pause pod until AUTO-UPLOAD SUCCESS."
echo "=================================================="

exit

# Post-Training Steps (after training completes)

## Step D.7: Comprehensive final probe

ssh ... 'cd /workspace/Demiurge && python3 -u scripts/cfg_sweep.py \
  --config configs/train/v9_cond.yaml \
  --checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --cfg-scales 0.0 0.5 1.0 1.5 2.0 3.0 5.0 \
  --n-scenes 200 \
  --head ema \
  --seed 42 \
  --drake-workers 24 \
  --out artifacts/v9_cfg_sweep.json'

## Step D.8: Per-prompt type accuracy

ssh ... 'cd /workspace/Demiurge && python3 -u scripts/probe_type_accuracy.py \
  --checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --n-prompts 200 \
  --n-samples 8 \
  --best-cfg 1.0 \
  --out artifacts/v9_type_accuracy.json'

## Step D.9: Eps correlation diagnostic

ssh ... 'cd /workspace/Demiurge && python3 -u scripts/eps_correlation.py \
  --checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --n-samples 100 \
  --out artifacts/v9_eps_corr.json'

## Step D.10: VLM scoring at best CFG

ssh ... 'cd /workspace/Demiurge && python3 -u scripts/generate_and_judge.py \
  --checkpoint checkpoints/conditional_v9_cond/latest.pt \
  --cfg-scale <best_from_cfg_sweep> \
  --n-scenes 100 \
  --vlm-model claude-haiku-4-5 \
  --out artifacts/v9_cond_vlm_scores.jsonl'

# Phase D Decision Gate

| Metric | v7 baseline | v9 target | Action if fail |
|---|---|---|---|
| Drake uncond (cfg=0) | 6.5% | > 5% | Halt if < 3% |
| Drake cond (best CFG) | 1.4% | > 3% | Continue but flag |
| Per-prompt type acc | 99.9% | > 95% | Investigate |
| Eps correlation | 0.994 | > 0.95 | Halt if < 0.9 |
| VLM mean | n/a | > 2.0 | Flag in writeup |

# Phase D Results Document

After all probes complete:

python3 - << 'PYEOF'
import json

cfg = json.load(open("artifacts/v9_cfg_sweep.json"))
type_acc = json.load(open("artifacts/v9_type_accuracy.json"))
eps = json.load(open("artifacts/v9_eps_corr.json"))

doc = """# Phase D Results: v9 Conditional Training

## CFG Sweep
"""
for scale, result in sorted(cfg.items()):
    if isinstance(result, dict):
        v = result.get("validity_rate", 0) * 100
        doc += f"  cfg={scale}: {v:.1f}%\n"

best_cfg = max(((s, r.get("validity_rate", 0)) for s, r in cfg.items() if isinstance(r, dict)), key=lambda x: x[1])
doc += f"\nBest CFG: {best_cfg[0]} = {best_cfg[1]*100:.1f}%\n"

doc += f"\n## Type Accuracy\n  Per-prompt type accuracy: {type_acc.get('accuracy', 0)*100:.1f}%\n"

doc += f"\n## Eps Correlation\n  Cond vs uncond: {eps.get('correlation', 0):.3f}\n"

doc += f"\n## Phase D Verdict\n"
best_v = best_cfg[1] * 100
if best_v > 3.0:
    doc += f"PROCEED TO PHASE E: cond validity {best_v:.1f}% > 3% target\n"
else:
    doc += f"REVIEW: cond validity {best_v:.1f}% below 3% target\n"

with open("artifacts/v9_cond_phase_d_results.md", "w") as f:
    f.write(doc)

print(doc)
PYEOF

# Constraints

- python3 -u for training
- DO NOT pause pod during training or before AUTO-UPLOAD SUCCESS
- DO NOT modify v9 uncond checkpoint

# Halt Conditions

- v9 uncond checkpoint missing: halt
- Smoke test shows identical cond/uncond outputs: halt
- Drake validity at step 100k < 1% conditional: halt
- Training loss NaN/Inf: halt
- Memory OOM: halt

# Cost Estimate

- Pod time: 10-12 hours × $0.74/hr = ~$7-9
- VLM API: $1-2 (final scoring)
- Total: ~$9-11

Begin with Step 0.1.
