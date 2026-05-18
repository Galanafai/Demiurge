# Antigravity Master Prompt — Phase B: v9 Unconditional Training (49M)

You are continuing Phase B of the v9 build. Phase A combined v1+v2 into
234k Drake-validated scenes with locked train/val/heldout splits.

Phase B: Train 49M unconditional baseline. Establish the architecture
converges at this scale before committing to expensive conditional
training in Phase D.

# Pre-Flight

## Step 0.1: Branch and pod verification

ssh -tt -i ~/.ssh/id_ed25519 -o StrictHostKeyChecking=no -o LogLevel=ERROR -o ServerAliveInterval=30 f0xpg6tkynmbk9-64411a32@ssh.runpod.io 'bash --noprofile --norc'

In session:

cd /workspace/Demiurge

# Switch from Phase A branch to Phase B branch
git checkout main
git pull origin main
git checkout week6/v9-phase-a 2>/dev/null && git pull origin week6/v9-phase-a || echo "Phase A branch not local"
git checkout -b week6/v9-phase-b

# Verify Phase A artifacts
ls -lh data/combined/shard_manifest.json
ls -lh data/combined/split_manifest.json
ls -lh data/combined/text_embeddings.pt
ls -lh data/combined/train_indices.json

python3 -c "
import json
with open('data/combined/split_manifest.json') as f:
    s = json.load(f)
print(f'Train: {s[\"train_count\"]:,}')
print(f'Val: {s[\"val_count\"]:,}')
print(f'Heldout: {s[\"heldout_count\"]:,}')
"

## Step 0.2: GPU and disk check

nvidia-smi --query-gpu=name,memory.used,memory.total,memory.free --format=csv,noheader
free -h
df -h /workspace

echo "Need at least 12 GB free disk for checkpoints + W&B cache"

## Step 0.3: Verify W&B reachable

python3 -c "
import wandb
api = wandb.Api()
print('W&B API OK')
# Check v7 still available as comparison reference
try:
    art = api.artifact('galanafai-self/demiurge/conditional_v7_checkpoint:latest')
    print(f'v7 reference: {art.size / 1024 / 1024:.1f} MB')
except Exception as e:
    print(f'v7 missing: {e}')
"

Halt conditions:
- Phase A artifacts missing
- GPU memory < 14 GB free
- Free disk < 12 GB
- W&B unreachable

# Before Doing Anything Else

1. Read AGENTS.md and .agents/skills/demiurge/SKILL.md.
2. Read artifacts/dataset_combined_v1v2_card.md (Phase A output).
3. Read configs/train/conditional_v7.yaml (reference architecture).
4. Read src/model/denoiser.py (verify d_model knob exists).
5. State in Plan Artifact:
   - Architecture target: d_model=384, n_layers=16, n_heads=8, ~49M params
   - Training: 200k steps, batch 64, lr 3e-4, from scratch
   - Validation: 100 unconditional scenes every 5k steps
   - Phase B success: Drake validity > 5% uncond at step 200k
   - Phase B halt: validity < 3% at step 100k

# Phase B Execution

## Step B.1: Architecture config

Create configs/train/v9_uncond.yaml:

cat > /workspace/Demiurge/configs/train/v9_uncond.yaml << 'YAMLEOF'
# v9 unconditional training config
# 49M params, 234k data, from scratch

model:
  d_model: 384
  n_layers: 16
  n_heads: 8
  ffn_multiplier: 4
  n_max: 12
  n_types: 12
  use_cross_attention: false  # Unconditional
  type_grad_isolation: true  # From v6 pattern - critical

diffusion:
  schedule: cosine
  n_train_steps: 1000
  n_inference_steps: 50

training:
  max_steps: 200000
  batch_size: 64
  lr: 3e-4
  warmup_steps: 2000
  schedule: cosine_decay
  warm_init_from: null  # Train from scratch
  cfg_dropout: 0.0  # No text for uncond
  val_every_steps: 5000
  val_n_scenes: 100
  num_workers: 8
  grad_clip_norm: 1.0
  use_class_balanced_sampler: false  # 234k should be sufficient
  
loss:
  pose_xyz: 1.0
  pose_rot: 1.0
  scale: 1.0
  type_ce: 0.1  # Low; no class balancing
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
  run_name: conditional_v9_uncond
  log_every: 100
  checkpoint_every: 10000
  checkpoint_dir: checkpoints/conditional_v9_uncond

drake_probe:
  enabled: true
  every_steps: 5000
  n_scenes: 100
  head: ema
  drake_workers: 4
YAMLEOF

# Verify config
cat /workspace/Demiurge/configs/train/v9_uncond.yaml | head -30

# Sanity check param count BEFORE launching training
python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import yaml
from model.denoiser import SceneDenoiser

with open("/workspace/Demiurge/configs/train/v9_uncond.yaml") as f:
    cfg = yaml.safe_load(f)

model = SceneDenoiser(
    d_model=cfg["model"]["d_model"],
    n_layers=cfg["model"]["n_layers"],
    n_heads=cfg["model"]["n_heads"],
    ffn_multiplier=cfg["model"]["ffn_multiplier"],
    n_max=cfg["model"]["n_max"],
    n_types=cfg["model"]["n_types"],
    use_cross_attention=False,
)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")
print(f"Trainable parameters: {trainable_params:,}")

if total_params < 40_000_000 or total_params > 60_000_000:
    print(f"WARNING: Param count outside 40-60M target range")
else:
    print(f"PASS: Param count in target range (40-60M)")
PYEOF

## Step B.2: Build auto-upload script BEFORE training launches

cat > /workspace/Demiurge/scripts/upload_v9_uncond.py << 'PYEOF'
#!/usr/bin/env python3
"""Upload v9 unconditional checkpoint to W&B with verification."""
import wandb
import os
import sys
import time

CHECKPOINT = "/workspace/Demiurge/checkpoints/conditional_v9_uncond/latest.pt"
ARTIFACT_NAME = "conditional_v9_uncond_checkpoint"
MAX_SIZE_MB = 500

print(f"Auto-upload v9 unconditional checkpoint")
print(f"Source: {CHECKPOINT}")
print(f"Target: galanafai-self/demiurge/{ARTIFACT_NAME}")

# Wait for checkpoint to stabilize
print("Waiting 60s for checkpoint to finalize...")
time.sleep(60)

if not os.path.exists(CHECKPOINT):
    print(f"FAIL: {CHECKPOINT} not found")
    sys.exit(1)

size_mb = os.path.getsize(CHECKPOINT) / 1024 / 1024
print(f"Checkpoint size: {size_mb:.1f} MB")

if size_mb > MAX_SIZE_MB:
    print(f"WARNING: Checkpoint > {MAX_SIZE_MB} MB ({size_mb:.1f})")

# Initialize W&B
run = wandb.init(
    project="demiurge",
    entity="galanafai-self",
    job_type="checkpoint_upload",
    name=f"v9_uncond_upload_{int(time.time())}",
)

# Upload artifact
artifact = wandb.Artifact(
    ARTIFACT_NAME,
    type="model",
    description=f"v9 unconditional: 49M params, 234k data, 200k steps, from scratch. Trained {time.strftime('%Y-%m-%d')}.",
    metadata={
        "params": "49M",
        "data_scenes": 234000,
        "max_steps": 200000,
        "batch_size": 64,
        "lr": 3e-4,
        "from_scratch": True,
    },
)
artifact.add_file(CHECKPOINT, name="checkpoint.pt")
run.log_artifact(artifact)
run.finish()

# Wait then verify
print("Upload submitted. Waiting 60s for W&B commit...")
time.sleep(60)

print("Verifying via API...")
api = wandb.Api()
try:
    check_art = api.artifact(f"galanafai-self/demiurge/{ARTIFACT_NAME}:latest")
    print(f"VERIFIED on W&B:")
    print(f"  Name: {check_art.name}")
    print(f"  Size: {check_art.size / 1024 / 1024:.1f} MB")
    print(f"  Created: {check_art.created_at}")
    print(f"UPLOAD SUCCESS")
    sys.exit(0)
except Exception as e:
    print(f"VERIFICATION FAILED: {e}")
    sys.exit(1)
PYEOF
chmod +x /workspace/Demiurge/scripts/upload_v9_uncond.py

## Step B.3: Smoke test (5 minutes)

Run a tiny training run to verify config + data pipeline works:

python3 - << 'PYEOF'
import sys
sys.path.insert(0, "src")
import yaml
import torch
from model.denoiser import SceneDenoiser
from data.reader import ShardReader

# Load config
with open("/workspace/Demiurge/configs/train/v9_uncond.yaml") as f:
    cfg = yaml.safe_load(f)

# Build model
model = SceneDenoiser(
    d_model=cfg["model"]["d_model"],
    n_layers=cfg["model"]["n_layers"],
    n_heads=cfg["model"]["n_heads"],
    ffn_multiplier=cfg["model"]["ffn_multiplier"],
    n_max=cfg["model"]["n_max"],
    n_types=cfg["model"]["n_types"],
    use_cross_attention=False,
).cuda()

# Test forward pass with dummy batch
batch_size = 4
x = torch.randn(batch_size, 12, 13).cuda()
t = torch.randint(0, 1000, (batch_size,)).cuda()

print("Smoke testing forward pass...")
with torch.no_grad():
    out = model(x, t, text_emb=None)
    print(f"Output shape: {out.shape}")
    print(f"Output dtype: {out.dtype}")
    print(f"No NaN: {not torch.isnan(out).any()}")
    print(f"Memory used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("Smoke test PASS")
PYEOF

If smoke fails: halt before launching full training.

## Step B.4: Launch training in tmux

# Clean any leftover sessions
tmux kill-server 2>/dev/null

# Launch training
tmux new-session -d -s train_v9_uncond "
cd /workspace/Demiurge
export WANDB_API_KEY=\$WANDB_API_KEY
PYTHONUNBUFFERED=1 \
PYTHONPATH=/workspace/Demiurge/src \
python3 -u scripts/train.py \
  --config configs/train/v9_uncond.yaml \
  --seed 42 \
  2>&1 | tee logs/train_v9_uncond.log
"

# Wait 2 minutes for training startup
echo "Waiting 120s for training to start..."
sleep 120

# Verify launch
echo "=== Active sessions ==="
tmux list-sessions

echo ""
echo "=== Training startup log ==="
tail -50 logs/train_v9_uncond.log

echo ""
echo "=== GPU state ==="
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

echo ""
echo "=== W&B run started ==="
grep -E "wandb: Run.*started|wandb.ai/" logs/train_v9_uncond.log | head -3

Pass criteria:
- tmux session active
- Training log shows step counter ticking (>step 50)
- No errors or NaN in first 50 steps
- GPU utilization > 70%
- W&B run visible in log

Halt if any fail.

## Step B.5: Launch auto-upload monitor

tmux new-session -d -s upload_monitor "
cd /workspace/Demiurge
export WANDB_API_KEY=\$WANDB_API_KEY

echo 'Monitor started, waiting for train_v9_uncond to complete' > logs/upload_v9_uncond_monitor.log

while tmux has-session -t train_v9_uncond 2>/dev/null; do
  sleep 600
  echo \"[\$(date)] training still running\" >> logs/upload_v9_uncond_monitor.log
done

echo \"[\$(date)] training ended, running upload\" >> logs/upload_v9_uncond_monitor.log

# Run upload
python3 /workspace/Demiurge/scripts/upload_v9_uncond.py 2>&1 | tee -a logs/upload_v9_uncond.log

if [ \$? -eq 0 ]; then
  echo \"AUTO-UPLOAD SUCCESS - pod safe to pause\" >> logs/upload_v9_uncond_monitor.log
else
  echo \"AUTO-UPLOAD FAILED - check logs/upload_v9_uncond.log\" >> logs/upload_v9_uncond_monitor.log
fi
"

echo ""
echo "=== Upload monitor active ==="
tmux list-sessions

## Step B.6: Stop here, training runs autonomously

The training takes ~8 hours. Surface status and stop:

echo ""
echo "=================================================="
echo "PHASE B TRAINING LAUNCHED AUTONOMOUSLY"
echo "=================================================="
echo ""
echo "Configuration:"
echo "  Model: 49M params (d_model=384, 16 layers, 8 heads)"
echo "  Data: 234k scenes (train split)"
echo "  Steps: 200,000"
echo "  Batch: 64"
echo "  LR: 3e-4"
echo "  Init: from scratch"
echo ""
echo "Expected:"
echo "  Wall-clock: ~8 hours at ~7 steps/sec"
echo "  Cost: ~\$6 pod time"
echo "  Completion: \$(date -d '+8 hours' 2>/dev/null || echo 'late evening')"
echo ""
echo "Drake probes during training:"
echo "  - Every 5k steps (first 100k)"
echo "  - Every 10k steps (last 100k)"
echo "  - 100 scenes per probe"
echo "  - Logged to W&B"
echo ""
echo "Auto-upload on completion:"
echo "  - upload_monitor watches train_v9_uncond"
echo "  - Uploads to W&B as conditional_v9_uncond_checkpoint"
echo "  - Verifies via API call"
echo "  - Logs SUCCESS/FAILED to logs/upload_v9_uncond_monitor.log"
echo ""
echo "Phase B Decision Gate (after training completes):"
echo "  - Drake uncond validity > 5%"
echo "  - All 12 object types present"
echo "  - Loss converged (< 1.5)"
echo "  - If gate passes: proceed to Phase C (VLM check)"
echo "  - If gate fails: halt, consider class balancing"
echo ""
echo "How to check progress:"
echo "  ssh ... 'tail -3 /workspace/Demiurge/logs/train_v9_uncond.log'"
echo "  ssh ... 'cat /workspace/Demiurge/logs/upload_v9_uncond_monitor.log'"
echo ""
echo "Do not pause pod until AUTO-UPLOAD SUCCESS appears."
echo "=================================================="

exit

# Constraints

- python3 -u for training
- DO NOT pause pod during training
- DO NOT modify v7 checkpoint
- DO NOT delete data/combined/

# Halt Conditions (autonomous training)

Built into the training script via val_every_steps callbacks:

- Step 10k: if Drake validity == 0%, halt
- Step 50k: if Drake validity < 1%, halt and surface
- Step 100k: if Drake validity < 3%, halt (gate fail)
- Any step: if loss NaN/Inf, halt
- Any step: if memory OOM, halt

# Cost Estimate

- Pod time: 8 hours × $0.74/hr = ~$6
- W&B storage: $0 (free tier)
- VLM API: $0 (no scoring in Phase B)
- Total: ~$6

# Phase B Decision Gate (After Training)

After auto-upload completes, run final probe:

ssh ... 'cd /workspace/Demiurge && python3 -u scripts/probe_drake.py \
  --config configs/train/v9_uncond.yaml \
  --checkpoint checkpoints/conditional_v9_uncond/latest.pt \
  --n-scenes 500 \
  --head ema \
  --seed 42 \
  --text-mode none \
  --drake-workers 24 \
  --out artifacts/v9_uncond_phase_b_results.json'

Required for proceeding to Phase C:
- Drake validity > 5%
- All 12 object types present
- Mean pairwise diversity > 0.10

Begin with Step 0.1.
