#!/bin/bash
set -euo pipefail
cd /workspace/Demiurge
source .venv/bin/activate
export PYTHONPATH=/workspace/Demiurge/src
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "[$(date)] Starting v9 unconditional training (48.66M, 150k steps)"
echo "[$(date)] Dataset: data/combined (234k scenes)"

PYTHONUNBUFFERED=1 python3 -u scripts/train.py \
  --config configs/train/v9_uncond.yaml \
  --seed 42 \
  2>&1 | tee logs/train_v9_uncond.log

EXIT_CODE=${PIPESTATUS[0]}
echo "[$(date)] Training exited with code $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
  echo "[$(date)] Running auto-upload..."
  python3 scripts/upload_v9_uncond.py 2>&1 | tee logs/upload_v9_uncond.log
  if [ $? -eq 0 ]; then
    echo "[$(date)] AUTO-UPLOAD SUCCESS - pod safe to pause"
  else
    echo "[$(date)] AUTO-UPLOAD FAILED - check logs/upload_v9_uncond.log"
  fi
else
  echo "[$(date)] Training failed - skipping upload"
fi
