#!/bin/bash
STEP=$1
CKPT="checkpoints/v9_uncond_v6/step_$(printf '%08d' $STEP).pt"
LOG="logs/v6_probe_${STEP}.log"
OUT="artifacts/v6_probe_${STEP}.json"
echo "[$(date)] Waiting for $CKPT..."
while [ ! -f "$CKPT" ]; do sleep 30; done
echo "[$(date)] Checkpoint found. Running Drake probe..."
python3 -u scripts/probe_drake.py \
    --config configs/train/v9_uncond_v6.yaml \
    --checkpoint "$CKPT" \
    --n-scenes 200 \
    --head ema --seed 42 \
    --text-mode none \
    --presence-threshold -0.589 \
    --drake-workers 6 \
    --out "$OUT" 2>&1 | tee "$LOG"
echo "[$(date)] Probe complete: $OUT"
