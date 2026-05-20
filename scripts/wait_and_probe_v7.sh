#!/bin/bash
TARGET_STEP=30000
CKPT="checkpoints/v9_uncond_v7/step_$(printf '%08d' $TARGET_STEP).pt"
echo "Waiting for $CKPT ..."
while [ ! -f "$CKPT" ]; do sleep 60; done
echo "Checkpoint found. Running probe..."
python3 scripts/probe_drake.py \
    --config configs/train/v9_uncond_v7.yaml \
    --checkpoint "$CKPT" \
    --n-scenes 200 \
    --drake-workers 6 \
    --seed 42 \
    2>&1 | tee logs/v7_probe_030000.log
