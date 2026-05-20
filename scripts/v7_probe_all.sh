#!/usr/bin/env bash
# v7_probe_all.sh — sequential watcher for all v7 Drake probe checkpoints.
# Each milestone waits until its checkpoint file appears, then runs probe_drake.py.
# Milestones are processed in order; no two probes run concurrently (GPU safe).
# Usage: nohup bash scripts/v7_probe_all.sh > logs/v7_probe_all.log 2>&1 &

set -euo pipefail

CONFIG="configs/train/v9_uncond_v7.yaml"
CKPT_DIR="checkpoints/v9_uncond_v7"
MILESTONES=(30000 60000 100000 150000 200000)
N_SCENES=200
WORKERS=6
SEED=42

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "v7 probe watcher starting."
log "Milestones: ${MILESTONES[*]}"
log "Training: $(pgrep -af train.py 2>/dev/null | grep -v grep | head -1 || echo 'process not found')"
log "---"

for step in "${MILESTONES[@]}"; do
    printf -v label "%08d" "$step"
    ckpt="${CKPT_DIR}/step_${label}.pt"
    logfile="logs/v7_probe_${label}.log"

    log "Watching for ${ckpt} ..."
    while [[ ! -f "$ckpt" ]]; do
        sleep 60
    done

    log "Found step_${label}.pt  -- launching probe (n=${N_SCENES}, workers=${WORKERS})"

    python3 scripts/probe_drake.py \
        --config      "$CONFIG"   \
        --checkpoint  "$ckpt"     \
        --n-scenes    "$N_SCENES" \
        --drake-workers "$WORKERS" \
        --seed        "$SEED"     \
        2>&1 | tee "$logfile"

    log "Probe step_${label} done. Log: ${logfile}"
    log "---"
done

log "All v7 probes complete."
