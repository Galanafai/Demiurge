#!/bin/bash
cd /workspace/Demiurge
source .venv/bin/activate
WANDB_KEY=$(grep -A 1 "machine api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
[ -z "$WANDB_KEY" ] && WANDB_KEY=$(grep "password" ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')
export WANDB_API_KEY="$WANDB_KEY"
LOG=/workspace/Demiurge/logs/auto_probe_phase_d.log
CKPT_DIR="checkpoints/run"

echo "[$(date)] Phase D probe scheduler (re)started" >> $LOG
echo "Schedule: 30k[kill], 50k, 60k, 70k, 80k, 90k, 100k, 110k, 120k, 130k, 140k, 150k, 160k, 170k, 180k, 190k, 200k" >> $LOG
echo "CKPT_DIR=$CKPT_DIR" >> $LOG

probe_step() {
    local STEP=$1; local N=$2; local LABEL=$3; local KILL_ON_COLLAPSE=$4
    local CKPT="$CKPT_DIR/step_$(printf '%08d' $STEP).pt"
    local OUT="artifacts/v9_phase_d_probe_${STEP}.json"
    mkdir -p artifacts
    echo "" >> $LOG
    echo "[$(date)] === Probing step $STEP ($LABEL) ===" | tee -a $LOG
    if [ ! -f "$CKPT" ]; then
        echo "  SKIP: checkpoint not found at $CKPT" | tee -a $LOG
        return
    fi
    PYTHONPATH=/workspace/Demiurge/src python3 -u scripts/probe_drake.py \
        --config configs/train/v9_cond_d.yaml \
        --checkpoint "$CKPT" --n-scenes $N --head ema --seed 42 \
        --presence-threshold -0.589 --drake-workers 6 --out "$OUT" 2>&1 | tee -a $LOG
    python3 - >> $LOG << PYEOF
import json, subprocess, sys
try:
    with open("$OUT") as f: d = json.load(f)
    n=d.get("n_scenes",$N); acc=d.get("n_accepted",0); ne=d.get("n_non_empty",0)
    v_pct=(acc/n)*100; e_pct=((n-ne)/n*100)
    print(f"  RESULT step=$STEP: validity={v_pct:.1f}% empty={e_pct:.1f}% (n={n}) [ref uncond: 3.0%]")
    if "$KILL_ON_COLLAPSE"=="yes" and e_pct>80:
        print(f"  !! COLLAPSE DETECTED: {e_pct:.1f}% empty > 80% -- killing training")
        subprocess.run(["pkill","-f","train.py"])
        open("artifacts/v9_phase_d_HALTED.txt","w").write(f"step=$STEP\nempty={e_pct:.1f}%\n")
except Exception as e:
    print(f"  parse error: {e}")
PYEOF
}

# -- Probe state flags (0=not yet done) --
declare -A DONE
for S in 30000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000; do
    DONE[$S]=0
    # Mark already-probed steps as done (resume support)
    [ -f "artifacts/v9_phase_d_probe_${S}.json" ] && DONE[$S]=1 && echo "[$(date)] Step $S already probed, skipping" >> $LOG
done

# -- N per step: 100 early, 200 mid, 300 final --
n_for_step() {
    local S=$1
    [ $S -ge 150000 ] && echo 300 || ( [ $S -ge 80000 ] && echo 200 || echo 100 )
}

while true; do
    [ -f artifacts/v9_phase_d_HALTED.txt ] && { echo "[$(date)] HALTED flag found, exiting" >> $LOG; break; }

    ALL_DONE=1
    for S in 30000 50000 60000 70000 80000 90000 100000 110000 120000 130000 140000 150000 160000 170000 180000 190000 200000; do
        if [ ${DONE[$S]} -eq 0 ] && [ -f "$CKPT_DIR/step_$(printf '%08d' $S).pt" ]; then
            N=$(n_for_step $S)
            KILL="no"
            [ $S -eq 30000 ] && KILL="yes"
            probe_step $S $N "step${S}" $KILL
            DONE[$S]=1
        fi
        [ ${DONE[$S]} -eq 0 ] && ALL_DONE=0
    done

    [ $ALL_DONE -eq 1 ] && break
    sleep 90
done

# Final trajectory summary
python3 - >> $LOG << 'PYEOF'
import json, glob
rows = []
for p in sorted(glob.glob("artifacts/v9_phase_d_probe_*.json")):
    step = int(p.split("_")[-1].replace(".json",""))
    try:
        d = json.load(open(p))
        n = d.get("n_scenes", 0)
        acc = d.get("n_accepted", 0)
        ne = d.get("n_non_empty", 0)
        rows.append((step, acc/n*100, (n-ne)/n*100))
    except:
        pass

print("\n=== PHASE D FULL TRAJECTORY ===")
print(f"{'Step':>7} | {'Validity':>8} | {'Empty':>7} | {'Bar':}")
print("-" * 50)
for step, v, e in rows:
    bar = "#" * int(v * 3)
    print(f"{step:>7} | {v:>7.1f}% | {e:>6.1f}% | {bar}")
PYEOF
touch artifacts/v9_phase_d_probes.DONE
echo "[$(date)] Probe scheduler complete" >> $LOG
