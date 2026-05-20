#!/bin/bash
cd /workspace/Demiurge
source .venv/bin/activate
WANDB_KEY=$(grep -A 1 "machine api.wandb.ai" ~/.netrc 2>/dev/null | grep password | awk '{print $2}')
[ -z "$WANDB_KEY" ] && WANDB_KEY=$(grep "password" ~/.netrc 2>/dev/null | head -1 | awk '{print $2}')
export WANDB_API_KEY="$WANDB_KEY"
LOG=/workspace/Demiurge/logs/auto_probe_phase_d.log

echo "[$(date)] Phase D probe scheduler started" > $LOG
echo "Schedule: 10k(collapse check), 50k, 100k, 150k, 200k" >> $LOG

probe_step() {
    local STEP=$1; local N=$2; local LABEL=$3; local KILL_ON_COLLAPSE=$4
    local CKPT="checkpoints/v9_cond_d/step_$(printf '%08d' $STEP).pt"
    local OUT="artifacts/v9_phase_d_probe_${STEP}.json"
    echo "[$(date)] === Probing step $STEP ($LABEL) ===" | tee -a $LOG
    PYTHONPATH=/workspace/Demiurge/src python3 -u scripts/probe_drake.py \
        --config configs/train/v9_cond_d.yaml \
        --checkpoint "$CKPT" --n-scenes $N --head ema --seed 42 \
        --presence-threshold -0.589 --drake-workers 6 --out "$OUT" 2>&1 | tee -a $LOG
    python3 - >> $LOG << PYEOF
import json, subprocess
try:
    with open("$OUT") as f: d = json.load(f)
    n=d.get("n_scenes",$N); acc=d.get("n_accepted",0); ne=d.get("n_non_empty",0)
    v_pct=(acc/n)*100; e_pct=((n-ne)/n*100)
    print(f"  Step $STEP: validity={v_pct:.1f}% empty={e_pct:.1f}% (ref: v9 uncond 3.0% overall)")
    if "$KILL_ON_COLLAPSE"=="yes" and e_pct>80:
        print(f"  KILLING: presence collapse ({e_pct:.1f}% empty > 80% threshold)")
        subprocess.run(["pkill","-f","train.py"])
        open("artifacts/v9_phase_d_HALTED.txt","w").write(f"Halted step $STEP\nempty={e_pct:.1f}%\n")
except Exception as e: print(f"  parse error: {e}")
PYEOF
    echo "" >> $LOG
}

PROBED_10K=0; PROBED_50K=0; PROBED_100K=0; PROBED_150K=0; PROBED_200K=0
while true; do
    [ -f artifacts/v9_phase_d_HALTED.txt ] && { echo "[$(date)] HALTED" >> $LOG; break; }
    [ $PROBED_10K -eq 0 ] && [ -f "checkpoints/v9_cond_d/step_00010000.pt" ] && { probe_step 10000 100 "COLLAPSE_CHECK" "yes"; PROBED_10K=1; }
    [ $PROBED_50K -eq 0 ] && [ -f "checkpoints/v9_cond_d/step_00050000.pt" ] && { probe_step 50000 100 "MID" "no"; PROBED_50K=1; }
    [ $PROBED_100K -eq 0 ] && [ -f "checkpoints/v9_cond_d/step_00100000.pt" ] && { probe_step 100000 200 "PRE-FINAL" "no"; PROBED_100K=1; }
    [ $PROBED_150K -eq 0 ] && [ -f "checkpoints/v9_cond_d/step_00150000.pt" ] && { probe_step 150000 200 "LATE" "no"; PROBED_150K=1; }
    if [ $PROBED_200K -eq 0 ] && [ -f "checkpoints/v9_cond_d/step_00200000.pt" ]; then
        probe_step 200000 300 "FINAL" "no"; PROBED_200K=1; break
    fi
    sleep 90
done

python3 - >> $LOG << 'PYEOF'
import json, glob
print("\n=== PHASE D TRAJECTORY ===")
print(f"{'Step':>6} | {'Validity':>8} | {'Non-empty':>10}")
print("-" * 40)
for p in sorted(glob.glob("artifacts/v9_phase_d_probe_*.json")):
    step=int(p.split("_")[-1].replace(".json",""))
    try:
        d=json.load(open(p))
        n=d.get("n_scenes",0); acc=d.get("n_accepted",0); ne=d.get("n_non_empty",0)
        print(f"{step:>6} | {(acc/n*100):>7.1f}% | {ne:>5}/{n}")
    except: pass
PYEOF
touch artifacts/v9_phase_d_probes.DONE
echo "[$(date)] Probe scheduler complete" >> $LOG
