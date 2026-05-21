#!/bin/bash
cd /workspace/Demiurge
source .venv/bin/activate
LOG=/workspace/Demiurge/logs/auto_probe_v5.log

echo "[$(date)] v5 probe scheduler started, watching for checkpoints" > $LOG

probe_checkpoint() {
    local STEP=$1
    local N_SCENES=$2
    local LABEL=$3
    local CKPT="checkpoints/v9_uncond_v5/step_$(printf '%08d' $STEP).pt"
    local OUT="artifacts/v5_probe_${STEP}.json"

    echo "[$(date)] === $LABEL probe (step $STEP) ===" >> $LOG
    sleep 30

    python3 -u scripts/probe_drake.py \
        --config configs/train/v9_uncond_v5.yaml \
        --checkpoint $CKPT \
        --n-scenes $N_SCENES \
        --head ema --seed 42 \
        --text-mode none \
        --presence-threshold -0.589 \
        --drake-workers 6 \
        --out $OUT 2>&1 | tee -a $LOG

    python3 - >> $LOG << PYEOF
import json
try:
    with open("$OUT") as f:
        d = json.load(f)
    v = d.get("validity_rate", 0) * 100
    acc = d.get("n_accepted", 0)
    n = d.get("n_scenes", $N_SCENES)
    rej = d.get("rejection_reasons", {})
    print(f"\nStep $STEP result: {v:.1f}% ({acc}/{n})")
    print(f"Rejections:")
    for r, c in sorted(rej.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c} ({100*c/n:.1f}%)")
except Exception as e:
    print(f"Result parse error: {e}")
PYEOF
}

PROBED_30K=0
PROBED_60K=0
PROBED_100K=0

while [ $PROBED_30K -eq 0 ] || [ $PROBED_60K -eq 0 ] || [ $PROBED_100K -eq 0 ]; do

    if [ $PROBED_30K -eq 0 ] && [ -f "checkpoints/v9_uncond_v5/step_00030000.pt" ]; then
        probe_checkpoint 30000 100 "EARLY (v4 peak comparison)"
        PROBED_30K=1
    fi

    if [ $PROBED_60K -eq 0 ] && [ -f "checkpoints/v9_uncond_v5/step_00060000.pt" ]; then
        probe_checkpoint 60000 200 "CRITICAL (collapse check)"
        PROBED_60K=1
    fi

    if [ $PROBED_100K -eq 0 ] && [ -f "checkpoints/v9_uncond_v5/step_00100000.pt" ]; then
        probe_checkpoint 100000 300 "FINAL (Phase B gate)"
        PROBED_100K=1
    fi

    sleep 60
done

echo "" >> $LOG
echo "[$(date)] === ALL PROBES DONE ===" >> $LOG

python3 - >> $LOG << 'PYEOF'
import json

results = {}
for step in [30000, 60000, 100000]:
    try:
        with open(f"artifacts/v5_probe_{step}.json") as f:
            d = json.load(f)
        results[step] = d.get("validity_rate", 0) * 100
    except:
        results[step] = None

print(f"\n=== v5 TRAJECTORY ===")
for step, v in results.items():
    label = f"{v:.1f}%" if v is not None else "MISSING"
    print(f"  step {step:>6}: {label}")

print(f"\n=== REFERENCE POINTS ===")
print(f"  Random procedural baseline:  9.0%")
print(f"  Lost v7 (9M):                6.5%")
print(f"  v4 best (step 20k):          4.0%")

v100 = results.get(100000)
v60  = results.get(60000)
v30  = results.get(30000)

print(f"\n=== PHASE B VERDICT ===")
if v100 is None:
    print(f"  Training didn't complete or 100k probe missing")
elif v100 >= 8.0:
    print(f"  STRONG PASS ({v100:.1f}%): v5 beats v7 baseline")
    print(f"  Action: Proceed to Phase C (VLM check)")
elif v100 >= 5.0:
    print(f"  PASS ({v100:.1f}%): Above Phase B gate")
    print(f"  Action: Proceed to Phase C")
elif v100 >= 3.0:
    print(f"  MARGINAL ({v100:.1f}%): Real signal, below gate")
    print(f"  Action: Ship as best v9 result, decide on Phase D path")
else:
    print(f"  BELOW GATE ({v100:.1f}%): Phase B fail")
    print(f"  Action: Ship best checkpoint, write negative result")

if v30 is not None and v60 is not None:
    print(f"\n=== COLLAPSE CHECK ===")
    if v60 >= v30 * 0.8:
        print(f"  No collapse: v5 fix is working")
    elif v60 >= v30 * 0.5:
        print(f"  Mild drift: holding around peak")
    else:
        print(f"  Collapse pattern: same as v4 (4% -> 0%)")
PYEOF

touch /workspace/Demiurge/artifacts/v5_auto_probes.DONE
