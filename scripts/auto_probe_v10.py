"""
v10 autonomous probe: polls for checkpoints at 20k, 50k, 80k, 110k, 130k, 150k.
Runs probe_drake.py at each milestone. Writes TSV + JSON artifacts.
Auto-uploads best checkpoint to W&B when all probes done.
"""
import subprocess, os, sys, time, json, glob, torch

CKPT_DIR  = "checkpoints/v9_uncond_v10"
CFG       = "configs/train/v9_uncond_v10.yaml"
LOG       = "logs/v10_probe_results.log"
TSV       = "logs/v10_probe_results.tsv"
ART_DIR   = "artifacts"
TARGETS   = [20000, 50000, 80000, 110000, 130000, 150000]
N_SCENES  = {20000:100, 50000:100, 80000:100, 110000:200, 130000:200, 150000:300}
POLL_SECS = 60

os.makedirs(ART_DIR, exist_ok=True)

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f: f.write(line + "\n")

def current_step():
    latest = f"{CKPT_DIR}/latest.pt"
    if not os.path.exists(latest): return 0
    try:
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        return int(ck.get("step", 0))
    except: return 0

def probe(step, n):
    ckpt = f"{CKPT_DIR}/step_{step:08d}.pt"
    out  = f"{ART_DIR}/v10_probe_{step}.json"
    if not os.path.exists(ckpt):
        log(f"  checkpoint {ckpt} not found, skipping")
        return None
    log(f"  Running probe: n={n} seed=42")
    r = subprocess.run([
        sys.executable, "scripts/probe_drake.py",
        "--config", CFG, "--checkpoint", ckpt,
        "--n-scenes", str(n), "--head", "ema", "--seed", "42",
        "--text-mode", "none", "--presence-threshold", "-0.589",
        "--drake-workers", "6", "--out", out,
    ], capture_output=True, text=True)
    if r.returncode != 0:
        log(f"  PROBE FAILED:\n{r.stderr[-500:]}")
        return None
    try:
        with open(out) as f: d = json.load(f)
        acc = d.get("n_accepted", 0)
        total = d.get("n_scenes", n)
        ne  = d.get("n_non_empty", total)
        v   = d.get("validity_rate", 0) * 100
        vc  = (acc/ne*100) if ne > 0 else 0
        rej = d.get("rejection_reasons", {})
        log(f"  accepted={acc}/{total} ({v:.1f}%) | non-empty={ne} | cond={vc:.1f}%")
        for reason, cnt in sorted(rej.items(), key=lambda x: -x[1]):
            log(f"    {reason}: {cnt} ({100*cnt/total:.1f}%)")
        with open(TSV, "a") as f:
            f.write(f"{step}\t{total}\t{ne}\t{acc}\t{v:.2f}\t{vc:.2f}\n")
        return v
    except Exception as e:
        log(f"  Parse error: {e}")
        return None

# TSV header
if not os.path.exists(TSV):
    with open(TSV, "w") as f:
        f.write("step\tn\tnon_empty\taccepted\tvalidity_pct\tcond_pct\n")

log("=== v10 auto-probe started ===")
log(f"Targets: {TARGETS}")
log(f"v9 reference: 3.0% overall, 11.3% conditional (step 170k)")

done = set()
best_step, best_v = 0, 0.0

for target in TARGETS:
    log(f"Waiting for step {target}...")
    while True:
        cs = current_step()
        if cs >= target: break
        log(f"  current={cs}, remaining={target-cs}")
        time.sleep(POLL_SECS)

    log(f"=== PROBE @ step {target} ===")
    time.sleep(30)  # let checkpoint finalize
    v = probe(target, N_SCENES[target])
    if v is not None and v > best_v:
        best_v, best_step = v, target
    done.add(target)

# Final summary
log("\n=== v10 ALL PROBES DONE ===")
results = {}
for step in TARGETS:
    p = f"{ART_DIR}/v10_probe_{step}.json"
    if os.path.exists(p):
        with open(p) as f: d = json.load(f)
        results[step] = {
            "validity": d.get("validity_rate",0)*100,
            "n_non_empty": d.get("n_non_empty", 0),
            "n_scenes": d.get("n_scenes", 0),
        }

log("\n  Step  |  Overall  | Non-empty")
log("  ------|-----------|----------")
for step, r in sorted(results.items()):
    log(f"  {step:>6}|  {r['validity']:>6.1f}%  | {r['n_non_empty']}/{r['n_scenes']}")

gate = results.get(150000, {}).get("validity", 0)
if gate >= 8.0:   log(f"\nSTRONG PASS ({gate:.1f}%): beats random baseline (9%)")
elif gate >= 5.0: log(f"\nPASS ({gate:.1f}%): above Phase B gate (5%)")
elif gate >= 3.0: log(f"\nMARGINAL ({gate:.1f}%): matches v9 step 170k")
else:             log(f"\nBELOW GATE ({gate:.1f}%): v9 step 170k remains best")

log(f"\nBest checkpoint: step {best_step} ({best_v:.1f}%)")

# Auto-upload best checkpoint
log("\n=== AUTO-UPLOAD to W&B ===")
best_ckpt = f"{CKPT_DIR}/step_{best_step:08d}.pt"
if not os.path.exists(best_ckpt):
    best_ckpt = sorted(glob.glob(f"{CKPT_DIR}/step_*.pt"))[-1] if glob.glob(f"{CKPT_DIR}/step_*.pt") else None

if best_ckpt:
    import wandb
    run = wandb.init(project="demiurge", entity="galanafai-self",
                     job_type="model_release", name=f"v10_FINAL_step{best_step}")
    art = wandb.Artifact("v10_final_model", type="model",
        description=f"v10 fresh training -- all audit fixes. Best step={best_step} ({best_v:.1f}% overall).",
        metadata={"step": best_step, "validity_overall": best_v, "fresh_training": True})
    art.add_file(best_ckpt, name="checkpoint.pt")
    for step in TARGETS:
        p = f"{ART_DIR}/v10_probe_{step}.json"
        if os.path.exists(p): art.add_file(p, name=f"probe_{step}.json")
    run.log_artifact(art)
    run.finish()
    log(f"Uploaded v10_final_model (step={best_step}) to W&B")
else:
    log("No checkpoint found for upload")

open(f"{ART_DIR}/v10_auto_probes.DONE", "w").close()
log("=== v10 probe pipeline complete ===")
