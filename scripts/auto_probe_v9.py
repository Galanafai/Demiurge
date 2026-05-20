"""
Auto-probe: wakes every 10k steps, runs 100-scene Drake validity probe,
logs results to logs/v9_probe_results.log and appends a TSV row.
Probe targets: 160k, 170k, 180k, 200k, 220k, 260k.
Run as background nohup.
"""
import sys, torch, time, os, math
sys.path.insert(0, "src")

PROBE_STEPS = [160000, 170000, 180000, 200000, 220000, 260000]
CKPT_DIR    = "checkpoints/v9_uncond_v9"
LOG_FILE    = "logs/v9_probe_results.log"
TSV_FILE    = "logs/v9_probe_results.tsv"
POLL_SECS   = 60   # check checkpoint step every 60s

_MEAN_XYZ   = torch.tensor([-0.049867, +0.378790, -0.751155])
_STD_XYZ    = torch.tensor([+0.419080, +0.457588, +0.127651])
_MEAN_SCALE = torch.tensor([-0.038706, -0.038706, -0.038706])
_STD_SCALE  = torch.tensor([+0.221619, +0.221619, +0.221619])
PRES_THRESH = -0.589

def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

def run_probe(step):
    from model.denoiser import DenoiserConfig, SceneDenoiser
    from model.schedule import CosineSchedule, DDIMSampler
    from model.rotations import rot6d_to_quat_wxyz
    from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
    from validator.core import SceneValidator
    import warnings; warnings.filterwarnings("ignore")

    DEV = torch.device("cuda")
    bounds = WorkspaceBounds.default()
    validator = SceneValidator(rrt_budget_s=2.0)

    ckpt = torch.load(f"{CKPT_DIR}/latest.pt", map_location=DEV, weights_only=False)
    arch = ckpt["arch"]
    cfg = DenoiserConfig(n_layers=arch["n_layers"], d_model=arch["d_model"],
        n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"],
        dropout=0.0, use_slot_id_embed=arch.get("use_slot_id_embed", True))
    model = SceneDenoiser(cfg).to(DEV)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()

    schedule = CosineSchedule(T=1000, zero_terminal_snr=True)
    sampler  = DDIMSampler(schedule, n_steps=50, prediction_type="v")

    N = 100
    with torch.no_grad():
        x0 = sampler.sample(model.noise_prediction_fn(None),
                            (N, N_MAX, 13), seed=step, device=DEV)
    x0 = x0.cpu()

    pres_mask = x0[:,:,12] > PRES_THRESH
    n_active  = pres_mask.sum().item()
    x_w       = x0[:,:,0][pres_mask]
    x_std     = x_w.std().item() if x_w.numel() > 1 else float("nan")

    accepted=0; interp=0; stable=0; ik=0; rrt=0; empty=0; n=0
    for i in range(N):
        r6d = x0[i,:,3:9].clone(); r6d[:,2]=0; r6d[:,5]=0
        xyz = (x0[i,:,:3]*_STD_XYZ + _MEAN_XYZ).clamp(-1,1)
        sc  = (x0[i,:,9:12]*_STD_SCALE + _MEAN_SCALE).clamp(-1,1)
        pres = x0[i,:,12] > PRES_THRESH
        if pres.sum() < 1: empty+=1; continue
        n += 1
        quats = rot6d_to_quat_wxyz(r6d)
        st = SceneTensor(object_types=torch.zeros(N_MAX,dtype=torch.long),
                         poses=torch.cat([xyz,quats],dim=-1),
                         scales=sc, presence=pres).denormalize(bounds)
        try:
            rpt = validator.validate(st, rrt_seed=i)
            if rpt.accepted: accepted+=1
            if rpt.no_interpenetration: interp+=1
            if rpt.stable_rest: stable+=1
            if rpt.ik_reachable: ik+=1
            if rpt.rrt_solvable: rrt+=1
        except: pass

    # log results
    pct = lambda a,b: f"{100*a/max(b,1):.1f}%"
    log(f"PROBE step={step} n={n} empty={empty} x_std={x_std:.3f}")
    log(f"  accepted={accepted}/{n} ({pct(accepted,n)})  interp={interp}/{n} ({pct(interp,n)})")
    log(f"  stable={stable}/{n} ({pct(stable,n)})  ik={ik}/{n} ({pct(ik,n)})  rrt={rrt}/{n} ({pct(rrt,n)})")

    # TSV row
    with open(TSV_FILE, "a") as f:
        f.write(f"{step}\t{n}\t{empty}\t{x_std:.4f}\t{accepted}\t{interp}\t{stable}\t{ik}\t{rrt}\n")

    del model, validator
    torch.cuda.empty_cache()
    return accepted, n

# Write TSV header if new
if not os.path.exists(TSV_FILE):
    with open(TSV_FILE, "w") as f:
        f.write("step\tn\tempty\tx_std\taccepted\tinterp\tstable\tik\trrt\n")

log("=== auto_probe_v9 started ===")
log(f"Probe schedule: {PROBE_STEPS}")
log(f"v7 baseline: accepted=2/180 (1.1%)  stable=9/180 (5.0%)  ik=131/180 (72.8%)")

completed = set()

while PROBE_STEPS:
    next_target = min(s for s in PROBE_STEPS if s not in completed)

    # Poll until latest.pt step >= target
    while True:
        try:
            ckpt = torch.load(f"{CKPT_DIR}/latest.pt", map_location="cpu",
                              weights_only=False)
            current_step = int(ckpt["step"])
            del ckpt
        except Exception as e:
            log(f"  checkpoint read failed: {e}")
            time.sleep(POLL_SECS)
            continue

        if current_step >= next_target:
            break

        remaining = next_target - current_step
        log(f"  waiting for step {next_target} (current={current_step}, remaining={remaining})")
        time.sleep(POLL_SECS)

    log(f"=== Running probe at step {next_target} ===")
    try:
        accepted, n = run_probe(next_target)
        log(f"=== Probe complete: {accepted}/{n} accepted ===")
    except Exception as e:
        log(f"ERROR during probe: {e}")

    completed.add(next_target)
    PROBE_STEPS = [s for s in PROBE_STEPS if s not in completed]

log("=== All probes complete ===")
