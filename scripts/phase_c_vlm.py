#!/usr/bin/env python3
"""Phase C: Generate scenes from v9 step 170k, render, VLM judge, analyze."""
import sys, os, json, hashlib, base64, random, time, statistics
from pathlib import Path
from collections import Counter
import torch

sys.path.insert(0, "src")
os.environ.setdefault("ANTHROPIC_API_KEY", "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")

from model.denoiser import DenoiserConfig, SceneDenoiser
from model.schedule import CosineSchedule, DDIMSampler
from model.rotations import rot6d_to_quat_wxyz
from scene.schema import N_MAX, SceneTensor, WorkspaceBounds
from validator.core import SceneValidator

CKPT = "checkpoints/v9_uncond_v9/step_00170000.pt"
SCENES_DIR = Path("artifacts/v9_phase_c_scenes")
RENDERS_DIR = Path("artifacts/v9_phase_c_renders")
VLM_DIR = Path("artifacts/v9_phase_c_vlm_cohort")
CACHE_DIR = Path(".vlm_cache_phase_c")
for d in [SCENES_DIR, RENDERS_DIR, VLM_DIR, CACHE_DIR, Path("src/eval/prompts"), Path("logs")]:
    d.mkdir(parents=True, exist_ok=True)

_MEAN_XYZ  = torch.tensor([-0.049867, +0.378790, -0.751155])
_STD_XYZ   = torch.tensor([+0.419080, +0.457588, +0.127651])
_MEAN_SC   = torch.tensor([-0.038706, -0.038706, -0.038706])
_STD_SC    = torch.tensor([+0.221619, +0.221619, +0.221619])

TYPE_NAMES = {0:"cube",1:"cylinder",2:"mug",3:"bowl",4:"bottle",5:"can",
              6:"box",7:"sphere",8:"cone",9:"tray",10:"plate",11:"cup"}

def yaw_project(r6d):
    o = r6d.clone(); o[...,2]=0; o[...,5]=0; return o

def log(msg): print(msg, flush=True)

# ── C.1 GENERATE ──────────────────────────────────────────────────────────────
log("=== C.1: Loading model ===")
DEV = torch.device("cuda")
bounds = WorkspaceBounds.default()

ckpt = torch.load(CKPT, map_location=DEV, weights_only=False)
arch = ckpt["arch"]
cfg = DenoiserConfig(n_layers=arch["n_layers"], d_model=arch["d_model"],
    n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"], dropout=0.0,
    use_slot_id_embed=arch.get("use_slot_id_embed", True))
model = SceneDenoiser(cfg).to(DEV)
model.load_state_dict(ckpt["ema_state"])
model.eval()
log(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params")

schedule = CosineSchedule(T=1000, zero_terminal_snr=True)
sampler = DDIMSampler(schedule, n_steps=50, prediction_type="v")
validator = SceneValidator(rrt_budget_s=2.0)

manifest_path = SCENES_DIR / "manifest.json"
if manifest_path.exists():
    log("Manifest already exists, skipping generation.")
    with open(manifest_path) as f: records = json.load(f)
else:
    N_ATTEMPTS, BATCH = 1500, 100
    records = []
    log(f"Generating {N_ATTEMPTS} scenes in batches of {BATCH}...")
    for bi in range(N_ATTEMPTS // BATCH):
        seed = 42 + bi
        log(f"  batch {bi+1}/{N_ATTEMPTS//BATCH} seed={seed}")
        with torch.no_grad():
            x0, tids = sampler.sample_with_types(
                model.conditional_sampling_fn(text_emb=None),
                (BATCH, N_MAX, 13), seed=seed, device=DEV)
        x0 = x0.cpu(); tids = tids.cpu()
        for i in range(BATCH):
            gi = bi * BATCH + i
            pres = x0[i,:,12] > -0.589
            if pres.sum() < 1: continue
            xyz = (x0[i,:,:3] * _STD_XYZ + _MEAN_XYZ).clamp(-1,1)
            r6d = yaw_project(x0[i,:,3:9])
            sc  = (x0[i,:,9:12] * _STD_SC + _MEAN_SC).clamp(-1,1)
            quats = rot6d_to_quat_wxyz(r6d)
            poses = torch.cat([xyz, quats], -1)
            st_n = SceneTensor(object_types=tids[i], poses=poses, scales=sc, presence=pres)
            st_p = st_n.denormalize(bounds)
            try:
                rpt = validator.validate(st_p, rrt_seed=gi)
                acc,iok,sok,ikok,rok = rpt.accepted,rpt.no_interpenetration,rpt.stable_rest,rpt.ik_reachable,rpt.rrt_solvable
            except: acc=iok=sok=ikok=rok=False
            rec = {"idx":gi,"seed":seed,"n_objects":int(pres.sum()),
                   "types":[int(tids[i,j]) for j in range(N_MAX) if pres[j]],
                   "drake_accepted":acc,"interp_ok":iok,"stable_ok":sok,"ik_ok":ikok,"rrt_ok":rok,
                   "poses_phys":st_p.poses[pres].tolist(),"scales_phys":st_p.scales[pres].tolist()}
            records.append(rec)
            torch.save({"object_types":tids[i],"poses":st_p.poses,"scales":st_p.scales,"presence":pres},
                       SCENES_DIR/f"scene_{gi:05d}.pt")
    with open(manifest_path,"w") as f: json.dump(records,f,indent=2)
    log(f"Generated: {len(records)} non-empty, {sum(r['drake_accepted'] for r in records)} Drake-accepted")

del model, sampler, schedule  # free GPU
torch.cuda.empty_cache()

# ── C.2 RENDER (matplotlib top-down view) ─────────────────────────────────────
log("\n=== C.2: Rendering scenes ===")
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    HAS_PLT = True
except ImportError:
    HAS_PLT = False; log("matplotlib missing -- skipping renders")

if HAS_PLT:
    for rec in records:
        gi = rec["idx"]
        png = RENDERS_DIR / f"scene_{gi:05d}.png"
        if png.exists(): continue
        pt = SCENES_DIR / f"scene_{gi:05d}.pt"
        if not pt.exists(): continue
        d = torch.load(pt, map_location="cpu", weights_only=False)
        pres = d["presence"]
        poses = d["poses"][pres].numpy()
        types = d["object_types"][pres].numpy()
        scales = d["scales"][pres].numpy()

        fig, ax = plt.subplots(1,1,figsize=(4,4))
        ax.set_xlim(-0.6,0.6); ax.set_ylim(-0.2,1.0)
        ax.set_aspect("equal"); ax.set_facecolor("#f5f0e8")
        # table surface
        table = patches.FancyBboxPatch((-0.5,-0.1),1.0,1.0,
            boxstyle="round,pad=0.02",linewidth=2,edgecolor="#8B6914",facecolor="#DEB887",alpha=0.3)
        ax.add_patch(table)
        COLORS = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6","#1abc9c",
                  "#e67e22","#e91e63","#00bcd4","#8bc34a","#ff5722","#607d8b"]
        for j,(pos,sc,ty) in enumerate(zip(poses,scales,types)):
            x,y = float(pos[0]),float(pos[1])
            r = max(float(abs(sc[0])),0.02)
            col = COLORS[int(ty)%len(COLORS)]
            c = plt.Circle((x,y),r*0.15+0.03,color=col,alpha=0.85,zorder=5)
            ax.add_patch(c)
            ax.text(x,y,TYPE_NAMES.get(int(ty),str(ty))[:3],ha="center",va="center",
                    fontsize=5,color="white",fontweight="bold",zorder=6)
        title_col = "#2ecc71" if rec["drake_accepted"] else "#e74c3c"
        verdict = "VALID" if rec["drake_accepted"] else ("IK-fail" if not rec["ik_ok"] else "unstable")
        ax.set_title(f"scene {gi:05d} | n={rec['n_objects']} | {verdict}",
                     fontsize=8,color=title_col,pad=4)
        ax.set_xlabel("x (m)",fontsize=6); ax.set_ylabel("y (m)",fontsize=6)
        ax.tick_params(labelsize=5)
        plt.tight_layout(pad=0.5)
        plt.savefig(png,dpi=80,bbox_inches="tight")
        plt.close()
    n_rendered = len(list(RENDERS_DIR.glob("*.png")))
    log(f"Rendered {n_rendered}/{len(records)} scenes")

# ── C.3 SELECT COHORT ─────────────────────────────────────────────────────────
log("\n=== C.3: Selecting VLM cohort ===")
valid   = [r for r in records if r["drake_accepted"]]
invalid = [r for r in records if not r["drake_accepted"]]
almost  = [r for r in records if r["interp_ok"] and not r["drake_accepted"]]
random.seed(42)
inv_sample    = random.sample(invalid, min(len(valid), len(invalid)))
almost_sample = random.sample(almost, min(20, len(almost)))
cohort = [{**r,"cohort":"drake_valid"} for r in valid] + \
         [{**r,"cohort":"drake_invalid"} for r in inv_sample] + \
         [{**r,"cohort":"interp_only"} for r in almost_sample]
log(f"  drake_valid: {len(valid)} | drake_invalid: {len(inv_sample)} | interp_only: {len(almost_sample)}")
log(f"  total to judge: {len(cohort)}")

import shutil
for rec in cohort:
    src = RENDERS_DIR / f"scene_{rec['idx']:05d}.png"
    dst = VLM_DIR / f"scene_{rec['idx']:05d}.png"
    if src.exists(): shutil.copy(src, dst)
with open(VLM_DIR/"cohort.json","w") as f: json.dump(cohort,f,indent=2)

# ── C.4 JUDGE PROMPT ──────────────────────────────────────────────────────────
PROMPT_PATH = Path("src/eval/prompts/scene_judge.txt")
if not PROMPT_PATH.exists():
    PROMPT_PATH.write_text("""You are evaluating a rendered top-down 2D diagram of a tabletop scene.
Each colored circle represents an object (labeled with a 3-letter abbreviation).
The beige rectangle is the table surface. The robot workspace is approx -0.5 to 0.5 m in x, 0 to 0.9 m in y.

Rate the scene 1-5:
1 = Broken: objects clearly off-table, all stacked at one point, or diagram unreadable
2 = Poor: objects heavily clustered or near workspace boundary
3 = Acceptable: objects spread across table, no obvious clustering
4 = Good: well-distributed objects, realistic tabletop density
5 = Excellent: diverse, well-spaced arrangement that looks like a real tabletop

Focus on: object spread, density, workspace coverage.
Ignore rendering style (this is a schematic, not photorealistic).

Respond EXACTLY in this format:
Score: <1-5>
Reasoning: <1-2 sentences>
Objects visible: <comma-separated list>
Issues: <specific problems or "none">""")
    log("Created judge prompt")
JUDGE_PROMPT = PROMPT_PATH.read_text()

# ── C.5 VLM JUDGING ───────────────────────────────────────────────────────────
log("\n=== C.5: VLM judging ===")
import anthropic
client = anthropic.Anthropic()

results = []
total_cost = 0.0
BUDGET = 5.0

for i, rec in enumerate(cohort):
    png = VLM_DIR / f"scene_{rec['idx']:05d}.png"
    if not png.exists():
        results.append({**rec,"vlm_score":None,"vlm_text":"PNG missing"}); continue
    if total_cost >= BUDGET:
        log(f"Budget ${BUDGET} reached at {i}/{len(cohort)}"); break

    raw = png.read_bytes()
    key = hashlib.sha256(raw + JUDGE_PROMPT.encode()).hexdigest()
    cache_file = CACHE_DIR / f"{key}.json"

    if cache_file.exists():
        rdata = json.loads(cache_file.read_text())
    else:
        b64 = base64.standard_b64encode(raw).decode()
        try:
            msg = client.messages.create(
                model="claude-haiku-4-5", max_tokens=300,
                messages=[{"role":"user","content":[
                    {"type":"image","source":{"type":"base64","media_type":"image/png","data":b64}},
                    {"type":"text","text":JUDGE_PROMPT}]}])
            rdata = {"text":msg.content[0].text,
                     "in":msg.usage.input_tokens,"out":msg.usage.output_tokens}
            cache_file.write_text(json.dumps(rdata))
            cost = rdata["in"]*1e-6 + rdata["out"]*5e-6
            total_cost += cost
            if i % 5 == 0: log(f"  [{i}/{len(cohort)}] ${total_cost:.3f} total")
        except Exception as e:
            log(f"  API error scene {rec['idx']}: {e}")
            results.append({**rec,"vlm_score":None,"vlm_text":f"ERR:{e}"}); continue

    text = rdata["text"]
    score = None
    for line in text.split("\n"):
        if line.strip().lower().startswith("score:"):
            try: score = float(line.split(":",1)[1].strip()); break
            except: pass
    results.append({**rec,"vlm_score":score,"vlm_text":text})

log(f"Judged {len(results)} scenes, total cost ${total_cost:.3f}")

out_path = Path("artifacts/v9_phase_c_vlm_scores.jsonl")
out_path.write_text("\n".join(json.dumps(r) for r in results) + "\n")

# ── C.6 ANALYSIS ──────────────────────────────────────────────────────────────
log("\n=== C.6: Analysis ===")
scored = [r for r in results if r.get("vlm_score") is not None]
all_sc = [r["vlm_score"] for r in scored]
dv = [r for r in scored if r.get("cohort")=="drake_valid"]
di = [r for r in scored if r.get("cohort")=="drake_invalid"]
io = [r for r in scored if r.get("cohort")=="interp_only"]

def st(xs, label):
    if not xs: return
    log(f"{label} (n={len(xs)}): mean={statistics.mean(xs):.2f} median={statistics.median(xs):.2f} "
        f"min={min(xs):.0f} max={max(xs):.0f} dist={dict(Counter(int(s) for s in xs))}")

st(all_sc, "OVERALL")
st([r["vlm_score"] for r in dv], "DRAKE-VALID")
st([r["vlm_score"] for r in di], "DRAKE-INVALID")

all_types = set()
for r in scored: all_types.update(r.get("types",[]))

summary = {
    "n_total":len(results),"n_scored":len(scored),"n_drake_valid_scored":len(dv),
    "overall_mean":statistics.mean(all_sc) if all_sc else None,
    "overall_median":statistics.median(all_sc) if all_sc else None,
    "drake_valid_mean":statistics.mean([r["vlm_score"] for r in dv]) if dv else None,
    "drake_valid_median":statistics.median([r["vlm_score"] for r in dv]) if dv else None,
    "drake_invalid_mean":statistics.mean([r["vlm_score"] for r in di]) if di else None,
    "types_covered":sorted(all_types),"n_types":len(all_types),
}
Path("artifacts/v9_phase_c_vlm_summary.json").write_text(json.dumps(summary,indent=2))

# ── C.7 GALLERY ───────────────────────────────────────────────────────────────
if HAS_PLT:
    gdir = Path("artifacts/v9_phase_c_gallery"); gdir.mkdir(exist_ok=True)
    for i,r in enumerate(sorted(dv,key=lambda x:x["vlm_score"],reverse=True)[:10]):
        src=RENDERS_DIR/f"scene_{r['idx']:05d}.png"
        if src.exists(): shutil.copy(src,gdir/f"dv_top_{i:02d}_s{int(r['vlm_score'])}.png")
    for i,r in enumerate(sorted(dv,key=lambda x:x["vlm_score"])[:5]):
        src=RENDERS_DIR/f"scene_{r['idx']:05d}.png"
        if src.exists(): shutil.copy(src,gdir/f"dv_worst_{i:02d}_s{int(r['vlm_score'])}.png")
    imgs = sorted(gdir.glob("*.png"))
    if len(imgs) >= 9:
        import numpy as np; from PIL import Image
        cols,rows=5,3; sample=Image.open(imgs[0]); w,h=sample.size
        grid=Image.new("RGB",(w*cols,h*rows),"white")
        for ii,p in enumerate(imgs[:cols*rows]):
            img=Image.open(p); grid.paste(img,((ii%cols)*w,(ii//cols)*h))
        grid.save("artifacts/v9_phase_c_gallery.png")
        log("Gallery grid saved")

# ── C.8 DECISION DOC ──────────────────────────────────────────────────────────
log("\n=== C.8: Decision document ===")
from datetime import datetime
dv_mean = summary.get("drake_valid_mean") or 0
di_mean = summary.get("drake_invalid_mean") or 0
n_types = summary["n_types"]
dv_count = summary["n_drake_valid_scored"]
dv_scores = [r["vlm_score"] for r in dv]
pct_ge3 = (sum(1 for s in dv_scores if s >= 3) / len(dv_scores) * 100) if dv_scores else 0

doc = f"""# Phase C Decision: VLM Quality Check on v9 Step 170k

Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Model: v9_uncond_v9/step_00170000.pt (49M params)

## Phase B Reference
- Drake validity (non-empty): 11.3%
- Drake validity (overall): 3.0%
- v7 baseline: 1.1% -- 10x improvement

## Phase C VLM Scoring
- Generation attempts: 1500
- Non-empty scenes: {len(records)}
- Drake-accepted: {sum(r['drake_accepted'] for r in records)}
- Scenes judged by VLM: {len(scored)}
- VLM model: claude-haiku-4-5
- Total cost: ${total_cost:.2f}

### Drake-Valid Scenes (priority cohort, n={dv_count})
- Mean VLM score: {dv_mean:.2f}
- Median VLM score: {summary.get('drake_valid_median',0):.2f}
- Pct scoring >= 3 (Acceptable): {pct_ge3:.0f}%

### Drake-Invalid Scenes (comparison)
- Mean VLM score: {di_mean:.2f}
- Delta: {dv_mean-di_mean:+.2f} ({'valid>invalid' if dv_mean>di_mean else 'INVERTED'})

## Type Diversity
- Types seen: {summary['types_covered']}
- Count: {n_types}/12

## Decision Gates
- Drake-valid mean >= 2.5: {'PASS' if dv_mean>=2.5 else 'FAIL'} ({dv_mean:.2f})
- Pct >= 3 on Drake-valid >= 30%: {'PASS' if pct_ge3>=30 else 'FAIL'} ({pct_ge3:.0f}%)
- Type diversity >= 8/12: {'PASS' if n_types>=8 else 'FAIL'} ({n_types})
- Drake-valid mean > invalid mean: {'PASS' if dv_mean>di_mean else 'FAIL'}

## Verdict
"""
if dv_mean >= 2.5 and n_types >= 8 and dv_count >= 20:
    doc += "**PROCEED** -- v9 demonstrates visual quality on Drake-valid subset.\nShip v9 as final result. Phase D conditional training optional.\n"
elif dv_mean >= 2.0 and dv_count >= 20:
    doc += "**MARGINAL** -- quality acceptable but below stretch goal.\nShip v9 with caveats. Document limitations.\n"
else:
    doc += "**BELOW EXPECTATIONS** -- ship v9 as honest negative result with debugging narrative.\n"

Path("artifacts/v9_phase_c_decision.md").write_text(doc)
log(doc)
log("=== Phase C complete ===")
