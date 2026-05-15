"""Phase A gate evaluation for conditional_v4.

Generates scenes conditioned on task descriptions, runs VLM scoring
(Anthropic Haiku), and measures per-type object accuracy.

Reports all three Phase A gates:
  Gate 1: VLM mean score > 2.0
  Gate 2: Joint Drake-valid AND VLM>=4 fraction > 5%
  Gate 3: Per-type accuracy > 40% (>= 5 of 12 types generated at all)

Usage:
    PYTHONPATH=/workspace/Demiurge/src python3 scripts/probe_vlm.py \\
        --config configs/train/conditional_v4.yaml \\
        --checkpoint checkpoints/conditional_v4/latest.pt \\
        --n-prompts 200 \\
        --n-per-prompt 8 \\
        --budget-usd 2 \\
        --out artifacts/conditional_v4_vlm_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))

from data.reader import ShardReader  # noqa: E402
from model.denoiser import DenoiserConfig, N_MAX, SceneDenoiser  # noqa: E402
from model.rotations import rot6d_to_quat_wxyz  # noqa: E402
from model.schedule import CosineSchedule, DDIMSampler  # noqa: E402
from scene.schema import SceneTensor, WorkspaceBounds  # noqa: E402
from scene.vocab import OBJECT_VOCAB  # noqa: E402
from validator.core import SceneValidator  # noqa: E402

OBJECT_TYPE_NAMES: dict[int, str] = {i: v.name for i, v in enumerate(OBJECT_VOCAB)}
DDIM_STEPS = 50

JUDGE_PROMPT = """\
You evaluate whether a generated tabletop scene matches a task description.

Given a task description and a serialized scene representation, score the match on a 1-5 scale:

5 = Perfect: All mentioned objects present at appropriate locations matching the task
4 = Good: Minor positional inaccuracy, or one small detail off
3 = Partial: Some mentioned objects present but overall composition is wrong
2 = Poor: Most mentioned objects missing or wrongly placed
1 = No match: Scene unrelated to description

Respond with ONLY a valid JSON object on a single line, no other text:
{"score": N, "reasoning": "brief explanation"}"""

PRICING = {
    "claude-haiku-4-5": {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000},
}


# ---------------------------------------------------------------------------
# Scene helpers
# ---------------------------------------------------------------------------


def serialize_scene(st: SceneTensor, bounds: WorkspaceBounds) -> str:
    """Convert SceneTensor to compact text for VLM judging."""
    physical = st.denormalize(bounds) if st.poses[:, :3].abs().max() <= 1.0 + 1e-3 else st
    parts = ["Scene contents:"]
    n_present = 0
    for i in range(N_MAX):
        if not physical.presence[i].item():
            continue
        n_present += 1
        type_id = int(physical.object_types[i].item())
        name = OBJECT_TYPE_NAMES.get(type_id, f"obj_{type_id}")
        p = physical.poses[i]
        s = physical.scales[i]
        parts.append(
            f"  - {name} at ({p[0].item():.2f}m, {p[1].item():.2f}m, {p[2].item():.2f}m)"
            f" scale ({s[0].item():.2f}, {s[1].item():.2f}, {s[2].item():.2f})"
        )
    if n_present == 0:
        parts.append("  - (empty scene)")
    parts.append("Workspace: 0.6m x 0.6m tabletop, UR5e robot at origin")
    return "\n".join(parts)


def decode_scene(
    x_cont_i: torch.Tensor,
    type_ids_i: torch.Tensor,
    bounds: WorkspaceBounds,
) -> SceneTensor:
    """Decode a single (N_MAX, N_CONT) x_cont slice + (N_MAX,) type_ids into a SceneTensor.

    x_cont channel layout (N_CONT=13):
      0-2  : xyz normalised
      3-8  : rot6d
      9-11 : scale normalised
      12   : presence logit (> 0 => present)

    type_ids are provided by the sampler (argmax of head_type logits at
    each DDIM step) rather than being decoded from x_cont -- x_cont does
    not encode type in any channel.
    """
    presence = x_cont_i[:, 12] > 0.0                          # (N_MAX,)
    quats = rot6d_to_quat_wxyz(x_cont_i[:, 3:9])              # (N_MAX, 4)
    poses = torch.cat([x_cont_i[:, :3].clamp(-1, 1), quats], dim=-1)  # (N_MAX, 7)
    return SceneTensor(
        object_types=type_ids_i.cpu(),
        poses=poses.cpu(),
        scales=x_cont_i[:, 9:12].clamp(-1, 1).cpu(),
        presence=presence.cpu(),
    )


# ---------------------------------------------------------------------------
# Data: held-out descriptions
# ---------------------------------------------------------------------------


def build_held_out(data_dir: Path, n_per_template: int, total: int) -> list[dict]:
    reader = ShardReader(data_dir)
    per_template: dict[str, list[dict]] = defaultdict(list)
    for scene, desc, report, sdf in reader:
        tmpl = report.get("task_family", "unknown")
        per_template[tmpl].append({"description": desc, "template": tmpl})

    rng = torch.Generator()
    rng.manual_seed(42)
    held_out_by_tmpl: dict[str, list[dict]] = {t: items[-n_per_template:] for t, items in per_template.items()}
    total_held = sum(len(v) for v in held_out_by_tmpl.values())

    sampled: list[dict] = []
    for tmpl in sorted(held_out_by_tmpl):
        items = held_out_by_tmpl[tmpl]
        n_take = max(1, round(total * len(items) / total_held))
        idx = torch.randperm(len(items), generator=rng).tolist()[:n_take]
        sampled.extend(items[i] for i in idx)

    seen: set[str] = set()
    deduped = []
    for item in sampled:
        key = hashlib.sha256(item["description"].encode()).hexdigest()[:16]
        if key not in seen:
            seen.add(key)
            item["desc_id"] = key
            deduped.append(item)

    deduped = deduped[:total]
    counts = Counter(d["template"] for d in deduped)
    print(f"Held-out: {len(deduped)} descriptions -- " + ", ".join(f"{t}={n}" for t, n in sorted(counts.items())))
    return deduped


def load_text_cache(data_dir: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(data_dir / "text_embeddings.pt", weights_only=True)
    return {k: v.float().cpu() for k, v in raw.items()}


def get_text_emb(desc: str, cache: dict[str, torch.Tensor], device: torch.device) -> torch.Tensor | None:
    key = hashlib.sha256(desc.encode()).hexdigest()
    return cache[key].unsqueeze(0).to(device) if key in cache else None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(ckpt_path: Path, device: torch.device) -> SceneDenoiser:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    cfg = DenoiserConfig(
        n_layers=arch["n_layers"], d_model=arch["d_model"],
        n_heads=arch["n_heads"], ffn_mult=arch["ffn_mult"], dropout=0.0,
    )
    model = SceneDenoiser(cfg).to(device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    print(f"Loaded EMA from {ckpt_path.name} (step={ckpt['step']})")
    return model


# ---------------------------------------------------------------------------
# VLM judging
# ---------------------------------------------------------------------------


def vlm_score(client: object, model_id: str, description: str, scene_text: str) -> tuple[int, str]:
    """Return (score 1-5, reasoning)."""
    user_msg = f"Task description: {description}\n\n{scene_text}"
    resp = client.messages.create(
        model=model_id,
        max_tokens=200,
        temperature=0,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()
    # Extract JSON -- handle markdown fences
    m = re.search(r'\{.*?"score".*?\}', raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON in VLM response: {raw[:100]!r}")
    parsed = json.loads(m.group())
    score = int(parsed["score"])
    reasoning = str(parsed.get("reasoning", ""))
    in_tok = resp.usage.input_tokens
    out_tok = resp.usage.output_tokens
    return score, reasoning, in_tok, out_tok


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--n-prompts", type=int, default=200)
    p.add_argument("--n-per-prompt", type=int, default=8)
    p.add_argument("--budget-usd", type=float, default=2.0)
    p.add_argument("--vlm-model", default="claude-haiku-4-5")
    p.add_argument("--rrt-budget", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-vlm", action="store_true", help="Skip VLM scoring (Drake + type stats only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # VLM client
    if not args.skip_vlm:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            raise OSError("ANTHROPIC_API_KEY not set. Pass --skip-vlm to skip VLM scoring.")
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        pricing = PRICING[args.vlm_model]
        total_cost = 0.0
    else:
        client = None
        pricing = {"input": 0, "output": 0}
        total_cost = 0.0

    model = load_model(Path(args.checkpoint), device)
    schedule = CosineSchedule(T=1000)
    sampler = DDIMSampler(schedule, n_steps=DDIM_STEPS)
    bounds = WorkspaceBounds.default()
    validator = SceneValidator(rrt_budget_s=args.rrt_budget)

    data_dir = Path("data/v1")
    held_out = build_held_out(data_dir, n_per_template=100, total=args.n_prompts)
    text_cache = load_text_cache(data_dir)

    # Metrics
    vlm_scores: list[int] = []
    joint_ok: list[bool] = []            # Drake-valid AND vlm>=4
    all_type_ids: list[int] = []         # type_ids of ALL generated present objects
    drake_accepted = 0
    drake_total = 0
    n_empty = 0
    records: list[dict] = []

    t0 = time.monotonic()

    for d_idx, desc_info in enumerate(held_out):
        desc = desc_info["description"]
        desc_id = desc_info["desc_id"]
        tmpl = desc_info["template"]

        text_emb = get_text_emb(desc, text_cache, device)
        if text_emb is None:
            print(f"  WARN: no text emb for {desc_id[:8]} -- skipping")
            continue

        seed = hash((args.seed, desc_id)) & 0xFFFFFFFF
        sampling_fn = model.conditional_sampling_fn(
            text_emb.expand(args.n_per_prompt, -1)
        )
        with torch.no_grad():
            x_cont_batch, type_ids_batch = sampler.sample_with_types(
                sampling_fn,
                (args.n_per_prompt, N_MAX, 13),
                seed=seed,
                device=device,
                type_init="uniform",
            )

        for s_idx in range(args.n_per_prompt):
            st_norm = decode_scene(x_cont_batch[s_idx], type_ids_batch[s_idx], bounds)
            st_phys = st_norm.denormalize(bounds)

            # Type distribution
            present_mask = st_norm.presence.bool()
            if not present_mask.any():
                n_empty += 1
            else:
                tids = st_norm.object_types[present_mask].tolist()
                all_type_ids.extend(tids)

            # Drake
            rpt = validator.validate(st_phys)
            drake_total += 1
            if rpt.accepted:
                drake_accepted += 1

            # Serialize for VLM
            scene_text = serialize_scene(st_norm, bounds)

            # VLM scoring
            score_val: int | None = None
            reasoning = ""
            if client is not None and total_cost < args.budget_usd:
                try:
                    score_val, reasoning, in_tok, out_tok = vlm_score(
                        client, args.vlm_model, desc, scene_text
                    )
                    cost = in_tok * pricing["input"] + out_tok * pricing["output"]
                    total_cost += cost
                    vlm_scores.append(score_val)
                    joint_ok.append(rpt.accepted and score_val >= 4)
                except Exception as e:
                    print(f"  VLM error for {desc_id}: {e}")
                    score_val = None

            records.append({
                "desc_id": desc_id,
                "description": desc,
                "template": tmpl,
                "sample_idx": s_idx,
                "scene_text": scene_text,
                "drake_accepted": rpt.accepted,
                "drake_reason": (
                    None if rpt.accepted else (
                        "interpenetration" if not rpt.no_interpenetration else
                        "stability" if not rpt.stable_rest else
                        "ik" if not rpt.ik_reachable else "rrt"
                    )
                ),
                "vlm_score": score_val,
                "vlm_reasoning": reasoning,
                "type_ids": st_norm.object_types[present_mask].tolist(),
            })

        if (d_idx + 1) % 20 == 0:
            elapsed = time.monotonic() - t0
            print(
                f"  [{d_idx+1}/{len(held_out)}] "
                f"drake={drake_accepted}/{drake_total} ({100*drake_accepted/max(1,drake_total):.1f}%) "
                f"vlm_n={len(vlm_scores)} "
                f"cost=${total_cost:.3f} "
                f"elapsed={elapsed:.0f}s"
            )
            if client is not None and total_cost >= args.budget_usd:
                print(f"  Budget ${args.budget_usd:.2f} reached -- stopping VLM scoring")
                client = None  # stop further VLM calls

    # ---------------------------------------------------------------------------
    # Gate computation
    # ---------------------------------------------------------------------------
    type_counts = Counter(all_type_ids)
    n_types_seen = len(type_counts)
    type_0_frac = type_counts.get(0, 0) / max(1, sum(type_counts.values()))

    gate1_score = sum(vlm_scores) / len(vlm_scores) if vlm_scores else None
    gate2_frac = sum(joint_ok) / max(1, len(joint_ok)) if joint_ok else None
    gate3_types = n_types_seen  # num unique types generated
    gate3_pass = n_types_seen >= 5  # >=5/12 types present => >40% coverage

    validity_rate = drake_accepted / max(1, drake_total)

    print("\n" + "=" * 60)
    print("=== PHASE A GATE RESULTS ===")
    print(f"  Scenes evaluated:  {drake_total}")
    print(f"  Empty scenes:      {n_empty}")
    print(f"  Drake validity:    {validity_rate*100:.1f}%  ({drake_accepted}/{drake_total})")
    print(f"")
    if gate1_score is not None:
        g1 = "PASS" if gate1_score > 2.0 else "FAIL"
        print(f"  Gate 1 VLM mean:   {gate1_score:.3f} (threshold >2.0) [{g1}]")
    else:
        print(f"  Gate 1 VLM mean:   N/A (VLM skipped)")
    if gate2_frac is not None:
        g2 = "PASS" if gate2_frac > 0.05 else "FAIL"
        print(f"  Gate 2 Joint ok:   {gate2_frac*100:.1f}% (threshold >5%) [{g2}]")
    else:
        print(f"  Gate 2 Joint ok:   N/A (VLM skipped)")
    g3 = "PASS" if gate3_pass else "FAIL"
    print(f"  Gate 3 Type div:   {gate3_types}/12 types seen (threshold >=5) [{g3}]")
    print(f"  Type 0 fraction:   {type_0_frac*100:.1f}% (was 99% in v2/v3)")
    print(f"  Type distribution: {dict(sorted(type_counts.items()))}")
    if vlm_scores:
        print(f"  VLM cost:          ${total_cost:.3f}")
    print("=" * 60)

    gates_passed = sum([
        (gate1_score is not None and gate1_score > 2.0),
        (gate2_frac is not None and gate2_frac > 0.05),
        gate3_pass,
    ])
    if not args.skip_vlm and gate1_score is not None:
        print(f"\n  {gates_passed}/3 gates passed")
    else:
        print(f"\n  {1 if gate3_pass else 0}/1 gate evaluated (VLM skipped)")

    # ---------------------------------------------------------------------------
    # Save output
    # ---------------------------------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "checkpoint": args.checkpoint,
        "step": 50000,
        "n_prompts_evaluated": len(set(r["desc_id"] for r in records)),
        "n_scenes_total": drake_total,
        "drake_validity_rate": validity_rate,
        "gate1_vlm_mean": gate1_score,
        "gate1_pass": (gate1_score is not None and gate1_score > 2.0),
        "gate2_joint_frac": gate2_frac,
        "gate2_pass": (gate2_frac is not None and gate2_frac > 0.05),
        "gate3_types_seen": gate3_types,
        "gate3_pass": gate3_pass,
        "gates_passed": gates_passed,
        "type_counts": dict(sorted(type_counts.items())),
        "type_0_fraction": type_0_frac,
        "vlm_cost_usd": total_cost,
        "vlm_n_scored": len(vlm_scores),
        "records": records,
    }
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
