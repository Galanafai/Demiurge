"""VLM judging for Universal Guidance sweep scenes.

Samples N scenes per config from the sweep output and submits them
to Claude for scoring. Writes per-record JSONL for the Pareto plotter.

Usage:
    python3 -u scripts/judge_sweep_with_vlm.py \\
        --sweep-dir artifacts/ug_sweep_v1/ \\
        --samples-per-config 100 \\
        --budget-usd 3.0 \\
        --out artifacts/ug_sweep_v1/vlm_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path

import anthropic

JUDGE_PROMPT = """You are a robotics scene evaluator. Given a task description
and a generated tabletop scene, rate how well the scene supports the task
on a scale of 1-5:
1 = scene is irrelevant or physically impossible
2 = scene is plausible but poorly suited to the task
3 = scene is acceptable for the task
4 = scene is well-suited to the task
5 = scene is excellent for the task

Respond ONLY with valid JSON: {"score": N, "reasoning": "..."}
Keep reasoning under 80 words."""

MODEL_ID = "claude-haiku-4-5"


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def vlm_score(client: anthropic.Anthropic, description: str, scene_text: str) -> tuple[int, str, int]:
    user_msg = f"Task description: {description}\n\n{scene_text}"
    resp = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        temperature=0,
        system=JUDGE_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = resp.content[0].text.strip()
    in_tok = resp.usage.input_tokens + resp.usage.output_tokens
    for candidate in (_strip_fences(raw), raw):
        m = re.search(r'\{.*?"score".*?\}', candidate, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group())
                return int(parsed["score"]), str(parsed.get("reasoning", "")), in_tok
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
    return 0, f"parse_fail: {raw[:100]}", in_tok


def scene_to_text(rec: dict) -> str:
    lines = [f"Objects present: {rec.get('n_present', '?')}"]
    tids = rec.get("type_ids", [])
    VOCAB = ["CUBE","SPHERE","CYLINDER","BOX_TALL","BOX_FLAT","MUSTARD_BOTTLE",
             "SUGAR_BOX","TOMATO_SOUP_CAN","BLEACH_CLEANSER","BANANA",
             "MASTER_CHEF_CAN","GELATIN_BOX","PAD"]
    type_names = [VOCAB[min(t, 12)] for t in tids if t < 12]
    if type_names:
        lines.append(f"Object types: {', '.join(type_names)}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-dir", required=True)
    p.add_argument("--samples-per-config", type=int, default=100)
    p.add_argument("--budget-usd", type=float, default=3.0)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.sweep_dir)
    rng = random.Random(args.seed)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    client = anthropic.Anthropic(api_key=api_key)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_cost = 0.0
    total_calls = 0

    run_files = sorted(sweep_dir.glob("run_gs*.json"))
    print(f"Found {len(run_files)} run files")

    with open(out_path, "a") as fout:
        for run_file in run_files:
            if total_cost >= args.budget_usd:
                print(f"HALT: budget ${args.budget_usd} reached")
                break

            data = json.loads(run_file.read_text())
            config = data["config"]
            records = data["records"]

            gs = config["guidance_scale"]
            seed = config["seed"]

            # Sample without replacement
            sample_size = min(args.samples_per_config, len(records))
            sampled = rng.sample(records, sample_size)

            batch_cost = 0.0
            batch_scores = []

            for rec in sampled:
                if total_cost + batch_cost >= args.budget_usd:
                    break
                desc = rec.get("description", "")
                scene_text = scene_to_text(rec)
                score, reasoning, tok = vlm_score(client, desc, scene_text)
                call_cost = tok * 0.80 / 1_000_000
                batch_cost += call_cost
                batch_scores.append(score)
                total_calls += 1

                out_rec = {
                    "guidance_scale": gs,
                    "seed": seed,
                    "desc_id": rec.get("desc_id", ""),
                    "description": desc,
                    "vlm_score": score,
                    "reasoning": reasoning,
                    "drake_accepted": rec.get("drake_accepted", False),
                }
                fout.write(json.dumps(out_rec) + "\n")
                fout.flush()

            total_cost += batch_cost
            mean_score = sum(batch_scores) / max(1, len(batch_scores))
            print(
                f"  gs={gs:.1f} seed={seed}: n={len(batch_scores)} "
                f"vlm_mean={mean_score:.3f} cost=${batch_cost:.3f} "
                f"total=${total_cost:.3f}"
            )

    print(f"\nVLM judging complete: {total_calls} calls, ${total_cost:.3f}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
