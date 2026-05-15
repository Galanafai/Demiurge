"""VLM judge runner for Week 3 Task 9.

Reads input JSONL of scene samples, calls Anthropic API with a fixed judge
prompt, writes output JSONL with vlm_score and vlm_reasoning per sample.

Usage:
    python3 scripts/run_vlm_eval.py \
        --model claude-haiku-4-5 \
        --input artifacts/week3_vlm_eval/samples.jsonl \
        --output artifacts/week3_vlm_eval/haiku_scores.jsonl \
        --temperature 0 --max-tokens 200 \
        --rate-limit 50 --budget-cap 5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# Pricing (per-token, in USD)
# ---------------------------------------------------------------------------

PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5":   {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000},
    "claude-sonnet-4-5":  {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "claude-sonnet-4-6":  {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
}

JUDGE_PROMPT_PATH = pathlib.Path("src/eval/prompts/scene_judge.txt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--subset-size", type=int, default=None)
    p.add_argument("--subset-seed", type=int, default=42)
    p.add_argument("--rate-limit", type=int, default=50, help="calls per minute")
    p.add_argument("--budget-cap", type=float, default=10.0, help="USD hard cap")
    return p.parse_args()


def load_samples(path: str, subset_size: int | None, seed: int) -> list[dict[str, Any]]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    if subset_size is not None and subset_size < len(samples):
        import random
        rng = random.Random(seed)
        samples = rng.sample(samples, subset_size)
        print(f"Subset: {len(samples)} samples (seed={seed})")
    return samples


def call_judge(
    client: Any,
    model: str,
    judge_prompt: str,
    description: str,
    scene_text: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], int, int]:
    """Returns (parsed_json, input_tokens, output_tokens). Raises on API error."""
    user_msg = f"TASK DESCRIPTION:\n{description}\n\nGENERATED SCENE:\n{scene_text}"
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=judge_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_text = resp.content[0].text.strip()
    # Parse JSON -- allow for minor formatting issues
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Try to extract JSON object with regex
        m = re.search(r'\{[^}]+\}', raw_text)
        if m:
            parsed = json.loads(m.group())
        else:
            raise ValueError(f"Malformed JSON from VLM: {raw_text!r}")

    if "score" not in parsed:
        raise ValueError(f"Missing 'score' in response: {parsed}")
    score = int(parsed["score"])
    if score < 1 or score > 5:
        raise ValueError(f"Score out of range: {score}")

    return parsed, resp.usage.input_tokens, resp.usage.output_tokens


def main() -> None:
    args = parse_args()
    judge_prompt = JUDGE_PROMPT_PATH.read_text().strip()
    samples = load_samples(args.input, args.subset_size, args.subset_seed)

    pricing = PRICING.get(args.model)
    if pricing is None:
        print(f"WARN: unknown model {args.model}, using Haiku pricing as fallback")
        pricing = PRICING["claude-haiku-4-5"]

    from anthropic import Anthropic
    client = Anthropic()

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    running_cost = 0.0
    malformed = 0
    total = len(samples)
    interval = 60.0 / args.rate_limit  # seconds between calls to stay under rate limit

    print(f"Model: {args.model}")
    print(f"Samples: {total}")
    print(f"Rate limit: {args.rate_limit}/min ({interval:.1f}s between calls)")
    print(f"Budget cap: ${args.budget_cap:.2f}")
    print(f"Output: {out_path}")
    print()

    t_start = time.monotonic()
    written = 0

    with open(out_path, "w") as out_f:
        for idx, sample in enumerate(samples):
            # Budget check
            if running_cost >= args.budget_cap:
                print(f"\nBUDGET CAP REACHED: ${running_cost:.4f} >= ${args.budget_cap:.2f}. Halting.")
                break

            t0 = time.monotonic()
            try:
                parsed, in_tok, out_tok = call_judge(
                    client, args.model, judge_prompt,
                    sample["description"], sample["scene_text"],
                    args.temperature, args.max_tokens,
                )
                call_cost = in_tok * pricing["input"] + out_tok * pricing["output"]
                running_cost += call_cost

                out_record = {
                    **sample,
                    "vlm_model": args.model,
                    "vlm_score": parsed["score"],
                    "vlm_reasoning": parsed.get("reasoning", ""),
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "call_cost_usd": call_cost,
                }
                out_f.write(json.dumps(out_record) + "\n")
                written += 1

            except Exception as e:
                malformed += 1
                print(f"  ERROR sample {idx} ({sample.get('sample_id', '?')}): {e}")
                # Write error record to preserve sample_id in output
                out_record = {
                    **sample,
                    "vlm_model": args.model,
                    "vlm_score": None,
                    "vlm_reasoning": f"ERROR: {e}",
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "call_cost_usd": 0.0,
                }
                out_f.write(json.dumps(out_record) + "\n")
                written += 1

                # Halt if >10% malformed
                if malformed > 5 and malformed / max(1, idx + 1) > 0.1:
                    print(f"\nHALT: malformed rate {malformed}/{idx+1} exceeds 10%. Surface for review.")
                    sys.exit(1)

            # Progress
            if (idx + 1) % 100 == 0 or idx == total - 1:
                elapsed = time.monotonic() - t_start
                rate = (idx + 1) / elapsed
                remaining = (total - idx - 1) / max(rate, 1e-6)
                print(f"  [{idx+1}/{total}]  cost=${running_cost:.4f}  "
                      f"malformed={malformed}  eta={remaining/60:.1f}min")
                out_f.flush()

            # Rate limiting: sleep to stay under rate limit
            elapsed_call = time.monotonic() - t0
            sleep_for = max(0.0, interval - elapsed_call)
            if sleep_for > 0:
                time.sleep(sleep_for)

    # Final summary
    elapsed_total = time.monotonic() - t_start
    print(f"\n{'='*60}")
    print(f"Done. Written: {written}/{total}")
    print(f"Malformed: {malformed}")
    print(f"Total cost: ${running_cost:.4f}")
    print(f"Wall-clock: {elapsed_total/60:.1f} min")
    print(f"Output: {out_path}")

    # Save cost summary
    summary = {
        "model": args.model,
        "total_samples": total,
        "written": written,
        "malformed": malformed,
        "total_cost_usd": running_cost,
        "elapsed_s": elapsed_total,
    }
    summary_path = out_path.with_suffix(".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
