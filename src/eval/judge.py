"""VLM-based scene quality judge for Demiurge evaluation.

Scores generated scenes against task descriptions using the Anthropic
Messages API. Supports two tiers:

  haiku   claude-haiku-4-5-20251001  bulk scoring (~$3 / 1600 scenes)
  sonnet  claude-sonnet-4-6          headline metrics + disagreement resolution

Usage
-----
    from eval.judge import SceneJudge, JudgeConfig, JudgeTier

    cfg = JudgeConfig(tier=JudgeTier.HAIKU)
    judge = SceneJudge(cfg)
    result = judge.score(task="place the mug on the table", scene=scene_tensor)
    print(result.score, result.reasoning)

Batch usage::

    results = judge.score_batch(task_scene_pairs, max_workers=8)

Environment
-----------
Requires CLAUDE_API_KEY to be set in the environment before instantiation.
Raises EnvironmentError if absent. Do not hard-code or log the key.

Design decisions
----------------
* **Tier enum**: Model names are pinned to specific versions so scoring
  is reproducible across sessions. Upgrading requires an explicit code
  change, not a config drift.
* **Prompt loading**: The system prompt is loaded from
  src/eval/prompts/scene_judge.txt at instantiation, not inlined, so
  it can be diff'd and version-controlled independently.
* **JSON-only output**: The prompt instructs the model to output only
  JSON. We parse with json.loads() and fall back to a score=0 sentinel
  with an error flag if parsing fails.
* **Retry logic**: Three retries with 2s backoff on HTTP 429 / 5xx.
  Rate-limit aware: haiku tier batches at max_workers=8, sonnet at 2.
* **Token budget**: Each call uses ~400 input tokens (prompt + scene
  JSON) + ~40 output tokens. Haiku pricing: ~$0.0008 per call.

Invariants
----------
  # APPROX: SceneJudge scores are VLM judgements, not Drake validity.
  # They approximate task relevance but are not a substitute for the
  # Drake validator for physical correctness.
"""
from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent / "prompts" / "scene_judge.txt"

# Object type id -> human-readable name (must match src/scene/schema.py ObjectTypeId)
_TYPE_NAMES: dict[int, str] = {
    0: "YCB_CRACKER_BOX",
    1: "YCB_SUGAR_BOX",
    2: "YCB_SOUP_CAN",
    3: "YCB_MUSTARD_BOTTLE",
    4: "YCB_POTTED_MEAT_CAN",
    5: "YCB_BANANA",
    6: "YCB_PITCHER_BASE",
    7: "YCB_BLEACH_CLEANSER",
    8: "YCB_BOWL",
    9: "YCB_MUG",
    10: "YCB_POWER_DRILL",
    11: "YCB_SCISSORS",
    12: "PAD",   # inactive slot; should not appear in decoded scenes
}


class JudgeTier(str, Enum):
    """Anthropic model tier selection."""

    HAIKU = "claude-haiku-4-5-20251001"
    SONNET = "claude-sonnet-4-6"


@dataclass(frozen=True)
class JudgeConfig:
    """Configuration for SceneJudge.

    Attributes:
        tier: Model tier. Haiku for bulk; Sonnet for headline metrics.
        max_tokens: Max output tokens per call. 100 is sufficient for
            the one-JSON-object response.
        temperature: Sampling temperature. 0.0 for deterministic scoring.
        max_retries: Retries on HTTP 429 / 5xx before raising.
        retry_delay_s: Base delay between retries (doubles each retry).
        max_workers: Concurrent API calls in score_batch().
    """

    tier: JudgeTier = JudgeTier.HAIKU
    max_tokens: int = 100
    temperature: float = 0.0
    max_retries: int = 3
    retry_delay_s: float = 2.0
    max_workers: int = 8


@dataclass
class JudgeResult:
    """Result of a single scene scoring call.

    Attributes:
        score: Integer 0-5 quality score.
        reasoning: One-sentence explanation from the model.
        parse_error: True if the model returned unparseable output.
            Score is set to -1 in this case.
        latency_s: Wall-clock time for the API call.
        model: Model name used.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens consumed.
    """

    score: int
    reasoning: str
    parse_error: bool = False
    latency_s: float = 0.0
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class SceneJudge:
    """VLM judge for Demiurge scene quality evaluation.

    Args:
        cfg: JudgeConfig. Defaults to haiku tier.

    Raises:
        EnvironmentError: If CLAUDE_API_KEY is not set.
        RuntimeError: If the anthropic package is not installed.
        FileNotFoundError: If the prompt file is missing.
    """

    cfg: JudgeConfig = field(default_factory=JudgeConfig)

    def __post_init__(self) -> None:
        api_key = os.environ.get("CLAUDE_API_KEY", "")
        if not api_key:
            raise OSError(
                "CLAUDE_API_KEY is not set. Export it before instantiating SceneJudge. "
                "Do not hard-code or log the key."
            )
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package is required. Install with: uv pip install anthropic"
            ) from e

        if not _PROMPT_PATH.exists():
            raise FileNotFoundError(
                f"Scene judge prompt not found: {_PROMPT_PATH}. "
                "Expected at src/eval/prompts/scene_judge.txt"
            )
        self._system_prompt = _PROMPT_PATH.read_text(encoding="utf-8")
        self._client = anthropic.Anthropic(api_key=api_key)
        log.info("SceneJudge initialised | model=%s", self.cfg.tier.value)

    # ------------------------------------------------------------------
    # Core scoring API
    # ------------------------------------------------------------------

    def score(self, task: str, scene: Any) -> JudgeResult:
        """Score a single scene against a task description.

        Args:
            task: Human-readable task description.
            scene: Either a SceneTensor (physical/denormalized) or a
                pre-formatted dict with 'n_objects' and 'objects' keys.

        Returns:
            JudgeResult with integer score [0, 5] and one-sentence reasoning.
        """
        scene_dict = self._format_scene(scene)
        user_content = json.dumps({"task": task, "scene": scene_dict}, indent=2)
        return self._call_api(user_content)

    def score_batch(
        self,
        pairs: list[tuple[str, Any]],
        max_workers: int | None = None,
        show_progress: bool = True,
    ) -> list[JudgeResult]:
        """Score a batch of (task, scene) pairs concurrently.

        Args:
            pairs: List of (task_description, scene) tuples.
            max_workers: Thread pool size. Defaults to cfg.max_workers.
            show_progress: Log progress every 50 completions.

        Returns:
            List of JudgeResult in the same order as input pairs.
        """
        workers = max_workers if max_workers is not None else self.cfg.max_workers
        results: list[JudgeResult | None] = [None] * len(pairs)
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.score, task, scene): idx
                for idx, (task, scene) in enumerate(pairs)
            }
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as exc:
                    log.warning("score_batch[%d] failed: %s", idx, exc)
                    results[idx] = JudgeResult(
                        score=-1,
                        reasoning=f"Exception: {exc}",
                        parse_error=True,
                    )
                completed += 1
                if show_progress and completed % 50 == 0:
                    log.info("SceneJudge: %d/%d complete", completed, len(pairs))

        # All slots should be filled at this point.
        assert all(r is not None for r in results)
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _format_scene(self, scene: Any) -> dict[str, Any]:
        """Convert scene to the dict format expected by the judge prompt."""
        # Accept a pre-formatted dict directly (for testing / flexibility).
        if isinstance(scene, dict):
            return scene

        # Assume SceneTensor with .object_types, .poses, .scales, .presence
        present = scene.presence.bool()
        objects = []
        for i in range(present.shape[0]):
            if not present[i]:
                continue
            type_id = int(scene.object_types[i].item())
            xyz = scene.poses[i, :3].tolist()
            scale = scene.scales[i].mean().item()
            objects.append({
                "type": _TYPE_NAMES.get(type_id, f"UNKNOWN_{type_id}"),
                "position_m": [round(v, 3) for v in xyz],
                "scale": round(scale, 3),
            })

        return {"n_objects": len(objects), "objects": objects}

    def _call_api(self, user_content: str) -> JudgeResult:
        """Make one API call with retry logic."""
        import anthropic

        delay = self.cfg.retry_delay_s
        last_exc: Exception | None = None

        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.monotonic()
            try:
                response = self._client.messages.create(
                    model=self.cfg.tier.value,
                    max_tokens=self.cfg.max_tokens,
                    temperature=self.cfg.temperature,
                    system=self._system_prompt,
                    messages=[{"role": "user", "content": user_content}],
                )
                latency = time.monotonic() - t0
                raw = response.content[0].text.strip()
                return self._parse_response(raw, latency, response)

            except anthropic.RateLimitError as exc:
                last_exc = exc
                log.warning("Rate limit (attempt %d/%d), sleeping %.1fs",
                            attempt + 1, self.cfg.max_retries + 1, delay)
                time.sleep(delay)
                delay *= 2.0

            except anthropic.APIStatusError as exc:
                if exc.status_code >= 500:
                    last_exc = exc
                    log.warning("Server error %d (attempt %d/%d), retrying",
                                exc.status_code, attempt + 1, self.cfg.max_retries + 1)
                    time.sleep(delay)
                    delay *= 2.0
                else:
                    raise

        raise RuntimeError(
            f"API call failed after {self.cfg.max_retries + 1} attempts"
        ) from last_exc

    def _parse_response(self, raw: str, latency: float, response: Any) -> JudgeResult:
        """Parse JSON response from the model."""
        try:
            # Strip markdown code fences if model emits them despite the prompt.
            clean = raw
            if clean.startswith("```"):
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            parsed = json.loads(clean.strip())
            score = int(parsed["score"])
            if not 0 <= score <= 5:
                raise ValueError(f"Score out of range: {score}")
            return JudgeResult(
                score=score,
                reasoning=str(parsed.get("reasoning", ""))[:200],
                parse_error=False,
                latency_s=latency,
                model=response.model,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
            )
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            log.warning("Failed to parse judge response: %s | raw=%r", exc, raw[:200])
            return JudgeResult(
                score=-1,
                reasoning=f"PARSE_ERROR: {exc}",
                parse_error=True,
                latency_s=latency,
                model=getattr(response, "model", ""),
            )
