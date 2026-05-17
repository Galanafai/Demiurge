"""Downstream evaluation: RRT planning success on generated scenes.

For each (scene, prompt) pair, we run the SceneValidator and extract the
rrt_solvable flag, plan length, and planning time from the ValidityReport.
The time budget per task is passed through to the validator's rrt_budget_s.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from scene.schema import SceneTensor

if TYPE_CHECKING:
    from validator.core import SceneValidator


@dataclass
class TaskRecord:
    """Per-task result from downstream evaluation."""

    prompt: str
    n_present: int
    no_interpenetration: bool
    stable_rest: bool
    ik_reachable: bool
    rrt_solvable: bool
    rrt_plan_length: float  # number of waypoints; 0 if unsolved
    planning_time_s: float


@dataclass
class DownstreamResult:
    """Aggregate result from DownstreamEvaluator.evaluate().

    Attributes:
        success_rate: Fraction of tasks where rrt_solvable is True.
        mean_plan_length: Mean waypoint count over successful plans.
        mean_planning_time_s: Mean wall-clock planning time per task.
        records: Per-task TaskRecord list for detailed analysis.
    """

    success_rate: float
    mean_plan_length: float
    mean_planning_time_s: float
    records: list[TaskRecord] = field(default_factory=list)


class DownstreamEvaluator:
    """Evaluates downstream RRT planning utility of generated scenes.

    Args:
        validator: A SceneValidator instance. The validator's rrt_budget_s
            controls how long each RRT attempt is allowed to run.
    """

    def __init__(self, validator: "SceneValidator") -> None:
        self._validator = validator

    def evaluate(
        self,
        scenes: list[SceneTensor],
        prompts: list[str],
    ) -> DownstreamResult:
        """Run the validator on each (scene, prompt) pair and aggregate.

        Args:
            scenes: Physical (denormalized) SceneTensors.
            prompts: One prompt string per scene (used only for logging).

        Returns:
            DownstreamResult with per-task records and aggregate stats.
        """
        import time

        if not scenes:
            return DownstreamResult(
                success_rate=0.0,
                mean_plan_length=0.0,
                mean_planning_time_s=0.0,
            )

        records: list[TaskRecord] = []
        for scene, prompt in zip(scenes, prompts):
            t0 = time.monotonic()
            report = self._validator.validate(scene)
            elapsed = time.monotonic() - t0

            pres = scene.presence.bool()
            n_present = int(pres.sum().item())

            # Extract plan length from the report if available.
            # ValidityReport stores rrt_plan_length when rrt_solvable is True.
            plan_length = float(getattr(report, "rrt_plan_length", 0) or 0)

            records.append(TaskRecord(
                prompt=prompt,
                n_present=n_present,
                no_interpenetration=bool(getattr(report, "no_interpenetration", False)),
                stable_rest=bool(getattr(report, "stable_rest", False)),
                ik_reachable=bool(getattr(report, "ik_reachable", False)),
                rrt_solvable=bool(getattr(report, "rrt_solvable", False)),
                rrt_plan_length=plan_length,
                planning_time_s=elapsed,
            ))

        successes = [r for r in records if r.rrt_solvable]
        success_rate = len(successes) / len(records)

        mean_plan_length = (
            sum(r.rrt_plan_length for r in successes) / len(successes)
            if successes
            else 0.0
        )
        mean_planning_time_s = sum(r.planning_time_s for r in records) / len(records)

        return DownstreamResult(
            success_rate=success_rate,
            mean_plan_length=mean_plan_length,
            mean_planning_time_s=mean_planning_time_s,
            records=records,
        )
