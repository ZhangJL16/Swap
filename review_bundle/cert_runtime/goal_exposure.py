from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


def goal_key(goal: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(goal, dtype=np.float64), decimals=decimals))


def goal_direction_statistics(directions: Iterable[np.ndarray], bins: int = 16) -> dict[str, float]:
    vectors = []
    for direction in directions:
        horizontal = np.asarray(direction, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(horizontal))
        if norm > 1e-12:
            vectors.append(horizontal / norm)
    if not vectors:
        return {"angular_coverage_degrees": 0.0, "direction_entropy": 0.0}
    angles = np.mod(np.arctan2(np.asarray(vectors)[:, 1], np.asarray(vectors)[:, 0]), 2.0 * np.pi)
    pairwise = np.abs(angles[:, None] - angles[None, :])
    pairwise = np.minimum(pairwise, 2.0 * np.pi - pairwise)
    counts, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
    probabilities = counts[counts > 0] / max(1, counts.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "angular_coverage_degrees": float(np.degrees(np.max(pairwise))),
        "direction_entropy": entropy,
    }


def task_completion_distance_invariant(
    mode_before: str,
    distance_after: float,
    goal_radius: float,
    task_completed_now: bool,
) -> bool:
    eligible = mode_before == "TASK_RL" and float(distance_after) <= float(goal_radius) + 1e-12
    return not eligible or bool(task_completed_now)


def goal_exposure_reset_boundary(
    step_number: int,
    total_steps: int,
    reset_interval: int | None,
    *,
    terminated: bool,
    truncated: bool,
) -> bool:
    return bool(
        reset_interval
        and step_number < total_steps
        and step_number % int(reset_interval) == 0
        and not terminated
        and not truncated
    )


def goal_exposure_reset_seed(base_seed: int, collector_reset_index: int) -> int:
    if collector_reset_index <= 0:
        raise ValueError("collector reset index must be positive")
    return int((int(base_seed) + 1_000_003 * int(collector_reset_index)) % (2**31 - 1))


def training_protocol_name(reset_interval: int | None) -> str:
    return "episodic_multi_goal_exposure" if reset_interval else "persistent_only"


def batch_goal_diversity(
    transitions: Sequence[object],
    position_slice: slice,
    goal_delta_slice: slice,
    world_size: np.ndarray,
) -> dict[str, float | int]:
    goals = []
    directions = []
    scale = np.asarray(world_size, dtype=np.float64)
    for transition in transitions:
        observation = np.asarray(transition.observation, dtype=np.float64)
        position = observation[position_slice] * scale
        direction = observation[goal_delta_slice] * scale
        task_goal = getattr(transition, "task_goal", None)
        goals.append(position + direction if task_goal is None else np.asarray(task_goal, dtype=np.float64))
        directions.append(direction)
    statistics = goal_direction_statistics(directions)
    return {
        "batch_unique_goal_count": len({goal_key(goal) for goal in goals}),
        "batch_goal_direction_entropy": statistics["direction_entropy"],
    }


@dataclass(slots=True)
class GoalExposureAccumulator:
    assignments: list[dict[str, object]] = field(default_factory=list)
    steps_by_goal: dict[tuple[float, ...], int] = field(default_factory=dict)
    completed_goal_steps: list[int] = field(default_factory=list)
    current_goal_key: tuple[float, ...] | None = None
    current_goal_age_steps: int = 0
    collector_resets: int = 0
    natural_task_completions: int = 0
    interval_new_goals: int = 0
    interval_goal_ages: list[int] = field(default_factory=list)

    def assign(
        self,
        goal: np.ndarray,
        position: np.ndarray,
        step: int,
        source: str,
        reset_seed: int | None,
    ) -> None:
        key = goal_key(goal)
        self.current_goal_key = key
        self.current_goal_age_steps = 0
        self.steps_by_goal.setdefault(key, 0)
        self.assignments.append({
            "step": int(step),
            "source": str(source),
            "reset_seed": None if reset_seed is None else int(reset_seed),
            "sampled_start": np.asarray(position, dtype=np.float64).tolist(),
            "sampled_goal": np.asarray(goal, dtype=np.float64).tolist(),
            "goal_direction": (np.asarray(goal, dtype=np.float64) - np.asarray(position, dtype=np.float64)).tolist(),
        })
        self.interval_new_goals += 1
        if source == "collector_reset":
            self.collector_resets += 1

    def observe_step(self, task_completed_now: bool) -> None:
        if self.current_goal_key is None:
            raise RuntimeError("goal exposure tracker has no current goal")
        self.current_goal_age_steps += 1
        self.steps_by_goal[self.current_goal_key] += 1
        self.interval_goal_ages.append(self.current_goal_age_steps)
        if task_completed_now:
            self.natural_task_completions += 1
            self.completed_goal_steps.append(self.current_goal_age_steps)

    def interval_summary(self, *, reset: bool) -> dict[str, float | int]:
        result = {
            "interval_new_goals": self.interval_new_goals,
            "median_current_goal_age": float(np.median(self.interval_goal_ages)) if self.interval_goal_ages else 0.0,
        }
        if reset:
            self.interval_new_goals = 0
            self.interval_goal_ages.clear()
        return result

    def summary(self) -> dict[str, object]:
        unique = {goal_key(np.asarray(item["sampled_goal"], dtype=np.float64)) for item in self.assignments}
        directions = [np.asarray(item["goal_direction"], dtype=np.float64) for item in self.assignments]
        statistics = goal_direction_statistics(directions)
        steps = list(self.steps_by_goal.values())
        return {
            "unique_goals_seen": len(unique),
            "goal_assignments": len(self.assignments),
            "collector_resets": self.collector_resets,
            "natural_task_completions": self.natural_task_completions,
            "steps_per_goal": steps,
            "median_steps_per_goal": float(np.median(steps)) if steps else 0.0,
            "steps_per_completed_goal": list(self.completed_goal_steps),
            "current_goal_age_steps": self.current_goal_age_steps,
            "goal_direction_angular_coverage_degrees": statistics["angular_coverage_degrees"],
            "goal_direction_entropy": statistics["direction_entropy"],
            "goal_sequence_length": len(self.assignments),
            "goal_exposure_gate": "PASS" if len(unique) > 1 else "FAIL",
            "distribution_annotation": "MULTI_GOAL_TRAINING_DISTRIBUTION" if len(unique) > 1 else "SINGLE_GOAL_TRAINING_DISTRIBUTION",
            "assignments": list(self.assignments),
        }
