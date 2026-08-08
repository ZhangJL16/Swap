from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Mapping

import numpy as np


COUNTER_FIELDS = (
    "tasks_completed",
    "voluntary_station_arrivals",
    "backup_recovery_count",
    "charging_visits",
    "charging_steps",
    "energy_charged",
    "departure_attempts",
    "departure_rejection_count",
    "generator_accepted_steps",
    "no_generator_steps",
    "energy_consumed",
    "collision_count",
    "energy_depletion_count",
    "uncertified_publication_count",
    "invalid_kappa_fallback_count",
)

LIST_FIELDS = (
    "task_completion_steps",
    "energy_margin_at_backup",
    "energy_margin_at_station_approach",
    "energy_on_station_arrival",
    "energy_on_departure",
    "charge_durations",
)


def metric_snapshot_delta(
    previous: Mapping[str, Any] | None,
    current: Mapping[str, Any],
) -> dict[str, Any]:
    before = {} if previous is None else previous
    delta: dict[str, Any] = {}
    for field_name in COUNTER_FIELDS:
        current_value = float(current.get(field_name, 0.0))
        previous_value = float(before.get(field_name, 0.0))
        difference = current_value - previous_value
        if difference < -1e-12:
            raise ValueError(f"persistent metric decreased within one episode: {field_name}")
        delta[field_name] = difference
    for field_name in LIST_FIELDS:
        current_values = list(current.get(field_name, ()))
        previous_values = list(before.get(field_name, ()))
        if len(current_values) < len(previous_values):
            raise ValueError(f"persistent list metric shrank within one episode: {field_name}")
        delta[field_name] = current_values[len(previous_values):]
    margin = current.get("minimum_energy_margin")
    delta["minimum_energy_margin"] = None if margin is None else float(margin)
    return delta


@dataclass(slots=True)
class PersistentMetricAccumulator:
    total_steps: int = 0
    total_reward: float = 0.0
    total_goal_progress: float = 0.0
    progressing_steps: int = 0
    counters: dict[str, float] = field(default_factory=lambda: {name: 0.0 for name in COUNTER_FIELDS})
    lists: dict[str, list[float]] = field(default_factory=lambda: {name: [] for name in LIST_FIELDS})
    minimum_energy_margin: float = float("inf")

    def observe(self, reward: float, info: Mapping[str, Any], delta: Mapping[str, Any]) -> None:
        self.total_steps += 1
        self.total_reward += float(reward)
        progress = float(info.get("goal_progress", 0.0))
        self.total_goal_progress += progress
        self.progressing_steps += int(progress > 0.0)
        for field_name in COUNTER_FIELDS:
            self.counters[field_name] += float(delta.get(field_name, 0.0))
        for field_name in LIST_FIELDS:
            self.lists[field_name].extend(float(value) for value in delta.get(field_name, ()))
        margin = delta.get("minimum_energy_margin")
        if margin is not None and np.isfinite(margin):
            self.minimum_energy_margin = min(self.minimum_energy_margin, float(margin))

    def summary(self) -> dict[str, Any]:
        steps = max(1, self.total_steps)
        tasks = int(round(self.counters["tasks_completed"]))
        task_steps = self.lists["task_completion_steps"]
        result: dict[str, Any] = {
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "tasks_completed": tasks,
            "tasks_per_1000_steps": 1000.0 * tasks / steps,
            "task_completion_count": len(task_steps),
            "task_completion_steps": list(task_steps),
            "mean_task_completion_steps": float(np.mean(task_steps)) if task_steps else None,
            "voluntary_station_arrivals": int(round(self.counters["voluntary_station_arrivals"])),
            "backup_recovery_count": int(round(self.counters["backup_recovery_count"])),
            "backup_rate": self.counters["backup_recovery_count"] / steps,
            "charging_visits": int(round(self.counters["charging_visits"])),
            "charging_steps": int(round(self.counters["charging_steps"])),
            "charging_fraction": self.counters["charging_steps"] / steps,
            "energy_charged": self.counters["energy_charged"],
            "departure_attempts": int(round(self.counters["departure_attempts"])),
            "departure_rejections": int(round(self.counters["departure_rejection_count"])),
            "generator_accepted_steps": int(round(self.counters["generator_accepted_steps"])),
            "generator_acceptance_rate": self.counters["generator_accepted_steps"] / steps,
            "no_generator_steps": int(round(self.counters["no_generator_steps"])),
            "no_generator_rate": self.counters["no_generator_steps"] / steps,
            "energy_consumed": self.counters["energy_consumed"],
            "energy_per_task": self.counters["energy_consumed"] / max(1, tasks),
            "minimum_energy_margin": None if not np.isfinite(self.minimum_energy_margin) else self.minimum_energy_margin,
            "collision_count": int(round(self.counters["collision_count"])),
            "energy_depletion_count": int(round(self.counters["energy_depletion_count"])),
            "uncertified_publication_count": int(round(self.counters["uncertified_publication_count"])),
            "invalid_kappa_fallback_count": int(round(self.counters["invalid_kappa_fallback_count"])),
            "total_goal_progress": self.total_goal_progress,
            "fraction_steps_progressing_to_goal": self.progressing_steps / steps,
            "mean_goal_progress_per_step": self.total_goal_progress / steps,
        }
        result.update({name: list(values) for name, values in self.lists.items() if name != "task_completion_steps"})
        return result


def episode_record(
    episode_id: int,
    episode_seed: int,
    reset_info: Mapping[str, Any],
    accumulator: PersistentMetricAccumulator,
    goal_sequence: list[np.ndarray],
    *,
    terminated: bool,
    truncated: bool,
    partial: bool,
) -> dict[str, Any]:
    metrics = accumulator.summary()
    start = reset_info.get("sampled_start")
    start_position = getattr(start, "position", start)
    initial_goal = reset_info.get("sampled_goal")
    return {
        "episode_id": int(episode_id),
        "episode_seed": int(episode_seed),
        "start_position": None if start_position is None else np.asarray(start_position, dtype=float).tolist(),
        "initial_goal": None if initial_goal is None else np.asarray(initial_goal, dtype=float).tolist(),
        "steps": metrics["total_steps"],
        "return": metrics["total_reward"],
        "tasks_completed": metrics["tasks_completed"],
        "voluntary_station_arrivals": metrics["voluntary_station_arrivals"],
        "backup_recoveries": metrics["backup_recovery_count"],
        "charging_visits": metrics["charging_visits"],
        "charging_steps": metrics["charging_steps"],
        "generator_accepted_steps": metrics["generator_accepted_steps"],
        "no_generator_steps": metrics["no_generator_steps"],
        "minimum_energy_margin": metrics["minimum_energy_margin"],
        "collision_count": metrics["collision_count"],
        "energy_depletion_count": metrics["energy_depletion_count"],
        "uncertified_publication_count": metrics["uncertified_publication_count"],
        "invalid_kappa_fallback_count": metrics["invalid_kappa_fallback_count"],
        "goal_progress": metrics["total_goal_progress"],
        "goal_sequence": [np.asarray(goal, dtype=float).tolist() for goal in goal_sequence],
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "partial": bool(partial),
    }


def write_jsonl(path, records: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )


def learning_curve_steps_monotonic(records: list[Mapping[str, Any]]) -> bool:
    steps = [int(record["step"]) for record in records]
    return bool(steps and all(right > left for left, right in zip(steps, steps[1:])))
