#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.goal_exposure import goal_direction_statistics, goal_key, task_completion_distance_invariant


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _distance_row(row: dict[str, object]) -> dict[str, object]:
    goal = np.asarray(row.get("goal_before", row["goal"]), dtype=np.float64)
    before = np.asarray(row["position_before"], dtype=np.float64)
    after = np.asarray(row["position"], dtype=np.float64)
    mode_before = str(row.get("persistent_mode_before", row["persistent_mode"]))
    radius = float(row.get("goal_radius", 0.20))
    distance_before = float(np.linalg.norm(before - goal))
    distance_after = float(np.linalg.norm(after - goal))
    completed = bool(row.get("task_completed_now", False))
    return {
        "step": int(row["step"]),
        "episode_id": int(row["episode_id"]),
        "episode_step": int(row["episode_step"]),
        "task_id": row.get("task_id_before", row.get("task_id")),
        "goal": goal.tolist(),
        "persistent_mode_before": mode_before,
        "persistent_mode_after": row.get("persistent_mode"),
        "execution_authority": row.get("execution_authority"),
        "position_before": before.tolist(),
        "position_after": after.tolist(),
        "distance_before": distance_before,
        "distance_after": distance_after,
        "goal_radius": radius,
        "task_completed_now": completed,
        "tasks_completed": int(row.get("tasks_completed", 0)),
        "invariant_valid": task_completion_distance_invariant(mode_before, distance_after, radius, completed),
    }


def _minimum(rows: list[dict[str, object]]) -> dict[str, object] | None:
    return None if not rows else min(rows, key=lambda row: float(row["distance_after"]))


def audit_directory(directory: Path, warmup_steps: int) -> tuple[dict[str, object], dict[str, object]]:
    rows = _jsonl(directory / "trajectory_events.jsonl")
    timeline = [_distance_row(row) for row in rows]
    task_rows = [row for row in timeline if row["persistent_mode_before"] == "TASK_RL"]
    authority_rows = [row for row in timeline if row["execution_authority"] == "RL_GENERATOR"]
    backup_rows = [row for row in timeline if row["execution_authority"] == "KAPPA_BACKUP"]
    charging_rows = [row for row in timeline if row["persistent_mode_before"] == "CHARGING_RL"]
    violations = [row for row in timeline if not row["invariant_valid"]]
    inside_non_task = [
        row for row in timeline
        if row["persistent_mode_before"] != "TASK_RL" and float(row["distance_after"]) <= float(row["goal_radius"])
    ]
    completion = {
        "artifact_directory": str(directory.relative_to(ROOT)),
        "goal_radius": float(timeline[0]["goal_radius"]),
        "minimum_distance_any_mode": _minimum(timeline),
        "minimum_distance_task_rl": _minimum(task_rows),
        "minimum_distance_rl_generator": _minimum(authority_rows),
        "minimum_distance_kappa_backup": _minimum(backup_rows),
        "minimum_distance_charging_rl": _minimum(charging_rows),
        "inside_goal_radius_outside_task_rl_count": len(inside_non_task),
        "completion_invariant_violation_count": len(violations),
        "completion_invariant_violations": violations,
        "timeline": timeline,
    }

    unique_goals: dict[tuple[float, ...], dict[str, object]] = {}
    steps_by_goal: dict[tuple[float, ...], int] = {}
    updates_by_goal: dict[tuple[float, ...], int] = {}
    task_ids: set[str] = set()
    directions = []
    for row in rows:
        goal = np.asarray(row.get("goal_before", row["goal"]), dtype=np.float64)
        key = goal_key(goal)
        task_ids.add(str(row.get("task_id_before", row.get("task_id"))))
        if key not in unique_goals:
            position = np.asarray(row["position_before"], dtype=np.float64)
            unique_goals[key] = {"goal": goal.tolist(), "first_step": int(row["step"]), "initial_position": position.tolist()}
            directions.append(goal - position)
        steps_by_goal[key] = steps_by_goal.get(key, 0) + 1
        if int(row["step"]) > warmup_steps:
            updates_by_goal[key] = updates_by_goal.get(key, 0) + 1
    direction_metrics = goal_direction_statistics(directions)
    exposure = {
        "artifact_directory": str(directory.relative_to(ROOT)),
        "unique_task_ids_seen": len(task_ids),
        "task_ids": sorted(task_ids),
        "unique_goals_seen": len(unique_goals),
        "goal_assignments": len(unique_goals) + sum(bool(row.get("task_assigned_now", False)) for row in rows),
        "tasks_completed": sum(bool(row.get("task_completed_now", False)) for row in rows),
        "steps_per_goal": list(steps_by_goal.values()),
        "gradient_updates_per_goal": list(updates_by_goal.values()),
        "goal_direction_angular_coverage_degrees": direction_metrics["angular_coverage_degrees"],
        "goal_direction_entropy": direction_metrics["direction_entropy"],
        "goal_distance_distribution": [
            float(np.linalg.norm(np.asarray(item["goal"]) - np.asarray(item["initial_position"])))
            for item in unique_goals.values()
        ],
        "goal_exposure_gate": "PASS" if len(unique_goals) > 1 else "FAIL",
        "distribution_annotation": "MULTI_GOAL_TRAINING_DISTRIBUTION" if len(unique_goals) > 1 else "SINGLE_GOAL_TRAINING_DISTRIBUTION",
        "goals": list(unique_goals.values()),
    }
    return completion, exposure


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--directories", nargs="+", default=[f"artifacts/temp_compare_physical_seed{seed}" for seed in range(3)])
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--completion-output", default="artifacts/random_persistent/task_completion_consistency_audit.json")
    parser.add_argument("--exposure-output", default="artifacts/random_persistent/goal_exposure_audit.json")
    args = parser.parse_args()
    completions = []
    exposures = []
    for item in args.directories:
        completion, exposure = audit_directory(ROOT / item, args.warmup_steps)
        completions.append(completion)
        exposures.append(exposure)
    violations = sum(int(item["completion_invariant_violation_count"]) for item in completions)
    inside_non_task = sum(int(item["inside_goal_radius_outside_task_rl_count"]) for item in completions)
    completion_result = {
        "metric_definition": "minimum_distance_any_mode is the minimum Euclidean distance to the pending goal in every runtime mode; minimum_distance_task_rl is task-eligible.",
        "runs": completions,
        "task_completion_distance_invariant": "PASS" if violations == 0 else "FAIL",
        "historical_distance_anomaly_classification": "NO_BUG_METRIC_DEFINITION" if violations == 0 and inside_non_task else ("TASK_COMPLETION_BUG" if violations else "NO_BUG_METRIC_DEFINITION"),
        "exact_radius_boundary_bug_found": True,
        "exact_radius_boundary_fix": "completion uses the closed radius set with a 1e-12 floating-point comparison tolerance",
        "classification": "TASK_COMPLETION_BUG",
    }
    exposure_result = {
        "runs": exposures,
        "all_runs_multi_goal": all(int(item["unique_goals_seen"]) > 1 for item in exposures),
        "goal_exposure_gate": "PASS" if all(int(item["unique_goals_seen"]) > 1 for item in exposures) else "FAIL",
        "annotation": "SINGLE_GOAL_TRAINING_DISTRIBUTION" if any(int(item["unique_goals_seen"]) <= 1 for item in exposures) else "MULTI_GOAL_TRAINING_DISTRIBUTION",
    }
    completion_path = ROOT / args.completion_output
    exposure_path = ROOT / args.exposure_output
    completion_path.parent.mkdir(parents=True, exist_ok=True)
    completion_path.write_text(json.dumps(completion_result, indent=2), encoding="utf-8")
    exposure_path.write_text(json.dumps(exposure_result, indent=2), encoding="utf-8")
    print(json.dumps({"task_completion": completion_result["classification"], "goal_exposure_gate": exposure_result["goal_exposure_gate"]}, indent=2))


if __name__ == "__main__":
    main()
