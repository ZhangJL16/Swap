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

from envs.certified_uav import make_random_persistent_uav_env


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _audit(path: Path, reward_config) -> dict[str, object]:
    rows = _rows(path)
    summary_path = path.with_name("summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    aggregate = summary.get("aggregate_metrics", {})
    components = {
        "goal_progress_reward": 0.0,
        "task_completion_reward": 0.0,
        "elapsed_time_cost": 0.0,
        "energy_cost": 0.0,
        "charging_dwell_cost": 0.0,
    }
    old_backup_steps = 0
    event_steps: list[int] = []
    previous_authority: str | None = None
    previous_episode: int | None = None
    for row in rows:
        recorded = row.get("reward_components")
        if isinstance(recorded, dict):
            components["goal_progress_reward"] += float(recorded.get("goal_progress_reward", 0.0))
            components["task_completion_reward"] += float(recorded.get("task_completion_reward", 0.0))
            components["elapsed_time_cost"] += float(recorded.get("elapsed_time_cost", 0.0))
            components["energy_cost"] += float(recorded.get("energy_cost", recorded.get("flight_energy_cost", 0.0)))
            components["charging_dwell_cost"] += float(recorded.get("charging_dwell_cost", recorded.get("charging_cost", 0.0)))
        authority = str(row.get("execution_authority", ""))
        episode = int(row.get("episode_id", 0))
        if episode != previous_episode:
            previous_authority = None
        if bool(row.get("backup_triggered", False)):
            old_backup_steps += 1
        if authority == "KAPPA_BACKUP" and previous_authority != "KAPPA_BACKUP":
            event_steps.append(int(row.get("step", 0)))
        previous_authority = authority
        previous_episode = episode
    if not any(abs(value) > 0.0 for value in components.values()):
        components["goal_progress_reward"] = reward_config.goal_progress_weight * sum(
            float(row.get("goal_progress", 0.0) or 0.0) for row in rows
        )
        components["task_completion_reward"] = reward_config.task_completion_reward * sum(
            bool(row.get("task_completed_now", False)) for row in rows
        )
        components["elapsed_time_cost"] = -reward_config.elapsed_time_cost * len(rows)
        components["energy_cost"] = -reward_config.flight_energy_cost * float(aggregate.get("energy_consumed", 0.0))
        components["charging_dwell_cost"] = -reward_config.charging_dwell_cost * float(aggregate.get("charging_steps", 0.0))
    backup_cost = float(reward_config.backup_intervention_cost)
    old_backup = -backup_cost * old_backup_steps
    new_backup = -backup_cost * len(event_steps)
    old_total = sum(components.values()) + old_backup
    new_total = sum(components.values()) + new_backup
    return {
        "trajectory": str(path.relative_to(ROOT)),
        "steps": len(rows),
        "old_per_step_backup_penalty_steps": old_backup_steps,
        "new_backup_intervention_events": len(event_steps),
        "backup_event_steps": event_steps,
        "old_per_step_backup_contribution": old_backup,
        "new_event_only_backup_contribution": new_backup,
        **components,
        "old_reconstructed_total": old_total,
        "new_reconstructed_total": new_total,
        "reward_scale_change": new_total - old_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectories",
        nargs="+",
        default=[f"artifacts/task_authority_smoke_open_seed{seed}/trajectory_events.jsonl" for seed in range(3)],
    )
    parser.add_argument("--output", default="artifacts/random_persistent/backup_reward_semantics_audit.json")
    args = parser.parse_args()
    environment = make_random_persistent_uav_env("random_persistent_open.json", seed=0)
    reward_config = environment.task_env.reward_config
    cost = float(reward_config.backup_intervention_cost)
    runs = [_audit(ROOT / path, reward_config) for path in args.trajectories]
    numeric = (
        "old_per_step_backup_contribution",
        "new_event_only_backup_contribution",
        "goal_progress_reward",
        "task_completion_reward",
        "elapsed_time_cost",
        "energy_cost",
        "charging_dwell_cost",
    )
    result = {
        "definition": "backup_intervention_cost is charged once on entry into KAPPA_BACKUP",
        "backup_intervention_cost": cost,
        "runs": runs,
        "aggregate_mean": {name: float(np.mean([float(run[name]) for run in runs])) for name in numeric},
        "reconstructed_only": True,
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
