#!/usr/bin/env python3
"""Evaluate checkpoints on identical held-out persistent random-goal streams."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.experiment_metrics import PersistentMetricAccumulator, metric_snapshot_delta
from cert_runtime.generator_sac import PersistentGeneratorSAC
from envs.certified_uav import make_random_persistent_uav_env


def _evaluate_checkpoint(environment, checkpoint: Path, seeds: list[int], steps_per_seed: int, device: str):
    initial_observation, _ = environment.reset(seed=seeds[0])
    agent = PersistentGeneratorSAC(initial_observation.size, seed=0, device=device)
    agent.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    aggregate = PersistentMetricAccumulator()
    seed_records = []
    for heldout_seed in seeds:
        observation, reset_info = environment.reset(seed=heldout_seed)
        previous_snapshot = None
        seed_metrics = PersistentMetricAccumulator()
        minimum_task_distance = float("inf")
        time_to_first_completion = None
        terminated = truncated = False
        for local_step in range(steps_per_seed):
            observation, reward, terminated, truncated, info = environment.step(
                agent.select_u(observation, deterministic=True)
            )
            snapshot = info["persistent_metrics"]
            delta = metric_snapshot_delta(previous_snapshot, snapshot)
            previous_snapshot = snapshot
            aggregate.observe(reward, info, delta)
            seed_metrics.observe(reward, info, delta)
            if info.get("persistent_mode_before") == "TASK_RL":
                distance = float(info.get("distance_to_goal_after", np.inf))
                if np.isfinite(distance):
                    minimum_task_distance = min(minimum_task_distance, distance)
            if info.get("task_completed_now") and time_to_first_completion is None:
                time_to_first_completion = local_step + 1
            if terminated or truncated:
                break
        metrics = seed_metrics.summary()
        seed_records.append({
            "heldout_seed": heldout_seed,
            "steps": metrics["total_steps"],
            "tasks_completed": metrics["tasks_completed"],
            "tasks_per_1000_steps": metrics["tasks_per_1000_steps"],
            "net_goal_progress": metrics["total_goal_progress"],
            "fraction_steps_progressing_to_goal": metrics["fraction_steps_progressing_to_goal"],
            "minimum_task_rl_distance": None if not np.isfinite(minimum_task_distance) else minimum_task_distance,
            "time_to_first_completion": time_to_first_completion,
            "unique_goals_completed": metrics["tasks_completed"],
            "voluntary_station_arrivals": metrics["voluntary_station_arrivals"],
            "backup_recovery_count": metrics["backup_recovery_count"],
            "charging_visits": metrics["charging_visits"],
            "collision_count": metrics["collision_count"],
            "energy_depletion_count": metrics["energy_depletion_count"],
            "uncertified_publication_count": metrics["uncertified_publication_count"],
            "invalid_kappa_fallback_count": metrics["invalid_kappa_fallback_count"],
            "fail_closed_steps": metrics["fail_closed_steps"],
            "accepted_into_kappa_only_count": metrics["accepted_into_kappa_only_count"],
            "sampled_start": np.asarray(reset_info["sampled_start"].position, dtype=float).tolist(),
            "initial_goal": np.asarray(reset_info["sampled_goal"], dtype=float).tolist(),
            "goal_sequence": [
                np.asarray(goal, dtype=float).tolist()
                for goal in getattr(environment.task_env.manager, "goal_sequence", ())
            ],
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
    aggregate_metrics = aggregate.summary()
    first_completion_times = [
        record["time_to_first_completion"]
        for record in seed_records
        if record["time_to_first_completion"] is not None
    ]
    minimum_distances = [
        record["minimum_task_rl_distance"]
        for record in seed_records
        if record["minimum_task_rl_distance"] is not None
    ]
    return {
        "checkpoint": str(checkpoint),
        "evaluation_protocol": "persistent_random_goal",
        "deterministic_actor": True,
        "heldout_seeds": seeds,
        "steps_per_seed": steps_per_seed,
        "aggregate_metrics": aggregate_metrics,
        "completion_run_fraction": float(np.mean([record["tasks_completed"] > 0 for record in seed_records])),
        "median_minimum_task_rl_distance": None if not minimum_distances else float(np.median(minimum_distances)),
        "mean_time_to_first_completion": None if not first_completion_times else float(np.mean(first_completion_times)),
        "seed_records": seed_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", required=True)
    parser.add_argument("--label", nargs="+")
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--steps-per-seed", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.steps_per_seed <= 0:
        raise ValueError("steps per held-out seed must be positive")
    labels = args.label or [Path(path).parent.name for path in args.checkpoint]
    if len(labels) != len(args.checkpoint):
        raise ValueError("labels and checkpoints must have equal length")

    environment = make_random_persistent_uav_env(f"{args.scenario}.json", seed=args.heldout_seeds[0])
    results = {}
    for label, checkpoint in zip(labels, args.checkpoint):
        results[label] = _evaluate_checkpoint(
            environment,
            ROOT / checkpoint,
            list(args.heldout_seeds),
            args.steps_per_seed,
            args.device,
        )
    payload = {
        "scenario": args.scenario,
        "evaluation_protocol": "persistent_random_goal",
        "periodic_exposure_resets": False,
        "deterministic_actor": True,
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
