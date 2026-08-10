#!/usr/bin/env python3
"""Deterministically evaluate one persistent checkpoint and record its trajectory."""

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

from cert_runtime.experiment_metrics import PersistentMetricAccumulator, metric_snapshot_delta, write_jsonl
from cert_runtime.generator_sac import PersistentGeneratorSAC
from envs.certified_uav import make_random_persistent_uav_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    environment = make_random_persistent_uav_env(f"{args.scenario}.json", seed=args.seed)
    observation, reset_info = environment.reset(seed=args.seed)
    agent = PersistentGeneratorSAC(observation.size, seed=0, device=args.device)
    agent.load_state_dict(torch.load(ROOT / args.checkpoint, map_location=args.device, weights_only=False))
    metrics = PersistentMetricAccumulator()
    previous_snapshot = None
    records: list[dict[str, object]] = []
    minimum_task_distance = float("inf")
    time_to_first_completion = None

    for local_step in range(args.steps):
        active_task = environment.task_env.manager.current_task
        goal_before = np.asarray(environment.task_env.manager.navigation_target, dtype=float).copy()
        task_id_before = getattr(active_task, "task_id", getattr(active_task, "edge_id", None))
        mode_before = environment.task_env.mode.name
        position_before = environment.plant.state.position.copy()
        actor_u = agent.select_u(observation, deterministic=True)
        observation, reward, terminated, truncated, info = environment.step(actor_u)
        telemetry = info["telemetry"]
        snapshot = info["persistent_metrics"]
        delta = metric_snapshot_delta(previous_snapshot, snapshot)
        previous_snapshot = snapshot
        metrics.observe(reward, info, delta)
        if mode_before == "TASK_RL" and np.isfinite(info.get("distance_to_goal_after", np.nan)):
            minimum_task_distance = min(minimum_task_distance, float(info["distance_to_goal_after"]))
        if info.get("task_completed_now") and time_to_first_completion is None:
            time_to_first_completion = local_step + 1
        records.append({
            "step": local_step + 1,
            "episode_id": 0,
            "episode_seed": args.seed,
            "task_id": info.get("task_id"),
            "task_id_before": task_id_before,
            "goal": None if info.get("current_goal") is None else np.asarray(info["current_goal"], dtype=float).tolist(),
            "goal_before": goal_before.tolist(),
            "position_before": np.asarray(position_before, dtype=float).tolist(),
            "position": np.asarray(telemetry.state_after.position, dtype=float).tolist(),
            "velocity": np.asarray(telemetry.state_after.velocity, dtype=float).tolist(),
            "energy": float(telemetry.state_after.energy),
            "energy_margin": float(info.get("energy_margin", np.nan)),
            "distance_to_goal_before": float(info.get("distance_to_goal_before", np.nan)),
            "distance_to_goal_after": float(info.get("distance_to_goal_after", np.nan)),
            "goal_progress": float(info.get("goal_progress", 0.0)),
            "reward": float(reward),
            "task_completed_now": bool(info.get("task_completed_now", False)),
            "tasks_completed": int(info.get("tasks_completed", 0)),
            "persistent_mode_before": mode_before,
            "persistent_mode": info.get("persistent_mode"),
            "execution_authority": info.get("execution_authority"),
            "charging": bool(info.get("charging", False)),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        })
        if terminated or truncated:
            break

    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "trajectory_events.jsonl"
    write_jsonl(trajectory_path, records)
    aggregate = metrics.summary()
    payload = {
        "checkpoint": args.checkpoint,
        "scenario": args.scenario,
        "evaluation_protocol": "persistent_random_goal",
        "periodic_exposure_resets": False,
        "deterministic_actor": True,
        "seed": args.seed,
        "steps": aggregate["total_steps"],
        "aggregate_metrics": aggregate,
        "minimum_task_rl_distance": None if not np.isfinite(minimum_task_distance) else minimum_task_distance,
        "time_to_first_completion": time_to_first_completion,
        "sampled_start": np.asarray(reset_info["sampled_start"].position, dtype=float).tolist(),
        "initial_goal": np.asarray(reset_info["sampled_goal"], dtype=float).tolist(),
        "trajectory_events": str(trajectory_path.relative_to(ROOT)),
    }
    (output_dir / "evaluation.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
