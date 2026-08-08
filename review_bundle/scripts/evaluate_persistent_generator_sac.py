#!/usr/bin/env python3
"""Evaluate a persistent single-policy checkpoint; this is a formal rollout script."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.generator_sac import PersistentGeneratorSAC
from cert_runtime.experiment_metrics import (
    PersistentMetricAccumulator,
    episode_record,
    metric_snapshot_delta,
)
from envs.certified_uav import make_persistent_uav_env, make_random_persistent_uav_env


def evaluation_episode_seed(base_seed: int, episode_id: int) -> int:
    return int(base_seed + episode_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--legacy-fixed-graph", action="store_true")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="artifacts/persistent_generator_sac/evaluation.json")
    args = parser.parse_args()
    factory = make_persistent_uav_env if args.legacy_fixed_graph else make_random_persistent_uav_env
    environment = factory(f"{args.scenario}.json", seed=args.seed)
    episode_id = 0
    episode_seed = evaluation_episode_seed(args.seed, episode_id)
    observation, reset_info = environment.reset(seed=episode_seed)
    agent = PersistentGeneratorSAC(observation.size, seed=args.seed, device=args.device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=False))
    run_metrics = PersistentMetricAccumulator()
    episode_metrics = PersistentMetricAccumulator()
    episode_records: list[dict[str, object]] = []
    previous_snapshot = None

    def goal_sequence():
        return [goal.copy() for goal in getattr(environment.task_env.manager, "goal_sequence", ())]

    for step in range(args.steps):
        observation, reward, terminated, truncated, info = environment.step(
            agent.select_u(observation, deterministic=True)
        )
        snapshot = info["persistent_metrics"]
        delta = metric_snapshot_delta(previous_snapshot, snapshot)
        previous_snapshot = snapshot
        run_metrics.observe(reward, info, delta)
        episode_metrics.observe(reward, info, delta)
        if terminated or truncated:
            episode_records.append(episode_record(
                episode_id,
                episode_seed,
                reset_info,
                episode_metrics,
                goal_sequence(),
                terminated=terminated,
                truncated=truncated,
                partial=False,
            ))
            episode_metrics = PersistentMetricAccumulator()
            previous_snapshot = None
            if step + 1 < args.steps:
                episode_id += 1
                episode_seed = evaluation_episode_seed(args.seed, episode_id)
                observation, reset_info = environment.reset(seed=episode_seed)
    if episode_metrics.total_steps:
        episode_records.append(episode_record(
            episode_id,
            episode_seed,
            reset_info,
            episode_metrics,
            goal_sequence(),
            terminated=False,
            truncated=False,
            partial=True,
        ))
    aggregate = run_metrics.summary()
    result = {
        "scenario": args.scenario,
        "legacy_fixed_graph": args.legacy_fixed_graph,
        "base_seed": args.seed,
        "steps": args.steps,
        "episodes": len(episode_records),
        "episode_seeds": [record["episode_seed"] for record in episode_records],
        "aggregate_metrics": aggregate,
        "episode_metrics": episode_records,
        "sampled_start_positions": [record["start_position"] for record in episode_records],
        "sampled_goal_sequences": [record["goal_sequence"] for record in episode_records],
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
