#!/usr/bin/env python3
"""Formal persistent single-policy training entry point; not a unit-test command."""

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

from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from envs.certified_uav import make_persistent_uav_env
from persistent_generator_common import transition_from_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one continuous persistent Generator-SAC policy.")
    parser.add_argument("--scenario", default="persistent_open")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/persistent_generator_sac")
    args = parser.parse_args()

    environment = make_persistent_uav_env(f"{args.scenario}.json", seed=args.seed)
    observation, reset_info = environment.reset(seed=args.seed)
    context = reset_info["action_context"]
    config = GeneratorSACConfig(batch_size=args.batch_size, warmup_steps=args.warmup_steps)
    agent = PersistentGeneratorSAC(observation.size, config, seed=args.seed, device=args.device)
    rng = np.random.default_rng(args.seed)
    episode_id = 0
    updates: list[dict[str, float | int | str]] = []
    for step in range(args.steps):
        actor_u = rng.normal(size=3) if step < args.warmup_steps else agent.select_u(observation)
        next_observation, reward, terminated, truncated, info = environment.step(actor_u)
        next_context = None if terminated or truncated else environment._refresh_context()
        agent.observe(transition_from_cycle(
            observation, next_observation, actor_u, reward, terminated, truncated,
            episode_id, context, next_context, info,
        ))
        if len(agent.replay) >= args.batch_size and step >= args.warmup_steps:
            updates.append(agent.update())
        if terminated or truncated:
            episode_id += 1
            observation, reset_info = environment.reset(seed=args.seed + episode_id)
            context = reset_info["action_context"]
        else:
            observation = next_observation
            context = next_context

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), output / "checkpoint_latest.pt")
    summary = {
        "scenario": args.scenario,
        "seed": args.seed,
        "steps": args.steps,
        "gradient_steps": agent.gradient_steps,
        "metrics": environment.metric_snapshot(),
        "last_update": None if not updates else updates[-1],
        "synthetic_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
