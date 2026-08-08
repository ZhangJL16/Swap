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
from envs.certified_uav import make_persistent_uav_env, make_random_persistent_uav_env


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
    observation, _ = environment.reset(seed=args.seed)
    agent = PersistentGeneratorSAC(observation.size, seed=args.seed, device=args.device)
    agent.load_state_dict(torch.load(args.checkpoint, map_location=args.device, weights_only=False))
    for _ in range(args.steps):
        observation, _, terminated, truncated, _ = environment.step(agent.select_u(observation, deterministic=True))
        if terminated or truncated:
            observation, _ = environment.reset(seed=args.seed)
    result = {
        "scenario": args.scenario,
        "legacy_fixed_graph": args.legacy_fixed_graph,
        "metrics": environment.metric_snapshot(),
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
