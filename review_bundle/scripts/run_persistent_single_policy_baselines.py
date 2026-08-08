#!/usr/bin/env python3
"""Formal rollout entry point for continuous-policy persistent baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_persistent_uav_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="persistent_open")
    parser.add_argument("--methods", nargs="+", default=["zero_latent", "random_latent"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--output", default="artifacts/persistent_single_policy/baselines.json")
    args = parser.parse_args()
    results = []
    for method in args.methods:
        environment = make_persistent_uav_env(f"{args.scenario}.json", seed=args.seed)
        environment.reset(seed=args.seed)
        rng = np.random.default_rng(args.seed)
        for _ in range(args.steps):
            actor_u = np.zeros(3) if method == "zero_latent" else rng.normal(size=3)
            _, _, terminated, truncated, _ = environment.step(actor_u)
            if terminated or truncated:
                environment.reset(seed=args.seed)
        results.append({"method": method, "metrics": environment.metric_snapshot(), "synthetic_only": True})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
