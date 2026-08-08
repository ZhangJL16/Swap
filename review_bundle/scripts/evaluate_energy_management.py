#!/usr/bin/env python3
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

from envs.certified_uav import make_persistent_energy_management_ablation_env


def main() -> None:
    parser = argparse.ArgumentParser(description="ABLATION ONLY: evaluate the deprecated hierarchical energy-management policy.")
    parser.add_argument("--scenario", default="persistent_energy_tight")
    parser.add_argument("--policy", default="energy_management_sac")
    parser.add_argument("--checkpoint")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/persistent/energy_management_evaluation.json")
    args = parser.parse_args()
    rows = []
    for seed in args.seeds:
        env = make_persistent_energy_management_ablation_env(
            f"{args.scenario}.json",
            energy_management_name=args.policy,
            seed=seed,
            timing_mode="functional",
            deterministic=True,
        )
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
            env.energy_management_policy.actor.load_state_dict(checkpoint["actor"])
        env.reset(seed=seed)
        for _ in range(args.steps):
            _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float64))
            if terminated or truncated:
                break
        rows.append({"seed": seed, **env.metric_snapshot()})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
