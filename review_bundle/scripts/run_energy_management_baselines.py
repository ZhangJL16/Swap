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

from envs.certified_uav import make_persistent_uav_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Run persistent energy-management baselines.")
    parser.add_argument("--methods", nargs="+", default=["reserve_only", "fixed_threshold_30", "fixed_threshold_50", "full_charge"])
    parser.add_argument("--scenarios", nargs="+", default=["persistent_open", "persistent_obstacle", "persistent_energy_tight"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/persistent/energy_management_baselines.json")
    args = parser.parse_args()
    rows = []
    for scenario in args.scenarios:
        for method in args.methods:
            for seed in args.seeds:
                env = make_persistent_uav_env(
                    f"{scenario}.json",
                    energy_management_name=method,
                    seed=seed,
                    timing_mode="functional",
                )
                env.reset(seed=seed)
                for _ in range(args.steps):
                    _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float64))
                    if terminated or truncated:
                        break
                rows.append({"scenario": scenario, "method": method, "seed": seed, **env.metric_snapshot()})
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} runs to {output}")


if __name__ == "__main__":
    main()
