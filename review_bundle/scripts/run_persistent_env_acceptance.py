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
    parser = argparse.ArgumentParser(description="Run a deterministic persistent software acceptance trajectory.")
    parser.add_argument("--scenario", default="persistent_open")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--output", default="artifacts/persistent/acceptance.json")
    args = parser.parse_args()
    env = make_persistent_uav_env(
        f"{args.scenario}.json",
        seed=args.seed,
        timing_mode="functional",
    )
    _, reset_info = env.reset(seed=args.seed)
    records = []
    for step in range(args.steps):
        _, reward, terminated, truncated, info = env.step(np.zeros(3, dtype=np.float64))
        records.append({
            "step": step,
            "reward": reward,
            "mode": info.get("persistent_mode"),
            "backup_triggered": info.get("backup_triggered"),
            "backup_reason": info.get("backup_reason"),
            "tasks_completed": info.get("persistent_metrics", {}).get("tasks_completed", 0),
            "energy": env.plant.state.energy,
            "command_source": info.get("command_source", "task_or_kappa"),
        })
        if terminated or truncated:
            break
    payload = {
        "scenario": args.scenario,
        "policy": "single_continuous_generator_sac",
        "reset": {key: str(value) for key, value in reset_info.items() if key != "action_context"},
        "metrics": env.metric_snapshot(),
        "records": records,
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
