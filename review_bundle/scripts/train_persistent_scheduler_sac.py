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

from envs.certified_uav import make_persistent_uav_env


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the event-level charging scheduler; low-level motion remains certified.")
    parser.add_argument("--scenario", default="persistent_energy_tight")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--output-dir", default="artifacts/persistent/scheduler_sac")
    args = parser.parse_args()
    env = make_persistent_uav_env(f"{args.scenario}.json", scheduler_name="scheduler_sac", seed=args.seed, timing_mode="functional")
    env.reset(seed=args.seed)
    updates = []
    for step in range(args.steps):
        _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float64))
        if len(env.scheduler.replay.records) >= env.scheduler.config.batch_size:
            updates.append({"step": step, **env.scheduler.update()})
        if terminated or truncated:
            env.reset(seed=args.seed + step + 1)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "actor": env.scheduler.actor.state_dict(),
        "critic_1": env.scheduler.critic_1.state_dict(),
        "critic_2": env.scheduler.critic_2.state_dict(),
        "log_alpha": env.scheduler.log_alpha.detach().cpu(),
    }, output / "checkpoint_latest.pt")
    (output / "metrics.json").write_text(json.dumps({"updates": updates, "metrics": env.metric_snapshot()}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
