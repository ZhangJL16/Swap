from __future__ import annotations

import argparse
import csv
import gc
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_certified_uav_env


def run(mode: str, seed: int, scenario: str) -> dict:
    environment = make_certified_uav_env(f"{scenario}.json", generator_center_mode=mode)
    environment.reset(seed=seed)
    minimum_distance = float("inf")
    volumes: list[float] = []
    centers: set[tuple[float, ...]] = set()
    accepted = fallback = 0
    for step in range(1, environment.config.episode_limit + 1):
        context = environment.action_context()
        if context.get("G") is not None:
            volumes.append(8.0 * abs(float(np.linalg.det(context["G"]))))
            centers.add(tuple(float(value) for value in np.round(context["c"], 8)))
        _, _, terminated, truncated, info = environment.step(np.zeros(3))
        accepted += int(info["accepted"])
        fallback += int(not info["accepted"])
        minimum_distance = min(minimum_distance, float(np.linalg.norm(environment.plant.state.position - environment.scenario.task_goal)))
        if terminated or truncated:
            break
    return {
        "mode": mode, "seed": seed, "scenario": scenario, "episode_length": step,
        "task_success": int(info["task_completed"]), "return_success": int(info["terminal_return_success"]),
        "collision": int(info.get("failure_reason") == "collision"), "minimum_distance_to_task": minimum_distance,
        "fallback_rate": fallback / step, "acceptance_rate": accepted / step,
        "mean_zonotope_volume": float(np.mean(volumes)) if volumes else 0.0,
        "unique_centers": len(centers),
        "evidence_scope": "deterministic synthetic center-construction ablation",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["braking", "task_oriented"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--scenario", default="mission_open")
    parser.add_argument("--output", default="artifacts/generator_center_ablation/results.csv")
    args = parser.parse_args()
    rows = []
    for mode in args.modes:
        for seed in args.seeds:
            rows.append(run(mode, seed, args.scenario))
            gc.collect()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    output.with_suffix(".json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
