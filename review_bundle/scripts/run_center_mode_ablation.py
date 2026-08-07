from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.agents import StatelessGeneratorPolicy
from experiments.metrics import write_csv
from experiments.runner import _evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=["mission_open", "mission_obstacle"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--output", default="artifacts/paper/center_ablation.csv")
    args = parser.parse_args()
    rows = []
    for scenario in args.scenarios:
        for mode in ("task_oriented", "zero", "braking"):
            for seed in args.seeds:
                episodes = _evaluate(
                    "center_only", StatelessGeneratorPolicy("center_only", seed), scenario,
                    seed * 1000, args.episodes, mode, "functional",
                )
                row = {
                    "scenario": scenario,
                    "center_mode": mode,
                    "seed": seed,
                    "episodes": len(episodes),
                    "task_success": sum(item["task_success"] for item in episodes) / len(episodes),
                    "return_success": sum(item["return_success"] for item in episodes) / len(episodes),
                    "collision": sum(item["collision"] for item in episodes) / len(episodes),
                    "minimum_distance_to_task": sum(item["minimum_distance_to_task"] for item in episodes) / len(episodes),
                    "outbound_intervention": sum(item["outbound_intervention_rate"] for item in episodes) / len(episodes),
                    "path_length": sum(item["total_path_length"] for item in episodes) / len(episodes),
                    "energy_consumed": sum(item["total_energy_consumed"] for item in episodes) / len(episodes),
                    "certificate_source": "same complete-set verifier",
                    "evidence_scope": "synthetic empirical center ablation",
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    write_csv(Path(args.output), rows)


if __name__ == "__main__":
    main()
