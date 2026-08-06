from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate import aggregate_results
from experiments.runner import ExperimentConfig, run_experiment
from experiments.registry import METHODS, SCENARIOS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts/comparison")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = []
    for scenario in args.scenarios:
        for method in args.methods:
            for seed in args.seeds:
                result = run_experiment(ExperimentConfig(method, scenario, seed, args.steps, args.warmup_steps, args.batch_size, output_root=args.output_dir, device=args.device))
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    aggregate = aggregate_results(args.output_dir)
    print(json.dumps({"runs": len(results), "aggregate_rows": aggregate}, indent=2))


if __name__ == "__main__":
    main()
