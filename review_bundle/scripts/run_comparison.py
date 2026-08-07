from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.aggregate import aggregate_results
from experiments.runner import ExperimentConfig, run_experiment
from experiments.registry import METHODS, SCENARIOS
import torch


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
    parser.add_argument("--evaluation-episodes", type=int, default=20)
    parser.add_argument("--generator-center-mode", choices=("braking", "zero", "task_oriented", "max_volume"), default="task_oriented")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--skip-aggregate", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    gc.disable()
    results = []
    for scenario in args.scenarios:
        for method in args.methods:
            for seed in args.seeds:
                result = run_experiment(ExperimentConfig(
                    method, scenario, seed, args.steps, args.warmup_steps, args.batch_size,
                    output_root=args.output_dir, device=args.device,
                    evaluation_episodes=args.evaluation_episodes,
                    generator_center_mode=args.generator_center_mode,
                ))
                results.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
                gc.enable()
                gc.collect()
                gc.disable()
    aggregate = [] if args.skip_aggregate else aggregate_results(args.output_dir)
    gc.enable()
    print(json.dumps({"runs": len(results), "aggregate_rows": aggregate}, indent=2))


if __name__ == "__main__":
    main()
