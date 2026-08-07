from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.runner import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="mission_open")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts/comparison")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    results = [run_experiment(ExperimentConfig("generator_sac", args.scenario, seed, args.steps, args.warmup_steps, args.batch_size, output_root=args.output_dir, device=args.device)) for seed in args.seeds]
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
