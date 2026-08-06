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
    parser.add_argument("method")
    parser.add_argument("scenario")
    parser.add_argument("seed", type=int)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts/comparison")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    result = run_experiment(ExperimentConfig(args.method, args.scenario, args.seed, args.steps, args.warmup_steps, args.batch_size, output_root=args.output_dir, device=args.device))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
