from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.multiscenario import MultiScenarioTrainingConfig, train_multiscenario


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario-index", default="artifacts/scenario_families/scenario_index.json")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/multiscenario_training")
    args = parser.parse_args()
    for seed in args.seeds:
        result = train_multiscenario(MultiScenarioTrainingConfig(
            scenario_index=args.scenario_index,
            total_steps=args.steps,
            seed=seed,
            warmup_steps=args.warmup_steps,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=args.output_dir,
        ))
        print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
