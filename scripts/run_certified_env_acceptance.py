#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.certified_uav.acceptance import run_acceptance_cycle, scenario_matrix


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="open_corridor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--matrix", action="store_true")
    args = parser.parse_args()
    output_dir = Path("artifacts/environment_acceptance")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace, _ = run_acceptance_cycle(args.scenario, args.seed)
    (output_dir / f"{args.scenario}_trace.json").write_text(json.dumps(trace, indent=2, sort_keys=True))
    summary = (
        f"scenario={args.scenario}\naccepted={trace['accepted']}\n"
        f"fallback_reason={trace['fallback_reason']}\nactor_called={trace['actor_called']}\n"
        f"published_once={trace['published_once']}\nelapsed_seconds={trace['elapsed_time']:.6f}\n"
        "evidence_scope=synthetic-software-only\n"
    )
    (output_dir / f"{args.scenario}_summary.txt").write_text(summary)
    result = {"trace": trace}
    if args.matrix:
        rows = scenario_matrix(args.seed)
        (output_dir / "scenario_matrix.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
        result["scenario_matrix"] = rows
        if not all(row["pass"] for row in rows):
            raise SystemExit(2)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
