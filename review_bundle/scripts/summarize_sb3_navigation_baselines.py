#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_algorithm(root: Path, algorithm: str) -> dict:
    rows = []
    for seed in (0, 1, 2):
        directory = root / f"seed{seed}"
        summary = load(directory / "summary.json")
        checkpoint_index = load(directory / "checkpoint_index.json")["checkpoints"]
        final = checkpoint_index[-1]
        stem = Path(final["checkpoint"]).stem
        primary_mode = "deterministic"
        evaluation = load(directory / f"{stem}_heldout_{primary_mode}.json")["aggregate"]
        rows.append({
            "seed": seed,
            "requested_timesteps": summary["requested_timesteps"],
            "actual_timesteps": summary["actual_timesteps"],
            "training_tasks_per_1000_steps": summary["tasks_per_1000_steps"],
            "training_last_10000_tasks_per_1000_steps": summary["last_10000_tasks_per_1000_steps"],
            "training_last_50000_tasks_per_1000_steps": summary["last_50000_tasks_per_1000_steps"],
            "heldout_tasks_per_1000_steps": evaluation["tasks_per_1000_steps"],
            "heldout_median_steps_per_goal": evaluation["goal_latency"]["median_steps_per_completed_goal"],
            "heldout_collision_rate": evaluation["collision_rate"],
            "heldout_boundary_lock_events": evaluation["boundary_lock_event_count"],
        })
    keys = (
        "training_tasks_per_1000_steps",
        "training_last_10000_tasks_per_1000_steps",
        "training_last_50000_tasks_per_1000_steps",
        "heldout_tasks_per_1000_steps",
        "heldout_median_steps_per_goal",
        "heldout_collision_rate",
    )
    aggregate = {}
    for key in keys:
        values = [row[key] for row in rows if row[key] is not None]
        aggregate[key] = {
            "mean": statistics.mean(values) if values else None,
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
        }
    return {
        "algorithm": algorithm,
        "experiment_class": "UNTUNED_STANDARD_SB3_BASELINES",
        "seed_results": rows,
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-root", type=Path, default=Path("artifacts/phase1_sb3_ppo_1m"))
    parser.add_argument("--ddpg-root", type=Path, default=Path("artifacts/phase1_sb3_ddpg_1m"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/ppo_ddpg_1m_summary.json"))
    args = parser.parse_args()
    payload = {
        "scope": "navigation algorithm baselines only; not energy/charging evidence",
        "solved_sac_reference": "DIRECT_SAC_BASELINE_SOLVED",
        "ppo": summarize_algorithm(args.ppo_root, "ppo"),
        "ddpg": summarize_algorithm(args.ddpg_root, "ddpg"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
