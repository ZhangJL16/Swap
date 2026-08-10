#!/usr/bin/env python3
"""Reconstruct exact batch goal telemetry for pre-metadata controlled runs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.goal_exposure import goal_direction_statistics, goal_key


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, records) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


def reconstruct(directory: Path) -> None:
    summary_path = directory / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trajectory = _read_jsonl(directory / "trajectory_events.jsonl")
    curve_path = directory / "learning_curve.jsonl"
    curve = _read_jsonl(curve_path)
    batch_size = 64
    warmup_steps = 300
    rng = np.random.default_rng(int(summary["seed"]))
    metrics_by_step = {}
    for step_number in range(1, len(trajectory) + 1):
        if step_number <= warmup_steps or step_number < batch_size:
            continue
        selected = rng.choice(step_number, size=batch_size, replace=False)
        goals = [np.asarray(trajectory[int(index)]["goal_before"], dtype=np.float64) for index in selected]
        directions = [
            np.asarray(trajectory[int(index)]["goal_before"], dtype=np.float64)
            - np.asarray(trajectory[int(index)]["position_before"], dtype=np.float64)
            for index in selected
        ]
        metrics_by_step[step_number] = {
            "batch_unique_goal_count": len({goal_key(goal) for goal in goals}),
            "batch_goal_direction_entropy": goal_direction_statistics(directions)["direction_entropy"],
        }
    for record in curve:
        metrics = metrics_by_step.get(int(record["step"]))
        if metrics is not None:
            record.update(metrics)
            record["batch_goal_telemetry_reconstructed_from_trajectory"] = True
    _write_jsonl(curve_path, curve)
    if summary.get("last_update") is not None:
        summary["last_update"].update(metrics_by_step[len(trajectory)])
    summary["batch_goal_telemetry_reconstructed_from_trajectory"] = True
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    for protocol in ("persistent", "exposure"):
        for seed in range(3):
            reconstruct(ROOT / f"artifacts/multigoal_5k/{protocol}_seed{seed}")


if __name__ == "__main__":
    main()
