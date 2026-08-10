#!/usr/bin/env python3
"""Fail-fast train/evaluate/render worker for one corrected-exposure seed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/zjl/mappo/.venv/bin/python")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _run(command: list[str]) -> None:
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _evaluate_and_render(checkpoint: Path, output_dir: Path, label: str) -> dict[str, str]:
    heldout = output_dir / "heldout_evaluation.json"
    trajectory_dir = output_dir / "eval_seed100"
    trajectory = trajectory_dir / "trajectory_events.jsonl"
    evaluation = trajectory_dir / "evaluation.json"
    gif = output_dir / "eval_seed100.gif"
    png = output_dir / "trajectory_eval_seed100.png"
    _run([
        str(PYTHON), "scripts/evaluate_multigoal_checkpoint_suite.py",
        "--checkpoint", _relative(checkpoint),
        "--label", label,
        "--scenario", "random_persistent_open",
        "--heldout-seeds", "100", "101", "102", "103", "104",
        "--steps-per-seed", "2000",
        "--output", _relative(heldout),
    ])
    _run([
        str(PYTHON), "scripts/evaluate_persistent_checkpoint_trajectory.py",
        "--checkpoint", _relative(checkpoint),
        "--scenario", "random_persistent_open",
        "--seed", "100",
        "--steps", "2000",
        "--output-dir", _relative(trajectory_dir),
    ])
    _run([
        str(PYTHON), "scripts/render_persistent_trajectory.py",
        "--trajectory", _relative(trajectory),
        "--scenario", "random_persistent_open",
        "--png", _relative(png),
        "--gif", _relative(gif),
        "--frame-stride", "10",
        "--fps", "12",
    ])
    return {
        "heldout_evaluation": _relative(heldout),
        "seed100_evaluation": _relative(evaluation),
        "seed100_trajectory": _relative(trajectory),
        "seed100_png": _relative(png),
        "seed100_gif": _relative(gif),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", default="artifacts/corrected_exposure_5k")
    args = parser.parse_args()
    output_dir = ROOT / args.output_root / f"seed{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    failed = output_dir / "PIPELINE_FAILED.json"
    done = output_dir / "PIPELINE_DONE.json"
    for marker in (failed, done):
        marker.unlink(missing_ok=True)
    try:
        train_command = [
            str(PYTHON), "scripts/train_persistent_generator_sac.py",
            "--scenario", "random_persistent_open",
            "--seed", str(args.seed),
            "--steps", "5000",
            "--warmup-steps", "300",
            "--batch-size", "64",
            "--temperature-coordinate", "physical",
            "--goal-exposure-reset-steps", "250",
            "--output-dir", _relative(output_dir),
        ]
        (output_dir / "pipeline_commands.json").write_text(
            json.dumps({"train": train_command}, indent=2), encoding="utf-8"
        )
        _run(train_command)
        checkpoint = output_dir / "checkpoint_latest.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        artifacts = _evaluate_and_render(checkpoint, output_dir, f"corrected_exposure_seed{args.seed}")
        if args.seed == 0:
            baseline_dir = ROOT / "artifacts/multigoal_5k/persistent_seed0"
            baseline_checkpoint = baseline_dir / "checkpoint_latest.pt"
            baseline_artifacts = _evaluate_and_render(baseline_checkpoint, baseline_dir, "persistent_seed0")
        else:
            baseline_dir = ROOT / f"artifacts/multigoal_5k/persistent_seed{args.seed}"
            baseline_checkpoint = baseline_dir / "checkpoint_latest.pt"
            baseline_heldout = baseline_dir / "heldout_evaluation.json"
            _run([
                str(PYTHON), "scripts/evaluate_multigoal_checkpoint_suite.py",
                "--checkpoint", _relative(baseline_checkpoint),
                "--label", f"persistent_seed{args.seed}",
                "--scenario", "random_persistent_open",
                "--heldout-seeds", "100", "101", "102", "103", "104",
                "--steps-per-seed", "2000",
                "--output", _relative(baseline_heldout),
            ])
            baseline_artifacts = {"heldout_evaluation": _relative(baseline_heldout)}
        payload = {
            "seed": args.seed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "training_exit_code": 0,
            "evaluation_exit_code": 0,
            "visualization_exit_code": 0,
            "artifacts": artifacts,
            "baseline_artifacts": baseline_artifacts,
        }
        done.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as error:
        failed.write_text(json.dumps({
            "seed": args.seed,
            "failed_at": datetime.now(timezone.utc).isoformat(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        }, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
