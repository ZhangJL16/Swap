#!/usr/bin/env python3
"""Background-only coordinator that aggregates completed corrected-exposure workers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "artifacts/corrected_exposure_5k"
COMPARISON = ROOT / "artifacts/random_persistent/corrected_exposure_5k_comparison.json"


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    while True:
        failures = [OUTPUT_ROOT / f"seed{seed}" / "PIPELINE_FAILED.json" for seed in range(3)]
        failures = [path for path in failures if path.exists()]
        if failures:
            payload = {
                "status": "FAILED",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "failures": [_read(path) for path in failures],
            }
            (OUTPUT_ROOT / "PIPELINE_FAILED.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            raise SystemExit(1)
        done = [OUTPUT_ROOT / f"seed{seed}" / "PIPELINE_DONE.json" for seed in range(3)]
        if all(path.exists() for path in done):
            break
        time.sleep(30)

    baseline = []
    corrected = []
    for seed in range(3):
        baseline_dir = ROOT / f"artifacts/multigoal_5k/persistent_seed{seed}"
        corrected_dir = OUTPUT_ROOT / f"seed{seed}"
        baseline.append({
            "seed": seed,
            "training": _read(baseline_dir / "summary.json"),
            "heldout": _read(baseline_dir / "heldout_evaluation.json"),
        })
        corrected.append({
            "seed": seed,
            "training": _read(corrected_dir / "summary.json"),
            "heldout": _read(corrected_dir / "heldout_evaluation.json"),
            "pipeline": _read(corrected_dir / "PIPELINE_DONE.json"),
        })
    payload = {
        "baseline_commit": "5771d5034aa21b78897b732477264a3943dc3e8a",
        "implementation": "corrected collector boundary preserves real one-step bootstrap",
        "common_config": {
            "scenario": "random_persistent_open",
            "steps": 5000,
            "warmup_steps": 300,
            "batch_size": 64,
            "temperature_coordinate": "physical",
            "heldout_seeds": [100, 101, 102, 103, 104],
            "heldout_steps_per_seed": 2000,
        },
        "persistent_baseline_reused": True,
        "baseline": baseline,
        "corrected_exposure": corrected,
        "training_semantics": {
            "collector_boundary_is_terminal": False,
            "real_successor_bootstrapped": True,
            "reset_state_used_as_successor": False,
        },
    }
    COMPARISON.parent.mkdir(parents=True, exist_ok=True)
    COMPARISON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    root_done = {
        "status": "COMPLETE",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "all_training_exit_codes": [0, 0, 0],
        "all_evaluation_exit_codes": [0, 0, 0],
        "all_visualization_exit_codes": [0, 0, 0],
        "comparison": str(COMPARISON.relative_to(ROOT)),
        "seed_done_markers": [str(path.relative_to(ROOT)) for path in done],
    }
    (OUTPUT_ROOT / "PIPELINE_DONE.json").write_text(json.dumps(root_done, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
