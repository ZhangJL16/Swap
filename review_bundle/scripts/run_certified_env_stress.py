#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.certified_uav import make_certified_uav_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--cycles-per-seed", type=int, default=50)
    args = parser.parse_args()
    counters = Counter()
    volumes, sigmas, conditions = [], [], []
    timings: dict[str, list[float]] = {}
    reasons = Counter()
    for seed in args.seeds:
        runtime = make_certified_uav_env(freeze_certificate_epoch=True)
        runtime.reset(seed=seed)
        rng = np.random.default_rng(seed)
        for _ in range(args.cycles_per_seed):
            _, _, terminated, truncated, info = runtime.step(rng.normal(size=3))
            counters["total_cycles"] += 1
            record = runtime.replay.records[-1]
            trace = info["telemetry"].action_trace
            counters["accepted_cycles" if record.accepted else "fallback_cycles"] += 1
            if record.zonotope_generators is not None:
                matrix = np.asarray(record.zonotope_generators)
                determinant = abs(float(np.linalg.det(matrix)))
                singular = np.linalg.svd(matrix, compute_uv=False)
                volumes.append(8.0 * determinant)
                sigmas.append(float(singular.min()))
                conditions.append(float(singular.max() / singular.min()))
                counters["generator_enabled_cycles"] += 1
                candidate = np.asarray(record.candidate_action)
                eta = np.asarray(record.squashed_eta)
                center = np.asarray(record.zonotope_center)
                if not np.allclose(candidate, center + matrix @ eta, atol=1e-12):
                    counters["candidate_membership_violations"] += 1
            if not np.array_equal(np.asarray(record.executed_action), trace.published):
                counters["executed_action_mismatch_count"] += 1
            if info["publication_count"] != 1:
                counters["one_shot_publication_violations"] += 1
            if not record.accepted:
                reasons[str(record.fallback_reason)] += 1
            telemetry = info["telemetry"]
            upper = runtime.calibration.energy.upper_cost(tuple(telemetry.state_before.velocity), tuple(trace.published))
            if telemetry.energy_cost > upper + 1e-12:
                counters["realized_cost_upper_bound_violations"] += 1
            counters["collision_count"] += int(telemetry.collision)
            counters["energy_depletion_count"] += int(info.get("failure_reason") == "energy_depleted")
            counters["terminal_success_count"] += int(telemetry.terminal_admissible)
            for name, elapsed in info["stage_timings"].items():
                timings.setdefault(name, []).append(elapsed)
            runtime.reset(seed=seed)
    result = dict(counters)
    total = counters["total_cycles"]
    result.update(
        acceptance_rate=counters["accepted_cycles"] / total,
        fallback_rate=counters["fallback_cycles"] / total,
        zonotope_volume={"mean": float(np.mean(volumes)), "min": float(np.min(volumes)), "max": float(np.max(volumes))},
        sigma_min={"mean": float(np.mean(sigmas)), "min": float(np.min(sigmas)), "max": float(np.max(sigmas))},
        condition_number={"mean": float(np.mean(conditions)), "min": float(np.min(conditions)), "max": float(np.max(conditions))},
        stage_timing_seconds={
            name: {
                "median": float(np.median(values)),
                "p95": float(np.percentile(values, 95)),
                "p99": float(np.percentile(values, 99)),
                "max": float(np.max(values)),
            }
            for name, values in timings.items()
        },
        fallback_reasons=dict(reasons),
        evidence_scope="synthetic software stress; not a proof",
    )
    required_zero = (
        "candidate_membership_violations",
        "executed_action_mismatch_count",
        "one_shot_publication_violations",
        "realized_cost_upper_bound_violations",
        "collision_count",
    )
    result["passed"] = all(counters[name] == 0 for name in required_zero) and total >= 500
    output = Path("artifacts/environment_acceptance/stress_summary.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
