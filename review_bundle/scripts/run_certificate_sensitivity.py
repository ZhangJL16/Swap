from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from envs.certified_uav.scenario import load_scenario
from envs.certified_uav.scenario_families.generator import _serialize
from experiments.agents import StatelessGeneratorPolicy
from experiments.metrics import write_csv
from experiments.runner import _evaluate, _certified_environment


def _scaled_payload(scenario: str, bound_scale: float, disturbance_fraction: float) -> dict:
    payload = _serialize(load_scenario(f"{scenario}.json"))
    payload["name"] = f"{scenario}-bounds-{bound_scale}-dist-{disturbance_fraction}"
    mission = payload["mission"]
    mission["synthetic_disturbance_fraction"] = disturbance_fraction
    bounds = mission.get("certificate_bounds", {})
    for key in ("position_residual_radius", "velocity_residual_radius", "wind_acceleration_radius"):
        if key in bounds:
            bounds[key] = (np.asarray(bounds[key], dtype=np.float64) * bound_scale).tolist()
    if "control_period_error" in bounds:
        bounds["control_period_error"] = float(bounds["control_period_error"] * bound_scale)
    configuration = payload["configuration_overrides"]
    configuration["tracking_error_bound"] = (
        np.asarray(configuration["tracking_error_bound"], dtype=np.float64) * bound_scale
    ).tolist()
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="mission_open")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--output", default="artifacts/paper/conservatism_robustness.csv")
    args = parser.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as directory:
        for bound_scale in (0.5, 0.75, 1.0):
            for disturbance in (0.0, 0.25, 0.5, 0.75, 1.0, 1.1):
                payload = _scaled_payload(args.scenario, bound_scale, disturbance)
                path = Path(directory) / f"scenario-{bound_scale}-{disturbance}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                runtime = _certified_environment(str(path), "task_oriented", "functional")
                runtime.reset(seed=0)
                gate = runtime.mission_provider.validation_report()["mission_certificate_gate"]
                if gate != "PASS":
                    rows.append({
                        "scenario": args.scenario, "bound_scale": bound_scale,
                        "disturbance_fraction": disturbance, "certificate_gate": gate,
                        "task_success": None, "return_success": None,
                        "outbound_intervention": None, "mean_zonotope_volume": None,
                        "classification": "out-of-contract diagnostic" if disturbance > 1.0 else "certificate construction failure",
                    })
                    continue
                episodes = _evaluate(
                    "center_only", StatelessGeneratorPolicy("center_only", 0), str(path),
                    0, args.episodes, "task_oriented", "functional",
                )
                volumes = []
                probe = _certified_environment(str(path), "task_oriented", "functional")
                probe.reset(seed=0)
                context = probe.action_context()
                if context.get("G") is not None:
                    volumes.append(8.0 * abs(float(np.linalg.det(context["G"]))))
                rows.append({
                    "scenario": args.scenario, "bound_scale": bound_scale,
                    "disturbance_fraction": disturbance, "certificate_gate": gate,
                    "task_success": float(np.mean([item["task_success"] for item in episodes])),
                    "return_success": float(np.mean([item["return_success"] for item in episodes])),
                    "outbound_intervention": float(np.mean([item["outbound_intervention_rate"] for item in episodes])),
                    "mean_zonotope_volume": float(np.mean(volumes)) if volumes else 0.0,
                    "classification": "synthetic sensitivity only; scaled bounds are not physical evidence",
                })
                print(json.dumps(rows[-1]), flush=True)
    write_csv(Path(args.output), rows)


if __name__ == "__main__":
    main()
