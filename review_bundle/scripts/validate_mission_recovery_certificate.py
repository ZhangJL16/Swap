from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.dynamics import integrate_double_integrator
from envs.certified_uav.state import UAVPhysicalState


SCENARIOS = ("mission_open", "mission_obstacle", "mission_narrow", "mission_energy_tight")


def validate_scenario(scenario: str, sampled_chains: int) -> dict:
    runtime = make_certified_uav_env(f"{scenario}.json")
    provider = runtime.mission_provider
    report = provider.validation_report()
    collisions = 0
    terminal_rollouts = 0
    level_failures = 0
    sampled_state_successor_failures = 0
    sampled_energy_upper_violations = 0
    rng = np.random.default_rng(0)
    selected = np.linspace(0, len(provider.manifest.chains) - 1, min(sampled_chains, len(provider.manifest.chains)), dtype=int)
    for chain_index in selected:
        chain = provider.manifest.chains[int(chain_index)]
        position = np.asarray(chain.root.reference_position, dtype=np.float64)
        velocity = np.asarray(chain.root.reference_velocity, dtype=np.float64)
        for cell, successor in zip(chain.cells[:-1], chain.cells[1:]):
            action = np.asarray(cell.reference_action, dtype=np.float64)
            next_position, next_velocity = integrate_double_integrator(position, velocity, action, runtime.config.dt)
            collisions += int(runtime.plant.world.swept_collision(position, next_position, runtime.config.body_radius))
            if successor.level >= cell.level:
                level_failures += 1
            position, velocity = next_position, next_velocity
        stride = max(1, (len(chain.cells) - 1) // 20)
        for cell_index in range(0, len(chain.cells) - 1, stride):
            cell, successor = chain.cells[cell_index], chain.cells[cell_index + 1]
            sampled_position = np.asarray(cell.reference_position, dtype=np.float64).copy()
            sampled_velocity = np.asarray(cell.reference_velocity, dtype=np.float64).copy()
            for axis, radius in enumerate(cell.ellipsoid_radii):
                direction = rng.normal(size=2)
                direction /= max(np.linalg.norm(direction), 1e-12)
                error = np.linalg.solve(np.linalg.cholesky(provider._matrix).T, direction * radius * rng.uniform(0.0, 0.9))
                sampled_position[axis] += error[0]
                sampled_velocity[axis] += error[1]
            action = (
                np.asarray(cell.reference_action)
                - provider.position_gain * (sampled_position - np.asarray(cell.reference_position))
                - provider.velocity_gain * (sampled_velocity - np.asarray(cell.reference_velocity))
            )
            next_position, next_velocity = integrate_double_integrator(sampled_position, sampled_velocity, action, runtime.config.dt)
            for axis in range(3):
                error = np.array((next_position[axis] - successor.reference_position[axis], next_velocity[axis] - successor.reference_velocity[axis]))
                sampled_state_successor_failures += int(float(error @ provider._matrix @ error) > successor.ellipsoid_radii[axis] ** 2 + 1e-10)
            realized = runtime.plant.energy_model.realized_cost(
                UAVPhysicalState(sampled_position, sampled_velocity, runtime.config.initial_energy, 0.0),
                action,
                runtime.config.dt,
            )
            sampled_energy_upper_violations += int(realized > cell.one_step_energy_upper + 1e-12)
        terminal_rollouts += int(
            np.all(position >= runtime.scenario.terminal.position_low)
            and np.all(position <= runtime.scenario.terminal.position_high)
            and np.all(np.abs(velocity) <= runtime.scenario.terminal.velocity_abs_max)
        )
    report.update(
        {
            "sampled_chains": int(len(selected)),
            "sampled_collision_count": collisions,
            "sampled_level_failures": level_failures,
            "sampled_terminal_arrivals": terminal_rollouts,
            "sampled_terminal_expected": int(len(selected)),
            "sampled_state_successor_failures": sampled_state_successor_failures,
            "sampled_energy_upper_violations": sampled_energy_upper_violations,
        }
    )
    report["mission_certificate_gate"] = (
        "PASS"
        if report["mission_certificate_gate"] == "PASS"
        and collisions == 0
        and level_failures == 0
        and terminal_rollouts == len(selected)
        and sampled_state_successor_failures == 0
        and sampled_energy_upper_violations == 0
        else "blocked-by-mission-certificate"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--sampled-chains", type=int, default=12)
    parser.add_argument("--output-dir", default="artifacts/mission_certificate")
    parser.add_argument("--single-process", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if len(args.scenarios) > 1 and not args.single_process:
        reports = []
        for scenario in args.scenarios:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--scenarios",
                    scenario,
                    "--sampled-chains",
                    str(args.sampled_chains),
                    "--output-dir",
                    str(output),
                    "--single-process",
                ],
                check=True,
            )
            reports.append(json.loads((output / f"{scenario}.json").read_text(encoding="utf-8")))
        gate = all(report["mission_certificate_gate"] == "PASS" for report in reports)
        (output / "gate.json").write_text(
            json.dumps({"MISSION_CERTIFICATE_GATE": "PASS" if gate else "blocked-by-mission-certificate", "scenarios": args.scenarios}, indent=2),
            encoding="utf-8",
        )
        if not gate:
            raise SystemExit(2)
        return
    reports = []
    for scenario in args.scenarios:
        report = validate_scenario(scenario, args.sampled_chains)
        reports.append(report)
        (output / f"{scenario}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, sort_keys=True), flush=True)
        gc.collect()
    gate = all(report["mission_certificate_gate"] == "PASS" for report in reports)
    (output / "gate.json").write_text(
        json.dumps({"MISSION_CERTIFICATE_GATE": "PASS" if gate else "blocked-by-mission-certificate", "scenarios": args.scenarios}, indent=2),
        encoding="utf-8",
    )
    if not gate:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
