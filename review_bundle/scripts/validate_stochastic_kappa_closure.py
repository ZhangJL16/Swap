#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.persistent_authority import ExecutionAuthority
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


def _set_state(environment, position: np.ndarray, velocity: np.ndarray, energy: float) -> None:
    environment.plant.state = UAVPhysicalState(position, velocity, float(energy), 0.0)
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        environment.plant.state,
        environment.plant.world,
        environment.plant.np_random,
    )
    environment._context_cache_key = None


def _near_terminal_cell(atlas):
    candidates = [
        cell for cell in atlas.manifest.cells
        if 1 <= cell.level <= 5 and cell.hash_valid and cell.complete_successor_containment
    ]
    if not candidates:
        raise RuntimeError("RECOVERY_ATLAS_HAS_NO_NEAR_TERMINAL_FIXTURE")
    return min(candidates, key=lambda cell: (cell.level, cell.cell_id))


def validate(seed_values: tuple[int, ...]) -> dict[str, object]:
    environment = make_random_persistent_uav_env("random_persistent_open.json", seed=seed_values[0])
    atlas = environment.atlas
    fixture = _near_terminal_cell(atlas)
    records: list[dict[str, object]] = []
    totals = {key: 0 for key in ("invalid_kappa", "fail_closed", "collision", "depletion", "uncertified")}
    for seed in seed_values:
        environment.reset(seed=seed)
        pending = environment.task_env.manager.current_task
        pending_id = pending.task_id
        pending_goal = pending.goal_position.copy()
        _set_state(
            environment,
            np.asarray(fixture.reference_position, dtype=np.float64),
            np.asarray(fixture.reference_velocity, dtype=np.float64),
            max(5.0, fixture.state_bounds.energy.low + 0.5),
        )
        environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        environment.task_env.phase = environment.task_env.mode
        atlas.recovery_active = True
        atlas.active_cell_id = fixture.cell_id
        reached_terminal = False
        charging_steps = 0
        authority_trace: list[str] = []
        failure_categories: list[str] = []
        for step in range(96):
            _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float64))
            context = info.get("action_context", {})
            authority_trace.append(str(info.get("execution_authority")))
            category = context.get("kappa_validation_failure_category")
            if category:
                failure_categories.append(str(category))
                totals["invalid_kappa"] += 1
            totals["fail_closed"] += int(info.get("execution_authority") == ExecutionAuthority.FAIL_CLOSED.value)
            totals["collision"] += int(bool(info.get("telemetry").collision))
            totals["depletion"] += int(info.get("failure_reason") == "energy_depleted")
            totals["uncertified"] += int(bool(info.get("accepted")) and not bool(context.get("recoverability_action_verified")))
            if environment.plant.terminal.is_charge_admissible(environment.plant.state):
                reached_terminal = True
            if environment.task_env.mode == PersistentMissionMode.CHARGING_RL:
                charging_steps += 1
                if charging_steps >= 20:
                    break
            if terminated or truncated:
                break
        current = environment.task_env.manager.current_task
        records.append({
            "seed": seed,
            "backup_reached_terminal": reached_terminal,
            "charging_steps": charging_steps,
            "pending_goal_preserved": bool(
                current is not None
                and current.task_id == pending_id
                and np.allclose(current.goal_position, pending_goal)
            ),
            "failure_categories": failure_categories,
            "authority_trace": authority_trace,
        })
    gate_pass = bool(
        all(record["backup_reached_terminal"] and record["charging_steps"] >= 20 and record["pending_goal_preserved"] for record in records)
        and all(value == 0 for value in totals.values())
    )
    return {
        "scenario": "random_persistent_open",
        "seeds": list(seed_values),
        "terminal_hold_certificate_hash": atlas.terminal_recovery_certificate.certificate_hash,
        "terminal_hold_rule_version": atlas.terminal_recovery_certificate.hold_rule_version,
        "counts": totals,
        "records": records,
        "STOCHASTIC_KAPPA_CLOSURE_GATE": "PASS" if gate_pass else "FAIL",
        "synthetic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=tuple(range(20)))
    parser.add_argument("--output", default="artifacts/random_persistent/stochastic_kappa_closure.json")
    args = parser.parse_args()
    result = validate(tuple(args.seeds))
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["STOCHASTIC_KAPPA_CLOSURE_GATE"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
