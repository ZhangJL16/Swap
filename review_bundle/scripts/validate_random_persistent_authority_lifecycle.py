#!/usr/bin/env python3
"""Deterministic synthetic validation of persistent execution-authority lifecycle."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.persistent_authority import ExecutionAuthority
from cert_runtime.task_authority import BestInGeneratorGoalOracle, latent_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


DEFAULT_SCENARIOS = ("random_persistent_open", "random_persistent_energy_tight")


def _longest_runs(values: list[str], selected: str) -> list[int]:
    runs: list[int] = []
    current = 0
    for value in values:
        if value == selected:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def _set_state(environment, cell, energy: float) -> None:
    environment.plant.state = UAVPhysicalState(
        np.asarray(cell.reference_position, dtype=np.float64),
        np.asarray(cell.reference_velocity, dtype=np.float64),
        float(energy),
        0.0,
    )
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        environment.plant.state,
        environment.plant.world,
        environment.plant.np_random,
    )


def _terminal_level_fixture(atlas, maximum_level: int = 5):
    candidates = [
        cell
        for cell in atlas.manifest.cells
        if 1 <= cell.level <= maximum_level and cell.hash_valid and cell.complete_successor_containment
    ]
    if not candidates:
        raise RuntimeError("RECOVERY_ATLAS_HAS_NO_NEAR_TERMINAL_FIXTURE")
    return min(candidates, key=lambda cell: (cell.level, cell.cell_id))


def validate_scenario(name: str, seeds: tuple[int, ...], endurance_steps_per_seed: int) -> dict[str, object]:
    environment = make_random_persistent_uav_env(f"{name}.json", seed=seeds[0])
    atlas = environment.atlas
    failures: list[str] = []
    authority_values: list[str] = []
    no_generator_count = 0
    accepted_into_kappa_only = 0
    accepted_into_kappa_only_witnesses: list[dict[str, object]] = []
    lifecycle_traces: list[dict[str, object]] = []
    rl_runs: list[int] = []

    for seed in seeds:
        _, reset_info = environment.reset(seed=seed)
        if reset_info["task_independence_gate"] != "PASS":
            failures.append(f"seed={seed}:TASK_INDEPENDENCE_GATE")
            continue
        seed_authorities: list[str] = []
        for _ in range(endurance_steps_per_seed):
            _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float64))
            authority = str(info.get("execution_authority"))
            authority_values.append(authority)
            seed_authorities.append(authority)
            no_generator_count += int(info.get("backup_reason") == "NO_GENERATOR_SET")
            if info.get("accepted") and not (terminated or truncated):
                next_context = environment._refresh_context()
                if not next_context.get("rl_authority_set_member") and next_context.get("persistent_mode") != "CHARGING_RL":
                    accepted_into_kappa_only += 1
                    accepted_into_kappa_only_witnesses.append({
                        "seed": seed,
                        "position": environment.plant.state.position.tolist(),
                        "velocity": environment.plant.state.velocity.tolist(),
                        "recovery_cell_id": next_context.get("recovery_cell_id"),
                        "execution_authority": next_context.get("execution_authority"),
                        "authority_reason": next_context.get("execution_authority_reason"),
                        "generator_available": next_context.get("generator_available"),
                        "recoverable_set_member": next_context.get("recoverable_set_member"),
                    })
            if terminated or truncated:
                failures.append(f"seed={seed}:endurance-ended:{info.get('failure_reason')}")
                break
        rl_runs.extend(_longest_runs(seed_authorities, ExecutionAuthority.RL_GENERATOR.value))

        _, reset_info = environment.reset(seed=seed)
        pending_task = environment.task_env.manager.current_task
        pending_id = pending_task.task_id
        pending_goal = pending_task.goal_position.copy()
        fixture = _terminal_level_fixture(atlas)
        _set_state(environment, fixture, max(5.0, fixture.state_bounds.energy.low + 0.5))
        environment.task_env.mode = PersistentMissionMode.BACKUP_RECOVERY
        environment.task_env.phase = environment.task_env.mode
        atlas.recovery_active = True
        atlas.active_cell_id = fixture.cell_id
        trace = [ExecutionAuthority.RL_GENERATOR.value, ExecutionAuthority.KAPPA_BACKUP.value]
        backup_reached_terminal = False
        invalid_kappa = False
        fail_closed = False
        for _ in range(32):
            _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float64))
            trace.append(str(info.get("execution_authority")))
            invalid_kappa |= info.get("backup_reason") == "KAPPA_CERTIFICATE_INVALID"
            fail_closed |= info.get("execution_authority") == ExecutionAuthority.FAIL_CLOSED.value
            if environment.plant.terminal.is_charge_admissible(environment.plant.state):
                backup_reached_terminal = True
                break
            if terminated or truncated:
                break
        environment.task_env.enter_charging(voluntary=False)
        environment.plant.state.energy = (
            environment.plant.scenario.terminal.minimum_energy + atlas.energy_reserve + 0.05
        )
        context = environment._refresh_context()
        terminal_valid = bool(
            context.get("certificate_valid")
            and context.get("terminal_recovery_certificate_hash") == atlas.terminal_recovery_certificate.certificate_hash
            and np.isfinite(context.get("recovery_energy_required"))
            and np.isfinite(context.get("energy_margin"))
        )
        energy_before_charge = environment.plant.state.energy
        for _ in range(4):
            _, _, terminated, truncated, info = environment.step(np.zeros(3, dtype=np.float64))
            trace.append(str(info.get("execution_authority")))
            invalid_kappa |= info.get("backup_reason") == "KAPPA_CERTIFICATE_INVALID"
            fail_closed |= info.get("execution_authority") == ExecutionAuthority.FAIL_CLOSED.value
            if terminated or truncated:
                break
        charging_increased = environment.plant.state.energy > energy_before_charge
        departure_succeeded = False
        post_departure_rl = False
        departure_oracle = BestInGeneratorGoalOracle()
        for _ in range(80):
            context = environment._refresh_context()
            task = environment.task_env.manager.current_task
            goal = environment.plant.state.position if task is None else task.goal_position
            eta = departure_oracle.select_eta(
                environment.plant.state,
                goal,
                np.asarray(context["c"], dtype=np.float64),
                np.asarray(context["G"], dtype=np.float64),
                environment.plant.config.dt,
            )
            _, _, terminated, truncated, info = environment.step(latent_from_eta(eta))
            trace.append(str(info.get("execution_authority")))
            invalid_kappa |= info.get("backup_reason") == "KAPPA_CERTIFICATE_INVALID"
            fail_closed |= info.get("execution_authority") == ExecutionAuthority.FAIL_CLOSED.value
            if environment.task_env.mode == PersistentMissionMode.TASK_RL:
                departure_succeeded = True
                next_context = environment._refresh_context()
                post_departure_rl = bool(
                    next_context.get("rl_authority_set_member")
                    and next_context.get("execution_authority") == ExecutionAuthority.RL_GENERATOR.value
                )
                break
            if terminated or truncated:
                break
        task = environment.task_env.manager.current_task
        pending_preserved = bool(
            task is not None
            and task.task_id == pending_id
            and np.allclose(task.goal_position, pending_goal)
        )
        lifecycle_traces.append({
            "seed": seed,
            "fixture_cell_id": fixture.cell_id,
            "backup_reached_terminal": backup_reached_terminal,
            "terminal_recovery_certificate_valid": terminal_valid,
            "terminal_recovery_energy_upper": atlas.terminal_recovery_certificate.recovery_energy_upper,
            "charging_increased_energy": charging_increased,
            "departure_succeeded": departure_succeeded,
            "post_departure_in_R_RL": post_departure_rl,
            "pending_goal_preserved": pending_preserved,
            "invalid_kappa": invalid_kappa,
            "fail_closed": fail_closed,
            "authority_trace": trace,
        })
        for condition, reason in (
            (backup_reached_terminal, "backup-did-not-reach-terminal"),
            (terminal_valid, "terminal-recovery-certificate-invalid"),
            (charging_increased, "charging-did-not-increase-energy"),
            (departure_succeeded, "departure-did-not-succeed"),
            (post_departure_rl, "post-departure-not-in-R_RL"),
            (pending_preserved, "pending-goal-not-preserved"),
            (not invalid_kappa, "invalid-kappa-event"),
            (not fail_closed, "fail-closed-event"),
        ):
            if not condition:
                failures.append(f"seed={seed}:{reason}")

    counts = Counter(authority_values)
    total = max(1, len(authority_values))
    metrics = atlas.authority_domain_report()
    metrics.update({
        "endurance_steps": len(authority_values),
        "seeds": list(seeds),
        "RL_GENERATOR_steps": counts[ExecutionAuthority.RL_GENERATOR.value],
        "KAPPA_BACKUP_steps": counts[ExecutionAuthority.KAPPA_BACKUP.value],
        "CHARGER_CONSTRAINED_steps": counts[ExecutionAuthority.CHARGER_CONSTRAINED.value],
        "FAIL_CLOSED_steps": counts[ExecutionAuthority.FAIL_CLOSED.value],
        "RL_GENERATOR_fraction": counts[ExecutionAuthority.RL_GENERATOR.value] / total,
        "KAPPA_BACKUP_fraction": counts[ExecutionAuthority.KAPPA_BACKUP.value] / total,
        "CHARGER_CONSTRAINED_fraction": counts[ExecutionAuthority.CHARGER_CONSTRAINED.value] / total,
        "mean_consecutive_RL_authority_duration": statistics.mean(rl_runs) if rl_runs else 0.0,
        "median_consecutive_RL_authority_duration": statistics.median(rl_runs) if rl_runs else 0.0,
        "minimum_consecutive_RL_authority_duration": min(rl_runs) if rl_runs else 0,
        "NO_GENERATOR_SET_count": no_generator_count,
        "accepted_actions_entering_kappa_only_cells": accepted_into_kappa_only,
    })
    safety = environment.metric_snapshot()
    for field in ("collision_count", "energy_depletion_count", "uncertified_publication_count", "invalid_kappa_fallback_count"):
        if safety.get(field, 0):
            failures.append(f"nonzero-{field}:{safety[field]}")
    if accepted_into_kappa_only:
        failures.append(f"accepted-actions-entered-kappa-only:{accepted_into_kappa_only}")
    passed = not failures
    return {
        "scenario": name,
        "AUTHORITY_LIFECYCLE_GATE": "PASS" if passed else "FAIL",
        "authority_domain": metrics,
        "accepted_into_kappa_only_witnesses": accepted_into_kappa_only_witnesses,
        "lifecycle_traces": lifecycle_traces,
        "failures": failures,
        "synthetic_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate random persistent authority lifecycle")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--endurance-steps-per-seed", type=int, default=200)
    parser.add_argument("--output", default="artifacts/random_persistent/authority_lifecycle_validation.json")
    args = parser.parse_args()
    results = [
        validate_scenario(name, tuple(args.seeds), args.endurance_steps_per_seed)
        for name in args.scenarios
    ]
    passed = all(item["AUTHORITY_LIFECYCLE_GATE"] == "PASS" for item in results)
    payload = {
        "AUTHORITY_LIFECYCLE_GATE": "PASS" if passed else "FAIL",
        "results": results,
        "evidence_scope": "deterministic synthetic software validation; not RL performance evidence",
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
