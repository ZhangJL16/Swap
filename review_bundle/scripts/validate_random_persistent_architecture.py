from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import make_random_persistent_uav_env


DEFAULT_SCENARIOS = (
    "random_persistent_open",
    "random_persistent_obstacle",
    "random_persistent_energy_tight",
)


def _array(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def validate_scenario(name: str, seeds: tuple[int, ...]) -> dict[str, object]:
    env = make_random_persistent_uav_env(f"{name}.json")
    atlas = env.atlas
    starts = []
    goals = []
    failures: list[str] = []
    equality_checks = {
        "required_return_energy": True,
        "recoverable_membership": True,
        "rl_authority_membership": True,
        "kappa_certificate": True,
        "center": True,
        "generators": True,
        "action_bounds": True,
        "continuation_support": True,
        "atlas_hash": True,
    }
    actor_action_changed = False
    minimum_sigma = float("inf")
    minimum_volume = float("inf")
    maximum_condition = 0.0
    complete_generator_in_safe_set = True
    generator_checks = 0
    kappa_checks = 0

    for seed in seeds:
        sampled = atlas.sample_initial_state(seed, env.charging.config.battery_capacity)
        env.plant.state = sampled.copy()
        state = env.runtime._certificate_state()
        atlas.reset()
        first_context = atlas.evaluate(state, sampled.timestamp)
        starts.append(sampled.position.tolist())
        kappa_valid = bool(atlas.contains_certificate_state(state) and first_context.recovery.certified)
        kappa_checks += 1
        if not kappa_valid:
            failures.append(f"seed={seed}:sampled-start-not-recoverable")
            continue

        first_goal = atlas.sample_goal(np.random.default_rng(seed + 1000), sampled.position, 0.5)
        second_goal = atlas.sample_goal(np.random.default_rng(seed + 2000), sampled.position, 0.5)
        goals.extend((first_goal.tolist(), second_goal.tolist()))
        supports = []
        for goal in (first_goal, second_goal):
            if env.task_env.manager.current_task is None:
                env.task_env.manager.reset(seed, sampled.position)
            env.task_env.manager.current_task.goal_position = goal.copy()
            atlas.reset()
            context = atlas.evaluate(state, sampled.timestamp)
            certificate = context.closure.zonotope_certificate
            if certificate is None or not certificate.verified:
                failures.append(f"seed={seed}:generator-unavailable")
                supports.append(None)
                continue
            zonotope = certificate.zonotope
            action_certificate = atlas.last_recoverability_action_certificate
            supports.append((
                context,
                _array(zonotope.center),
                _array(zonotope.generators),
                _array(zonotope.action_bounds.low),
                _array(zonotope.action_bounds.high),
            ))
            generator_checks += 1
            minimum_sigma = min(minimum_sigma, float(zonotope.sigma_min_lower_bound))
            minimum_volume = min(minimum_volume, float(8.0 * abs(zonotope.determinant)))
            maximum_condition = max(maximum_condition, float(zonotope.condition_number_upper_bound))
            complete_generator_in_safe_set &= bool(action_certificate is not None and action_certificate.verified)
        if any(item is None for item in supports):
            continue
        first, second = supports
        equality_checks["required_return_energy"] &= first[0].required_energy == second[0].required_energy
        equality_checks["recoverable_membership"] &= bool(
            atlas.contains_certificate_state(state)
            and atlas.last_recoverable_set_certificate.recoverable
        )
        equality_checks["rl_authority_membership"] &= atlas.contains_rl_authority_state(state)
        equality_checks["kappa_certificate"] &= first[0].recovery.certificate_hash == second[0].recovery.certificate_hash
        equality_checks["center"] &= bool(np.allclose(first[1], second[1]))
        equality_checks["generators"] &= bool(np.allclose(first[2], second[2]))
        equality_checks["action_bounds"] &= bool(np.allclose(first[3], second[3]) and np.allclose(first[4], second[4]))
        equality_checks["continuation_support"] &= bool(
            atlas.last_continuation_verified
            and first[0].task_successor_cell_id == second[0].task_successor_cell_id
        )
        equality_checks["atlas_hash"] &= atlas.atlas_hash == env.manifest_hash
        actor_action_changed |= not np.allclose(
            np.clip(first_goal - sampled.position, -1.0, 1.0),
            np.clip(second_goal - sampled.position, -1.0, 1.0),
        )

    pending_goal_preserved = True
    sampled = atlas.sample_initial_state(seeds[0], env.charging.config.battery_capacity)
    env.plant.state = sampled.copy()
    env.task_env.manager.reset(seeds[0], sampled.position)
    pending = env.task_env.manager.current_task.goal_position.copy()
    pending_id = env.task_env.manager.current_task.task_id
    env.task_env.enter_charging(voluntary=True)
    env.task_env.leave_station()
    pending_goal_preserved &= bool(
        env.task_env.manager.current_task.task_id == pending_id
        and np.allclose(env.task_env.manager.current_task.goal_position, pending)
    )
    env.task_env.begin_backup_recovery("ARCHITECTURE_PROBE")
    pending_goal_preserved &= bool(
        env.task_env.manager.current_task.task_id == pending_id
        and np.allclose(env.task_env.manager.current_task.goal_position, pending)
    )

    gate = bool(
        atlas.gate_pass
        and atlas.task_independent
        and not atlas.consumes_task_edges
        and not atlas.consumes_task_waypoints
        and not hasattr(atlas, "task_reference")
        and all(equality_checks.values())
        and actor_action_changed
        and complete_generator_in_safe_set
        and generator_checks == 2 * len(seeds)
        and kappa_checks == len(seeds)
        and pending_goal_preserved
        and not failures
    )
    return {
        "scenario": name,
        "TASK_INDEPENDENCE_GATE": "PASS" if gate else "FAIL",
        "atlas_gate": "PASS" if atlas.gate_pass else "FAIL",
        "atlas_hash": atlas.atlas_hash,
        "number_of_recovery_cells": len(atlas.manifest.cells),
        "certified_coverage_fraction": atlas.persistent_manifest.certified_coverage_fraction,
        "authority_domain": atlas.authority_domain_report(),
        "sampled_starts": starts,
        "sampled_goals": goals,
        "goal_independence": equality_checks,
        "actor_action_changed_with_goal": actor_action_changed,
        "minimum_sigma_min_G": None if not np.isfinite(minimum_sigma) else minimum_sigma,
        "maximum_condition_G": maximum_condition,
        "minimum_zonotope_volume": None if not np.isfinite(minimum_volume) else minimum_volume,
        "complete_generator_inside_A_safe": complete_generator_in_safe_set,
        "generator_checks": generator_checks,
        "kappa_recovery_checks": kappa_checks,
        "pending_goal_preserved_across_charge_and_backup": pending_goal_preserved,
        "main_path_consumes_TASK_EDGE": atlas.consumes_task_edges,
        "main_path_consumes_task_waypoints": atlas.consumes_task_waypoints,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate task-independent random persistent architecture")
    parser.add_argument("--scenarios", nargs="+", default=list(DEFAULT_SCENARIOS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--output",
        default="artifacts/random_persistent/architecture_validation.json",
    )
    args = parser.parse_args()
    results = [validate_scenario(name, tuple(args.seeds)) for name in args.scenarios]
    overall = all(result["TASK_INDEPENDENCE_GATE"] == "PASS" for result in results)
    payload = {
        "TASK_INDEPENDENCE_GATE": "PASS" if overall else "FAIL",
        "evidence_scope": "deterministic synthetic architecture validation; not training or task-success evidence",
        "results": results,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
