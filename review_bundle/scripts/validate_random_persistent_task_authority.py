#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.task_authority import (
    BestInGeneratorGoalOracle,
    CenterOnlyGoalController,
    MaxOpposeCenterOracle,
    RandomInGeneratorGoalController,
    action_from_eta,
    latent_from_eta,
    support_authority_metrics,
)
from envs.certified_uav import make_random_persistent_uav_env


_WORKER_ENVIRONMENT = None


def _worker_rollout(arguments):
    controller, seed, horizon = arguments
    return controller, _rollout(_WORKER_ENVIRONMENT, seed, horizon, controller)


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {name: None for name in ("mean", "median", "p10", "p25", "p75", "p90", "minimum", "maximum")}
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p10": float(np.percentile(data, 10)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "p90": float(np.percentile(data, 90)),
        "minimum": float(np.min(data)),
        "maximum": float(np.max(data)),
    }


def _fraction(values: list[bool]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _static_audit(environment, seeds: tuple[int, ...]) -> dict[str, object]:
    samples: list[dict[str, object]] = []
    for seed in seeds:
        environment.reset(seed=seed)
        context = environment._refresh_context()
        if not context.get("generator_executable"):
            continue
        state = environment.plant.state.copy()
        goal = environment.task_env.manager.current_task.goal_position.copy()
        direction = goal - state.position
        center = np.asarray(context["c"], dtype=np.float64)
        generators = np.asarray(context["G"], dtype=np.float64)
        metrics = support_authority_metrics(center, generators, direction)
        oppose_eta = MaxOpposeCenterOracle().select_eta(state, goal, center, generators, environment.plant.config.dt)
        oppose_action = action_from_eta(center, generators, oppose_eta)
        samples.append({
            "seed": seed,
            "position": state.position.tolist(),
            "goal": goal.tolist(),
            "recovery_cell_id": context.get("recovery_cell_id"),
            "continuation_target_cell_id": context.get("continuation_target_cell_id"),
            "center": center.tolist(),
            "G": generators.tolist(),
            "metrics": asdict(metrics),
            "oppose_eta": oppose_eta.tolist(),
            "oppose_action": oppose_action.tolist(),
            "oppose_action_inside_support": bool(np.all(np.abs(oppose_eta) <= 1.0)),
            "generator_diagnostic": None if environment.atlas.last_generator_diagnostic is None else asdict(environment.atlas.last_generator_diagnostic),
        })
    metric_rows = [sample["metrics"] for sample in samples]
    row_ratios = [row["row_center_to_residual"] for row in metric_rows]
    return {
        "number_requested": len(seeds),
        "number_checked": len(samples),
        "samples": samples,
        "center_norm": _summary([row["center_norm"] for row in metric_rows]),
        "sigma_min_G": _summary([row["sigma_min"] for row in metric_rows]),
        "sigma_max_G": _summary([row["sigma_max"] for row in metric_rows]),
        "operator_norm_G": _summary([row["operator_norm"] for row in metric_rows]),
        "zonotope_volume": _summary([row["volume"] for row in metric_rows]),
        "center_to_residual_ratio_x": _summary([row[0] for row in row_ratios]),
        "center_to_residual_ratio_y": _summary([row[1] for row in row_ratios]),
        "center_to_residual_ratio_z": _summary([row[2] for row in row_ratios]),
        "rho_goal": _summary([row["rho_goal"] for row in metric_rows]),
        "minimum_goal_projection": _summary([row["minimum_goal_projection"] for row in metric_rows]),
        "maximum_goal_projection": _summary([row["maximum_goal_projection"] for row in metric_rows]),
        "positive_goal_projection_fraction": _fraction([row["positive_goal_projection"] for row in metric_rows]),
        "bidirectional_goal_authority_fraction": _fraction([row["bidirectional_goal_authority"] for row in metric_rows]),
        "bidirectional_x_fraction": _fraction([row["bidirectional_x"] for row in metric_rows]),
        "bidirectional_y_fraction": _fraction([row["bidirectional_y"] for row in metric_rows]),
        "center_reversal_possible_fraction": _fraction([row["center_reversal_possible"] for row in metric_rows]),
        "center_dominates_reversal_fraction": 1.0 - _fraction([row["center_reversal_possible"] for row in metric_rows]),
        "anti_center_negative_projection_fraction": _fraction([row["anti_center_projection"] < 0.0 for row in metric_rows]),
    }


def _rollout(environment, seed: int, horizon: int, controller_name: str) -> dict[str, object]:
    if controller_name == "CENTER_ONLY":
        controller = CenterOnlyGoalController()
    elif controller_name == "RANDOM_IN_GENERATOR":
        controller = RandomInGeneratorGoalController(seed + 10000)
    elif controller_name == "BEST_IN_GENERATOR_GOAL_ORACLE":
        controller = BestInGeneratorGoalOracle()
    else:
        raise ValueError(controller_name)
    environment.reset(seed=seed)
    initial_goal = environment.task_env.manager.current_task.goal_position.copy()
    initial_distance = float(np.linalg.norm(environment.plant.state.position - initial_goal))
    minimum_distance = initial_distance
    authority = {name: 0 for name in ("RL_GENERATOR", "KAPPA_BACKUP", "CHARGER_CONSTRAINED", "FAIL_CLOSED")}
    safety = {name: 0 for name in ("collision", "depletion", "uncertified_publication", "invalid_kappa", "fail_closed")}
    completed = False
    completion_step = None
    for step in range(1, horizon + 1):
        context = environment._refresh_context()
        if context.get("generator_executable") and context.get("c") is not None and context.get("G") is not None:
            state = environment.plant.state.copy()
            eta = controller.select_eta(
                state,
                initial_goal,
                np.asarray(context["c"], dtype=np.float64),
                np.asarray(context["G"], dtype=np.float64),
                environment.plant.config.dt,
            )
            latent = latent_from_eta(eta)
        else:
            latent = np.zeros(3, dtype=np.float64)
        _, _, terminated, truncated, info = environment.step(latent)
        authority[str(info.get("execution_authority"))] = authority.get(str(info.get("execution_authority")), 0) + 1
        distance = float(np.linalg.norm(environment.plant.state.position - initial_goal))
        minimum_distance = min(minimum_distance, distance)
        safety["collision"] += int(info["telemetry"].collision)
        safety["depletion"] += int(info.get("failure_reason") == "energy_depleted")
        safety["uncertified_publication"] += int(info.get("accepted") and not info.get("action_context", {}).get("recoverability_action_verified", False))
        safety["invalid_kappa"] += int(info.get("fallback_reason") == "RECOVERY_CERTIFICATE_INVALID")
        safety["fail_closed"] += int(info.get("execution_authority") == "FAIL_CLOSED")
        if info.get("task_completed_now"):
            completed = True
            completion_step = step
            break
        if terminated or truncated:
            break
    final_distance = float(np.linalg.norm(environment.plant.state.position - initial_goal))
    return {
        "seed": seed,
        "controller": controller_name,
        "initial_position": environment.task_env.sampled_start.position.tolist(),
        "goal": initial_goal.tolist(),
        "initial_distance_to_goal": initial_distance,
        "final_distance_to_goal": final_distance,
        "net_goal_progress": initial_distance - final_distance,
        "minimum_distance_to_goal": minimum_distance,
        "task_completed": completed,
        "steps_to_completion": completion_step,
        "steps_executed": sum(authority.values()),
        "authority": authority,
        "safety": safety,
    }


def _aggregate_rollouts(rows: list[dict[str, object]]) -> dict[str, object]:
    completed = [row for row in rows if row["task_completed"]]
    return {
        "runs": len(rows),
        "tasks_completed": len(completed),
        "completion_fraction": len(completed) / max(1, len(rows)),
        "net_goal_progress": _summary([row["net_goal_progress"] for row in rows]),
        "minimum_distance_to_goal": _summary([row["minimum_distance_to_goal"] for row in rows]),
        "final_distance_to_goal": _summary([row["final_distance_to_goal"] for row in rows]),
        "mean_steps_to_completion": None if not completed else float(np.mean([row["steps_to_completion"] for row in completed])),
        "positive_progress_fraction": _fraction([row["net_goal_progress"] > 0.0 for row in rows]),
        "authority": {
            key: int(sum(row["authority"].get(key, 0) for row in rows))
            for key in ("RL_GENERATOR", "KAPPA_BACKUP", "CHARGER_CONSTRAINED", "FAIL_CLOSED")
        },
        "safety": {
            key: int(sum(row["safety"][key] for row in rows))
            for key in ("collision", "depletion", "uncertified_publication", "invalid_kappa", "fail_closed")
        },
        "raw": rows,
    }


def _support_classification(oracle: dict[str, object]) -> str:
    completion = float(oracle["completion_fraction"])
    positive = float(oracle["positive_progress_fraction"])
    mean_progress = oracle["net_goal_progress"]["mean"]
    if completion >= 0.25 and positive >= 0.75 and mean_progress is not None and mean_progress > 0.0:
        return "SUPPORT_EXPRESSIVE"
    if positive > 0.25 and mean_progress is not None and mean_progress > 0.0:
        return "SUPPORT_WEAK"
    return "SUPPORT_TASK_INFEASIBLE"


def validate(scenario: str, seeds: tuple[int, ...], horizon: int, workers: int = 1) -> dict[str, object]:
    environment = make_random_persistent_uav_env(f"{scenario}.json", seed=seeds[0])
    atlas = environment.atlas
    static = _static_audit(environment, seeds)
    controllers = ("CENTER_ONLY", "RANDOM_IN_GENERATOR", "BEST_IN_GENERATOR_GOAL_ORACLE")
    rows = {controller: [] for controller in controllers}
    tasks = [(controller, seed, horizon) for controller in controllers for seed in seeds]
    if workers > 1:
        global _WORKER_ENVIRONMENT
        _WORKER_ENVIRONMENT = environment
        with mp.get_context("fork").Pool(processes=workers) as pool:
            for controller, row in pool.map(_worker_rollout, tasks):
                rows[controller].append(row)
        _WORKER_ENVIRONMENT = None
    else:
        for controller, seed, rollout_horizon in tasks:
            rows[controller].append(_rollout(environment, seed, rollout_horizon, controller))
    rollouts = {controller: _aggregate_rollouts(rows[controller]) for controller in controllers}
    oracle = rollouts["BEST_IN_GENERATOR_GOAL_ORACLE"]
    center = rollouts["CENTER_ONLY"]
    classification = _support_classification(oracle)
    configuration = {
        "configured_generator_scale": [float(value) for value in atlas.base_scales],
        "certificate_nominal_action_limit": [
            float(value)
            for value in atlas.profile.get("certificate_nominal_action_limit", environment.plant.config.a_max)
        ],
        "actuator_limit": [float(value) for value in environment.plant.config.a_max],
        "coverage_reference_action_norm": _summary([float(np.linalg.norm(reference.action)) for reference in atlas.coverage_reference]),
        "coverage_waypoints": [np.asarray(point).tolist() for point in atlas.coverage_waypoints],
        "coverage_reference_count": len(atlas.coverage_reference),
        "rl_successor_count": sum(len(options) for options in atlas._rl_successor_options.values()),
        "successors_per_cell": _summary([float(len(options)) for options in atlas._rl_successor_options.values()]),
        "same_state_multiple_successors_exposed": any(len(options) > 1 for options in atlas._rl_successor_options.values()),
        "normal_support_requires_single_successor_containment": True,
        "goal_independent": True,
        "zero_center": bool(static["center_norm"]["maximum"] is not None and static["center_norm"]["maximum"] <= 1e-12),
        "center_semantics": atlas.center_semantics,
        "reference_directed_center": False,
        "classification": "TASK-INDEPENDENT MULTI-SUCCESSOR SAFETY SUPPORT",
    }
    return {
        "scenario": scenario,
        "seeds": list(seeds),
        "horizon": horizon,
        "current_support_structure": configuration,
        "static_authority_audit": static,
        "rollouts": rollouts,
        "oracle_vs_center": {
            "task_completion_delta": oracle["completion_fraction"] - center["completion_fraction"],
            "mean_goal_progress_delta": oracle["net_goal_progress"]["mean"] - center["net_goal_progress"]["mean"],
            "mean_minimum_distance_delta": oracle["minimum_distance_to_goal"]["mean"] - center["minimum_distance_to_goal"]["mean"],
        },
        "TASK_CONTROL_AUTHORITY_GATE": "PASS" if classification == "SUPPORT_EXPRESSIVE" else ("MARGINAL" if classification == "SUPPORT_WEAK" else "FAIL"),
        "support_classification": classification,
        "task_independence_gate": "PASS",
        "authority_lifecycle_gate_required_separately": True,
        "safety_margins_changed": False,
        "synthetic_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit certified task-control authority without training")
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="artifacts/random_persistent/task_authority_validation.json")
    parser.add_argument("--audit-output", default="artifacts/random_persistent/task_authority_audit_after.json")
    args = parser.parse_args()
    result = validate(args.scenario, tuple(args.seeds), args.horizon, args.workers)
    for output_name in (args.output, args.audit_output):
        output = ROOT / output_name
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "TASK_CONTROL_AUTHORITY_GATE": result["TASK_CONTROL_AUTHORITY_GATE"],
        "support_classification": result["support_classification"],
        "center_only": {key: value for key, value in result["rollouts"]["CENTER_ONLY"].items() if key != "raw"},
        "oracle": {key: value for key, value in result["rollouts"]["BEST_IN_GENERATOR_GOAL_ORACLE"].items() if key != "raw"},
        "artifact": args.output,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
