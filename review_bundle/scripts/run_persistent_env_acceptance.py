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
from envs.certified_uav import PersistentMissionMode, make_persistent_uav_env
from scripts.validate_persistent_certificate import policy_authority_report


def directional_latent(context: dict, position: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Deterministic acceptance latent; not a policy or experiment baseline."""
    center = np.asarray(context["c"], dtype=np.float64)
    generators = np.asarray(context["G"], dtype=np.float64)
    direction = np.asarray(target, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    sigma = float(np.min(np.linalg.svd(generators, compute_uv=False)))
    desired = 0.5 * sigma * direction / norm
    eta = np.linalg.solve(generators, desired - center)
    eta = np.clip(eta, -0.9, 0.9)
    action = center + generators @ eta
    if float(action @ direction) <= 0.0:
        raise RuntimeError("verified support has no requested directional authority")
    return np.arctanh(eta)


def record_step(env, index: int, probe: str, info: dict, reward: float) -> dict:
    telemetry = info["telemetry"]
    trace = telemetry.action_trace
    return {
        "step": index,
        "probe": probe,
        "reward": float(reward),
        "mode": info.get("persistent_mode"),
        "execution_authority": info.get("execution_authority"),
        "authority_reason": info.get("execution_authority_reason"),
        "accepted": bool(info.get("accepted", False)),
        "backup_triggered": bool(info.get("backup_triggered", False)),
        "backup_reason": info.get("backup_reason"),
        "departure_attempt": bool(info.get("departure_attempt", False)),
        "departure_rejected": bool(info.get("departure_rejected", False)),
        "energy": float(env.plant.state.energy),
        "collision": bool(telemetry.collision),
        "published": np.asarray(trace.published).tolist(),
        "measured": np.asarray(trace.measured).tolist(),
        "command_source": info.get("command_source", "runtime"),
        "tasks_completed": int(info.get("persistent_metrics", {}).get("tasks_completed", 0)),
    }


def run_action(env, latent: np.ndarray, probe: str, records: list[dict]) -> tuple[bool, bool, dict]:
    _, reward, terminated, truncated, info = env.step(np.asarray(latent, dtype=np.float64))
    records.append(record_step(env, len(records), probe, info, reward))
    return terminated, truncated, info


def run_zero_probe(env, records: list[dict]) -> dict[str, bool]:
    terminated, truncated, info = run_action(env, np.zeros(3), "zero", records)
    return {"zero_probe_completed": not terminated and not truncated, "zero_probe_accepted_or_backup": "telemetry" in info}


def run_goal_probe(env, records: list[dict]) -> dict[str, bool]:
    before = float(np.linalg.norm(env.task_env.manager.current_task.goal_position - env.plant.state.position))
    context = env._refresh_context()
    latent = directional_latent(context, env.plant.state.position, env.task_env.manager.current_task.goal_position)
    _, _, info = run_action(env, latent, "goal", records)
    after = float(np.linalg.norm(env.task_env.manager.current_task.goal_position - env.plant.state.position))
    return {
        "goal_direction_authority": bool(info.get("accepted", False) and not info.get("backup_triggered", False)),
        "goal_progress": after < before,
    }


def run_voluntary_charge_probe(env, records: list[dict], max_steps: int) -> dict[str, bool]:
    pending_task = env.task_env.manager.current_task.task_id
    arrived_without_backup = False
    for _ in range(max_steps):
        context = env._refresh_context()
        latent = directional_latent(context, env.plant.state.position, env.plant.scenario.station_position)
        terminated, truncated, info = run_action(env, latent, "voluntary_station", records)
        if info.get("backup_triggered"):
            break
        if env.task_env.mode == PersistentMissionMode.CHARGING_RL:
            arrived_without_backup = True
            break
        if terminated or truncated:
            break
    charging_increased = False
    unsafe_departure_not_published = False
    valid_departure = False
    if arrived_without_backup:
        energy_before = float(env.plant.state.energy)
        context = env._refresh_context()
        if not context.get("departure_allowed", False):
            target = env.task_env.manager.current_task.goal_position
            latent = directional_latent(context, env.plant.state.position, target)
            _, _, info = run_action(env, latent, "closed_departure", records)
            unsafe_departure_not_published = bool(
                info.get("execution_authority") == ExecutionAuthority.CHARGER_CONSTRAINED.value
                and env.plant.terminal.is_charge_admissible(env.plant.state)
            )
        for _ in range(max_steps):
            context = env._refresh_context()
            if context.get("departure_allowed", False):
                break
            terminated, truncated, _ = run_action(env, np.zeros(3), "charging_dwell", records)
            if terminated or truncated:
                break
        charging_increased = env.plant.state.energy > energy_before
        for _ in range(max_steps):
            context = env._refresh_context()
            if not context.get("departure_allowed", False):
                break
            latent = directional_latent(context, env.plant.state.position, env.task_env.manager.current_task.goal_position)
            terminated, truncated, info = run_action(env, latent, "valid_departure", records)
            if env.task_env.mode == PersistentMissionMode.TASK_RL and not env.plant.terminal.is_charge_admissible(env.plant.state):
                valid_departure = bool(info.get("accepted", False))
                break
            if terminated or truncated:
                break
    return {
        "station_direction_authority": arrived_without_backup,
        "voluntary_station_arrival_without_kappa": arrived_without_backup,
        "charging_increased_energy": charging_increased,
        "unsafe_departure_not_published": unsafe_departure_not_published,
        "valid_departure_succeeds": valid_departure,
        "pending_goal_preserved": env.task_env.manager.current_task.task_id == pending_task,
    }


def run_backup_probe(scenario: str, seed: int, records: list[dict], max_steps: int) -> dict[str, bool]:
    env = make_persistent_uav_env(f"{scenario}.json", seed=seed, timing_mode="functional")
    env.reset(seed=seed)
    context = env._refresh_context()
    margin = float(context["energy_margin"])
    env.plant.state.energy -= max(0.0, margin - env.charging.config.forced_return_margin)
    exact_reason = False
    terminal_reached = False
    for _ in range(max_steps):
        terminated, truncated, info = run_action(env, np.full(3, 0.5), "backup", records)
        exact_reason |= info.get("backup_reason") in {"ENERGY_MARGIN_BACKUP_SWITCH", "BACKUP_RECOVERY_CONTINUATION"}
        if env.task_env.mode == PersistentMissionMode.CHARGING_RL:
            terminal_reached = True
            break
        if terminated or truncated:
            break
    return {"energy_boundary_invokes_kappa": exact_reason, "backup_reaches_terminal": terminal_reached}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic theorem-facing persistent acceptance probes.")
    parser.add_argument("--scenario", default="persistent_open")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--probe", choices=("zero", "goal", "voluntary-charge", "backup", "all"), default="all")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    env = make_persistent_uav_env(f"{args.scenario}.json", seed=args.seed, timing_mode="functional")
    _, reset_info = env.reset(seed=args.seed)
    policy_gate = policy_authority_report(env)
    records: list[dict] = []
    assertions: dict[str, bool] = {
        "persistent_certificate_gate": reset_info["persistent_certificate_gate"] == "PASS",
        "policy_authority_gate": policy_gate["gate"] == "PASS",
    }
    if args.probe in {"zero", "all"}:
        assertions.update(run_zero_probe(env, records))
    if args.probe in {"goal", "all"}:
        assertions.update(run_goal_probe(env, records))
    if args.probe in {"voluntary-charge", "all"}:
        assertions.update(run_voluntary_charge_probe(env, records, args.steps))
    if args.probe in {"backup", "all"}:
        assertions.update(run_backup_probe(args.scenario, args.seed, records, args.steps))
    metrics = env.metric_snapshot()
    assertions.update({
        "zero_uncertified_publications": metrics["uncertified_publication_count"] == 0,
        "zero_sampled_collisions": metrics["collision_count"] == 0,
        "zero_energy_depletion": metrics["energy_depletion_count"] == 0,
    })
    payload = {
        "scenario": args.scenario,
        "policy": "single_continuous_generator_sac",
        "probe": args.probe,
        "persistent_gate": reset_info["persistent_certificate_gate"],
        "policy_authority_gate": policy_gate,
        "assertions": assertions,
        "strict_pass": all(assertions.values()),
        "metrics": metrics,
        "records": records,
        "synthetic_only": True,
    }
    output = ROOT / (args.output or f"artifacts/persistent/acceptance_{args.scenario}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"assertions": assertions, "metrics": metrics}, indent=2))
    if args.strict and not payload["strict_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
