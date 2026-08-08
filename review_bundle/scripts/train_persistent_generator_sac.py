#!/usr/bin/env python3
"""Formal persistent single-policy training entry point; not a unit-test command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.experiment_metrics import (
    PersistentMetricAccumulator,
    episode_record,
    learning_curve_steps_monotonic,
    metric_snapshot_delta,
    write_jsonl,
)
from envs.certified_uav import make_persistent_uav_env, make_random_persistent_uav_env
from persistent_generator_common import transition_from_cycle


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one continuous persistent Generator-SAC policy.")
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--legacy-fixed-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="artifacts/persistent_generator_sac")
    parser.add_argument("--log-interval", type=int, default=500)
    args = parser.parse_args()
    if args.log_interval <= 0:
        raise ValueError("log interval must be positive")

    factory = make_persistent_uav_env if args.legacy_fixed_graph else make_random_persistent_uav_env
    environment = factory(f"{args.scenario}.json", seed=args.seed)
    observation, reset_info = environment.reset(seed=args.seed)
    context = reset_info["action_context"]
    config = GeneratorSACConfig(batch_size=args.batch_size, warmup_steps=args.warmup_steps)
    agent = PersistentGeneratorSAC(observation.size, config, seed=args.seed, device=args.device)
    rng = np.random.default_rng(args.seed)
    episode_id = 0
    episode_seed = args.seed
    run_metrics = PersistentMetricAccumulator()
    episode_metrics = PersistentMetricAccumulator()
    interval_metrics = PersistentMetricAccumulator()
    episode_records: list[dict[str, object]] = []
    curve_records: list[dict[str, object]] = []
    trajectory_records: list[dict[str, object]] = []
    previous_snapshot = None
    last_update: dict[str, float | int | str | None] | None = None
    actor_updates = 0
    critic_updates = 0
    accepted_actor_samples = 0

    def goal_sequence() -> list[np.ndarray]:
        manager = environment.task_env.manager
        return [np.asarray(goal, dtype=np.float64).copy() for goal in getattr(manager, "goal_sequence", ())]

    def learning_curve_record(step_number: int) -> dict[str, object]:
        interval = interval_metrics.summary()
        update = {} if last_update is None else last_update
        target_count = max(1, int(update.get("target_batch_count", 0) or 0))
        return {
            "step": step_number,
            "episode_id": episode_id,
            "interval_reward": interval["total_reward"],
            "cumulative_reward": run_metrics.total_reward,
            "interval_goal_progress": interval["total_goal_progress"],
            "interval_tasks_completed": interval["tasks_completed"],
            "tasks_per_1000_steps": interval["tasks_per_1000_steps"],
            "interval_voluntary_station_arrivals": interval["voluntary_station_arrivals"],
            "interval_backup_recoveries": interval["backup_recovery_count"],
            "backup_rate": interval["backup_rate"],
            "charging_fraction": interval["charging_fraction"],
            "generator_acceptance_rate": interval["generator_acceptance_rate"],
            "no_generator_rate": interval["no_generator_rate"],
            "rl_generator_fraction": interval["rl_generator_fraction"],
            "kappa_backup_fraction": interval["kappa_backup_fraction"],
            "charger_constrained_fraction": interval["charger_constrained_fraction"],
            "fail_closed_fraction": interval["fail_closed_fraction"],
            "minimum_energy_margin": interval["minimum_energy_margin"],
            "actor_loss": update.get("actor_loss"),
            "critic_loss_1": update.get("critic_loss_1"),
            "critic_loss_2": update.get("critic_loss_2"),
            "alpha": update.get("alpha"),
            "alpha_loss": update.get("alpha_loss"),
            "mean_log_prob_u": update.get("mean_log_prob_u"),
            "mean_negative_tanh_log_jacobian": update.get("mean_negative_tanh_log_jacobian"),
            "mean_log_det_G": update.get("mean_log_det_G"),
            "mean_negative_log_det_G": update.get("mean_negative_log_det_G"),
            "mean_normalized_log_prob": update.get("mean_normalized_log_prob"),
            "mean_physical_log_prob": update.get("mean_log_prob"),
            "entropy_target_residual": update.get("entropy_target_residual"),
            "alpha_gradient": update.get("alpha_gradient"),
            "accepted_batch_fraction": update.get("accepted_batch_fraction"),
            "kappa_target_fraction": float(update.get("kappa_target_count", 0) or 0) / target_count,
            "charger_target_fraction": float(update.get("charger_target_count", 0) or 0) / target_count,
            "rl_generator_target_fraction": float(update.get("rl_generator_target_count", 0) or 0) / target_count,
            "fail_closed_target_fraction": float(update.get("fail_closed_target_count", 0) or 0) / target_count,
            "gradient_steps": agent.gradient_steps,
            "replay_size": len(agent.replay),
        }

    for step in range(args.steps):
        actor_u = rng.normal(size=3) if step < args.warmup_steps else agent.select_u(observation)
        certificate_state_before = environment.runtime._certificate_state()
        candidate_action = environment._candidate_from_context(actor_u, context)
        next_observation, reward, terminated, truncated, info = environment.step(actor_u)
        snapshot = info["persistent_metrics"]
        delta = metric_snapshot_delta(previous_snapshot, snapshot)
        previous_snapshot = snapshot
        run_metrics.observe(reward, info, delta)
        episode_metrics.observe(reward, info, delta)
        interval_metrics.observe(reward, info, delta)
        telemetry = info["telemetry"]
        trajectory_records.append({
            "step": step + 1,
            "episode_id": episode_id,
            "episode_seed": episode_seed,
            "episode_step": int(info.get("episode_step", episode_metrics.total_steps)),
            "task_id": info.get("task_id"),
            "goal": None if info.get("current_goal") is None else np.asarray(info["current_goal"], dtype=float).tolist(),
            "task_completed_now": bool(info.get("task_completed_now", False)),
            "tasks_completed": int(info.get("tasks_completed", 0)),
            "persistent_mode": info.get("persistent_mode"),
            "execution_authority": info.get("execution_authority"),
            "execution_authority_reason": info.get("execution_authority_reason"),
            "accepted": bool(info.get("accepted", False)),
            "backup_triggered": bool(info.get("backup_triggered", False)),
            "backup_reason": info.get("backup_reason"),
            "voluntary_station_approach": bool(info.get("voluntary_station_approach", False)),
            "voluntary_station_arrival": bool(delta.get("voluntary_station_arrivals", 0.0) > 0.0),
            "charging": bool(info.get("charging", False)),
            "energy_charged": float(delta.get("energy_charged", 0.0)),
            "departure_attempt": bool(info.get("departure_attempt", False)),
            "departure_rejected": bool(info.get("departure_rejected", False)),
            "position": np.asarray(telemetry.state_after.position, dtype=float).tolist(),
            "velocity": np.asarray(telemetry.state_after.velocity, dtype=float).tolist(),
            "energy": float(telemetry.state_after.energy),
            "position_before": np.asarray(telemetry.state_before.position, dtype=float).tolist(),
            "velocity_before": np.asarray(telemetry.state_before.velocity, dtype=float).tolist(),
            "energy_before": float(telemetry.state_before.energy),
            "energy_error_radius": float(certificate_state_before.energy_error_radius),
            "energy_margin": float(info.get("energy_margin", np.nan)),
            "required_return_energy": float(info.get("required_return_energy", np.nan)),
            "goal_progress": float(info.get("goal_progress", 0.0)),
            "reward": float(reward),
            "reward_components": info.get("reward_components"),
            "actor_u": np.asarray(actor_u, dtype=float).tolist(),
            "actor_eta": np.tanh(np.asarray(actor_u, dtype=float)).tolist(),
            "generator_center": None if context.get("c") is None else np.asarray(context["c"], dtype=float).tolist(),
            "generator_matrix": None if context.get("G") is None else np.asarray(context["G"], dtype=float).tolist(),
            "candidate_action": None if candidate_action is None else np.asarray(candidate_action, dtype=float).tolist(),
            "executed_action": np.asarray(info.get("critic_action", telemetry.action_trace.published), dtype=float).tolist(),
            "measured_action": np.asarray(telemetry.action_trace.measured, dtype=float).tolist(),
            "generator_available": bool(context.get("generator_available", False)),
            "generator_executable": bool(context.get("generator_executable", False)),
            "recoverable_set_member": context.get("recoverable_set_member"),
            "rl_authority_set_member": context.get("rl_authority_set_member"),
            "recoverability_action_verified": context.get("recoverability_action_verified"),
            "continuation_action_verified": context.get("continuation_action_verified"),
            "kappa_valid": bool(context.get("certificate_valid", False) and context.get("kappa") is not None),
            "kappa_cell_id": context.get("recovery_cell_id"),
            "kappa_level": context.get("recovery_level"),
            "kappa_certificate_hash": context.get("recovery_hash"),
            "kappa_action": None if context.get("kappa") is None else np.asarray(context["kappa"], dtype=float).tolist(),
            "kappa_validation_failure_category": context.get("kappa_validation_failure_category"),
            "kappa_validation_failure_detail": context.get("kappa_validation_failure_detail"),
            "continuation_target_cell_id": context.get("continuation_target_cell_id"),
            "terminal_recovery_certificate_hash": context.get("terminal_recovery_certificate_hash"),
            "terminal_admissible": bool(telemetry.terminal_admissible),
            "departure_allowed": context.get("departure_allowed"),
            "station_hold_valid": context.get("station_hold_valid"),
            "charging_support_verified": context.get("charging_support_verified"),
            "command_source": info.get("command_source"),
            "fallback_reason": info.get("fallback_reason"),
            "atlas_hash": context.get("atlas_hash"),
            "certificate_valid": bool(context.get("certificate_valid", False)),
            "persistent_certificate_valid": bool(context.get("persistent_certificate_valid", False)),
            "certificate_epoch": context.get("certificate_epoch"),
            "geometry_version": context.get("geometry_version"),
            "dynamics_version": context.get("dynamics_version"),
            "tracking_version": context.get("tracking_version"),
            "energy_version": context.get("energy_version"),
            "terminal_version": context.get("terminal_version"),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "failure_reason": info.get("failure_reason"),
        })
        next_context = None if terminated or truncated else environment._refresh_context()
        if (
            info.get("accepted")
            and next_context is not None
            and not next_context.get("rl_authority_set_member")
            and next_context.get("persistent_mode") != "CHARGING_RL"
        ):
            environment.metrics.accepted_into_kappa_only_count += 1
        agent.observe(transition_from_cycle(
            observation, next_observation, actor_u, reward, terminated, truncated,
            episode_id, context, next_context, info,
        ))
        if len(agent.replay) >= args.batch_size and step >= args.warmup_steps:
            last_update = agent.update()
            critic_updates += 1
            actor_updates += int(last_update["actor_status"] == "updated")
            accepted_actor_samples += int(last_update["accepted_batch_count"])
        step_number = step + 1
        if step_number % args.log_interval == 0 or step_number == args.steps:
            curve_records.append(learning_curve_record(step_number))
            interval_metrics = PersistentMetricAccumulator()
        if terminated or truncated:
            episode_records.append(episode_record(
                episode_id,
                episode_seed,
                reset_info,
                episode_metrics,
                goal_sequence(),
                terminated=terminated,
                truncated=truncated,
                partial=False,
            ))
            episode_metrics = PersistentMetricAccumulator()
            previous_snapshot = None
            if step_number < args.steps:
                episode_id += 1
                episode_seed = args.seed + episode_id
                observation, reset_info = environment.reset(seed=episode_seed)
                context = reset_info["action_context"]
        else:
            observation = next_observation
            context = next_context

    if episode_metrics.total_steps:
        episode_records.append(episode_record(
            episode_id,
            episode_seed,
            reset_info,
            episode_metrics,
            goal_sequence(),
            terminated=False,
            truncated=False,
            partial=True,
        ))

    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "checkpoint_latest.pt"
    episodes_path = output / "episodes.jsonl"
    learning_curve_path = output / "learning_curve.jsonl"
    trajectory_path = output / "trajectory_events.jsonl"
    torch.save(agent.state_dict(), checkpoint_path)
    write_jsonl(episodes_path, episode_records)
    if not learning_curve_steps_monotonic(curve_records):
        raise RuntimeError("learning-curve steps are not strictly increasing")
    write_jsonl(learning_curve_path, curve_records)
    write_jsonl(trajectory_path, trajectory_records)
    aggregate = run_metrics.summary()
    summary = {
        "scenario": args.scenario,
        "legacy_fixed_graph": args.legacy_fixed_graph,
        "seed": args.seed,
        "steps": args.steps,
        "episodes": len(episode_records),
        "gradient_steps": agent.gradient_steps,
        "actor_updates": actor_updates,
        "critic_updates": critic_updates,
        "accepted_actor_samples": accepted_actor_samples,
        "total_reward": aggregate["total_reward"],
        "aggregate_metrics": aggregate,
        "last_episode_metrics": None if not episode_records else episode_records[-1],
        "last_update": last_update,
        "artifact_paths": {
            "episodes": str(episodes_path.relative_to(ROOT)),
            "learning_curve": str(learning_curve_path.relative_to(ROOT)),
            "trajectory_events": str(trajectory_path.relative_to(ROOT)),
            "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        },
        "synthetic_only": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
