from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from envs.certified_uav import (
    EnergyNavigationConfig,
    NavigationRewardConfig,
    PersistentEnergyNavigationEnv,
)
from scripts.sb3_navigation_harness import (
    NavigationMetricsCallback,
    append_jsonl,
    augment_attempt_record,
    initial_distance_statistics,
    latency_statistics,
)


def make_energy_environment(args, *, max_episode_steps: int | None = None) -> PersistentEnergyNavigationEnv:
    reward_config = NavigationRewardConfig(
        distance_potential_scale=args.distance_potential_scale,
        gamma=args.gamma,
        velocity_toward_goal_weight=args.velocity_reward_weight,
        time_cost=args.time_cost,
        task_completion_reward=args.completion_reward,
        collision_penalty=args.collision_penalty,
        energy_cost_weight=args.energy_cost_weight,
        backup_intervention_cost=args.backup_intervention_cost,
    )
    energy_config = EnergyNavigationConfig(
        battery_capacity=args.battery_capacity,
        charging_rate=args.charging_rate,
        charging_radius=args.charging_radius,
        charging_velocity_limit=tuple(args.charging_velocity_limit),
        initial_energy_fraction_min=args.initial_energy_fraction_min,
        initial_energy_fraction_max=args.initial_energy_fraction_max,
    )
    return PersistentEnergyNavigationEnv(
        args.scenario,
        max_episode_steps=args.max_episode_steps if max_episode_steps is None else max_episode_steps,
        goal_radius=args.goal_radius,
        minimum_goal_separation=args.minimum_goal_separation,
        sampling_margin=args.sampling_margin,
        reward_config=reward_config,
        energy_config=energy_config,
    )


class EnergyMetricsCallback(NavigationMetricsCallback):
    def __init__(self, args, output_dir: Path) -> None:
        super().__init__(args, output_dir)
        self.total_flight_energy = 0.0
        self.total_charge_received = 0.0
        self.minimum_soc_global = 1.0
        self.soc_sum_global = 0.0
        self.soc_samples_global = 0
        self.station_visits = 0
        self.charging_sessions = 0
        self.successful_charging_sessions = 0
        self.successful_resumes = 0
        self.energy_stranding_events = 0
        self.stranded_episodes = 0
        self.charging_session_records: list[dict[str, Any]] = []

    def snapshot(self) -> dict[str, Any]:
        base = super().snapshot()
        entries = [record["charge_start_soc"] for record in self.charging_session_records]
        departures = [
            record["departure_soc"]
            for record in self.charging_session_records
            if record["status"] == "departed"
        ]
        durations = [record["charging_duration_steps"] for record in self.charging_session_records]
        tasks_between = [record["tasks_between_charges"] for record in self.charging_session_records]
        base["energy_metrics"] = {
            "total_flight_energy": self.total_flight_energy,
            "total_charge_received": self.total_charge_received,
            "energy_per_completed_task": (
                self.total_flight_energy / self.global_tasks_completed
                if self.global_tasks_completed > 0
                else None
            ),
            "minimum_soc": self.minimum_soc_global,
            "mean_soc": self.soc_sum_global / max(1, self.soc_samples_global),
            "station_visit_count": self.station_visits,
            "charging_visit_rate_per_1000_steps": 1000.0 * self.station_visits / max(1, self.num_timesteps),
            "charging_session_count": self.charging_sessions,
            "voluntary_charging_session_count": self.charging_sessions,
            "successful_charging_session_count": self.successful_charging_sessions,
            "successful_charge_rate": self.successful_charging_sessions / max(1, self.charging_sessions),
            "mean_soc_at_charge_entry": float(np.mean(entries)) if entries else None,
            "mean_soc_at_departure": float(np.mean(departures)) if departures else None,
            "mean_charge_duration_steps": float(np.mean(durations)) if durations else None,
            "mean_charge_duration_seconds": (
                float(np.mean(durations)) * self.args.dt if durations else None
            ),
            "mean_tasks_between_charges": float(np.mean(tasks_between)) if tasks_between else None,
            "successful_resume_count": self.successful_resumes,
            "task_resume_success_rate": self.successful_resumes / max(1, self.charging_sessions),
            "energy_stranded_count": self.energy_stranding_events,
            "stranding_rate_per_1000_steps": 1000.0 * self.energy_stranding_events / max(1, self.num_timesteps),
            "stranded_episode_count": self.stranded_episodes,
            "stranded_episode_rate": self.stranded_episodes / max(1, self.episode_index),
        }
        return base

    def _on_step(self) -> bool:
        info = self.locals["infos"][0]
        self.total_flight_energy += float(info["flight_energy_used"])
        self.total_charge_received += float(info["gross_charge_received"])
        soc = float(info["state_of_charge"])
        self.minimum_soc_global = min(self.minimum_soc_global, soc)
        self.soc_sum_global += soc
        self.soc_samples_global += 1
        self.station_visits += int(info["station_visit_now"])
        self.charging_sessions += int(info["charging_session_started_now"])
        self.energy_stranding_events += int(info["energy_stranded_now"])
        if info["stranding_event"] is not None:
            event = dict(info["stranding_event"])
            event |= {"training_seed": self.args.seed, "training_step": self.num_timesteps}
            append_jsonl(self.output_dir / "energy_stranding_events.jsonl", event)
        for record in info["charging_session_records"]:
            augmented = dict(record) | {
                "training_seed": self.args.seed,
                "recorded_training_step": self.num_timesteps,
            }
            self.charging_session_records.append(augmented)
            self.successful_charging_sessions += int(augmented["successful_charge"])
            self.successful_resumes += int(augmented["successful_resume"])
            append_jsonl(self.output_dir / "charging_sessions.jsonl", augmented)
        if bool(self.locals["dones"][0]):
            self.stranded_episodes += int(info["energy_stranded_count"] > 0)
            append_jsonl(
                self.output_dir / "energy_episodes.jsonl",
                {
                    "training_seed": self.args.seed,
                    "training_step": self.num_timesteps,
                    "minimum_soc": info["minimum_soc"],
                    "mean_soc": info["mean_soc"],
                    "station_visit_count": info["station_visit_count"],
                    "charging_session_count": info["charging_session_count"],
                    "successful_charging_session_count": info["successful_charging_session_count"],
                    "successful_resume_count": info["successful_resume_count"],
                    "energy_stranded_count": info["energy_stranded_count"],
                    "flight_energy": info["cumulative_energy_usage"],
                    "tasks_completed": info["tasks_completed"],
                },
            )
        return super()._on_step()


def _energy_eval_sampling_seed(args, actual_step: int, heldout_seed: int, mode: str, soc_group: str) -> int:
    offsets = {
        ("deterministic", "full"): 0,
        ("deterministic", "low_soc"): 10_000_000,
        ("stochastic", "full"): 20_000_000,
        ("stochastic", "low_soc"): 30_000_000,
    }
    return int(args.evaluation_seed_base + args.seed * 100_000_000 + actual_step * 10 + heldout_seed + offsets[(mode, soc_group)])


def evaluate_energy_model(
    model,
    args,
    requested_step: int,
    actual_step: int,
    *,
    evaluation_mode: str,
    soc_group: str,
) -> dict[str, Any]:
    if evaluation_mode not in {"deterministic", "stochastic"}:
        raise ValueError("energy evaluation mode must be deterministic or stochastic")
    if soc_group not in {"full", "low_soc"}:
        raise ValueError("unknown SOC evaluation group")
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rows = []
    try:
        for heldout_seed in args.heldout_seeds:
            sampling_seed = _energy_eval_sampling_seed(
                args, actual_step, heldout_seed, evaluation_mode, soc_group
            )
            torch.manual_seed(sampling_seed)
            np.random.seed(sampling_seed % (2**32 - 1))
            environment = make_energy_environment(args, max_episode_steps=args.evaluation_steps)
            reset_options = (
                {"initial_energy_fraction": 1.0}
                if soc_group == "full"
                else {"initial_energy_fraction_range": [0.30, 0.45]}
            )
            observation, reset_info = environment.reset(seed=heldout_seed, options=reset_options)
            total_return = 0.0
            completed_records = []
            charge_records = []
            stranding_events = []
            signed_velocity_sum = 0.0
            for evaluation_step in range(1, args.evaluation_steps + 1):
                action, _ = model.predict(
                    observation,
                    deterministic=evaluation_mode == "deterministic",
                )
                observation, reward, terminated, truncated, info = environment.step(action)
                if terminated:
                    raise RuntimeError("finite-energy environment terminated unexpectedly")
                if not np.isfinite(reward) or not np.all(np.isfinite(observation)):
                    raise FloatingPointError("nonfinite finite-energy evaluation transition")
                total_return += float(reward)
                signed_velocity_sum += float(info["signed_velocity_toward_goal"])
                for record in info["goal_attempt_records"]:
                    if record["completed"]:
                        completed_records.append(
                            augment_attempt_record(record, evaluation_step, step_prefix="evaluation")
                        )
                charge_records.extend(info["charging_session_records"])
                if info["stranding_event"] is not None:
                    stranding_events.append(info["stranding_event"])
                if truncated:
                    break
            departed = [record for record in charge_records if record["status"] == "departed"]
            successful = [record for record in charge_records if record["successful_charge"]]
            resumed = [record for record in charge_records if record["successful_resume"]]
            steps = environment.episode_step
            rows.append({
                "training_seed": args.seed,
                "heldout_environment_seed": heldout_seed,
                "policy_sampling_seed": sampling_seed if evaluation_mode == "stochastic" else None,
                "requested_checkpoint_step": requested_step,
                "actual_checkpoint_step": actual_step,
                "evaluation_mode": evaluation_mode,
                "soc_group": soc_group,
                "initial_soc": reset_info["initial_soc"],
                "steps": steps,
                "tasks_completed": environment.tasks_completed,
                "tasks_per_1000_steps": 1000.0 * environment.tasks_completed / max(1, steps),
                "episode_return": total_return,
                "goal_latency": latency_statistics(completed_records),
                "completed_goal_records": completed_records,
                "minimum_goal_distance": environment.minimum_goal_distance,
                "mean_goal_distance": environment.goal_distance_sum / max(1, environment.goal_distance_samples),
                "mean_signed_velocity_toward_goal": signed_velocity_sum / max(1, steps),
                "collision_count": environment.collision_count,
                "collision_rate": environment.collision_count / max(1, steps),
                "velocity_saturation_count": environment.velocity_saturation_count,
                "velocity_saturation_rate": environment.velocity_saturation_count / max(1, steps),
                "boundary_lock_event_count": environment.boundary_lock_event_count,
                "maximum_consecutive_boundary_contacts": environment.maximum_consecutive_boundary_contacts,
                "total_flight_energy": environment.cumulative_energy_usage,
                "energy_per_completed_task": (
                    environment.cumulative_energy_usage / environment.tasks_completed
                    if environment.tasks_completed > 0
                    else None
                ),
                "minimum_soc": environment.minimum_soc,
                "mean_soc": environment.soc_sum / max(1, environment.soc_samples),
                "station_visit_count": environment.station_visit_count,
                "charging_visit_rate_per_1000_steps": 1000.0 * environment.station_visit_count / max(1, steps),
                "charging_session_count": environment.charging_session_count,
                "successful_charging_session_count": len(successful),
                "successful_charge_rate": len(successful) / max(1, environment.charging_session_count),
                "mean_soc_at_charge_entry": float(np.mean([r["charge_start_soc"] for r in charge_records])) if charge_records else None,
                "mean_soc_at_departure": float(np.mean([r["departure_soc"] for r in departed])) if departed else None,
                "mean_charge_duration_steps": float(np.mean([r["charging_duration_steps"] for r in charge_records])) if charge_records else None,
                "mean_tasks_between_charges": float(np.mean([r["tasks_between_charges"] for r in charge_records])) if charge_records else None,
                "successful_resume_count": len(resumed),
                "task_resume_success_rate": len(resumed) / max(1, environment.charging_session_count),
                "energy_stranded_count": len(stranding_events),
                "stranded": len(stranding_events) > 0,
                "charging_sessions": charge_records,
                "stranding_events": stranding_events,
            })
            environment.close()
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    total_steps = sum(row["steps"] for row in rows)
    total_tasks = sum(row["tasks_completed"] for row in rows)
    total_sessions = sum(row["charging_session_count"] for row in rows)
    total_successful_charges = sum(row["successful_charging_session_count"] for row in rows)
    total_resumes = sum(row["successful_resume_count"] for row in rows)
    total_stranded = sum(row["stranded"] for row in rows)
    entries = [record["charge_start_soc"] for row in rows for record in row["charging_sessions"]]
    departed = [record for row in rows for record in row["charging_sessions"] if record["status"] == "departed"]
    durations = [record["charging_duration_steps"] for row in rows for record in row["charging_sessions"]]
    all_completed = [record for row in rows for record in row["completed_goal_records"]]
    return {
        "training_seed": args.seed,
        "requested_checkpoint_step": requested_step,
        "actual_checkpoint_step": actual_step,
        "evaluation_mode": evaluation_mode,
        "soc_group": soc_group,
        "heldout_environment_seeds": list(args.heldout_seeds),
        "seed_results": rows,
        "aggregate": {
            "steps": total_steps,
            "tasks_completed": total_tasks,
            "tasks_per_1000_steps": 1000.0 * total_tasks / max(1, total_steps),
            "goal_latency": latency_statistics(all_completed),
            "mean_goal_distance": float(np.mean([row["mean_goal_distance"] for row in rows])),
            "mean_minimum_goal_distance": float(np.mean([row["minimum_goal_distance"] for row in rows])),
            "collision_rate": sum(row["collision_count"] for row in rows) / max(1, total_steps),
            "velocity_saturation_rate": sum(row["velocity_saturation_count"] for row in rows) / max(1, total_steps),
            "boundary_lock_event_count": sum(row["boundary_lock_event_count"] for row in rows),
            "maximum_consecutive_boundary_contacts": max(row["maximum_consecutive_boundary_contacts"] for row in rows),
            "charging_visit_rate_per_1000_steps": 1000.0 * sum(row["station_visit_count"] for row in rows) / max(1, total_steps),
            "charging_session_count": total_sessions,
            "successful_charge_rate": total_successful_charges / max(1, total_sessions),
            "stranding_rate": total_stranded / max(1, len(rows)),
            "mean_soc_at_charge_entry": float(np.mean(entries)) if entries else None,
            "mean_soc_at_departure": float(np.mean([r["departure_soc"] for r in departed])) if departed else None,
            "mean_charge_duration_steps": float(np.mean(durations)) if durations else None,
            "mean_tasks_between_charges": float(np.mean([r["tasks_between_charges"] for row in rows for r in row["charging_sessions"]])) if total_sessions else None,
            "task_resume_success_rate": total_resumes / max(1, total_sessions),
            "energy_per_completed_task": (
                sum(row["total_flight_energy"] for row in rows) / total_tasks
                if total_tasks > 0
                else None
            ),
        },
    }
