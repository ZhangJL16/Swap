from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback

from envs.certified_uav import NavigationRewardConfig, PersistentNavigationEnv


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(jsonable(payload), sort_keys=True) + "\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def latency_statistics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    values = np.asarray(
        [float(record["steps_to_goal"]) for record in records if record.get("completed")],
        dtype=np.float64,
    )
    if values.size == 0:
        return {
            "completed_goal_count": 0,
            "mean_steps_per_completed_goal": None,
            "median_steps_per_completed_goal": None,
            "p25_steps_per_completed_goal": None,
            "p75_steps_per_completed_goal": None,
        }
    return {
        "completed_goal_count": int(values.size),
        "mean_steps_per_completed_goal": float(np.mean(values)),
        "median_steps_per_completed_goal": float(np.median(values)),
        "p25_steps_per_completed_goal": float(np.percentile(values, 25)),
        "p75_steps_per_completed_goal": float(np.percentile(values, 75)),
    }


def initial_distance_statistics(records: list[dict[str, Any]]) -> dict[str, float | None]:
    if not records:
        return {
            "mean_initial_goal_distance": None,
            "mean_initial_xy_distance": None,
            "mean_initial_z_distance": None,
        }
    return {
        "mean_initial_goal_distance": float(np.mean([record["goal_initial_distance"] for record in records])),
        "mean_initial_xy_distance": float(np.mean([record["initial_xy_distance"] for record in records])),
        "mean_initial_z_distance": float(np.mean([record["initial_z_distance"] for record in records])),
    }


def augment_attempt_record(
    record: dict[str, Any],
    end_step: int,
    *,
    step_prefix: str,
) -> dict[str, Any]:
    augmented = dict(record)
    attempt_steps = int(record["attempt_steps"])
    augmented[f"goal_start_{step_prefix}_step"] = int(end_step - attempt_steps)
    augmented[f"attempt_end_{step_prefix}_step"] = int(end_step)
    augmented[f"completion_{step_prefix}_step"] = int(end_step) if record.get("completed") else None
    return augmented


def make_navigation_environment(args, *, max_episode_steps: int | None = None) -> PersistentNavigationEnv:
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
    return PersistentNavigationEnv(
        args.scenario,
        max_episode_steps=args.max_episode_steps if max_episode_steps is None else max_episode_steps,
        navigation_energy_capacity=args.navigation_energy_capacity,
        goal_radius=args.goal_radius,
        minimum_goal_separation=args.minimum_goal_separation,
        sampling_margin=args.sampling_margin,
        reward_config=reward_config,
    )


def policy_sampling_seed(args, actual_step: int, heldout_seed: int, mode_offset: int = 0) -> int:
    return int(
        args.evaluation_seed_base
        + args.seed * 10_000_000
        + actual_step * 10
        + heldout_seed
        + mode_offset
    )


def rollout_aligned_step_at_or_after(requested_step: int, rollout_steps: int) -> int:
    if requested_step <= 0 or rollout_steps <= 0:
        raise ValueError("requested_step and rollout_steps must be positive")
    return int(((requested_step + rollout_steps - 1) // rollout_steps) * rollout_steps)


def evaluate_navigation_model(
    model,
    args,
    requested_step: int,
    actual_step: int,
    *,
    evaluation_mode: str,
) -> dict[str, Any]:
    if evaluation_mode not in {"deterministic", "stochastic", "ddpg_exploration_noise"}:
        raise ValueError(f"unsupported evaluation mode {evaluation_mode}")
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    seed_rows: list[dict[str, Any]] = []
    all_completed: list[dict[str, Any]] = []
    all_attempts: list[dict[str, Any]] = []
    all_boundary_locks: list[dict[str, Any]] = []
    try:
        for heldout_seed in args.heldout_seeds:
            mode_offset = 90_000_000 if evaluation_mode == "ddpg_exploration_noise" else 0
            sampling_seed = None
            if evaluation_mode != "deterministic":
                sampling_seed = policy_sampling_seed(args, actual_step, heldout_seed, mode_offset)
                torch.manual_seed(sampling_seed)
                np.random.seed(sampling_seed % (2**32 - 1))
            environment = make_navigation_environment(args, max_episode_steps=args.evaluation_steps)
            observation, _ = environment.reset(seed=heldout_seed)
            total_return = 0.0
            signed_velocity_sum = 0.0
            action_abs_sum = np.zeros(3, dtype=np.float64)
            action_squared_sum = np.zeros(3, dtype=np.float64)
            stream_completed: list[dict[str, Any]] = []
            stream_attempts: list[dict[str, Any]] = []
            stream_boundary_locks: list[dict[str, Any]] = []
            for evaluation_step in range(1, args.evaluation_steps + 1):
                deterministic_prediction = evaluation_mode != "stochastic"
                action, _ = model.predict(observation, deterministic=deterministic_prediction)
                action = np.asarray(action, dtype=np.float32)
                if evaluation_mode == "ddpg_exploration_noise":
                    action = np.clip(
                        action + np.random.normal(0.0, args.action_noise_sigma, size=3),
                        -1.0,
                        1.0,
                    ).astype(np.float32)
                observation, reward, terminated, truncated, info = environment.step(action)
                if terminated:
                    raise RuntimeError("held-out navigation environment terminated unexpectedly")
                if not np.isfinite(reward) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(observation)):
                    raise FloatingPointError("nonfinite held-out transition")
                total_return += float(reward)
                signed_velocity_sum += float(info["signed_velocity_toward_goal"])
                action_abs_sum += np.abs(action)
                action_squared_sum += action * action
                for record in info["goal_attempt_records"]:
                    augmented = augment_attempt_record(record, evaluation_step, step_prefix="evaluation")
                    augmented |= {
                        "training_seed": args.seed,
                        "heldout_environment_seed": heldout_seed,
                        "policy_sampling_seed": sampling_seed,
                        "requested_checkpoint_step": requested_step,
                        "actual_checkpoint_step": actual_step,
                        "evaluation_mode": evaluation_mode,
                    }
                    stream_attempts.append(augmented)
                    if augmented["completed"]:
                        stream_completed.append(augmented)
                if info["boundary_lock_event"] is not None:
                    event = dict(info["boundary_lock_event"])
                    event |= {
                        "training_seed": args.seed,
                        "heldout_environment_seed": heldout_seed,
                        "policy_sampling_seed": sampling_seed,
                        "requested_checkpoint_step": requested_step,
                        "actual_checkpoint_step": actual_step,
                        "evaluation_mode": evaluation_mode,
                    }
                    stream_boundary_locks.append(event)
                if truncated:
                    break
            steps = environment.episode_step
            row = {
                "training_seed": args.seed,
                "heldout_environment_seed": heldout_seed,
                "policy_sampling_seed": sampling_seed,
                "requested_checkpoint_step": requested_step,
                "actual_checkpoint_step": actual_step,
                "evaluation_mode": evaluation_mode,
                "steps": steps,
                "tasks_completed": environment.tasks_completed,
                "tasks_per_1000_steps": 1000.0 * environment.tasks_completed / max(1, steps),
                "successful_goal_attempts": len(stream_completed),
                "stream_success": len(stream_completed) > 0,
                "episode_return": total_return,
                "minimum_goal_distance": environment.minimum_goal_distance,
                "mean_goal_distance": environment.goal_distance_sum / max(1, environment.goal_distance_samples),
                "mean_signed_velocity_toward_goal": signed_velocity_sum / max(1, steps),
                "collision_count": environment.collision_count,
                "collision_rate": environment.collision_count / max(1, steps),
                "boundary_collision_count": environment.boundary_collision_count,
                "boundary_collision_rate": environment.boundary_collision_count / max(1, steps),
                "velocity_saturation_count": environment.velocity_saturation_count,
                "velocity_saturation_rate": environment.velocity_saturation_count / max(1, steps),
                "boundary_lock_event_count": len(stream_boundary_locks),
                "maximum_consecutive_boundary_contacts": environment.maximum_consecutive_boundary_contacts,
                "mean_abs_action": action_abs_sum / max(1, steps),
                "rms_action": np.sqrt(action_squared_sum / max(1, steps)),
                "goal_latency": latency_statistics(stream_completed),
                "goal_attempts": stream_attempts,
                "boundary_lock_events": stream_boundary_locks,
            }
            seed_rows.append(row)
            all_completed.extend(stream_completed)
            all_attempts.extend(stream_attempts)
            all_boundary_locks.extend(stream_boundary_locks)
            environment.close()
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    total_steps = sum(row["steps"] for row in seed_rows)
    total_tasks = sum(row["tasks_completed"] for row in seed_rows)
    total_collisions = sum(row["collision_count"] for row in seed_rows)
    total_boundary = sum(row["boundary_collision_count"] for row in seed_rows)
    total_saturations = sum(row["velocity_saturation_count"] for row in seed_rows)
    aggregate = {
        "evaluation_mode": evaluation_mode,
        "training_seed": args.seed,
        "requested_checkpoint_step": requested_step,
        "actual_checkpoint_step": actual_step,
        "evaluation_stream_count": len(seed_rows),
        "successful_evaluation_streams": sum(bool(row["stream_success"]) for row in seed_rows),
        "stream_success_rate": float(np.mean([row["stream_success"] for row in seed_rows])),
        "steps": total_steps,
        "tasks_completed": total_tasks,
        "tasks_per_1000_steps": 1000.0 * total_tasks / max(1, total_steps),
        "mean_episode_return": float(np.mean([row["episode_return"] for row in seed_rows])),
        "mean_goal_distance": float(np.mean([row["mean_goal_distance"] for row in seed_rows])),
        "mean_minimum_goal_distance": float(np.mean([row["minimum_goal_distance"] for row in seed_rows])),
        "mean_signed_velocity_toward_goal": float(
            np.mean([row["mean_signed_velocity_toward_goal"] for row in seed_rows])
        ),
        "collision_count": total_collisions,
        "collision_rate": total_collisions / max(1, total_steps),
        "boundary_collision_count": total_boundary,
        "boundary_collision_rate": total_boundary / max(1, total_steps),
        "velocity_saturation_count": total_saturations,
        "velocity_saturation_rate": total_saturations / max(1, total_steps),
        "boundary_lock_event_count": len(all_boundary_locks),
        "maximum_consecutive_boundary_contacts": max(
            row["maximum_consecutive_boundary_contacts"] for row in seed_rows
        ),
        "goal_latency": latency_statistics(all_completed),
        "goal_initial_distances": initial_distance_statistics(all_attempts),
    }
    return {
        "training_seed": args.seed,
        "requested_checkpoint_step": requested_step,
        "actual_checkpoint_step": actual_step,
        "evaluation_mode": evaluation_mode,
        "heldout_environment_seeds": list(args.heldout_seeds),
        "seed_results": seed_rows,
        "aggregate": aggregate,
    }


class NavigationMetricsCallback(BaseCallback):
    def __init__(self, args, output_dir: Path) -> None:
        super().__init__(verbose=0)
        self.args = args
        self.output_dir = output_dir
        self.global_tasks_completed = 0
        self.global_collision_count = 0
        self.global_boundary_collision_count = 0
        self.global_velocity_saturation_count = 0
        self.global_energy_usage = 0.0
        self.global_goal_progress = 0.0
        self.global_signed_velocity = 0.0
        self.global_minimum_goal_distance = float("inf")
        self.global_goal_distance_sum = 0.0
        self.action_sum = np.zeros(3, dtype=np.float64)
        self.action_abs_sum = np.zeros(3, dtype=np.float64)
        self.action_squared_sum = np.zeros(3, dtype=np.float64)
        self.action_min = np.full(3, np.inf)
        self.action_max = np.full(3, -np.inf)
        self.reward_component_totals: dict[str, float] = {}
        self.episode_return = 0.0
        self.episode_steps = 0
        self.episode_index = 0
        self.completed_goal_records: list[dict[str, Any]] = []
        self.all_goal_attempt_records: list[dict[str, Any]] = []
        self.completion_training_steps: list[int] = []
        self.boundary_lock_event_count = 0
        self.maximum_consecutive_boundary_contacts = 0

    def assert_finite_training_state(self) -> None:
        if any(not torch.isfinite(parameter).all() for parameter in self.model.policy.parameters()):
            raise FloatingPointError("nonfinite SB3 policy parameters")
        for name in ("actor", "critic", "critic_target"):
            module = getattr(self.model, name, None)
            if module is not None and any(not torch.isfinite(parameter).all() for parameter in module.parameters()):
                raise FloatingPointError(f"nonfinite SB3 {name} parameters")
        for name, value in self.model.logger.name_to_value.items():
            if name.startswith("train/") and isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                raise FloatingPointError(f"nonfinite SB3 metric {name}={value}")

    def _window_task_rate(self, window_steps: int) -> float:
        lower_bound = self.num_timesteps - window_steps
        count = sum(step > lower_bound for step in self.completion_training_steps)
        denominator = min(window_steps, self.num_timesteps)
        return 1000.0 * count / max(1, denominator)

    def snapshot(self) -> dict[str, Any]:
        steps = max(1, self.num_timesteps)
        logger_values = {
            name: value
            for name, value in self.model.logger.name_to_value.items()
            if name.startswith("train/") and isinstance(value, (int, float, np.number))
        }
        return {
            "actual_training_step": self.num_timesteps,
            "tasks_completed": self.global_tasks_completed,
            "tasks_per_1000_steps": 1000.0 * self.global_tasks_completed / steps,
            "last_10000_tasks_per_1000_steps": self._window_task_rate(10_000),
            "last_50000_tasks_per_1000_steps": self._window_task_rate(50_000),
            "goal_latency": latency_statistics(self.completed_goal_records),
            "goal_initial_distances": initial_distance_statistics(self.all_goal_attempt_records),
            "mean_goal_progress": self.global_goal_progress / steps,
            "minimum_goal_distance": self.global_minimum_goal_distance,
            "mean_goal_distance": self.global_goal_distance_sum / steps,
            "mean_signed_velocity_toward_goal": self.global_signed_velocity / steps,
            "collision_count": self.global_collision_count,
            "collision_rate": self.global_collision_count / steps,
            "boundary_collision_count": self.global_boundary_collision_count,
            "boundary_collision_rate": self.global_boundary_collision_count / steps,
            "velocity_saturation_count": self.global_velocity_saturation_count,
            "velocity_saturation_rate": self.global_velocity_saturation_count / steps,
            "boundary_lock_event_count": self.boundary_lock_event_count,
            "maximum_consecutive_boundary_contacts": self.maximum_consecutive_boundary_contacts,
            "energy_usage": self.global_energy_usage,
            "reward_component_totals": dict(self.reward_component_totals),
            "episode_index": self.episode_index,
            "current_episode_return": self.episode_return,
            "action_mean": self.action_sum / steps,
            "action_mean_abs": self.action_abs_sum / steps,
            "action_rms": np.sqrt(self.action_squared_sum / steps),
            "action_min": self.action_min,
            "action_max": self.action_max,
            "sb3_train_metrics": logger_values,
        }

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        actions = np.asarray(self.locals["actions"], dtype=np.float64)
        info = self.locals["infos"][0]
        if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(actions)):
            raise FloatingPointError("nonfinite reward or action in SB3 rollout")
        action = np.asarray(info["normalized_action"], dtype=np.float64)
        self.global_tasks_completed += int(info["task_completed_now"])
        self.global_collision_count += int(info["collision"])
        self.global_boundary_collision_count += int(info["boundary_collision"])
        self.global_velocity_saturation_count += int(info["velocity_saturated"])
        self.global_energy_usage += float(info["energy_usage"])
        self.global_goal_progress += float(info["goal_progress"])
        self.global_signed_velocity += float(info["signed_velocity_toward_goal"])
        self.global_goal_distance_sum += float(info["distance_to_goal_after"])
        self.global_minimum_goal_distance = min(
            self.global_minimum_goal_distance,
            float(info["distance_to_goal_after"]),
        )
        self.maximum_consecutive_boundary_contacts = max(
            self.maximum_consecutive_boundary_contacts,
            int(info["maximum_consecutive_boundary_contacts"]),
        )
        for name, value in info["reward_components"].items():
            self.reward_component_totals[name] = self.reward_component_totals.get(name, 0.0) + float(value)
        self.action_sum += action
        self.action_abs_sum += np.abs(action)
        self.action_squared_sum += action * action
        self.action_min = np.minimum(self.action_min, action)
        self.action_max = np.maximum(self.action_max, action)
        self.episode_return += float(rewards[0])
        self.episode_steps += 1

        for record in info["goal_attempt_records"]:
            augmented = augment_attempt_record(record, self.num_timesteps, step_prefix="training")
            augmented["training_seed"] = self.args.seed
            self.all_goal_attempt_records.append(augmented)
            append_jsonl(self.output_dir / "goal_attempts.jsonl", augmented)
            if augmented["completed"]:
                self.completed_goal_records.append(augmented)
                self.completion_training_steps.append(self.num_timesteps)
                append_jsonl(self.output_dir / "goal_completions.jsonl", augmented)
        if info["boundary_lock_event"] is not None:
            event = dict(info["boundary_lock_event"])
            event |= {"training_seed": self.args.seed, "training_step": self.num_timesteps}
            self.boundary_lock_event_count += 1
            append_jsonl(self.output_dir / "boundary_lock_events.jsonl", event)

        if bool(self.locals["dones"][0]):
            append_jsonl(
                self.output_dir / "episodes.jsonl",
                {
                    "episode_index": self.episode_index,
                    "actual_training_step": self.num_timesteps,
                    "episode_steps": self.episode_steps,
                    "episode_return": self.episode_return,
                    "tasks_completed": info["tasks_completed"],
                    "collision_count": info["collision_count"],
                    "boundary_collision_count": info["boundary_collision_count"],
                    "velocity_saturation_count": info["velocity_saturation_count"],
                    "boundary_lock_event_count": info["boundary_lock_event_count"],
                    "maximum_consecutive_boundary_contacts": info[
                        "maximum_consecutive_boundary_contacts"
                    ],
                    "energy_usage": info["cumulative_energy_usage"],
                    "minimum_goal_distance": info["minimum_goal_distance"],
                    "mean_goal_distance": info["mean_goal_distance"],
                },
            )
            self.episode_index += 1
            self.episode_return = 0.0
            self.episode_steps = 0
        if self.num_timesteps % self.args.log_interval == 0:
            self.assert_finite_training_state()
            append_jsonl(self.output_dir / "learning_curve.jsonl", self.snapshot())
        return True


def model_device_metadata(model, requested_device: str) -> dict[str, Any]:
    metadata = {
        "requested_device": requested_device,
        "model_device": str(model.device),
        "policy_parameter_devices": sorted({str(parameter.device) for parameter in model.policy.parameters()}),
    }
    for name in ("actor", "critic", "critic_target"):
        module = getattr(model, name, None)
        if module is not None:
            metadata[f"{name}_parameter_devices"] = sorted(
                {str(parameter.device) for parameter in module.parameters()}
            )
    return metadata
