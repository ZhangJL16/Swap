#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any

import gymnasium
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import NavigationRewardConfig, PersistentNavigationEnv


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_jsonable(payload), sort_keys=True) + "\n")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _latency_statistics(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
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


def _initial_distance_statistics(records: list[dict[str, Any]]) -> dict[str, float | None]:
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


def _augment_attempt_record(
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


def make_environment(args: argparse.Namespace, *, max_episode_steps: int | None = None) -> PersistentNavigationEnv:
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


def _policy_sampling_seed(args: argparse.Namespace, training_step: int, heldout_seed: int) -> int:
    return int(
        args.stochastic_policy_seed_base
        + args.seed * 10_000_000
        + training_step * 10
        + heldout_seed
    )


def evaluate_model(
    model,
    args: argparse.Namespace,
    training_step: int,
    *,
    deterministic: bool,
) -> dict[str, Any]:
    torch_state = torch.random.get_rng_state()
    numpy_state = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    mode_name = "deterministic" if deterministic else "stochastic"
    seed_rows: list[dict[str, Any]] = []
    all_completed_records: list[dict[str, Any]] = []
    all_attempt_records: list[dict[str, Any]] = []
    all_boundary_lock_events: list[dict[str, Any]] = []
    try:
        for heldout_seed in args.heldout_seeds:
            policy_seed = None if deterministic else _policy_sampling_seed(args, training_step, heldout_seed)
            if policy_seed is not None:
                torch.manual_seed(policy_seed)
                np.random.seed(policy_seed % (2**32 - 1))
            environment = make_environment(args, max_episode_steps=args.evaluation_steps)
            observation, _ = environment.reset(seed=heldout_seed)
            total_return = 0.0
            progress_sum = 0.0
            signed_velocity_sum = 0.0
            action_abs_sum = np.zeros(3, dtype=np.float64)
            action_squared_sum = np.zeros(3, dtype=np.float64)
            stream_completed: list[dict[str, Any]] = []
            stream_attempts: list[dict[str, Any]] = []
            stream_boundary_locks: list[dict[str, Any]] = []
            for evaluation_step in range(1, args.evaluation_steps + 1):
                action, _ = model.predict(observation, deterministic=deterministic)
                observation, reward, terminated, truncated, info = environment.step(action)
                if terminated:
                    raise RuntimeError("held-out navigation environment terminated unexpectedly")
                if not np.isfinite(reward) or not np.all(np.isfinite(action)) or not np.all(np.isfinite(observation)):
                    raise FloatingPointError("nonfinite held-out transition")
                total_return += float(reward)
                progress_sum += float(info["goal_progress"])
                signed_velocity_sum += float(info["signed_velocity_toward_goal"])
                action_array = np.asarray(action, dtype=np.float64)
                action_abs_sum += np.abs(action_array)
                action_squared_sum += action_array * action_array
                for record in info["goal_attempt_records"]:
                    augmented = _augment_attempt_record(record, evaluation_step, step_prefix="evaluation")
                    augmented |= {
                        "training_seed": args.seed,
                        "heldout_environment_seed": heldout_seed,
                        "policy_sampling_seed": policy_seed,
                        "checkpoint_step": training_step,
                        "evaluation_mode": mode_name,
                    }
                    stream_attempts.append(augmented)
                    if augmented["completed"]:
                        stream_completed.append(augmented)
                if info["boundary_lock_event"] is not None:
                    event = dict(info["boundary_lock_event"])
                    event |= {
                        "training_seed": args.seed,
                        "heldout_environment_seed": heldout_seed,
                        "policy_sampling_seed": policy_seed,
                        "checkpoint_step": training_step,
                        "evaluation_mode": mode_name,
                    }
                    stream_boundary_locks.append(event)
                if truncated:
                    break
            steps = environment.episode_step
            row = {
                "training_seed": args.seed,
                "heldout_environment_seed": heldout_seed,
                "policy_sampling_seed": policy_seed,
                "checkpoint_step": training_step,
                "evaluation_mode": mode_name,
                "steps": steps,
                "tasks_completed": environment.tasks_completed,
                "tasks_per_1000_steps": 1000.0 * environment.tasks_completed / max(1, steps),
                "successful_goal_attempts": len(stream_completed),
                "goal_attempt_count": len(stream_attempts),
                "stream_success": len(stream_completed) > 0,
                "episode_return": total_return,
                "mean_goal_progress": progress_sum / max(1, steps),
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
                "energy_usage": environment.cumulative_energy_usage,
                "mean_abs_action": action_abs_sum / max(1, steps),
                "rms_action": np.sqrt(action_squared_sum / max(1, steps)),
                "goal_latency": _latency_statistics(stream_completed),
                "goal_attempts": stream_attempts,
                "boundary_lock_events": stream_boundary_locks,
            }
            seed_rows.append(row)
            all_completed_records.extend(stream_completed)
            all_attempt_records.extend(stream_attempts)
            all_boundary_lock_events.extend(stream_boundary_locks)
            environment.close()
    finally:
        torch.random.set_rng_state(torch_state)
        np.random.set_state(numpy_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)

    total_steps = sum(row["steps"] for row in seed_rows)
    total_tasks = sum(row["tasks_completed"] for row in seed_rows)
    total_collisions = sum(row["collision_count"] for row in seed_rows)
    total_boundary_collisions = sum(row["boundary_collision_count"] for row in seed_rows)
    total_velocity_saturations = sum(row["velocity_saturation_count"] for row in seed_rows)
    aggregate = {
        "evaluation_mode": mode_name,
        "training_seed": args.seed,
        "checkpoint_step": training_step,
        "evaluation_stream_count": len(seed_rows),
        "successful_evaluation_streams": sum(bool(row["stream_success"]) for row in seed_rows),
        "stream_success_rate": float(np.mean([row["stream_success"] for row in seed_rows])),
        "steps": total_steps,
        "tasks_completed": total_tasks,
        "tasks_per_1000_steps": 1000.0 * total_tasks / max(1, total_steps),
        "successful_goal_attempts": len(all_completed_records),
        "goal_attempt_count": len(all_attempt_records),
        "mean_episode_return": float(np.mean([row["episode_return"] for row in seed_rows])),
        "mean_goal_progress": float(np.mean([row["mean_goal_progress"] for row in seed_rows])),
        "mean_minimum_goal_distance": float(np.mean([row["minimum_goal_distance"] for row in seed_rows])),
        "mean_goal_distance": float(np.mean([row["mean_goal_distance"] for row in seed_rows])),
        "mean_signed_velocity_toward_goal": float(
            np.mean([row["mean_signed_velocity_toward_goal"] for row in seed_rows])
        ),
        "collision_count": total_collisions,
        "collision_rate": total_collisions / max(1, total_steps),
        "boundary_collision_count": total_boundary_collisions,
        "boundary_collision_rate": total_boundary_collisions / max(1, total_steps),
        "velocity_saturation_count": total_velocity_saturations,
        "velocity_saturation_rate": total_velocity_saturations / max(1, total_steps),
        "boundary_lock_event_count": len(all_boundary_lock_events),
        "maximum_consecutive_boundary_contacts": max(
            row["maximum_consecutive_boundary_contacts"] for row in seed_rows
        ),
        "mean_energy_usage": float(np.mean([row["energy_usage"] for row in seed_rows])),
        "mean_abs_action": np.mean([row["mean_abs_action"] for row in seed_rows], axis=0),
        "rms_action": np.mean([row["rms_action"] for row in seed_rows], axis=0),
        "goal_latency": _latency_statistics(all_completed_records),
        "goal_initial_distances": _initial_distance_statistics(all_attempt_records),
    }
    return {
        "training_seed": args.seed,
        "training_step": training_step,
        "evaluation_mode": mode_name,
        "heldout_environment_seeds": list(args.heldout_seeds),
        "policy_sampling_seed_rule": None if deterministic else (
            "base + training_seed*10000000 + checkpoint_step*10 + heldout_environment_seed"
        ),
        "seed_results": seed_rows,
        "aggregate": aggregate,
    }


def build_callback(args: argparse.Namespace, output_dir: Path):
    from stable_baselines3.common.callbacks import BaseCallback

    class NavigationMetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
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
            self.checkpoint_steps = set(args.checkpoint_steps)

        def _assert_finite_training_state(self) -> None:
            for module_name in ("actor", "critic", "critic_target"):
                module = getattr(self.model, module_name)
                if any(not torch.isfinite(parameter).all() for parameter in module.parameters()):
                    raise FloatingPointError(f"nonfinite SB3 SAC {module_name} parameters")
            log_ent_coef = getattr(self.model, "log_ent_coef", None)
            if log_ent_coef is not None and not torch.isfinite(log_ent_coef).all():
                raise FloatingPointError("nonfinite SB3 SAC entropy coefficient")
            for name, value in self.model.logger.name_to_value.items():
                if name.startswith("train/") and isinstance(value, (int, float, np.number)) and not np.isfinite(value):
                    raise FloatingPointError(f"nonfinite SB3 SAC metric {name}={value}")

        def _window_task_rate(self, window_steps: int) -> float:
            lower_bound = self.num_timesteps - window_steps
            count = sum(step > lower_bound for step in self.completion_training_steps)
            denominator = min(window_steps, self.num_timesteps)
            return 1000.0 * count / max(1, denominator)

        def _training_snapshot(self) -> dict[str, Any]:
            steps = max(1, self.num_timesteps)
            logger_values = {
                name: value
                for name, value in self.model.logger.name_to_value.items()
                if name.startswith("train/") and isinstance(value, (int, float, np.number))
            }
            return {
                "training_step": self.num_timesteps,
                "tasks_completed": self.global_tasks_completed,
                "tasks_per_1000_steps": 1000.0 * self.global_tasks_completed / steps,
                "last_10000_tasks_per_1000_steps": self._window_task_rate(10_000),
                "last_50000_tasks_per_1000_steps": self._window_task_rate(50_000),
                "goal_latency": _latency_statistics(self.completed_goal_records),
                "goal_initial_distances": _initial_distance_statistics(self.all_goal_attempt_records),
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

        def _save_checkpoint(self) -> None:
            checkpoint_path = output_dir / f"checkpoint_step_{self.num_timesteps:06d}"
            self.model.save(checkpoint_path)
            snapshot = self._training_snapshot()
            _write_json(
                output_dir / f"checkpoint_summary_step_{self.num_timesteps:06d}.json",
                snapshot,
            )
            for deterministic in (True, False):
                evaluation = evaluate_model(
                    self.model,
                    args,
                    self.num_timesteps,
                    deterministic=deterministic,
                )
                mode_name = evaluation["evaluation_mode"]
                _write_json(
                    output_dir / f"heldout_{mode_name}_step_{self.num_timesteps:06d}.json",
                    evaluation,
                )
                _append_jsonl(output_dir / f"evaluations_{mode_name}.jsonl", evaluation)

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
            actions = np.asarray(self.locals["actions"], dtype=np.float64)
            info = self.locals["infos"][0]
            if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(actions)):
                raise FloatingPointError("nonfinite reward or action in SB3 rollout")
            for key in (
                "goal_progress",
                "signed_velocity_toward_goal",
                "energy_usage",
                "distance_to_goal_after",
            ):
                if not np.isfinite(float(info[key])):
                    raise FloatingPointError(f"nonfinite environment metric {key}")

            action = actions[0]
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
                augmented = _augment_attempt_record(record, self.num_timesteps, step_prefix="training")
                augmented["training_seed"] = args.seed
                self.all_goal_attempt_records.append(augmented)
                _append_jsonl(output_dir / "goal_attempts.jsonl", augmented)
                if augmented["completed"]:
                    self.completed_goal_records.append(augmented)
                    self.completion_training_steps.append(self.num_timesteps)
                    _append_jsonl(output_dir / "goal_completions.jsonl", augmented)
            if info["boundary_lock_event"] is not None:
                event = dict(info["boundary_lock_event"])
                event |= {"training_seed": args.seed, "training_step": self.num_timesteps}
                self.boundary_lock_event_count += 1
                _append_jsonl(output_dir / "boundary_lock_events.jsonl", event)

            if bool(self.locals["dones"][0]):
                _append_jsonl(
                    output_dir / "episodes.jsonl",
                    {
                        "episode_index": self.episode_index,
                        "training_step": self.num_timesteps,
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
            if self.num_timesteps % args.log_interval == 0:
                self._assert_finite_training_state()
                _append_jsonl(output_dir / "learning_curve.jsonl", self._training_snapshot())
            if self.num_timesteps in self.checkpoint_steps:
                self._assert_finite_training_state()
                self._save_checkpoint()
            return True

    return NavigationMetricsCallback()


def train(args: argparse.Namespace) -> None:
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.monitor import Monitor

    sb3_version = importlib.metadata.version("stable-baselines3")
    if sb3_version != "2.8.0":
        raise RuntimeError(f"formal baseline requires stable-baselines3==2.8.0, found {sb3_version}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = make_environment(args)
    check_env(environment, warn=True, skip_render_check=True)
    environment.reset(seed=args.seed)
    monitored_environment = Monitor(environment)
    torch.set_num_threads(args.torch_threads)

    package_versions = {
        "stable_baselines3": sb3_version,
        "torch": torch.__version__,
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
    }
    config = vars(args) | {
        "pid": os.getpid(),
        "started_at": _utc_now(),
        "package_versions": package_versions,
        "observation_fields": list(environment.observation_fields),
        "observation_dimension": int(environment.observation_space.shape[0]),
        "action_space_low": environment.action_space.low.tolist(),
        "action_space_high": environment.action_space.high.tolist(),
        "physical_acceleration_limit": environment.config.a_max.tolist(),
        "velocity_limit": environment.config.v_max.tolist(),
        "dt": environment.config.dt,
        "energy_semantics": "navigation_baseline_nonterminating_large_budget",
        "phase_scope": "DOES_NOT_ESTABLISH_CHARGING_OR_ENERGY_MANAGEMENT_LEARNABILITY",
        "initialization": "FROM_SCRATCH",
        "sac_implementation": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "policy_kwargs": None,
        "ent_coef": "auto",
        "target_entropy": "auto",
        "continuous_goal_progress_reward": "REMOVED",
    }
    _write_json(output_dir / "config.json", config)
    _write_json(
        output_dir / "RUNNING.json",
        {"status": "RUNNING", "pid": os.getpid(), "started_at": config["started_at"]},
    )

    model = SAC(
        "MlpPolicy",
        monitored_environment,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=(args.train_frequency, "step"),
        gradient_steps=args.gradient_steps,
        ent_coef="auto",
        target_entropy="auto",
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.set_logger(configure(str(output_dir / "sb3_logger"), ["stdout", "csv", "json"]))
    callback = build_callback(args, output_dir)
    try:
        model.learn(total_timesteps=args.steps, callback=callback, log_interval=1, progress_bar=False)
        callback._assert_finite_training_state()
        if args.steps not in args.checkpoint_steps:
            model.save(output_dir / f"checkpoint_step_{args.steps:06d}")
        summary = callback._training_snapshot()
        summary |= {
            "status": "COMPLETED",
            "completed_at": _utc_now(),
            "phase_scope": "DOES_NOT_ESTABLISH_CHARGING_OR_ENERGY_MANAGEMENT_LEARNABILITY",
        }
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "COMPLETED.json", summary)
        _write_json(
            output_dir / "RUNNING.json",
            {"status": "COMPLETED", "pid": os.getpid(), "completed_at": summary["completed_at"]},
        )
    except Exception as error:
        failure = {
            "status": "IMPLEMENTATION_FAILURE",
            "failed_at": _utc_now(),
            "training_step": model.num_timesteps,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "FAILED.json", failure)
        _write_json(
            output_dir / "RUNNING.json",
            {"status": "IMPLEMENTATION_FAILURE", "pid": os.getpid(), "failed_at": failure["failed_at"]},
        )
        raise
    finally:
        monitored_environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal 1M pinned-SB3 SAC persistent navigation baseline")
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--navigation-energy-capacity", type=float, default=1000.0)
    parser.add_argument("--goal-radius", type=float, default=0.20)
    parser.add_argument("--minimum-goal-separation", type=float, default=0.60)
    parser.add_argument("--sampling-margin", type=float, default=0.20)
    parser.add_argument("--distance-potential-scale", type=float, default=0.25)
    parser.add_argument("--progress-weight", type=float, default=0.0, help=argparse.SUPPRESS)
    parser.add_argument("--velocity-reward-weight", type=float, default=0.1)
    parser.add_argument("--time-cost", type=float, default=0.01)
    parser.add_argument("--completion-reward", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=1.2)
    parser.add_argument("--energy-cost-weight", type=float, default=0.01)
    parser.add_argument("--backup-intervention-cost", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=[10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000],
    )
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--evaluation-steps", type=int, default=5000)
    parser.add_argument("--stochastic-policy-seed-base", type=int, default=73_000_000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.learning_starts < 0 or args.learning_starts >= args.steps:
        parser.error("steps and learning-starts define an empty training interval")
    if any(step <= 0 or step > args.steps for step in args.checkpoint_steps):
        parser.error("checkpoint steps must lie inside the training budget")
    if args.log_interval <= 0 or args.evaluation_steps <= 0:
        parser.error("logging and evaluation intervals must be positive")
    if abs(args.progress_weight) > 1e-12:
        parser.error("continuous goal progress reward has been removed; progress-weight must be zero")
    if abs(args.gamma - 0.99) > 1e-12:
        parser.error("the formal baseline fixes gamma=0.99 for both SAC and potential shaping")
    return args


if __name__ == "__main__":
    train(parse_args())
