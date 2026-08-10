#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
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


def make_environment(args: argparse.Namespace, *, max_episode_steps: int | None = None) -> PersistentNavigationEnv:
    reward_config = NavigationRewardConfig(
        progress_weight=args.progress_weight,
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


def evaluate_model(model, args: argparse.Namespace, training_step: int) -> dict[str, Any]:
    seed_rows: list[dict[str, Any]] = []
    for heldout_seed in args.heldout_seeds:
        environment = make_environment(args, max_episode_steps=args.evaluation_steps)
        observation, _ = environment.reset(seed=heldout_seed)
        total_return = 0.0
        progress_sum = 0.0
        velocity_toward_goal_sum = 0.0
        collision_count = 0
        action_abs_sum = np.zeros(3, dtype=np.float64)
        action_squared_sum = np.zeros(3, dtype=np.float64)
        for _ in range(args.evaluation_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = environment.step(action)
            total_return += float(reward)
            progress_sum += float(info["goal_progress"])
            velocity_toward_goal_sum += float(info["velocity_toward_goal"])
            collision_count += int(info["collision"])
            action_array = np.asarray(action, dtype=np.float64)
            action_abs_sum += np.abs(action_array)
            action_squared_sum += action_array * action_array
            if terminated:
                raise RuntimeError("held-out navigation environment terminated unexpectedly")
            if truncated:
                break
        steps = environment.episode_step
        seed_rows.append(
            {
                "seed": heldout_seed,
                "steps": steps,
                "tasks_completed": environment.tasks_completed,
                "tasks_per_1000_steps": 1000.0 * environment.tasks_completed / max(1, steps),
                "goal_success": environment.tasks_completed > 0,
                "goal_attempt_success_rate": environment.tasks_completed / max(1, environment.tasks_completed + 1),
                "episode_return": total_return,
                "mean_goal_progress": progress_sum / max(1, steps),
                "minimum_goal_distance": environment.minimum_goal_distance,
                "mean_goal_distance": environment.goal_distance_sum / max(1, environment.goal_distance_samples),
                "mean_velocity_toward_goal": velocity_toward_goal_sum / max(1, steps),
                "collision_count": collision_count,
                "collision_rate": collision_count / max(1, steps),
                "energy_usage": environment.cumulative_energy_usage,
                "mean_abs_action": action_abs_sum / max(1, steps),
                "rms_action": np.sqrt(action_squared_sum / max(1, steps)),
            }
        )
        environment.close()
    total_steps = sum(row["steps"] for row in seed_rows)
    total_tasks = sum(row["tasks_completed"] for row in seed_rows)
    total_collisions = sum(row["collision_count"] for row in seed_rows)
    return {
        "training_step": training_step,
        "heldout_seeds": list(args.heldout_seeds),
        "seed_results": seed_rows,
        "aggregate": {
            "steps": total_steps,
            "tasks_completed": total_tasks,
            "tasks_per_1000_steps": 1000.0 * total_tasks / max(1, total_steps),
            "seed_success_rate": float(np.mean([row["goal_success"] for row in seed_rows])),
            "goal_attempt_success_rate": total_tasks / max(1, total_tasks + len(seed_rows)),
            "mean_episode_return": float(np.mean([row["episode_return"] for row in seed_rows])),
            "mean_goal_progress": float(np.mean([row["mean_goal_progress"] for row in seed_rows])),
            "mean_minimum_goal_distance": float(np.mean([row["minimum_goal_distance"] for row in seed_rows])),
            "mean_goal_distance": float(np.mean([row["mean_goal_distance"] for row in seed_rows])),
            "mean_velocity_toward_goal": float(np.mean([row["mean_velocity_toward_goal"] for row in seed_rows])),
            "collision_count": total_collisions,
            "collision_rate": total_collisions / max(1, total_steps),
            "mean_energy_usage": float(np.mean([row["energy_usage"] for row in seed_rows])),
            "mean_abs_action": np.mean([row["mean_abs_action"] for row in seed_rows], axis=0),
            "rms_action": np.mean([row["rms_action"] for row in seed_rows], axis=0),
        },
    }


def build_callback(args: argparse.Namespace, output_dir: Path):
    from stable_baselines3.common.callbacks import BaseCallback

    class NavigationMetricsCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.global_tasks_completed = 0
            self.global_collision_count = 0
            self.global_energy_usage = 0.0
            self.global_goal_progress = 0.0
            self.global_velocity_toward_goal = 0.0
            self.global_minimum_goal_distance = float("inf")
            self.action_sum = np.zeros(3, dtype=np.float64)
            self.action_abs_sum = np.zeros(3, dtype=np.float64)
            self.action_squared_sum = np.zeros(3, dtype=np.float64)
            self.action_min = np.full(3, np.inf)
            self.action_max = np.full(3, -np.inf)
            self.episode_return = 0.0
            self.episode_steps = 0
            self.episode_index = 0
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
                "mean_goal_progress": self.global_goal_progress / steps,
                "minimum_goal_distance": self.global_minimum_goal_distance,
                "mean_velocity_toward_goal": self.global_velocity_toward_goal / steps,
                "collision_count": self.global_collision_count,
                "collision_rate": self.global_collision_count / steps,
                "energy_usage": self.global_energy_usage,
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
            evaluation = evaluate_model(self.model, args, self.num_timesteps)
            _write_json(output_dir / f"heldout_step_{self.num_timesteps:06d}.json", evaluation)
            _append_jsonl(output_dir / "evaluations.jsonl", evaluation)

        def _on_step(self) -> bool:
            rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
            actions = np.asarray(self.locals["actions"], dtype=np.float64)
            infos = self.locals["infos"]
            if not np.all(np.isfinite(rewards)) or not np.all(np.isfinite(actions)):
                raise FloatingPointError("nonfinite reward or action in SB3 rollout")
            action = actions[0]
            info = infos[0]
            for key in ("goal_progress", "velocity_toward_goal", "energy_usage", "distance_to_goal_after"):
                if not np.isfinite(float(info[key])):
                    raise FloatingPointError(f"nonfinite environment metric {key}")
            self.global_tasks_completed += int(info["task_completed_now"])
            self.global_collision_count += int(info["collision"])
            self.global_energy_usage += float(info["energy_usage"])
            self.global_goal_progress += float(info["goal_progress"])
            self.global_velocity_toward_goal += float(info["velocity_toward_goal"])
            self.global_minimum_goal_distance = min(
                self.global_minimum_goal_distance,
                float(info["distance_to_goal_after"]),
            )
            self.action_sum += action
            self.action_abs_sum += np.abs(action)
            self.action_squared_sum += action * action
            self.action_min = np.minimum(self.action_min, action)
            self.action_max = np.maximum(self.action_max, action)
            self.episode_return += float(rewards[0])
            self.episode_steps += 1
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    environment = make_environment(args)
    check_env(environment, warn=True, skip_render_check=True)
    environment.reset(seed=args.seed)
    monitored_environment = Monitor(environment)
    torch.set_num_threads(args.torch_threads)

    package_versions = {
        "stable_baselines3": importlib.metadata.version("stable-baselines3"),
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
        "energy_semantics": "navigation_baseline_nonterminating_large_budget",
        "sac_implementation": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "policy_kwargs": None,
    }
    _write_json(output_dir / "config.json", config)
    _write_json(output_dir / "RUNNING.json", {"status": "RUNNING", "pid": os.getpid(), "started_at": config["started_at"]})

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
        summary["status"] = "COMPLETED"
        summary["completed_at"] = _utc_now()
        _write_json(output_dir / "summary.json", summary)
        _write_json(output_dir / "COMPLETED.json", summary)
        _write_json(
            output_dir / "RUNNING.json",
            {"status": "COMPLETED", "pid": os.getpid(), "completed_at": summary["completed_at"]},
        )
    except Exception as error:
        failure = {
            "status": "FAILED",
            "failed_at": _utc_now(),
            "training_step": model.num_timesteps,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_dir / "FAILED.json", failure)
        _write_json(
            output_dir / "RUNNING.json",
            {"status": "FAILED", "pid": os.getpid(), "failed_at": failure["failed_at"]},
        )
        raise
    finally:
        monitored_environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pinned Stable-Baselines3 SAC persistent navigation baseline")
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--navigation-energy-capacity", type=float, default=1000.0)
    parser.add_argument("--goal-radius", type=float, default=0.20)
    parser.add_argument("--minimum-goal-separation", type=float, default=0.60)
    parser.add_argument("--sampling-margin", type=float, default=0.20)
    parser.add_argument("--progress-weight", type=float, default=2.5)
    parser.add_argument("--velocity-reward-weight", type=float, default=0.1)
    parser.add_argument("--time-cost", type=float, default=0.01)
    parser.add_argument("--completion-reward", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=1.2)
    parser.add_argument("--energy-cost-weight", type=float, default=0.01)
    parser.add_argument("--backup-intervention-cost", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-frequency", type=int, default=1)
    parser.add_argument("--gradient-steps", type=int, default=1)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[10000, 50000, 100000, 200000])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--evaluation-steps", type=int, default=2000)
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
    return args


if __name__ == "__main__":
    train(parse_args())
