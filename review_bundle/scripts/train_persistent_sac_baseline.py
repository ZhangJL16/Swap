#!/usr/bin/env python3
"""Train the direct-action SAC learnability baseline on persistent random goals."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, TextIO

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.certified_uav import RandomPersistentTaskWrapper, make_random_persistent_uav_env
from experiments.agents import DirectSACAgent, DirectTransition


BASE_COMMIT = "3211a4b10311108da5133ab7d4144d7c3953ed73"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(stream: TextIO, record: dict[str, Any]) -> None:
    stream.write(json.dumps(record, sort_keys=True) + "\n")
    stream.flush()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def repository_state() -> dict[str, str | None]:
    def run(*command: str) -> str | None:
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "base_commit": BASE_COMMIT,
        "head_commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "branch", "--show-current"),
    }


def make_direct_environment(scenario: str, seed: int) -> RandomPersistentTaskWrapper:
    certified_environment = make_random_persistent_uav_env(f"{scenario}.json", seed=seed)
    return certified_environment.task_env


@dataclass
class ActionStatistics:
    count: int = 0
    action_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    action_square_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    absolute_sum: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    norm_sum: float = 0.0
    saturated_steps: int = 0

    def observe(self, action: np.ndarray, action_max: np.ndarray) -> None:
        values = np.asarray(action, dtype=np.float64)
        self.count += 1
        self.action_sum += values
        self.action_square_sum += values * values
        self.absolute_sum += np.abs(values)
        self.norm_sum += float(np.linalg.norm(values))
        self.saturated_steps += int(np.any(np.abs(values) >= 0.95 * action_max))

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {
                "action_count": 0,
                "action_mean": [0.0, 0.0, 0.0],
                "action_std": [0.0, 0.0, 0.0],
                "action_mean_abs": [0.0, 0.0, 0.0],
                "action_mean_norm": 0.0,
                "action_saturation_fraction": 0.0,
            }
        mean = self.action_sum / self.count
        variance = np.maximum(0.0, self.action_square_sum / self.count - mean * mean)
        return {
            "action_count": self.count,
            "action_mean": mean.tolist(),
            "action_std": np.sqrt(variance).tolist(),
            "action_mean_abs": (self.absolute_sum / self.count).tolist(),
            "action_mean_norm": self.norm_sum / self.count,
            "action_saturation_fraction": self.saturated_steps / self.count,
        }


@dataclass
class RunStatistics:
    steps: int = 0
    total_return: float = 0.0
    total_goal_progress: float = 0.0
    progressing_steps: int = 0
    tasks_completed: int = 0
    minimum_goal_distance: float = float("inf")
    failures: Counter = field(default_factory=Counter)
    actions: ActionStatistics = field(default_factory=ActionStatistics)

    def observe(
        self,
        reward: float,
        info: dict[str, Any],
        action: np.ndarray,
        action_max: np.ndarray,
    ) -> None:
        progress = float(info.get("goal_progress", 0.0))
        distance = float(info.get("distance_to_goal_after", np.inf))
        self.steps += 1
        self.total_return += float(reward)
        self.total_goal_progress += progress
        self.progressing_steps += int(progress > 0.0)
        self.tasks_completed += int(bool(info.get("task_completed_now", False)))
        if np.isfinite(distance):
            self.minimum_goal_distance = min(self.minimum_goal_distance, distance)
        failure_reason = info.get("failure_reason")
        if failure_reason:
            self.failures[str(failure_reason)] += 1
        self.actions.observe(action, action_max)

    def summary(self) -> dict[str, Any]:
        denominator = max(1, self.steps)
        physical_failures = int(sum(self.failures.values()))
        return {
            "steps": self.steps,
            "total_return": self.total_return,
            "tasks_completed": self.tasks_completed,
            "tasks_per_1000_steps": 1000.0 * self.tasks_completed / denominator,
            "total_goal_progress": self.total_goal_progress,
            "mean_goal_progress": self.total_goal_progress / denominator,
            "progressing_step_fraction": self.progressing_steps / denominator,
            "minimum_goal_distance": None if not np.isfinite(self.minimum_goal_distance) else self.minimum_goal_distance,
            "physical_failure_count": physical_failures,
            "physical_failure_counts": dict(sorted(self.failures.items())),
            "kappa_takeover_count": 0,
            "fallback_count": 0,
            **self.actions.summary(),
        }


@dataclass
class WindowStatistics:
    start_step: int
    start_task_count: int
    return_sum: float = 0.0
    goal_progress: float = 0.0
    minimum_goal_distance: float = float("inf")
    failures: Counter = field(default_factory=Counter)
    actions: ActionStatistics = field(default_factory=ActionStatistics)

    def observe(self, reward: float, info: dict[str, Any], action: np.ndarray, action_max: np.ndarray) -> None:
        self.return_sum += float(reward)
        self.goal_progress += float(info.get("goal_progress", 0.0))
        distance = float(info.get("distance_to_goal_after", np.inf))
        if np.isfinite(distance):
            self.minimum_goal_distance = min(self.minimum_goal_distance, distance)
        failure_reason = info.get("failure_reason")
        if failure_reason:
            self.failures[str(failure_reason)] += 1
        self.actions.observe(action, action_max)

    def summary(self, end_step: int, total_tasks: int) -> dict[str, Any]:
        steps = end_step - self.start_step + 1
        completions = total_tasks - self.start_task_count
        return {
            "window_start_step": self.start_step,
            "window_end_step": end_step,
            "window_steps": steps,
            "window_return": self.return_sum,
            "window_goal_progress": self.goal_progress,
            "window_tasks_completed": completions,
            "tasks_per_1000_steps": 1000.0 * completions / max(1, steps),
            "minimum_goal_distance": None if not np.isfinite(self.minimum_goal_distance) else self.minimum_goal_distance,
            "physical_failure_count": int(sum(self.failures.values())),
            "physical_failure_counts": dict(sorted(self.failures.items())),
            "kappa_takeover_count": 0,
            "fallback_count": 0,
            **self.actions.summary(),
        }


@dataclass
class EpisodeStatistics:
    episode_id: int
    episode_seed: int
    start_step: int
    start_position: list[float]
    initial_goal: list[float]
    return_sum: float = 0.0
    goal_progress: float = 0.0
    tasks_completed: int = 0
    minimum_goal_distance: float = float("inf")
    failures: Counter = field(default_factory=Counter)

    def observe(self, reward: float, info: dict[str, Any]) -> None:
        self.return_sum += float(reward)
        self.goal_progress += float(info.get("goal_progress", 0.0))
        self.tasks_completed += int(bool(info.get("task_completed_now", False)))
        distance = float(info.get("distance_to_goal_after", np.inf))
        if np.isfinite(distance):
            self.minimum_goal_distance = min(self.minimum_goal_distance, distance)
        failure_reason = info.get("failure_reason")
        if failure_reason:
            self.failures[str(failure_reason)] += 1

    def record(self, end_step: int, *, terminated: bool, truncated: bool, partial: bool) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_seed": self.episode_seed,
            "start_step": self.start_step,
            "end_step": end_step,
            "steps": end_step - self.start_step + 1,
            "return": self.return_sum,
            "goal_progress": self.goal_progress,
            "tasks_completed": self.tasks_completed,
            "minimum_goal_distance": None if not np.isfinite(self.minimum_goal_distance) else self.minimum_goal_distance,
            "physical_failure_counts": dict(sorted(self.failures.items())),
            "start_position": self.start_position,
            "initial_goal": self.initial_goal,
            "terminated": terminated,
            "truncated": truncated,
            "partial": partial,
        }


@dataclass
class SegmentStatistics:
    segment_id: int
    episode_id: int
    task_id: str
    start_step: int
    initial_distance: float
    return_sum: float = 0.0
    goal_progress: float = 0.0
    minimum_goal_distance: float = float("inf")

    def observe(self, reward: float, info: dict[str, Any]) -> None:
        self.return_sum += float(reward)
        self.goal_progress += float(info.get("goal_progress", 0.0))
        distance = float(info.get("distance_to_goal_after", np.inf))
        if np.isfinite(distance):
            self.minimum_goal_distance = min(self.minimum_goal_distance, distance)

    def record(self, end_step: int, *, completed: bool, interruption: str | None) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "start_step": self.start_step,
            "end_step": end_step,
            "steps": end_step - self.start_step + 1,
            "return": self.return_sum,
            "goal_progress": self.goal_progress,
            "initial_goal_distance": self.initial_distance,
            "minimum_goal_distance": None if not np.isfinite(self.minimum_goal_distance) else self.minimum_goal_distance,
            "completed": completed,
            "interruption": interruption,
        }


def sampled_start(reset_info: dict[str, Any]) -> list[float]:
    start = reset_info["sampled_start"]
    return np.asarray(getattr(start, "position", start), dtype=float).tolist()


def new_episode(episode_id: int, episode_seed: int, start_step: int, reset_info: dict[str, Any]) -> EpisodeStatistics:
    return EpisodeStatistics(
        episode_id,
        episode_seed,
        start_step,
        sampled_start(reset_info),
        np.asarray(reset_info["sampled_goal"], dtype=float).tolist(),
    )


def new_segment(
    environment: RandomPersistentTaskWrapper,
    segment_id: int,
    episode_id: int,
    start_step: int,
) -> SegmentStatistics:
    task = environment.manager.current_task
    if task is None:
        raise RuntimeError("persistent task is unavailable")
    distance = float(np.linalg.norm(environment.plant.state.position - task.goal_position))
    return SegmentStatistics(segment_id, episode_id, task.task_id, start_step, distance)


def checkpoint_name(step: int) -> str:
    return f"checkpoint_step_{step:06d}.pt"


def checkpoint_payload(
    agent: DirectSACAgent,
    args: argparse.Namespace,
    observation_dim: int,
    action_max: np.ndarray,
    step: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "algorithm": "direct_sac",
        "environment": "random_persistent_task_direct_action",
        "strict_safety_layer": False,
        "step": step,
        "seed": args.seed,
        "observation_dim": observation_dim,
        "action_max": np.asarray(action_max, dtype=np.float32),
        "agent": agent.state_dict(),
        "config": vars(args),
        "repository": repository_state(),
    }


def evaluate_agent(
    agent: DirectSACAgent,
    scenario: str,
    heldout_seeds: list[int],
    steps_per_seed: int,
) -> dict[str, Any]:
    environment = make_direct_environment(scenario, heldout_seeds[0])
    action_max = np.asarray(environment.plant.config.a_max, dtype=np.float64)
    seed_records = []
    aggregate_steps = aggregate_tasks = 0
    aggregate_return = aggregate_progress = 0.0
    aggregate_failures: Counter = Counter()
    minimum_distances: list[float] = []
    for heldout_seed in heldout_seeds:
        observation, reset_info = environment.reset(seed=heldout_seed)
        metrics = RunStatistics()
        episode_count = 1
        first_completion_step = None
        for local_step in range(1, steps_per_seed + 1):
            action = agent.select_action(observation, deterministic=True)
            next_observation, reward, terminated, truncated, info = environment.step(action)
            metrics.observe(reward, info, action, action_max)
            if info.get("task_completed_now") and first_completion_step is None:
                first_completion_step = local_step
            if terminated or truncated:
                if local_step < steps_per_seed:
                    reset_seed = heldout_seed + 10_000 * episode_count
                    observation, _ = environment.reset(seed=reset_seed)
                    episode_count += 1
            else:
                observation = next_observation
        summary = metrics.summary()
        aggregate_steps += metrics.steps
        aggregate_tasks += metrics.tasks_completed
        aggregate_return += metrics.total_return
        aggregate_progress += metrics.total_goal_progress
        aggregate_failures.update(metrics.failures)
        if np.isfinite(metrics.minimum_goal_distance):
            minimum_distances.append(metrics.minimum_goal_distance)
        seed_records.append({
            "heldout_seed": heldout_seed,
            "initial_start": sampled_start(reset_info),
            "initial_goal": np.asarray(reset_info["sampled_goal"], dtype=float).tolist(),
            "episodes": episode_count,
            "time_to_first_completion": first_completion_step,
            "success": metrics.tasks_completed > 0,
            **summary,
        })
    return {
        "evaluation_protocol": "direct_sac_persistent_random_goal",
        "deterministic_actor": True,
        "heldout_seeds": heldout_seeds,
        "steps_per_seed": steps_per_seed,
        "total_evaluation_steps": aggregate_steps,
        "heldout_success_rate": float(np.mean([record["success"] for record in seed_records])),
        "tasks_completed": aggregate_tasks,
        "tasks_per_1000_steps": 1000.0 * aggregate_tasks / max(1, aggregate_steps),
        "total_return": aggregate_return,
        "total_goal_progress": aggregate_progress,
        "mean_goal_progress": aggregate_progress / max(1, aggregate_steps),
        "median_minimum_goal_distance": None if not minimum_distances else float(np.median(minimum_distances)),
        "physical_failure_count": int(sum(aggregate_failures.values())),
        "physical_failure_counts": dict(sorted(aggregate_failures.items())),
        "kappa_takeover_count": 0,
        "fallback_count": 0,
        "seed_records": seed_records,
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.warmup_steps < 0 or args.batch_size <= 0:
        raise ValueError("steps and batch size must be positive; warmup must be nonnegative")
    if args.log_interval <= 0 or args.evaluation_steps <= 0:
        raise ValueError("log interval and evaluation steps must be positive")
    milestones = sorted(set(args.checkpoint_steps))
    if milestones != args.checkpoint_steps:
        raise ValueError("checkpoint steps must be unique and sorted")
    if not milestones or milestones[-1] != args.steps:
        raise ValueError("the final checkpoint step must equal the training budget")
    if milestones[0] <= 0 or milestones[-1] > args.steps:
        raise ValueError("checkpoint steps must lie within the training budget")


def train(args: argparse.Namespace) -> dict[str, Any]:
    validate_args(args)
    torch.set_num_threads(args.torch_threads)
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    protected = [output / checkpoint_name(step) for step in args.checkpoint_steps]
    protected += [
        output / "checkpoint_latest.pt",
        output / "config.json",
        output / "RUNNING.json",
        output / "FAILED.json",
        output / "DONE.json",
        output / "summary.json",
        output / "learning_curve.jsonl",
        output / "episodes.jsonl",
        output / "segments.jsonl",
        output / "evaluations.jsonl",
    ]
    existing = [str(path) for path in protected if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing experiment artifacts: {existing}")

    config = {
        **vars(args),
        "algorithm": "DirectSACAgent",
        "critic_architecture": "existing twin QNetwork",
        "task_environment": "RandomPersistentTaskWrapper",
        "action_execution": "direct physical action; no Generator/certificate/kappa",
        "observation_semantics": "master layout; certificate-only map/corridor/energy-margin fields use wrapper defaults",
        "reward_semantics": "unchanged RandomPersistentTaskWrapper reward",
        "repository": repository_state(),
        "started_at": utc_now(),
    }
    write_json(output / "config.json", config)
    write_json(output / "RUNNING.json", {"started_at": config["started_at"], "pid": str(Path("/proc/self").resolve().name)})

    environment = make_direct_environment(args.scenario, args.seed)
    observation, reset_info = environment.reset(seed=args.seed)
    action_max = np.asarray(environment.plant.config.a_max, dtype=np.float64)
    agent = DirectSACAgent(
        observation.size,
        action_max,
        seed=args.seed,
        batch_size=args.batch_size,
        capacity=args.replay_capacity,
        hidden_dim=args.hidden_dim,
        gamma=args.gamma,
        tau=args.tau,
        device=args.device,
    )
    rng = np.random.default_rng(args.seed)
    run_metrics = RunStatistics()
    window = WindowStatistics(1, 0)
    episode_id = 0
    episode_seed = args.seed
    episode = new_episode(episode_id, episode_seed, 1, reset_info)
    segment_id = 0
    segment = new_segment(environment, segment_id, episode_id, 1)
    episode_records = 0
    segment_records = 0
    last_update: dict[str, float | int] | None = None
    evaluations: list[dict[str, Any]] = []

    with (
        (output / "learning_curve.jsonl").open("a", encoding="utf-8") as curve_stream,
        (output / "episodes.jsonl").open("a", encoding="utf-8") as episode_stream,
        (output / "segments.jsonl").open("a", encoding="utf-8") as segment_stream,
        (output / "evaluations.jsonl").open("a", encoding="utf-8") as evaluation_stream,
    ):
        for step_number in range(1, args.steps + 1):
            if step_number <= args.warmup_steps:
                action = rng.uniform(-action_max, action_max)
            else:
                action = agent.select_action(observation)
            if not np.isfinite(action).all():
                raise FloatingPointError(f"nonfinite action at step {step_number}")
            next_observation, reward, terminated, truncated, info = environment.step(action)
            if not np.isfinite(next_observation).all() or not np.isfinite(reward):
                raise FloatingPointError(f"nonfinite transition at step {step_number}")
            agent.observe(DirectTransition(
                observation,
                next_observation,
                reward,
                terminated,
                truncated,
                action,
            ))
            if step_number > args.warmup_steps and len(agent.replay) >= args.batch_size:
                last_update = agent.update()
                if last_update is None or not all(
                    np.isfinite(value) for value in last_update.values() if isinstance(value, float)
                ):
                    raise FloatingPointError(f"nonfinite SAC update at step {step_number}: {last_update}")

            run_metrics.observe(reward, info, action, action_max)
            window.observe(reward, info, action, action_max)
            episode.observe(reward, info)
            segment.observe(reward, info)

            task_completed = bool(info.get("task_completed_now", False))
            if task_completed:
                append_jsonl(segment_stream, segment.record(step_number, completed=True, interruption=None))
                segment_records += 1
                segment_id += 1
                if not (terminated or truncated):
                    segment = new_segment(environment, segment_id, episode_id, step_number + 1)

            if terminated or truncated:
                if not task_completed:
                    append_jsonl(
                        segment_stream,
                        segment.record(step_number, completed=False, interruption=info.get("failure_reason") or "time_limit"),
                    )
                    segment_records += 1
                    segment_id += 1
                append_jsonl(
                    episode_stream,
                    episode.record(step_number, terminated=terminated, truncated=truncated, partial=False),
                )
                episode_records += 1
                episode_id += 1
                episode_seed = args.seed + episode_id
                observation, reset_info = environment.reset(seed=episode_seed)
                episode = new_episode(episode_id, episode_seed, step_number + 1, reset_info)
                segment = new_segment(environment, segment_id, episode_id, step_number + 1)
            else:
                observation = next_observation

            if step_number % args.log_interval == 0 or step_number == args.steps:
                curve_record = {
                    "step": step_number,
                    **window.summary(step_number, run_metrics.tasks_completed),
                    "cumulative": run_metrics.summary(),
                    "episode_id": episode_id,
                    "gradient_steps": agent.gradient_steps,
                    "replay_size": len(agent.replay),
                    "last_update": last_update,
                }
                append_jsonl(curve_stream, curve_record)
                print(
                    f"step={step_number} tasks={run_metrics.tasks_completed} "
                    f"tasks_per_1k={run_metrics.summary()['tasks_per_1000_steps']:.3f} "
                    f"progress={run_metrics.total_goal_progress:.3f} failures={sum(run_metrics.failures.values())} "
                    f"gradient_steps={agent.gradient_steps}",
                    flush=True,
                )
                window = WindowStatistics(step_number + 1, run_metrics.tasks_completed)

            if step_number in args.checkpoint_steps:
                payload = checkpoint_payload(agent, args, observation.size, action_max, step_number)
                checkpoint_path = output / checkpoint_name(step_number)
                torch.save(payload, checkpoint_path)
                torch.save(payload, output / "checkpoint_latest.pt")
                evaluation = evaluate_agent(
                    agent,
                    args.scenario,
                    list(args.heldout_seeds),
                    args.evaluation_steps,
                )
                evaluation["checkpoint_step"] = step_number
                evaluation["checkpoint"] = str(checkpoint_path.relative_to(ROOT))
                write_json(output / f"heldout_step_{step_number:06d}.json", evaluation)
                append_jsonl(evaluation_stream, evaluation)
                evaluations.append(evaluation)
                partial_summary = {
                    "status": "running" if step_number < args.steps else "complete",
                    "step": step_number,
                    "seed": args.seed,
                    "training": run_metrics.summary(),
                    "last_update": last_update,
                    "latest_evaluation": evaluation,
                }
                write_json(output / "summary.json", partial_summary)
                print(
                    f"checkpoint={step_number} heldout_success={evaluation['heldout_success_rate']:.3f} "
                    f"heldout_tasks_per_1k={evaluation['tasks_per_1000_steps']:.3f}",
                    flush=True,
                )

        if episode.start_step <= args.steps:
            append_jsonl(
                episode_stream,
                episode.record(args.steps, terminated=False, truncated=False, partial=True),
            )
            episode_records += 1
        if segment.start_step <= args.steps:
            append_jsonl(
                segment_stream,
                segment.record(args.steps, completed=False, interruption="training_budget"),
            )
            segment_records += 1

    final_summary = {
        "status": "complete",
        "completed_at": utc_now(),
        "seed": args.seed,
        "steps": args.steps,
        "training": run_metrics.summary(),
        "gradient_steps": agent.gradient_steps,
        "episodes": episode_records,
        "segments": segment_records,
        "last_update": last_update,
        "evaluations": evaluations,
        "artifact_paths": {
            "config": str((output / "config.json").relative_to(ROOT)),
            "learning_curve": str((output / "learning_curve.jsonl").relative_to(ROOT)),
            "episodes": str((output / "episodes.jsonl").relative_to(ROOT)),
            "segments": str((output / "segments.jsonl").relative_to(ROOT)),
            "evaluations": str((output / "evaluations.jsonl").relative_to(ROOT)),
            "summary": str((output / "summary.json").relative_to(ROOT)),
            "checkpoint_latest": str((output / "checkpoint_latest.pt").relative_to(ROOT)),
        },
    }
    write_json(output / "summary.json", final_summary)
    write_json(output / "DONE.json", {"completed_at": final_summary["completed_at"], "exit_code": 0})
    print(json.dumps(final_summary, indent=2, sort_keys=True), flush=True)
    return final_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=1_000_000)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=1_000)
    parser.add_argument("--checkpoint-steps", nargs="+", type=int, default=[10_000, 50_000, 100_000, 200_000])
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[100, 101, 102, 103, 104])
    parser.add_argument("--evaluation-steps", type=int, default=2_000)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = ROOT / args.output_dir
    try:
        train(args)
    except Exception as error:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "FAILED.json", {
            "failed_at": utc_now(),
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        raise


if __name__ == "__main__":
    main()
