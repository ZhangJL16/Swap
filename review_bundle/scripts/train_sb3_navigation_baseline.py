#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import traceback

import gymnasium
import numpy as np
import torch
from stable_baselines3 import DDPG, PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb3_navigation_harness import (
    NavigationMetricsCallback,
    evaluate_navigation_model,
    make_navigation_environment,
    model_device_metadata,
    rollout_aligned_step_at_or_after,
    utc_now,
    write_json,
)


def build_model(args, environment):
    if args.algorithm == "ppo":
        return PPO(
            "MlpPolicy",
            environment,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            policy_kwargs=None,
            seed=args.seed,
            device=args.device,
            verbose=1,
        )
    action_noise = NormalActionNoise(
        mean=np.zeros(3, dtype=np.float32),
        sigma=np.full(3, args.action_noise_sigma, dtype=np.float32),
    )
    return DDPG(
        "MlpPolicy",
        environment,
        learning_rate=1e-3,
        buffer_size=1_000_000,
        learning_starts=100,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        action_noise=action_noise,
        policy_kwargs=None,
        seed=args.seed,
        device=args.device,
        verbose=1,
    )


def algorithm_config(args) -> dict:
    if args.algorithm == "ppo":
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "clip_range_vf": None,
            "normalize_advantage": True,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "use_sde": False,
            "policy_architecture": "stable_baselines3.PPO.MlpPolicy default",
        }
    return {
        "learning_rate": 1e-3,
        "buffer_size": 1_000_000,
        "learning_starts": 100,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": [1, "step"],
        "gradient_steps": 1,
        "policy_architecture": "stable_baselines3.DDPG.MlpPolicy default",
        "action_noise_type": "stable_baselines3.common.noise.NormalActionNoise",
        "action_noise_mean": [0.0, 0.0, 0.0],
        "action_noise_sigma": [args.action_noise_sigma] * 3,
    }


def run(args) -> None:
    sb3_version = importlib.metadata.version("stable-baselines3")
    if sb3_version != "2.8.0":
        raise RuntimeError(f"baseline requires stable-baselines3==2.8.0, found {sb3_version}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise FileExistsError(f"refusing to overwrite {output_dir / 'config.json'}")
    environment = make_navigation_environment(args)
    monitored_environment = Monitor(environment)
    torch.set_num_threads(args.torch_threads)
    model = build_model(args, monitored_environment)
    model.set_logger(configure(str(output_dir / "sb3_logger"), ["stdout", "csv", "json"]))
    callback = NavigationMetricsCallback(args, output_dir)
    config = vars(args) | {
        "experiment_class": "UNTUNED_STANDARD_SB3_BASELINES",
        "requested_timesteps": args.steps,
        "actual_timesteps": None,
        "initialization": "FROM_SCRATCH",
        "stable_baselines3_version": sb3_version,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "algorithm_class": f"stable_baselines3.{args.algorithm.upper()}",
        "policy": "MlpPolicy",
        "actual_policy_class": type(model.policy).__qualname__,
        "actual_policy_architecture": getattr(model.policy, "net_arch", None),
        "policy_repr": repr(model.policy),
        "algorithm_hyperparameters": algorithm_config(args),
        "environment_class": "PersistentNavigationEnv",
        "solved_sac_reference": "DIRECT_SAC_BASELINE_SOLVED",
        "solved_sac_artifact": "artifacts/phase1_sb3_sac_1m_gpu",
        "observation_fields": list(environment.observation_fields),
        "observation_dimension": int(environment.observation_space.shape[0]),
        "action_space_low": environment.action_space.low.tolist(),
        "action_space_high": environment.action_space.high.tolist(),
        "physical_acceleration_limit": environment.config.a_max.tolist(),
        "velocity_limit": environment.config.v_max.tolist(),
        "dt": environment.config.dt,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": utc_now(),
    }
    config |= model_device_metadata(model, args.device)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "RUNNING.json", {"status": "RUNNING", "started_at": config["started_at"]})
    print("DEVICE_CHECK " + json.dumps(model_device_metadata(model, args.device), sort_keys=True), flush=True)

    checkpoint_records = []
    try:
        for requested_step in args.checkpoint_steps:
            remaining = requested_step - model.num_timesteps
            if remaining > 0:
                model.learn(
                    total_timesteps=remaining,
                    callback=callback,
                    log_interval=1,
                    reset_num_timesteps=False,
                    progress_bar=False,
                )
            callback.assert_finite_training_state()
            actual_step = int(model.num_timesteps)
            if args.algorithm == "ppo":
                expected_actual = rollout_aligned_step_at_or_after(requested_step, model.n_steps)
                if actual_step != expected_actual:
                    raise RuntimeError(
                        f"PPO checkpoint is not rollout aligned: requested={requested_step} "
                        f"actual={actual_step} expected={expected_actual}"
                    )
            checkpoint_stem = (
                f"checkpoint_requested_{requested_step:07d}_actual_{actual_step:07d}"
            )
            model.save(output_dir / checkpoint_stem)
            snapshot = callback.snapshot() | {
                "requested_checkpoint_step": requested_step,
                "actual_checkpoint_step": actual_step,
            }
            write_json(output_dir / f"{checkpoint_stem}_summary.json", snapshot)
            evaluation_modes = (
                ("deterministic", "stochastic")
                if args.algorithm == "ppo"
                else ("deterministic", "ddpg_exploration_noise")
            )
            for mode in evaluation_modes:
                evaluation = evaluate_navigation_model(
                    model,
                    args,
                    requested_step,
                    actual_step,
                    evaluation_mode=mode,
                )
                write_json(output_dir / f"{checkpoint_stem}_heldout_{mode}.json", evaluation)
            record = {
                "requested_checkpoint_step": requested_step,
                "actual_checkpoint_step": actual_step,
                "checkpoint": f"{checkpoint_stem}.zip",
            }
            checkpoint_records.append(record)
            write_json(output_dir / "checkpoint_index.json", {"checkpoints": checkpoint_records})

        summary = callback.snapshot() | {
            "status": "COMPLETED",
            "requested_timesteps": args.steps,
            "actual_timesteps": int(model.num_timesteps),
            "completed_at": utc_now(),
            "checkpoint_index": checkpoint_records,
        }
        config["actual_timesteps"] = int(model.num_timesteps)
        config["checkpoint_index"] = checkpoint_records
        write_json(output_dir / "config.json", config)
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "COMPLETED.json", summary)
        write_json(output_dir / "RUNNING.json", {"status": "COMPLETED", "completed_at": summary["completed_at"]})
    except Exception as error:
        config["actual_timesteps"] = int(model.num_timesteps)
        write_json(output_dir / "config.json", config)
        failure = {
            "status": "IMPLEMENTATION_FAILURE",
            "failed_at": utc_now(),
            "requested_timesteps": args.steps,
            "actual_timesteps": int(model.num_timesteps),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "FAILED.json", failure)
        write_json(output_dir / "RUNNING.json", failure)
        raise
    finally:
        monitored_environment.close()


def parse_args(algorithm: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Untuned standard SB3 {algorithm.upper()} navigation baseline")
    parser.add_argument("--algorithm", choices=[algorithm], default=algorithm)
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
    parser.add_argument("--navigation-energy-capacity", type=float, default=1000.0)
    parser.add_argument("--goal-radius", type=float, default=0.20)
    parser.add_argument("--minimum-goal-separation", type=float, default=0.60)
    parser.add_argument("--sampling-margin", type=float, default=0.20)
    parser.add_argument("--distance-potential-scale", type=float, default=0.25)
    parser.add_argument("--velocity-reward-weight", type=float, default=0.1)
    parser.add_argument("--time-cost", type=float, default=0.01)
    parser.add_argument("--completion-reward", type=float, default=10.0)
    parser.add_argument("--collision-penalty", type=float, default=1.2)
    parser.add_argument("--energy-cost-weight", type=float, default=0.01)
    parser.add_argument("--backup-intervention-cost", type=float, default=0.1)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--evaluation-steps", type=int, default=5000)
    parser.add_argument("--evaluation-seed-base", type=int, default=83_000_000)
    parser.add_argument("--action-noise-sigma", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.gamma = 0.99
    if args.steps <= 0 or args.checkpoint_steps[-1] != args.steps:
        parser.error("final checkpoint must equal requested training budget")
    if sorted(set(args.checkpoint_steps)) != args.checkpoint_steps:
        parser.error("checkpoint steps must be strictly increasing")
    return args


def main(algorithm: str) -> None:
    run(parse_args(algorithm))
