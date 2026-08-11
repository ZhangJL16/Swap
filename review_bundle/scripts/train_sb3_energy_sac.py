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
from stable_baselines3 import SAC
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sb3_energy_harness import EnergyMetricsCallback, evaluate_energy_model, make_energy_environment
from scripts.sb3_navigation_harness import model_device_metadata, utc_now, write_json


def run(args) -> None:
    sb3_version = importlib.metadata.version("stable-baselines3")
    if sb3_version != "2.8.0":
        raise RuntimeError(f"energy baseline requires stable-baselines3==2.8.0, found {sb3_version}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise FileExistsError(f"refusing to overwrite {output_dir / 'config.json'}")
    environment = make_energy_environment(args)
    monitored_environment = Monitor(environment)
    torch.set_num_threads(args.torch_threads)
    model = SAC(
        "MlpPolicy",
        monitored_environment,
        learning_rate=3e-4,
        buffer_size=1_000_000,
        learning_starts=5000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=(1, "step"),
        gradient_steps=1,
        ent_coef="auto",
        target_entropy="auto",
        seed=args.seed,
        device=args.device,
        verbose=1,
    )
    model.set_logger(configure(str(output_dir / "sb3_logger"), ["stdout", "csv", "json"]))
    callback = EnergyMetricsCallback(args, output_dir)
    config = vars(args) | {
        "initialization": "FROM_SCRATCH",
        "algorithm_class": "stable_baselines3.SAC",
        "policy": "MlpPolicy",
        "actual_policy_class": type(model.policy).__qualname__,
        "actual_policy_architecture": getattr(model.policy, "net_arch", None),
        "policy_repr": repr(model.policy),
        "stable_baselines3_version": sb3_version,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "numpy_version": np.__version__,
        "python_version": sys.version,
        "environment_class": "PersistentEnergyNavigationEnv",
        "energy_mode": "finite_charging",
        "charging_trigger": "continuous_position_and_velocity_only",
        "discrete_charge_action": False,
        "strict_safety_runtime": False,
        "phase_scope": "DIRECT_SAC_ENERGY_LEARNABILITY",
        "observation_fields": list(environment.observation_fields),
        "observation_dimension": int(environment.observation_space.shape[0]),
        "action_space_low": environment.action_space.low.tolist(),
        "action_space_high": environment.action_space.high.tolist(),
        "dt": environment.config.dt,
        "physical_acceleration_limit": environment.config.a_max.tolist(),
        "velocity_limit": environment.config.v_max.tolist(),
        "sac_hyperparameters": {
            "learning_rate": 3e-4,
            "buffer_size": 1_000_000,
            "learning_starts": 5000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "train_freq": [1, "step"],
            "gradient_steps": 1,
            "ent_coef": "auto",
            "target_entropy": "auto",
        },
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "started_at": utc_now(),
    }
    config |= model_device_metadata(model, args.device)
    write_json(output_dir / "config.json", config)
    write_json(output_dir / "RUNNING.json", {"status": "RUNNING", "started_at": config["started_at"]})
    print("DEVICE_CHECK " + json.dumps(model_device_metadata(model, args.device), sort_keys=True), flush=True)
    try:
        for checkpoint_step in args.checkpoint_steps:
            remaining = checkpoint_step - model.num_timesteps
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
            model.save(output_dir / f"checkpoint_step_{actual_step:07d}")
            write_json(
                output_dir / f"checkpoint_summary_step_{actual_step:07d}.json",
                callback.snapshot() | {"requested_checkpoint_step": checkpoint_step},
            )
            for evaluation_mode in ("deterministic", "stochastic"):
                for soc_group in ("full", "low_soc"):
                    evaluation = evaluate_energy_model(
                        model,
                        args,
                        checkpoint_step,
                        actual_step,
                        evaluation_mode=evaluation_mode,
                        soc_group=soc_group,
                    )
                    write_json(
                        output_dir / f"heldout_{evaluation_mode}_{soc_group}_step_{actual_step:07d}.json",
                        evaluation,
                    )
        summary = callback.snapshot() | {
            "status": "COMPLETED",
            "requested_timesteps": args.steps,
            "actual_timesteps": int(model.num_timesteps),
            "completed_at": utc_now(),
        }
        write_json(output_dir / "summary.json", summary)
        write_json(output_dir / "COMPLETED.json", summary)
        write_json(output_dir / "RUNNING.json", {"status": "COMPLETED", "completed_at": summary["completed_at"]})
    except Exception as error:
        failure = {
            "status": "IMPLEMENTATION_FAILURE",
            "failed_at": utc_now(),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standard SB3 SAC finite-energy persistent navigation")
    parser.add_argument("--scenario", default="random_persistent_open.json")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--max-episode-steps", type=int, default=5000)
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
    parser.add_argument("--battery-capacity", type=float, default=30.0)
    parser.add_argument("--charging-rate", type=float, default=2.0)
    parser.add_argument("--charging-radius", type=float, default=0.18)
    parser.add_argument("--charging-velocity-limit", type=float, nargs=3, default=[0.05, 0.05, 0.04])
    parser.add_argument("--initial-energy-fraction-min", type=float, default=0.30)
    parser.add_argument("--initial-energy-fraction-max", type=float, default=1.00)
    parser.add_argument("--checkpoint-steps", type=int, nargs="+", default=[10_000, 50_000, 100_000, 200_000, 300_000, 500_000, 750_000, 1_000_000])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[100, 101, 102, 103, 104])
    parser.add_argument("--evaluation-steps", type=int, default=5000)
    parser.add_argument("--evaluation-seed-base", type=int, default=93_000_000)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    args.gamma = 0.99
    args.dt = 0.2
    if args.steps <= 5000 or args.checkpoint_steps[-1] != args.steps:
        parser.error("energy run must include warm-up and final checkpoint")
    return args


if __name__ == "__main__":
    run(parse_args())
