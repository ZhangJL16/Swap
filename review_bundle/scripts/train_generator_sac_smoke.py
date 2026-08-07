#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch.set_num_threads(1)

from cert_runtime.smoke_training import MinimalGeneratorSAC, SmokeTransition, density_gradient_acceptance
from envs.certified_uav import make_certified_uav_env


def train_seed(seed: int, scenario: str, steps: int, warmup: int, batch_size: int, update_every: int) -> dict[str, object]:
    runtime = make_certified_uav_env(f"{scenario}.json", freeze_certificate_epoch=True)
    observation, reset_info = runtime.reset(seed=seed)
    if not reset_info["certificate_ready"]:
        raise RuntimeError(reset_info["certificate_failure_reason"])
    trainer = MinimalGeneratorSAC(observation.size, seed)
    replay: deque[SmokeTransition] = deque(maxlen=10000)
    rng = np.random.default_rng(seed)
    rows = []
    accepted = fallback = gradient_steps = epoch_rejections = nonfinite = 0
    actor_before = torch.cat([parameter.detach().flatten() for parameter in trainer.actor.parameters()]).clone()
    for environment_step in range(1, steps + 1):
        if environment_step <= warmup:
            actor_u = rng.normal(size=3)
        else:
            actor_u = np.asarray(trainer.actor.sample_u(observation), dtype=np.float64)
        next_observation, reward, terminated, truncated, info = runtime.step(actor_u)
        record = runtime.replay.records[-1]
        trainer.freeze_epoch(record)
        replay.append(SmokeTransition(observation.copy(), next_observation.copy(), reward, terminated or truncated, record))
        accepted += int(record.accepted)
        fallback += int(not record.accepted)
        losses: dict[str, object] = {}
        if len(replay) >= batch_size and environment_step % update_every == 0:
            indices = rng.choice(len(replay), size=batch_size, replace=False)
            batch = tuple(replay[int(index)] for index in indices)
            try:
                losses = trainer.update(batch)
                gradient_steps += 1
            except ValueError:
                epoch_rejections += 1
                raise
            except FloatingPointError:
                nonfinite += 1
                raise
        zonotope = record.zonotope_generators
        matrix = None if zonotope is None else np.asarray(zonotope)
        rows.append(
            {
                "environment_steps": environment_step,
                "gradient_steps": gradient_steps,
                "episode_return": reward,
                "episode_length": 1,
                "task_success": int(info.get("task_goal_reached", False)),
                "terminal_return_success": int(info["telemetry"].terminal_admissible),
                "collision": int(info["telemetry"].collision),
                "energy_depleted": int(info.get("failure_reason") == "energy_depleted"),
                "accepted_count": accepted,
                "fallback_count": fallback,
                "acceptance_rate": accepted / environment_step,
                "fallback_rate": fallback / environment_step,
                "no_generator_set_rate": 0.0,
                "certificate_failure_rate": fallback / environment_step,
                "mean_zonotope_volume": None if matrix is None else 8.0 * abs(float(np.linalg.det(matrix))),
                "min_sigma": None if matrix is None else float(np.linalg.svd(matrix, compute_uv=False).min()),
                "mean_condition_number": None if matrix is None else float(np.linalg.cond(matrix)),
                **losses,
                "q_value_candidate_if_available": losses.get("q_value_exec"),
                "replay_size": len(replay),
                "epoch_rejection_count": epoch_rejections,
                "nonfinite_count": nonfinite,
                "deadline_fallback_count": sum(item.record.fallback_reason == "WATCHDOG_DEADLINE" for item in replay),
            }
        )
        observation, _ = runtime.reset(seed=seed)
    actor_after = torch.cat([parameter.detach().flatten() for parameter in trainer.actor.parameters()])
    if gradient_steps == 0 or torch.equal(actor_before, actor_after):
        raise AssertionError("smoke run did not update the actor")
    output_dir = Path("artifacts/smoke_training")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f"seed_{seed}_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "seed": seed,
        "environment_steps": steps,
        "gradient_steps": gradient_steps,
        "accepted_transitions": accepted,
        "fallback_transitions": fallback,
        "acceptance_rate": accepted / steps,
        "epoch_rejections": epoch_rejections,
        "nonfinite_count": nonfinite,
        "actor_parameter_change_norm": float(torch.linalg.vector_norm(actor_after - actor_before)),
        "final_metrics": rows[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="open_corridor")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--training-steps", type=int, default=1000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--update-every", type=int, default=10)
    args = parser.parse_args()
    summaries = [
        train_seed(seed, args.scenario, args.training_steps, args.warmup_steps, args.batch_size, args.update_every)
        for seed in args.seeds
    ]
    result = {
        "scenario": args.scenario,
        "seeds": summaries,
        "density_gradient_acceptance": density_gradient_acceptance(),
        "evidence_scope": "minimal synthetic training smoke; not convergence or flight evidence",
    }
    path = Path("artifacts/smoke_training/summary.json")
    path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
