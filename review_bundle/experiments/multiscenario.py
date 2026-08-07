from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from cert_runtime.generator_sac import GeneratorSAC, GeneratorSACConfig, GeneratorTransition
from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.scenario_families import scenario_file_hash
from .metrics import write_csv


@dataclass(frozen=True)
class MultiScenarioTrainingConfig:
    scenario_index: str = "artifacts/scenario_families/scenario_index.json"
    total_steps: int = 50_000
    seed: int = 0
    warmup_steps: int = 1_000
    batch_size: int = 128
    hidden_dim: int = 128
    device: str = "cpu"
    output_dir: str = "artifacts/multiscenario_training"


def _terminal_context(context: dict) -> dict:
    return {
        "certificate_valid": False,
        "generator_available": False,
        "c": None,
        "G": None,
        "kappa": np.asarray(context["kappa"], dtype=np.float32),
        "certificate_epoch": context["certificate_epoch"],
    }


def train_multiscenario(config: MultiScenarioTrainingConfig) -> dict:
    index = json.loads(Path(config.scenario_index).read_text(encoding="utf-8"))
    records = [record for record in index if record["split"] == "training"]
    if not records:
        raise ValueError("scenario index contains no training scenarios")
    if any(record.get("certificate_gate") != "PASS" for record in records):
        raise RuntimeError("blocked-by-scenario-certificate")
    rng = np.random.default_rng(config.seed)
    first = make_certified_uav_env(records[0]["path"], timing_mode="functional")
    observation, _ = first.reset(seed=config.seed)
    agent = GeneratorSAC(
        observation.size,
        GeneratorSACConfig(
            batch_size=config.batch_size,
            warmup_steps=config.warmup_steps,
            hidden_dim=config.hidden_dim,
            replay_capacity=max(100_000, 2 * config.total_steps),
            epoch_replay_policy="group",
        ),
        seed=config.seed,
        device=config.device,
    )
    output = Path(config.output_dir) / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
    environment = first
    active = records[0]
    context = environment.action_context()
    if scenario_file_hash(active["path"]) != active["scenario_hash"]:
        raise RuntimeError("scenario file hash mismatch before episode")
    episode_id = 0
    episode_return = 0.0
    training_rows = []
    episode_rows = []
    scenario_sequence = []
    latest_update = {}
    for environment_step in range(1, config.total_steps + 1):
        if context["certificate_epoch"] != active["certificate_manifest_hash"]:
            raise RuntimeError("scenario/certificate manifest mismatch")
        phase = environment.task_env.phase.name
        u = rng.normal(size=3) if environment_step <= config.warmup_steps else agent.select_u(observation)
        next_observation, reward, terminated, truncated, info = environment.step(u)
        next_context = _terminal_context(context) if terminated else environment.preview_next_action_context()
        record = environment.replay.records[-1]
        transition = GeneratorTransition(
            observation, next_observation, reward, terminated, truncated, episode_id, phase,
            info.get("mission_phase", phase), str(context["certificate_epoch"]), str(next_context["certificate_epoch"]),
            record.nominal_pre_squash_u, record.squashed_eta, context.get("c"), context.get("G"),
            record.candidate_action, record.recovery_action, record.executed_action, record.measured_tracking_action,
            record.accepted, record.fallback_reason, next_context.get("c"), next_context.get("G"),
            next_context["kappa"], bool(next_context["generator_available"]), bool(next_context["certificate_valid"]),
            str(context["geometry_version"]), str(context["corridor_version"]), str(context["energy_version"]),
            (context.get("recovery_hash"), context.get("zonotope_hash")),
            scenario_id=active["scenario_id"], scenario_family=active["family"],
            scenario_hash=active["scenario_hash"], certificate_manifest_hash=active["certificate_manifest_hash"],
        )
        agent.observe(transition)
        episode_return += reward
        if environment_step > config.warmup_steps:
            try:
                latest_update = agent.update()
            except ValueError as error:
                if "complete batch" not in str(error):
                    raise
        training_rows.append({
            "environment_step": environment_step,
            "episode_id": episode_id,
            "scenario_id": active["scenario_id"],
            "scenario_family": active["family"],
            "certificate_manifest_hash": active["certificate_manifest_hash"],
            "reward": reward,
            "accepted": int(record.accepted),
            "fallback_reason": record.fallback_reason or "",
        } | latest_update)
        if terminated or truncated:
            episode_rows.append({
                "episode_id": episode_id,
                "scenario_id": active["scenario_id"],
                "scenario_family": active["family"],
                "episode_return": episode_return,
                "episode_length": info["episode_step"],
                "task_success": int(info.get("task_completed", False)),
                "return_success": int(info.get("terminal_return_success", False)),
                "collision": int(info.get("failure_reason") == "collision"),
            })
            episode_id += 1
            active = records[int(rng.integers(len(records)))]
            scenario_sequence.append(active["scenario_id"])
            environment = make_certified_uav_env(active["path"], timing_mode="functional")
            observation, _ = environment.reset(seed=config.seed + episode_id)
            if scenario_file_hash(active["path"]) != active["scenario_hash"]:
                raise RuntimeError("scenario file hash mismatch before episode")
            if not environment.mission_provider.gate_pass:
                raise RuntimeError(f"blocked-by-scenario-certificate:{active['scenario_id']}")
            context = environment.action_context()
            episode_return = 0.0
        else:
            observation = next_observation
            context = next_context
    write_csv(output / "training_metrics.csv", training_rows)
    write_csv(output / "episode_metrics.csv", episode_rows)
    torch.save(agent.state_dict(), output / "checkpoint_latest.pt")
    summary = {
        "seed": config.seed,
        "environment_steps": config.total_steps,
        "episodes": len(episode_rows),
        "unique_training_scenarios": len(set(scenario_sequence)),
        "task_success": float(np.mean([row["task_success"] for row in episode_rows])) if episode_rows else None,
        "return_success": float(np.mean([row["return_success"] for row in episode_rows])) if episode_rows else None,
        "collision": float(np.mean([row["collision"] for row in episode_rows])) if episode_rows else None,
        "gradient_steps": agent.gradient_steps,
        "evidence_scope": "multi-scenario synthetic empirical training",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
