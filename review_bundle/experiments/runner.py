from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import subprocess
from time import monotonic

import numpy as np
import torch

from cert_runtime.generator_sac import GeneratorSAC, GeneratorSACConfig, GeneratorTransition
from envs.certified_uav import make_certified_uav_env
from envs.certified_uav.config import CertifiedUAVConfig, apply_configuration_overrides
from envs.certified_uav.plant_env import CertifiedSingleUAVPlantEnv
from envs.certified_uav.scenario import FixedCertificationScenario
from envs.certified_uav.task_wrapper import CertifiedTaskWrapper

from .agents import DirectSACAgent, DirectTransition
from .metrics import write_csv
from .registry import validate_method


@dataclass(frozen=True)
class ExperimentConfig:
    method: str
    scenario: str
    seed: int
    total_steps: int = 10_000
    warmup_steps: int = 500
    batch_size: int = 64
    hidden_dim: int = 128
    updates_per_step: int = 1
    checkpoint_interval: int = 5_000
    evaluation_interval: int = 5_000
    output_root: str = "artifacts/comparison"
    device: str = "cpu"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _direct_environment(scenario_name: str):
    scenario = FixedCertificationScenario(f"{scenario_name}.json").definition
    config = apply_configuration_overrides(CertifiedUAVConfig(world_size=scenario.world_size), scenario.configuration_overrides)
    return CertifiedTaskWrapper(CertifiedSingleUAVPlantEnv(config, scenario))


def _terminal_context(current: dict) -> dict:
    return {
        "certificate_valid": False, "generator_available": False,
        "c": None, "G": None, "kappa": np.asarray(current.get("kappa", np.zeros(3)), dtype=np.float32),
        "certificate_epoch": current.get("certificate_epoch", "terminal"),
        "geometry_version": current.get("geometry_version", "terminal"),
        "corridor_version": current.get("corridor_version", "terminal"),
        "energy_version": current.get("energy_version", "terminal"),
        "recovery_hash": current.get("recovery_hash"), "zonotope_hash": None,
    }


def _evaluate(method: str, agent, scenario: str, seed: int, episodes: int = 2) -> list[dict]:
    certified = method in {"shield_sac", "generator_sac"}
    environment = make_certified_uav_env(f"{scenario}.json") if certified else _direct_environment(scenario)
    rows = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        episode_return = 0.0
        task_success = return_success = False
        for step in range(1, environment.plant.config.episode_limit + 1):
            if method == "generator_sac":
                next_observation, reward, terminated, truncated, info = environment.step(agent.select_u(observation, deterministic=True))
            else:
                action = agent.select_action(observation, {}, deterministic=True)
                if method == "shield_sac":
                    next_observation, reward, terminated, truncated, info = environment.step_nominal_action(action)
                else:
                    next_observation, reward, terminated, truncated, info = environment.step(action)
            if method == "penalty_sac":
                reward -= 20.0 * float(info.get("failure_reason") in {"collision", "energy_depleted", "velocity_limit_exceeded"})
            episode_return += reward
            task_success |= bool(info.get("task_completed", False))
            return_success |= bool(info.get("terminal_return_success", False))
            observation = next_observation
            if terminated or truncated:
                break
        rows.append({
            "evaluation_episode": episode, "episode_return": episode_return, "episode_length": step,
            "task_success": int(task_success), "return_success": int(return_success),
            "collision": int(info.get("failure_reason") == "collision"),
            "energy_depleted": int(info.get("failure_reason") == "energy_depleted"),
        })
    return rows


def run_experiment(config: ExperimentConfig) -> dict:
    method = validate_method(config.method)
    output = Path(config.output_root) / config.scenario / method / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config) | {
        "git_sha": _git_sha(), "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "evidence_scope": "synthetic empirical training only",
    }, indent=2), encoding="utf-8")

    certified = method in {"shield_sac", "generator_sac"}
    environment = make_certified_uav_env(f"{config.scenario}.json") if certified else _direct_environment(config.scenario)
    observation, reset_info = environment.reset(seed=config.seed)
    rng = np.random.default_rng(config.seed)
    action_max = environment.config.a_max if certified else environment.plant.config.a_max
    if method == "generator_sac":
        agent = GeneratorSAC(observation.size, GeneratorSACConfig(
            batch_size=config.batch_size, warmup_steps=config.warmup_steps, hidden_dim=config.hidden_dim,
            updates_per_step=config.updates_per_step, replay_capacity=max(10000, config.total_steps * 2),
            epoch_replay_policy="group",
        ), seed=config.seed, device=config.device)
        current_context = environment.action_context()
    else:
        agent = DirectSACAgent(observation.size, action_max, seed=config.seed, batch_size=config.batch_size, hidden_dim=config.hidden_dim, device=config.device)
        current_context = {}

    episode_id = episode_step = 0
    episode_return = path_length = outbound_path = return_path = 0.0
    previous_position = environment.plant.state.position.copy()
    training_rows: list[dict] = []
    episode_rows: list[dict] = []
    safety_events: list[dict] = []
    latest_update: dict = {}
    accepted_total = fallback_total = no_generator_total = 0
    zonotope_volumes: list[float] = []
    sigma_values: list[float] = []
    condition_values: list[float] = []
    cycle_times: list[float] = []
    started = monotonic()

    for environment_step in range(1, config.total_steps + 1):
        phase_before = environment.task_env.phase.name if certified else environment.phase.name
        if method == "generator_sac":
            actor_u = rng.normal(size=3) if environment_step <= config.warmup_steps else agent.select_u(observation)
            next_observation, reward, terminated, truncated, info = environment.step(actor_u)
        else:
            nominal_action = rng.uniform(-action_max, action_max) if environment_step <= config.warmup_steps else agent.select_action(observation, current_context)
            if method == "shield_sac":
                next_observation, reward, terminated, truncated, info = environment.step_nominal_action(nominal_action)
            else:
                next_observation, reward, terminated, truncated, info = environment.step(nominal_action)
        telemetry = info["telemetry"]
        cycle_times.append(float(info.get("stage_timings", {}).get("T_total", 0.0)))
        executed = telemetry.action_trace.published.copy()
        measured = telemetry.action_trace.measured.copy()
        accepted = bool(info.get("accepted", method in {"sac", "penalty_sac"}))
        fallback_reason = info.get("fallback_reason")
        if method == "penalty_sac":
            reward -= 20.0 * float(info.get("failure_reason") in {"collision", "energy_depleted", "velocity_limit_exceeded"})
        next_phase = info.get("mission_phase", environment.task_env.phase.name if certified else environment.phase.name)
        if method == "generator_sac":
            next_context = _terminal_context(current_context) if terminated else environment.preview_next_action_context()
            record = environment.replay.records[-1]
            transition = GeneratorTransition(
                observation, next_observation, reward, terminated, truncated, episode_id, phase_before, next_phase,
                str(current_context["certificate_epoch"]), str(next_context["certificate_epoch"]),
                record.nominal_pre_squash_u, record.squashed_eta, current_context.get("c"), current_context.get("G"),
                record.candidate_action, record.recovery_action, record.executed_action, record.measured_tracking_action,
                record.accepted, record.fallback_reason, next_context.get("c"), next_context.get("G"), next_context["kappa"],
                bool(next_context["generator_available"]), bool(next_context["certificate_valid"]),
                str(current_context["geometry_version"]), str(current_context["corridor_version"]), str(current_context["energy_version"]),
                (current_context.get("recovery_hash"), current_context.get("zonotope_hash")),
            )
            agent.observe(transition)
            current_context = next_context
            if current_context.get("G") is not None:
                determinant = abs(float(np.linalg.det(current_context["G"])))
                zonotope_volumes.append(8.0 * determinant)
                singular = np.linalg.svd(current_context["G"], compute_uv=False)
                sigma_values.append(float(singular.min()))
                condition_values.append(float(singular.max() / singular.min()))
        else:
            agent.observe(DirectTransition(observation, next_observation, reward, terminated, truncated, executed))
        if accepted:
            accepted_total += 1
        else:
            fallback_total += 1
            no_generator_total += int(fallback_reason == "NO_GENERATOR_SET")
        if environment_step > config.warmup_steps:
            for _ in range(config.updates_per_step):
                try:
                    result = agent.update()
                except ValueError as error:
                    if "batch" not in str(error) and "epoch" not in str(error):
                        raise
                    result = None
                if result:
                    latest_update = result
                    if any(isinstance(value, float) and not np.isfinite(value) for key, value in result.items() if "loss" in key and key != "actor_loss"):
                        raise FloatingPointError(f"nonfinite update at step {environment_step}: {result}")

        displacement = float(np.linalg.norm(environment.plant.state.position - previous_position))
        path_length += displacement
        if phase_before == "OUTBOUND": outbound_path += displacement
        else:
            return_path += displacement
        previous_position = environment.plant.state.position.copy()
        episode_step += 1
        episode_return += reward
        if info.get("failure_reason"):
            safety_events.append({"step": environment_step, "episode_id": episode_id, "reason": info["failure_reason"], "executed_action": executed.tolist()})
        row = {
            "environment_step": environment_step, "episode_id": episode_id, "episode_step": episode_step,
            "reward": reward, "mission_phase": next_phase, "accepted": int(accepted), "fallback": int(not accepted),
            "fallback_reason": fallback_reason or "", "energy": environment.plant.state.energy,
            "zonotope_volume": zonotope_volumes[-1] if zonotope_volumes else "", "elapsed_seconds": monotonic() - started,
        } | latest_update
        training_rows.append(row)
        if terminated or truncated:
            terminal_success = bool(info.get("terminal_return_success", False))
            episode_rows.append({
                "episode_id": episode_id, "episode_return": episode_return, "episode_length": episode_step,
                "task_success": int(info.get("task_completed", False)), "return_success": int(terminal_success),
                "collision": int(info.get("failure_reason") == "collision"),
                "energy_depleted": int(info.get("failure_reason") == "energy_depleted"),
                "timeout": int(truncated), "outbound_path_length": outbound_path, "return_path_length": return_path,
                "total_path_length": path_length, "terminal_energy": environment.plant.state.energy,
            })
            episode_id += 1
            observation, reset_info = environment.reset(seed=config.seed + episode_id)
            current_context = environment.action_context() if method == "generator_sac" else {}
            episode_step = 0
            episode_return = path_length = outbound_path = return_path = 0.0
            previous_position = environment.plant.state.position.copy()
        else:
            observation = next_observation
        if environment_step % config.checkpoint_interval == 0:
            state = agent.state_dict() if method == "generator_sac" else {
                "actor": agent.actor.state_dict(), "critic_1": agent.critic_1.state_dict(), "critic_2": agent.critic_2.state_dict()
            }
            torch.save(state, output / "checkpoint_latest.pt")

    write_csv(output / "training_metrics.csv", training_rows)
    write_csv(output / "episode_metrics.csv", episode_rows)
    evaluation_rows = _evaluate(method, agent, config.scenario, config.seed + 10_000)
    write_csv(output / "evaluation_metrics.csv", evaluation_rows)
    with (output / "safety_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in safety_events:
            handle.write(json.dumps(event) + "\n")
    runtime_summary = {
        "method": method, "scenario": config.scenario, "seed": config.seed, "environment_steps": config.total_steps,
        "episodes": len(episode_rows), "mean_episode_length": float(np.mean([row["episode_length"] for row in episode_rows])) if episode_rows else 0.0,
        "task_success_rate": float(np.mean([row["task_success"] for row in episode_rows])) if episode_rows else 0.0,
        "return_success_rate": float(np.mean([row["return_success"] for row in episode_rows])) if episode_rows else 0.0,
        "collision_episode_rate": float(np.mean([row["collision"] for row in episode_rows])) if episode_rows else 0.0,
        "energy_depletion_rate": float(np.mean([row["energy_depleted"] for row in episode_rows])) if episode_rows else 0.0,
        "mean_episode_return": float(np.mean([row["episode_return"] for row in episode_rows])) if episode_rows else 0.0,
        "generator_acceptance_rate": accepted_total / config.total_steps if method == "generator_sac" else 0.0,
        "shield_acceptance_rate": accepted_total / config.total_steps if method == "shield_sac" else 0.0,
        "task_action_execution_rate": accepted_total / config.total_steps,
        "fallback_rate": fallback_total / config.total_steps,
        "no_generator_set_rate": no_generator_total / config.total_steps,
        "mean_zonotope_volume": float(np.mean(zonotope_volumes)) if zonotope_volumes else 0.0,
        "min_zonotope_volume": float(np.min(zonotope_volumes)) if zonotope_volumes else 0.0,
        "mean_sigma_min": float(np.mean(sigma_values)) if sigma_values else 0.0,
        "mean_condition_number": float(np.mean(condition_values)) if condition_values else 0.0,
        "runtime_median_seconds": float(np.median(cycle_times)) if cycle_times else 0.0,
        "runtime_p95_seconds": float(np.quantile(cycle_times, 0.95)) if cycle_times else 0.0,
        "runtime_p99_seconds": float(np.quantile(cycle_times, 0.99)) if cycle_times else 0.0,
        "runtime_max_seconds": float(np.max(cycle_times)) if cycle_times else 0.0,
        "wall_time_seconds": monotonic() - started, "evidence_scope": "synthetic empirical training only",
        "evaluation_task_success_rate": float(np.mean([row["task_success"] for row in evaluation_rows])),
        "evaluation_return_success_rate": float(np.mean([row["return_success"] for row in evaluation_rows])),
    }
    final_state = agent.state_dict() if method == "generator_sac" else {
        "actor": agent.actor.state_dict(), "critic_1": agent.critic_1.state_dict(), "critic_2": agent.critic_2.state_dict(),
        "target_critic_1": agent.target_critic_1.state_dict(), "target_critic_2": agent.target_critic_2.state_dict(),
    }
    torch.save(final_state, output / "checkpoint_latest.pt")
    torch.save(final_state, output / "checkpoint_best.pt")
    (output / "runtime_profile.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    return runtime_summary
