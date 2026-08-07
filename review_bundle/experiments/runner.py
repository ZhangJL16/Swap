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
    evaluation_episodes: int = 20
    generator_center_mode: str = "task_oriented"


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


def _evaluate(method: str, agent, scenario: str, seed: int, episodes: int = 20, generator_center_mode: str = "task_oriented") -> list[dict]:
    certified = method in {"shield_sac", "generator_sac"}
    environment = make_certified_uav_env(f"{scenario}.json", generator_center_mode=generator_center_mode) if certified else _direct_environment(scenario)
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
            "termination_reason": info.get("mission_termination_reason") or "OTHER_FAILURE",
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
    environment = make_certified_uav_env(
        f"{config.scenario}.json",
        generator_center_mode=config.generator_center_mode,
    ) if certified else _direct_environment(config.scenario)
    observation, reset_info = environment.reset(seed=config.seed)
    if certified and environment.mission_provider.validation_report()["mission_certificate_gate"] != "PASS":
        raise RuntimeError(f"blocked-by-mission-certificate: {config.scenario}")
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
        current_context = environment.action_context() if certified else {}

    episode_id = episode_step = 0
    episode_return = path_length = outbound_path = return_path = 0.0
    previous_position = environment.plant.state.position.copy()
    training_rows: list[dict] = []
    episode_rows: list[dict] = []
    safety_events: list[dict] = []
    latest_update: dict = {}
    accepted_total = fallback_total = no_generator_total = 0
    uncertified_task_publications = fallback_with_invalid_kappa = 0
    zonotope_volumes: list[float] = []
    sigma_values: list[float] = []
    condition_values: list[float] = []
    cycle_times: list[float] = []
    stage_samples: dict[str, list[float]] = {name: [] for name in ("T_policy", "T_certificate", "T_watchdog", "T_plant", "T_total")}
    trajectory_rows: list[dict] = []
    minimum_distance_to_task = float("inf")
    previous_distance_to_task = float(np.linalg.norm(environment.plant.state.position - environment.plant.scenario.task_goal))
    progressing_steps = outbound_steps = outbound_accepted = return_steps = return_accepted = 0
    cosine_exec_sum = cosine_center_sum = cosine_center_count = 0.0
    started = monotonic()

    for environment_step in range(1, config.total_steps + 1):
        cycle_started = monotonic()
        phase_before = environment.task_env.phase.name if certified else environment.phase.name
        state_before = environment.plant.state.copy()
        context_before = dict(current_context)
        policy_started = monotonic()
        if method == "generator_sac":
            actor_u = rng.normal(size=3) if environment_step <= config.warmup_steps else agent.select_u(observation)
            stage_samples["T_policy"].append(monotonic() - policy_started)
            next_observation, reward, terminated, truncated, info = environment.step(actor_u)
        else:
            nominal_action = rng.uniform(-action_max, action_max) if environment_step <= config.warmup_steps else agent.select_action(observation, current_context)
            stage_samples["T_policy"].append(monotonic() - policy_started)
            if method == "shield_sac":
                next_observation, reward, terminated, truncated, info = environment.step_nominal_action(nominal_action)
            else:
                next_observation, reward, terminated, truncated, info = environment.step(nominal_action)
        telemetry = info["telemetry"]
        stage_info = info.get("stage_timings", {})
        measured_total = monotonic() - cycle_started
        cycle_times.append(float(stage_info.get("T_total", measured_total)))
        stage_samples["T_total"].append(measured_total)
        stage_samples["T_plant"].append(float(stage_info.get("T_plant", measured_total)))
        if certified:
            stage_samples["T_certificate"].append(float(stage_info.get("T_certificate", 0.0)))
            stage_samples["T_watchdog"].append(float(stage_info.get("T_recheck", 0.0)) + float(stage_info.get("T_publish", 0.0)))
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
            if certified:
                current_context = environment.action_context()
        if accepted:
            accepted_total += 1
            uncertified_task_publications += int(certified and not bool(context_before.get("certificate_valid", False)))
        else:
            fallback_total += 1
            no_generator_total += int(fallback_reason == "NO_GENERATOR_SET")
            fallback_with_invalid_kappa += int(certified and not bool(context_before.get("certificate_valid", False)))
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
        distance_to_task = float(np.linalg.norm(environment.plant.state.position - environment.plant.scenario.task_goal))
        distance_to_station = float(np.linalg.norm(environment.plant.state.position - environment.plant.scenario.station_position))
        minimum_distance_to_task = min(minimum_distance_to_task, distance_to_task)
        progressing_steps += int(distance_to_task < previous_distance_to_task - 1e-12)
        previous_distance_to_task = distance_to_task
        goal_vector = environment.plant.scenario.task_goal - state_before.position
        def cosine(action):
            action = np.asarray(action, dtype=np.float64)
            denominator = float(np.linalg.norm(action) * np.linalg.norm(goal_vector))
            return 0.0 if denominator <= 1e-12 else float(action @ goal_vector / denominator)
        center = context_before.get("c") if certified else None
        kappa = context_before.get("kappa") if certified else None
        trajectory_rows.append({
            "environment_step": environment_step, "episode_id": episode_id, "episode_step": episode_step,
            "position": json.dumps(environment.plant.state.position.tolist()),
            "velocity": json.dumps(environment.plant.state.velocity.tolist()),
            "distance_to_task": distance_to_task, "distance_to_station": distance_to_station,
            "minimum_distance_to_task_so_far": minimum_distance_to_task,
            "executed_action": json.dumps(executed.tolist()),
            "candidate_action": "" if telemetry.action_trace.candidate is None else json.dumps(telemetry.action_trace.candidate.tolist()),
            "kappa_action": "" if kappa is None else json.dumps(np.asarray(kappa).tolist()),
            "zonotope_center": "" if center is None else json.dumps(np.asarray(center).tolist()),
            "zonotope_G": "" if context_before.get("G") is None else json.dumps(np.asarray(context_before["G"]).tolist()),
            "cos_exec_goal": cosine(executed),
            "cos_candidate_goal": "" if telemetry.action_trace.candidate is None else cosine(telemetry.action_trace.candidate),
            "cos_center_goal": "" if center is None else cosine(center),
            "cos_kappa_goal": "" if kappa is None else cosine(kappa),
            "zonotope_volume": "" if context_before.get("G") is None else 8.0 * abs(float(np.linalg.det(context_before["G"]))),
            "sigma_min": "" if context_before.get("G") is None else float(np.linalg.svd(context_before["G"], compute_uv=False).min()),
            "condition_number": "" if context_before.get("G") is None else float(np.linalg.cond(context_before["G"])),
            "certificate_valid": int(bool(context_before.get("certificate_valid", False))) if certified else "",
            "generator_available": int(bool(context_before.get("generator_available", False))) if certified else "",
            "fallback_reason": fallback_reason or "", "mission_phase": next_phase,
            "energy_margin": context_before.get("energy_margin", "") if certified else "",
            "recovery_energy_required": context_before.get("recovery_energy_required", "") if certified else "",
        })
        cosine_exec_sum += cosine(executed)
        if center is not None:
            cosine_center_sum += cosine(center)
            cosine_center_count += 1
        if phase_before == "OUTBOUND":
            outbound_steps += 1
            outbound_accepted += int(accepted)
        else:
            return_steps += 1
            return_accepted += int(accepted)
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
                "termination_reason": info.get("mission_termination_reason") or "OTHER_FAILURE",
                "minimum_distance_to_task": minimum_distance_to_task,
                "fraction_steps_progressing_to_task": progressing_steps / max(1, episode_step),
                "mean_cos_exec_goal": cosine_exec_sum / max(1, episode_step),
                "mean_cos_center_goal": cosine_center_sum / max(1.0, cosine_center_count),
                "outbound_acceptance_rate": outbound_accepted / max(1, outbound_steps),
                "outbound_fallback_rate": 1.0 - outbound_accepted / max(1, outbound_steps),
                "return_acceptance_rate": return_accepted / max(1, return_steps),
                "return_fallback_rate": 1.0 - return_accepted / max(1, return_steps),
            })
            episode_id += 1
            observation, reset_info = environment.reset(seed=config.seed + episode_id)
            current_context = environment.action_context() if certified else {}
            episode_step = 0
            episode_return = path_length = outbound_path = return_path = 0.0
            previous_position = environment.plant.state.position.copy()
            minimum_distance_to_task = float("inf")
            previous_distance_to_task = float(np.linalg.norm(environment.plant.state.position - environment.plant.scenario.task_goal))
            progressing_steps = outbound_steps = outbound_accepted = return_steps = return_accepted = 0
            cosine_exec_sum = cosine_center_sum = cosine_center_count = 0.0
        else:
            observation = next_observation
        if environment_step % config.checkpoint_interval == 0:
            state = agent.state_dict() if method == "generator_sac" else {
                "actor": agent.actor.state_dict(), "critic_1": agent.critic_1.state_dict(), "critic_2": agent.critic_2.state_dict()
            }
            torch.save(state, output / "checkpoint_latest.pt")

    write_csv(output / "training_metrics.csv", training_rows)
    write_csv(output / "episode_metrics.csv", episode_rows)
    write_csv(output / "trajectory_diagnostics.csv", trajectory_rows)
    evaluation_rows = _evaluate(method, agent, config.scenario, config.seed + 10_000, config.evaluation_episodes, config.generator_center_mode)
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
        "uncertified_task_publication_count": uncertified_task_publications,
        "fallback_with_invalid_kappa": fallback_with_invalid_kappa,
        "mean_zonotope_volume": float(np.mean(zonotope_volumes)) if zonotope_volumes else 0.0,
        "min_zonotope_volume": float(np.min(zonotope_volumes)) if zonotope_volumes else 0.0,
        "mean_sigma_min": float(np.mean(sigma_values)) if sigma_values else 0.0,
        "mean_condition_number": float(np.mean(condition_values)) if condition_values else 0.0,
        "runtime_median_seconds": float(np.median(cycle_times)) if cycle_times else None,
        "runtime_p95_seconds": float(np.quantile(cycle_times, 0.95)) if cycle_times else 0.0,
        "runtime_p99_seconds": float(np.quantile(cycle_times, 0.99)) if cycle_times else 0.0,
        "runtime_max_seconds": float(np.max(cycle_times)) if cycle_times else 0.0,
        "wall_time_seconds": monotonic() - started, "evidence_scope": "synthetic empirical training only",
        "evaluation_task_success_rate": float(np.mean([row["task_success"] for row in evaluation_rows])),
        "evaluation_return_success_rate": float(np.mean([row["return_success"] for row in evaluation_rows])),
        "terminal_energy_mean": float(np.mean([row["terminal_energy"] for row in episode_rows])) if episode_rows else None,
        "minimum_energy_margin": min(
            (float(row["energy_margin"]) for row in trajectory_rows if row["energy_margin"] not in ("", None)),
            default=None,
        ),
        "minimum_distance_to_task": min((row["minimum_distance_to_task"] for row in episode_rows), default=None),
        "task_completion_time_mean": float(np.mean([row["episode_length"] for row in episode_rows if row["task_success"]])) if any(row["task_success"] for row in episode_rows) else None,
        "outbound_fallback_rate": float(np.mean([row["outbound_fallback_rate"] for row in episode_rows])) if episode_rows else None,
        "mission_certificate_gate": "PASS" if certified else "not_applicable",
        "termination_reason_counts": {
            reason: sum(row["termination_reason"] == reason for row in episode_rows)
            for reason in sorted({row["termination_reason"] for row in episode_rows})
        },
        "premature_terminal_rate": float(np.mean([row["termination_reason"] == "PREMATURE_TERMINAL" for row in episode_rows])) if episode_rows else 0.0,
        "corridor_exit_rate": float(np.mean([row["termination_reason"] == "CORRIDOR_EXIT" for row in episode_rows])) if episode_rows else 0.0,
        "certificate_failure_rate": float(np.mean([row["termination_reason"] in {"RECOVERY_CERTIFICATE_INVALID", "CERTIFICATE_EXPIRED"} for row in episode_rows])) if episode_rows else 0.0,
        "velocity_failure_rate": float(np.mean([row["termination_reason"] == "VELOCITY_LIMIT" for row in episode_rows])) if episode_rows else 0.0,
        "other_failure_rate": float(np.mean([row["termination_reason"] == "OTHER_FAILURE" for row in episode_rows])) if episode_rows else 0.0,
        "timing_semantics": "profiled" if certified else "policy-and-plant-profiled; certificate not_applicable",
        "T_policy_p99_seconds": float(np.quantile(stage_samples["T_policy"], 0.99)),
        "T_certificate_p99_seconds": float(np.quantile(stage_samples["T_certificate"], 0.99)) if certified and stage_samples["T_certificate"] else None,
        "T_watchdog_p99_seconds": float(np.quantile(stage_samples["T_watchdog"], 0.99)) if certified and stage_samples["T_watchdog"] else None,
        "T_plant_p99_seconds": float(np.quantile(stage_samples["T_plant"], 0.99)),
        "T_total_p99_seconds": float(np.quantile(stage_samples["T_total"], 0.99)),
    }
    final_state = agent.state_dict() if method == "generator_sac" else {
        "actor": agent.actor.state_dict(), "critic_1": agent.critic_1.state_dict(), "critic_2": agent.critic_2.state_dict(),
        "target_critic_1": agent.target_critic_1.state_dict(), "target_critic_2": agent.target_critic_2.state_dict(),
    }
    torch.save(final_state, output / "checkpoint_latest.pt")
    torch.save(final_state, output / "checkpoint_best.pt")
    (output / "runtime_profile.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    return runtime_summary
