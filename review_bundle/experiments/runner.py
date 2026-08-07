from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import hashlib
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

from .agents import DirectSACAgent, DirectTransition, StatelessGeneratorPolicy
from .metrics import write_csv
from .registry import CERTIFIED_METHODS, GENERATOR_METHODS, TRAINED_METHODS, validate_method


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
    checkpoint_path: str | None = None
    timing_mode: str = "wall_clock"


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unavailable"


def _scenario_source(scenario_name: str) -> str:
    path = Path(scenario_name)
    return str(path) if path.exists() else f"{scenario_name}.json"


def _scenario_label(scenario_name: str) -> str:
    return Path(scenario_name).stem


def _scenario_hash(scenario_name: str) -> str:
    source = Path(_scenario_source(scenario_name))
    if not source.exists():
        scenario = FixedCertificationScenario(_scenario_source(scenario_name)).definition
        payload = {
            "name": scenario.name,
            "initial_position": scenario.initial_state.position.tolist(),
            "initial_velocity": scenario.initial_state.velocity.tolist(),
            "initial_energy": scenario.initial_state.energy,
            "task_goal": scenario.task_goal.tolist(),
            "mission": scenario.mission_config,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    else:
        encoded = source.read_bytes()
    return hashlib.sha256(encoded).hexdigest()


def _scenario_definition(environment):
    return environment.plant.scenario


def _direct_environment(scenario_name: str):
    scenario = FixedCertificationScenario(_scenario_source(scenario_name)).definition
    config = apply_configuration_overrides(CertifiedUAVConfig(world_size=scenario.world_size), scenario.configuration_overrides)
    return CertifiedTaskWrapper(CertifiedSingleUAVPlantEnv(config, scenario))


def _certified_environment(scenario_name: str, center_mode: str, timing_mode: str = "wall_clock"):
    return make_certified_uav_env(
        _scenario_source(scenario_name),
        generator_center_mode=center_mode,
        timing_mode=timing_mode,
    )


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


def _evaluate(
    method: str,
    agent,
    scenario: str,
    seed: int,
    episodes: int = 20,
    generator_center_mode: str = "task_oriented",
    timing_mode: str = "wall_clock",
) -> list[dict]:
    certified = method in CERTIFIED_METHODS
    environment = (
        _certified_environment(scenario, generator_center_mode, timing_mode)
        if certified
        else _direct_environment(scenario)
    )
    rows = []
    for episode in range(episodes):
        observation, _ = environment.reset(seed=seed + episode)
        episode_return = outbound_path = return_path = 0.0
        scenario_definition = _scenario_definition(environment)
        initial_energy = environment.plant.state.energy
        energy_at_task = None
        task_success = return_success = False
        task_completion_step = None
        minimum_distance = float("inf")
        progressing = outbound_steps = return_steps = 0
        outbound_interventions = return_handoffs = safety_fallbacks = 0
        no_generator = certificate_fallbacks = deadline_fallbacks = 0
        uncertified_task_publications = invalid_kappa_fallbacks = 0
        action_norms: list[float] = []
        action_deltas: list[float] = []
        velocity_integral = acceleration_integral = 0.0
        residual_norms: list[float] = []
        center_norms: list[float] = []
        residual_ratios: list[float] = []
        cosine_center_goal: list[float] = []
        cosine_residual_goal: list[float] = []
        cosine_exec_goal: list[float] = []
        previous_action = np.zeros(3)
        previous_position = environment.plant.state.position.copy()
        previous_distance = float(np.linalg.norm(previous_position - scenario_definition.task_goal))
        context = environment.action_context() if certified else {}
        for step in range(1, environment.plant.config.episode_limit + 1):
            phase_before = environment.task_env.phase.name if certified else environment.phase.name
            if method in GENERATOR_METHODS:
                next_observation, reward, terminated, truncated, info = environment.step(
                    agent.select_u(observation, deterministic=method != "random_generator")
                )
            else:
                action = agent.select_action(observation, {}, deterministic=True)
                if method == "shield_sac":
                    next_observation, reward, terminated, truncated, info = environment.step_nominal_action(action)
                else:
                    next_observation, reward, terminated, truncated, info = environment.step(action)
            if method == "penalty_sac":
                reward -= 20.0 * float(info.get("failure_reason") in {"collision", "energy_depleted", "velocity_limit_exceeded"})
            telemetry = info["telemetry"]
            executed = np.asarray(telemetry.action_trace.published)
            displacement = float(np.linalg.norm(environment.plant.state.position - previous_position))
            if phase_before == "OUTBOUND":
                outbound_path += displacement
                outbound_steps += 1
            else:
                return_path += displacement
                return_steps += 1
            previous_position = environment.plant.state.position.copy()
            episode_return += reward
            distance = float(np.linalg.norm(environment.plant.state.position - scenario_definition.task_goal))
            minimum_distance = min(minimum_distance, distance)
            progressing += int(phase_before == "OUTBOUND" and distance < previous_distance - 1e-12)
            previous_distance = distance
            if info.get("task_completed_now") and task_completion_step is None:
                task_completion_step = step
                energy_at_task = environment.plant.state.energy
            task_success |= bool(info.get("task_completed", False))
            return_success |= bool(info.get("terminal_return_success", False))
            accepted = bool(info.get("accepted", not certified))
            reason = info.get("fallback_reason") or ""
            if certified:
                uncertified_task_publications += int(accepted and not bool(context.get("certificate_valid", False)))
                invalid_kappa_fallbacks += int(not accepted and not bool(context.get("certificate_valid", False)))
            if certified and not accepted:
                if phase_before == "OUTBOUND":
                    outbound_interventions += 1
                else:
                    return_handoffs += 1
                no_generator += int(reason == "NO_GENERATOR_SET")
                deadline_fallbacks += int("DEADLINE" in reason)
                certificate_fallbacks += int("CERTIFICATE" in reason or "STALE" in reason or "BUNDLE" in reason)
                safety_fallbacks += int(phase_before == "OUTBOUND" and reason not in {"RECOVERY_TAKEOVER", "NO_GENERATOR_SET"})
            residual = np.zeros(3)
            center = context.get("c")
            goal_direction = scenario_definition.task_goal - telemetry.state_before.position
            def direction_cosine(action: np.ndarray) -> float:
                denominator = float(np.linalg.norm(action) * np.linalg.norm(goal_direction))
                return 0.0 if denominator <= 1e-12 else float(np.asarray(action) @ goal_direction / denominator)
            if accepted and center is not None and telemetry.action_trace.candidate is not None:
                center_array = np.asarray(center, dtype=np.float64)
                residual = np.asarray(telemetry.action_trace.candidate) - center_array
                center_norm = float(np.linalg.norm(center_array))
                residual_norm = float(np.linalg.norm(residual))
                center_norms.append(center_norm)
                residual_norms.append(residual_norm)
                residual_ratios.append(residual_norm / max(center_norm, 1e-12))
                cosine_center_goal.append(direction_cosine(center_array))
                cosine_residual_goal.append(direction_cosine(residual))
            cosine_exec_goal.append(direction_cosine(executed))
            action_norms.append(float(np.linalg.norm(executed)))
            action_deltas.append(float(np.linalg.norm(executed - previous_action)))
            previous_action = executed.copy()
            velocity_integral += float(np.linalg.norm(environment.plant.state.velocity)) * environment.plant.config.dt
            acceleration_integral += float(np.linalg.norm(telemetry.action_trace.measured)) * environment.plant.config.dt
            observation = next_observation
            if terminated or truncated:
                break
            if certified:
                context = environment.preview_next_action_context()
        final_energy = environment.plant.state.energy
        rows.append({
            "evaluation_episode": episode,
            "scenario_id": scenario_definition.name,
            "certificate_manifest_hash": environment.mission_provider.manifest.manifest_hash if certified else "",
            "episode_return": episode_return,
            "episode_length": step,
            "mission_completion_steps": step if return_success else None,
            "task_completion_steps": task_completion_step,
            "steps_to_first_task_reach": task_completion_step,
            "task_success": int(task_success),
            "return_success": int(return_success),
            "collision": int(info.get("failure_reason") == "collision"),
            "energy_depleted": int(info.get("failure_reason") == "energy_depleted"),
            "termination_reason": info.get("mission_termination_reason") or "OTHER_FAILURE",
            "outbound_path_length": outbound_path,
            "return_path_length": return_path,
            "total_path_length": outbound_path + return_path,
            "total_energy_consumed": initial_energy - final_energy,
            "energy_to_task": None if energy_at_task is None else initial_energy - energy_at_task,
            "energy_return": None if energy_at_task is None else energy_at_task - final_energy,
            "terminal_energy": final_energy,
            "minimum_distance_to_task": minimum_distance,
            "fraction_steps_progressing_to_task": progressing / max(1, outbound_steps),
            "mean_task_progress_per_step": (float(np.linalg.norm(scenario_definition.initial_state.position - scenario_definition.task_goal)) - minimum_distance) / max(1, outbound_steps),
            "mean_action_norm": float(np.mean(action_norms)),
            "mean_action_delta": float(np.mean(action_deltas)),
            "action_jerk_proxy": float(np.mean(action_deltas)) / environment.plant.config.dt,
            "velocity_integral": velocity_integral,
            "acceleration_integral": acceleration_integral,
            "mean_residual_norm": float(np.mean(residual_norms)) if residual_norms else 0.0,
            "mean_center_norm": float(np.mean(center_norms)) if center_norms else 0.0,
            "mean_residual_to_center_ratio": float(np.mean(residual_ratios)) if residual_ratios else 0.0,
            "mean_cos_center_goal": float(np.mean(cosine_center_goal)) if cosine_center_goal else 0.0,
            "mean_cos_residual_goal": float(np.mean(cosine_residual_goal)) if cosine_residual_goal else 0.0,
            "mean_cos_exec_goal": float(np.mean(cosine_exec_goal)) if cosine_exec_goal else 0.0,
            "outbound_intervention_rate": outbound_interventions / max(1, outbound_steps),
            "return_handoff_rate": return_handoffs / max(1, return_steps),
            "safety_failure_fallback_rate": safety_fallbacks / max(1, step),
            "no_generator_rate": no_generator / max(1, step),
            "certificate_failure_fallback_rate": certificate_fallbacks / max(1, step),
            "deadline_fallback_rate": deadline_fallbacks / max(1, step),
            "uncertified_task_publication_count": uncertified_task_publications,
            "invalid_kappa_fallback_count": invalid_kappa_fallbacks,
        })
    return rows


def run_experiment(config: ExperimentConfig) -> dict:
    method = validate_method(config.method)
    output = Path(config.output_root) / _scenario_label(config.scenario) / method / f"seed_{config.seed}"
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config) | {
        "git_sha": _git_sha(), "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "evidence_scope": "synthetic empirical training only",
    }, indent=2), encoding="utf-8")

    certified = method in CERTIFIED_METHODS
    environment = (
        _certified_environment(config.scenario, config.generator_center_mode, config.timing_mode)
        if certified
        else _direct_environment(config.scenario)
    )
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
        if config.checkpoint_path is not None:
            agent.load_state_dict(torch.load(config.checkpoint_path, map_location=config.device, weights_only=False))
        current_context = environment.action_context()
    elif method in {"center_only", "random_generator"}:
        agent = StatelessGeneratorPolicy(method, config.seed)
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
    episode_initial_energy = environment.plant.state.energy
    energy_at_task: float | None = None
    task_completion_step: int | None = None
    action_norm_sum = action_delta_sum = velocity_integral = acceleration_integral = 0.0
    residual_norm_sum = center_norm_sum = residual_ratio_sum = residual_cosine_sum = 0.0
    residual_count = 0
    previous_executed = np.zeros(3, dtype=np.float64)
    outbound_interventions = return_handoffs = safety_failure_fallbacks = 0
    certificate_fallbacks = deadline_fallbacks = 0
    started = monotonic()

    for environment_step in range(1, config.total_steps + 1):
        cycle_started = monotonic()
        phase_before = environment.task_env.phase.name if certified else environment.phase.name
        state_before = environment.plant.state.copy()
        context_before = dict(current_context)
        policy_started = monotonic()
        if method in GENERATOR_METHODS:
            actor_u = (
                rng.normal(size=3)
                if method == "generator_sac" and environment_step <= config.warmup_steps
                else agent.select_u(observation)
            )
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
            scenario_definition = _scenario_definition(environment)
            transition = GeneratorTransition(
                observation, next_observation, reward, terminated, truncated, episode_id, phase_before, next_phase,
                str(current_context["certificate_epoch"]), str(next_context["certificate_epoch"]),
                record.nominal_pre_squash_u, record.squashed_eta, current_context.get("c"), current_context.get("G"),
                record.candidate_action, record.recovery_action, record.executed_action, record.measured_tracking_action,
                record.accepted, record.fallback_reason, next_context.get("c"), next_context.get("G"), next_context["kappa"],
                bool(next_context["generator_available"]), bool(next_context["certificate_valid"]),
                str(current_context["geometry_version"]), str(current_context["corridor_version"]), str(current_context["energy_version"]),
                (current_context.get("recovery_hash"), current_context.get("zonotope_hash")),
                scenario_id=scenario_definition.name,
                scenario_family=str(scenario_definition.mission_config.get("scenario_family", _scenario_label(config.scenario))),
                scenario_hash=_scenario_hash(config.scenario),
                certificate_manifest_hash=str(current_context["certificate_epoch"]),
            )
            agent.observe(transition)
            current_context = next_context
            if current_context.get("G") is not None:
                determinant = abs(float(np.linalg.det(current_context["G"])))
                zonotope_volumes.append(8.0 * determinant)
                singular = np.linalg.svd(current_context["G"], compute_uv=False)
                sigma_values.append(float(singular.min()))
                condition_values.append(float(singular.max() / singular.min()))
        elif method in {"sac", "penalty_sac", "shield_sac"}:
            agent.observe(DirectTransition(observation, next_observation, reward, terminated, truncated, executed))
            if certified:
                current_context = environment.action_context()
        else:
            current_context = _terminal_context(current_context) if terminated else environment.preview_next_action_context()
        if accepted:
            accepted_total += 1
            uncertified_task_publications += int(certified and not bool(context_before.get("certificate_valid", False)))
        else:
            fallback_total += 1
            no_generator_total += int(fallback_reason == "NO_GENERATOR_SET")
            fallback_with_invalid_kappa += int(certified and not bool(context_before.get("certificate_valid", False)))
            if certified and phase_before == "OUTBOUND":
                outbound_interventions += 1
                safety_failure_fallbacks += int(fallback_reason not in {"RECOVERY_TAKEOVER", "NO_GENERATOR_SET"})
            elif certified:
                return_handoffs += 1
            certificate_fallbacks += int(bool(fallback_reason) and ("CERTIFICATE" in fallback_reason or "STALE" in fallback_reason or "BUNDLE" in fallback_reason))
            deadline_fallbacks += int(bool(fallback_reason) and "DEADLINE" in fallback_reason)
        if method in TRAINED_METHODS and environment_step > config.warmup_steps:
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
        action_norm_sum += float(np.linalg.norm(executed))
        action_delta_sum += float(np.linalg.norm(executed - previous_executed))
        previous_executed = executed.copy()
        velocity_integral += float(np.linalg.norm(environment.plant.state.velocity)) * environment.plant.config.dt
        acceleration_integral += float(np.linalg.norm(measured)) * environment.plant.config.dt
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
            "residual_action": "" if center is None or telemetry.action_trace.candidate is None else json.dumps((np.asarray(telemetry.action_trace.candidate) - np.asarray(center)).tolist()),
            "residual_norm": "" if center is None or telemetry.action_trace.candidate is None else float(np.linalg.norm(np.asarray(telemetry.action_trace.candidate) - np.asarray(center))),
            "center_norm": "" if center is None else float(np.linalg.norm(center)),
        })
        if accepted and center is not None and telemetry.action_trace.candidate is not None:
            residual = np.asarray(telemetry.action_trace.candidate) - np.asarray(center)
            residual_norm = float(np.linalg.norm(residual))
            center_norm = float(np.linalg.norm(center))
            residual_norm_sum += residual_norm
            center_norm_sum += center_norm
            residual_ratio_sum += residual_norm / max(center_norm, 1e-12)
            residual_cosine_sum += cosine(residual)
            residual_count += 1
        if info.get("task_completed_now") and task_completion_step is None:
            task_completion_step = episode_step
            energy_at_task = environment.plant.state.energy
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
                "mission_completion_steps": episode_step if terminal_success else None,
                "task_completion_steps": task_completion_step,
                "total_energy_consumed": episode_initial_energy - environment.plant.state.energy,
                "energy_to_task": None if energy_at_task is None else episode_initial_energy - energy_at_task,
                "energy_return": None if energy_at_task is None else energy_at_task - environment.plant.state.energy,
                "termination_reason": info.get("mission_termination_reason") or "OTHER_FAILURE",
                "minimum_distance_to_task": minimum_distance_to_task,
                "fraction_steps_progressing_to_task": progressing_steps / max(1, episode_step),
                "mean_cos_exec_goal": cosine_exec_sum / max(1, episode_step),
                "mean_cos_center_goal": cosine_center_sum / max(1.0, cosine_center_count),
                "outbound_acceptance_rate": outbound_accepted / max(1, outbound_steps),
                "outbound_fallback_rate": 1.0 - outbound_accepted / max(1, outbound_steps),
                "return_acceptance_rate": return_accepted / max(1, return_steps),
                "return_fallback_rate": 1.0 - return_accepted / max(1, return_steps),
                "outbound_intervention_rate": outbound_interventions / max(1, outbound_steps),
                "return_handoff_rate": return_handoffs / max(1, return_steps),
                "safety_failure_fallback_rate": safety_failure_fallbacks / max(1, episode_step),
                "certificate_failure_fallback_rate": certificate_fallbacks / max(1, episode_step),
                "deadline_fallback_rate": deadline_fallbacks / max(1, episode_step),
                "mean_action_norm": action_norm_sum / max(1, episode_step),
                "mean_action_delta": action_delta_sum / max(1, episode_step),
                "action_jerk_proxy": action_delta_sum / max(1, episode_step) / environment.plant.config.dt,
                "velocity_integral": velocity_integral,
                "acceleration_integral": acceleration_integral,
                "mean_residual_norm": residual_norm_sum / max(1, residual_count),
                "mean_center_norm": center_norm_sum / max(1, residual_count),
                "mean_residual_to_center_ratio": residual_ratio_sum / max(1, residual_count),
                "mean_cos_residual_goal": residual_cosine_sum / max(1, residual_count),
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
            episode_initial_energy = environment.plant.state.energy
            energy_at_task = None
            task_completion_step = None
            action_norm_sum = action_delta_sum = velocity_integral = acceleration_integral = 0.0
            residual_norm_sum = center_norm_sum = residual_ratio_sum = residual_cosine_sum = 0.0
            residual_count = 0
            previous_executed = np.zeros(3, dtype=np.float64)
            outbound_interventions = return_handoffs = safety_failure_fallbacks = 0
            certificate_fallbacks = deadline_fallbacks = 0
        else:
            observation = next_observation
        if method in TRAINED_METHODS and environment_step % config.checkpoint_interval == 0:
            state = agent.state_dict() if method == "generator_sac" else {
                "actor": agent.actor.state_dict(), "critic_1": agent.critic_1.state_dict(), "critic_2": agent.critic_2.state_dict()
            }
            torch.save(state, output / "checkpoint_latest.pt")

    write_csv(output / "training_metrics.csv", training_rows)
    write_csv(output / "episode_metrics.csv", episode_rows)
    write_csv(output / "trajectory_diagnostics.csv", trajectory_rows)
    evaluation_rows = _evaluate(
        method,
        agent,
        config.scenario,
        config.seed + 10_000,
        config.evaluation_episodes,
        config.generator_center_mode,
        config.timing_mode,
    )
    write_csv(output / "evaluation_metrics.csv", evaluation_rows)
    with (output / "safety_events.jsonl").open("w", encoding="utf-8") as handle:
        for event in safety_events:
            handle.write(json.dumps(event) + "\n")
    runtime_summary = {
        "method": method, "scenario": _scenario_label(config.scenario), "seed": config.seed, "environment_steps": config.total_steps,
        "episodes": len(episode_rows), "mean_episode_length": float(np.mean([row["episode_length"] for row in episode_rows])) if episode_rows else 0.0,
        "task_success_rate": float(np.mean([row["task_success"] for row in episode_rows])) if episode_rows else 0.0,
        "return_success_rate": float(np.mean([row["return_success"] for row in episode_rows])) if episode_rows else 0.0,
        "collision_episode_rate": float(np.mean([row["collision"] for row in episode_rows])) if episode_rows else 0.0,
        "energy_depletion_rate": float(np.mean([row["energy_depleted"] for row in episode_rows])) if episode_rows else 0.0,
        "mean_episode_return": float(np.mean([row["episode_return"] for row in episode_rows])) if episode_rows else 0.0,
        "generator_acceptance_rate": accepted_total / config.total_steps if method in GENERATOR_METHODS else 0.0,
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
    if method in TRAINED_METHODS:
        final_state = agent.state_dict() if method == "generator_sac" else {
            "actor": agent.actor.state_dict(), "critic_1": agent.critic_1.state_dict(), "critic_2": agent.critic_2.state_dict(),
            "target_critic_1": agent.target_critic_1.state_dict(), "target_critic_2": agent.target_critic_2.state_dict(),
        }
        torch.save(final_state, output / "checkpoint_latest.pt")
        torch.save(final_state, output / "checkpoint_best.pt")
    (output / "runtime_profile.json").write_text(json.dumps(runtime_summary, indent=2), encoding="utf-8")
    return runtime_summary
