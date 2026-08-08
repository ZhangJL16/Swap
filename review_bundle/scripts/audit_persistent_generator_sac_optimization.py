#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.optimization_diagnostics import (
    affine_scale_entropy_audit,
    entropy_decomposition,
    goal_projection_metrics,
    observation_component_statistics,
)
from cert_runtime.task_authority import BestInGeneratorGoalOracle, action_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {key: float("nan") for key in ("mean", "std", "p10", "p50", "p90", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p10": float(np.percentile(array, 10)),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _restore(environment, row: dict[str, object]) -> tuple[np.ndarray, dict[str, object]]:
    environment.plant.state = UAVPhysicalState(
        np.asarray(row["position"], dtype=np.float64),
        np.asarray(row["velocity"], dtype=np.float64),
        float(row["energy"]),
        0.0,
    )
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        environment.plant.state,
        environment.plant.world,
        environment.plant.np_random,
    )
    goal = np.asarray(row["goal"], dtype=np.float64)
    environment.task_env.manager.current_task.goal_position = goal.copy()
    environment.task_env.mode = PersistentMissionMode.TASK_RL
    environment.task_env.phase = environment.task_env.mode
    environment._context_cache_key = None
    context = environment._refresh_context()
    observation = environment.task_env.build_observation(
        environment.runtime._map_encoding(), environment.runtime._corridor_encoding()
    )
    return observation, context


def _immediate_reward(environment, state: UAVPhysicalState, goal: np.ndarray, action: np.ndarray) -> tuple[float, float, float]:
    dt = environment.plant.config.dt
    next_position = state.position + state.velocity * dt + 0.5 * np.asarray(action) * dt * dt
    progress = float(np.linalg.norm(goal - state.position) - np.linalg.norm(goal - next_position))
    energy = float(environment.plant.energy_model.realized_cost(state, np.asarray(action), dt))
    config = environment.task_env.reward_config
    reward = config.goal_progress_weight * progress - config.elapsed_time_cost - config.flight_energy_cost * energy
    return reward, progress, energy


def _classification(critic: dict[str, float], entropy: dict[str, float], actor: dict[str, object], observation: dict[str, object]) -> tuple[str, list[str]]:
    secondary: list[str] = []
    if critic["fraction_Q_oracle_gt_Q_actor"] < 0.55 and critic["fraction_Q_oracle_gt_Q_opposite"] < 0.60:
        primary = "CRITIC_NOT_GOAL_AWARE"
    elif critic["fraction_Q_oracle_gt_Q_actor"] >= 0.70 and float(actor["oracle_gap"]["p50"]) > 0.0:
        primary = "ACTOR_NOT_EXPLOITING_CRITIC"
    elif abs(entropy["mean_alpha_times_physical_log_prob"]) > 5.0 * max(abs(entropy["mean_immediate_reward"]), 1e-6):
        primary = "ENTROPY_SCALE_MISMATCH"
    else:
        primary = "MIXED"
    goal_stats = observation.get("goal_delta", {})
    if max(goal_stats.get("fraction_clipped_negative_2", 0.0), goal_stats.get("fraction_clipped_positive_2", 0.0)) > 0.1:
        secondary.append("OBSERVATION_SCALING_PROBLEM")
    if entropy["mean_goal_progress_reward_abs"] < entropy["mean_nonprogress_cost_abs"]:
        secondary.append("REWARD_SIGNAL_WEAK")
    if (
        primary != "ENTROPY_SCALE_MISMATCH"
        and abs(entropy["mean_alpha_times_physical_log_prob"]) > 5.0 * max(abs(entropy["mean_immediate_reward"]), 1e-6)
    ):
        secondary.append("ENTROPY_SCALE_MISMATCH")
    return primary, secondary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/task_authority_smoke_open_seed0/checkpoint_latest.pt")
    parser.add_argument("--trajectory", default="artifacts/task_authority_smoke_open_seed0/trajectory_events.jsonl")
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--output", default="artifacts/random_persistent/generator_sac_optimization_audit.json")
    args = parser.parse_args()

    environment = make_random_persistent_uav_env(f"{args.scenario}.json", seed=args.seed)
    observation, _ = environment.reset(seed=args.seed)
    checkpoint = torch.load(ROOT / args.checkpoint, map_location="cpu", weights_only=False)
    config = GeneratorSACConfig(**checkpoint["config"])
    agent = PersistentGeneratorSAC(observation.size, config, seed=args.seed)
    agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()
    rows = _load_rows(ROOT / args.trajectory)
    eligible = [row for row in rows if row.get("execution_authority") == "RL_GENERATOR" and row.get("goal") is not None]
    if not eligible:
        raise RuntimeError("trajectory contains no RL_GENERATOR states")
    indices = np.linspace(0, len(eligible) - 1, min(args.samples, len(eligible)), dtype=int)
    selected = [eligible[index] for index in indices]
    rng = np.random.default_rng(args.seed)
    observations: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    generators: list[np.ndarray] = []
    u_means: list[np.ndarray] = []
    u_samples: list[np.ndarray] = []
    u_stds: list[np.ndarray] = []
    eta_values: list[np.ndarray] = []
    action_norms: list[float] = []
    residual_norms: list[float] = []
    alignments: list[dict[str, float]] = []
    q_values = {name: [] for name in ("oracle", "actor", "center", "random", "opposite")}
    immediate = {name: [] for name in ("oracle", "actor", "center", "opposite")}
    entropy_rows: list[dict[str, float]] = []
    bellman_rows: list[dict[str, float]] = []
    torch.manual_seed(args.seed + 2026)

    for row in selected:
        obs, context = _restore(environment, row)
        if not context.get("generator_executable"):
            continue
        center = np.asarray(context["c"], dtype=np.float64)
        generators_matrix = np.asarray(context["G"], dtype=np.float64)
        state = environment.plant.state.copy()
        goal = np.asarray(row["goal"], dtype=np.float64)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        distribution = agent.actor.distribution(obs_tensor)
        u_mean = distribution.mean
        u_sample = distribution.sample()
        eta_mean = torch.tanh(u_mean)
        eta_sample = torch.tanh(u_sample)
        actor_action = center + generators_matrix @ eta_mean.detach().cpu().numpy()[0]
        oracle_eta = BestInGeneratorGoalOracle().select_eta(state, goal, center, generators_matrix, environment.plant.config.dt)
        oracle_action = action_from_eta(center, generators_matrix, oracle_eta)
        opposite_action = action_from_eta(center, generators_matrix, -oracle_eta)
        random_action = action_from_eta(center, generators_matrix, rng.uniform(-1.0, 1.0, 3))
        alignment = goal_projection_metrics(
            state.position, state.velocity, goal, actor_action, center, oracle_action, environment.plant.config.dt
        )
        observations.append(obs)
        centers.append(center)
        generators.append(generators_matrix)
        u_means.append(u_mean.detach().cpu().numpy()[0])
        u_samples.append(u_sample.detach().cpu().numpy()[0])
        u_stds.append(distribution.stddev.detach().cpu().numpy()[0])
        eta_values.append(eta_sample.detach().cpu().numpy()[0])
        action_norms.append(float(np.linalg.norm(actor_action)))
        residual_norms.append(float(np.linalg.norm(actor_action - center)))
        alignments.append(alignment)
        action_map = {
            "oracle": oracle_action,
            "actor": actor_action,
            "center": center,
            "random": random_action,
            "opposite": opposite_action,
        }
        with torch.no_grad():
            for name, action in action_map.items():
                action_tensor = torch.as_tensor(action, dtype=torch.float32).unsqueeze(0)
                value = torch.minimum(agent.critic_1(obs_tensor, action_tensor), agent.critic_2(obs_tensor, action_tensor))
                q_values[name].append(float(value.item()))
        for name in immediate:
            immediate[name].append(_immediate_reward(environment, state, goal, action_map[name])[0])
        terms = entropy_decomposition(
            distribution,
            u_sample,
            torch.as_tensor(generators_matrix, dtype=torch.float32).unsqueeze(0),
        )
        reward_value = immediate["actor"][-1]
        with torch.no_grad():
            next_position = state.position + state.velocity * environment.plant.config.dt + 0.5 * actor_action * environment.plant.config.dt ** 2
            next_velocity = state.velocity + actor_action * environment.plant.config.dt
            next_state = UAVPhysicalState(next_position, next_velocity, state.energy, 0.0)
            environment.plant.state = next_state
            environment.plant.last_lidar = environment.plant.lidar_model.measure(next_state, environment.plant.world, environment.plant.np_random)
            environment._context_cache_key = None
            next_obs = environment.task_env.build_observation(environment.runtime._map_encoding(), environment.runtime._corridor_encoding())
            next_obs_tensor = torch.as_tensor(next_obs, dtype=torch.float32).unsqueeze(0)
            next_distribution = agent.actor.distribution(next_obs_tensor)
            next_u = next_distribution.sample()
            next_action = center + generators_matrix @ torch.tanh(next_u).cpu().numpy()[0]
            next_action_tensor = torch.as_tensor(next_action, dtype=torch.float32).unsqueeze(0)
            q_next = torch.minimum(agent.target_critic_1(next_obs_tensor, next_action_tensor), agent.target_critic_2(next_obs_tensor, next_action_tensor))
            entropy_term = agent.alpha * terms.physical_log_prob
            target = reward_value + config.gamma * (q_next - entropy_term)
            actor_action_tensor = torch.as_tensor(actor_action, dtype=torch.float32).unsqueeze(0)
            current_q = torch.minimum(agent.critic_1(obs_tensor, actor_action_tensor), agent.critic_2(obs_tensor, actor_action_tensor))
        entropy_rows.append({
            "normal": float(terms.normal_term.item()),
            "tanh": float(terms.negative_tanh_log_jacobian_term.item()),
            "log_det": float(terms.log_det_G.item()),
            "negative_log_det": float(terms.negative_log_det_G_term.item()),
            "normalized": float(terms.normalized_log_prob.item()),
            "physical": float(terms.physical_log_prob.item()),
        })
        bellman_rows.append({
            "reward": reward_value,
            "q_next": float(q_next.item()),
            "entropy_term": float(entropy_term.item()),
            "target": float(target.item()),
            "current_q": float(current_q.item()),
            "td_error": float((target - current_q).item()),
        })

    observation_array = np.stack(observations)
    generator_tensor = torch.as_tensor(np.stack(generators), dtype=torch.float32)
    observation_tensor = torch.as_tensor(observation_array, dtype=torch.float32)
    distribution = agent.actor.distribution(observation_tensor)
    torch.manual_seed(args.seed + 2026)
    u_tensor = distribution.sample()
    affine = affine_scale_entropy_audit(distribution, u_tensor, generator_tensor, config.target_entropy, agent.log_alpha)
    entropy_output = {
        "mean_log_prob_u": float(np.mean([row["normal"] for row in entropy_rows])),
        "mean_negative_tanh_log_jacobian": float(np.mean([row["tanh"] for row in entropy_rows])),
        "mean_log_det_G": float(np.mean([row["log_det"] for row in entropy_rows])),
        "mean_negative_log_det_G": float(np.mean([row["negative_log_det"] for row in entropy_rows])),
        "mean_normalized_log_prob": float(np.mean([row["normalized"] for row in entropy_rows])),
        "mean_physical_log_prob": float(np.mean([row["physical"] for row in entropy_rows])),
        "alpha": float(agent.alpha.detach()),
        "log_alpha": float(agent.log_alpha.detach()),
        "target_entropy": config.target_entropy,
    }
    entropy_output["entropy_target_residual"] = entropy_output["mean_physical_log_prob"] + config.target_entropy
    entropy_output["normalized_entropy_target_residual"] = entropy_output["mean_normalized_log_prob"] + config.target_entropy
    critic_output = {
        "fraction_Q_oracle_gt_Q_actor": float(np.mean(np.asarray(q_values["oracle"]) > np.asarray(q_values["actor"]))),
        "fraction_Q_oracle_gt_Q_center": float(np.mean(np.asarray(q_values["oracle"]) > np.asarray(q_values["center"]))),
        "fraction_Q_oracle_gt_Q_opposite": float(np.mean(np.asarray(q_values["oracle"]) > np.asarray(q_values["opposite"]))),
        "mean_Q_oracle_minus_actor": float(np.mean(np.asarray(q_values["oracle"]) - np.asarray(q_values["actor"]))),
        "mean_Q_oracle_minus_center": float(np.mean(np.asarray(q_values["oracle"]) - np.asarray(q_values["center"]))),
        "Q": {name: _stats(values) for name, values in q_values.items()},
    }
    reward_output = {
        "immediate_reward": {name: _stats(values) for name, values in immediate.items()},
        "fraction_oracle_reward_gt_opposite": float(np.mean(np.asarray(immediate["oracle"]) > np.asarray(immediate["opposite"]))),
    }
    reward_components = {
        "goal_progress_reward": 0.0,
        "task_completion_reward": 0.0,
        "elapsed_time_cost": 0.0,
        "flight_energy_cost": 0.0,
        "backup_cost": 0.0,
        "charging_cost": 0.0,
    }
    for index, row in enumerate(rows):
        progress = float(row.get("goal_progress", 0.0) or 0.0)
        reward_components["goal_progress_reward"] += environment.task_env.reward_config.goal_progress_weight * progress
        reward_components["task_completion_reward"] += environment.task_env.reward_config.task_completion_reward * float(row.get("task_completed_now", False))
        reward_components["elapsed_time_cost"] -= environment.task_env.reward_config.elapsed_time_cost
        if index:
            consumed = max(0.0, float(rows[index - 1].get("energy", 0.0)) - float(row.get("energy", 0.0)) + float(row.get("energy_charged", 0.0) or 0.0))
            reward_components["flight_energy_cost"] -= environment.task_env.reward_config.flight_energy_cost * consumed
        reward_components["backup_cost"] -= environment.task_env.reward_config.backup_intervention_cost * float(row.get("backup_triggered", False))
        reward_components["charging_cost"] -= environment.task_env.reward_config.charging_dwell_cost * float(row.get("charging", False))
    absolute_total = sum(abs(value) for value in reward_components.values()) or 1.0
    reward_output["trajectory_contributions"] = reward_components
    reward_output["absolute_contribution_fraction"] = {key: abs(value) / absolute_total for key, value in reward_components.items()}
    actor_output = {
        "deterministic_u_mean": _stats(np.asarray(u_means).reshape(-1).tolist()),
        "sampled_u": _stats(np.asarray(u_samples).reshape(-1).tolist()),
        "mean_abs_sampled_u": float(np.mean(np.abs(np.asarray(u_samples)))),
        "max_abs_sampled_u": float(np.max(np.abs(np.asarray(u_samples)))),
        "u_std_parameter": _stats(np.asarray(u_stds).reshape(-1).tolist()),
        "sampled_eta": _stats(np.asarray(eta_values).reshape(-1).tolist()),
        "fraction_abs_eta_gt_090": float(np.mean(np.abs(np.asarray(eta_values)) > 0.90)),
        "fraction_abs_eta_gt_095": float(np.mean(np.abs(np.asarray(eta_values)) > 0.95)),
        "fraction_abs_eta_gt_099": float(np.mean(np.abs(np.asarray(eta_values)) > 0.99)),
        "fraction_abs_u_gt_2": float(np.mean(np.abs(np.asarray(u_samples)) > 2.0)),
        "fraction_abs_u_gt_3": float(np.mean(np.abs(np.asarray(u_samples)) > 3.0)),
        "fraction_abs_u_gt_5": float(np.mean(np.abs(np.asarray(u_samples)) > 5.0)),
        "physical_action_norm": _stats(action_norms),
        "residual_action_norm": _stats(residual_norms),
        "actor_goal_projection": _stats([row["actor_goal_projection"] for row in alignments]),
        "oracle_goal_projection": _stats([row["oracle_goal_projection"] for row in alignments]),
        "oracle_gap": _stats([row["oracle_gap"] for row in alignments]),
        "fraction_actor_within_10_percent_oracle": float(np.mean([row["predicted_actor_progress"] >= 0.90 * row["predicted_oracle_progress"] for row in alignments])),
        "fraction_actor_within_25_percent_oracle": float(np.mean([row["predicted_actor_progress"] >= 0.75 * row["predicted_oracle_progress"] for row in alignments])),
        "fraction_actor_within_50_percent_oracle": float(np.mean([row["predicted_actor_progress"] >= 0.50 * row["predicted_oracle_progress"] for row in alignments])),
    }
    observation_output = observation_component_statistics(observation_array, environment.task_env.observation_layout)
    bellman_output = {name: _stats([row[name] for row in bellman_rows]) for name in bellman_rows[0]}
    classifier_entropy = {
        "mean_alpha_times_physical_log_prob": entropy_output["alpha"] * entropy_output["mean_physical_log_prob"],
        "mean_immediate_reward": float(np.mean(immediate["actor"])),
        "mean_goal_progress_reward_abs": abs(reward_components["goal_progress_reward"]) / max(1, len(rows)),
        "mean_nonprogress_cost_abs": sum(abs(value) for key, value in reward_components.items() if key != "goal_progress_reward") / max(1, len(rows)),
    }
    primary, secondary = _classification(critic_output, classifier_entropy, actor_output, observation_output)
    goal_sensitivity: list[dict[str, object]] = []
    base_position = np.asarray(selected[0]["position"], dtype=np.float64)
    _restore(environment, selected[0])
    for offset in (np.array((1.0, 0.0, 0.0)), np.array((-1.0, 0.0, 0.0)), np.array((0.0, 1.0, 0.0)), np.array((0.0, -1.0, 0.0))):
        selected_goal = base_position + offset
        environment.task_env.manager.current_task.goal_position = selected_goal
        goal_observation = environment.task_env.build_observation(environment.runtime._map_encoding(), environment.runtime._corridor_encoding())
        with torch.no_grad():
            mean = agent.actor.distribution(torch.as_tensor(goal_observation, dtype=torch.float32).unsqueeze(0)).mean[0]
        goal_sensitivity.append({"goal": selected_goal.tolist(), "u_mean": mean.tolist(), "eta": torch.tanh(mean).tolist()})
    latent_values = np.asarray([entry["u_mean"] for entry in goal_sensitivity], dtype=np.float64)
    action_values = np.asarray([
        np.asarray(centers[0]) + np.asarray(generators[0]) @ np.asarray(entry["eta"], dtype=np.float64)
        for entry in goal_sensitivity
    ])
    pairwise_latent = []
    pairwise_action = []
    for first in range(len(goal_sensitivity)):
        for second in range(first + 1, len(goal_sensitivity)):
            pairwise_latent.append(float(np.linalg.norm(latent_values[first] - latent_values[second])))
            pairwise_action.append(float(np.linalg.norm(action_values[first] - action_values[second])))
    goal_sensitivity_output = {
        "goals": goal_sensitivity,
        "pairwise_latent_distance": _stats(pairwise_latent),
        "pairwise_action_distance": _stats(pairwise_action),
    }
    result = {
        "checkpoint": args.checkpoint,
        "trajectory": args.trajectory,
        "scenario": args.scenario,
        "sample_count": len(observations),
        "ACTOR": actor_output,
        "TRAINED_GOAL_SENSITIVITY_DIAGNOSTIC": goal_sensitivity_output,
        "OBSERVATION": observation_output,
        "ENTROPY": entropy_output,
        "AFFINE_SCALE_AUDIT": affine,
        "CRITIC": critic_output,
        "REWARD": reward_output,
        "BELLMAN": bellman_output,
        "CLASSIFICATION": {"primary": primary, "secondary": secondary},
        "algorithm_changed": False,
        "synthetic_only": True,
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    affine_path = output.with_name("entropy_affine_scale_audit.json")
    affine_path.write_text(json.dumps(affine, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
