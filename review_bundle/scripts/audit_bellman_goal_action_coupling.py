#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.bellman_goal_action_diagnostics import (
    additive_decomposition_metrics,
    bellman_target_decomposition,
    contrast_decomposition,
    finite_preferred_actions,
    mean_pairwise_action_distance,
    nominal_physical_transition,
    immediate_reward_components,
)
from cert_runtime.counterfactual_goal_diagnostics import residual_alignment
from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.optimization_diagnostics import entropy_decomposition
from cert_runtime.persistent_authority import ExecutionAuthority
from cert_runtime.task_authority import BestInGeneratorGoalOracle, action_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState
from scripts.audit_counterfactual_goal_critic import (
    _counterfactual_goals,
    _goal_direction,
    _goal_pool,
    _load_rows,
    _state_goal_observations,
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _stats(values) -> dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {name: float("nan") for name in ("mean", "median", "p10", "p90", "min", "max")}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _unique_action_candidates(center, generator, actor_eta, oracle_etas, seed):
    rng = np.random.default_rng(seed)
    candidates: list[tuple[str, np.ndarray]] = [("center", np.zeros(3)), ("actor", actor_eta)]
    candidates.extend((f"corner_{index}", np.asarray(eta)) for index, eta in enumerate(product((-1.0, 1.0), repeat=3)))
    candidates.extend((f"random_{index}", eta) for index, eta in enumerate(rng.uniform(-1.0, 1.0, size=(6, 3))))
    for index, eta in enumerate(oracle_etas):
        candidates.append((f"oracle_{index}", eta))
        candidates.append((f"opposite_{index}", -eta))
    labels = []
    etas = []
    seen = set()
    for label, eta in candidates:
        selected = np.clip(np.asarray(eta, dtype=np.float64), -1.0, 1.0)
        key = tuple(np.round(selected, 8))
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
        etas.append(selected)
    eta_array = np.stack(etas)
    actions = center[None, :] + np.einsum("ij,nj->ni", generator, eta_array)
    return labels, eta_array, actions


def _set_successor_state(environment, state: UAVPhysicalState) -> None:
    environment.plant.state = state.copy()
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        state,
        environment.plant.world,
        environment.plant.np_random,
    )
    environment.task_env.mode = PersistentMissionMode.TASK_RL
    environment.task_env.phase = PersistentMissionMode.TASK_RL
    environment._context_cache_key = None
    environment._context_cache = None


def _next_observations(environment, goals, successor: UAVPhysicalState):
    _set_successor_state(environment, successor)
    task = environment.task_env.manager.current_task
    task.goal_position = np.asarray(goals[0], dtype=np.float64).copy()
    context = environment._refresh_context()
    local_map = environment.runtime._map_encoding()
    corridor = environment.runtime._corridor_encoding()
    observations = []
    for goal in goals:
        task.goal_position = np.asarray(goal, dtype=np.float64).copy()
        observations.append(environment.task_env.build_observation(local_map, corridor))
    return np.stack(observations), context


def _target_terms(agent, observations, context, common_noise):
    count = observations.shape[0]
    authority = str(context.get("execution_authority", ExecutionAuthority.FAIL_CLOSED.value))
    generator = (
        authority in {ExecutionAuthority.RL_GENERATOR.value, ExecutionAuthority.CHARGER_CONSTRAINED.value}
        and context.get("generator_executable") is True
    )
    bootstrap = np.ones(count, dtype=np.float64)
    entropy = torch.zeros(count, dtype=torch.float32)
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
    if authority == ExecutionAuthority.FAIL_CLOSED.value:
        bootstrap.fill(0.0)
        actions = torch.zeros((count, 3), dtype=torch.float32)
    elif generator:
        center = torch.as_tensor(context["c"], dtype=torch.float32).reshape(1, 3).expand(count, -1)
        generators = torch.as_tensor(context["G"], dtype=torch.float32).reshape(1, 3, 3).expand(count, -1, -1)
        distribution = agent.actor.distribution(observation_tensor)
        noise = torch.as_tensor(common_noise, dtype=torch.float32).reshape(1, 3).expand_as(distribution.mean)
        u = distribution.mean + distribution.stddev * noise
        actions = center + torch.bmm(generators, torch.tanh(u).unsqueeze(-1)).squeeze(-1)
        entropy = agent.alpha.detach().cpu() * entropy_decomposition(distribution, u, generators).physical_log_prob.detach().cpu()
    elif authority == ExecutionAuthority.CHARGER_CONSTRAINED.value:
        actions = torch.zeros((count, 3), dtype=torch.float32)
    else:
        actions = torch.as_tensor(context.get("kappa", np.zeros(3)), dtype=torch.float32).reshape(1, 3).expand(count, -1)
    with torch.no_grad():
        next_q = torch.minimum(
            agent.target_critic_1(observation_tensor, actions),
            agent.target_critic_2(observation_tensor, actions),
        )
    return next_q.numpy(), entropy.numpy(), bootstrap, authority


def _reversal(preferred_actions, labels, directions, axis):
    positive = labels.index("+x" if axis == 0 else "+y")
    negative = labels.index("-x" if axis == 0 else "-y")
    return bool(
        preferred_actions[positive, axis] * directions[positive, axis] > 0.0
        and preferred_actions[negative, axis] * directions[negative, axis] > 0.0
        and np.sign(preferred_actions[positive, axis]) != np.sign(preferred_actions[negative, axis])
    )


def _audit_checkpoint(path: Path, scenario: str, sample_count: int, goals_per_state: int, seed: int):
    environment = make_random_persistent_uav_env(f"{scenario}.json", seed=seed)
    initial_observation, _ = environment.reset(seed=seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    agent = PersistentGeneratorSAC(initial_observation.size, GeneratorSACConfig(**checkpoint["config"]), seed=seed)
    agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()
    agent.target_critic_1.eval()
    agent.target_critic_2.eval()
    trajectory = path.parent / "trajectory_events.jsonl"
    rows = [
        row for row in _load_rows(trajectory)
        if row.get("execution_authority") == "RL_GENERATOR"
        and row.get("goal") is not None
        and row.get("rl_authority_set_member") is not False
    ]
    indices = np.linspace(0, len(rows) - 1, min(sample_count, len(rows)), dtype=int)
    selected_rows = [rows[index] for index in indices]
    pool = _goal_pool(environment)
    oracle = BestInGeneratorGoalOracle()
    records = []
    representatives = []
    completion_pairs = 0
    invariance_failures = []
    for state_index, row in enumerate(selected_rows):
        goals_labeled = _counterfactual_goals(
            np.asarray(row["position"], dtype=np.float64),
            np.asarray(row["goal"], dtype=np.float64),
            pool,
            goals_per_state,
            environment.task_env.manager.minimum_goal_separation,
        )
        observations, contexts, _, observation_failures, certificate_failures = _state_goal_observations(
            environment, row, goals_labeled
        )
        if observation_failures or certificate_failures:
            invariance_failures.append({"state_index": state_index, "observation": observation_failures, "certificate": certificate_failures})
            continue
        context = contexts[0]
        if not all((
            context.get("generator_available"),
            context.get("recoverable_set_member") is True,
            context.get("rl_authority_set_member") is True,
            context.get("recoverability_action_verified") is True,
            context.get("continuation_action_verified") is True,
        )):
            continue
        state = environment.plant.state.copy()
        goals = [goal for _, goal in goals_labeled]
        goal_labels = [label for label, _ in goals_labeled]
        center = np.asarray(context["c"], dtype=np.float64)
        generator = np.asarray(context["G"], dtype=np.float64)
        with torch.no_grad():
            original_distribution = agent.actor.distribution(torch.as_tensor(observations[0:1], dtype=torch.float32))
            actor_eta = torch.tanh(original_distribution.mean)[0].numpy()
        oracle_etas = np.stack([
            oracle.select_eta(state, goal, center, generator, environment.plant.config.dt)
            for goal in goals
        ])
        oracle_actions = np.stack([action_from_eta(center, generator, eta) for eta in oracle_etas])
        action_labels, action_etas, actions = _unique_action_candidates(
            center, generator, actor_eta, oracle_etas, seed * 100000 + state_index
        )
        reward = np.zeros((len(goals), len(actions)), dtype=np.float64)
        task_reward = np.zeros_like(reward)
        generic_reward = np.zeros_like(reward)
        completion = np.zeros_like(reward, dtype=bool)
        progress = np.zeros_like(reward)
        next_q = np.zeros_like(reward)
        entropy = np.zeros_like(reward)
        bootstrap = np.ones_like(reward)
        authorities = []
        physical_invariance = True
        rng = np.random.default_rng(seed * 1000000 + state_index)
        for action_index, action in enumerate(actions):
            transition = nominal_physical_transition(environment, state, action)
            environment.task_env.manager.current_task.goal_position = np.asarray(goals[-1], dtype=np.float64).copy()
            repeated_transition = nominal_physical_transition(environment, state, action)
            next_observations, next_context = _next_observations(environment, goals, transition.state)
            q_values, entropy_values, bootstrap_values, authority = _target_terms(
                agent, next_observations, next_context, rng.standard_normal(3)
            )
            authorities.append(authority)
            next_q[:, action_index] = q_values
            entropy[:, action_index] = entropy_values
            bootstrap[:, action_index] = bootstrap_values
            for goal_index, goal in enumerate(goals):
                components = immediate_reward_components(environment, state, goal, transition)
                reward[goal_index, action_index] = float(components["total_reward"])
                task_reward[goal_index, action_index] = float(
                    components["goal_progress_reward"] + components["task_completion_reward"]
                )
                generic_reward[goal_index, action_index] = float(
                    components["elapsed_time_cost"]
                    + components["energy_cost"]
                    + components["backup_intervention_event_cost"]
                    + components["charging_dwell_cost"]
                )
                progress[goal_index, action_index] = float(components["goal_progress"])
                completion[goal_index, action_index] = bool(components["task_completed"])
            physical_invariance = physical_invariance and bool(
                np.array_equal(transition.state.position, repeated_transition.state.position)
                and np.array_equal(transition.state.velocity, repeated_transition.state.velocity)
                and transition.state.energy == repeated_transition.state.energy
                and transition.energy_cost == repeated_transition.energy_cost
                and transition.collision == repeated_transition.collision
                and transition.velocity_violation == repeated_transition.velocity_violation
            )
        if np.any(completion):
            completion_pairs += int(np.count_nonzero(completion))
            continue
        decomposition = bellman_target_decomposition(reward, next_q, entropy, agent.config.gamma, bootstrap)
        target = decomposition["target"]
        observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        with torch.no_grad():
            q_matrix = np.stack([
                torch.minimum(
                    agent.critic_1(observation_tensor[goal_index:goal_index + 1].expand(len(actions), -1), action_tensor),
                    agent.critic_2(observation_tensor[goal_index:goal_index + 1].expand(len(actions), -1), action_tensor),
                ).numpy()
                for goal_index in range(len(goals))
            ])
        reward_indices, reward_preferred = finite_preferred_actions(reward, actions)
        target_indices, target_preferred = finite_preferred_actions(target, actions)
        critic_indices, critic_preferred = finite_preferred_actions(q_matrix, actions)
        nearest_eta_index = lambda eta: int(np.argmin(np.linalg.norm(action_etas - np.asarray(eta), axis=1)))
        oracle_indices = np.asarray([nearest_eta_index(eta) for eta in oracle_etas])
        opposite_indices = np.asarray([nearest_eta_index(-eta) for eta in oracle_etas])
        center_index = nearest_eta_index(np.zeros(3))
        actor_index = nearest_eta_index(actor_eta)
        contrasts = {name: [] for name in ("oracle_opposite", "oracle_actor", "oracle_center")}
        comparison_indices = {
            "oracle_opposite": opposite_indices,
            "oracle_actor": np.full(len(goals), actor_index),
            "oracle_center": np.full(len(goals), center_index),
        }
        for name, compared in comparison_indices.items():
            for goal_index in range(len(goals)):
                row_decomposition = {key: value[goal_index] for key, value in decomposition.items()}
                contrasts[name].append(contrast_decomposition(
                    row_decomposition, int(oracle_indices[goal_index]), int(compared[goal_index])
                ))
        directions = np.stack([_goal_direction(state.position, goal) for goal in goals])
        target_alignment = [residual_alignment(target_preferred[index], oracle_actions[index], center) for index in range(len(goals))]
        critic_alignment = [residual_alignment(critic_preferred[index], oracle_actions[index], center) for index in range(len(goals))]
        critic_target_alignment = [residual_alignment(critic_preferred[index], target_preferred[index], center) for index in range(len(goals))]
        reward_additive = additive_decomposition_metrics(reward)
        target_additive = additive_decomposition_metrics(target)
        critic_additive = additive_decomposition_metrics(q_matrix)
        record = {
            "checkpoint_seed": seed,
            "state_index": state_index,
            "step": int(row.get("step", 0)),
            "position": state.position,
            "velocity": state.velocity,
            "energy": state.energy,
            "recovery_cell_id": context.get("recovery_cell_id"),
            "goal_count": len(goals),
            "action_count": len(actions),
            "physical_transition_invariant": physical_invariance,
            "S_R": mean_pairwise_action_distance(reward_preferred),
            "S_Y": mean_pairwise_action_distance(target_preferred),
            "S_Q": mean_pairwise_action_distance(critic_preferred),
            "S_oracle": mean_pairwise_action_distance(oracle_actions),
            "target_x_reversal": _reversal(target_preferred, goal_labels, directions, 0),
            "target_y_reversal": _reversal(target_preferred, goal_labels, directions, 1),
            "critic_x_reversal": _reversal(critic_preferred, goal_labels, directions, 0),
            "critic_y_reversal": _reversal(critic_preferred, goal_labels, directions, 1),
            "target_oracle_cosine": float(np.mean(target_alignment)),
            "critic_oracle_cosine": float(np.mean(critic_alignment)),
            "critic_target_cosine": float(np.mean(critic_target_alignment)),
            "reward_interaction_variance": reward_additive["interaction_variance"],
            "goal_dependent_reward_variance_fraction": float(np.var(task_reward)) / max(
                float(np.var(task_reward) + np.var(generic_reward)), 1e-12
            ),
            "target_interaction_variance": target_additive["interaction_variance"],
            "critic_interaction_variance": critic_additive["interaction_variance"],
            "reward_additive_explained": reward_additive["additive_explained_variance"],
            "target_additive_explained": target_additive["additive_explained_variance"],
            "critic_additive_explained": critic_additive["additive_explained_variance"],
            "interaction_preservation_ratio": float(target_additive["interaction_variance"]) / max(float(reward_additive["interaction_variance"]), 1e-12),
            "mean_reward": float(np.mean(reward)),
            "mean_gamma_next_q": float(np.mean(decomposition["gamma_next_q"])),
            "mean_negative_gamma_entropy": float(np.mean(decomposition["negative_gamma_entropy"])),
            "mean_target": float(np.mean(target)),
            "contrasts": contrasts,
            "authorities": {value: authorities.count(value) for value in sorted(set(authorities))},
        }
        records.append(record)
        if len(representatives) < 4:
            representatives.append({
                "state": {"checkpoint_seed": seed, "step": record["step"], "position": state.position, "velocity": state.velocity, "energy": state.energy, "recovery_cell_id": context.get("recovery_cell_id")},
                "goals": [{"label": label, "position": goal} for label, goal in goals_labeled],
                "c": center,
                "G": generator,
                "candidate_action_labels": action_labels,
                "candidate_etas": action_etas,
                "candidate_actions": actions,
                "R_ij": reward,
                "R_task_ij": task_reward,
                "R_generic_ij": generic_reward,
                "next_Q_ij": next_q,
                "entropy_ij": entropy,
                "Y_ij": target,
                "learned_Q_ij": q_matrix,
                "one_step_progress_ij": progress,
                "environment_oracle_actions": oracle_actions,
                "reward_preferred_actions": reward_preferred,
                "target_preferred_actions": target_preferred,
                "critic_preferred_actions": critic_preferred,
                "additive_interaction_residuals": {
                    "reward": reward_additive["interaction_residual"],
                    "target": target_additive["interaction_residual"],
                    "critic": critic_additive["interaction_residual"],
                },
                "metrics": record,
            })
    if invariance_failures:
        raise RuntimeError(f"TASK_INDEPENDENCE_REGRESSION: {invariance_failures[0]}")
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "physical_states": len(records),
        "records": records,
        "task_completion_pairs_excluded_from_primary": completion_pairs,
    }, representatives


def _checkpoint_worker(arguments):
    torch.set_num_threads(1)
    return _audit_checkpoint(*arguments)


def _flatten_contrasts(records, comparison, field):
    return [float(item[field]) for record in records for item in record["contrasts"][comparison]]


def _aggregate(checkpoints, representatives, goals_per_state):
    records = [record for checkpoint in checkpoints for record in checkpoint["records"]]
    values = lambda name: [float(record[name]) for record in records]
    interaction_ratio = float(np.mean(values("interaction_preservation_ratio")))
    target_sensitivity = float(np.mean(values("S_Y")))
    reward_sensitivity = float(np.mean(values("S_R")))
    oracle_sensitivity = float(np.mean(values("S_oracle")))
    target_alignment = float(np.mean(values("target_oracle_cosine")))
    target_reversal = 0.5 * (float(np.mean(values("target_x_reversal"))) + float(np.mean(values("target_y_reversal"))))
    positive_target_contrast = float(np.mean(np.asarray(_flatten_contrasts(records, "oracle_opposite", "target")) > 0.0))
    strong = interaction_ratio >= 0.25 and target_sensitivity >= 0.25 * max(oracle_sensitivity, 1e-12) and target_alignment > 0.0 and positive_target_contrast >= 0.6
    partial = (
        target_sensitivity >= 0.05 * max(oracle_sensitivity, 1e-12)
        or target_reversal >= 0.1
        or (positive_target_contrast >= 0.55 and target_alignment >= 0.25)
    )
    gate = "PASS" if strong else ("MARGINAL" if partial else "FAIL")
    rationale = (
        "Bellman targets preserve substantial goal-action interaction and certified action preference"
        if strong else
        "Bellman targets preserve only partial goal-action coupling"
        if partial else
        "Bellman targets substantially suppress immediate goal-action preference"
    )
    summary = {
        "metadata": {
            "checkpoints": [checkpoint["checkpoint"] for checkpoint in checkpoints],
            "physical_states": len(records),
            "goals_per_state": goals_per_state,
            "actions_per_state": _stats(record["action_count"] for record in records),
            "total_counterfactual_transitions": int(sum(record["goal_count"] * record["action_count"] for record in records)),
            "target_sampling_convention": "fixed common standard-normal reparameterization noise per physical state/action across goals",
            "primary_analysis_noncompletion": True,
            "task_completion_pairs_excluded": int(sum(checkpoint["task_completion_pairs_excluded_from_primary"] for checkpoint in checkpoints)),
            "training_steps": 0,
            "actor_updated": False,
            "critic_updated": False,
            "replay_modified": False,
            "safety_support_modified": False,
            "reward_modified": False,
        },
        "physical_transition_invariance": {
            "passed": all(record["physical_transition_invariant"] for record in records),
            "goal_affects_dynamics": False,
        },
        "immediate_reward": {
            "oracle_vs_opposite_delta_R": _stats(_flatten_contrasts(records, "oracle_opposite", "reward")),
            "oracle_vs_actor_delta_R": _stats(_flatten_contrasts(records, "oracle_actor", "reward")),
            "oracle_vs_center_delta_R": _stats(_flatten_contrasts(records, "oracle_center", "reward")),
            "oracle_vs_opposite_positive_fraction": float(np.mean(np.asarray(_flatten_contrasts(records, "oracle_opposite", "reward")) > 0.0)),
            "goal_action_interaction_variance": _stats(values("reward_interaction_variance")),
            "goal_dependent_variance_fraction": _stats(values("goal_dependent_reward_variance_fraction")),
        },
        "bellman_target": {
            "mean_reward": _stats(values("mean_reward")),
            "mean_gamma_Q_next": _stats(values("mean_gamma_next_q")),
            "mean_negative_gamma_entropy": _stats(values("mean_negative_gamma_entropy")),
            "mean_target": _stats(values("mean_target")),
            "oracle_vs_opposite": {
                "delta_R": _stats(_flatten_contrasts(records, "oracle_opposite", "reward")),
                "gamma_delta_Q_next": _stats(_flatten_contrasts(records, "oracle_opposite", "gamma_next_q")),
                "negative_gamma_delta_entropy": _stats(_flatten_contrasts(records, "oracle_opposite", "negative_gamma_entropy")),
                "delta_Y": _stats(_flatten_contrasts(records, "oracle_opposite", "target")),
                "positive_fraction": positive_target_contrast,
                "contrast_preservation_ratio": _stats(_flatten_contrasts(records, "oracle_opposite", "bellman_contrast_preservation_ratio")),
                "bootstrap_dominance_ratio": _stats(_flatten_contrasts(records, "oracle_opposite", "bootstrap_dominance_ratio")),
                "bootstrap_base_scale_ratio": _stats(_flatten_contrasts(records, "oracle_opposite", "bootstrap_base_scale_ratio")),
            },
            "goal_action_interaction_variance": _stats(values("target_interaction_variance")),
            "interaction_preservation_ratio": _stats(values("interaction_preservation_ratio")),
        },
        "target_preference": {
            "S_R": reward_sensitivity,
            "S_Y": target_sensitivity,
            "S_Q": float(np.mean(values("S_Q"))),
            "S_oracle": oracle_sensitivity,
            "target_x_reversal_fraction": float(np.mean(values("target_x_reversal"))),
            "target_y_reversal_fraction": float(np.mean(values("target_y_reversal"))),
            "critic_x_reversal_fraction": float(np.mean(values("critic_x_reversal"))),
            "critic_y_reversal_fraction": float(np.mean(values("critic_y_reversal"))),
            "target_vs_oracle_cosine": _stats(values("target_oracle_cosine")),
            "critic_vs_oracle_cosine": _stats(values("critic_oracle_cosine")),
            "critic_vs_target_cosine": _stats(values("critic_target_cosine")),
            "target_reversal_mean": target_reversal,
        },
        "additive_decomposition": {
            "reward": {"additive_explained_variance": _stats(values("reward_additive_explained")), "interaction_variance": _stats(values("reward_interaction_variance"))},
            "bellman_target": {"additive_explained_variance": _stats(values("target_additive_explained")), "interaction_variance": _stats(values("target_interaction_variance"))},
            "learned_critic": {"additive_explained_variance": _stats(values("critic_additive_explained")), "interaction_variance": _stats(values("critic_interaction_variance"))},
        },
        "BELLMAN_GOAL_ACTION_COUPLING_GATE": gate,
        "gate_rationale": rationale,
        "checkpoint_summaries": [{key: value for key, value in checkpoint.items() if key != "records"} for checkpoint in checkpoints],
        "representative_case_count": len(representatives),
        "synthetic_only": True,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", default=[f"artifacts/temp_compare_physical_seed{seed}/checkpoint_latest.pt" for seed in range(3)])
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--sample-count", type=int, default=75, help="states sampled per checkpoint")
    parser.add_argument("--goals-per-state", type=int, default=8)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", default="artifacts/random_persistent/bellman_goal_action_coupling_audit.json")
    parser.add_argument("--representative-output", default="artifacts/random_persistent/bellman_goal_action_representative_cases.json")
    args = parser.parse_args()
    tasks = [
        (ROOT / checkpoint, args.scenario, args.sample_count, args.goals_per_state, seed)
        for seed, checkpoint in enumerate(args.checkpoint)
    ]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(processes=min(args.workers, len(tasks))) as pool:
            results = pool.map(_checkpoint_worker, tasks)
    else:
        results = [_checkpoint_worker(task) for task in tasks]
    checkpoints = []
    representatives = []
    for summary, cases in results:
        checkpoints.append(summary)
        representatives.extend(cases)
    output = _aggregate(checkpoints, representatives[:10], args.goals_per_state)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default) + "\n", encoding="utf-8")
    representative_path = ROOT / args.representative_output
    representative_path.parent.mkdir(parents=True, exist_ok=True)
    representative_path.write_text(json.dumps({"representative_cases": representatives[:10]}, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"BELLMAN_GOAL_ACTION_COUPLING_GATE = {output['BELLMAN_GOAL_ACTION_COUPLING_GATE']}")
    print(json.dumps({key: output[key] for key in ("metadata", "immediate_reward", "bellman_target", "target_preference", "additive_decomposition", "gate_rationale")}, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
