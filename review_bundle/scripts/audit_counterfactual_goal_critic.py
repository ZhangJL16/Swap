#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cert_runtime.actor_gradient_diagnostics import critic_goal_jacobian, scalar_statistics
from cert_runtime.counterfactual_goal_diagnostics import (
    GOAL_DERIVED_OBSERVATION_FIELDS,
    action_gradient_goal_jacobian,
    certificate_invariance_snapshot,
    certificate_snapshots_equal,
    changed_observation_fields,
    critic_action_gradient,
    cross_goal_q_matrix,
    diagonal_preference,
    environment_cross_goal_matrix,
    finite_difference_action_gradient_goal_jacobian,
    mean_pairwise_cosine,
    mean_pairwise_distance,
    opposite_goal_preference_reversal,
    residual_alignment,
    searched_critic_preferred_actions,
)
from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.task_authority import BestInGeneratorGoalOracle, action_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _restore_physical_state(environment, row: dict[str, object]) -> None:
    environment.plant.state = UAVPhysicalState(
        np.asarray(row["position"], dtype=np.float64),
        np.asarray(row["velocity"], dtype=np.float64),
        float(row["energy"]),
        float(row.get("energy_error_radius", 0.0) or 0.0),
    )
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        environment.plant.state,
        environment.plant.world,
        environment.plant.np_random,
    )
    environment.task_env.mode = PersistentMissionMode.TASK_RL
    environment.task_env.phase = PersistentMissionMode.TASK_RL


def _goal_pool(environment) -> np.ndarray:
    terminal = environment.plant.scenario.terminal
    values = []
    for cell in environment.atlas.root_cells:
        if not cell.hash_valid or not cell.complete_successor_containment or cell.minimum_geometry_slack < -1e-12:
            continue
        point = np.asarray(cell.reference_position, dtype=np.float64)
        if np.all(point >= terminal.position_low) and np.all(point <= terminal.position_high):
            continue
        values.append(point)
    return np.unique(np.asarray(values), axis=0)


def _counterfactual_goals(
    position: np.ndarray,
    original_goal: np.ndarray,
    pool: np.ndarray,
    count: int,
    minimum_separation: float,
) -> tuple[tuple[str, np.ndarray], ...]:
    direction_specs = (
        ("+x", np.array((1.0, 0.0, 0.0))),
        ("-x", np.array((-1.0, 0.0, 0.0))),
        ("+y", np.array((0.0, 1.0, 0.0))),
        ("-y", np.array((0.0, -1.0, 0.0))),
        ("+xy", np.array((1.0, 1.0, 0.0))),
        ("-xy", np.array((-1.0, -1.0, 0.0))),
        ("+x-y", np.array((1.0, -1.0, 0.0))),
        ("-x+y", np.array((-1.0, 1.0, 0.0))),
    )
    result = [("original", np.asarray(original_goal, dtype=np.float64).copy())]
    used = {tuple(np.round(result[0][1], 9))}
    offsets = pool - np.asarray(position, dtype=np.float64)
    distances = np.linalg.norm(offsets, axis=1)
    valid = distances >= minimum_separation
    unit = offsets / np.maximum(distances[:, None], 1e-12)
    for label, requested in direction_specs:
        if len(result) >= count:
            break
        requested = requested / np.linalg.norm(requested)
        scores = unit @ requested + 0.02 * distances / max(float(np.max(distances)), 1e-12)
        for index in np.argsort(scores)[::-1]:
            candidate = pool[index]
            key = tuple(np.round(candidate, 9))
            if valid[index] and key not in used:
                result.append((label, candidate.copy()))
                used.add(key)
                break
    if len(result) < count:
        raise RuntimeError("COUNTERFACTUAL_GOAL_POOL_INSUFFICIENT")
    return tuple(result)


def _goal_direction(position: np.ndarray, goal: np.ndarray) -> np.ndarray:
    difference = np.asarray(goal, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    return difference / max(float(np.linalg.norm(difference)), 1e-12)


def _goal_projection(action: np.ndarray, position: np.ndarray, goal: np.ndarray) -> float:
    return float(np.asarray(action, dtype=np.float64) @ _goal_direction(position, goal))


def _state_goal_observations(environment, row, goals):
    _restore_physical_state(environment, row)
    task = environment.task_env.manager.current_task
    observations = []
    contexts = []
    snapshots = []
    for _, goal in goals:
        task.goal_position = np.asarray(goal, dtype=np.float64).copy()
        environment._context_cache_key = None
        environment._context_cache = None
        context = environment._refresh_context()
        observation = environment.task_env.build_observation(
            environment.runtime._map_encoding(), environment.runtime._corridor_encoding()
        )
        observations.append(observation)
        contexts.append(context)
        snapshots.append(certificate_invariance_snapshot(context))
    baseline_observation = observations[0]
    baseline_snapshot = snapshots[0]
    observation_failures = []
    certificate_failures = []
    for index in range(1, len(goals)):
        changed = changed_observation_fields(
            baseline_observation,
            observations[index],
            environment.task_env.observation_layout,
            tolerance=1e-7,
        )
        if not set(changed).issubset(GOAL_DERIVED_OBSERVATION_FIELDS):
            observation_failures.append({"goal": goals[index][0], "changed_fields": changed})
        equal, failures = certificate_snapshots_equal(baseline_snapshot, snapshots[index])
        if not equal:
            certificate_failures.append({"goal": goals[index][0], "fields": failures})
    task.goal_position = np.asarray(goals[0][1], dtype=np.float64).copy()
    environment._context_cache_key = None
    environment._context_cache = None
    return (
        np.stack(observations),
        contexts,
        snapshots,
        observation_failures,
        certificate_failures,
    )


def _network_snapshot(agent) -> dict[str, torch.Tensor]:
    values = {}
    for prefix, network in (("actor", agent.actor), ("critic_1", agent.critic_1), ("critic_2", agent.critic_2)):
        values.update({f"{prefix}.{name}": value.detach().clone() for name, value in network.state_dict().items()})
    return values


def _network_unchanged(before, agent) -> bool:
    after = _network_snapshot(agent)
    return before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before)


def _checkpoint_audit(checkpoint_path: Path, scenario: str, sample_count: int, goals_per_state: int, seed: int):
    environment = make_random_persistent_uav_env(f"{scenario}.json", seed=seed)
    initial_observation, _ = environment.reset(seed=seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    agent = PersistentGeneratorSAC(
        initial_observation.size,
        GeneratorSACConfig(**checkpoint["config"]),
        seed=seed,
    )
    agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic_1.eval()
    agent.critic_2.eval()
    network_before = _network_snapshot(agent)
    checkpoint_hash_before = _sha256(checkpoint_path)
    trajectory_path = checkpoint_path.parent / "trajectory_events.jsonl"
    trajectory_hash_before = _sha256(trajectory_path)
    rows = [
        row for row in _load_rows(trajectory_path)
        if row.get("execution_authority") == "RL_GENERATOR"
        and row.get("goal") is not None
        and row.get("rl_authority_member") is not False
    ]
    selected_indices = np.linspace(0, len(rows) - 1, min(sample_count, len(rows)), dtype=int)
    selected_rows = [rows[index] for index in selected_indices]
    pool = _goal_pool(environment)
    oracle = BestInGeneratorGoalOracle()
    records = []
    representatives = []
    invariance_failures = []
    finite_difference_errors = []
    for state_index, row in enumerate(selected_rows):
        original_goal = np.asarray(row["goal"], dtype=np.float64)
        goals = _counterfactual_goals(
            np.asarray(row["position"], dtype=np.float64),
            original_goal,
            pool,
            goals_per_state,
            environment.task_env.manager.minimum_goal_separation,
        )
        observations, contexts, snapshots, observation_failures, certificate_failures = _state_goal_observations(
            environment, row, goals
        )
        if observation_failures or certificate_failures:
            invariance_failures.append({
                "state_index": state_index,
                "observation": observation_failures,
                "certificate": certificate_failures,
            })
            continue
        context = contexts[0]
        if not (
            context.get("generator_available")
            and context.get("recoverable_set_member") is True
            and context.get("rl_authority_set_member") is True
            and context.get("recoverability_action_verified") is True
            and context.get("continuation_action_verified") is True
        ):
            continue
        center = np.asarray(context["c"], dtype=np.float64)
        generator = np.asarray(context["G"], dtype=np.float64)
        observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
        center_tensor = torch.as_tensor(center, dtype=torch.float32).reshape(1, 3)
        generator_tensor = torch.as_tensor(generator, dtype=torch.float32).reshape(1, 3, 3)
        with torch.no_grad():
            actor_u = agent.actor.distribution(observation_tensor).mean
            actor_eta = torch.tanh(actor_u)
            actor_actions = center_tensor + torch.matmul(generator_tensor, actor_eta.unsqueeze(-1)).squeeze(-1)
        state = environment.plant.state.copy()
        goal_values = [goal for _, goal in goals]
        oracle_eta = np.stack([
            oracle.select_eta(state, goal, center, generator, environment.plant.config.dt)
            for goal in goal_values
        ])
        oracle_actions = np.stack([
            action_from_eta(center, generator, eta) for eta in oracle_eta
        ])
        preferred = searched_critic_preferred_actions(
            agent,
            observation_tensor,
            center,
            generator,
            actor_eta.detach().cpu().numpy(),
            seed=seed * 100000 + state_index,
        )
        preferred_eta = np.stack([item.eta for item in preferred])
        preferred_actions = np.stack([item.action for item in preferred])
        center_action = torch.as_tensor(center, dtype=torch.float32).reshape(1, 3)
        original_actor_action = actor_actions[0:1].detach()
        center_gradients = []
        actor_reference_gradients = []
        fixed_center_q = []
        fixed_actor_q = []
        fixed_neutral_q = []
        neutral_eta = np.array((0.25, 0.0, 0.0), dtype=np.float64)
        neutral_action = torch.as_tensor(action_from_eta(center, generator, neutral_eta), dtype=torch.float32).reshape(1, 3)
        directional_to_oracle = []
        direct_goal_jacobians = []
        for goal_index in range(len(goals)):
            selected_observation = observation_tensor[goal_index:goal_index + 1]
            q_center, center_gradient = critic_action_gradient(agent, selected_observation, center_action)
            q_actor, actor_gradient = critic_action_gradient(agent, selected_observation, original_actor_action)
            q_neutral, _ = critic_action_gradient(agent, selected_observation, neutral_action)
            center_gradients.append(center_gradient[0].detach().numpy())
            actor_reference_gradients.append(actor_gradient[0].detach().numpy())
            fixed_center_q.append(float(q_center))
            fixed_actor_q.append(float(q_actor))
            fixed_neutral_q.append(float(q_neutral))
            direction = torch.as_tensor(oracle_actions[goal_index], dtype=torch.float32).reshape(1, 3) - original_actor_action
            direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True) + 1e-12
            directional_to_oracle.append(float((actor_gradient * direction).sum()))
            direct_goal_jacobians.append(float(torch.linalg.vector_norm(
                critic_goal_jacobian(
                    agent,
                    selected_observation,
                    center_action,
                    environment.task_env.observation_layout["goal_delta"],
                )
            )))
        mixed = action_gradient_goal_jacobian(
            agent,
            observation_tensor[0:1],
            center_action,
            environment.task_env.observation_layout["goal_delta"],
        )
        if state_index < 5:
            finite = finite_difference_action_gradient_goal_jacobian(
                agent,
                observation_tensor[0:1],
                center_action,
                environment.task_env.observation_layout["goal_delta"],
            )
            finite_difference_errors.append(float(torch.linalg.matrix_norm(mixed - finite)))
        q_matrix = cross_goal_q_matrix(
            agent,
            observation_tensor,
            torch.as_tensor(preferred_actions, dtype=torch.float32),
        )
        env_oracle_reward, env_oracle_progress = environment_cross_goal_matrix(
            state,
            goal_values,
            oracle_actions,
            environment.plant.config.dt,
            environment.plant.energy_model,
            environment.task_env.reward_config,
        )
        _, preferred_progress = environment_cross_goal_matrix(
            state,
            goal_values,
            preferred_actions,
            environment.plant.config.dt,
            environment.plant.energy_model,
            environment.task_env.reward_config,
        )
        q_diagonal_fraction, q_diagonal_advantage = diagonal_preference(q_matrix)
        env_diagonal_fraction, env_diagonal_advantage = diagonal_preference(env_oracle_reward)
        alignments = [
            residual_alignment(preferred_actions[index], oracle_actions[index], center)
            for index in range(len(goals))
        ]
        action_distances = [
            float(np.linalg.norm(preferred_actions[index] - oracle_actions[index]))
            for index in range(len(goals))
        ]
        labels = [label for label, _ in goals]
        label_index = {label: index for index, label in enumerate(labels)}
        x_reversal = opposite_goal_preference_reversal(
            preferred_actions[label_index["+x"]],
            preferred_actions[label_index["-x"]],
            _goal_direction(state.position, goal_values[label_index["+x"]]),
            _goal_direction(state.position, goal_values[label_index["-x"]]),
            0,
        )
        y_reversal = opposite_goal_preference_reversal(
            preferred_actions[label_index["+y"]],
            preferred_actions[label_index["-y"]],
            _goal_direction(state.position, goal_values[label_index["+y"]]),
            _goal_direction(state.position, goal_values[label_index["-y"]]),
            1,
        )
        actor_sensitivity = mean_pairwise_distance(actor_actions.detach().cpu().numpy())
        preferred_sensitivity = mean_pairwise_distance(preferred_actions)
        oracle_sensitivity = mean_pairwise_distance(oracle_actions)
        record = {
            "checkpoint_seed": seed,
            "state_index": state_index,
            "step": int(row.get("step", 0)),
            "position": state.position.copy(),
            "velocity": state.velocity.copy(),
            "energy": state.energy,
            "energy_margin": float(context["energy_margin"]),
            "recovery_cell_id": context.get("recovery_cell_id"),
            "goal_count": len(goals),
            "Q_variance_center": float(np.var(fixed_center_q)),
            "Q_variance_actor_reference": float(np.var(fixed_actor_q)),
            "Q_variance_neutral": float(np.var(fixed_neutral_q)),
            "critic_goal_jacobian_norm": float(np.mean(direct_goal_jacobians)),
            "mixed_derivative_norm": float(torch.linalg.matrix_norm(mixed)),
            "gradient_pairwise_distance": mean_pairwise_distance(center_gradients),
            "gradient_pairwise_cosine": mean_pairwise_cosine(center_gradients),
            "actor_reference_gradient_pairwise_distance": mean_pairwise_distance(actor_reference_gradients),
            "preferred_action_sensitivity": preferred_sensitivity,
            "preferred_eta_sensitivity": mean_pairwise_distance(preferred_eta),
            "actor_action_sensitivity": actor_sensitivity,
            "oracle_action_sensitivity": oracle_sensitivity,
            "actor_to_critic_sensitivity_ratio": actor_sensitivity / max(preferred_sensitivity, 1e-12),
            "critic_to_oracle_sensitivity_ratio": preferred_sensitivity / max(oracle_sensitivity, 1e-12),
            "x_reversal": x_reversal,
            "y_reversal": y_reversal,
            "alignment_mean": float(np.mean(alignments)),
            "alignment_median": float(np.median(alignments)),
            "alignment_positive_fraction": float(np.mean(np.asarray(alignments) > 0.0)),
            "alignment_above_half_fraction": float(np.mean(np.asarray(alignments) > 0.5)),
            "preferred_oracle_action_distance": float(np.mean(action_distances)),
            "preferred_goal_projection": float(np.mean([
                _goal_projection(preferred_actions[index], state.position, goal_values[index])
                for index in range(len(goals))
            ])),
            "oracle_goal_projection": float(np.mean([
                _goal_projection(oracle_actions[index], state.position, goal_values[index])
                for index in range(len(goals))
            ])),
            "preferred_one_step_progress": float(np.mean(np.diag(preferred_progress))),
            "oracle_one_step_progress": float(np.mean(np.diag(env_oracle_progress))),
            "directional_to_oracle_positive_fraction": float(np.mean(np.asarray(directional_to_oracle) > 0.0)),
            "q_diagonal_preference_fraction": q_diagonal_fraction,
            "q_diagonal_advantage": q_diagonal_advantage,
            "environment_diagonal_preference_fraction": env_diagonal_fraction,
            "environment_diagonal_advantage": env_diagonal_advantage,
        }
        records.append(record)
        if len(representatives) < 4:
            representatives.append({
                "state": {
                    "checkpoint_seed": seed,
                    "step": record["step"],
                    "position": state.position,
                    "velocity": state.velocity,
                    "energy": state.energy,
                    "recovery_cell_id": context.get("recovery_cell_id"),
                },
                "goals": [
                    {
                        "label": labels[index],
                        "position": goal_values[index],
                        "direction": _goal_direction(state.position, goal_values[index]),
                        "actor_action": actor_actions[index].detach().cpu().numpy(),
                        "critic_preferred_action": preferred_actions[index],
                        "critic_preferred_eta": preferred_eta[index],
                        "critic_preferred_Q": preferred[index].q_value,
                        "critic_search_source": preferred[index].source,
                        "oracle_action": oracle_actions[index],
                        "grad_a_Q_at_center": center_gradients[index],
                        "critic_preferred_goal_projection": _goal_projection(
                            preferred_actions[index], state.position, goal_values[index]
                        ),
                        "oracle_goal_projection": _goal_projection(
                            oracle_actions[index], state.position, goal_values[index]
                        ),
                    }
                    for index in range(len(goals))
                ],
                "c": center,
                "G": generator,
                "cross_goal_Q_matrix": q_matrix,
                "environment_oracle_reward_matrix": env_oracle_reward,
                "environment_oracle_progress_matrix": env_oracle_progress,
                "metrics": record,
            })
    if invariance_failures:
        raise RuntimeError(f"TASK_INDEPENDENCE_REGRESSION: {invariance_failures[0]}")
    checkpoint_summary = {
        "checkpoint": str(checkpoint_path.relative_to(ROOT)),
        "trajectory": str(trajectory_path.relative_to(ROOT)),
        "physical_states": len(records),
        "state_goal_pairs": sum(record["goal_count"] for record in records),
        "unique_recovery_cells": len({record["recovery_cell_id"] for record in records}),
        "finite_difference_mixed_derivative_error": scalar_statistics(finite_difference_errors),
        "network_unchanged": _network_unchanged(network_before, agent),
        "checkpoint_hash_unchanged": checkpoint_hash_before == _sha256(checkpoint_path),
        "trajectory_hash_unchanged": trajectory_hash_before == _sha256(trajectory_path),
        "records": records,
    }
    return checkpoint_summary, representatives


def _aggregate(checkpoints, representatives, goals_per_state):
    records = [record for checkpoint in checkpoints for record in checkpoint["records"]]

    def values(name):
        return [float(record[name]) for record in records]

    preferred_sensitivity = float(np.mean(values("preferred_action_sensitivity")))
    actor_sensitivity = float(np.mean(values("actor_action_sensitivity")))
    oracle_sensitivity = float(np.mean(values("oracle_action_sensitivity")))
    critic_oracle_ratio = preferred_sensitivity / max(oracle_sensitivity, 1e-12)
    actor_critic_ratio = actor_sensitivity / max(preferred_sensitivity, 1e-12)
    alignment_mean = float(np.mean(values("alignment_mean")))
    directional_positive = float(np.mean(values("directional_to_oracle_positive_fraction")))
    x_reversal = float(np.mean(values("x_reversal")))
    y_reversal = float(np.mean(values("y_reversal")))
    diagonal_advantage = float(np.mean(values("q_diagonal_advantage")))
    if (
        critic_oracle_ratio >= 0.5
        and alignment_mean > 0.0
        and directional_positive > 0.5
        and diagonal_advantage > 0.0
        and 0.5 * (x_reversal + y_reversal) >= 0.5
    ):
        classification = "CRITIC_GOAL_CONDITIONED"
        gate = "PASS"
        rationale = "preferred actions vary comparably to the task oracle and usually align with it"
        next_candidate = "actor objective exploitation-balance audit"
    elif critic_oracle_ratio < 0.25 and 0.5 * (x_reversal + y_reversal) < 0.25:
        classification = "CRITIC_GOAL_INSENSITIVE"
        gate = "FAIL"
        rationale = "goal-dependent values do not induce comparable certified-action preference changes"
        next_candidate = "critic Bellman goal-learning and reward-horizon audit"
    elif critic_oracle_ratio >= 0.25 and (alignment_mean <= 0.0 or directional_positive <= 0.5):
        classification = "CRITIC_GOAL_MISALIGNED"
        gate = "FAIL"
        rationale = "critic preferences change with goals but do not consistently align with task progress"
        next_candidate = "reward-to-Q propagation and Bellman target audit"
    else:
        classification = "MIXED"
        gate = "MARGINAL"
        rationale = "checkpoint or state subsets provide conflicting control-preference evidence"
        next_candidate = "subdivide critic preference by checkpoint and recovery-cell distribution"
    return {
        "metadata": {
            "physical_states": len(records),
            "goals_per_state": goals_per_state,
            "state_goal_pairs": sum(record["goal_count"] for record in records),
            "checkpoints": [checkpoint["checkpoint"] for checkpoint in checkpoints],
            "training_run": False,
            "oracle_evaluation_only": True,
            "replay_loaded": False,
        },
        "certificate_invariance": {
            "passed": True,
            "c_invariant": True,
            "G_invariant": True,
            "R_invariant": True,
            "R_RL_invariant": True,
            "E_kappa_invariant": True,
            "energy_margin_invariant": True,
            "certificate_identity_invariant": True,
            "continuation_support_invariant": True,
            "atlas_hash_invariant": True,
        },
        "value_goal_sensitivity": {
            "Q_variance_center": scalar_statistics(values("Q_variance_center")),
            "Q_variance_actor_reference": scalar_statistics(values("Q_variance_actor_reference")),
            "Q_variance_neutral": scalar_statistics(values("Q_variance_neutral")),
            "critic_goal_jacobian_norm": scalar_statistics(values("critic_goal_jacobian_norm")),
        },
        "action_gradient_goal_sensitivity": {
            "mixed_derivative_norm": scalar_statistics(values("mixed_derivative_norm")),
            "pairwise_gradient_distance": scalar_statistics(values("gradient_pairwise_distance")),
            "pairwise_gradient_cosine": scalar_statistics(values("gradient_pairwise_cosine")),
            "actor_reference_pairwise_gradient_distance": scalar_statistics(values("actor_reference_gradient_pairwise_distance")),
        },
        "critic_preferred_action": {
            "search_method": "actor/center/corners/fixed-seed random starts plus projected gradient ascent and random cross-check",
            "all_actions_inside_generator_support": True,
            "preferred_action_pairwise_distance": scalar_statistics(values("preferred_action_sensitivity")),
            "preferred_eta_pairwise_distance": scalar_statistics(values("preferred_eta_sensitivity")),
        },
        "opposite_goals": {
            "x_reversal_fraction": x_reversal,
            "y_reversal_fraction": y_reversal,
        },
        "critic_vs_environment_oracle": {
            "action_cosine": scalar_statistics(values("alignment_mean")),
            "median_action_cosine_by_state": scalar_statistics(values("alignment_median")),
            "positive_cosine_fraction": float(np.mean(values("alignment_positive_fraction"))),
            "cosine_above_half_fraction": float(np.mean(values("alignment_above_half_fraction"))),
            "action_distance": scalar_statistics(values("preferred_oracle_action_distance")),
            "critic_preferred_goal_projection": scalar_statistics(values("preferred_goal_projection")),
            "oracle_goal_projection": scalar_statistics(values("oracle_goal_projection")),
            "critic_preferred_one_step_progress": scalar_statistics(values("preferred_one_step_progress")),
            "oracle_one_step_progress": scalar_statistics(values("oracle_one_step_progress")),
        },
        "counterfactual_directional_gradient": {
            "positive_fraction": directional_positive,
        },
        "cross_goal_Q_matrix": {
            "diagonal_preference_fraction": float(np.mean(values("q_diagonal_preference_fraction"))),
            "mean_diagonal_advantage": diagonal_advantage,
            "environment_diagonal_preference_fraction": float(np.mean(values("environment_diagonal_preference_fraction"))),
            "environment_mean_diagonal_advantage": float(np.mean(values("environment_diagonal_advantage"))),
        },
        "actor_vs_critic_vs_oracle": {
            "S_actor": actor_sensitivity,
            "S_Q": preferred_sensitivity,
            "S_oracle": oracle_sensitivity,
            "S_actor_over_S_Q": actor_critic_ratio,
            "S_Q_over_S_oracle": critic_oracle_ratio,
        },
        "PRIMARY_CLASSIFICATION": classification,
        "COUNTERFACTUAL_GOAL_CRITIC_GATE": gate,
        "classification_rationale": rationale,
        "NEXT_SINGLE_CANDIDATE": next_candidate,
        "representative_cases": representatives[:10],
        "checkpoint_summaries": [
            {key: value for key, value in checkpoint.items() if key != "records"}
            for checkpoint in checkpoints
        ],
        "network_and_artifact_integrity": {
            "all_networks_unchanged": all(checkpoint["network_unchanged"] for checkpoint in checkpoints),
            "all_checkpoint_hashes_unchanged": all(checkpoint["checkpoint_hash_unchanged"] for checkpoint in checkpoints),
            "all_trajectory_hashes_unchanged": all(checkpoint["trajectory_hash_unchanged"] for checkpoint in checkpoints),
            "replay_modified": False,
            "checkpoint_modified": False,
            "oracle_used_for_training": False,
        },
        "synthetic_only": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        default=[f"artifacts/temp_compare_physical_seed{seed}/checkpoint_latest.pt" for seed in range(3)],
    )
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--sample-count", type=int, default=100, help="physical states per checkpoint")
    parser.add_argument("--goals-per-state", type=int, default=8)
    parser.add_argument("--output", default="artifacts/random_persistent/counterfactual_goal_critic_audit.json")
    args = parser.parse_args()
    if not 6 <= args.goals_per_state <= 9:
        raise ValueError("goals-per-state must be between 6 and 9")
    checkpoints = []
    representatives = []
    for seed, name in enumerate(args.checkpoint):
        checkpoint, selected_representatives = _checkpoint_audit(
            (ROOT / name).resolve(),
            args.scenario,
            args.sample_count,
            args.goals_per_state,
            seed,
        )
        checkpoints.append(checkpoint)
        representatives.extend(selected_representatives)
    output = _aggregate(checkpoints, representatives, args.goals_per_state)
    if output["metadata"]["physical_states"] < 200:
        raise RuntimeError("COUNTERFACTUAL_AUDIT_REQUIRES_AT_LEAST_200_PHYSICAL_STATES")
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(path.relative_to(ROOT)),
        "physical_states": output["metadata"]["physical_states"],
        "state_goal_pairs": output["metadata"]["state_goal_pairs"],
        "PRIMARY_CLASSIFICATION": output["PRIMARY_CLASSIFICATION"],
        "COUNTERFACTUAL_GOAL_CRITIC_GATE": output["COUNTERFACTUAL_GOAL_CRITIC_GATE"],
        "NEXT_SINGLE_CANDIDATE": output["NEXT_SINGLE_CANDIDATE"],
    }, indent=2))


if __name__ == "__main__":
    main()
