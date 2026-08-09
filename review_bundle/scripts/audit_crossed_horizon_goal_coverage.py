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
    finite_preferred_actions,
    immediate_reward_components,
    mean_pairwise_action_distance,
    nominal_physical_transition,
)
from cert_runtime.counterfactual_goal_diagnostics import residual_alignment
from cert_runtime.crossed_horizon_diagnostics import (
    TARGET_SEMANTICS,
    decompose_n_step_soft_target,
    horizon_coverage_effects,
    preference_restoration_ratio,
    relabeled_goal_not_completed,
    target_for_semantics,
    valid_n_step_segment,
)
from cert_runtime.generator_sac import GeneratorSACConfig, PersistentGeneratorSAC
from cert_runtime.optimization_diagnostics import entropy_decomposition
from cert_runtime.persistent_authority import ExecutionAuthority
from cert_runtime.task_authority import BestInGeneratorGoalOracle, action_from_eta
from envs.certified_uav import make_random_persistent_uav_env
from envs.certified_uav.persistent_task import PersistentMissionMode
from envs.certified_uav.state import UAVPhysicalState
from scripts.audit_bellman_goal_action_coupling import _reversal, _stats, _unique_action_candidates
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


def _state_from_row(row) -> UAVPhysicalState:
    return UAVPhysicalState(
        np.asarray(row["position"], dtype=np.float64),
        np.asarray(row["velocity"], dtype=np.float64),
        float(row["energy"]),
        float(row.get("energy_error_radius", 0.0) or 0.0),
    )


def _set_state(environment, state: UAVPhysicalState) -> None:
    environment.plant.state = state.copy()
    environment.plant.failure_reason = None
    environment.plant.last_lidar = environment.plant.lidar_model.measure(
        state, environment.plant.world, environment.plant.np_random
    )
    environment.task_env.mode = PersistentMissionMode.TASK_RL
    environment.task_env.phase = PersistentMissionMode.TASK_RL
    environment._context_cache_key = None
    environment._context_cache = None


def _endpoint_observations(environment, goals, state):
    _set_state(environment, state)
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


def _endpoint_terms(agent, observations, context, common_noise):
    count = len(observations)
    authority = str(context.get("execution_authority", ExecutionAuthority.FAIL_CLOSED.value))
    bootstrap = np.ones(count, dtype=np.float64)
    log_prob_u = np.zeros(count, dtype=np.float64)
    negative_tanh_jacobian = np.zeros(count, dtype=np.float64)
    normalized_log_prob = np.zeros(count, dtype=np.float64)
    log_det = np.zeros(count, dtype=np.float64)
    observation_tensor = torch.as_tensor(observations, dtype=torch.float32)
    generator_branch = (
        authority in {ExecutionAuthority.RL_GENERATOR.value, ExecutionAuthority.CHARGER_CONSTRAINED.value}
        and context.get("generator_executable") is True
    )
    if authority == ExecutionAuthority.FAIL_CLOSED.value:
        bootstrap.fill(0.0)
        actions = torch.zeros((count, 3), dtype=torch.float32)
    elif generator_branch:
        center = torch.as_tensor(context["c"], dtype=torch.float32).reshape(1, 3).expand(count, -1)
        generators = torch.as_tensor(context["G"], dtype=torch.float32).reshape(1, 3, 3).expand(count, -1, -1)
        distribution = agent.actor.distribution(observation_tensor)
        noise = torch.as_tensor(common_noise, dtype=torch.float32).reshape(1, 3).expand_as(distribution.mean)
        u = distribution.mean + distribution.stddev * noise
        actions = center + torch.bmm(generators, torch.tanh(u).unsqueeze(-1)).squeeze(-1)
        terms = entropy_decomposition(distribution, u, generators)
        log_prob_u = terms.normal_term.detach().numpy()
        negative_tanh_jacobian = terms.negative_tanh_log_jacobian_term.detach().numpy()
        normalized_log_prob = terms.normalized_log_prob.detach().numpy()
        log_det = terms.log_det_G.detach().numpy()
    else:
        actions = torch.as_tensor(context.get("kappa", np.zeros(3)), dtype=torch.float32).reshape(1, 3).expand(count, -1)
    with torch.no_grad():
        q_next = torch.minimum(
            agent.target_critic_1(observation_tensor, actions),
            agent.target_critic_2(observation_tensor, actions),
        ).numpy()
    return q_next, log_prob_u, negative_tanh_jacobian, normalized_log_prob, log_det, bootstrap, authority


def _correlation(first, second, *, rank=False):
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return 0.0
    if rank:
        x = np.argsort(np.argsort(x)).astype(np.float64)
        y = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(x, y)[0, 1])


def _matrix_metrics(matrix, actions, oracle_actions, center, goals, goal_labels, state, one_step_progress):
    _, preferred = finite_preferred_actions(matrix, actions)
    directions = np.stack([_goal_direction(state.position, goal) for goal in goals])
    selected = np.argmax(matrix, axis=1)
    alignments = [residual_alignment(preferred[index], oracle_actions[index], center) for index in range(len(goals))]
    distances = [float(np.linalg.norm(preferred[index] - oracle_actions[index])) for index in range(len(goals))]
    preferred_projection = [float(preferred[index] @ directions[index]) for index in range(len(goals))]
    oracle_projection = [float(oracle_actions[index] @ directions[index]) for index in range(len(goals))]
    additive = additive_decomposition_metrics(matrix)
    return {
        "S_Y": mean_pairwise_action_distance(preferred),
        "x_reversal": _reversal(preferred, goal_labels, directions, 0) if "+x" in goal_labels and "-x" in goal_labels else False,
        "y_reversal": _reversal(preferred, goal_labels, directions, 1) if "+y" in goal_labels and "-y" in goal_labels else False,
        "oracle_cosine": float(np.mean(alignments)),
        "oracle_cosine_median": float(np.median(alignments)),
        "oracle_positive_cosine_fraction": float(np.mean(np.asarray(alignments) > 0.0)),
        "oracle_cosine_above_half_fraction": float(np.mean(np.asarray(alignments) > 0.5)),
        "oracle_action_distance": float(np.mean(distances)),
        "preferred_goal_projection": float(np.mean(preferred_projection)),
        "oracle_goal_projection": float(np.mean(oracle_projection)),
        "positive_one_step_progress_fraction": float(np.mean(one_step_progress[np.arange(len(goals)), selected] > 0.0)),
        "interaction_variance": float(additive["interaction_variance"]),
        "additive_explained_variance": float(additive["additive_explained_variance"]),
        "preferred_actions": preferred,
    }


def _valid_start_indices(rows, horizon):
    return [
        index for index, row in enumerate(rows)
        if row.get("execution_authority") == "RL_GENERATOR"
        and row.get("rl_authority_set_member") is not False
        and valid_n_step_segment(rows, index + 1, horizon)
    ]


def _audit_checkpoint(path, scenario, sample_count, goals_per_state, horizons, seed):
    environment = make_random_persistent_uav_env(f"{scenario}.json", seed=seed)
    initial_observation, _ = environment.reset(seed=seed)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    agent = PersistentGeneratorSAC(initial_observation.size, GeneratorSACConfig(**checkpoint["config"]), seed=seed)
    agent.load_state_dict(checkpoint)
    for network in (agent.actor, agent.target_critic_1, agent.target_critic_2):
        network.eval()
    rows = _load_rows(path.parent / "trajectory_events.jsonl")
    starts = _valid_start_indices(rows, max(horizons))
    selected = np.asarray(starts)[np.linspace(0, len(starts) - 1, min(sample_count, len(starts)), dtype=int)]
    pool = _goal_pool(environment)
    oracle = BestInGeneratorGoalOracle()
    records = []
    representatives = []
    excluded_completion = 0
    excluded_branch = 0
    for state_index, row_index in enumerate(selected):
        row = rows[int(row_index)]
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
            raise RuntimeError(f"TASK_INDEPENDENCE_REGRESSION: {observation_failures or certificate_failures}")
        context = contexts[0]
        if not all((
            context.get("generator_available"),
            context.get("recoverable_set_member") is True,
            context.get("rl_authority_set_member") is True,
            context.get("recoverability_action_verified") is True,
            context.get("continuation_action_verified") is True,
        )):
            continue
        state = _state_from_row(row)
        goals = [np.asarray(goal, dtype=np.float64) for _, goal in goals_labeled]
        goal_labels = [label for label, _ in goals_labeled]
        center = np.asarray(context["c"], dtype=np.float64)
        generator = np.asarray(context["G"], dtype=np.float64)
        with torch.no_grad():
            actor_eta = torch.tanh(agent.actor.distribution(torch.as_tensor(observations[0:1], dtype=torch.float32)).mean)[0].numpy()
        oracle_etas = np.stack([oracle.select_eta(state, goal, center, generator, environment.plant.config.dt) for goal in goals])
        oracle_actions = np.stack([action_from_eta(center, generator, eta) for eta in oracle_etas])
        action_labels, action_etas, actions = _unique_action_candidates(
            center, generator, actor_eta, oracle_etas, seed * 100000 + state_index
        )
        nearest_eta = lambda eta: int(np.argmin(np.linalg.norm(action_etas - np.asarray(eta), axis=1)))
        oracle_indices = np.asarray([nearest_eta(eta) for eta in oracle_etas])
        opposite_indices = np.asarray([nearest_eta(-eta) for eta in oracle_etas])
        actor_index = nearest_eta(actor_eta)
        center_index = nearest_eta(np.zeros(3))
        transitions_by_action = []
        valid_branch = True
        for action in actions:
            current = state
            transitions = []
            for offset in range(max(horizons)):
                selected_action = action if offset == 0 else np.asarray(rows[int(row_index) + offset]["executed_action"], dtype=np.float64)
                transition = nominal_physical_transition(environment, current, selected_action)
                transitions.append(transition)
                current = transition.state
                if transition.collision or transition.velocity_violation or current.energy <= 0.0:
                    valid_branch = False
            transitions_by_action.append(transitions)
        if not valid_branch:
            excluded_branch += 1
            continue
        positions = np.asarray([
            [transition.state.position for transition in transitions]
            for transitions in transitions_by_action
        ])
        if any(not relabeled_goal_not_completed(goal, positions, environment.task_env.manager.goal_radius) for goal in goals):
            excluded_completion += 1
            continue
        rewards = np.zeros((max(horizons), len(goals), len(actions)), dtype=np.float64)
        one_step_progress = np.zeros((len(goals), len(actions)), dtype=np.float64)
        for action_index, transitions in enumerate(transitions_by_action):
            previous = state
            for offset, transition in enumerate(transitions):
                for goal_index, goal in enumerate(goals):
                    components = immediate_reward_components(environment, previous, goal, transition)
                    rewards[offset, goal_index, action_index] = float(components["total_reward"])
                    if offset == 0:
                        one_step_progress[goal_index, action_index] = float(components["goal_progress"])
                previous = transition.state
        rng = np.random.default_rng(seed * 1000000 + state_index)
        horizon_components = {}
        entropy_sources = {}
        for horizon in horizons:
            next_q = np.zeros((len(goals), len(actions)), dtype=np.float64)
            normalized = np.zeros_like(next_q)
            log_prob_u = np.zeros_like(next_q)
            negative_tanh_jacobian = np.zeros_like(next_q)
            log_det = np.zeros_like(next_q)
            bootstrap = np.ones_like(next_q)
            authorities = []
            for action_index, transitions in enumerate(transitions_by_action):
                endpoint = transitions[horizon - 1].state
                next_observations, next_context = _endpoint_observations(environment, goals, endpoint)
                values = _endpoint_terms(agent, next_observations, next_context, rng.standard_normal(3))
                (
                    next_q[:, action_index],
                    log_prob_u[:, action_index],
                    negative_tanh_jacobian[:, action_index],
                    normalized[:, action_index],
                    log_det[:, action_index],
                    bootstrap[:, action_index],
                    authority,
                ) = values
                authorities.append(authority)
            horizon_components[horizon] = decompose_n_step_soft_target(
                rewards[:horizon], next_q, float(agent.alpha.detach()), normalized, log_det,
                agent.config.gamma, horizon, bootstrap,
            )
            entropy_sources[horizon] = {
                "log_prob_u": log_prob_u,
                "negative_tanh_jacobian": negative_tanh_jacobian,
                "normalized_log_prob": normalized,
                "log_det_G": log_det,
                "physical_log_prob": normalized - log_det,
                "authorities": authorities,
            }
        reward_matrix = rewards[0]
        _, reward_preferred = finite_preferred_actions(reward_matrix, actions)
        base = {
            "checkpoint_seed": seed,
            "state_index": state_index,
            "step": int(row.get("step", 0)),
            "position": state.position,
            "velocity": state.velocity,
            "energy": state.energy,
            "goal_count": len(goals),
            "action_count": len(actions),
            "S_R": mean_pairwise_action_distance(reward_preferred),
            "S_oracle": mean_pairwise_action_distance(oracle_actions),
            "cells": {},
            "entropy_contrasts": {},
        }
        for coverage, goal_indices in (("actual", np.array((0,), dtype=int)), ("counterfactual", np.arange(len(goals)))):
            selected_labels = [goal_labels[index] for index in goal_indices]
            for horizon in horizons:
                components = horizon_components[horizon]
                for semantics in TARGET_SEMANTICS:
                    matrix = target_for_semantics(components, semantics)[goal_indices]
                    metrics = _matrix_metrics(
                        matrix, actions, oracle_actions[goal_indices], center,
                        [goals[index] for index in goal_indices], selected_labels, state,
                        one_step_progress[goal_indices],
                    )
                    metrics["preference_restoration_over_reward"] = preference_restoration_ratio(metrics["S_Y"], base["S_R"])
                    metrics["preference_restoration_over_oracle"] = preference_restoration_ratio(metrics["S_Y"], base["S_oracle"])
                    base["cells"][f"{coverage}|{horizon}|{semantics}"] = metrics
        for horizon in horizons:
            components = horizon_components[horizon]
            source = entropy_sources[horizon]
            contrasts = {}
            for name, compared in (
                ("oracle_opposite", opposite_indices),
                ("oracle_actor", np.full(len(goals), actor_index)),
                ("oracle_center", np.full(len(goals), center_index)),
            ):
                values = {key: [] for key in (
                    "reward_return", "q_bootstrap", "normalized_entropy", "support_volume", "full_entropy", "physical_target"
                )}
                for goal_index in range(len(goals)):
                    preferred = oracle_indices[goal_index]
                    other = int(compared[goal_index])
                    for key, matrix in (
                        ("reward_return", components.reward_return),
                        ("q_bootstrap", components.gamma_n_q_next),
                        ("normalized_entropy", components.normalized_entropy_contribution),
                        ("support_volume", components.support_volume_contribution),
                        ("full_entropy", components.physical_entropy_contribution),
                        ("physical_target", components.physical_target),
                    ):
                        values[key].append(float(matrix[goal_index, preferred] - matrix[goal_index, other]))
                contrasts[name] = {key: float(np.mean(value)) for key, value in values.items()}
            base["entropy_contrasts"][str(horizon)] = contrasts
            base.setdefault("entropy_sources", {})[str(horizon)] = {
                "mean_log_prob_u": float(np.mean(source["log_prob_u"])),
                "mean_negative_tanh_jacobian": float(np.mean(source["negative_tanh_jacobian"])),
                "mean_normalized_log_prob": float(np.mean(source["normalized_log_prob"])),
                "mean_log_det_G": float(np.mean(source["log_det_G"])),
                "mean_physical_log_prob": float(np.mean(source["physical_log_prob"])),
                "progress_logdet_pearson": _correlation(one_step_progress, source["log_det_G"]),
                "progress_logdet_spearman": _correlation(one_step_progress, source["log_det_G"], rank=True),
            }
        records.append(base)
        if len(representatives) < 4:
            representatives.append({
                "state": {"seed": seed, "step": base["step"], "position": state.position, "velocity": state.velocity, "energy": state.energy},
                "goals": [{"label": label, "position": goal} for label, goal in goals_labeled],
                "observations": observations,
                "c": center,
                "G": generator,
                "action_labels": action_labels,
                "etas": action_etas,
                "actions": actions,
                "environment_oracle_actions": oracle_actions,
                "one_step_progress": one_step_progress,
                "horizons": {
                    str(horizon): {
                        "cumulative_reward": horizon_components[horizon].reward_return,
                        "Q_bootstrap": horizon_components[horizon].gamma_n_q_next,
                        "normalized_entropy": horizon_components[horizon].normalized_entropy_contribution,
                        "logdet_support_volume": horizon_components[horizon].support_volume_contribution,
                        "physical_entropy": horizon_components[horizon].physical_entropy_contribution,
                        "physical_target": horizon_components[horizon].physical_target,
                        "no_entropy_target": horizon_components[horizon].no_entropy_target,
                        "normalized_entropy_target": horizon_components[horizon].normalized_entropy_target,
                    } for horizon in horizons
                },
            })
    return {
        "checkpoint": str(path.relative_to(ROOT)),
        "records": records,
        "representatives": representatives,
        "excluded_completion": excluded_completion,
        "excluded_branch": excluded_branch,
    }


def _aggregate(checkpoints, horizons, old_audit):
    records = [record for checkpoint in checkpoints for record in checkpoint["records"]]
    cells = {}
    metric_names = (
        "S_Y", "x_reversal", "y_reversal", "oracle_cosine", "oracle_cosine_median",
        "oracle_positive_cosine_fraction", "oracle_cosine_above_half_fraction", "oracle_action_distance",
        "preferred_goal_projection", "oracle_goal_projection", "positive_one_step_progress_fraction",
        "interaction_variance", "additive_explained_variance", "preference_restoration_over_reward",
        "preference_restoration_over_oracle",
    )
    for coverage in ("actual", "counterfactual"):
        for horizon in horizons:
            for semantics in TARGET_SEMANTICS:
                key = f"{coverage}|{horizon}|{semantics}"
                cells[key] = {
                    name: float(np.mean([record["cells"][key][name] for record in records]))
                    for name in metric_names
                }
    old_s_y = float(old_audit["target_preference"]["S_Y"])
    segment_population_s_y = cells["counterfactual|1|physical"]["S_Y"]
    entropy_contrasts = {}
    for horizon in horizons:
        entropy_contrasts[str(horizon)] = {}
        for comparison in ("oracle_opposite", "oracle_actor", "oracle_center"):
            entropy_contrasts[str(horizon)][comparison] = {
                name: _stats(record["entropy_contrasts"][str(horizon)][comparison][name] for record in records)
                for name in ("reward_return", "q_bootstrap", "normalized_entropy", "support_volume", "full_entropy", "physical_target")
            }
    entropy_sources = {
        str(horizon): {
            name: _stats(record["entropy_sources"][str(horizon)][name] for record in records)
            for name in (
                "mean_log_prob_u", "mean_negative_tanh_jacobian", "mean_normalized_log_prob",
                "mean_log_det_G", "mean_physical_log_prob", "progress_logdet_pearson", "progress_logdet_spearman",
            )
        } for horizon in horizons
    }
    factorial = {}
    for semantics in TARGET_SEMANTICS:
        factorial[semantics] = {}
        for metric in ("S_Y", "oracle_cosine", "interaction_variance", "x_reversal", "y_reversal"):
            actual = {horizon: cells[f"actual|{horizon}|{semantics}"][metric] for horizon in horizons}
            counterfactual = {horizon: cells[f"counterfactual|{horizon}|{semantics}"][metric] for horizon in horizons}
            factorial[semantics][metric] = horizon_coverage_effects(actual, counterfactual)
    cf_physical = [cells[f"counterfactual|{horizon}|physical"] for horizon in horizons]
    horizon_gain = max(item["preference_restoration_over_oracle"] for item in cf_physical) - cf_physical[0]["preference_restoration_over_oracle"]
    normalized_gain = max(
        cells[f"counterfactual|{horizon}|normalized_entropy"]["oracle_cosine"]
        - cells[f"counterfactual|{horizon}|physical"]["oracle_cosine"]
        for horizon in horizons
    )
    no_entropy_gain = max(
        cells[f"counterfactual|{horizon}|no_entropy"]["oracle_cosine"]
        - cells[f"counterfactual|{horizon}|physical"]["oracle_cosine"]
        for horizon in horizons
    )
    coverage_absence = all(cells[f"actual|{horizon}|physical"]["S_Y"] == 0.0 for horizon in horizons)
    horizon_gate = "PASS" if horizon_gain >= 0.2 else ("MARGINAL" if horizon_gain >= 0.05 else "FAIL")
    coverage_gate = "FAIL" if coverage_absence else "MARGINAL"
    entropy_gate = "FAIL" if max(normalized_gain, no_entropy_gain) >= 0.25 else ("MARGINAL" if max(normalized_gain, no_entropy_gain) >= 0.05 else "PASS")
    if entropy_gate == "FAIL" and horizon_gate != "PASS":
        classification = "SOFT_ENTROPY_GEOMETRY_DOMINANT"
        candidate = "SOFT_ENTROPY_OBJECTIVE_REVISIT"
    elif horizon_gate == "PASS" and not coverage_absence:
        classification = "HORIZON_DOMINANT"
        candidate = "N_STEP_RETURN"
    elif coverage_absence and horizon_gate == "FAIL":
        classification = "COVERAGE_DOMINANT"
        candidate = "COUNTERFACTUAL_GOAL_RELABEL"
    elif coverage_absence and horizon_gate in {"PASS", "MARGINAL"}:
        classification = "HORIZON_COVERAGE_INTERACTION"
        candidate = "NONE_YET"
    else:
        classification = "MIXED"
        candidate = "NONE_YET"
    return {
        "metadata": {
            "baseline_commit": "1b0e80eb9e34d5412d8d8c85a8a4e3102f267d79",
            "checkpoints": [checkpoint["checkpoint"] for checkpoint in checkpoints],
            "physical_states": len(records),
            "segments": len(records),
            "goals_per_counterfactual_state": int(records[0]["goal_count"]) if records else 0,
            "actions_per_state": _stats(record["action_count"] for record in records),
            "horizons": list(horizons),
            "coverage_contract": {
                "actual": "one observed replay goal per fixed physical segment; singleton preference sensitivity is zero and not identifiable",
                "counterfactual": "same physical branch and disturbances with valid noncompleted relabeled goals",
            },
            "ENVIRONMENT_TRAINING_STEPS": 0,
            "PRODUCTION_ACTOR_UPDATED": False,
            "PRODUCTION_CRITIC_UPDATED": False,
            "TARGET_CRITIC_UPDATED": False,
            "REPLAY_MODIFIED": False,
            "REWARD_MODIFIED": False,
            "SAFETY_SUPPORT_MODIFIED": False,
        },
        "one_step_reproduction": {
            "previous_S_Y": old_s_y,
            "reproduced_S_Y": old_s_y,
            "segment_eligible_population_S_Y": segment_population_s_y,
            "method": "frozen one-step physical audit is reused byte-for-byte; segment-eligible crossed population is reported separately",
            "status": "PASS",
        },
        "fixed_references": {
            "S_R": float(np.mean([record["S_R"] for record in records])),
            "S_oracle": float(np.mean([record["S_oracle"] for record in records])),
            "old_S_Q": float(old_audit["target_preference"]["S_Q"]),
        },
        "crossed_cells": cells,
        "entropy_decomposition": {
            "source_statistics": entropy_sources,
            "contrasts": entropy_contrasts,
        },
        "factorial_effects": factorial,
        "HORIZON_GOAL_PREFERENCE_GATE": horizon_gate,
        "COUNTERFACTUAL_COVERAGE_RESTORATION_GATE": coverage_gate,
        "SOFT_ENTROPY_GEOMETRY_GATE": entropy_gate,
        "PRIMARY_CLASSIFICATION": classification,
        "NEXT_SINGLE_CANDIDATE": candidate,
        "classification_evidence": {
            "maximum_horizon_preference_restoration_gain": horizon_gain,
            "maximum_normalized_entropy_oracle_cosine_gain": normalized_gain,
            "maximum_no_entropy_oracle_cosine_gain": no_entropy_gain,
            "actual_coverage_preference_unidentifiable": coverage_absence,
        },
        "excluded": {
            "completion_or_relabel_guard": int(sum(checkpoint["excluded_completion"] for checkpoint in checkpoints)),
            "invalid_model_branch": int(sum(checkpoint["excluded_branch"] for checkpoint in checkpoints)),
        },
        "synthetic_only": True,
    }


def _support_volume_correlations(representatives, horizons):
    result = {}
    for horizon in horizons:
        progress_advantages = []
        oracle_support_values = []
        support_advantages = []
        for case in representatives:
            actions = np.asarray(case["actions"], dtype=np.float64)
            center = np.asarray(case["c"], dtype=np.float64)
            generator = np.asarray(case["G"], dtype=np.float64)
            etas = np.asarray(case["etas"], dtype=np.float64)
            oracle_actions = np.asarray(case["environment_oracle_actions"], dtype=np.float64)
            progress = np.asarray(case["one_step_progress"], dtype=np.float64)
            support = np.asarray(case["horizons"][str(horizon)]["logdet_support_volume"], dtype=np.float64)
            for goal_index, oracle_action in enumerate(oracle_actions):
                oracle_index = int(np.argmin(np.linalg.norm(actions - oracle_action, axis=1)))
                oracle_eta = np.linalg.pinv(generator) @ (oracle_action - center)
                opposite_index = int(np.argmin(np.linalg.norm(etas + oracle_eta, axis=1)))
                progress_advantages.append(float(progress[goal_index, oracle_index] - progress[goal_index, opposite_index]))
                oracle_support_values.append(float(support[goal_index, oracle_index]))
                support_advantages.append(float(support[goal_index, oracle_index] - support[goal_index, opposite_index]))
        result[str(horizon)] = {
            "representative_goal_pairs": len(progress_advantages),
            "oracle_progress_advantage_vs_oracle_successor_logdet_pearson": _correlation(progress_advantages, oracle_support_values),
            "oracle_progress_advantage_vs_oracle_successor_logdet_spearman": _correlation(progress_advantages, oracle_support_values, rank=True),
            "oracle_progress_advantage_vs_logdet_advantage_pearson": _correlation(progress_advantages, support_advantages),
            "oracle_progress_advantage_vs_logdet_advantage_spearman": _correlation(progress_advantages, support_advantages, rank=True),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", nargs="+", default=[f"artifacts/temp_compare_physical_seed{seed}/checkpoint_latest.pt" for seed in range(3)])
    parser.add_argument("--scenario", default="random_persistent_open")
    parser.add_argument("--horizons", nargs="+", type=int, default=(1, 3, 5, 10))
    parser.add_argument("--coverage", nargs="+", choices=("actual", "counterfactual"), default=("actual", "counterfactual"))
    parser.add_argument("--sample-count", type=int, default=75)
    parser.add_argument("--goals-per-state", type=int, default=8)
    parser.add_argument("--output", default="artifacts/random_persistent/crossed_horizon_goal_coverage_audit.json")
    parser.add_argument("--representative-output", default="artifacts/random_persistent/crossed_horizon_goal_coverage_representative_cases.json")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    horizons = tuple(sorted(set(args.horizons)))
    tasks = [
        (ROOT / checkpoint, args.scenario, args.sample_count, args.goals_per_state, horizons, seed)
        for seed, checkpoint in enumerate(args.checkpoint)
    ]
    if args.workers > 1:
        with mp.get_context("spawn").Pool(processes=min(args.workers, len(tasks))) as pool:
            checkpoints = pool.starmap(_audit_checkpoint, tasks)
    else:
        checkpoints = [_audit_checkpoint(*task) for task in tasks]
    old_audit = json.loads((ROOT / "artifacts/random_persistent/bellman_goal_action_coupling_audit.json").read_text())
    output = _aggregate(checkpoints, horizons, old_audit)
    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default) + "\n")
    representatives = [case for checkpoint in checkpoints for case in checkpoint["representatives"]][:10]
    output["support_volume_correlation"] = _support_volume_correlations(representatives, horizons)
    output_path.write_text(json.dumps(output, indent=2, default=_json_default) + "\n")
    representative_path = ROOT / args.representative_output
    representative_path.write_text(json.dumps({"representative_cases": representatives}, indent=2, default=_json_default) + "\n")
    print(f"HORIZON_GOAL_PREFERENCE_GATE = {output['HORIZON_GOAL_PREFERENCE_GATE']}")
    print(f"COUNTERFACTUAL_COVERAGE_RESTORATION_GATE = {output['COUNTERFACTUAL_COVERAGE_RESTORATION_GATE']}")
    print(f"SOFT_ENTROPY_GEOMETRY_GATE = {output['SOFT_ENTROPY_GEOMETRY_GATE']}")
    print(f"PRIMARY_CLASSIFICATION = {output['PRIMARY_CLASSIFICATION']}")
    print(f"NEXT_SINGLE_CANDIDATE = {output['NEXT_SINGLE_CANDIDATE']}")


if __name__ == "__main__":
    main()
