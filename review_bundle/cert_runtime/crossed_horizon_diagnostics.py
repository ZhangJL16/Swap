from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch

from .generator_sac import QNetwork


EPSILON = 1e-12
TARGET_SEMANTICS = ("physical", "no_entropy", "normalized_entropy")


@dataclass(frozen=True, slots=True)
class NStepTargetComponents:
    reward_return: np.ndarray
    gamma_n_q_next: np.ndarray
    normalized_entropy_contribution: np.ndarray
    support_volume_contribution: np.ndarray
    physical_entropy_contribution: np.ndarray
    physical_target: np.ndarray
    no_entropy_target: np.ndarray
    normalized_entropy_target: np.ndarray


def valid_n_step_segment(rows: Sequence[Mapping[str, object]], start: int, horizon: int) -> bool:
    if horizon < 1 or start < 0 or start + horizon > len(rows):
        return False
    selected = rows[start:start + horizon]
    first = selected[0]
    episode_id = first.get("episode_id")
    task_id = first.get("task_id")
    expected_step = int(first.get("episode_step", -1))
    for offset, row in enumerate(selected):
        if row.get("episode_id") != episode_id or row.get("task_id") != task_id:
            return False
        if int(row.get("episode_step", -2)) != expected_step + offset:
            return False
        if row.get("task_completed_now") or row.get("terminated") or row.get("truncated"):
            return False
        if row.get("goal") is None:
            return False
    return True


def discounted_reward_return(rewards: np.ndarray, gamma: float) -> np.ndarray:
    values = np.asarray(rewards, dtype=np.float64)
    if values.ndim < 1 or values.shape[0] < 1:
        raise ValueError("rewards must have a nonempty horizon dimension")
    discounts = np.power(float(gamma), np.arange(values.shape[0], dtype=np.float64))
    return np.tensordot(discounts, values, axes=(0, 0))


def decompose_n_step_soft_target(
    rewards: np.ndarray,
    next_q: np.ndarray,
    alpha: float,
    normalized_log_prob: np.ndarray,
    log_det_generator: np.ndarray,
    gamma: float,
    horizon: int,
    bootstrap: np.ndarray | float = 1.0,
) -> NStepTargetComponents:
    if horizon < 1:
        raise ValueError("horizon must be positive")
    reward_return = discounted_reward_return(rewards, gamma)
    factor = float(gamma) ** int(horizon) * np.asarray(bootstrap, dtype=np.float64)
    q_term = factor * np.asarray(next_q, dtype=np.float64)
    normalized_entropy = -factor * float(alpha) * np.asarray(normalized_log_prob, dtype=np.float64)
    support_volume = factor * float(alpha) * np.asarray(log_det_generator, dtype=np.float64)
    physical_entropy = normalized_entropy + support_volume
    no_entropy = reward_return + q_term
    normalized_target = no_entropy + normalized_entropy
    physical_target = normalized_target + support_volume
    return NStepTargetComponents(
        reward_return=reward_return,
        gamma_n_q_next=q_term,
        normalized_entropy_contribution=normalized_entropy,
        support_volume_contribution=support_volume,
        physical_entropy_contribution=physical_entropy,
        physical_target=physical_target,
        no_entropy_target=no_entropy,
        normalized_entropy_target=normalized_target,
    )


def target_for_semantics(components: NStepTargetComponents, semantics: str) -> np.ndarray:
    if semantics == "physical":
        return components.physical_target
    if semantics == "no_entropy":
        return components.no_entropy_target
    if semantics == "normalized_entropy":
        return components.normalized_entropy_target
    raise ValueError(f"unsupported target semantics: {semantics}")


def relabeled_goal_not_completed(
    goal: np.ndarray,
    successor_positions: np.ndarray,
    goal_radius: float,
) -> bool:
    positions = np.asarray(successor_positions, dtype=np.float64)
    selected_goal = np.asarray(goal, dtype=np.float64)
    return bool(np.all(np.linalg.norm(positions - selected_goal, axis=-1) > float(goal_radius)))


def preference_restoration_ratio(preference_sensitivity: float, reference_sensitivity: float) -> float:
    return float(preference_sensitivity) / (float(reference_sensitivity) + EPSILON)


def horizon_coverage_effects(
    actual: Mapping[int, float],
    counterfactual: Mapping[int, float],
    baseline_horizon: int = 1,
) -> dict[int, dict[str, float]]:
    if baseline_horizon not in actual or baseline_horizon not in counterfactual:
        raise ValueError("baseline horizon is missing")
    result = {}
    for horizon in sorted(set(actual) & set(counterfactual)):
        horizon_effect = float(actual[horizon] - actual[baseline_horizon])
        coverage_effect = float(counterfactual[horizon] - actual[horizon])
        interaction = float(
            (counterfactual[horizon] - counterfactual[baseline_horizon])
            - (actual[horizon] - actual[baseline_horizon])
        )
        result[horizon] = {
            "horizon_main_effect": horizon_effect,
            "coverage_main_effect": coverage_effect,
            "horizon_coverage_interaction": interaction,
        }
    return result


def physical_entropy_identity(
    normalized_log_prob: np.ndarray,
    log_det_generator: np.ndarray,
) -> np.ndarray:
    return np.asarray(normalized_log_prob, dtype=np.float64) - np.asarray(log_det_generator, dtype=np.float64)


def fit_disposable_critic(
    observations: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_dim: int,
    steps: int = 500,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> tuple[QNetwork, dict[str, float]]:
    observation_values = np.asarray(observations, dtype=np.float32).copy()
    action_values = np.asarray(actions, dtype=np.float32).copy()
    target_values = np.asarray(targets, dtype=np.float32).copy()
    if observation_values.ndim != 2 or action_values.shape != (len(observation_values), 3):
        raise ValueError("disposable critic inputs have incompatible shapes")
    if target_values.shape != (len(observation_values),):
        raise ValueError("disposable critic targets have incompatible shape")
    torch.manual_seed(seed)
    network = QNetwork(observation_values.shape[1], hidden_dim)
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    observations_tensor = torch.as_tensor(observation_values)
    actions_tensor = torch.as_tensor(action_values)
    targets_tensor = torch.as_tensor(target_values)
    with torch.no_grad():
        initial_loss = torch.mean((network(observations_tensor, actions_tensor) - targets_tensor) ** 2)
    network.train()
    for _ in range(int(steps)):
        predicted = network(observations_tensor, actions_tensor)
        loss = torch.mean((predicted - targets_tensor) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    network.eval()
    with torch.no_grad():
        final_loss = torch.mean((network(observations_tensor, actions_tensor) - targets_tensor) ** 2)
    return network, {
        "initial_mse": float(initial_loss),
        "final_mse": float(final_loss),
        "steps": int(steps),
    }
