from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from envs.certified_uav.dynamics import integrate_double_integrator
from envs.certified_uav.state import UAVPhysicalState


EPSILON = 1e-12


@dataclass(frozen=True, slots=True)
class NominalPhysicalTransition:
    state: UAVPhysicalState
    energy_cost: float
    collision: bool
    velocity_violation: bool


def nominal_physical_transition(environment, state: UAVPhysicalState, action: np.ndarray) -> NominalPhysicalTransition:
    selected_action = np.asarray(action, dtype=np.float64)
    position, velocity = integrate_double_integrator(
        state.position,
        state.velocity,
        selected_action,
        environment.plant.config.dt,
    )
    energy_cost = float(environment.plant.energy_model.realized_cost(
        state,
        selected_action,
        environment.plant.config.dt,
    ))
    successor = UAVPhysicalState(
        position,
        velocity,
        state.energy - energy_cost,
        state.timestamp + environment.plant.config.dt,
    )
    return NominalPhysicalTransition(
        state=successor,
        energy_cost=energy_cost,
        collision=bool(environment.plant.world.swept_collision(
            state.position,
            position,
            environment.plant.config.body_radius,
        )),
        velocity_violation=bool(np.any(np.abs(velocity) > environment.plant.config.v_max + 1e-12)),
    )


def immediate_reward_components(
    environment,
    state: UAVPhysicalState,
    goal: np.ndarray,
    transition: NominalPhysicalTransition,
    *,
    backup_started_now: bool = False,
    charging: bool = False,
) -> dict[str, float | bool]:
    selected_goal = np.asarray(goal, dtype=np.float64)
    distance_before = float(np.linalg.norm(selected_goal - state.position))
    distance_after = float(np.linalg.norm(selected_goal - transition.state.position))
    progress = distance_before - distance_after
    completed = distance_after <= environment.task_env.manager.goal_radius
    config = environment.task_env.reward_config
    components = {
        "goal_progress_reward": float(config.goal_progress_weight * progress),
        "task_completion_reward": float(config.task_completion_reward * completed),
        "elapsed_time_cost": float(-config.elapsed_time_cost),
        "energy_cost": float(-config.flight_energy_cost * transition.energy_cost),
        "backup_intervention_event_cost": float(-config.backup_intervention_cost * backup_started_now),
        "charging_dwell_cost": float(-config.charging_dwell_cost * charging),
    }
    return components | {
        "total_reward": float(sum(components.values())),
        "goal_progress": progress,
        "task_completed": bool(completed),
    }


def bellman_target_decomposition(
    reward: np.ndarray,
    next_q: np.ndarray,
    entropy: np.ndarray,
    gamma: float,
    bootstrap: np.ndarray | float = 1.0,
) -> dict[str, np.ndarray]:
    reward_array = np.asarray(reward, dtype=np.float64)
    next_q_array = np.asarray(next_q, dtype=np.float64)
    entropy_array = np.asarray(entropy, dtype=np.float64)
    bootstrap_array = np.asarray(bootstrap, dtype=np.float64)
    gamma_next_q = gamma * bootstrap_array * next_q_array
    negative_gamma_entropy = -gamma * bootstrap_array * entropy_array
    target = reward_array + gamma_next_q + negative_gamma_entropy
    return {
        "reward": reward_array,
        "next_q": next_q_array,
        "entropy": entropy_array,
        "gamma_next_q": gamma_next_q,
        "negative_gamma_entropy": negative_gamma_entropy,
        "target": target,
    }


def contrast_decomposition(
    decomposition: Mapping[str, np.ndarray],
    preferred_index: int,
    comparison_index: int,
) -> dict[str, float]:
    result = {}
    for name in ("reward", "gamma_next_q", "negative_gamma_entropy", "target"):
        values = np.asarray(decomposition[name], dtype=np.float64)
        result[name] = float(values[preferred_index] - values[comparison_index])
    delta_reward = result["reward"]
    bootstrap_contrast = result["gamma_next_q"] + result["negative_gamma_entropy"]
    result["bellman_contrast_preservation_ratio"] = abs(result["target"]) / (abs(delta_reward) + EPSILON)
    result["bootstrap_dominance_ratio"] = abs(bootstrap_contrast) / (abs(delta_reward) + EPSILON)
    gamma_next_q = np.asarray(decomposition["gamma_next_q"], dtype=np.float64)
    negative_gamma_entropy = np.asarray(decomposition["negative_gamma_entropy"], dtype=np.float64)
    result["bootstrap_base_scale_ratio"] = abs(
        gamma_next_q[preferred_index] + negative_gamma_entropy[preferred_index]
    ) / (abs(delta_reward) + EPSILON)
    return result


def goal_action_interaction(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("goal-action matrix must be two-dimensional")
    return values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()


def additive_decomposition_metrics(matrix: np.ndarray) -> dict[str, float | np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    interaction = goal_action_interaction(values)
    total_variance = float(np.var(values))
    interaction_variance = float(np.var(interaction))
    explained = 1.0 if total_variance <= EPSILON else max(0.0, 1.0 - interaction_variance / total_variance)
    fitted = values - interaction
    return {
        "additive_explained_variance": float(explained),
        "interaction_variance": interaction_variance,
        "total_variance": total_variance,
        "interaction_residual": interaction,
        "additive_fit": fitted,
    }


def finite_preferred_actions(matrix: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    candidates = np.asarray(actions, dtype=np.float64)
    if values.ndim != 2 or candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("invalid goal-action preference inputs")
    if values.shape[1] != candidates.shape[0]:
        raise ValueError("matrix action dimension does not match candidate actions")
    indices = np.argmax(values, axis=1)
    return indices, candidates[indices]


def mean_pairwise_action_distance(actions: Sequence[np.ndarray]) -> float:
    values = np.asarray(actions, dtype=np.float64)
    if values.shape[0] < 2:
        return 0.0
    distances = [
        float(np.linalg.norm(values[first] - values[second]))
        for first in range(values.shape[0])
        for second in range(first + 1, values.shape[0])
    ]
    return float(np.mean(distances))


def matrix_diagonal_preference(matrix: np.ndarray) -> tuple[float, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("cross-goal matrix must be square")
    diagonal = np.diag(values)
    alternatives = np.asarray([
        np.max(np.delete(values[row], row)) if values.shape[1] > 1 else values[row, row]
        for row in range(values.shape[0])
    ])
    return float(np.mean(diagonal >= alternatives)), float(np.mean(diagonal - alternatives))
