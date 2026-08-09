from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from .generator_sac import GeneratorSAC


EPSILON = 1e-12
GOAL_DERIVED_OBSERVATION_FIELDS = frozenset(("goal_delta", "distance_to_goal"))


def minimum_q(agent: GeneratorSAC, observations: Tensor, actions: Tensor) -> Tensor:
    return torch.minimum(agent.critic_1(observations, actions), agent.critic_2(observations, actions))


def changed_observation_fields(
    original: np.ndarray,
    counterfactual: np.ndarray,
    layout: Mapping[str, slice],
    tolerance: float = 0.0,
) -> tuple[str, ...]:
    first = np.asarray(original)
    second = np.asarray(counterfactual)
    changed = []
    for name, selected in layout.items():
        if not np.allclose(first[selected], second[selected], atol=tolerance, rtol=0.0):
            changed.append(name)
    return tuple(changed)


def certificate_invariance_snapshot(context: Mapping[str, object]) -> dict[str, object]:
    keys = (
        "recoverable_set_member",
        "rl_authority_set_member",
        "recovery_energy_required",
        "energy_margin",
        "recovery_hash",
        "recoverable_set_hash",
        "recoverability_action_hash",
        "continuation_action_verified",
        "continuation_target_cell_id",
        "atlas_hash",
        "certificate_epoch",
        "certificate_version",
        "geometry_version",
        "dynamics_version",
        "tracking_version",
        "energy_version",
        "terminal_version",
        "corridor_version",
        "terminal_recovery_certificate_hash",
        "zonotope_hash",
    )
    snapshot: dict[str, object] = {}
    for key in keys:
        value = context.get(key)
        if isinstance(value, np.ndarray):
            snapshot[key] = value.copy()
        elif isinstance(value, np.generic):
            snapshot[key] = value.item()
        else:
            snapshot[key] = value
    snapshot["c"] = np.asarray(context["c"], dtype=np.float64).copy()
    snapshot["G"] = np.asarray(context["G"], dtype=np.float64).copy()
    return snapshot


def certificate_snapshots_equal(
    first: Mapping[str, object],
    second: Mapping[str, object],
    tolerance: float = 1e-12,
) -> tuple[bool, tuple[str, ...]]:
    failures = []
    for key in first:
        left = first[key]
        right = second.get(key)
        if isinstance(left, np.ndarray):
            if right is None or not np.allclose(left, np.asarray(right), atol=tolerance, rtol=0.0):
                failures.append(key)
        elif isinstance(left, float):
            if right is None or not np.isclose(left, float(right), atol=tolerance, rtol=0.0):
                failures.append(key)
        elif left != right:
            failures.append(key)
    return not failures, tuple(failures)


def critic_action_gradient(
    agent: GeneratorSAC,
    observation: Tensor,
    action: Tensor,
    *,
    create_graph: bool = False,
) -> tuple[Tensor, Tensor]:
    selected_action = action.detach().clone().requires_grad_(True)
    value = minimum_q(agent, observation, selected_action)
    gradient = torch.autograd.grad(value.sum(), selected_action, create_graph=create_graph)[0]
    return value, gradient


def action_gradient_goal_jacobian(
    agent: GeneratorSAC,
    observation: Tensor,
    action: Tensor,
    goal_slice: slice,
) -> Tensor:
    selected_observation = observation.detach().clone().requires_grad_(True)
    selected_action = action.detach().clone().requires_grad_(True)
    value = minimum_q(agent, selected_observation, selected_action)
    action_gradient = torch.autograd.grad(value.sum(), selected_action, create_graph=True)[0]
    rows = []
    for axis in range(action_gradient.shape[-1]):
        derivative = None
        if action_gradient[:, axis].requires_grad:
            derivative = torch.autograd.grad(
                action_gradient[:, axis].sum(),
                selected_observation,
                retain_graph=True,
                allow_unused=True,
            )[0]
        if derivative is None:
            derivative = torch.zeros_like(selected_observation)
        rows.append(derivative[:, goal_slice])
    return torch.stack(rows, dim=1).detach()


def finite_difference_action_gradient_goal_jacobian(
    agent: GeneratorSAC,
    observation: Tensor,
    action: Tensor,
    goal_slice: slice,
    epsilon_goal: float = 1e-3,
) -> Tensor:
    width = goal_slice.stop - goal_slice.start
    columns = []
    for axis in range(width):
        positive = observation.detach().clone()
        negative = observation.detach().clone()
        index = goal_slice.start + axis
        positive[:, index] += epsilon_goal
        negative[:, index] -= epsilon_goal
        _, positive_gradient = critic_action_gradient(agent, positive, action)
        _, negative_gradient = critic_action_gradient(agent, negative, action)
        columns.append((positive_gradient - negative_gradient) / (2.0 * epsilon_goal))
    return torch.stack(columns, dim=-1).detach()


@dataclass(frozen=True, slots=True)
class CriticPreferredAction:
    eta: np.ndarray
    action: np.ndarray
    q_value: float
    source: str
    candidates_evaluated: int


def searched_critic_preferred_action(
    agent: GeneratorSAC,
    observation: Tensor,
    center: np.ndarray,
    generator: np.ndarray,
    actor_eta: np.ndarray,
    *,
    seed: int,
    gradient_steps: int = 24,
    random_starts: int = 6,
    random_evaluations: int = 48,
) -> CriticPreferredAction:
    rng = np.random.default_rng(seed)
    corners = np.asarray(tuple(product((-1.0, 1.0), repeat=3)), dtype=np.float32)
    initial = np.concatenate((
        np.asarray(actor_eta, dtype=np.float32).reshape(1, 3),
        np.zeros((1, 3), dtype=np.float32),
        corners,
        rng.uniform(-1.0, 1.0, size=(random_starts, 3)).astype(np.float32),
    ))
    centers = torch.as_tensor(center, dtype=torch.float32).reshape(1, 3)
    generators = torch.as_tensor(generator, dtype=torch.float32).reshape(1, 3, 3)
    eta = torch.as_tensor(initial, dtype=torch.float32).clone().requires_grad_(True)
    optimizer = torch.optim.Adam((eta,), lr=0.08)
    repeated_observation = observation.detach().expand(eta.shape[0], -1)
    for _ in range(gradient_steps):
        actions = centers + torch.matmul(generators, eta.unsqueeze(-1)).squeeze(-1)
        loss = -minimum_q(agent, repeated_observation, actions).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            eta.clamp_(-1.0, 1.0)
    random_candidates = rng.uniform(-1.0, 1.0, size=(random_evaluations, 3)).astype(np.float32)
    all_eta = torch.cat((
        torch.as_tensor(initial),
        eta.detach(),
        torch.as_tensor(random_candidates),
    ))
    all_actions = centers + torch.matmul(generators, all_eta.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        values = minimum_q(agent, observation.detach().expand(all_eta.shape[0], -1), all_actions)
    best_index = int(torch.argmax(values))
    source = "projected_gradient" if best_index >= initial.shape[0] and best_index < 2 * initial.shape[0] else (
        "initial_or_corner" if best_index < initial.shape[0] else "random_evaluation"
    )
    return CriticPreferredAction(
        eta=all_eta[best_index].detach().cpu().numpy().astype(np.float64),
        action=all_actions[best_index].detach().cpu().numpy().astype(np.float64),
        q_value=float(values[best_index]),
        source=source,
        candidates_evaluated=int(all_eta.shape[0]),
    )


def searched_critic_preferred_actions(
    agent: GeneratorSAC,
    observations: Tensor,
    center: np.ndarray,
    generator: np.ndarray,
    actor_eta: np.ndarray,
    *,
    seed: int,
    gradient_steps: int = 24,
    random_starts: int = 6,
    random_evaluations: int = 48,
) -> tuple[CriticPreferredAction, ...]:
    rng = np.random.default_rng(seed)
    goal_count = observations.shape[0]
    corners = np.asarray(tuple(product((-1.0, 1.0), repeat=3)), dtype=np.float32)
    fixed = np.concatenate((np.zeros((1, 3), dtype=np.float32), corners))
    initial = []
    for goal_index in range(goal_count):
        initial.append(np.concatenate((
            np.asarray(actor_eta[goal_index], dtype=np.float32).reshape(1, 3),
            fixed,
            rng.uniform(-1.0, 1.0, size=(random_starts, 3)).astype(np.float32),
        )))
    initial_array = np.stack(initial)
    center_tensor = torch.as_tensor(center, dtype=torch.float32).reshape(1, 1, 3)
    generator_tensor = torch.as_tensor(generator, dtype=torch.float32).reshape(1, 1, 3, 3)
    eta = torch.as_tensor(initial_array).clone().requires_grad_(True)
    optimizer = torch.optim.Adam((eta,), lr=0.08)
    repeated_observations = observations[:, None, :].expand(-1, eta.shape[1], -1).reshape(-1, observations.shape[-1])
    for _ in range(gradient_steps):
        actions = center_tensor + torch.matmul(generator_tensor, eta.unsqueeze(-1)).squeeze(-1)
        values = minimum_q(agent, repeated_observations, actions.reshape(-1, 3)).reshape(goal_count, -1)
        loss = -values.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            eta.clamp_(-1.0, 1.0)
    random_candidates = rng.uniform(
        -1.0, 1.0, size=(goal_count, random_evaluations, 3)
    ).astype(np.float32)
    all_eta = torch.cat((
        torch.as_tensor(initial_array),
        eta.detach(),
        torch.as_tensor(random_candidates),
    ), dim=1)
    all_actions = center_tensor + torch.matmul(generator_tensor, all_eta.unsqueeze(-1)).squeeze(-1)
    all_observations = observations[:, None, :].expand(-1, all_eta.shape[1], -1).reshape(-1, observations.shape[-1])
    with torch.no_grad():
        values = minimum_q(agent, all_observations, all_actions.reshape(-1, 3)).reshape(goal_count, -1)
    results = []
    initial_count = initial_array.shape[1]
    for goal_index in range(goal_count):
        best_index = int(torch.argmax(values[goal_index]))
        source = "projected_gradient" if initial_count <= best_index < 2 * initial_count else (
            "initial_or_corner" if best_index < initial_count else "random_evaluation"
        )
        results.append(CriticPreferredAction(
            eta=all_eta[goal_index, best_index].detach().cpu().numpy().astype(np.float64),
            action=all_actions[goal_index, best_index].detach().cpu().numpy().astype(np.float64),
            q_value=float(values[goal_index, best_index]),
            source=source,
            candidates_evaluated=int(all_eta.shape[1]),
        ))
    return tuple(results)


def mean_pairwise_distance(values: Sequence[np.ndarray]) -> float:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    distances = [float(np.linalg.norm(first - second)) for first, second in combinations(arrays, 2)]
    return 0.0 if not distances else float(np.mean(distances))


def mean_pairwise_cosine(values: Sequence[np.ndarray]) -> float:
    arrays = [np.asarray(value, dtype=np.float64) for value in values]
    cosines = []
    for first, second in combinations(arrays, 2):
        denominator = max(float(np.linalg.norm(first) * np.linalg.norm(second)), EPSILON)
        cosines.append(float(first @ second / denominator))
    return 1.0 if not cosines else float(np.mean(cosines))


def residual_alignment(
    preferred_action: np.ndarray,
    oracle_action: np.ndarray,
    center: np.ndarray,
) -> float:
    preferred = np.asarray(preferred_action, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    oracle = np.asarray(oracle_action, dtype=np.float64) - np.asarray(center, dtype=np.float64)
    return float(preferred @ oracle / max(np.linalg.norm(preferred) * np.linalg.norm(oracle), EPSILON))


def opposite_goal_preference_reversal(
    positive_action: np.ndarray,
    negative_action: np.ndarray,
    positive_direction: np.ndarray,
    negative_direction: np.ndarray,
    axis: int,
) -> bool:
    positive = np.asarray(positive_action, dtype=np.float64)
    negative = np.asarray(negative_action, dtype=np.float64)
    return bool(
        positive @ np.asarray(positive_direction, dtype=np.float64) > 0.0
        and negative @ np.asarray(negative_direction, dtype=np.float64) > 0.0
        and positive[axis] * negative[axis] < 0.0
    )


def cross_goal_q_matrix(
    agent: GeneratorSAC,
    observations: Tensor,
    actions: Tensor,
) -> np.ndarray:
    goal_count = observations.shape[0]
    tiled_observations = observations[:, None, :].expand(goal_count, goal_count, -1).reshape(-1, observations.shape[-1])
    tiled_actions = actions[None, :, :].expand(goal_count, goal_count, -1).reshape(-1, actions.shape[-1])
    with torch.no_grad():
        values = minimum_q(agent, tiled_observations, tiled_actions)
    return values.reshape(goal_count, goal_count).cpu().numpy().astype(np.float64)


def environment_cross_goal_matrix(
    state,
    goals: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    dt: float,
    energy_model,
    reward_config,
) -> tuple[np.ndarray, np.ndarray]:
    reward = np.zeros((len(goals), len(actions)), dtype=np.float64)
    progress = np.zeros_like(reward)
    position = np.asarray(state.position, dtype=np.float64)
    velocity = np.asarray(state.velocity, dtype=np.float64)
    for goal_index, goal in enumerate(goals):
        selected_goal = np.asarray(goal, dtype=np.float64)
        initial_distance = float(np.linalg.norm(selected_goal - position))
        for action_index, action in enumerate(actions):
            selected_action = np.asarray(action, dtype=np.float64)
            next_position = position + velocity * dt + 0.5 * selected_action * dt * dt
            selected_progress = initial_distance - float(np.linalg.norm(selected_goal - next_position))
            energy = float(energy_model.realized_cost(state, selected_action, dt))
            progress[goal_index, action_index] = selected_progress
            reward[goal_index, action_index] = (
                reward_config.goal_progress_weight * selected_progress
                - reward_config.elapsed_time_cost
                - reward_config.flight_energy_cost * energy
            )
    return reward, progress


def diagonal_preference(matrix: np.ndarray) -> tuple[float, float]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("cross-goal matrix must be square")
    preferred = []
    advantages = []
    for index in range(values.shape[0]):
        off_diagonal = np.delete(values[index], index)
        best_other = float(np.max(off_diagonal)) if off_diagonal.size else float(values[index, index])
        preferred.append(float(values[index, index]) >= best_other - 1e-9)
        advantages.append(float(values[index, index]) - best_other)
    return float(np.mean(preferred)), float(np.mean(advantages))
