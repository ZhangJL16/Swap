from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from envs.certified_uav.dynamics import integrate_double_integrator


LATENT_LIMIT = 1.0 - 1e-6


def support_interval(center: np.ndarray, generators: np.ndarray, direction: np.ndarray) -> tuple[float, float, float]:
    center_array = np.asarray(center, dtype=np.float64)
    generator_array = np.asarray(generators, dtype=np.float64)
    direction_array = np.asarray(direction, dtype=np.float64)
    residual = float(np.sum(np.abs(generator_array.T @ direction_array)))
    projection = float(direction_array @ center_array)
    return projection - residual, projection + residual, residual


def action_from_eta(center: np.ndarray, generators: np.ndarray, eta: np.ndarray) -> np.ndarray:
    selected = np.asarray(eta, dtype=np.float64)
    if selected.shape != (3,) or np.any(selected < -1.0) or np.any(selected > 1.0):
        raise ValueError("eta must belong to [-1, 1]^3")
    return np.asarray(center, dtype=np.float64) + np.asarray(generators, dtype=np.float64) @ selected


def latent_from_eta(eta: np.ndarray) -> np.ndarray:
    return np.arctanh(np.clip(np.asarray(eta, dtype=np.float64), -LATENT_LIMIT, LATENT_LIMIT))


@dataclass(frozen=True, slots=True)
class SupportAuthorityMetrics:
    center_norm: float
    sigma_min: float
    sigma_max: float
    operator_norm: float
    volume: float
    row_center_to_residual: tuple[float, float, float]
    minimum_goal_projection: float
    maximum_goal_projection: float
    rho_goal: float
    bidirectional_goal_authority: bool
    positive_goal_projection: bool
    center_reversal_possible: bool
    bidirectional_x: bool
    bidirectional_y: bool
    anti_center_projection: float


def support_authority_metrics(center: np.ndarray, generators: np.ndarray, goal_direction: np.ndarray) -> SupportAuthorityMetrics:
    center_array = np.asarray(center, dtype=np.float64)
    generator_array = np.asarray(generators, dtype=np.float64)
    direction = np.asarray(goal_direction, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("goal direction must be nonzero")
    direction = direction / norm
    singular_values = np.linalg.svd(generator_array, compute_uv=False)
    minimum, maximum, residual = support_interval(center_array, generator_array, direction)
    center_projection = float(direction @ center_array)
    if center_projection > 1e-12:
        reversal = minimum < 0.0
    elif center_projection < -1e-12:
        reversal = maximum > 0.0
    else:
        reversal = minimum < 0.0 < maximum
    row_capacity = np.sum(np.abs(generator_array), axis=1)
    row_ratio = tuple(float(abs(center_array[index]) / max(row_capacity[index], 1e-12)) for index in range(3))
    x_min, x_max, _ = support_interval(center_array, generator_array, np.array((1.0, 0.0, 0.0)))
    y_min, y_max, _ = support_interval(center_array, generator_array, np.array((0.0, 1.0, 0.0)))
    anti_center = float(center_array @ center_array - np.sum(np.abs(generator_array.T @ center_array)))
    return SupportAuthorityMetrics(
        center_norm=float(np.linalg.norm(center_array)),
        sigma_min=float(np.min(singular_values)),
        sigma_max=float(np.max(singular_values)),
        operator_norm=float(np.linalg.norm(generator_array, ord=2)),
        volume=float((2.0 ** 3) * abs(np.linalg.det(generator_array))),
        row_center_to_residual=row_ratio,
        minimum_goal_projection=minimum,
        maximum_goal_projection=maximum,
        rho_goal=float(residual / (abs(center_projection) + 1e-12)),
        bidirectional_goal_authority=minimum < 0.0 < maximum,
        positive_goal_projection=maximum > 0.0,
        center_reversal_possible=reversal,
        bidirectional_x=x_min < 0.0 < x_max,
        bidirectional_y=y_min < 0.0 < y_max,
        anti_center_projection=anti_center,
    )


class CenterOnlyGoalController:
    def select_eta(self, state, goal: np.ndarray, center: np.ndarray, generators: np.ndarray, dt: float) -> np.ndarray:
        del state, goal, center, generators, dt
        return np.zeros(3, dtype=np.float64)


class RandomInGeneratorGoalController:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def select_eta(self, state, goal: np.ndarray, center: np.ndarray, generators: np.ndarray, dt: float) -> np.ndarray:
        del state, goal, center, generators, dt
        return self.rng.uniform(-LATENT_LIMIT, LATENT_LIMIT, 3)


class BestInGeneratorGoalOracle:
    def select_eta(self, state, goal: np.ndarray, center: np.ndarray, generators: np.ndarray, dt: float) -> np.ndarray:
        candidates = [np.zeros(3, dtype=np.float64)]
        candidates.extend(np.asarray(values, dtype=np.float64) * LATENT_LIMIT for values in product((-1.0, 1.0), repeat=3))
        best_eta = candidates[0]
        best_distance = float("inf")
        for eta in candidates:
            action = action_from_eta(center, generators, eta)
            next_position, _ = integrate_double_integrator(state.position, state.velocity, action, dt)
            distance = float(np.linalg.norm(next_position - np.asarray(goal, dtype=np.float64)))
            if distance < best_distance:
                best_distance = distance
                best_eta = eta
        return best_eta.copy()


class MaxOpposeCenterOracle:
    def select_eta(self, state, goal: np.ndarray, center: np.ndarray, generators: np.ndarray, dt: float) -> np.ndarray:
        del state, goal, dt
        coefficients = np.asarray(generators, dtype=np.float64).T @ np.asarray(center, dtype=np.float64)
        return -np.sign(coefficients) * LATENT_LIMIT
