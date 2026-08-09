from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


EPSILON = 1e-12


def unit_directions(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPSILON)


def physical_feature(row: Mapping[str, object], world_size: np.ndarray, v_max: np.ndarray, energy_scale: float) -> np.ndarray:
    return np.concatenate((
        np.asarray(row["position"], dtype=np.float64) / np.asarray(world_size, dtype=np.float64),
        np.asarray(row["velocity"], dtype=np.float64) / np.asarray(v_max, dtype=np.float64),
        np.asarray((float(row["energy"]) / float(energy_scale),), dtype=np.float64),
    ))


def physical_neighborhood_key(row: Mapping[str, object], decimals: int = 2) -> tuple[object, ...]:
    return (
        row.get("kappa_cell_id"),
        row.get("persistent_mode"),
        *np.round(np.asarray(row["position"], dtype=np.float64), decimals).tolist(),
        *np.round(np.asarray(row["velocity"], dtype=np.float64), decimals).tolist(),
        round(float(row["energy_margin"]), decimals),
    )


def knn_physical_neighborhoods(
    rows: Sequence[Mapping[str, object]],
    features: np.ndarray,
    *,
    neighbors: int = 32,
    anchors_per_cell: int = 8,
    minimum_size: int = 8,
) -> list[np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    if values.shape[0] != len(rows):
        raise ValueError("feature count must match rows")
    cells: dict[tuple[object, object], list[int]] = {}
    for index, row in enumerate(rows):
        cells.setdefault((row.get("kappa_cell_id"), row.get("persistent_mode")), []).append(index)
    result: list[np.ndarray] = []
    seen: set[tuple[int, ...]] = set()
    for indices in cells.values():
        if len(indices) < minimum_size:
            continue
        pool = np.asarray(indices, dtype=int)
        anchors = pool[np.linspace(0, len(pool) - 1, min(anchors_per_cell, len(pool)), dtype=int)]
        for anchor in anchors:
            distances = np.linalg.norm(values[pool] - values[anchor], axis=1)
            selected = np.sort(pool[np.argsort(distances)[: min(neighbors, len(pool))]])
            key = tuple(int(value) for value in selected)
            if len(selected) >= minimum_size and key not in seen:
                seen.add(key)
                result.append(selected)
    return result


def goal_direction_diversity(goal_directions: np.ndarray, bins: int = 8) -> dict[str, float | int | bool]:
    directions = unit_directions(goal_directions)
    if directions.shape[0] == 0:
        return {"distinct_directions": 0, "angular_spread_degrees": 0.0, "direction_entropy": 0.0, "near_opposite": False}
    angles = np.arctan2(directions[:, 1], directions[:, 0])
    histogram, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
    probabilities = histogram[histogram > 0] / max(int(histogram.sum()), 1)
    entropy = float(-np.sum(probabilities * np.log(probabilities)) / np.log(bins)) if probabilities.size else 0.0
    cosine = np.clip(directions @ directions.T, -1.0, 1.0)
    spread = float(np.degrees(np.max(np.arccos(cosine))))
    return {
        "distinct_directions": int(np.count_nonzero(histogram)),
        "angular_spread_degrees": spread,
        "direction_entropy": entropy,
        "near_opposite": bool(spread > 120.0),
    }


def action_covariance_metrics(actions: np.ndarray, tolerance: float = 1e-8) -> dict[str, float | int | np.ndarray]:
    values = np.asarray(actions, dtype=np.float64)
    covariance = np.cov(values, rowvar=False, bias=True) if values.shape[0] > 1 else np.zeros((3, 3))
    eigenvalues = np.linalg.eigvalsh(covariance)
    rank = int(np.count_nonzero(eigenvalues > tolerance))
    return {
        "rank": rank,
        "eigenvalues": eigenvalues,
        "trace": float(np.trace(covariance)),
    }


def goal_action_interaction_features(goal_directions: np.ndarray, action_coordinates: np.ndarray) -> np.ndarray:
    goals = unit_directions(goal_directions)
    actions = np.asarray(action_coordinates, dtype=np.float64)
    if goals.shape != actions.shape or goals.ndim != 2 or goals.shape[1] != 3:
        raise ValueError("goal and action coordinates must both have shape (n,3)")
    return np.einsum("ni,nj->nij", goals, actions).reshape(len(goals), 9)


def effective_rank(matrix: np.ndarray, tolerance: float = 1e-8) -> dict[str, float | int | np.ndarray]:
    values = np.asarray(matrix, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    if singular.size == 0:
        return {"rank": 0, "effective_rank": 0.0, "smallest_nonzero_singular_value": 0.0, "condition_number": float("inf"), "singular_values": singular}
    threshold = max(tolerance, tolerance * float(singular[0]))
    nonzero = singular[singular > threshold]
    probabilities = singular / max(float(singular.sum()), EPSILON)
    entropy_rank = float(np.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))))
    return {
        "rank": int(nonzero.size),
        "effective_rank": entropy_rank,
        "smallest_nonzero_singular_value": float(nonzero[-1]) if nonzero.size else 0.0,
        "condition_number": float(nonzero[0] / nonzero[-1]) if nonzero.size else float("inf"),
        "singular_values": singular,
    }


def observed_action_support_coverage(etas: np.ndarray) -> dict[str, float | np.ndarray]:
    values = np.asarray(etas, dtype=np.float64)
    standard_deviation = np.std(values, axis=0)
    return {
        "per_axis_std_over_half_width": standard_deviation,
        "mean_std_over_half_width": float(np.mean(standard_deviation)),
        "max_abs_eta": float(np.max(np.abs(values))) if values.size else 0.0,
    }


def counterfactual_augmented_features(goal_directions: np.ndarray, action_coordinates: np.ndarray) -> np.ndarray:
    goals = unit_directions(goal_directions)
    actions = np.asarray(action_coordinates, dtype=np.float64)
    expanded_goals = np.repeat(goals, len(actions), axis=0)
    expanded_actions = np.tile(actions, (len(goals), 1))
    return goal_action_interaction_features(expanded_goals, expanded_actions)

