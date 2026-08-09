from __future__ import annotations

from typing import Iterable

import numpy as np


def goal_key(goal: np.ndarray, decimals: int = 6) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(goal, dtype=np.float64), decimals=decimals))


def goal_direction_statistics(directions: Iterable[np.ndarray], bins: int = 16) -> dict[str, float]:
    vectors = []
    for direction in directions:
        horizontal = np.asarray(direction, dtype=np.float64)[:2]
        norm = float(np.linalg.norm(horizontal))
        if norm > 1e-12:
            vectors.append(horizontal / norm)
    if not vectors:
        return {"angular_coverage_degrees": 0.0, "direction_entropy": 0.0}
    angles = np.mod(np.arctan2(np.asarray(vectors)[:, 1], np.asarray(vectors)[:, 0]), 2.0 * np.pi)
    pairwise = np.abs(angles[:, None] - angles[None, :])
    pairwise = np.minimum(pairwise, 2.0 * np.pi - pairwise)
    counts, _ = np.histogram(angles, bins=bins, range=(0.0, 2.0 * np.pi))
    probabilities = counts[counts > 0] / max(1, counts.sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities)))
    return {
        "angular_coverage_degrees": float(np.degrees(np.max(pairwise))),
        "direction_entropy": entropy,
    }


def task_completion_distance_invariant(
    mode_before: str,
    distance_after: float,
    goal_radius: float,
    task_completed_now: bool,
) -> bool:
    eligible = mode_before == "TASK_RL" and float(distance_after) <= float(goal_radius) + 1e-12
    return not eligible or bool(task_completed_now)

