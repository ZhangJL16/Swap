from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Normal

from .actor import FeedForwardAffineTanhActor


@dataclass(frozen=True, slots=True)
class EntropyDecomposition:
    normal_term: Tensor
    tanh_log_jacobian: Tensor
    negative_tanh_log_jacobian_term: Tensor
    log_det_G: Tensor
    negative_log_det_G_term: Tensor
    normalized_log_prob: Tensor
    physical_log_prob: Tensor


def temperature_log_probability(
    terms: EntropyDecomposition,
    coordinate: str,
) -> Tensor:
    if coordinate == "physical":
        return terms.physical_log_prob
    if coordinate == "normalized":
        return terms.normalized_log_prob
    raise ValueError(f"unsupported temperature coordinate: {coordinate}")


def entropy_decomposition(distribution: Normal, u: Tensor, generators: Tensor) -> EntropyDecomposition:
    normal = distribution.log_prob(u).sum(-1)
    jacobian = FeedForwardAffineTanhActor.stable_tanh_log_jacobian(u)
    log_det = torch.linalg.slogdet(generators.detach()).logabsdet
    normalized = normal - jacobian
    physical = normalized - log_det
    return EntropyDecomposition(
        normal,
        jacobian,
        -jacobian,
        log_det,
        -log_det,
        normalized,
        physical,
    )


def affine_scale_entropy_audit(
    distribution: Normal,
    u: Tensor,
    generators: Tensor,
    target_entropy: float,
    log_alpha: Tensor,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for factor in (0.5, 1.0, 2.0):
        terms = entropy_decomposition(distribution, u, generators * factor)
        physical_residual = terms.physical_log_prob + target_entropy
        normalized_residual = terms.normalized_log_prob + target_entropy
        result[f"{factor:g}G"] = {
            "normalized_log_prob": float(terms.normalized_log_prob.mean().detach()),
            "physical_log_prob": float(terms.physical_log_prob.mean().detach()),
            "negative_log_det_G_term": float(terms.negative_log_det_G_term.mean().detach()),
            "physical_entropy_residual": float(physical_residual.mean().detach()),
            "normalized_entropy_residual": float(normalized_residual.mean().detach()),
            "physical_alpha_loss": float((-(log_alpha * physical_residual.detach())).mean().detach()),
            "normalized_alpha_loss": float((-(log_alpha * normalized_residual.detach())).mean().detach()),
            "physical_alpha_gradient": float((-physical_residual.detach()).mean()),
            "normalized_alpha_gradient": float((-normalized_residual.detach()).mean()),
        }
    return result


def goal_projection_metrics(
    position: np.ndarray,
    velocity: np.ndarray,
    goal: np.ndarray,
    actor_action: np.ndarray,
    center_action: np.ndarray,
    oracle_action: np.ndarray,
    dt: float,
) -> dict[str, float]:
    position_array = np.asarray(position, dtype=np.float64)
    velocity_array = np.asarray(velocity, dtype=np.float64)
    goal_array = np.asarray(goal, dtype=np.float64)
    direction = goal_array - position_array
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        direction = np.zeros(3, dtype=np.float64)
    else:
        direction /= norm

    def progress(action: np.ndarray) -> float:
        next_position = position_array + velocity_array * dt + 0.5 * np.asarray(action) * dt * dt
        return norm - float(np.linalg.norm(goal_array - next_position))

    actor = np.asarray(actor_action, dtype=np.float64)
    center = np.asarray(center_action, dtype=np.float64)
    oracle = np.asarray(oracle_action, dtype=np.float64)
    actor_progress = progress(actor)
    oracle_progress = progress(oracle)
    return {
        "actor_goal_projection": float(actor @ direction),
        "center_goal_projection": float(center @ direction),
        "oracle_goal_projection": float(oracle @ direction),
        "actor_vs_oracle_cosine": float(actor @ oracle / max(np.linalg.norm(actor) * np.linalg.norm(oracle), 1e-12)),
        "actor_vs_goal_cosine": float(actor @ direction / max(np.linalg.norm(actor), 1e-12)),
        "center_vs_goal_cosine": float(center @ direction / max(np.linalg.norm(center), 1e-12)),
        "predicted_actor_progress": actor_progress,
        "predicted_oracle_progress": oracle_progress,
        "oracle_gap": oracle_progress - actor_progress,
    }


def observation_component_statistics(observations: np.ndarray, layout: dict[str, slice]) -> dict[str, dict[str, float]]:
    values = np.asarray(observations, dtype=np.float64)
    result: dict[str, dict[str, float]] = {}
    for name, selected in layout.items():
        component = values[:, selected].reshape(-1)
        result[name] = {
            "mean": float(np.mean(component)),
            "std": float(np.std(component)),
            "min": float(np.min(component)),
            "max": float(np.max(component)),
            "fraction_clipped_negative_2": float(np.mean(component <= -2.0 + 1e-12)),
            "fraction_clipped_positive_2": float(np.mean(component >= 2.0 - 1e-12)),
        }
    return result
