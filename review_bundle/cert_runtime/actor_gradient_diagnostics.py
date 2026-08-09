from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .generator_sac import GeneratorSAC
from .optimization_diagnostics import entropy_decomposition


EPSILON = 1e-12


def scalar_statistics(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
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


def _minimum_q(agent: GeneratorSAC, observations: Tensor, actions: Tensor) -> Tensor:
    return torch.minimum(agent.critic_1(observations, actions), agent.critic_2(observations, actions))


def critic_action_gradient(
    agent: GeneratorSAC,
    observation: Tensor,
    action: Tensor,
) -> tuple[Tensor, Tensor]:
    selected = action.detach().clone().requires_grad_(True)
    value = _minimum_q(agent, observation, selected)
    gradient = torch.autograd.grad(value.sum(), selected)[0]
    return value.detach(), gradient.detach()


def directional_finite_difference(
    agent: GeneratorSAC,
    observation: Tensor,
    action: Tensor,
    direction: Tensor,
    epsilon_action: float = 1e-4,
) -> Tensor:
    with torch.no_grad():
        positive = _minimum_q(agent, observation, action + epsilon_action * direction)
        negative = _minimum_q(agent, observation, action - epsilon_action * direction)
    return (positive - negative) / (2.0 * epsilon_action)


@dataclass(frozen=True, slots=True)
class TransmissionResult:
    grad_action: Tensor
    grad_eta: Tensor
    grad_u: Tensor
    tanh_derivative: Tensor
    ratio_G: Tensor
    ratio_tanh: Tensor
    ratio_total: Tensor


def action_to_latent_gradient(
    grad_action: Tensor,
    generators: Tensor,
    u: Tensor,
) -> TransmissionResult:
    eta = torch.tanh(u)
    tanh_derivative = 1.0 - eta.square()
    grad_eta = torch.bmm(generators.transpose(1, 2), grad_action.unsqueeze(-1)).squeeze(-1)
    grad_u = grad_eta * tanh_derivative
    action_norm = torch.linalg.vector_norm(grad_action, dim=-1)
    eta_norm = torch.linalg.vector_norm(grad_eta, dim=-1)
    u_norm = torch.linalg.vector_norm(grad_u, dim=-1)
    return TransmissionResult(
        grad_action,
        grad_eta,
        grad_u,
        tanh_derivative,
        eta_norm / (action_norm + EPSILON),
        u_norm / (eta_norm + EPSILON),
        u_norm / (action_norm + EPSILON),
    )


def flatten_parameter_gradients(
    gradients: Sequence[Tensor | None],
    parameters: Sequence[nn.Parameter],
) -> Tensor:
    values = [
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for gradient, parameter in zip(gradients, parameters)
    ]
    return torch.cat(values) if values else torch.zeros(0)


def gradient_cosine(first: Tensor, second: Tensor) -> float:
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    if float(denominator) <= EPSILON:
        return 0.0
    return float((first @ second / denominator).detach())


def _layer_name(parameter_name: str) -> str:
    if parameter_name.startswith("backbone.0"):
        return "backbone_input"
    if parameter_name.startswith("backbone.2"):
        return "backbone_hidden"
    if parameter_name.startswith("mean"):
        return "mean_head"
    if parameter_name.startswith("log_standard_deviation"):
        return "log_std_head"
    return parameter_name.split(".", 1)[0]


def actor_gradient_decomposition(
    agent: GeneratorSAC,
    observations: Tensor,
    centers: Tensor,
    generators: Tensor,
    noise: Tensor,
) -> dict[str, object]:
    parameters = list(agent.actor.parameters())
    parameter_names = [name for name, _ in agent.actor.named_parameters()]
    distribution = agent.actor.distribution(observations)
    u = distribution.mean + distribution.stddev * noise
    actions = agent._mapped_action(centers.detach(), generators.detach(), u)
    q_loss = -_minimum_q(agent, observations, actions).mean()
    entropy_loss = agent.alpha.detach() * entropy_decomposition(distribution, u, generators).physical_log_prob.mean()
    q_gradients = torch.autograd.grad(q_loss, parameters, retain_graph=True, allow_unused=True)
    entropy_gradients = torch.autograd.grad(entropy_loss, parameters, retain_graph=True, allow_unused=True)
    q_flat = flatten_parameter_gradients(q_gradients, parameters)
    entropy_flat = flatten_parameter_gradients(entropy_gradients, parameters)
    total_flat = q_flat + entropy_flat
    layers: dict[str, dict[str, float]] = defaultdict(lambda: {"Q": 0.0, "entropy": 0.0, "total": 0.0})
    for name, q_gradient, entropy_gradient, parameter in zip(
        parameter_names, q_gradients, entropy_gradients, parameters
    ):
        layer = _layer_name(name)
        q_value = torch.zeros_like(parameter) if q_gradient is None else q_gradient
        entropy_value = torch.zeros_like(parameter) if entropy_gradient is None else entropy_gradient
        layers[layer]["Q"] += float(q_value.square().sum())
        layers[layer]["entropy"] += float(entropy_value.square().sum())
        layers[layer]["total"] += float((q_value + entropy_value).square().sum())
    for values in layers.values():
        for name in values:
            values[name] = float(np.sqrt(values[name]))
    q_norm = float(torch.linalg.vector_norm(q_flat))
    entropy_norm = float(torch.linalg.vector_norm(entropy_flat))
    total_norm = float(torch.linalg.vector_norm(total_flat))
    return {
        "Q_gradient_norm": q_norm,
        "entropy_gradient_norm": entropy_norm,
        "total_gradient_norm": total_norm,
        "entropy_to_Q_gradient_ratio": entropy_norm / (q_norm + EPSILON),
        "total_to_Q_gradient_ratio": total_norm / (q_norm + EPSILON),
        "Q_entropy_cosine": gradient_cosine(q_flat, entropy_flat),
        "gradient_sum_consistency_error": float(torch.linalg.vector_norm(total_flat - q_flat - entropy_flat)),
        "layer_norms": dict(layers),
    }


def actor_goal_jacobians(
    agent: GeneratorSAC,
    observation: Tensor,
    center: Tensor,
    generator: Tensor,
    goal_slice: slice,
) -> tuple[Tensor, Tensor]:
    selected = observation.detach().clone().requires_grad_(True)
    mean = agent.actor.distribution(selected).mean
    mean_rows = []
    action_rows = []
    action = center.detach() + torch.bmm(
        generator.detach(), torch.tanh(mean).unsqueeze(-1)
    ).squeeze(-1)
    for output_axis in range(3):
        mean_gradient = torch.autograd.grad(mean[:, output_axis].sum(), selected, retain_graph=True)[0]
        action_gradient = torch.autograd.grad(action[:, output_axis].sum(), selected, retain_graph=True)[0]
        mean_rows.append(mean_gradient[:, goal_slice])
        action_rows.append(action_gradient[:, goal_slice])
    return torch.stack(mean_rows, dim=1).detach(), torch.stack(action_rows, dim=1).detach()


def critic_goal_jacobian(
    agent: GeneratorSAC,
    observation: Tensor,
    action: Tensor,
    goal_slice: slice,
) -> Tensor:
    selected = observation.detach().clone().requires_grad_(True)
    value = _minimum_q(agent, selected, action.detach())
    gradient = torch.autograd.grad(value.sum(), selected)[0]
    return gradient[:, goal_slice].detach()


def q_through_actor_goal_gradient(
    agent: GeneratorSAC,
    observation: Tensor,
    center: Tensor,
    generator: Tensor,
    goal_slice: slice,
) -> Tensor:
    selected = observation.detach().clone().requires_grad_(True)
    mean = agent.actor.distribution(selected).mean
    action = center.detach() + torch.bmm(
        generator.detach(), torch.tanh(mean).unsqueeze(-1)
    ).squeeze(-1)
    value = _minimum_q(agent, selected, action)
    return torch.autograd.grad(value.sum(), selected)[0][:, goal_slice].detach()


def first_layer_column_statistics(
    linear: nn.Linear,
    groups: Mapping[str, slice],
) -> dict[str, dict[str, float]]:
    weight = linear.weight.detach()
    result = {}
    for name, selected in groups.items():
        values = weight[:, selected]
        result[name] = {
            "L2": float(torch.linalg.vector_norm(values)),
            "mean_abs": float(values.abs().mean()),
            "max_abs": float(values.abs().max()),
        }
    return result


def critic_action_column_statistics(critic: nn.Module, observation_dim: int) -> dict[str, object]:
    linear = critic.network[0]
    action = linear.weight.detach()[:, observation_dim:]
    observation = linear.weight.detach()[:, :observation_dim]
    return {
        "action_column_L2": float(torch.linalg.vector_norm(action)),
        "observation_column_L2": float(torch.linalg.vector_norm(observation)),
        "action_to_observation_weight_ratio": float(
            torch.linalg.vector_norm(action) / (torch.linalg.vector_norm(observation) + EPSILON)
        ),
        "per_action_axis_L2": [float(torch.linalg.vector_norm(action[:, axis])) for axis in range(3)],
    }


def interpolation_landscape(
    agent: GeneratorSAC,
    observation: Tensor,
    actor_action: Tensor,
    oracle_action: Tensor,
) -> dict[str, object]:
    lambdas = torch.linspace(0.0, 1.0, 11, dtype=actor_action.dtype, device=actor_action.device)
    actions = actor_action + lambdas[:, None] * (oracle_action - actor_action)
    repeated = observation.expand(actions.shape[0], -1)
    with torch.no_grad():
        values = _minimum_q(agent, repeated, actions)
    differences = values[1:] - values[:-1]
    tolerance = max(1e-7, 1e-5 * float(values.abs().max()))
    monotonic = bool(torch.all(differences >= -tolerance))
    initial = float(differences[0])
    flat = abs(initial) <= tolerance
    if monotonic and initial > tolerance:
        classification = "MONOTONIC_TOWARD_ORACLE"
    elif flat:
        classification = "FLAT_NEAR_ACTOR"
    elif initial < -tolerance:
        classification = "LOCAL_WRONG_DIRECTION"
    elif not monotonic:
        classification = "NONMONOTONIC"
    else:
        classification = "INITIAL_POSITIVE_SLOPE"
    return {
        "lambda": [float(value) for value in lambdas],
        "Q": [float(value) for value in values],
        "classification": classification,
        "initial_difference": initial,
        "tolerance": tolerance,
    }


def parameters_unchanged(before: Iterable[Tensor], after: Iterable[Tensor]) -> bool:
    return all(torch.equal(first, second) for first, second in zip(before, after))
