from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .actor import FeedForwardAffineTanhActor
from .runtime import ReplayRecord
from .trainer import CertificateEpoch


@dataclass(frozen=True)
class SmokeTransition:
    observation: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    record: ReplayRecord


class SmokeCritic(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


class MinimalGeneratorSAC:
    """Small epoch-frozen SAC optimizer used only for semantic smoke tests."""

    def __init__(self, observation_dim: int, seed: int, learning_rate: float = 3e-4) -> None:
        torch.manual_seed(seed)
        self.actor = FeedForwardAffineTanhActor(observation_dim, 64)
        self.critic_1 = SmokeCritic(observation_dim)
        self.critic_2 = SmokeCritic(observation_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer_1 = torch.optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_optimizer_2 = torch.optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        self.log_alpha = torch.tensor(np.log(0.2), dtype=torch.float32, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=learning_rate)
        self.target_entropy = -3.0
        self.epoch: CertificateEpoch | None = None
        self.generator_log_density_calls = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    def freeze_epoch(self, record: ReplayRecord) -> CertificateEpoch:
        epoch = CertificateEpoch.from_snapshot(record.certificate_state)
        if self.epoch is None:
            self.epoch = epoch
        elif self.epoch != epoch:
            raise ValueError("certificate epoch changed during smoke optimization")
        return epoch

    def _validate(self, transitions: Sequence[SmokeTransition]) -> None:
        if self.epoch is None or not transitions:
            raise ValueError("a nonempty frozen-epoch batch is required")
        if any(not self.epoch.accepts(transition.record) for transition in transitions):
            raise ValueError("mixed certificate epoch/version batch")

    @staticmethod
    def _finite_gradients(module: nn.Module) -> bool:
        return all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in module.parameters())

    def update(self, transitions: Sequence[SmokeTransition]) -> dict[str, float | int | str]:
        self._validate(transitions)
        observations = torch.as_tensor(np.stack([item.observation for item in transitions]), dtype=torch.float32)
        executed_actions = torch.as_tensor(
            np.asarray([item.record.executed_action for item in transitions]), dtype=torch.float32
        )
        rewards = torch.as_tensor([item.reward for item in transitions], dtype=torch.float32)
        prediction_1 = self.critic_1(observations, executed_actions)
        prediction_2 = self.critic_2(observations, executed_actions)
        critic_loss_1 = torch.nn.functional.mse_loss(prediction_1, rewards)
        critic_loss_2 = torch.nn.functional.mse_loss(prediction_2, rewards)
        for optimizer, loss, critic in (
            (self.critic_optimizer_1, critic_loss_1, self.critic_1),
            (self.critic_optimizer_2, critic_loss_2, self.critic_2),
        ):
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(loss) or not self._finite_gradients(critic):
                raise FloatingPointError("nonfinite critic loss or gradient")
            optimizer.step()
        accepted = [index for index, item in enumerate(transitions) if item.record.accepted]
        actor_status = "updated" if accepted else "zero-accepted-sample"
        actor_loss_value = float("nan")
        alpha_loss_value = float("nan")
        mean_log_prob = float("nan")
        mean_log_det = float("nan")
        mean_jacobian = float("nan")
        if accepted:
            actor_terms, log_probs, log_dets, jacobians = [], [], [], []
            for index in accepted:
                record = transitions[index].record
                center = torch.as_tensor(record.zonotope_center, dtype=torch.float32).detach()
                generators = torch.as_tensor(record.zonotope_generators, dtype=torch.float32).detach()
                action, log_probability, u = self.actor.action_and_log_density(
                    observations[index], center, generators
                )
                self.generator_log_density_calls += 1
                actor_terms.append(self.alpha.detach() * log_probability - torch.minimum(
                    self.critic_1(observations[index], action),
                    self.critic_2(observations[index], action),
                ).squeeze())
                log_probs.append(log_probability)
                log_dets.append(torch.linalg.slogdet(generators).logabsdet)
                jacobians.append(self.actor.stable_tanh_log_jacobian(u))
                if center.requires_grad or generators.requires_grad:
                    raise AssertionError("certificate c,G were not detached")
            actor_loss = torch.stack(actor_terms).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if not torch.isfinite(actor_loss) or not self._finite_gradients(self.actor):
                raise FloatingPointError("nonfinite actor loss or gradient")
            self.actor_optimizer.step()
            detached_log_probs = torch.stack(log_probs).detach()
            alpha_loss = -(self.log_alpha * (detached_log_probs + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            if not torch.isfinite(alpha_loss) or not torch.isfinite(self.log_alpha.grad):
                raise FloatingPointError("nonfinite temperature loss or gradient")
            self.alpha_optimizer.step()
            actor_loss_value = float(actor_loss.detach())
            alpha_loss_value = float(alpha_loss.detach())
            mean_log_prob = float(detached_log_probs.mean())
            mean_log_det = float(torch.stack(log_dets).mean())
            mean_jacobian = float(torch.stack(jacobians).mean())
        return {
            "critic_loss_1": float(critic_loss_1.detach()),
            "critic_loss_2": float(critic_loss_2.detach()),
            "actor_loss": actor_loss_value,
            "alpha_loss": alpha_loss_value,
            "alpha": float(self.alpha.detach()),
            "mean_log_prob": mean_log_prob,
            "mean_log_det_G": mean_log_det,
            "mean_tanh_log_jacobian": mean_jacobian,
            "accepted_batch_count": len(accepted),
            "fallback_batch_count": len(transitions) - len(accepted),
            "actor_status": actor_status,
            "q_value_exec": float(torch.minimum(prediction_1, prediction_2).mean().detach()),
        }


def density_gradient_acceptance(seed: int = 0) -> dict[str, object]:
    torch.manual_seed(seed)
    actor = FeedForwardAffineTanhActor(4, 8).double()
    observation = torch.zeros(4, dtype=torch.float64)
    epsilon = torch.tensor([0.2, -0.7, 1.1], dtype=torch.float64)
    results = []
    max_formula_error = 0.0
    max_gradient_error = 0.0
    for scale in (0.01, 0.05, 0.2):
        generators = torch.diag(torch.full((3,), scale, dtype=torch.float64))
        distribution = actor.distribution(observation)
        u = distribution.mean + distribution.stddev * epsilon
        implemented = actor.log_density_from_u(distribution, u, generators)
        manual = (
            distribution.log_prob(u).sum()
            - torch.log(1.0 - torch.tanh(u).square()).sum()
            - torch.log(torch.abs(torch.linalg.det(generators)))
        )
        formula_error = abs(float(implemented - manual))
        max_formula_error = max(max_formula_error, formula_error)
        actor.zero_grad(set_to_none=True)
        implemented.backward()
        analytical = actor.mean.bias.grad.detach().clone()
        finite = []
        step = 1e-5
        with torch.no_grad():
            for index in range(3):
                actor.mean.bias[index] += step
                plus_distribution = actor.distribution(observation)
                plus_u = plus_distribution.mean + plus_distribution.stddev * epsilon
                plus = actor.log_density_from_u(plus_distribution, plus_u, generators)
                actor.mean.bias[index] -= 2.0 * step
                minus_distribution = actor.distribution(observation)
                minus_u = minus_distribution.mean + minus_distribution.stddev * epsilon
                minus = actor.log_density_from_u(minus_distribution, minus_u, generators)
                actor.mean.bias[index] += step
                finite.append(float((plus - minus) / (2.0 * step)))
        gradient_error = float(torch.max(torch.abs(analytical - torch.tensor(finite, dtype=torch.float64))))
        max_gradient_error = max(max_gradient_error, gradient_error)
        results.append(
            {
                "scale": scale,
                "log_density": float(implemented.detach()),
                "log_abs_det_G": float(torch.linalg.slogdet(generators).logabsdet),
                "formula_abs_error": formula_error,
                "mean_gradient_abs_max": float(analytical.abs().max()),
                "finite_difference_abs_error": gradient_error,
                "finite": bool(torch.isfinite(implemented)),
            }
        )
    return {
        "cases": results,
        "maximum_formula_absolute_error": max_formula_error,
        "maximum_gradient_absolute_error": max_gradient_error,
        "scope": "accepted affine-tanh density only; not T9A realizability",
    }
