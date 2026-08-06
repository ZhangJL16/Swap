from __future__ import annotations

from typing import Sequence
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.distributions import Normal


class FeedForwardAffineTanhActor(nn.Module):
    """Torch realization of T12A; c and G are detached certificate outputs."""

    def __init__(self, observation_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, 3)
        self.log_standard_deviation = nn.Linear(hidden_dim, 3)

    def distribution(self, observation: Tensor) -> Normal:
        feature = self.backbone(observation)
        mean = self.mean(feature)
        log_std = self.log_standard_deviation(feature).clamp(-10.0, 2.0)
        return Normal(mean, log_std.exp())

    def sample_u(self, observation: Sequence[float]) -> Sequence[float]:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32)
            return self.distribution(tensor).sample().cpu().tolist()

    def action_and_log_density(
        self,
        observation: Tensor,
        center: Tensor,
        generators: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.distribution(observation)
        u = distribution.rsample()
        eta = torch.tanh(u)
        frozen_center = center.detach()
        frozen_generators = generators.detach()
        action = frozen_center + frozen_generators @ eta
        log_abs_det = torch.linalg.slogdet(frozen_generators).logabsdet
        tanh_log_jacobian = self.stable_tanh_log_jacobian(u)
        log_density = distribution.log_prob(u).sum(-1) - tanh_log_jacobian - log_abs_det
        return action, log_density, u

    @staticmethod
    def stable_tanh_log_jacobian(u: Tensor) -> Tensor:
        """Sum log(1-tanh(u)^2) using the exact softplus identity."""

        return (2.0 * (log(2.0) - u - F.softplus(-2.0 * u))).sum(-1)

    @classmethod
    def log_density_from_u(
        cls,
        distribution: Normal,
        u: Tensor,
        generators: Tensor,
    ) -> Tensor:
        frozen_generators = generators.detach()
        return (
            distribution.log_prob(u).sum(-1)
            - cls.stable_tanh_log_jacobian(u)
            - torch.linalg.slogdet(frozen_generators).logabsdet
        )
