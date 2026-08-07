from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SchedulerBinaryDecision(IntEnum):
    CHARGE_OR_STAY = 0
    SERVE_OR_LEAVE = 1


class ChargingScheduler(Protocol):
    def select_action(self, observation: np.ndarray, context: dict, deterministic: bool = False) -> SchedulerBinaryDecision:
        ...


@dataclass(frozen=True, slots=True)
class SchedulerTransition:
    observation: np.ndarray
    requested_decision: int
    executed_decision: int
    cumulative_reward: float
    duration_steps: int
    next_observation: np.ndarray
    terminated: bool
    forced_override: bool
    override_reason: str | None
    scenario_id: str
    manifest_hash: str

    def __post_init__(self) -> None:
        observation = np.asarray(self.observation, dtype=np.float32).copy()
        next_observation = np.asarray(self.next_observation, dtype=np.float32).copy()
        if observation.ndim != 1 or next_observation.shape != observation.shape:
            raise ValueError("scheduler observations must be matching vectors")
        if self.duration_steps <= 0:
            raise ValueError("scheduler SMDP duration must be positive")
        if self.requested_decision not in (0, 1) or self.executed_decision not in (0, 1):
            raise ValueError("scheduler decisions must be binary")
        if not self.scenario_id or not self.manifest_hash:
            raise ValueError("scheduler replay requires scenario and manifest binding")
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "next_observation", next_observation)


class SchedulerReplayBuffer:
    def __init__(self, capacity: int, scenario_manifests: dict[str, str] | None = None) -> None:
        if capacity <= 0:
            raise ValueError("scheduler replay capacity must be positive")
        self.capacity = capacity
        self.scenario_manifests = dict(scenario_manifests or {})
        self.records: list[SchedulerTransition] = []

    def add(self, transition: SchedulerTransition) -> None:
        expected = self.scenario_manifests.get(transition.scenario_id)
        if expected is not None and transition.manifest_hash != expected:
            raise ValueError("scheduler scenario/manifest mismatch")
        self.records.append(SchedulerTransition(
            transition.observation.copy(),
            transition.requested_decision,
            transition.executed_decision,
            transition.cumulative_reward,
            transition.duration_steps,
            transition.next_observation.copy(),
            transition.terminated,
            transition.forced_override,
            transition.override_reason,
            transition.scenario_id,
            transition.manifest_hash,
        ))
        if len(self.records) > self.capacity:
            self.records.pop(0)

    def sample(self, batch_size: int, rng: np.random.Generator) -> list[SchedulerTransition]:
        if batch_size <= 0 or len(self.records) < batch_size:
            raise ValueError("insufficient scheduler replay")
        groups: dict[str, list[SchedulerTransition]] = {}
        for transition in self.records:
            groups.setdefault(transition.manifest_hash, []).append(transition)
        eligible = [records for records in groups.values() if len(records) >= batch_size]
        if not eligible:
            raise ValueError("no manifest-compatible scheduler batch")
        selected = max(eligible, key=len)
        indices = rng.choice(len(selected), size=batch_size, replace=False)
        return [selected[int(index)] for index in indices]


class ReserveOnlyScheduler:
    def select_action(self, observation: np.ndarray, context: dict, deterministic: bool = False) -> SchedulerBinaryDecision:
        del observation, deterministic
        if context.get("charging", False):
            return SchedulerBinaryDecision.SERVE_OR_LEAVE if context.get("departure_allowed", False) else SchedulerBinaryDecision.CHARGE_OR_STAY
        return SchedulerBinaryDecision.SERVE_OR_LEAVE


class FixedThresholdScheduler:
    def __init__(self, threshold: float) -> None:
        if not 0.0 < threshold <= 1.0:
            raise ValueError("charge threshold must lie in (0,1]")
        self.threshold = threshold

    def select_action(self, observation: np.ndarray, context: dict, deterministic: bool = False) -> SchedulerBinaryDecision:
        del observation, deterministic
        energy_fraction = float(context["energy_fraction"])
        if context.get("charging", False):
            leave = energy_fraction >= self.threshold and context.get("departure_allowed", False)
            return SchedulerBinaryDecision.SERVE_OR_LEAVE if leave else SchedulerBinaryDecision.CHARGE_OR_STAY
        return SchedulerBinaryDecision.CHARGE_OR_STAY if energy_fraction < self.threshold else SchedulerBinaryDecision.SERVE_OR_LEAVE


class FullChargeScheduler(FixedThresholdScheduler):
    def __init__(self) -> None:
        super().__init__(1.0)


class _MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


@dataclass(frozen=True, slots=True)
class ChargingSchedulerSACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    initial_alpha: float = 0.2
    target_entropy: float = 0.5
    hidden_dim: int = 128
    batch_size: int = 128
    replay_capacity: int = 100_000


class ChargingSchedulerSAC:
    """Discrete soft actor-critic over event-to-event SMDP transitions."""

    uses_generator_density = False

    def __init__(self, observation_dim: int, config: ChargingSchedulerSACConfig | None = None, seed: int = 0, device: str = "cpu") -> None:
        self.config = config or ChargingSchedulerSACConfig()
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.actor = _MLP(observation_dim, 2, self.config.hidden_dim).to(self.device)
        self.critic_1 = _MLP(observation_dim, 2, self.config.hidden_dim).to(self.device)
        self.critic_2 = _MLP(observation_dim, 2, self.config.hidden_dim).to(self.device)
        self.target_1 = _MLP(observation_dim, 2, self.config.hidden_dim).to(self.device)
        self.target_2 = _MLP(observation_dim, 2, self.config.hidden_dim).to(self.device)
        self.target_1.load_state_dict(self.critic_1.state_dict())
        self.target_2.load_state_dict(self.critic_2.state_dict())
        self.log_alpha = nn.Parameter(torch.tensor(np.log(self.config.initial_alpha), dtype=torch.float32, device=self.device))
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.config.critic_lr)
        self.alpha_optimizer = torch.optim.Adam((self.log_alpha,), lr=self.config.alpha_lr)
        self.replay = SchedulerReplayBuffer(self.config.replay_capacity)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, observation: np.ndarray, context: dict, deterministic: bool = False) -> SchedulerBinaryDecision:
        del context
        with torch.no_grad():
            logits = self.actor(torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))[0]
            if deterministic:
                action = int(logits.argmax().item())
            else:
                action = int(torch.distributions.Categorical(logits=logits).sample().item())
        return SchedulerBinaryDecision(action)

    @staticmethod
    def smdp_discount(gamma: float, duration_steps: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.full_like(duration_steps, gamma), duration_steps)

    def observe(self, transition: SchedulerTransition) -> None:
        self.replay.add(transition)

    def _soft_update(self) -> None:
        for target, online in ((self.target_1, self.critic_1), (self.target_2, self.critic_2)):
            for target_parameter, parameter in zip(target.parameters(), online.parameters()):
                target_parameter.data.mul_(1.0 - self.config.tau).add_(self.config.tau * parameter.data)

    def update(self) -> dict[str, float | str]:
        batch = self.replay.sample(self.config.batch_size, self.rng)
        observations = torch.as_tensor(np.stack([item.observation for item in batch]), dtype=torch.float32, device=self.device)
        next_observations = torch.as_tensor(np.stack([item.next_observation for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([item.executed_decision for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.cumulative_reward for item in batch], dtype=torch.float32, device=self.device)
        durations = torch.as_tensor([item.duration_steps for item in batch], dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor([item.terminated for item in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_log_probabilities = F.log_softmax(self.actor(next_observations), dim=-1)
            next_probabilities = next_log_probabilities.exp()
            next_q = torch.minimum(self.target_1(next_observations), self.target_2(next_observations))
            next_value = (next_probabilities * (next_q - self.alpha.detach() * next_log_probabilities)).sum(dim=-1)
            target = rewards + (1.0 - terminated) * self.smdp_discount(self.config.gamma, durations) * next_value
        q1 = self.critic_1(observations).gather(1, actions[:, None]).squeeze(1)
        q2 = self.critic_2(observations).gather(1, actions[:, None]).squeeze(1)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()
        log_probabilities = F.log_softmax(self.actor(observations), dim=-1)
        probabilities = log_probabilities.exp()
        q = torch.minimum(self.critic_1(observations), self.critic_2(observations)).detach()
        actor_loss = (probabilities * (self.alpha.detach() * log_probabilities - q)).sum(dim=-1).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        entropy = -(probabilities.detach() * log_probabilities).sum(dim=-1).mean()
        alpha_loss = -(self.log_alpha * (self.config.target_entropy - entropy).detach())
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self._soft_update()
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "scheduler_density": "categorical-softmax; independent of Generator affine-tanh density",
        }


def make_scheduler(name: str, observation_dim: int | None = None, **kwargs):
    if name == "reserve_only":
        return ReserveOnlyScheduler()
    if name == "fixed_threshold_30":
        return FixedThresholdScheduler(0.30)
    if name == "fixed_threshold_50":
        return FixedThresholdScheduler(0.50)
    if name == "full_charge":
        return FullChargeScheduler()
    if name == "scheduler_sac":
        if observation_dim is None:
            raise ValueError("scheduler_sac requires observation_dim")
        return ChargingSchedulerSAC(observation_dim, **kwargs)
    raise ValueError(f"unknown charging scheduler: {name}")
