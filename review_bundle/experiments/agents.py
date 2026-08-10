from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from cert_runtime.actor import FeedForwardAffineTanhActor
from cert_runtime.generator_sac import QNetwork


class ContinuousAgent(Protocol):
    def select_action(self, observation: np.ndarray, context: dict, deterministic: bool = False) -> np.ndarray: ...
    def observe(self, transition) -> None: ...
    def update(self) -> dict[str, float | int | str] | None: ...


class StatelessGeneratorPolicy:
    """Non-learning Generator ablation sharing the certified c, G, and kappa."""

    def __init__(self, mode: str, seed: int) -> None:
        if mode not in {"center_only", "random_generator"}:
            raise ValueError(f"unsupported stateless Generator mode: {mode}")
        self.mode = mode
        self.rng = np.random.default_rng(seed)

    def select_u(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        del observation
        if self.mode == "center_only" or deterministic:
            return np.zeros(3, dtype=np.float64)
        return self.rng.normal(size=3)

    def observe(self, transition) -> None:
        del transition

    def update(self) -> None:
        return None


@dataclass(frozen=True)
class DirectTransition:
    observation: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    executed_action: np.ndarray

    def __post_init__(self) -> None:
        for name in ("observation", "next_observation", "executed_action"):
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=np.float32).copy())


class DirectActor(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(observation_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.mean = nn.Linear(hidden_dim, 3)
        self.log_std = nn.Linear(hidden_dim, 3)

    def distribution(self, observation):
        features = self.backbone(observation)
        return Normal(self.mean(features), self.log_std(features).clamp(-10.0, 2.0).exp())


class DirectSACAgent:
    """Vanilla/penalty/shield SAC; critics always consume executed actions."""

    def __init__(self, observation_dim: int, action_max: np.ndarray, *, seed: int, batch_size: int = 64, capacity: int = 100000, hidden_dim: int = 128, gamma: float = 0.99, tau: float = 0.005, device: str = "cpu") -> None:
        torch.manual_seed(seed)
        self.device = torch.device(device)
        self.action_max = torch.as_tensor(action_max, dtype=torch.float32, device=self.device)
        self.actor = DirectActor(observation_dim, hidden_dim).to(self.device)
        self.critic_1 = QNetwork(observation_dim, hidden_dim).to(self.device)
        self.critic_2 = QNetwork(observation_dim, hidden_dim).to(self.device)
        self.target_critic_1 = QNetwork(observation_dim, hidden_dim).to(self.device)
        self.target_critic_2 = QNetwork(observation_dim, hidden_dim).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=3e-4)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=3e-4)
        self.gamma, self.tau, self.batch_size, self.capacity = gamma, tau, batch_size, capacity
        self.replay: list[DirectTransition] = []
        self.rng = np.random.default_rng(seed)
        self.gradient_steps = 0

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _sample(self, observations, deterministic=False):
        distribution = self.actor.distribution(observations)
        u = distribution.mean if deterministic else distribution.rsample()
        eta = torch.tanh(u)
        action = self.action_max * eta
        log_prob = distribution.log_prob(u).sum(-1) - FeedForwardAffineTanhActor.stable_tanh_log_jacobian(u) - torch.log(self.action_max).sum()
        return action, log_prob

    def select_action(self, observation: np.ndarray, context: dict | None = None, deterministic: bool = False) -> np.ndarray:
        del context
        with torch.no_grad():
            action, _ = self._sample(torch.as_tensor(observation, dtype=torch.float32, device=self.device), deterministic)
            return action.cpu().numpy().astype(np.float64)

    def observe(self, transition: DirectTransition) -> None:
        self.replay.append(transition)
        if len(self.replay) > self.capacity:
            del self.replay[0]

    def update(self):
        if len(self.replay) < self.batch_size:
            return None
        indices = self.rng.choice(len(self.replay), self.batch_size, replace=False)
        batch = [self.replay[int(index)] for index in indices]
        obs = torch.as_tensor(np.stack([item.observation for item in batch]), dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(np.stack([item.next_observation for item in batch]), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.stack([item.executed_action for item in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        done = torch.as_tensor([float(item.terminated) for item in batch], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            next_action, next_log_prob = self._sample(next_obs)
            target_q = torch.minimum(self.target_critic_1(next_obs, next_action), self.target_critic_2(next_obs, next_action))
            target = rewards + self.gamma * (1.0 - done) * (target_q - self.alpha.detach() * next_log_prob)
        pred_1, pred_2 = self.critic_1(obs, actions), self.critic_2(obs, actions)
        loss_1, loss_2 = nn.functional.mse_loss(pred_1, target), nn.functional.mse_loss(pred_2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        (loss_1 + loss_2).backward()
        critic_grad = float(torch.nn.utils.clip_grad_norm_(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), 100.0))
        self.critic_optimizer.step()
        sampled_action, log_prob = self._sample(obs)
        actor_loss = (self.alpha.detach() * log_prob - torch.minimum(self.critic_1(obs, sampled_action), self.critic_2(obs, sampled_action))).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0))
        self.actor_optimizer.step()
        alpha_loss = -(self.log_alpha * (log_prob.detach() - 3.0)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for online, target_network in ((self.critic_1, self.target_critic_1), (self.critic_2, self.target_critic_2)):
                for source, target_parameter in zip(online.parameters(), target_network.parameters()):
                    target_parameter.mul_(1.0 - self.tau).add_(self.tau * source)
        self.gradient_steps += 1
        return {
            "actor_loss": float(actor_loss.detach()), "critic_loss_1": float(loss_1.detach()), "critic_loss_2": float(loss_2.detach()),
            "alpha": float(self.alpha.detach()), "entropy": float(-log_prob.mean().detach()),
            "q_value_exec": float(torch.minimum(pred_1, pred_2).mean().detach()), "gradient_norm": max(actor_grad, critic_grad),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "gradient_steps": self.gradient_steps,
            "rng_state": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic_1.load_state_dict(state["critic_1"])
        self.critic_2.load_state_dict(state["critic_2"])
        if "target_critic_1" in state:
            self.target_critic_1.load_state_dict(state["target_critic_1"])
            self.target_critic_2.load_state_dict(state["target_critic_2"])
        else:
            self.target_critic_1.load_state_dict(self.critic_1.state_dict())
            self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        self.gradient_steps = int(state.get("gradient_steps", 0))
        if "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
