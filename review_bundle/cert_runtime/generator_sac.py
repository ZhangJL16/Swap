from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from .actor import FeedForwardAffineTanhActor
from .persistent_authority import ExecutionAuthority
from .optimization_diagnostics import entropy_decomposition


EpochReplayPolicy = Literal["reject", "group", "clear_on_change"]


@dataclass(frozen=True)
class GeneratorSACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256
    replay_capacity: int = 1_000_000
    warmup_steps: int = 5_000
    updates_per_step: int = 1
    target_entropy: float = -3.0
    hidden_dim: int = 128
    bootstrap_on_truncation: bool = True
    epoch_replay_policy: EpochReplayPolicy = "clear_on_change"


def _copy_array(value):
    return None if value is None else np.asarray(value, dtype=np.float32).copy()


@dataclass(frozen=True)
class GeneratorTransition:
    observation: np.ndarray
    next_observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    episode_id: int
    mission_phase: str
    next_mission_phase: str
    certificate_epoch: str
    next_certificate_epoch: str
    u: np.ndarray | None
    eta: np.ndarray | None
    c: np.ndarray | None
    G: np.ndarray | None
    candidate_action: np.ndarray | None
    kappa_action: np.ndarray
    executed_action: np.ndarray
    measured_action: np.ndarray
    accepted: bool
    fallback_reason: str | None
    next_c: np.ndarray | None
    next_G: np.ndarray | None
    next_kappa: np.ndarray
    next_generator_available: bool
    next_certificate_valid: bool
    geometry_version: str
    corridor_version: str
    energy_version: str
    certificate_hashes: tuple[str | None, str | None]
    scenario_id: str | None = None
    scenario_family: str | None = None
    scenario_hash: str | None = None
    certificate_manifest_hash: str | None = None
    backup_triggered: bool = False
    backup_reason: str | None = None
    energy: float | None = None
    required_return_energy: float | None = None
    energy_margin: float | None = None
    charging: bool = False
    station_arrival: bool = False
    departure_attempt: bool = False
    departure_rejected: bool = False
    task_id: str | None = None
    goal_id: str | None = None
    tasks_completed: int = 0
    recoverable_set_version: str | None = None
    recoverability_action_rule_version: str | None = None
    execution_authority: str | None = None
    next_execution_authority: str | None = None
    next_generator_executable: bool | None = None
    next_backup_required: bool | None = None
    next_backup_reason: str | None = None
    next_recoverable_set_member: bool | None = None
    next_recoverability_action_verified: bool | None = None
    next_policy_authority_pass: bool | None = None
    next_energy_margin: float | None = None
    next_departure_allowed: bool | None = None
    next_charging_state: bool | None = None
    next_charging_restriction: bool | None = None
    next_authority_action: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("observation", "next_observation", "u", "eta", "c", "G", "candidate_action", "kappa_action", "executed_action", "measured_action", "next_c", "next_G", "next_kappa", "next_authority_action"):
            object.__setattr__(self, name, _copy_array(getattr(self, name)))
        if self.executed_action is None or self.executed_action.shape != (3,):
            raise ValueError("executed_action must have shape (3,)")
        if self.next_kappa is None or self.next_kappa.shape != (3,):
            raise ValueError("next_kappa must have shape (3,)")
        if self.next_authority_action is None:
            object.__setattr__(self, "next_authority_action", self.next_kappa.copy())
        elif self.next_authority_action.shape != (3,):
            raise ValueError("next_authority_action must have shape (3,)")
        if self.accepted and (self.c is None or self.G is None or self.u is None):
            raise ValueError("accepted transition requires u,c,G")
        if self.next_generator_available and (self.next_c is None or self.next_G is None):
            raise ValueError("generator-valid next state requires next c,G")
        if (
            self.certificate_manifest_hash is not None
            and self.certificate_manifest_hash != self.certificate_epoch
        ):
            raise ValueError("scenario certificate manifest does not match certificate epoch")
        if self.accepted and self.backup_triggered:
            raise ValueError("a transition cannot be both accepted and backup-controlled")
        if self.tasks_completed < 0:
            raise ValueError("tasks_completed must be nonnegative")


class GeneratorReplayBuffer:
    def __init__(self, capacity: int, epoch_policy: EpochReplayPolicy, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.epoch_policy = epoch_policy
        self.rng = np.random.default_rng(seed)
        self.transitions: list[GeneratorTransition] = []
        self.active_epoch: str | None = None
        self.epoch_rejection_count = 0

    def __len__(self) -> int:
        return len(self.transitions)

    def add(self, transition: GeneratorTransition) -> bool:
        if (
            transition.certificate_manifest_hash is not None
            and transition.certificate_manifest_hash != transition.certificate_epoch
        ):
            raise ValueError("scenario/certificate manifest mismatch")
        epoch = transition.certificate_epoch
        if self.active_epoch is None:
            self.active_epoch = epoch
        elif epoch != self.active_epoch:
            if self.epoch_policy == "reject":
                self.epoch_rejection_count += 1
                return False
            if self.epoch_policy == "clear_on_change":
                self.transitions.clear()
                self.active_epoch = epoch
        self.transitions.append(transition)
        if len(self.transitions) > self.capacity:
            del self.transitions[: len(self.transitions) - self.capacity]
        return True

    def sample(self, batch_size: int) -> list[GeneratorTransition]:
        if not self.transitions:
            raise ValueError("cannot sample empty replay")
        if self.epoch_policy == "group":
            epochs: dict[str, list[int]] = {}
            for index, transition in enumerate(self.transitions):
                epochs.setdefault(transition.certificate_epoch, []).append(index)
            eligible = [indices for indices in epochs.values() if len(indices) >= batch_size]
            if not eligible:
                raise ValueError("no certificate epoch contains a complete batch")
            pool = eligible[int(self.rng.integers(len(eligible)))]
        else:
            pool = list(range(len(self.transitions)))
            if len({self.transitions[index].certificate_epoch for index in pool}) != 1:
                raise ValueError("incompatible certificate epochs in replay")
        selected = self.rng.choice(pool, size=batch_size, replace=False)
        return [self.transitions[int(index)] for index in selected]


class QNetwork(nn.Module):
    def __init__(self, observation_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim + 3, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
        )

    def forward(self, observation: Tensor, action: Tensor) -> Tensor:
        return self.network(torch.cat((observation, action), dim=-1)).squeeze(-1)


class GeneratorSAC:
    """Off-policy Generator-SAC with a fallback atomic Bellman branch."""

    def __init__(self, observation_dim: int, config: GeneratorSACConfig | None = None, *, seed: int = 0, device: str = "cpu") -> None:
        self.config = GeneratorSACConfig() if config is None else config
        self.device = torch.device(device)
        torch.manual_seed(seed)
        self.actor = FeedForwardAffineTanhActor(observation_dim, self.config.hidden_dim).to(self.device)
        self.critic_1 = QNetwork(observation_dim, self.config.hidden_dim).to(self.device)
        self.critic_2 = QNetwork(observation_dim, self.config.hidden_dim).to(self.device)
        self.target_critic_1 = QNetwork(observation_dim, self.config.hidden_dim).to(self.device)
        self.target_critic_2 = QNetwork(observation_dim, self.config.hidden_dim).to(self.device)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=self.config.critic_lr)
        self.log_alpha = torch.tensor(0.0, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.config.alpha_lr)
        self.replay = GeneratorReplayBuffer(self.config.replay_capacity, self.config.epoch_replay_policy, seed)
        self.gradient_steps = 0
        self.generator_log_density_calls = 0

    @property
    def alpha(self) -> Tensor:
        return self.log_alpha.exp()

    @property
    def action_dimension(self) -> int:
        return 3

    def select_u(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
            distribution = self.actor.distribution(tensor)
            u = distribution.mean if deterministic else distribution.sample()
        return u.cpu().numpy().astype(np.float64)

    def observe(self, transition: GeneratorTransition) -> bool:
        return self.replay.add(transition)

    def _tensor(self, values) -> Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    @staticmethod
    def _mapped_action(c: Tensor, G: Tensor, u: Tensor) -> Tensor:
        return c + torch.bmm(G, torch.tanh(u).unsqueeze(-1)).squeeze(-1)

    def _sample_generator_actions(self, observations: Tensor, centers: Tensor, generators: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        distribution = self.actor.distribution(observations)
        u = distribution.rsample()
        actions = self._mapped_action(centers.detach(), generators.detach(), u)
        log_prob = distribution.log_prob(u).sum(-1) - self.actor.stable_tanh_log_jacobian(u) - torch.linalg.slogdet(generators.detach()).logabsdet
        self.generator_log_density_calls += int(observations.shape[0])
        return actions, log_prob, u

    def bellman_target(self, batch: Sequence[GeneratorTransition]) -> tuple[Tensor, dict[str, int]]:
        rewards = self._tensor([transition.reward for transition in batch])
        terminated = self._tensor([float(transition.terminated) for transition in batch])
        truncated = self._tensor([float(transition.truncated) for transition in batch])
        done = terminated if self.config.bootstrap_on_truncation else torch.maximum(terminated, truncated)
        next_observations = self._tensor(np.stack([transition.next_observation for transition in batch]))
        next_actions = self._tensor(np.stack([transition.next_kappa for transition in batch]))
        entropy = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        generator_indices = [index for index, transition in enumerate(batch) if transition.next_generator_available and transition.next_certificate_valid and not transition.terminated]
        if generator_indices:
            index_tensor = torch.as_tensor(generator_indices, dtype=torch.long, device=self.device)
            centers = self._tensor(np.stack([batch[index].next_c for index in generator_indices]))
            generators = self._tensor(np.stack([batch[index].next_G for index in generator_indices]))
            actions, log_prob, _ = self._sample_generator_actions(next_observations[index_tensor], centers, generators)
            next_actions = next_actions.clone()
            next_actions[index_tensor] = actions
            entropy[index_tensor] = self.alpha.detach() * log_prob
        with torch.no_grad():
            q_next = torch.minimum(self.target_critic_1(next_observations, next_actions), self.target_critic_2(next_observations, next_actions))
            target = rewards + self.config.gamma * (1.0 - done) * (q_next - entropy)
        return target, {"generator_target_count": len(generator_indices), "fallback_target_count": len(batch) - len(generator_indices)}

    @staticmethod
    def _finite_gradients(parameters) -> bool:
        return all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters)

    def update(self, batch: Sequence[GeneratorTransition] | None = None) -> dict[str, float | int | str]:
        selected = list(batch) if batch is not None else self.replay.sample(self.config.batch_size)
        if len({transition.certificate_epoch for transition in selected}) != 1:
            raise ValueError("mixed certificate epoch batch")
        observations = self._tensor(np.stack([transition.observation for transition in selected]))
        executed_actions = self._tensor(np.stack([transition.executed_action for transition in selected]))
        target, branch_counts = self.bellman_target(selected)
        prediction_1 = self.critic_1(observations, executed_actions)
        prediction_2 = self.critic_2(observations, executed_actions)
        critic_loss_1 = nn.functional.mse_loss(prediction_1, target)
        critic_loss_2 = nn.functional.mse_loss(prediction_2, target)
        critic_loss = critic_loss_1 + critic_loss_2
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        parameters = list(self.critic_1.parameters()) + list(self.critic_2.parameters())
        if not torch.isfinite(critic_loss) or not self._finite_gradients(parameters):
            raise FloatingPointError("nonfinite critic loss or gradient")
        critic_gradient_norm = float(torch.nn.utils.clip_grad_norm_(parameters, 100.0))
        self.critic_optimizer.step()
        accepted_indices = [index for index, transition in enumerate(selected) if transition.accepted and transition.c is not None and transition.G is not None]
        actor_status = "zero-accepted-sample"
        actor_loss_value = alpha_loss_value = mean_log_prob = mean_log_det = mean_jacobian = None
        entropy_metrics: dict[str, float | None] = {
            "mean_log_prob_u": None,
            "mean_negative_tanh_log_jacobian": None,
            "mean_negative_log_det_G": None,
            "mean_normalized_log_prob": None,
            "entropy_target_residual": None,
            "alpha_gradient": None,
            "mean_u": None,
            "std_u": None,
            "mean_abs_u": None,
            "max_abs_u": None,
            "mean_eta": None,
            "std_eta": None,
            "eta_abs_gt_090": None,
            "eta_abs_gt_095": None,
            "eta_abs_gt_099": None,
            "u_abs_gt_2": None,
            "u_abs_gt_3": None,
            "u_abs_gt_5": None,
        }
        actor_gradient_norm = 0.0
        if accepted_indices:
            index_tensor = torch.as_tensor(accepted_indices, dtype=torch.long, device=self.device)
            centers = self._tensor(np.stack([selected[index].c for index in accepted_indices])).detach()
            generators = self._tensor(np.stack([selected[index].G for index in accepted_indices])).detach()
            actions, log_prob, u = self._sample_generator_actions(observations[index_tensor], centers, generators)
            distribution = self.actor.distribution(observations[index_tensor])
            terms = entropy_decomposition(distribution, u, generators)
            q_value = torch.minimum(self.critic_1(observations[index_tensor], actions), self.critic_2(observations[index_tensor], actions))
            actor_loss = (self.alpha.detach() * log_prob - q_value).mean()
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            if not torch.isfinite(actor_loss) or not self._finite_gradients(self.actor.parameters()):
                raise FloatingPointError("nonfinite actor loss or gradient")
            actor_gradient_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 100.0))
            self.actor_optimizer.step()
            alpha_loss = -(self.log_alpha * (log_prob.detach() + self.config.target_entropy)).mean()
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            if not torch.isfinite(alpha_loss) or self.log_alpha.grad is None or not torch.isfinite(self.log_alpha.grad):
                raise FloatingPointError("nonfinite alpha loss or gradient")
            self.alpha_optimizer.step()
            actor_status = "updated"
            actor_loss_value = float(actor_loss.detach())
            alpha_loss_value = float(alpha_loss.detach())
            mean_log_prob = float(log_prob.mean().detach())
            mean_log_det = float(torch.linalg.slogdet(generators).logabsdet.mean().detach())
            mean_jacobian = float(self.actor.stable_tanh_log_jacobian(u).mean().detach())
            eta = torch.tanh(u)
            entropy_metrics = {
                "mean_log_prob_u": float(terms.normal_term.mean().detach()),
                "mean_negative_tanh_log_jacobian": float(terms.negative_tanh_log_jacobian_term.mean().detach()),
                "mean_negative_log_det_G": float(terms.negative_log_det_G_term.mean().detach()),
                "mean_normalized_log_prob": float(terms.normalized_log_prob.mean().detach()),
                "entropy_target_residual": float((terms.physical_log_prob + self.config.target_entropy).mean().detach()),
                "alpha_gradient": float(self.log_alpha.grad.detach()),
                "mean_u": float(u.mean().detach()),
                "std_u": float(u.std(unbiased=False).detach()),
                "mean_abs_u": float(u.abs().mean().detach()),
                "max_abs_u": float(u.abs().max().detach()),
                "mean_eta": float(eta.mean().detach()),
                "std_eta": float(eta.std(unbiased=False).detach()),
                "eta_abs_gt_090": float((eta.abs() > 0.90).float().mean().detach()),
                "eta_abs_gt_095": float((eta.abs() > 0.95).float().mean().detach()),
                "eta_abs_gt_099": float((eta.abs() > 0.99).float().mean().detach()),
                "u_abs_gt_2": float((u.abs() > 2.0).float().mean().detach()),
                "u_abs_gt_3": float((u.abs() > 3.0).float().mean().detach()),
                "u_abs_gt_5": float((u.abs() > 5.0).float().mean().detach()),
            }
        self.polyak_update()
        self.gradient_steps += 1
        return {
            "critic_loss_1": float(critic_loss_1.detach()), "critic_loss_2": float(critic_loss_2.detach()),
            "actor_loss": actor_loss_value, "alpha_loss": alpha_loss_value, "alpha": float(self.alpha.detach()),
            "mean_log_prob": mean_log_prob, "mean_log_det_G": mean_log_det,
            "mean_tanh_log_jacobian": mean_jacobian,
            "q_value_exec": float(torch.minimum(prediction_1, prediction_2).mean().detach()),
            "accepted_batch_count": len(accepted_indices), "fallback_batch_count": len(selected) - len(accepted_indices),
            "accepted_batch_fraction": len(accepted_indices) / len(selected), "actor_status": actor_status,
            "critic_gradient_norm": critic_gradient_norm, "actor_gradient_norm": actor_gradient_norm,
            "mean_reward": float(np.mean([transition.reward for transition in selected])),
            "mean_bellman_target": float(target.mean().detach()),
            "mean_td_error": float((target - torch.minimum(prediction_1, prediction_2)).mean().detach()),
            **entropy_metrics,
            **branch_counts,
        }

    def polyak_update(self) -> None:
        with torch.no_grad():
            for online, target in ((self.critic_1, self.target_critic_1), (self.critic_2, self.target_critic_2)):
                for online_parameter, target_parameter in zip(online.parameters(), target.parameters()):
                    target_parameter.data.mul_(1.0 - self.config.tau)
                    target_parameter.data.add_(self.config.tau * online_parameter.data)

    def state_dict(self) -> dict[str, object]:
        return {
            "actor": self.actor.state_dict(), "critic_1": self.critic_1.state_dict(), "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(), "target_critic_2": self.target_critic_2.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(), "config": self.config.__dict__.copy(), "gradient_steps": self.gradient_steps,
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic_1.load_state_dict(state["critic_1"])
        self.critic_2.load_state_dict(state["critic_2"])
        self.target_critic_1.load_state_dict(state["target_critic_1"])
        self.target_critic_2.load_state_dict(state["target_critic_2"])
        with torch.no_grad():
            self.log_alpha.copy_(torch.as_tensor(state["log_alpha"], device=self.device))
        self.gradient_steps = int(state.get("gradient_steps", 0))


class PersistentGeneratorSAC(GeneratorSAC):
    """Main persistent agent: one continuous three-dimensional policy."""

    _GENERATOR_AUTHORITIES = {
        ExecutionAuthority.RL_GENERATOR.value,
        ExecutionAuthority.CHARGER_CONSTRAINED.value,
    }

    @staticmethod
    def _validate_persistent_transition(transition: GeneratorTransition) -> None:
        if transition.next_execution_authority is None:
            raise ValueError("persistent transition requires next execution authority")
        try:
            authority = ExecutionAuthority(transition.next_execution_authority)
        except ValueError as error:
            raise ValueError("unknown persistent next execution authority") from error
        if authority in {ExecutionAuthority.RL_GENERATOR, ExecutionAuthority.CHARGER_CONSTRAINED}:
            if transition.next_generator_executable:
                if transition.next_c is None or transition.next_G is None:
                    raise ValueError("persistent Generator branch requires next c,G")
                if not all((
                    transition.next_certificate_valid,
                    transition.next_recoverable_set_member,
                    transition.next_recoverability_action_verified,
                    transition.next_policy_authority_pass,
                )):
                    raise ValueError("persistent Generator authority lacks certified prerequisites")
            elif authority == ExecutionAuthority.RL_GENERATOR:
                raise ValueError("RL_GENERATOR authority must be executable")
        if authority == ExecutionAuthority.KAPPA_BACKUP and not transition.next_backup_required:
            raise ValueError("KAPPA_BACKUP authority requires backup metadata")

    def observe(self, transition: GeneratorTransition) -> bool:
        self._validate_persistent_transition(transition)
        return super().observe(transition)

    def bellman_target(self, batch: Sequence[GeneratorTransition]) -> tuple[Tensor, dict[str, int]]:
        for transition in batch:
            self._validate_persistent_transition(transition)
        rewards = self._tensor([transition.reward for transition in batch])
        terminated = self._tensor([float(transition.terminated) for transition in batch])
        truncated = self._tensor([float(transition.truncated) for transition in batch])
        done = terminated if self.config.bootstrap_on_truncation else torch.maximum(terminated, truncated)
        fail_closed = self._tensor([
            float(transition.next_execution_authority == ExecutionAuthority.FAIL_CLOSED.value)
            for transition in batch
        ])
        bootstrap = (1.0 - done) * (1.0 - fail_closed)
        next_observations = self._tensor(np.stack([transition.next_observation for transition in batch]))
        next_actions = self._tensor(np.stack([transition.next_authority_action for transition in batch]))
        entropy = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        generator_indices = [
            index
            for index, transition in enumerate(batch)
            if (
                transition.next_execution_authority in self._GENERATOR_AUTHORITIES
                and transition.next_generator_executable is True
                and not transition.terminated
            )
        ]
        if generator_indices:
            index_tensor = torch.as_tensor(generator_indices, dtype=torch.long, device=self.device)
            centers = self._tensor(np.stack([batch[index].next_c for index in generator_indices]))
            generators = self._tensor(np.stack([batch[index].next_G for index in generator_indices]))
            actions, log_prob, _ = self._sample_generator_actions(next_observations[index_tensor], centers, generators)
            next_actions = next_actions.clone()
            next_actions[index_tensor] = actions
            entropy[index_tensor] = self.alpha.detach() * log_prob
        with torch.no_grad():
            q_next = torch.minimum(
                self.target_critic_1(next_observations, next_actions),
                self.target_critic_2(next_observations, next_actions),
            )
            target = rewards + self.config.gamma * bootstrap * (q_next - entropy)
        return target, {
            "generator_target_count": len(generator_indices),
            "fallback_target_count": len(batch) - len(generator_indices),
            "target_batch_count": len(batch),
            "rl_generator_target_count": sum(
                transition.next_execution_authority == ExecutionAuthority.RL_GENERATOR.value
                and transition.next_generator_executable is True
                for transition in batch
            ),
            "charger_target_count": sum(
                transition.next_execution_authority == ExecutionAuthority.CHARGER_CONSTRAINED.value
                for transition in batch
            ),
            "kappa_target_count": sum(
                transition.next_execution_authority == ExecutionAuthority.KAPPA_BACKUP.value
                for transition in batch
            ),
            "charger_atomic_target_count": sum(
                transition.next_execution_authority == ExecutionAuthority.CHARGER_CONSTRAINED.value
                and transition.next_generator_executable is not True
                for transition in batch
            ),
            "fail_closed_target_count": int(fail_closed.sum().item()),
        }
