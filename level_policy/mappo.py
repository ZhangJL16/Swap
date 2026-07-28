import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

POLICY_SCOPE = "level_policy"


class DiscreteActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.logits(x)


class GaussianActorNetwork(nn.Module):
    def __init__(self, input_dims, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        mean = torch.tanh(self.mean(x))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std


class HybridHighActorNetwork(nn.Module):
    def __init__(
        self,
        input_dims,
        n_modes,
        continuous_dim,
        hidden_dim=128,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mode_logits = nn.Linear(hidden_dim, n_modes)
        self.mean = nn.Linear(hidden_dim, continuous_dim)
        self.log_std = nn.Parameter(torch.zeros(continuous_dim))

    def encode(self, state):
        x = torch.tanh(self.fc1(state))
        return torch.tanh(self.fc2(x))

    def forward(self, state):
        h = self.encode(state)
        mode_logits = self.mode_logits(h)
        mean = torch.tanh(self.mean(h))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mode_logits, mean, std, h


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return self.v(x)


class MAPPO:
    def __init__(self, args):
        self.args = args
        self.n_agents = args.n_agents
        self.low_n_actions = args.n_actions
        self.low_action_type = getattr(args, "low_action_type", "discrete")
        self.low_continuous = self.low_action_type == "continuous"
        self.low_action_dim = int(getattr(args, "low_action_dim", self.low_n_actions))
        self.low_obs_shape = args.obs_shape
        self.low_state_shape = args.state_shape
        self.high_n_actions = int(getattr(args, "high_level_n_actions", 0))
        self.high_mode_n_actions = int(getattr(args, "high_level_mode_n_actions", 0))
        self.high_continuous_dim = max(0, self.high_n_actions - 1)
        self.use_hybrid_high_policy = (
            self.high_mode_n_actions > 1 and self.high_n_actions >= 2
        )
        self.high_obs_shape = int(getattr(args, "high_level_obs_shape", 0))
        self.high_state_shape = int(getattr(args, "high_level_state_shape", 0))
        self.high_actor_obs_shape = self.high_obs_shape
        self.last_train_metrics = {}
        self.high_level_enabled = (
            self.high_n_actions > 0
            and self.high_obs_shape > 0
            and self.high_state_shape > 0
        )

        self.device = torch.device(
            f"cuda:{getattr(self.args, 'gpu_id', 0)}"
            if self.args.cuda and torch.cuda.is_available()
            else "cpu"
        )

        actor_hidden_dim = getattr(args, "actor_hidden_dim", 128)
        critic_hidden_dim = getattr(args, "critic_hidden_dim", 128)
        high_actor_hidden_dim = getattr(args, "high_actor_hidden_dim", actor_hidden_dim)
        high_critic_hidden_dim = getattr(args, "high_critic_hidden_dim", critic_hidden_dim)
        high_lr_actor = getattr(args, "high_lr_actor", args.lr_actor)
        high_lr_critic = getattr(args, "high_lr_critic", args.lr_critic)

        low_actor_cls = GaussianActorNetwork if self.low_continuous else DiscreteActorNetwork
        low_actor_action_dim = self.low_action_dim if self.low_continuous else self.low_n_actions
        self.low_actors = nn.ModuleList(
            [
                low_actor_cls(
                    self.low_obs_shape, low_actor_action_dim, actor_hidden_dim
                )
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.low_critics = nn.ModuleList(
            [
                CriticNetwork(self.low_state_shape, critic_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        if self.high_level_enabled:
            self.high_actors = nn.ModuleList(
                [
                    (
                        HybridHighActorNetwork(
                            self.high_actor_obs_shape,
                            self.high_mode_n_actions,
                            self.high_continuous_dim,
                            high_actor_hidden_dim,
                        )
                        if self.use_hybrid_high_policy
                        else GaussianActorNetwork(
                            self.high_actor_obs_shape,
                            self.high_n_actions,
                            high_actor_hidden_dim,
                        )
                    )
                    for _ in range(self.n_agents)
                ]
            ).to(self.device)
            self.high_critics = nn.ModuleList(
                [
                    CriticNetwork(self.high_state_shape, high_critic_hidden_dim)
                    for _ in range(self.n_agents)
                ]
            ).to(self.device)
        else:
            self.high_actors = nn.ModuleList().to(self.device)
            self.high_critics = nn.ModuleList().to(self.device)
        self.low_actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
            for actor in self.low_actors
        ]
        self.low_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
            for critic in self.low_critics
        ]
        self.high_actor_optimizers = (
            [
                torch.optim.Adam(actor.parameters(), lr=high_lr_actor)
                for actor in self.high_actors
            ]
            if self.high_level_enabled
            else []
        )
        self.high_critic_optimizers = (
            [
                torch.optim.Adam(critic.parameters(), lr=high_lr_critic)
                for critic in self.high_critics
            ]
            if self.high_level_enabled
            else []
        )
        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map
        self.eval_hidden = None

        auto_load_low = (
            getattr(self.args, "alg", "") == "hmappo_cbf_flow"
            and getattr(self.args, "training_phase", "low_cbf") != "low_cbf"
        )
        if self.args.load_model:
            self._load_models()
        elif auto_load_low:
            self._try_load_existing_low_models()
        pretrained_low_dir = getattr(self.args, "hmappo_pretrained_low_model_dir", "")
        if pretrained_low_dir:
            self._load_pretrained_low_models(pretrained_low_dir)

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, 1), device=self.device
        )

    def _masked_categorical(self, logits, avail_actions):
        masked_logits = logits.masked_fill(avail_actions <= 0, -1e10)
        return Categorical(logits=masked_logits)

    def _gaussian_dist(self, network, observation):
        mean, std = network(observation)
        return Normal(mean, std)

    def _hybrid_high_dist(
        self,
        network,
        observation,
        agent_idx=None,
        return_aux=False,
        mode_avail=None,
    ):
        mode_logits, mean, std, h = network(observation)
        adjusted_logits = mode_logits
        if mode_avail is not None and mode_avail.numel() > 0:
            mode_avail = mode_avail[:, : self.high_mode_n_actions]
            if mode_avail.size(-1) == self.high_mode_n_actions:
                all_blocked = mode_avail.sum(dim=-1, keepdim=True) <= 0.0
                mode_avail = torch.where(all_blocked, torch.ones_like(mode_avail), mode_avail)
                adjusted_logits = adjusted_logits.masked_fill(mode_avail <= 0.0, -1e10)
        mode_dist = Categorical(logits=adjusted_logits)
        cont_dist = Normal(mean, std)
        if return_aux:
            return mode_dist, cont_dist, h, mode_logits
        return mode_dist, cont_dist

    @torch.no_grad()
    def _choose_from_network(self, network, observation, avail_actions, evaluate=False):
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        avail = torch.tensor(
            avail_actions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        if torch.sum(avail) <= 0:
            return 0
        logits = network(obs)
        dist = self._masked_categorical(logits, avail)
        if evaluate:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    @torch.no_grad()
    def choose_action(self, observation, agent_idx, avail_actions, evaluate=False):
        if self.low_continuous:
            obs = torch.tensor(
                observation, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            dist = self._gaussian_dist(self.low_actors[agent_idx], obs)
            action = dist.mean if evaluate else dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            return action.squeeze(0).detach().cpu().numpy().astype("float32")
        return self._choose_from_network(
            self.low_actors[agent_idx], observation, avail_actions, evaluate
        )

    @torch.no_grad()
    def get_low_action_log_probs(self, observations, actions):
        if not self.low_continuous:
            return None
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        log_probs = np.zeros((self.n_agents, 1), dtype=np.float32)
        for agent_idx in range(self.n_agents):
            obs = torch.tensor(
                observations[agent_idx], dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            action = torch.tensor(
                actions[agent_idx], dtype=torch.float32, device=self.device
            ).view(1, -1)
            dist = self._gaussian_dist(self.low_actors[agent_idx], obs)
            log_probs[agent_idx, 0] = float(dist.log_prob(action).sum(dim=-1).item())
        return log_probs

    @torch.no_grad()
    def choose_high_level_action(
        self, observation, agent_idx, avail_actions=None, evaluate=False
    ):
        if not self.high_level_enabled:
            return np.zeros((0,), dtype=np.float32)
        obs = torch.tensor(
            observation, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        mode_avail = None
        if avail_actions is not None:
            mode_avail = torch.tensor(
                avail_actions, dtype=torch.float32, device=self.device
            ).view(1, -1)
        if self.use_hybrid_high_policy:
            mode_dist, continuous_dist = self._hybrid_high_dist(
                self.high_actors[agent_idx],
                obs,
                agent_idx=agent_idx,
                mode_avail=mode_avail,
            )
            if evaluate:
                mode = torch.argmax(mode_dist.logits, dim=-1)
                continuous = continuous_dist.mean
            else:
                mode = mode_dist.sample()
                continuous = continuous_dist.sample()
            continuous = torch.clamp(continuous, -1.0, 1.0)
            action = torch.cat([mode.float().unsqueeze(-1), continuous], dim=-1)
            return action.squeeze(0).detach().cpu().numpy().astype("float32")

        dist = self._gaussian_dist(self.high_actors[agent_idx], obs)
        if evaluate:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        return action.squeeze(0).detach().cpu().numpy().astype("float32")

    def _prepare_batch(self, batch):
        tensor_batch = {}
        for key, value in batch.items():
            if key == "u" and not self.low_continuous:
                tensor_batch[key] = torch.tensor(
                    value, dtype=torch.long, device=self.device
                )
            else:
                tensor_batch[key] = torch.tensor(
                    value, dtype=torch.float32, device=self.device
                )
        return tensor_batch

    def _compute_advantages(self, rewards, values, next_values, terminated, mask):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.size(0), device=self.device)
        for t in reversed(range(rewards.size(1))):
            not_done = 1 - terminated[:, t]
            delta = rewards[:, t] + self.args.gamma * next_values[:, t] * not_done - values[:, t]
            gae = delta + self.args.gamma * self.args.gae_lambda * not_done * gae
            gae = gae * mask[:, t]
            advantages[:, t] = gae
        returns = advantages + values

        valid_advantages = advantages[mask > 0]
        if valid_advantages.numel() > 0:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-8
            )
        return advantages, returns

    def _compute_smdp_advantages(
        self, rewards, values, next_values, terminated, mask, durations
    ):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(rewards.size(0), device=self.device)
        durations = torch.clamp(durations, min=1.0)
        gamma_k = torch.pow(
            torch.full_like(durations, float(self.args.gamma)),
            durations,
        )
        lambda_k = torch.pow(
            torch.full_like(durations, float(self.args.gamma * self.args.gae_lambda)),
            durations,
        )
        for t in reversed(range(rewards.size(1))):
            not_done = 1 - terminated[:, t]
            delta = (
                rewards[:, t]
                + gamma_k[:, t] * next_values[:, t] * not_done
                - values[:, t]
            )
            gae = delta + lambda_k[:, t] * not_done * gae
            gae = gae * mask[:, t]
            advantages[:, t] = gae
        returns = advantages + values

        valid_advantages = advantages[mask > 0]
        if valid_advantages.numel() > 0:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-8
            )
        return advantages, returns

    def learn(self, batch, max_episode_len, train_step, epsilon):
        del max_episode_len, epsilon
        self.last_train_metrics = {}
        batch = self._prepare_batch(batch)
        freeze_low = bool(getattr(self.args, "hmappo_freeze_low_level", False))
        freeze_high = bool(getattr(self.args, "hmappo_freeze_high_level", False))
        if not freeze_low:
            if self.low_continuous:
                low_action_key = (
                    "u_raw"
                    if self.args.alg == "hmappo_cbf_flow" and "u_raw" in batch
                    else "u"
                )
                self._learn_continuous_level(
                    batch=batch,
                    actors=self.low_actors,
                    critics=self.low_critics,
                    actor_optimizers=self.low_actor_optimizers,
                    critic_optimizers=self.low_critic_optimizers,
                    obs_key="o",
                    state_key="s",
                    next_state_key="s_next",
                    action_key=low_action_key,
                    reward_key="r",
                    padded_key="padded",
                    terminated_key="terminated",
                    active_key="agent_active_mask",
                    action_dim=self.low_action_dim,
                    obs_dim=self.low_obs_shape,
                    state_dim=self.low_state_shape,
                    intervention_key="guard_applied",
                    correct_action_key="u_correct",
                    raw_log_prob_key="raw_log_prob",
                )
            else:
                self._learn_level(
                    batch=batch,
                    actors=self.low_actors,
                    critics=self.low_critics,
                    actor_optimizers=self.low_actor_optimizers,
                    critic_optimizers=self.low_critic_optimizers,
                    obs_key="o",
                    state_key="s",
                    next_state_key="s_next",
                    action_key="u",
                    avail_key="avail_u",
                    reward_key="r",
                    padded_key="padded",
                    terminated_key="terminated",
                    active_key="agent_active_mask",
                    action_dim=self.low_n_actions,
                    obs_dim=self.low_obs_shape,
                    state_dim=self.low_state_shape,
                    guard_key="guard_applied",
                )
        if not self.high_level_enabled or "high_o" not in batch or freeze_high:
            return

        learn_high = (
            self._learn_hybrid_high_level
            if self.use_hybrid_high_policy
            else self._learn_continuous_level
        )
        high_action_key = "high_u"
        learn_high(
            batch=batch,
            actors=self.high_actors,
            critics=self.high_critics,
            actor_optimizers=self.high_actor_optimizers,
            critic_optimizers=self.high_critic_optimizers,
            obs_key="high_o",
            next_obs_key="high_o_next",
            state_key="high_s",
            next_state_key="high_s_next",
            action_key=high_action_key,
            avail_key="high_avail_u",
            reward_key="high_r",
            padded_key="high_padded",
            terminated_key="high_terminated",
            active_key="high_agent_active_mask",
            action_dim=self.high_n_actions,
            obs_dim=self.high_obs_shape,
            state_dim=self.high_state_shape,
            energy_margin_key="high_energy_margin",
            energy_order_mask_key="high_energy_order_mask",
            mode_mask_key="high_mode_train_mask",
            duration_key="high_duration",
            intervention_key="high_intervention_mask",
        )

    def _learn_hybrid_high_level(
        self,
        batch,
        actors,
        critics,
        actor_optimizers,
        critic_optimizers,
        obs_key,
        next_obs_key,
        state_key,
        next_state_key,
        action_key,
        avail_key,
        reward_key,
        padded_key,
        terminated_key,
        active_key,
        action_dim,
        obs_dim,
        state_dim,
        energy_margin_key=None,
        energy_order_mask_key=None,
        mode_mask_key=None,
        duration_key=None,
        intervention_key=None,
    ):
        del action_dim, next_obs_key
        del energy_margin_key, energy_order_mask_key
        states = batch[state_key]
        next_states = batch[next_state_key]
        obs = batch[obs_key]
        actions = batch[action_key]
        avail_actions = batch.get(avail_key, None)
        terminated = batch[terminated_key].squeeze(-1)
        mask = 1 - batch[padded_key].squeeze(-1)
        active_mask = batch.get(active_key, None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        mode_train_mask = (
            batch.get(mode_mask_key, None) if mode_mask_key is not None else None
        )
        if mode_train_mask is None:
            mode_train_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        mode_train_mask = mode_train_mask.squeeze(-1)
        intervention_mask = (
            batch.get(intervention_key, None) if intervention_key is not None else None
        )
        if intervention_mask is None:
            intervention_mask = torch.zeros(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        intervention_mask = intervention_mask.squeeze(-1).clamp(0.0, 1.0)
        durations = batch.get(duration_key, None) if duration_key is not None else None
        if durations is None:
            durations = torch.ones(
                (*mask.shape, 1), dtype=mask.dtype, device=self.device
            )
        durations = durations.squeeze(-1)

        rewards = batch[reward_key]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, obs_dim
            )
            actor_states = torch.nan_to_num(actor_states, nan=0.0, posinf=1.0, neginf=-1.0)
            actor_policy_states = actor_states
            agent_actions = actions[:, :, agent_idx, :].reshape(-1, self.high_n_actions)
            agent_actions = torch.nan_to_num(
                agent_actions, nan=0.0, posinf=1.0, neginf=-1.0
            )
            if avail_actions is not None:
                agent_avail_actions = avail_actions[:, :, agent_idx, :].reshape(
                    -1, avail_actions.size(-1)
                )
            else:
                agent_avail_actions = torch.ones(
                    (episode_num * time_len, self.high_mode_n_actions),
                    dtype=actor_states.dtype,
                    device=self.device,
                )
            mode_actions = torch.clamp(
                torch.round(agent_actions[:, 0]).long(),
                0,
                self.high_mode_n_actions - 1,
            )
            continuous_actions = agent_actions[:, 1 : 1 + self.high_continuous_dim]
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            agent_actor_mask = agent_mask * (1.0 - intervention_mask[:, :, agent_idx])
            agent_mode_mask = agent_actor_mask * mode_train_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0
            flat_actor_mask = agent_actor_mask.reshape(-1) > 0
            flat_mode_mask = agent_mode_mask.reshape(-1).clamp(0.0, 1.0)

            with torch.no_grad():
                values = critics[agent_idx](states.reshape(-1, state_dim)).reshape(
                    episode_num, time_len
                )
                next_values = critics[agent_idx](
                    next_states.reshape(-1, state_dim)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_smdp_advantages(
                    agent_rewards,
                    values,
                    next_values,
                    terminated,
                    agent_mask,
                    durations,
                )
                old_mode_dist, old_cont_dist = self._hybrid_high_dist(
                    actors[agent_idx],
                    actor_policy_states,
                    agent_idx=agent_idx,
                    mode_avail=agent_avail_actions,
                )
                old_mode_log_probs = old_mode_dist.log_prob(mode_actions)
                old_cont_log_probs = old_cont_dist.log_prob(
                    continuous_actions
                ).sum(dim=-1)
                old_log_probs = old_cont_log_probs + old_mode_log_probs * flat_mode_mask

            valid_states = flat_states[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            valid_actor_states_pg = actor_policy_states[flat_actor_mask]
            valid_avail_pg = agent_avail_actions[flat_actor_mask]
            valid_modes_pg = mode_actions[flat_actor_mask]
            valid_continuous_pg = continuous_actions[flat_actor_mask]
            valid_advantages_pg = advantages.reshape(-1)[flat_actor_mask]
            valid_old_log_probs_pg = old_log_probs[flat_actor_mask]
            valid_mode_mask_pg = flat_mode_mask[flat_actor_mask]
            if valid_states.size(0) == 0:
                continue

            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        mode_dist, cont_dist, h, _ = self._hybrid_high_dist(
                            actors[agent_idx],
                            valid_actor_states_pg[indices],
                            agent_idx=agent_idx,
                            return_aux=True,
                            mode_avail=valid_avail_pg[indices],
                        )
                        new_mode_log_probs = mode_dist.log_prob(valid_modes_pg[indices])
                        new_cont_log_probs = cont_dist.log_prob(
                            valid_continuous_pg[indices]
                        ).sum(dim=-1)
                        mode_mask = valid_mode_mask_pg[indices]
                        new_log_probs = new_cont_log_probs + new_mode_log_probs * mode_mask
                        log_ratio = torch.clamp(
                            new_log_probs - valid_old_log_probs_pg[indices],
                            -20.0,
                            20.0,
                        )
                        ratio = torch.exp(log_ratio)
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        entropy = (
                            cont_dist.entropy().sum(dim=-1)
                            + mode_dist.entropy() * mode_mask
                        )
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        actor_loss = actor_loss.mean()

                        if not torch.isfinite(actor_loss):
                            continue
                        actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actors[agent_idx].parameters(), self.args.grad_norm_clip
                        )
                        has_bad_grad = any(
                            param.grad is not None
                            and not torch.isfinite(param.grad).all()
                            for param in actors[agent_idx].parameters()
                        )
                        if has_bad_grad:
                            actor_optimizers[agent_idx].zero_grad()
                            continue
                        actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    if not torch.isfinite(critic_loss):
                        continue
                    critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critics[agent_idx].parameters(), self.args.grad_norm_clip
                    )
                    has_bad_grad = any(
                        param.grad is not None
                        and not torch.isfinite(param.grad).all()
                        for param in critics[agent_idx].parameters()
                    )
                    if has_bad_grad:
                        critic_optimizers[agent_idx].zero_grad()
                        continue
                    critic_optimizers[agent_idx].step()

    def _learn_continuous_level(
        self,
        batch,
        actors,
        critics,
        actor_optimizers,
        critic_optimizers,
        obs_key,
        state_key,
        next_state_key,
        action_key,
        reward_key,
        padded_key,
        terminated_key,
        active_key,
        action_dim,
        obs_dim,
        state_dim,
        energy_margin_key=None,
        energy_order_mask_key=None,
        mode_mask_key=None,
        **unused_kwargs,
    ):
        del mode_mask_key
        del energy_margin_key, energy_order_mask_key
        duration_key = unused_kwargs.get("duration_key", None)
        intervention_key = unused_kwargs.get("intervention_key", None)
        correct_action_key = unused_kwargs.get("correct_action_key", None)
        raw_log_prob_key = unused_kwargs.get("raw_log_prob_key", None)
        cbf_flow = self.args.alg == "hmappo_cbf_flow"
        align_coef = float(getattr(self.args, "actor_correct_align_coef", 0.0))
        states = batch[state_key]
        next_states = batch[next_state_key]
        obs = batch[obs_key]
        actions = batch[action_key]
        correct_actions = (
            batch.get(correct_action_key, None) if correct_action_key is not None else None
        )
        raw_log_probs_saved = (
            batch.get(raw_log_prob_key, None) if raw_log_prob_key is not None else None
        )
        terminated = batch[terminated_key].squeeze(-1)
        mask = 1 - batch[padded_key].squeeze(-1)
        active_mask = batch.get(active_key, None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        intervention_mask = (
            batch.get(intervention_key, None) if intervention_key is not None else None
        )
        if intervention_mask is None:
            intervention_mask = torch.zeros(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        intervention_mask = intervention_mask.squeeze(-1).clamp(0.0, 1.0)
        durations = batch.get(duration_key, None) if duration_key is not None else None
        if durations is None:
            durations = torch.ones(
                (*mask.shape, 1), dtype=mask.dtype, device=self.device
            )
        durations = durations.squeeze(-1)

        rewards = batch[reward_key]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)
        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, obs_dim
            )
            agent_actions = actions[:, :, agent_idx, :].reshape(-1, action_dim)
            agent_correct_actions = (
                correct_actions[:, :, agent_idx, :].reshape(-1, action_dim)
                if correct_actions is not None
                else None
            )
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            if cbf_flow:
                agent_actor_mask = agent_mask
            else:
                agent_actor_mask = agent_mask * (1.0 - intervention_mask[:, :, agent_idx])
            flat_agent_mask = agent_mask.reshape(-1) > 0
            flat_actor_mask = agent_actor_mask.reshape(-1) > 0

            with torch.no_grad():
                values = critics[agent_idx](states.reshape(-1, state_dim)).reshape(
                    episode_num, time_len
                )
                next_values = critics[agent_idx](
                    next_states.reshape(-1, state_dim)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_smdp_advantages(
                    agent_rewards,
                    values,
                    next_values,
                    terminated,
                    agent_mask,
                    durations,
                )
                if raw_log_probs_saved is not None:
                    old_log_probs = raw_log_probs_saved[:, :, agent_idx, :].reshape(-1)
                else:
                    old_dist = self._gaussian_dist(actors[agent_idx], actor_states)
                    old_log_probs = old_dist.log_prob(agent_actions).sum(dim=-1)

            valid_states = flat_states[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            valid_actor_states_pg = actor_states[flat_actor_mask]
            valid_actions_pg = agent_actions[flat_actor_mask]
            valid_correct_actions_pg = (
                agent_correct_actions[flat_actor_mask]
                if agent_correct_actions is not None
                else None
            )
            valid_advantages_pg = advantages.reshape(-1)[flat_actor_mask]
            valid_old_log_probs_pg = old_log_probs[flat_actor_mask]
            if valid_states.size(0) == 0:
                continue

            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        dist = self._gaussian_dist(
                            actors[agent_idx], valid_actor_states_pg[indices]
                        )
                        new_log_probs = dist.log_prob(valid_actions_pg[indices]).sum(dim=-1)
                        ratio = torch.exp(new_log_probs - valid_old_log_probs_pg[indices])
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        entropy = dist.entropy().sum(dim=-1)
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        if (
                            cbf_flow
                            and align_coef > 0.0
                            and valid_correct_actions_pg is not None
                        ):
                            mean, _ = actors[agent_idx](valid_actor_states_pg[indices])
                            align_loss = (mean - valid_correct_actions_pg[indices].detach()).pow(2).sum(dim=-1)
                            actor_loss = actor_loss + align_coef * align_loss
                        actor_loss = actor_loss.mean()
                        actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actors[agent_idx].parameters(), self.args.grad_norm_clip
                        )
                        actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critics[agent_idx].parameters(), self.args.grad_norm_clip
                    )
                    critic_optimizers[agent_idx].step()

    def _learn_level(
        self,
        batch,
        actors,
        critics,
        actor_optimizers,
        critic_optimizers,
        obs_key,
        state_key,
        next_state_key,
        action_key,
        avail_key,
        reward_key,
        padded_key,
        terminated_key,
        active_key,
        action_dim,
        obs_dim,
        state_dim,
        guard_key=None,
    ):
        states = batch[state_key]
        next_states = batch[next_state_key]
        obs = batch[obs_key]
        actions = batch[action_key].squeeze(-1)
        avail_actions = batch[avail_key]
        terminated = batch[terminated_key].squeeze(-1)
        mask = 1 - batch[padded_key].squeeze(-1)
        active_mask = batch.get(active_key, None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        guard_applied = batch.get(guard_key, None) if guard_key is not None else None
        if guard_applied is not None:
            guard_applied = guard_applied.squeeze(-1)

        rewards = batch[reward_key]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(
                episode_num * time_len, -1
            )
            agent_actions = actions[:, :, agent_idx].reshape(-1)
            agent_avail = avail_actions[:, :, agent_idx, :].reshape(-1, action_dim)
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0

            with torch.no_grad():
                values = critics[agent_idx](states.reshape(-1, state_dim)).reshape(
                    episode_num, time_len
                )
                next_values = critics[agent_idx](
                    next_states.reshape(-1, state_dim)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_advantages(
                    agent_rewards, values, next_values, terminated, agent_mask
                )
                old_dist = self._masked_categorical(
                    actors[agent_idx](actor_states), agent_avail
                )
                old_log_probs = old_dist.log_prob(agent_actions)

            valid_states = flat_states[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            if guard_applied is not None:
                actor_mask = agent_mask * (1 - guard_applied[:, :, agent_idx])
                flat_actor_mask = actor_mask.reshape(-1) > 0
            else:
                flat_actor_mask = flat_agent_mask

            valid_actor_states_pg = actor_states[flat_actor_mask]
            valid_actions_pg = agent_actions[flat_actor_mask]
            valid_avail_pg = agent_avail[flat_actor_mask]
            valid_advantages_pg = advantages.reshape(-1)[flat_actor_mask]
            valid_old_log_probs_pg = old_log_probs[flat_actor_mask]

            if valid_states.size(0) == 0:
                continue

            batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        dist = self._masked_categorical(
                            actors[agent_idx](valid_actor_states_pg[indices]),
                            valid_avail_pg[indices],
                        )
                        new_log_probs = dist.log_prob(valid_actions_pg[indices])
                        ratio = torch.exp(new_log_probs - valid_old_log_probs_pg[indices])
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        entropy = dist.entropy()
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        actor_loss = actor_loss.mean()

                        actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            actors[agent_idx].parameters(), self.args.grad_norm_clip
                        )
                        actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        critics[agent_idx].parameters(), self.args.grad_norm_clip
                    )
                    critic_optimizers[agent_idx].step()

    def _model_filename(self, level, role, agent_idx, prefix="", model_dir=None):
        model_dir = self.model_dir if model_dir is None else model_dir
        return os.path.join(
            model_dir,
            f"{prefix}{level}_agent_{agent_idx}_{role}_discrete_ppo",
        )

    def _latest_paths(self, level, role, agent_idx, model_dir=None):
        direct = self._model_filename(level, role, agent_idx, model_dir=model_dir)
        if os.path.exists(direct):
            return direct
        candidates = glob.glob(
            self._model_filename(level, role, agent_idx, prefix="*_", model_dir=model_dir)
        )
        if not candidates:
            return None

        def _sort_key(path):
            prefix = os.path.basename(path).split("_", 1)[0]
            return int(prefix) if prefix.isdigit() else -1

        return max(candidates, key=_sort_key)

    def _resolve_checkpoint_dir(self, checkpoint_dir):
        checkpoint_dir = os.path.normpath(str(checkpoint_dir))
        candidates = [
            checkpoint_dir,
            os.path.join(checkpoint_dir, self.args.alg, self.args.map),
            os.path.join(checkpoint_dir, "hmappo", self.args.map),
        ]
        for candidate in candidates:
            if self._latest_paths("low", "actor", 0, model_dir=candidate) is not None:
                return candidate
        raise Exception(f"No pretrained low-level model found under {checkpoint_dir}!")

    def _load_level_models(self, level, actors, critics, model_dir=None):
        def _load_compatible(module, state, label):
            current = module.state_dict()
            compatible = {
                key: value
                for key, value in state.items()
                if key in current and current[key].shape == value.shape
            }
            skipped = sorted(set(state.keys()) - set(compatible.keys()))
            current.update(compatible)
            module.load_state_dict(current)
            if skipped:
                print(
                    f"Skipped {len(skipped)} incompatible tensors while loading {label}: "
                    + ", ".join(skipped[:6])
                    + ("..." if len(skipped) > 6 else "")
                )

        for agent_idx in range(self.n_agents):
            actor_path = self._latest_paths(level, "actor", agent_idx, model_dir=model_dir)
            critic_path = self._latest_paths(level, "critic", agent_idx, model_dir=model_dir)
            if actor_path is None or critic_path is None:
                raise Exception(f"No {level} model for agent {agent_idx}!")
            actor_state = torch.load(actor_path, map_location=self.device)
            critic_state = torch.load(critic_path, map_location=self.device)
            _load_compatible(
                actors[agent_idx],
                actor_state,
                f"{level} actor agent {agent_idx}",
            )
            _load_compatible(
                critics[agent_idx],
                critic_state,
                f"{level} critic agent {agent_idx}",
            )

    def _load_pretrained_low_models(self, checkpoint_dir):
        checkpoint_dir = self._resolve_checkpoint_dir(checkpoint_dir)
        self._load_level_models(
            "low",
            self.low_actors,
            self.low_critics,
            model_dir=checkpoint_dir,
        )

    def _try_load_existing_low_models(self):
        if self._latest_paths("low", "actor", 0) is None:
            return
        self._load_level_models("low", self.low_actors, self.low_critics)

    def _load_models(self):
        levels = [("low", self.low_actors, self.low_critics)]
        if self.high_level_enabled:
            levels.append(("high", self.high_actors, self.high_critics))
        for level, actors, critics in levels:
            self._load_level_models(level, actors, critics)

    def save_model(self, train_step):
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)
        levels = [("low", self.low_actors, self.low_critics)]
        if self.high_level_enabled:
            levels.append(("high", self.high_actors, self.high_critics))
        for level, actors, critics in levels:
            for agent_idx in range(self.n_agents):
                for prefix in (f"{num}_", ""):
                    torch.save(
                        actors[agent_idx].state_dict(),
                        self._model_filename(level, "actor", agent_idx, prefix=prefix),
                    )
                    torch.save(
                        critics[agent_idx].state_dict(),
                        self._model_filename(level, "critic", agent_idx, prefix=prefix),
                    )
