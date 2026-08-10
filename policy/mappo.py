import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


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
        self.n_actions = args.n_actions
        self.action_type = getattr(args, "low_action_type", "discrete")
        self.continuous_action = self.action_type == "continuous"
        self.action_dim = int(getattr(args, "low_action_dim", self.n_actions))
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.device = torch.device(
            f"cuda:{getattr(self.args, 'gpu_id', 0)}"
            if self.args.cuda and torch.cuda.is_available()
            else "cpu"
        )

        actor_hidden_dim = getattr(args, "actor_hidden_dim", 128)
        critic_hidden_dim = getattr(args, "critic_hidden_dim", 128)
        actor_cls = GaussianActorNetwork if self.continuous_action else DiscreteActorNetwork
        actor_action_dim = self.action_dim if self.continuous_action else self.n_actions
        self.actors = nn.ModuleList(
            [
                actor_cls(self.obs_shape, actor_action_dim, actor_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.critics = nn.ModuleList(
            [
                CriticNetwork(self.state_shape, critic_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)

        self.actor_optimizers = [
            torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
            for actor in self.actors
        ]
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
            for critic in self.critics
        ]

        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map
        self.eval_hidden = None

        if self.args.load_model:
            model_paths = self._get_model_paths()
            if model_paths is None:
                raise Exception("No model!")
            for agent_idx, (actor_path, critic_path) in enumerate(model_paths):
                self.actors[agent_idx].load_state_dict(
                    torch.load(actor_path, map_location=self.device)
                )
                self.critics[agent_idx].load_state_dict(
                    torch.load(critic_path, map_location=self.device)
                )

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros((episode_num, self.n_agents, 1), device=self.device)

    def _masked_categorical(self, logits, avail_actions):
        masked_logits = logits.masked_fill(avail_actions <= 0, -1e10)
        return Categorical(logits=masked_logits)

    def _gaussian_dist(self, actor, obs):
        mean, std = actor(obs)
        return Normal(mean, std)

    @torch.no_grad()
    def choose_action(self, observation, agent_idx, avail_actions, evaluate=False):
        obs = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        if self.continuous_action:
            dist = self._gaussian_dist(self.actors[agent_idx], obs)
            action = dist.mean if evaluate else dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            return action.squeeze(0).detach().cpu().numpy().astype("float32")
        avail = torch.tensor(avail_actions, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actors[agent_idx](obs)
        dist = self._masked_categorical(logits, avail)
        if evaluate:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    def _prepare_batch(self, batch):
        tensor_batch = {}
        for key, value in batch.items():
            if key == "u" and not self.continuous_action:
                tensor_batch[key] = torch.tensor(value, dtype=torch.long, device=self.device)
            else:
                tensor_batch[key] = torch.tensor(value, dtype=torch.float32, device=self.device)
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

    def learn(self, batch, max_episode_len, train_step, epsilon):
        batch = self._prepare_batch(batch)
        states = batch["s"]
        next_states = batch["s_next"]
        obs = batch["o"]
        actions = batch["u"] if self.continuous_action else batch["u"].squeeze(-1)
        avail_actions = batch["avail_u"]
        terminated = batch["terminated"].squeeze(-1)
        mask = 1 - batch["padded"].squeeze(-1)
        active_mask = batch.get("agent_active_mask", None)
        if active_mask is None:
            active_mask = torch.ones(
                (*mask.shape, self.n_agents, 1),
                dtype=mask.dtype,
                device=self.device,
            )
        active_mask = active_mask.squeeze(-1)
        guard_applied = batch.get("guard_applied", None)
        if guard_applied is not None:
            guard_applied = guard_applied.squeeze(-1)
        rewards = batch["r"]
        if rewards.size(-1) == 1:
            rewards = rewards.expand(-1, -1, self.n_agents)

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_mask = mask.reshape(-1) > 0

        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(episode_num * time_len, -1)
            if self.continuous_action:
                agent_actions = actions[:, :, agent_idx, :].reshape(-1, self.action_dim)
            else:
                agent_actions = actions[:, :, agent_idx].reshape(-1)
            agent_avail = avail_actions[:, :, agent_idx, :].reshape(-1, self.n_actions)
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0

            with torch.no_grad():
                values = self.critics[agent_idx](states.reshape(-1, self.state_shape)).reshape(
                    episode_num, time_len
                )
                next_values = self.critics[agent_idx](
                    next_states.reshape(-1, self.state_shape)
                ).reshape(episode_num, time_len)
                advantages, returns = self._compute_advantages(
                    agent_rewards, values, next_values, terminated, agent_mask
                )
                if self.continuous_action:
                    old_dist = self._gaussian_dist(self.actors[agent_idx], actor_states)
                    old_log_probs = old_dist.log_prob(agent_actions).sum(dim=-1)
                else:
                    old_dist = self._masked_categorical(
                        self.actors[agent_idx](actor_states), agent_avail
                    )
                    old_log_probs = old_dist.log_prob(agent_actions)

            valid_actor_states = actor_states[flat_agent_mask]
            valid_states = flat_states[flat_agent_mask]
            valid_actions = agent_actions[flat_agent_mask]
            valid_advantages = advantages.reshape(-1)[flat_agent_mask]
            valid_returns = returns.reshape(-1)[flat_agent_mask]
            valid_old_log_probs = old_log_probs[flat_agent_mask]
            if guard_applied is not None:
                actor_mask = agent_mask * (1 - guard_applied[:, :, agent_idx])
                flat_actor_mask = actor_mask.reshape(-1) > 0
            else:
                flat_actor_mask = flat_agent_mask

            valid_actor_states_pg = actor_states[flat_actor_mask]
            valid_actions_pg = agent_actions[flat_actor_mask]
            valid_advantages_pg = advantages.reshape(-1)[flat_actor_mask]
            valid_old_log_probs_pg = old_log_probs[flat_actor_mask]
            if not self.continuous_action:
                valid_avail_pg = agent_avail[flat_actor_mask]

            if valid_states.size(0) == 0:
                continue

            batch_size = min(
                getattr(self.args, "batch_size", 64), valid_states.size(0)
            )

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]

                        if self.continuous_action:
                            dist = self._gaussian_dist(
                                self.actors[agent_idx],
                                valid_actor_states_pg[indices],
                            )
                            new_log_probs = dist.log_prob(
                                valid_actions_pg[indices]
                            ).sum(dim=-1)
                            entropy = dist.entropy().sum(dim=-1)
                        else:
                            dist = self._masked_categorical(
                                self.actors[agent_idx](valid_actor_states_pg[indices]),
                                valid_avail_pg[indices],
                            )
                            new_log_probs = dist.log_prob(valid_actions_pg[indices])
                            entropy = dist.entropy()
                        ratio = torch.exp(new_log_probs - valid_old_log_probs_pg[indices])
                        clipped_ratio = torch.clamp(
                            ratio,
                            1.0 - self.args.clip_param,
                            1.0 + self.args.clip_param,
                        )
                        surrogate_1 = ratio * valid_advantages_pg[indices]
                        surrogate_2 = clipped_ratio * valid_advantages_pg[indices]
                        actor_loss = -torch.min(surrogate_1, surrogate_2)
                        actor_loss -= self.args.entropy_coef * entropy
                        actor_loss = actor_loss.mean()

                        self.actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actors[agent_idx].parameters(),
                            self.args.grad_norm_clip,
                        )
                        self.actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]

                    critic_values = self.critics[agent_idx](valid_states[indices]).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    self.critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[agent_idx].parameters(),
                        self.args.grad_norm_clip,
                    )
                    self.critic_optimizers[agent_idx].step()

    def _actor_filename(self, agent_idx, prefix=""):
        return os.path.join(
            self.model_dir, f"{prefix}agent_{agent_idx}_actor_discrete_ppo"
        )

    def _critic_filename(self, agent_idx, prefix=""):
        return os.path.join(
            self.model_dir, f"{prefix}agent_{agent_idx}_critic_continuous_ppo"
        )

    def _get_model_paths(self):
        latest_paths = []
        for agent_idx in range(self.n_agents):
            actor_path = self._actor_filename(agent_idx)
            critic_path = self._critic_filename(agent_idx)
            if not (os.path.exists(actor_path) and os.path.exists(critic_path)):
                latest_paths = []
                break
            latest_paths.append((actor_path, critic_path))
        if latest_paths:
            return latest_paths

        fallback_paths = []
        for agent_idx in range(self.n_agents):
            actor_candidates = glob.glob(self._actor_filename(agent_idx, prefix="*_" ))
            critic_candidates = glob.glob(self._critic_filename(agent_idx, prefix="*_" ))
            if not actor_candidates or not critic_candidates:
                return None

            def _sort_key(path):
                prefix = os.path.basename(path).split("_", 1)[0]
                return int(prefix) if prefix.isdigit() else -1

            fallback_paths.append(
                (
                    max(actor_candidates, key=_sort_key),
                    max(critic_candidates, key=_sort_key),
                )
            )
        return fallback_paths

    def save_model(self, train_step):
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)
        for agent_idx in range(self.n_agents):
            torch.save(
                self.actors[agent_idx].state_dict(),
                self._actor_filename(agent_idx, prefix=f"{num}_"),
            )
            torch.save(
                self.critics[agent_idx].state_dict(),
                self._critic_filename(agent_idx, prefix=f"{num}_"),
            )
            torch.save(
                self.actors[agent_idx].state_dict(),
                self._actor_filename(agent_idx),
            )
            torch.save(
                self.critics[agent_idx].state_dict(),
                self._critic_filename(agent_idx),
            )
