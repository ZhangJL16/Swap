import glob
import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


class SharedDiscreteActor(nn.Module):
    def __init__(self, input_dims, n_actions, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.logits = nn.Linear(hidden_dim, n_actions)

    def forward(self, obs):
        return self.logits(self.net(obs))


class SharedGaussianActor(nn.Module):
    def __init__(self, input_dims, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs):
        mean = torch.tanh(self.mean(self.net(obs)))
        log_std = torch.clamp(self.log_std, -5.0, 2.0)
        std = torch.exp(log_std).expand_as(mean)
        return mean, std


class SharedCentralCritic(nn.Module):
    def __init__(self, input_dims, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dims, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.v = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        return self.v(self.net(state))


class OfficialMAPPO:
    """Shared-parameter MAPPO following the official on-policy implementation style.

    The repository's original ``policy/mappo.py`` owns one actor/critic pair per
    agent.  This implementation keeps a single shared actor and a single shared
    centralized critic, then trains them on the flattened
    episode x timestep x agent batch, matching the official MAPPO data layout.
    """

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
        actor_cls = SharedGaussianActor if self.continuous_action else SharedDiscreteActor
        actor_action_dim = self.action_dim if self.continuous_action else self.n_actions
        self.actor = actor_cls(
            self.obs_shape,
            actor_action_dim,
            actor_hidden_dim,
        ).to(self.device)
        self.critic = SharedCentralCritic(
            self.state_shape,
            critic_hidden_dim,
        ).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=args.lr_critic,
        )

        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map
        self.eval_hidden = None
        self.last_train_metrics = {}

        if self.args.load_model:
            model_paths = self._get_model_paths()
            if model_paths is None:
                raise Exception("No official MAPPO shared model!")
            actor_path, critic_path = model_paths
            self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
            self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, 1),
            device=self.device,
        )

    def _masked_categorical(self, logits, avail_actions):
        masked_logits = logits.masked_fill(avail_actions <= 0, -1e10)
        return Categorical(logits=masked_logits)

    def _gaussian_dist(self, obs):
        mean, std = self.actor(obs)
        return Normal(mean, std)

    @torch.no_grad()
    def choose_action(self, observation, agent_idx, avail_actions, evaluate=False):
        del agent_idx
        obs = torch.tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        if self.continuous_action:
            dist = self._gaussian_dist(obs)
            action = dist.mean if evaluate else dist.sample()
            action = torch.clamp(action, -1.0, 1.0)
            return action.squeeze(0).detach().cpu().numpy().astype("float32")
        avail = torch.tensor(
            avail_actions,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        dist = self._masked_categorical(self.actor(obs), avail)
        if evaluate:
            action = torch.argmax(dist.logits, dim=-1)
        else:
            action = dist.sample()
        return int(action.item())

    def _prepare_batch(self, batch):
        tensor_batch = {}
        for key, value in batch.items():
            if key == "u" and not self.continuous_action:
                tensor_batch[key] = torch.tensor(
                    value,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                tensor_batch[key] = torch.tensor(
                    value,
                    dtype=torch.float32,
                    device=self.device,
                )
        return tensor_batch

    def _compute_advantages(self, rewards, values, next_values, terminated, mask):
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(
            rewards.size(0),
            rewards.size(2),
            device=self.device,
        )
        terminated = terminated.unsqueeze(-1)
        for t in reversed(range(rewards.size(1))):
            not_done = 1.0 - terminated[:, t]
            delta = (
                rewards[:, t]
                + self.args.gamma * next_values[:, t] * not_done
                - values[:, t]
            )
            gae = (
                delta
                + self.args.gamma * self.args.gae_lambda * not_done * gae
            )
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
        del max_episode_len, train_step, epsilon
        batch = self._prepare_batch(batch)
        states = batch["s"]
        next_states = batch["s_next"]
        obs = batch["o"]
        actions = batch["u"] if self.continuous_action else batch["u"].squeeze(-1)
        avail_actions = batch["avail_u"]
        terminated = batch["terminated"].squeeze(-1)
        mask = 1.0 - batch["padded"].squeeze(-1)
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
        agent_mask = mask.unsqueeze(-1) * active_mask

        flat_obs = obs.reshape(episode_num * time_len * self.n_agents, -1)
        if self.continuous_action:
            flat_actions = actions.reshape(-1, self.action_dim)
        else:
            flat_actions = actions.reshape(-1)
        flat_avail = avail_actions.reshape(-1, self.n_actions)
        flat_states = (
            states.unsqueeze(2)
            .expand(-1, -1, self.n_agents, -1)
            .reshape(-1, self.state_shape)
        )
        flat_next_states = (
            next_states.unsqueeze(2)
            .expand(-1, -1, self.n_agents, -1)
            .reshape(-1, self.state_shape)
        )

        with torch.no_grad():
            values = self.critic(flat_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            next_values = self.critic(flat_next_states).reshape(
                episode_num,
                time_len,
                self.n_agents,
            )
            advantages, returns = self._compute_advantages(
                rewards,
                values,
                next_values,
                terminated,
                agent_mask,
            )
            if self.continuous_action:
                old_dist = self._gaussian_dist(flat_obs)
                old_log_probs = old_dist.log_prob(flat_actions).sum(dim=-1)
            else:
                old_dist = self._masked_categorical(self.actor(flat_obs), flat_avail)
                old_log_probs = old_dist.log_prob(flat_actions)

        flat_agent_mask = agent_mask.reshape(-1) > 0
        if guard_applied is not None:
            actor_mask = agent_mask * (1.0 - guard_applied)
        else:
            actor_mask = agent_mask
        flat_actor_mask = actor_mask.reshape(-1) > 0

        valid_states = flat_states[flat_agent_mask]
        valid_returns = returns.reshape(-1)[flat_agent_mask]
        valid_old_values = values.reshape(-1)[flat_agent_mask]

        valid_actor_obs = flat_obs[flat_actor_mask]
        valid_actor_actions = flat_actions[flat_actor_mask]
        valid_actor_advantages = advantages.reshape(-1)[flat_actor_mask]
        valid_actor_old_log_probs = old_log_probs[flat_actor_mask]
        if not self.continuous_action:
            valid_actor_avail = flat_avail[flat_actor_mask]

        if valid_states.size(0) == 0:
            self.last_train_metrics = {}
            return

        batch_size = min(getattr(self.args, "batch_size", 64), valid_states.size(0))
        last_actor_loss = None
        last_critic_loss = None
        last_entropy = None

        for _ in range(self.args.ppo_epoch):
            if valid_actor_obs.size(0) > 0:
                actor_batch_size = min(batch_size, valid_actor_obs.size(0))
                actor_permutation = torch.randperm(
                    valid_actor_obs.size(0),
                    device=self.device,
                )
                for start in range(0, valid_actor_obs.size(0), actor_batch_size):
                    indices = actor_permutation[start : start + actor_batch_size]
                    if self.continuous_action:
                        dist = self._gaussian_dist(valid_actor_obs[indices])
                        new_log_probs = dist.log_prob(
                            valid_actor_actions[indices]
                        ).sum(dim=-1)
                        entropy = dist.entropy().sum(dim=-1)
                    else:
                        dist = self._masked_categorical(
                            self.actor(valid_actor_obs[indices]),
                            valid_actor_avail[indices],
                        )
                        new_log_probs = dist.log_prob(valid_actor_actions[indices])
                        entropy = dist.entropy()
                    ratio = torch.exp(
                        new_log_probs - valid_actor_old_log_probs[indices]
                    )
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - self.args.clip_param,
                        1.0 + self.args.clip_param,
                    )
                    surrogate_1 = ratio * valid_actor_advantages[indices]
                    surrogate_2 = clipped_ratio * valid_actor_advantages[indices]
                    actor_loss = -torch.min(surrogate_1, surrogate_2)
                    actor_loss -= self.args.entropy_coef * entropy
                    actor_loss = actor_loss.mean()

                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.actor.parameters(),
                        self.args.grad_norm_clip,
                    )
                    self.actor_optimizer.step()
                    last_actor_loss = actor_loss.detach()
                    last_entropy = entropy.mean().detach()

            critic_permutation = torch.randperm(valid_states.size(0), device=self.device)
            for start in range(0, valid_states.size(0), batch_size):
                indices = critic_permutation[start : start + batch_size]
                critic_values = self.critic(valid_states[indices]).squeeze(-1)
                if getattr(self.args, "use_clipped_value_loss", True):
                    value_pred_clipped = valid_old_values[indices] + (
                        critic_values - valid_old_values[indices]
                    ).clamp(-self.args.clip_param, self.args.clip_param)
                    value_losses = (critic_values - valid_returns[indices]).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - valid_returns[indices]
                    ).pow(2)
                    critic_loss = 0.5 * torch.max(
                        value_losses,
                        value_losses_clipped,
                    ).mean()
                else:
                    critic_loss = 0.5 * (
                        critic_values - valid_returns[indices]
                    ).pow(2).mean()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    self.args.grad_norm_clip,
                )
                self.critic_optimizer.step()
                last_critic_loss = critic_loss.detach()

        self.last_train_metrics = {}
        if last_actor_loss is not None:
            self.last_train_metrics["official_mappo_actor_loss"] = float(
                last_actor_loss.item()
            )
        if last_critic_loss is not None:
            self.last_train_metrics["official_mappo_critic_loss"] = float(
                last_critic_loss.item()
            )
        if last_entropy is not None:
            self.last_train_metrics["official_mappo_entropy"] = float(
                last_entropy.item()
            )

    def _actor_filename(self, prefix=""):
        return os.path.join(self.model_dir, f"{prefix}shared_actor_ppo")

    def _critic_filename(self, prefix=""):
        return os.path.join(self.model_dir, f"{prefix}shared_critic_ppo")

    def _get_model_paths(self):
        actor_path = self._actor_filename()
        critic_path = self._critic_filename()
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            return actor_path, critic_path

        actor_candidates = glob.glob(self._actor_filename(prefix="*_"))
        critic_candidates = glob.glob(self._critic_filename(prefix="*_"))
        if not actor_candidates or not critic_candidates:
            return None

        def _sort_key(path):
            prefix = os.path.basename(path).split("_", 1)[0]
            return int(prefix) if prefix.isdigit() else -1

        return (
            max(actor_candidates, key=_sort_key),
            max(critic_candidates, key=_sort_key),
        )

    def save_model(self, train_step):
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(self.actor.state_dict(), self._actor_filename(prefix=f"{num}_"))
        torch.save(self.critic.state_dict(), self._critic_filename(prefix=f"{num}_"))
        torch.save(self.actor.state_dict(), self._actor_filename())
        torch.save(self.critic.state_dict(), self._critic_filename())
