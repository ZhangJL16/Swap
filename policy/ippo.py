import os

import torch

from policy.mappo import CriticNetwork, MAPPO


class IPPO(MAPPO):
    """Independent PPO: per-agent actor and local-observation value critic."""

    def __init__(self, args):
        super().__init__(args)
        critic_hidden_dim = getattr(args, "critic_hidden_dim", 128)
        self.critics = torch.nn.ModuleList(
            [
                CriticNetwork(self.obs_shape, critic_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=args.lr_critic)
            for critic in self.critics
        ]
        if self.args.load_model:
            self._load_local_critics()

    def learn(self, batch, max_episode_len, train_step, epsilon):
        batch = self._prepare_batch(batch)
        obs = batch["o"]
        next_obs = batch["o_next"]
        actions = batch["u"].squeeze(-1)
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

        episode_num = obs.size(0)
        time_len = obs.size(1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(episode_num * time_len, -1)
            next_actor_states = next_obs[:, :, agent_idx, :].reshape(
                episode_num * time_len,
                -1,
            )
            agent_actions = actions[:, :, agent_idx].reshape(-1)
            agent_avail = avail_actions[:, :, agent_idx, :].reshape(-1, self.n_actions)
            agent_rewards = rewards[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0

            with torch.no_grad():
                values = self.critics[agent_idx](actor_states).reshape(
                    episode_num,
                    time_len,
                )
                next_values = self.critics[agent_idx](next_actor_states).reshape(
                    episode_num,
                    time_len,
                )
                advantages, returns = self._compute_advantages(
                    agent_rewards,
                    values,
                    next_values,
                    terminated,
                    agent_mask,
                )
                old_dist = self._masked_categorical(
                    self.actors[agent_idx](actor_states), agent_avail
                )
                old_log_probs = old_dist.log_prob(agent_actions)

            valid_actor_states = actor_states[flat_agent_mask]
            valid_actions = agent_actions[flat_agent_mask]
            valid_avail = agent_avail[flat_agent_mask]
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

            if valid_actor_states.size(0) == 0:
                continue

            batch_size = min(
                getattr(self.args, "batch_size", 64), valid_actor_states.size(0)
            )

            for _ in range(self.args.ppo_epoch):
                if valid_actor_states_pg.size(0) > 0:
                    actor_batch_size = min(batch_size, valid_actor_states_pg.size(0))
                    actor_permutation = torch.randperm(
                        valid_actor_states_pg.size(0), device=self.device
                    )
                    for start in range(0, valid_actor_states_pg.size(0), actor_batch_size):
                        indices = actor_permutation[start : start + actor_batch_size]
                        dist = self._masked_categorical(
                            self.actors[agent_idx](valid_actor_states_pg[indices]),
                            valid_avail_pg[indices],
                        )
                        new_log_probs = dist.log_prob(valid_actions_pg[indices])
                        ratio = torch.exp(
                            new_log_probs - valid_old_log_probs_pg[indices]
                        )
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

                        self.actor_optimizers[agent_idx].zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            self.actors[agent_idx].parameters(),
                            self.args.grad_norm_clip,
                        )
                        self.actor_optimizers[agent_idx].step()

                critic_permutation = torch.randperm(
                    valid_actor_states.size(0), device=self.device
                )
                for start in range(0, valid_actor_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]
                    critic_values = self.critics[agent_idx](
                        valid_actor_states[indices]
                    ).squeeze(-1)
                    critic_loss = (critic_values - valid_returns[indices]).pow(2).mean()
                    self.critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[agent_idx].parameters(),
                        self.args.grad_norm_clip,
                    )
                    self.critic_optimizers[agent_idx].step()

    def _critic_filename(self, agent_idx, prefix=""):
        return os.path.join(
            self.model_dir, f"{prefix}agent_{agent_idx}_critic_ippo_obs"
        )

    def _load_local_critics(self):
        paths = self._get_model_paths()
        if paths is None:
            return
        for agent_idx, (_, critic_path) in enumerate(paths):
            self.critics[agent_idx].load_state_dict(
                torch.load(critic_path, map_location=self.device)
            )
