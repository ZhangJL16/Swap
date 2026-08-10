import glob
import os

import torch

from policy.mappo import CriticNetwork, MAPPO


class MAPPOLagrangian(MAPPO):
    """MAPPO with a Lagrangian penalty on environment cost signals."""

    def __init__(self, args):
        super().__init__(args)

        critic_hidden_dim = getattr(args, "critic_hidden_dim", 128)
        self.cost_critics = torch.nn.ModuleList(
            [
                CriticNetwork(self.state_shape, critic_hidden_dim)
                for _ in range(self.n_agents)
            ]
        ).to(self.device)
        self.cost_critic_optimizers = [
            torch.optim.Adam(critic.parameters(), lr=args.lr_cost_critic)
            for critic in self.cost_critics
        ]
        self.lagrange_multiplier = torch.tensor(
            float(args.lambda_init),
            dtype=torch.float32,
            device=self.device,
        )

        if self.args.load_model:
            self._load_cost_models()

    def _prepare_costs(self, batch, rewards):
        costs = batch.get("c", None)
        if costs is None:
            costs = batch.get("warning_signal", None)
        if costs is None:
            return torch.zeros_like(rewards)
        if costs.dim() == 3:
            return costs
        if costs.dim() == 4 and costs.size(-1) == 1:
            return costs.squeeze(-1)
        if costs.size(-1) == 1:
            return costs.expand(-1, -1, self.n_agents)
        return costs

    def _update_lagrange_multiplier(self, costs, mask, active_mask):
        agent_mask = mask.unsqueeze(-1) * active_mask
        mean_cost = (costs * agent_mask).sum() / agent_mask.sum().clamp(min=1.0)
        updated = self.lagrange_multiplier + self.args.lambda_lr * (
            mean_cost.detach() - self.args.cost_limit
        )
        self.lagrange_multiplier = torch.clamp(updated, min=0.0)

    def learn(self, batch, max_episode_len, train_step, epsilon):
        batch = self._prepare_batch(batch)
        states = batch["s"]
        next_states = batch["s_next"]
        obs = batch["o"]
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
        costs = self._prepare_costs(batch, rewards)
        self._update_lagrange_multiplier(costs, mask, active_mask)

        episode_num = states.size(0)
        time_len = states.size(1)
        flat_states = states.reshape(episode_num * time_len, -1)

        for agent_idx in range(self.n_agents):
            actor_states = obs[:, :, agent_idx, :].reshape(episode_num * time_len, -1)
            agent_actions = actions[:, :, agent_idx].reshape(-1)
            agent_avail = avail_actions[:, :, agent_idx, :].reshape(-1, self.n_actions)
            agent_rewards = rewards[:, :, agent_idx]
            agent_costs = costs[:, :, agent_idx]
            agent_mask = mask * active_mask[:, :, agent_idx]
            flat_agent_mask = agent_mask.reshape(-1) > 0

            with torch.no_grad():
                values = self.critics[agent_idx](
                    states.reshape(-1, self.state_shape)
                ).reshape(episode_num, time_len)
                next_values = self.critics[agent_idx](
                    next_states.reshape(-1, self.state_shape)
                ).reshape(episode_num, time_len)
                reward_advantages, reward_returns = self._compute_advantages(
                    agent_rewards, values, next_values, terminated, agent_mask
                )
                cost_values = self.cost_critics[agent_idx](
                    states.reshape(-1, self.state_shape)
                ).reshape(episode_num, time_len)
                next_cost_values = self.cost_critics[agent_idx](
                    next_states.reshape(-1, self.state_shape)
                ).reshape(episode_num, time_len)
                cost_advantages, cost_returns = self._compute_advantages(
                    agent_costs,
                    cost_values,
                    next_cost_values,
                    terminated,
                    agent_mask,
                )
                old_dist = self._masked_categorical(
                    self.actors[agent_idx](actor_states), agent_avail
                )
                old_log_probs = old_dist.log_prob(agent_actions)

            constrained_advantages = reward_advantages - (
                self.args.cost_coef * self.lagrange_multiplier.detach() * cost_advantages
            )
            valid_actor_states = actor_states[flat_agent_mask]
            valid_states = flat_states[flat_agent_mask]
            valid_actions = agent_actions[flat_agent_mask]
            valid_avail = agent_avail[flat_agent_mask]
            valid_reward_returns = reward_returns.reshape(-1)[flat_agent_mask]
            valid_cost_returns = cost_returns.reshape(-1)[flat_agent_mask]

            if guard_applied is not None:
                actor_mask = agent_mask * (1 - guard_applied[:, :, agent_idx])
                flat_actor_mask = actor_mask.reshape(-1) > 0
            else:
                flat_actor_mask = flat_agent_mask

            valid_actor_states_pg = actor_states[flat_actor_mask]
            valid_actions_pg = agent_actions[flat_actor_mask]
            valid_avail_pg = agent_avail[flat_actor_mask]
            valid_advantages_pg = constrained_advantages.reshape(-1)[flat_actor_mask]
            valid_old_log_probs_pg = old_log_probs[flat_actor_mask]

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
                    valid_states.size(0), device=self.device
                )
                for start in range(0, valid_states.size(0), batch_size):
                    indices = critic_permutation[start : start + batch_size]

                    critic_values = self.critics[agent_idx](
                        valid_states[indices]
                    ).squeeze(-1)
                    critic_loss = (
                        critic_values - valid_reward_returns[indices]
                    ).pow(2).mean()
                    self.critic_optimizers[agent_idx].zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.critics[agent_idx].parameters(),
                        self.args.grad_norm_clip,
                    )
                    self.critic_optimizers[agent_idx].step()

                    cost_values = self.cost_critics[agent_idx](
                        valid_states[indices]
                    ).squeeze(-1)
                    cost_loss = (
                        cost_values - valid_cost_returns[indices]
                    ).pow(2).mean()
                    self.cost_critic_optimizers[agent_idx].zero_grad()
                    cost_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.cost_critics[agent_idx].parameters(),
                        self.args.grad_norm_clip,
                    )
                    self.cost_critic_optimizers[agent_idx].step()

    def _cost_critic_filename(self, agent_idx, prefix=""):
        return os.path.join(
            self.model_dir, f"{prefix}agent_{agent_idx}_cost_critic_lagrangian"
        )

    def _lambda_filename(self):
        return os.path.join(self.model_dir, "lagrange_multiplier.pt")

    def _load_cost_models(self):
        for agent_idx in range(self.n_agents):
            path = self._cost_critic_filename(agent_idx)
            if not os.path.exists(path):
                candidates = glob.glob(
                    self._cost_critic_filename(agent_idx, prefix="*_")
                )
                if not candidates:
                    continue

                def _sort_key(candidate):
                    prefix = os.path.basename(candidate).split("_", 1)[0]
                    return int(prefix) if prefix.isdigit() else -1

                path = max(candidates, key=_sort_key)
            self.cost_critics[agent_idx].load_state_dict(
                torch.load(path, map_location=self.device)
            )
        lambda_path = self._lambda_filename()
        if os.path.exists(lambda_path):
            self.lagrange_multiplier = torch.load(
                lambda_path, map_location=self.device
            ).float()

    def save_model(self, train_step):
        super().save_model(train_step)
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)
        for agent_idx in range(self.n_agents):
            torch.save(
                self.cost_critics[agent_idx].state_dict(),
                self._cost_critic_filename(agent_idx, prefix=f"{num}_"),
            )
            torch.save(
                self.cost_critics[agent_idx].state_dict(),
                self._cost_critic_filename(agent_idx),
            )
        torch.save(self.lagrange_multiplier.detach(), self._lambda_filename())
