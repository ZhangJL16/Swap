import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.base_net import RNN


class MACPOCritic(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.fc1 = nn.Linear(input_shape, args.critic_dim)
        self.fc2 = nn.Linear(args.critic_dim, args.critic_dim)
        self.fc3 = nn.Linear(args.critic_dim, args.n_agents)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class MACPO:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        actor_input_shape = self.obs_shape
        critic_input_shape = self.state_shape

        if args.last_action:
            actor_input_shape += self.n_actions
        if args.reuse_network:
            actor_input_shape += self.n_agents

        self.eval_rnn = RNN(actor_input_shape, args)
        self.eval_reward_critic = MACPOCritic(critic_input_shape, args)
        self.eval_cost_critic = MACPOCritic(critic_input_shape, args)

        if self.args.cuda:
            self.eval_rnn.cuda()
            self.eval_reward_critic.cuda()
            self.eval_cost_critic.cuda()

        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map

        self.rnn_parameters = list(self.eval_rnn.parameters())
        self.reward_critic_parameters = list(self.eval_reward_critic.parameters())
        self.cost_critic_parameters = list(self.eval_cost_critic.parameters())

        if args.optimizer == "RMS":
            self.actor_optimizer = torch.optim.RMSprop(
                self.rnn_parameters, lr=args.lr_actor
            )
            self.reward_critic_optimizer = torch.optim.RMSprop(
                self.reward_critic_parameters, lr=args.lr_critic
            )
            self.cost_critic_optimizer = torch.optim.RMSprop(
                self.cost_critic_parameters, lr=args.lr_cost_critic
            )
        else:
            self.actor_optimizer = torch.optim.Adam(
                self.rnn_parameters, lr=args.lr_actor
            )
            self.reward_critic_optimizer = torch.optim.Adam(
                self.reward_critic_parameters, lr=args.lr_critic
            )
            self.cost_critic_optimizer = torch.optim.Adam(
                self.cost_critic_parameters, lr=args.lr_cost_critic
            )

        lambda_device = "cuda" if self.args.cuda else "cpu"
        self.lagrange_multiplier = torch.tensor(
            float(args.lambda_init), dtype=torch.float32, device=lambda_device
        )

        if self.args.load_model:
            model_paths = self._get_model_paths()
            if model_paths is None:
                raise Exception("No model!")
            (
                path_rnn,
                path_reward_critic,
                path_cost_critic,
                path_lambda,
            ) = model_paths
            map_location = (
                f"cuda:{getattr(self.args, 'gpu_id', 0)}" if self.args.cuda else "cpu"
            )
            self.eval_rnn.load_state_dict(
                torch.load(path_rnn, map_location=map_location)
            )
            self.eval_reward_critic.load_state_dict(
                torch.load(path_reward_critic, map_location=map_location)
            )
            self.eval_cost_critic.load_state_dict(
                torch.load(path_cost_critic, map_location=map_location)
            )
            if path_lambda is not None and os.path.exists(path_lambda):
                self.lagrange_multiplier = torch.load(
                    path_lambda, map_location=map_location
                ).float()

        self.eval_hidden = None

    def learn(self, batch, max_episode_len, train_step, epsilon):
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)

        if self.args.cuda:
            for key in batch.keys():
                if isinstance(batch[key], np.ndarray):
                    batch[key] = torch.from_numpy(batch[key]).cuda()
                elif isinstance(batch[key], torch.Tensor) and not batch[key].is_cuda:
                    batch[key] = batch[key].cuda()
            batch["u"] = batch["u"].long()
            for key in batch.keys():
                if key != "u" and batch[key].dtype != torch.float32:
                    batch[key] = batch[key].float()
        else:
            for key in batch.keys():
                if key == "u":
                    batch[key] = torch.tensor(batch[key], dtype=torch.long)
                else:
                    batch[key] = torch.tensor(batch[key], dtype=torch.float32)

        u = batch["u"]
        avail_u = batch["avail_u"]
        mask = (1 - batch["padded"].float()).repeat(1, 1, self.n_agents)
        active_mask = batch.get("agent_active_mask", None)
        if active_mask is not None:
            mask = mask * active_mask.squeeze(-1)
        if mask.sum() <= 0:
            return

        reward_advantages, reward_returns = self._compute_gae(
            batch=batch,
            max_episode_len=max_episode_len,
            mask=mask,
            signal_key="r",
            critic_type="reward",
        )
        cost_advantages, cost_returns = self._compute_gae(
            batch=batch,
            max_episode_len=max_episode_len,
            mask=mask,
            signal_key="c",
            critic_type="cost",
        )

        reward_advantages = reward_advantages.detach()
        reward_returns = reward_returns.detach()
        cost_advantages = cost_advantages.detach()
        cost_returns = cost_returns.detach()

        with torch.no_grad():
            old_action_log_probs = self._get_action_log_probs(
                batch, max_episode_len, u
            )

        episode_cost = (batch["c"] * mask).sum() / mask.sum().clamp(min=1.0)
        lambda_update = self.lagrange_multiplier + self.args.lambda_lr * (
            episode_cost.detach() - self.args.cost_limit
        )
        self.lagrange_multiplier = torch.clamp(lambda_update, min=0.0)

        for _ in range(self.args.ppo_epoch):
            action_log_probs = self._get_action_log_probs(batch, max_episode_len, u)
            dist_entropy = self._get_policy_entropy(batch, max_episode_len, avail_u)

            ratio = torch.exp(action_log_probs - old_action_log_probs)
            clipped_ratio = torch.clamp(
                ratio, 1.0 - self.args.clip_param, 1.0 + self.args.clip_param
            )

            reward_surr1 = ratio * reward_advantages
            reward_surr2 = clipped_ratio * reward_advantages
            reward_surrogate = torch.min(reward_surr1, reward_surr2)

            cost_surr1 = ratio * cost_advantages
            cost_surr2 = clipped_ratio * cost_advantages
            cost_surrogate = torch.max(cost_surr1, cost_surr2)

            combined_objective = reward_surrogate - (
                self.args.cost_coef * self.lagrange_multiplier * cost_surrogate
            )
            policy_loss = -(combined_objective * mask).sum() / mask.sum().clamp(min=1.0)
            dist_entropy = (dist_entropy * mask).sum() / mask.sum().clamp(min=1.0)
            actor_loss = policy_loss - self.args.entropy_coef * dist_entropy

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.rnn_parameters, self.args.grad_norm_clip
            )
            self.actor_optimizer.step()

            reward_values = self._get_values(batch, max_episode_len, critic_type="reward")
            reward_value_loss = F.mse_loss(
                reward_values[mask > 0], reward_returns[mask > 0]
            )
            self.reward_critic_optimizer.zero_grad()
            reward_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.reward_critic_parameters, self.args.grad_norm_clip
            )
            self.reward_critic_optimizer.step()

            cost_values = self._get_values(batch, max_episode_len, critic_type="cost")
            cost_value_loss = F.mse_loss(
                cost_values[mask > 0], cost_returns[mask > 0]
            )
            self.cost_critic_optimizer.zero_grad()
            cost_value_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.cost_critic_parameters, self.args.grad_norm_clip
            )
            self.cost_critic_optimizer.step()

    def _compute_gae(self, batch, max_episode_len, mask, signal_key, critic_type):
        with torch.no_grad():
            values = self._get_values(batch, max_episode_len, critic_type=critic_type)

            signal = batch[signal_key]
            if signal.size(-1) == 1:
                signal = signal.expand(-1, -1, self.n_agents)
            terminated = batch["terminated"].squeeze(-1)

            advantages = torch.zeros_like(signal)
            returns = torch.zeros_like(signal)

            gae = torch.zeros_like(signal[:, 0, :])
            for t in reversed(range(max_episode_len)):
                next_value = (
                    torch.zeros_like(values[:, t, :])
                    if t == max_episode_len - 1
                    else values[:, t + 1, :]
                )
                not_done = (1 - terminated[:, t]).unsqueeze(-1)
                step_mask = mask[:, t, :]
                delta = (
                    signal[:, t, :]
                    + self.args.gamma * next_value * not_done
                    - values[:, t, :]
                )
                gae = delta + self.args.gamma * self.args.gae_lambda * not_done * gae
                gae = gae * step_mask
                advantages[:, t, :] = gae
                returns[:, t, :] = (gae + values[:, t, :]) * step_mask

        valid_advantages = advantages[mask > 0]
        if valid_advantages.numel() > 0:
            advantages = (advantages - valid_advantages.mean()) / (
                valid_advantages.std(unbiased=False) + 1e-8
            )

        return advantages, returns

    def _get_values(self, batch, max_episode_len, critic_type):
        critic = (
            self.eval_reward_critic
            if critic_type == "reward"
            else self.eval_cost_critic
        )
        values = []
        for transition_idx in range(max_episode_len):
            state = batch["s"][:, transition_idx]
            value = critic(state)
            values.append(value)
        return torch.stack(values, dim=1)

    def _get_action_log_probs(self, batch, max_episode_len, actions):
        action_probs = self._get_action_prob(batch, max_episode_len, 0)
        pi_taken = torch.gather(action_probs, dim=3, index=actions).squeeze(3)
        pi_taken = torch.clamp(pi_taken, min=1e-10, max=1.0)
        return torch.log(pi_taken)

    def _get_policy_entropy(self, batch, max_episode_len, avail_actions):
        action_probs = self._get_action_prob(batch, max_episode_len, 0)
        action_probs = action_probs.clone()
        action_probs[avail_actions == 0] = 1e-10
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp(
            min=1e-10
        )
        return -(action_probs * torch.log(action_probs + 1e-10)).sum(dim=-1)

    def _get_actor_inputs(self, batch, transition_idx):
        obs = batch["o"][:, transition_idx]
        u_onehot = batch["u_onehot"][:]
        episode_num = obs.shape[0]
        inputs = [obs]

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])

        if self.args.reuse_network:
            inputs.append(
                torch.eye(self.args.n_agents, device=obs.device)
                .unsqueeze(0)
                .expand(episode_num, -1, -1)
            )

        return torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1
        )

    def _get_action_prob(self, batch, max_episode_len, epsilon):
        episode_num = batch["o"].shape[0]
        avail_actions = batch["avail_u"]
        action_prob = []
        hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim),
            device=batch["o"].device,
        )

        for transition_idx in range(max_episode_len):
            inputs = self._get_actor_inputs(batch, transition_idx)
            outputs, hidden = self.eval_rnn(inputs, hidden)
            outputs = outputs.view(episode_num, self.n_agents, -1)
            prob = torch.nn.functional.softmax(outputs, dim=-1)
            action_prob.append(prob)

        action_prob = torch.stack(action_prob, dim=1)
        action_prob[avail_actions == 0] = 0.0
        action_prob = action_prob / action_prob.sum(dim=-1, keepdim=True).clamp(
            min=1e-10
        )
        action_prob[avail_actions == 0] = 0.0
        return action_prob

    def init_hidden(self, episode_num):
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )

    def _get_model_paths(self):
        latest_rnn = os.path.join(self.model_dir, "rnn_params.pkl")
        latest_reward_critic = os.path.join(self.model_dir, "reward_critic_params.pkl")
        latest_cost_critic = os.path.join(self.model_dir, "cost_critic_params.pkl")
        latest_lambda = os.path.join(self.model_dir, "lagrange_multiplier.pt")
        if (
            os.path.exists(latest_rnn)
            and os.path.exists(latest_reward_critic)
            and os.path.exists(latest_cost_critic)
        ):
            return latest_rnn, latest_reward_critic, latest_cost_critic, latest_lambda

        rnn_candidates = glob.glob(os.path.join(self.model_dir, "*_rnn_params.pkl"))
        reward_critic_candidates = glob.glob(
            os.path.join(self.model_dir, "*_reward_critic_params.pkl")
        )
        cost_critic_candidates = glob.glob(
            os.path.join(self.model_dir, "*_cost_critic_params.pkl")
        )
        if not rnn_candidates or not reward_critic_candidates or not cost_critic_candidates:
            return None

        def _sort_key(path):
            prefix = os.path.basename(path).split("_", 1)[0]
            return int(prefix) if prefix.isdigit() else -1

        path_rnn = max(rnn_candidates, key=_sort_key)
        path_reward_critic = max(reward_critic_candidates, key=_sort_key)
        path_cost_critic = max(cost_critic_candidates, key=_sort_key)
        return path_rnn, path_reward_critic, path_cost_critic, latest_lambda

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(self.eval_rnn.state_dict(), self.model_dir + "/" + num + "_rnn_params.pkl")
        torch.save(
            self.eval_reward_critic.state_dict(),
            self.model_dir + "/" + num + "_reward_critic_params.pkl",
        )
        torch.save(
            self.eval_cost_critic.state_dict(),
            self.model_dir + "/" + num + "_cost_critic_params.pkl",
        )
        torch.save(self.eval_rnn.state_dict(), self.model_dir + "/rnn_params.pkl")
        torch.save(
            self.eval_reward_critic.state_dict(),
            self.model_dir + "/reward_critic_params.pkl",
        )
        torch.save(
            self.eval_cost_critic.state_dict(),
            self.model_dir + "/cost_critic_params.pkl",
        )
        torch.save(
            self.lagrange_multiplier.detach().cpu(),
            self.model_dir + "/lagrange_multiplier.pt",
        )
