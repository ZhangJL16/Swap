import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from policy.actor_critic import Actor


class TwinCritic(nn.Module):
    """Centralized twin-Q critic used by MATD3."""

    def __init__(self, args):
        super().__init__()
        self.max_action = args.high_action
        input_dim = sum(args.obs_shape) + sum(args.action_shape)
        self.q1 = self._make_q_network(input_dim)
        self.q2 = self._make_q_network(input_dim)

    @staticmethod
    def _make_q_network(input_dim):
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def _merge_inputs(self, state, action):
        state = torch.cat(state, dim=1)
        scaled_action = [agent_action / self.max_action for agent_action in action]
        action = torch.cat(scaled_action, dim=1)
        return torch.cat([state, action], dim=1)

    def forward(self, state, action):
        x = self._merge_inputs(state, action)
        return self.q1(x), self.q2(x)

    def q1_value(self, state, action):
        return self.q1(self._merge_inputs(state, action))


class MATD3:
    """MATD3 implementation compatible with the repository's RGMComm runner."""

    def __init__(self, args, agent_id):
        self.args = args
        self.agent_id = agent_id
        self.train_step = 0

        self.actor_network = Actor(args, agent_id)
        self.actor_target_network = Actor(args, agent_id)
        self.critic_network = TwinCritic(args)
        self.critic_target_network = TwinCritic(args)

        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())

        self.actor_optim = torch.optim.Adam(
            self.actor_network.parameters(), lr=self.args.lr_actor
        )
        self.critic_optim = torch.optim.Adam(
            self.critic_network.parameters(), lr=self.args.lr_critic
        )

        self.model_path = os.path.join(
            self.args.save_dir, self.args.scenario_name, f"agent_{agent_id}"
        )
        os.makedirs(self.model_path, exist_ok=True)
        self.load_path = os.path.join(
            self.args.load_dir, self.args.scenario_name, f"agent_{agent_id}"
        )
        self._load_if_available()

    def _load_if_available(self):
        actor_path = os.path.join(self.load_path, "3999_actor_params.pkl")
        critic_path = os.path.join(self.load_path, "3999_critic_params.pkl")
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            self.actor_network.load_state_dict(torch.load(actor_path))
            self.critic_network.load_state_dict(torch.load(critic_path))
            self.actor_target_network.load_state_dict(self.actor_network.state_dict())
            self.critic_target_network.load_state_dict(self.critic_network.state_dict())
            print(
                f"Agent {self.agent_id} successfully loaded MATD3 actor: {actor_path}"
            )
            print(
                f"Agent {self.agent_id} successfully loaded MATD3 critic: {critic_path}"
            )

    def _soft_update_target_network(self, update_actor=True):
        if update_actor:
            for target_param, param in zip(
                self.actor_target_network.parameters(), self.actor_network.parameters()
            ):
                target_param.data.copy_(
                    (1 - self.args.tau) * target_param.data + self.args.tau * param.data
                )

        for target_param, param in zip(
            self.critic_target_network.parameters(), self.critic_network.parameters()
        ):
            target_param.data.copy_(
                (1 - self.args.tau) * target_param.data + self.args.tau * param.data
            )

    def _target_action(self, actor, obs):
        action = actor(obs)
        noise_std = float(getattr(self.args, "matd3_target_noise", 0.2))
        noise_clip = float(getattr(self.args, "matd3_target_noise_clip", 0.5))
        if noise_std > 0:
            noise = torch.randn_like(action) * noise_std * self.args.high_action
            noise = torch.clamp(
                noise,
                -noise_clip * self.args.high_action,
                noise_clip * self.args.high_action,
            )
            action = action + noise
        return torch.clamp(action, -self.args.high_action, self.args.high_action)

    def train(self, transitions, other_agents):
        transitions = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in transitions.items()
        }
        rewards = transitions[f"r_{self.agent_id}"]
        obs, actions, next_obs = [], [], []
        for agent_id in range(self.args.n_agents):
            obs.append(transitions[f"o_{agent_id}"])
            actions.append(transitions[f"u_{agent_id}"])
            next_obs.append(transitions[f"o_next_{agent_id}"])

        next_actions = []
        with torch.no_grad():
            other_idx = 0
            for agent_id in range(self.args.n_agents):
                if agent_id == self.agent_id:
                    next_actions.append(
                        self._target_action(
                            self.actor_target_network, next_obs[agent_id]
                        )
                    )
                else:
                    next_actions.append(
                        self._target_action(
                            other_agents[other_idx].policy.actor_target_network,
                            next_obs[agent_id],
                        )
                    )
                    other_idx += 1
            target_q1, target_q2 = self.critic_target_network(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards.unsqueeze(1) + self.args.gamma * target_q

        current_q1, current_q2 = self.critic_network(obs, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(
            current_q2, target_q
        )

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        policy_delay = max(1, int(getattr(self.args, "matd3_policy_delay", 2)))
        update_actor = self.train_step % policy_delay == 0
        if update_actor:
            actor_actions = list(actions)
            actor_actions[self.agent_id] = self.actor_network(obs[self.agent_id])
            actor_loss = -self.critic_network.q1_value(obs, actor_actions).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

        self._soft_update_target_network(update_actor=update_actor)
        if self.train_step > 0 and self.train_step % self.args.save_rate == 0:
            self.save_model(self.train_step)
        self.train_step += 1

    def save_model(self, train_step):
        num = str(train_step // self.args.save_rate)
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(
            self.actor_network.state_dict(),
            os.path.join(self.model_path, f"{num}_actor_params.pkl"),
        )
        torch.save(
            self.critic_network.state_dict(),
            os.path.join(self.model_path, f"{num}_critic_params.pkl"),
        )

    def trainit(self, transitions):
        transitions = {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in transitions.items()
        }
        obs, actions = [], []
        for agent_id in range(self.args.n_agents):
            obs.append(transitions[f"o_{agent_id}"])
            actions.append(transitions[f"u_{agent_id}"])
        q_values = self.critic_network.q1_value(obs, actions).detach().cpu().numpy()
        return [[row.tolist(), q_values[idx].tolist()] for idx, row in enumerate(obs[self.agent_id])]
