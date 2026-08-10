#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random

import numpy as np
import torch as T
from torch import nn

from common.Experience_replay import ReplayBuffer


class CommunicationCNet(nn.Module):
    def __init__(
        self, input_dim, msg_dim, n_share_levels, lr, chkpt_dir, use_cuda=False, gpu_id=0
    ):
        super().__init__()
        self.chkpt_dir = chkpt_dir
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.ratio_head = nn.Linear(128, n_share_levels)
        self.device = T.device(
            f"cuda:{int(gpu_id)}" if use_cuda and T.cuda.is_available() else "cpu"
        )
        self.to(self.device)
        self.optimizer = T.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = T.relu(self.fc1(x))
        h = T.relu(self.fc2(h))
        return self.ratio_head(h)

    def save_checkpoint(self, save_file):
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        T.save(self.state_dict(), save_file)

    def load_checkpoint(self, load_file):
        self.load_state_dict(T.load(load_file, map_location=self.device))


class DQN(object):
    """Paper-style shared C-network.

    Input: sender local observation.
    Output:
    - discrete communication sharing level
    """

    def __init__(
        self,
        state_dim,
        n_actions,
        max_keep_dim=None,
        gamma=0.95,
        lr=0.0001,
        batch_size=16,
        INITIAL_EPSILON=0.2,
        FINAL_EPSILON=0.0001,
        max_episode=1e6,
        replace_target_cnt=1000,
        chkpt_dir="./model/",
        buffer_size=1e3,
        tau=0.01,
        use_cuda=False,
        gpu_id=0,
    ):
        del tau
        self.gamma = gamma
        self.lr = lr
        self.use_cuda = bool(use_cuda)
        self.gpu_id = int(gpu_id)
        self.msg_dim = int(n_actions)
        if max_keep_dim is None:
            self.max_keep_dim = self.msg_dim
        else:
            self.max_keep_dim = int(np.clip(int(max_keep_dim), 0, self.msg_dim))
        self.n_share_levels = self.max_keep_dim + 1
        self.state_dim = int(state_dim)
        self.batch_size = batch_size
        self.INITIAL_EPSILON = INITIAL_EPSILON
        self.FINAL_EPSILON = FINAL_EPSILON
        self.max_episode = max_episode
        self.replace_target_cnt = replace_target_cnt
        self.chkpt_dir = chkpt_dir
        self.buffer_size = int(buffer_size)
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(self.buffer_size)
        self.risk_memory = ReplayBuffer(self.buffer_size)

        self.q_eval = CommunicationCNet(
            input_dim=self.state_dim,
            msg_dim=self.msg_dim,
            n_share_levels=self.n_share_levels,
            lr=self.lr,
            chkpt_dir=self.chkpt_dir,
            use_cuda=self.use_cuda,
            gpu_id=self.gpu_id,
        )
        self.q_next = CommunicationCNet(
            input_dim=self.state_dim,
            msg_dim=self.msg_dim,
            n_share_levels=self.n_share_levels,
            lr=self.lr,
            chkpt_dir=self.chkpt_dir,
            use_cuda=self.use_cuda,
            gpu_id=self.gpu_id,
        )
        self.q_next.load_state_dict(self.q_eval.state_dict())

        self.td_loss = nn.MSELoss()
        self.decrement_epsilon(0)

    def choose_action(self, observation, evaluate=False, return_info=False):
        state = T.tensor(observation, dtype=T.float32, device=self.q_eval.device).reshape(
            1, -1
        )
        with T.no_grad():
            ratio_values = self.q_eval.forward(state)
        if evaluate or np.random.random() > self.epsilon:
            action = int(T.argmax(ratio_values, dim=1).item())
        else:
            action = random.randint(0, self.n_share_levels - 1)
        self.decrement_epsilon(self.learn_step_counter)
        if not return_info:
            return action
        return {
            "action": action,
            "share_ratio": float(action / max(self.msg_dim, 1)),
        }

    def store_transition(self, state, action, reward, state_, warning=0):
        if warning == 0:
            self.memory.add(state, action, reward, state_)
        else:
            self.risk_memory.add(state, action, reward, state_)

    def sample_memory(self):
        state, action, reward, new_state = self.memory.sample_batch(self.batch_size)
        r_state, r_action, r_reward, r_new_state = self.risk_memory.sample_batch(
            self.batch_size
        )
        state.extend(r_state)
        action.extend(r_action)
        reward.extend(r_reward)
        new_state.extend(r_new_state)

        if not state:
            return None

        states = T.tensor(np.array(state), dtype=T.float32, device=self.q_eval.device)
        actions = T.tensor(np.array(action), dtype=T.long, device=self.q_eval.device)
        rewards = T.tensor(np.array(reward), dtype=T.float32, device=self.q_eval.device)
        states_ = T.tensor(np.array(new_state), dtype=T.float32, device=self.q_eval.device)
        return states, actions, rewards, states_

    def replace_target_network(self):
        self.learn_step_counter += 1
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self, episode):
        self.epsilon = max(
            self.FINAL_EPSILON,
            self.INITIAL_EPSILON
            - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode,
        )

    def save_models(self, save_file):
        self.q_eval.save_checkpoint(save_file + "_eval")
        self.q_next.save_checkpoint(save_file + "_next")

    def load_models(self, load_file):
        self.q_eval.load_checkpoint(load_file + "_eval")
        self.q_next.load_checkpoint(load_file + "_next")

    def learn(self):
        sampled = self.sample_memory()
        if sampled is None:
            return
        states, actions, rewards, states_ = sampled
        if states.numel() == 0:
            return

        ratio_eval = self.q_eval.forward(states)
        with T.no_grad():
            ratio_next = self.q_next.forward(states_)
            max_q_value = T.max(ratio_next, dim=1)[0]
            q_target = rewards.view(max_q_value.size()) + self.gamma * max_q_value

        q_eval_replaced = ratio_eval.clone()
        q_eval_replaced[T.arange(ratio_eval.shape[0]), actions] = q_target
        loss = self.td_loss(q_eval_replaced, ratio_eval)

        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()

        self.replace_target_network()
