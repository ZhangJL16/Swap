import copy
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DecayThenFlatSchedule:
    def __init__(self, start, finish, time_length):
        self.start = float(start)
        self.finish = float(finish)
        self.time_length = max(int(time_length), 1)
        self.delta = (self.start - self.finish) / self.time_length

    def eval(self, t_env):
        return max(self.finish, self.start - self.delta * int(t_env))


class EpsilonGreedyActionSelector:
    def __init__(self, args):
        self.schedule = DecayThenFlatSchedule(
            getattr(args, "epsilon", 1.0),
            getattr(args, "min_epsilon", 0.05),
            getattr(args, "epsilon_anneal_time", 50_000),
        )
        self.epsilon = self.schedule.eval(0)

    def select_action(self, q_values, avail_actions, t_env, test_mode=False, epsilon_override=None):
        if epsilon_override is None:
            epsilon = self.schedule.eval(t_env)
        else:
            epsilon = float(epsilon_override)
        if test_mode:
            epsilon = 0.0
        self.epsilon = epsilon

        masked_q_values = q_values.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")
        avail_actions_ind = torch.nonzero(avail_actions[0] > 0.0, as_tuple=False).squeeze(-1)
        if avail_actions_ind.numel() == 0:
            return torch.zeros(1, dtype=torch.long, device=q_values.device)
        if np.random.uniform() < epsilon:
            random_idx = np.random.choice(avail_actions_ind.cpu().numpy())
            return torch.tensor([random_idx], dtype=torch.long, device=q_values.device)
        return masked_q_values.max(dim=1)[1]


class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super().__init__()
        self.args = args
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


class QMixer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.state_dim = int(np.prod(args.state_shape))
        self.embed_dim = getattr(args, "mixing_embed_dim", args.qmix_hidden_dim)
        hypernet_layers = getattr(args, "hypernet_layers", 1)
        hypernet_embed = getattr(args, "hypernet_embed", args.hyper_hidden_dim)

        if hypernet_layers == 1:
            self.hyper_w_1 = nn.Linear(self.state_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.state_dim, self.embed_dim)
        elif hypernet_layers == 2:
            self.hyper_w_1 = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim * self.n_agents),
            )
            self.hyper_w_final = nn.Sequential(
                nn.Linear(self.state_dim, hypernet_embed),
                nn.ReLU(),
                nn.Linear(hypernet_embed, self.embed_dim),
            )
        else:
            raise ValueError("Only 1 or 2 hypernet layers are supported.")

        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.state_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, agent_qs, states):
        episode_num = agent_qs.size(0)
        states = states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        w1 = torch.abs(self.hyper_w_1(states))
        b1 = self.hyper_b_1(states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(states))
        w_final = w_final.view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        y = torch.bmm(hidden, w_final) + v
        return y.view(episode_num, -1, 1)


class QMIX:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        gpu_id = int(getattr(args, "gpu_id", 0))
        self.device = torch.device(
            f"cuda:{gpu_id}" if self.args.cuda and torch.cuda.is_available() else "cpu"
        )

        input_shape = self.obs_shape
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        self.eval_rnn = RNNAgent(input_shape, args).to(self.device)
        self.target_rnn = copy.deepcopy(self.eval_rnn).to(self.device)
        self.eval_qmix_net = QMixer(args).to(self.device)
        self.target_qmix_net = copy.deepcopy(self.eval_qmix_net).to(self.device)
        self.action_selector = EpsilonGreedyActionSelector(args)
        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map

        self.eval_parameters = list(self.eval_rnn.parameters()) + list(
            self.eval_qmix_net.parameters()
        )
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(
                self.eval_parameters,
                lr=args.lr,
                alpha=getattr(args, "optim_alpha", 0.99),
                eps=getattr(args, "optim_eps", 1e-5),
            )
        else:
            self.optimizer = torch.optim.Adam(self.eval_parameters, lr=args.lr)

        if self.args.load_model:
            self._load_model()

        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

        self.eval_hidden = None
        self.target_hidden = None
        print("Init alg QMIX")

    def _load_model(self):
        agent_path = os.path.join(self.model_dir, "agent.th")
        mixer_path = os.path.join(self.model_dir, "mixer.th")
        optimizer_path = os.path.join(self.model_dir, "opt.th")
        legacy_rnn = os.path.join(self.model_dir, "rnn_net_params.pkl")
        legacy_mixer = os.path.join(self.model_dir, "qmix_net_params.pkl")
        map_location = self.device

        if os.path.exists(agent_path) and os.path.exists(mixer_path):
            self.eval_rnn.load_state_dict(torch.load(agent_path, map_location=map_location))
            self.eval_qmix_net.load_state_dict(
                torch.load(mixer_path, map_location=map_location)
            )
            if os.path.exists(optimizer_path):
                self.optimizer.load_state_dict(
                    torch.load(optimizer_path, map_location=map_location)
                )
            return

        if os.path.exists(legacy_rnn) and os.path.exists(legacy_mixer):
            self.eval_rnn.load_state_dict(
                torch.load(legacy_rnn, map_location=map_location)
            )
            self.eval_qmix_net.load_state_dict(
                torch.load(legacy_mixer, map_location=map_location)
            )
            return

        numbered_agent = glob.glob(os.path.join(self.model_dir, "*_agent.th"))
        numbered_mixer = glob.glob(os.path.join(self.model_dir, "*_mixer.th"))
        if numbered_agent and numbered_mixer:
            def _sort_key(path):
                prefix = os.path.basename(path).split("_", 1)[0]
                return int(prefix) if prefix.isdigit() else -1

            self.eval_rnn.load_state_dict(
                torch.load(max(numbered_agent, key=_sort_key), map_location=map_location)
            )
            self.eval_qmix_net.load_state_dict(
                torch.load(max(numbered_mixer, key=_sort_key), map_location=map_location)
            )
            return

        raise Exception("No model!")

    def init_hidden(self, episode_num):
        hidden = self.eval_rnn.init_hidden().to(self.device)
        self.eval_hidden = hidden.unsqueeze(0).expand(episode_num, self.n_agents, -1).contiguous()
        self.target_hidden = hidden.unsqueeze(0).expand(episode_num, self.n_agents, -1).contiguous()

    def choose_action(
        self,
        obs,
        last_action,
        agent_num,
        avail_actions,
        epsilon,
        timestep_cur,
        test_mode=False,
    ):
        inputs = obs.copy()
        agent_id = np.zeros(self.n_agents, dtype=np.float32)
        agent_id[agent_num] = 1.0

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))

        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device).unsqueeze(0)
        avail_actions = torch.tensor(
            avail_actions, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        hidden_state = self.eval_hidden[:, agent_num, :]

        q_value, next_hidden = self.eval_rnn(inputs, hidden_state)
        self.eval_hidden[:, agent_num, :] = next_hidden
        action = self.action_selector.select_action(
            q_value,
            avail_actions,
            timestep_cur,
            test_mode=test_mode,
            epsilon_override=epsilon,
        )
        return int(action.item())

    def learn(self, batch, max_episode_len, train_step, epsilon=None):
        del epsilon
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)

        tensor_batch = {}
        for key, value in batch.items():
            if key == "u":
                tensor_batch[key] = torch.tensor(
                    value, dtype=torch.long, device=self.device
                )
            else:
                tensor_batch[key] = torch.tensor(
                    value, dtype=torch.float32, device=self.device
                )

        s = tensor_batch["s"]
        s_next = tensor_batch["s_next"]
        u = tensor_batch["u"]
        r = tensor_batch["r"]
        avail_u = tensor_batch["avail_u"]
        avail_u_next = tensor_batch["avail_u_next"]
        terminated = tensor_batch["terminated"]
        mask = 1 - tensor_batch["padded"].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        agent_active_mask = tensor_batch.get("agent_active_mask", None)
        if agent_active_mask is not None:
            agent_active_mask = agent_active_mask.squeeze(-1)
            transition_active_mask = (
                agent_active_mask.sum(dim=2, keepdim=True) > 0
            ).float()
            mask = mask * transition_active_mask

        q_evals, q_targets, q_eval_next = self.get_q_values(tensor_batch, max_episode_len)
        chosen_action_qvals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        if agent_active_mask is not None:
            chosen_action_qvals = chosen_action_qvals * agent_active_mask

        q_targets[avail_u_next == 0.0] = -9999999
        if getattr(self.args, "double_q", True):
            q_eval_next_detach = q_eval_next.detach().clone()
            q_eval_next_detach[avail_u_next == 0.0] = -9999999
            cur_max_actions = q_eval_next_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(q_targets, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = q_targets.max(dim=3)[0]
        if agent_active_mask is not None:
            target_max_qvals = target_max_qvals * agent_active_mask

        q_total_eval = self.eval_qmix_net(chosen_action_qvals, s)
        q_total_target = self.target_qmix_net(target_max_qvals, s_next)
        targets = r + self.args.gamma * (1 - terminated) * q_total_target

        td_error = q_total_eval - targets.detach()
        mask = mask.expand_as(td_error)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum().clamp(min=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())

    def _get_inputs(self, batch, transition_idx):
        obs = batch["o"][:, transition_idx]
        obs_next = batch["o_next"][:, transition_idx]
        u_onehot = batch["u_onehot"]
        episode_num = obs.shape[0]
        inputs = [obs]
        inputs_next = [obs_next]

        if self.args.last_action:
            if transition_idx == 0:
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])

        if self.args.reuse_network:
            agent_eye = (
                torch.eye(self.args.n_agents, device=obs.device)
                .unsqueeze(0)
                .expand(episode_num, -1, -1)
            )
            inputs.append(agent_eye)
            inputs_next.append(agent_eye)

        inputs = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1
        )
        inputs_next = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next], dim=1
        )
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals, q_targets, q_eval_next = [], [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(batch, transition_idx)
            q_eval, self.eval_hidden = self.eval_rnn(inputs, self.eval_hidden)
            q_eval_next_step, _ = self.eval_rnn(
                inputs_next,
                self.eval_hidden.detach().clone(),
            )
            q_target, self.target_hidden = self.target_rnn(inputs_next, self.target_hidden)

            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_eval_next_step = q_eval_next_step.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_eval_next.append(q_eval_next_step)
            q_targets.append(q_target)

        q_evals = torch.stack(q_evals, dim=1)
        q_eval_next = torch.stack(q_eval_next, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets, q_eval_next

    def save_model(self, train_step):
        num = str(train_step // max(self.args.save_cycle, 1))
        os.makedirs(self.model_dir, exist_ok=True)

        torch.save(self.eval_rnn.state_dict(), os.path.join(self.model_dir, "agent.th"))
        torch.save(self.eval_qmix_net.state_dict(), os.path.join(self.model_dir, "mixer.th"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.model_dir, "opt.th"))
        torch.save(
            self.eval_rnn.state_dict(),
            os.path.join(self.model_dir, f"{num}_agent.th"),
        )
        torch.save(
            self.eval_qmix_net.state_dict(),
            os.path.join(self.model_dir, f"{num}_mixer.th"),
        )
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.model_dir, f"{num}_opt.th"),
        )

        # Legacy files kept for compatibility with older tooling.
        torch.save(
            self.eval_qmix_net.state_dict(),
            os.path.join(self.model_dir, "qmix_net_params.pkl"),
        )
        torch.save(
            self.eval_rnn.state_dict(),
            os.path.join(self.model_dir, "rnn_net_params.pkl"),
        )
