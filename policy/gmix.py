from logging import warn, warning
import torch
import os
from network.base_net import RNN
from network.base_net_nonegetive import RNN_NONEGETIVE
from network.qmix_net import QMixNet
from network.vdn_net import VDNNet


class GMIX:
    def __init__(self, args):
        self.device = torch.device(
            f"cuda:{getattr(self.args, 'gpu_id', 0)}"
            if self.args.cuda and torch.cuda.is_available()
            else "cpu"
        )
        # self.device = torch.device("cpu")
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        input_shape = self.obs_shape
        # 根据参数决定RNN的输入维度
        if args.last_action:
            input_shape += self.n_actions
        if args.reuse_network:
            input_shape += self.n_agents

        # 神经网络
        self.eval_rnn = RNN(input_shape, args)  # 每个agent选动作的网络
        self.target_rnn = RNN(input_shape, args)
        self.eval_qmix_net = QMixNet(args)  # 把agentsQ值加起来的网络
        self.target_qmix_net = QMixNet(args)

        self.distributed = args.distributed

        self.eval_guide = RNN_NONEGETIVE(input_shape + 1, args)  # G-network
        self.target_guide = RNN_NONEGETIVE(input_shape + 1, args)
        # 这两个 mix net 用来 更新 G-network，与 Q-network 无关
        if args.guide_mix_network_type == "vdn":
            self.eval_gmix_net = VDNNet()
            self.target_gmix_net = VDNNet()
        elif args.guide_mix_network_type == "qmix":
            self.eval_gmix_net = QMixNet(args)
            self.target_gmix_net = QMixNet(args)
        else:
            raise Exception("No such type of mix network")

        self.args = args
        if self.args.cuda:
            self.eval_rnn.cuda()
            self.target_rnn.cuda()
            self.eval_qmix_net.cuda()
            self.target_qmix_net.cuda()
            self.eval_guide.cuda()
            self.target_guide.cuda()
        self.model_dir = args.model_dir + "/" + args.alg + "/" + args.map
        # 如果存在模型则加载模型
        if self.args.load_model:
            if os.path.exists(self.model_dir + "/rnn_net_params.pkl"):
                path_rnn = self.model_dir + "/rnn_net_params.pkl"
                path_qmix = self.model_dir + "/qmix_net_params.pkl"
                path_guide = self.model_dir + "/guide_net_params.pkl"
                map_location = (
                    f"cuda:{getattr(self.args, 'gpu_id', 0)}" if self.args.cuda else "cpu"
                )
                self.eval_rnn.load_state_dict(
                    torch.load(path_rnn, map_location=map_location)
                )
                self.eval_qmix_net.load_state_dict(
                    torch.load(path_qmix, map_location=map_location)
                )
                self.eval_guide.load_state_dict(
                    torch.load(path_guide, map_location=map_location)
                )
                print(
                    "Successfully load the model: {} and {}".format(path_rnn, path_qmix)
                )
            else:
                raise Exception("No model!")

        # 让target_net和eval_net的网络参数相同
        self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
        self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
        self.target_guide.load_state_dict(self.eval_guide.state_dict())

        self.eval_parameters = (
            list(self.eval_qmix_net.parameters())
            + list(self.eval_rnn.parameters())
            + list(self.eval_guide.parameters())
        )
        if args.optimizer == "RMS":
            self.optimizer = torch.optim.RMSprop(self.eval_parameters, lr=args.lr)

        # 执行过程中，要为每个agent都维护一个eval_hidden
        # 学习过程中，要为每个episode的每个agent都维护一个eval_hidden、target_hidden
        self.eval_hidden = []
        self.target_hidden = []

        self.eval_hidden_guide = []
        self.target_hidden_guide = []
        print("Init alg GMIX")

    def learn(
        self, batch, max_episode_len, train_step, epsilon=None
    ):  # train_step表示是第几次学习，用来控制更新target_net网络的参数
        """
        在learn的时候，抽取到的数据是四维的，四个维度分别为 1——第几个episode 2——episode中第几个transition
        3——第几个agent的数据 4——具体obs维度。因为在选动作时不仅需要输入当前的inputs，还要给神经网络输入hidden_state，
        hidden_state和之前的经验相关，因此就不能随机抽取经验进行学习。所以这里一次抽取多个episode，然后一次给神经网络
        传入每个episode的同一个位置的transition
        """
        episode_num = batch["o"].shape[0]
        self.init_hidden(episode_num)
        for key in batch.keys():  # 把batch里的数据转化成tensor
            if key == "u":
                batch[key] = torch.tensor(batch[key], dtype=torch.long)
            else:
                batch[key] = torch.tensor(batch[key], dtype=torch.float32)
        s, s_next, u, r, avail_u, avail_u_next, terminated, warning_signal = (
            batch["s"],
            batch["s_next"],
            batch["u"],
            batch["r"],
            batch["avail_u"],
            batch["avail_u_next"],
            batch["terminated"],
            batch["warning_signal"],
        )
        mask = (
            1 - batch["padded"].float()
        )  # 用来把那些填充的经验的TD-error置0，从而不让它们影响到学习
        agent_active_mask = batch.get("agent_active_mask", None)
        if agent_active_mask is not None:
            agent_active_mask = agent_active_mask.squeeze(-1)
            transition_active_mask = (
                agent_active_mask.sum(dim=2, keepdim=True) > 0
            ).float()
            mask = mask * transition_active_mask

        if self.args.cuda:
            s = s.cuda()
            u = u.cuda()
            r = r.cuda()
            s_next = s_next.cuda()
            terminated = terminated.cuda()
            mask = mask.cuda()
            warning_signal = warning_signal.cuda()
            if agent_active_mask is not None:
                agent_active_mask = agent_active_mask.cuda()

        # 得到每个agent对应的Q值，维度为(episode个数, max_episode_len, n_agents, n_actions)
        q_evals, q_targets, g_evals, g_targets = self.get_q_g_values(
            batch, max_episode_len
        )
        # 取每个agent动作对应的Q值，并且把最后不需要的一维去掉，因为最后一维只有一个值了
        q_evals = torch.gather(q_evals, dim=3, index=u).squeeze(3)
        if agent_active_mask is not None:
            q_evals = q_evals * agent_active_mask
        q_total_eval = self.eval_qmix_net(q_evals, s)

        g_evals = torch.gather(g_evals, dim=3, index=u).squeeze(3)
        if agent_active_mask is not None:
            g_evals = g_evals * agent_active_mask
        g_tot_eval = self.eval_gmix_net(g_evals)

        # 得到target_q
        q_targets[avail_u_next == 0.0] = -9999999
        q_targets = q_targets.max(dim=3)[0]
        if agent_active_mask is not None:
            q_targets = q_targets * agent_active_mask
        q_total_target = self.target_qmix_net(q_targets, s_next)
        targets = r + self.args.gamma * q_total_target * (1 - terminated)
        td_error = q_total_eval - targets.detach()
        masked_td_error = mask * td_error  # 抹掉填充的经验的td_error

        # target_g
        beta = 0.8
        g_targets = -(warning_signal + g_evals.unsqueeze(3) * beta)

        g_tot_target = self.target_gmix_net(g_targets).squeeze(3)
        g_error = g_tot_eval - g_tot_target.detach()
        masked_g_error = mask * g_error

        # 不能直接用mean，因为还有许多经验是没用的，所以要求和再比真实的经验数，才是真正的均值
        loss = ((masked_td_error + masked_g_error) ** 2).sum() / mask.sum().clamp(min=1.0)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.eval_parameters, self.args.grad_norm_clip)
        self.optimizer.step()

        if train_step > 0 and train_step % self.args.target_update_cycle == 0:
            self.target_rnn.load_state_dict(self.eval_rnn.state_dict())
            self.target_qmix_net.load_state_dict(self.eval_qmix_net.state_dict())
            self.target_guide.load_state_dict(self.eval_guide.state_dict())

    def _get_inputs(self, batch, transition_idx):
        # 取出所有episode上该transition_idx的经验，u_onehot要取出所有，因为要用到上一条
        obs, obs_next, u_onehot = (
            batch["o"][:, transition_idx],
            batch["o_next"][:, transition_idx],
            batch["u_onehot"][:],
        )

        episode_num = obs.shape[0]
        inputs, inputs_next = [], []
        inputs.append(obs)
        inputs_next.append(obs_next)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            if transition_idx == 0:  # 如果是第一条经验，就让前一个动作为0向量
                inputs.append(torch.zeros_like(u_onehot[:, transition_idx]))
            else:
                inputs.append(u_onehot[:, transition_idx - 1])
            inputs_next.append(u_onehot[:, transition_idx])
        if self.args.reuse_network:
            # 因为当前的obs三维的数据，每一维分别代表(episode编号，agent编号，obs维度)，直接在dim_1上添加对应的向量
            # 即可，比如给agent_0后面加(1, 0, 0, 0, 0)，表示5个agent中的0号。而agent_0的数据正好在第0行，那么需要加的
            # agent编号恰好就是一个单位矩阵，即对角线为1，其余为0
            inputs.append(
                torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
            )
            inputs_next.append(
                torch.eye(self.args.n_agents).unsqueeze(0).expand(episode_num, -1, -1)
            )
        # 要把obs中的三个拼起来，并且要把episode_num个episode、self.args.n_agents个agent的数据拼成40条(40,96)的数据，
        # 因为这里所有agent共享一个神经网络，每条数据中带上了自己的编号，所以还是自己的数据
        inputs = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs], dim=1
        )
        inputs_next = torch.cat(
            [x.reshape(episode_num * self.args.n_agents, -1) for x in inputs_next],
            dim=1,
        )
        return inputs, inputs_next

    def get_q_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals, q_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(
                batch, transition_idx
            )  # 给obs加last_action、agent_id
            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(
                inputs, self.eval_hidden
            )  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(
                inputs_next, self.target_hidden
            )

            # 把q_eval维度重新变回(8, 5,n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)
        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)
        return q_evals, q_targets

    def get_q_g_values(self, batch, max_episode_len):
        episode_num = batch["o"].shape[0]
        q_evals, q_targets = [], []
        g_evals, g_targets = [], []
        for transition_idx in range(max_episode_len):
            inputs, inputs_next = self._get_inputs(
                batch, transition_idx
            )  # 给obs加last_action、agent_id

            if self.args.cuda:
                inputs = inputs.cuda()
                inputs_next = inputs_next.cuda()
                self.eval_hidden = self.eval_hidden.cuda()
                self.target_hidden = self.target_hidden.cuda()
            q_eval, self.eval_hidden = self.eval_rnn(
                inputs, self.eval_hidden
            )  # inputs维度为(40,96)，得到的q_eval维度为(40,n_actions)
            q_target, self.target_hidden = self.target_rnn(
                inputs_next, self.target_hidden
            )

            # 把q_eval维度重新变回(8, 5, n_actions)
            q_eval = q_eval.view(episode_num, self.n_agents, -1)
            q_target = q_target.view(episode_num, self.n_agents, -1)
            q_evals.append(q_eval)
            q_targets.append(q_target)

            warning_signals = batch["warning_signal"][:, transition_idx].reshape(-1, 1)
            # print(inputs.shape, warning_signals.shape)
            # TODO 这里 reshape 成当前形状，是为了将 warning signal 拼接到 input 上，这样操作是否正确有待验证。
            warning_signals_next = batch["warning_signal"][
                :, transition_idx + 1 if transition_idx + 1 < max_episode_len else 0
            ].reshape(-1, 1)
            if self.args.cuda:
                warning_signals = warning_signals.cuda()
                warning_signals_next = warning_signals_next.cuda()

            inputs_exp = torch.cat([inputs, warning_signals], dim=1)
            inputs_next_exp = torch.cat([inputs_next, warning_signals_next], dim=1)

            if self.args.cuda:
                inputs_exp = inputs_exp.cuda()
                inputs_next_exp = inputs_next_exp.cuda()
                self.eval_hidden_guide = self.eval_hidden_guide.cuda()
                self.target_hidden_guide = self.target_hidden_guide.cuda()

            g_eval, self.eval_hidden_guide = self.eval_guide(
                inputs_exp, self.eval_hidden_guide
            )
            g_target, self.target_hidden_guide = self.target_guide(
                inputs_next_exp, self.target_hidden_guide
            )
            g_eval = g_eval.view(episode_num, self.n_agents, -1)
            g_target = g_target.view(episode_num, self.n_agents, -1)
            g_evals.append(g_eval)
            g_targets.append(g_target)

        # 得的q_eval和q_target是一个列表，列表里装着max_episode_len个数组，数组的的维度是(episode个数, n_agents，n_actions)
        # 把该列表转化成(episode个数, max_episode_len， n_agents，n_actions)的数组
        q_evals = torch.stack(q_evals, dim=1)
        q_targets = torch.stack(q_targets, dim=1)

        g_evals = torch.stack(g_evals, dim=1)
        g_targets = torch.stack(g_targets, dim=1)
        return q_evals, q_targets, g_evals, g_targets

    def init_hidden(self, episode_num):
        # 为每个episode中的每个agent都初始化一个eval_hidden、target_hidden
        self.eval_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )
        self.target_hidden = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )

        self.eval_hidden_guide = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )
        self.target_hidden_guide = torch.zeros(
            (episode_num, self.n_agents, self.args.rnn_hidden_dim)
        )

    def save_model(self, train_step):
        num = str(train_step // self.args.save_cycle)
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        torch.save(
            self.eval_qmix_net.state_dict(),
            self.model_dir + "/" + num + "_qmix_net_params.pkl",
        )
        torch.save(
            self.eval_rnn.state_dict(),
            self.model_dir + "/" + num + "_rnn_net_params.pkl",
        )
        torch.save(
            self.eval_guide.state_dict(),
            self.model_dir + "/" + num + "_guide_net_params.pkl",
        )

    def fix_q_values(self, q_value, inputs, agent_num, timestep_cur, timestep_max):
        g_value, self.eval_hidden_guide[:, agent_num, :] = self.eval_guide(
            inputs, self.eval_hidden_guide[:, agent_num, :]
        )
        assert (
            q_value.shape == g_value.shape
        ), "q_values and g_values should have the same shape"
        # G 权重退火
        # phi = 0.5 * (1 - timestep_cur / timestep_max)
        phi = 0.5
        # (0, x)
        upper_bound = 5.0
        lower_bound = -upper_bound

        q_value = lower_bound + (upper_bound - lower_bound) * torch.sigmoid(q_value)
        g_value = upper_bound * torch.sigmoid(g_value)

        return q_value - g_value * phi

    def guard(self, agent_num, inputs, action, risk_level):
        risk_level = torch.tensor([risk_level], dtype=torch.float32).unsqueeze(0)
        max_risk_levels = {"Basic2P": 20, "IoV": 50, "SMAC": 100}
        max_risk_level = max_risk_levels.get(self.args.map, 100)
        # whether activate guard
        if risk_level < max_risk_level:
            safe_action = action
        else:
            inputs_exp = torch.cat([inputs, risk_level], dim=1)
            if self.args.cuda:
                risk_level = risk_level.cuda()

            if self.args.cuda:
                inputs_exp = inputs_exp.cuda()
                self.eval_hidden_guide = self.eval_hidden_guide.cuda()
            g_value, self.eval_hidden_guide[:, agent_num, :] = self.eval_guide(
                inputs_exp, self.eval_hidden_guide[:, agent_num, :]
            )
            safe_action = torch.argmin(g_value).cpu()

        return safe_action
