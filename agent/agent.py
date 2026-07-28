from copy import deepcopy
from networkx import neighbors
import numpy as np
import torch
from torch.distributions import Categorical

from common.rollout import SMAC_MAPS, SMAC_SAFE_DISABLED_ALGS
from policy.dqn import DQN
from common.aoi_comm import AoIMessageEnhancer

# Agent no communication
class Agents:
    def __init__(self, args, env, agent_id=None):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args
        self.env = env
        self.agent_id = agent_id

        if args.alg.find("vdn") != -1:
            from policy.vdn import VDN

            self.policy = VDN(args)
        elif args.alg.find("iql") != -1:
            from policy.iql import IQL

            self.policy = IQL(args)
        elif args.alg.find("qmix") != -1:
            from policy.qmix import QMIX

            self.policy = QMIX(args)
        elif args.alg == "coma":
            from policy.coma import COMA

            self.policy = COMA(args)
        elif args.alg == "qtran_alt":
            from policy.qtran_alt import QtranAlt

            self.policy = QtranAlt(args)
        elif args.alg == "qtran_base":
            from policy.qtran_base import QtranBase

            self.policy = QtranBase(args)
        elif args.alg == "maven":
            from policy.maven import MAVEN

            self.policy = MAVEN(args)
        elif args.alg == "central_v":
            from policy.central_v import CentralV

            self.policy = CentralV(args)
        elif args.alg == "reinforce":
            from policy.reinforce import Reinforce

            self.policy = Reinforce(args)
        elif args.alg.find("gmix") > -1:
            from policy.gmix import GMIX

            self.policy = GMIX(args)
        elif args.alg.find("ippo") > -1:
            if getattr(args, "use_level_policy", False):
                raise ValueError("ippo is only implemented for flat PPO.")
            from policy.ippo import IPPO

            self.policy = IPPO(args)
        elif args.alg == "official_mappo":
            if getattr(args, "use_level_policy", False):
                raise ValueError("official_mappo is only implemented for flat MAPPO.")
            from policy.official_mappo import OfficialMAPPO

            self.policy = OfficialMAPPO(args)
        elif args.alg == "official_hmappo":
            if not getattr(args, "use_level_policy", False):
                raise ValueError("official_hmappo requires level-policy training.")
            from level_policy.official_hmappo import OfficialHMAPPO

            self.policy = OfficialHMAPPO(args)
        elif args.alg == "hmappo_cbf_flow":
            if not getattr(args, "use_level_policy", False):
                raise ValueError("hmappo_cbf_flow requires level-policy training.")
            from level_policy.mappo import MAPPO

            self.policy = MAPPO(args)
        elif args.alg.find("mappo_lagrangian") > -1:
            if getattr(args, "use_level_policy", False):
                raise ValueError("mappo_lagrangian is only implemented for flat MAPPO.")
            from policy.mappo_lagrangian import MAPPOLagrangian

            self.policy = MAPPOLagrangian(args)
        elif args.alg.find("mappo") > -1:
            if getattr(args, "use_level_policy", False):
                from level_policy.mappo import MAPPO
            else:
                from policy.mappo import MAPPO

            self.policy = MAPPO(args)
        elif args.alg.find("macpo") > -1:
            from policy.macpo import MACPO

            self.policy = MACPO(args)
        elif args.alg.lower().find("rgmcomm") > -1:
            if getattr(args, "use_level_policy", False):
                from level_policy.maddpg import MADDPG
                self.policy = MADDPG(args, agent_id)
            elif args.alg.lower().find("matd3") > -1:
                from policy.matd3 import MATD3

                self.policy = MATD3(args, agent_id)
            else:
                from policy.maddpg import MADDPG
                self.policy = MADDPG(args, agent_id)
        else:
            raise Exception("No such algorithm")

        self.obs_ = {}
        self.comm_buffer = {
            "obs": [],
            "action": [],
            "sender": [],
            "aoi": [],
            "share_ratio": [],
            "fresh_ratio": [],
        }
        self.state_ = []
        self.safety_guard = None
        self.use_comm_plugin = bool(
            self.args.alg.find("Comm") > -1 and getattr(args, "msg_shape", 0) > 0
        )
        self.prev_comm_lambda = np.zeros(self.n_agents, dtype=np.float32)
        self.cbf_teacher = None
        self.correction_records = []
        self.energy_records = []
        if args.alg == "hmappo_cbf_flow":
            from safety.joint_cbf_qp import JointCBFQP

            self.cbf_teacher = JointCBFQP(
                alpha1=getattr(args, "cbf_alpha1", 1.0),
                alpha2=getattr(args, "cbf_alpha2", 1.0),
                teacher_margin=getattr(args, "cbf_teacher_margin", 0.0),
            )
            self.last_cbf_constraint_A = None
            self.last_cbf_constraint_c = None

        if self.use_comm_plugin:
            dqn_kwargs = dict(
                state_dim=getattr(args, "raw_obs_shape", args.msg_shape) + 1,
                n_actions=args.msg_shape,
                max_keep_dim=getattr(args, "comm_max_keep_dim", args.msg_shape),
                max_episode=args.n_steps,
                use_cuda=args.cuda,
            )
            if getattr(args, "comm_lr", None) is not None:
                dqn_kwargs["lr"] = float(args.comm_lr)
            try:
                self.comm_policy = DQN(
                    gpu_id=getattr(args, "gpu_id", 0),
                    **dqn_kwargs,
                )
            except TypeError:
                self.comm_policy = DQN(**dqn_kwargs)
            self.aoi_comm = AoIMessageEnhancer(
                n_agents=args.n_agents,
                msg_dim=args.msg_shape,
                aoi_threshold=getattr(args, "aoi_threshold", 0.25),
                max_keep_dim=getattr(args, "comm_max_keep_dim", args.msg_shape),
            )
            if getattr(args, "load_model", False):
                comm_path = (
                    args.model_dir + "/" + args.alg + "/" + args.map + "/comm_policy"
                )
                try:
                    self.comm_policy.load_models(comm_path)
                except Exception:
                    pass

        is_safe_alg = self.args.alg.lower().find("safe") > -1
        smac_safe_blocked = (
            self.args.map in SMAC_MAPS and self.args.alg in SMAC_SAFE_DISABLED_ALGS
        )
        supports_safety = (
            self.args.alg.find("mappo") > -1
            or self.args.alg.find("qmix") > -1
            or self.args.alg.find("vdn") > -1
        )
        if (
            supports_safety
            and is_safe_alg
            and not smac_safe_blocked
            and self.args.alg.lower().find("rgmcomm") < 0
        ):
            from policy.mappo_safety import MAPPOSafetyGuide

            self.safety_guard = MAPPOSafetyGuide(args)

    @staticmethod
    def _state_dict_to_cpu(state_dict):
        return {
            key: value.detach().cpu().clone() if torch.is_tensor(value) else value
            for key, value in state_dict.items()
        }

    def export_rollout_state(self):
        snapshot = {}
        if self.args.alg.find("vdn") != -1 or self.args.alg.find("qmix") != -1:
            snapshot["policy_eval_rnn"] = self._state_dict_to_cpu(
                self.policy.eval_rnn.state_dict()
            )
        if self.use_comm_plugin:
            snapshot["comm_q_eval"] = self._state_dict_to_cpu(
                self.comm_policy.q_eval.state_dict()
            )
        if self.safety_guard is not None:
            snapshot["safety_guides"] = self._state_dict_to_cpu(
                self.safety_guard.guides.state_dict()
            )
        return snapshot

    def load_rollout_state(self, snapshot):
        if not snapshot:
            return
        if "policy_eval_rnn" in snapshot and (
            self.args.alg.find("vdn") != -1 or self.args.alg.find("qmix") != -1
        ):
            self.policy.eval_rnn.load_state_dict(snapshot["policy_eval_rnn"])
        if self.use_comm_plugin and "comm_q_eval" in snapshot:
            self.comm_policy.q_eval.load_state_dict(snapshot["comm_q_eval"])
        if self.safety_guard is not None and "safety_guides" in snapshot:
            self.safety_guard.guides.load_state_dict(snapshot["safety_guides"])

    def reset_episode_state(self):
        self.obs_ = {}
        self.state_ = []
        self.prev_comm_lambda = np.zeros(self.n_agents, dtype=np.float32)
        self.comm_buffer = {
            "obs": [],
            "action": [],
            "sender": [],
            "aoi": [],
            "share_ratio": [],
            "fresh_ratio": [],
        }
        if hasattr(self, "aoi_comm"):
            self.aoi_comm.reset()

    def _current_warning_levels(self):
        info = getattr(self.env, "_last_info", None)
        if isinstance(info, dict) and "warning_signal" in info:
            warning = np.asarray(info["warning_signal"], dtype=np.float32).reshape(-1)
        else:
            warning = np.asarray(
                getattr(self.env, "warning_signal", np.zeros(self.n_agents)),
                dtype=np.float32,
            ).reshape(-1)

        if warning.size == 0:
            warning = np.zeros(self.n_agents, dtype=np.float32)
        elif warning.size == 1:
            warning = np.repeat(warning.item(), self.n_agents).astype(np.float32)
        elif warning.size != self.n_agents:
            warning = np.resize(warning, self.n_agents).astype(np.float32)
        return warning

    def _apply_comm_cutoff(self, msg_vec, cutoff):
        masked_msg = np.asarray(msg_vec, dtype=np.float32).copy()
        if masked_msg.size == 0:
            return masked_msg
        cutoff = int(np.clip(cutoff, 0, masked_msg.shape[0] - 1))
        masked_msg[: cutoff + 1] = 0.0
        return masked_msg

    def choose_action(
        self,
        obs,
        last_action,
        agent_num,
        avail_actions,
        epsilon,
        maven_z=None,
        timestep_cur=0,
        timestep_max=int(1e6),
        msg=None,
    ):
        if (
            msg is not None
            and self.use_comm_plugin
            and self.args.alg.lower().find("rgmcomm") < 0
        ):
            msg = np.array(msg, dtype=np.float32).reshape((self.args.n_agents - 1, -1))
            msg = msg.ravel()
            obs = np.concatenate((obs, msg)).reshape(-1)
            self.obs_[agent_num] = obs

        if (
            msg is None
            and self.use_comm_plugin
            and self.args.alg.lower().find("rgmcomm") < 0
        ):
            self.obs_[agent_num] = np.asarray(obs, dtype=np.float32).reshape(-1)

        if self.args.alg.find("ippo") > -1 or self.args.alg.find("mappo") > -1:
            action = self.policy.choose_action(
                obs,
                agent_num,
                avail_actions,
                evaluate=(epsilon == 0),
            )
            if isinstance(action, torch.Tensor):
                action = action.cpu().item()
            return action

        if self.args.alg.find("qmix") != -1:
            action = self.policy.choose_action(
                obs,
                last_action,
                agent_num,
                avail_actions,
                epsilon,
                timestep_cur,
                test_mode=False,
            )
            if isinstance(action, torch.Tensor):
                action = action.cpu().item()
            return action

        inputs = obs.copy()
        avail_actions_ind = np.nonzero(avail_actions)[
            0
        ]  # index of actions which can be choose

        # transform agent_num to onehot vector
        agent_id = np.zeros(self.n_agents)
        agent_id[agent_num] = 1.0

        if self.args.last_action:
            inputs = np.hstack((inputs, last_action))
        if self.args.reuse_network:
            inputs = np.hstack((inputs, agent_id))
        hidden_state = self.policy.eval_hidden[:, agent_num, :]

        # transform the shape of inputs from (42,) to (1,42)
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
            hidden_state = hidden_state.cuda()

        # get q value
        if self.args.alg == "maven":
            maven_z = torch.tensor(maven_z, dtype=torch.float32).unsqueeze(0)
            if self.args.cuda:
                maven_z = maven_z.cuda()
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(
                inputs, hidden_state, maven_z
            )
        else:
            q_value, self.policy.eval_hidden[:, agent_num, :] = self.policy.eval_rnn(
                inputs, hidden_state
            )

        # choose action from q value
        if self.args.alg in ["coma", "central_v", "reinforce"]:
            action = self._choose_action_from_softmax(
                q_value.cpu(), avail_actions, epsilon
            )
        elif self.args.alg.find("macpo") > -1:
            if epsilon == 0:
                q_value[avail_actions == 0.0] = -float("inf")
                action = torch.argmax(q_value)
            else:
                action = self._choose_action_from_softmax(
                    q_value.cpu(), avail_actions, 0
                )
        else:
            q_value[avail_actions == 0.0] = -float("inf")
            if np.random.uniform() < epsilon:
                action = np.random.choice(avail_actions_ind)  # action是一个整数
            else:
                action = torch.argmax(q_value)

        # adaptive guardian
        if self.args.alg.find("g") > -1:
            warning_signal = float(self._current_warning_levels()[agent_num])
            action = self.policy.guard(agent_num, inputs, action, warning_signal)

        if isinstance(action, torch.Tensor):
            action = action.cpu().item()

        return action

    def choose_high_level_action(
        self,
        obs,
        agent_num,
        avail_actions=None,
        epsilon=0.0,
    ):
        if not hasattr(self.policy, "choose_high_level_action"):
            raise AttributeError("Current policy does not expose high-level actions.")
        action = self.policy.choose_high_level_action(
            obs,
            agent_num,
            avail_actions,
            evaluate=(epsilon == 0),
        )
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        return np.asarray(action, dtype=np.float32)

    def prepare_comm_obs(self, observations, epsilon, active_agent_mask=None):
        raw_obs = [
            np.asarray(obs, dtype=np.float32).reshape(-1)
            for obs in np.asarray(observations, dtype=np.float32)
        ]
        if active_agent_mask is None:
            active_agent_mask = np.ones(self.n_agents, dtype=np.float32)
        else:
            active_agent_mask = np.asarray(active_agent_mask, dtype=np.float32).reshape(-1)
        sender_packets = []
        self.obs_ = {}
        self.state_ = []
        sender_aoi_values = [[] for _ in range(self.n_agents)]
        sender_fresh_values = [[] for _ in range(self.n_agents)]

        for sender_idx, sender_obs in enumerate(raw_obs):
            sender_input = np.concatenate(
                (
                    sender_obs.astype(np.float32),
                    np.array([self.prev_comm_lambda[sender_idx]], dtype=np.float32),
                )
            ).astype(np.float32)
            if active_agent_mask[sender_idx] <= 0.0:
                sender_packets.append(
                    {
                        "obs": sender_input,
                        "action": 0,
                        "transmitted": np.zeros_like(sender_obs, dtype=np.float32),
                        "share_ratio": 0.0,
                        "sent": 0.0,
                        "active": 0.0,
                    }
                )
                continue
            decision = self.comm_policy.choose_action(
                sender_input, evaluate=(epsilon == 0.0), return_info=True
            )
            transmitted, tx_stats = self.aoi_comm.build_transmission(
                sender_obs,
                action_idx=decision["action"],
            )
            sender_packets.append(
                {
                    "obs": sender_input,
                    "action": int(decision["action"]),
                    "transmitted": transmitted.astype(np.float32),
                    "share_ratio": float(tx_stats["share_ratio"]),
                    "sent": float(tx_stats["sent"]),
                    "active": 1.0,
                }
            )

        processed_obs = []
        for receiver_idx in range(self.n_agents):
            masked_msgs = []
            receiver_aoi = []
            receiver_fresh = []
            for sender_idx in range(self.n_agents):
                if sender_idx == receiver_idx:
                    continue
                filtered, rx_stats = self.aoi_comm.receive_message(
                    receiver_idx=receiver_idx,
                    sender_idx=sender_idx,
                    transmitted_message=sender_packets[sender_idx]["transmitted"],
                    sent=sender_packets[sender_idx]["sent"],
                )
                masked_msgs.append(filtered)
                receiver_aoi.append(float(rx_stats["mean_aoi"]))
                receiver_fresh.append(float(rx_stats["fresh_ratio"]))
                sender_aoi_values[sender_idx].append(float(rx_stats["mean_aoi"]))
                sender_fresh_values[sender_idx].append(float(rx_stats["fresh_ratio"]))

            if masked_msgs:
                msg = np.asarray(masked_msgs, dtype=np.float32).ravel()
                final_obs = np.concatenate((raw_obs[receiver_idx], msg)).reshape(-1)
                mean_aoi = float(np.mean(receiver_aoi))
                mean_fresh = float(np.mean(receiver_fresh))
            else:
                final_obs = raw_obs[receiver_idx].reshape(-1)
                mean_aoi = 0.0
                mean_fresh = 1.0

            processed_obs.append(final_obs.astype(np.float32))
            self.obs_[receiver_idx] = final_obs.astype(np.float32)
            self.state_.append(final_obs.astype(np.float32))

        for sender_idx, packet in enumerate(sender_packets):
            if packet.get("active", 1.0) <= 0.0:
                self.prev_comm_lambda[sender_idx] = 0.0
                continue
            self.comm_buffer["obs"].append(packet["obs"])
            self.comm_buffer["action"].append(packet["action"])
            self.comm_buffer["sender"].append(int(sender_idx))
            self.comm_buffer["aoi"].append(
                float(np.mean(sender_aoi_values[sender_idx]))
                if sender_aoi_values[sender_idx]
                else 0.0
            )
            self.comm_buffer["share_ratio"].append(float(packet["share_ratio"]))
            self.comm_buffer["fresh_ratio"].append(
                float(np.mean(sender_fresh_values[sender_idx]))
                if sender_fresh_values[sender_idx]
                else 1.0
            )
            self.prev_comm_lambda[sender_idx] = float(packet["share_ratio"])

        return np.asarray(processed_obs, dtype=np.float32)

    def _choose_action_from_softmax(self, inputs, avail_actions, epsilon):
        """
        :param inputs: # q_value of all actions
        """
        action_num = (
            avail_actions.sum(dim=1, keepdim=True)
            .float()
            .repeat(1, avail_actions.shape[-1])
        )  # num of avail_actions
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(inputs, dim=-1)
        # add noise of epsilon
        prob = (1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        action = Categorical(prob).sample().long()
        return action

    def get_low_action_log_probs(self, observations, actions):
        if not hasattr(self.policy, "get_low_action_log_probs"):
            return None
        return self.policy.get_low_action_log_probs(observations, actions)

    def drain_correction_records(self):
        records = self.correction_records
        self.correction_records = []
        return records

    def drain_energy_records(self):
        records = self.energy_records
        self.energy_records = []
        return records

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(self, batch, train_step, epsilon=None):  # coma needs epsilon for training
        # different episode has different length, so we need to get max length of the batch
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            if key != "z":
                batch[key] = batch[key][:, :max_episode_len]
        metrics = {}
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        policy_metrics = getattr(self.policy, "last_train_metrics", None)
        if isinstance(policy_metrics, dict):
            for key, value in policy_metrics.items():
                try:
                    metrics[key] = float(value)
                except (TypeError, ValueError):
                    continue

        if self.safety_guard is not None:
            safety_loss = self.safety_guard.learn(batch, max_episode_len, train_step)
            if safety_loss is not None:
                metrics["safety_loss"] = float(safety_loss)

        if self.use_comm_plugin:
            self.comm_policy.learn()

        # if train_step > 0 and train_step % self.args.save_cycle == 0:
        #     self.policy.save_model(train_step)
        return metrics

    def AoI_update(
        self,
    ):
        if hasattr(self, "aoi_comm"):
            return self.aoi_comm.last_mean_aoi
        return 0.0

    def revise_safe_actions(self, observations, avail_actions, base_actions):
        if self.args.alg == "hmappo_cbf_flow":
            del observations, avail_actions
            active_mask = (
                self.env.get_active_agent_mask()
                if hasattr(self.env, "get_active_agent_mask")
                else np.ones(self.n_agents, dtype=np.float32)
            )
            corrected, A, c = self.cbf_teacher.solve_from_env(
                self.env.env if hasattr(self.env, "env") else self.env,
                np.asarray(base_actions, dtype=np.float32),
                active_mask=active_mask,
            )
            raw = np.asarray(base_actions, dtype=np.float32)
            delta = np.linalg.norm(corrected - raw, axis=-1)
            flags = ((delta > 1e-6).astype(np.float32) * active_mask.reshape(-1))
            self.last_guard_applied = flags.tolist()
            self.last_cbf_constraint_A = A
            self.last_cbf_constraint_c = c
            env_ref = self.env.env if hasattr(self.env, "env") else self.env
            if hasattr(env_ref, "get_cbf_safety_state"):
                self.correction_records.append(
                    {
                        "safety_state_abs": env_ref.get_cbf_safety_state(),
                        "raw_joint_action": raw.reshape(-1).copy(),
                        "correct_joint_action": corrected.reshape(-1).copy(),
                        "constraint_A": A.copy(),
                        "constraint_c": c.copy(),
                        "active_mask": active_mask.copy(),
                    }
                )
            return corrected.astype(np.float32)
        if getattr(self.args, "low_action_type", "discrete") == "continuous":
            self.last_guard_applied = [0.0 for _ in range(self.n_agents)]
            return None
        guard_flags = np.zeros(self.n_agents, dtype=np.float32)
        revised_actions = [int(action) for action in base_actions]

        if self.safety_guard is None:
            self.last_guard_applied = guard_flags.tolist()
        else:
            revised_actions = self.safety_guard.revise_actions(
                observations=observations,
                avail_actions=avail_actions,
                base_actions=base_actions,
                env=self.env,
            )
            guard_flags = np.asarray(
                getattr(
                    self.safety_guard,
                    "last_guard_applied",
                    [0 for _ in range(self.n_agents)],
                ),
                dtype=np.float32,
            )

        if (
            getattr(self.args, "hrl_safe_action_guard_enabled", False)
            and hasattr(self.env, "revise_safe_actions")
        ):
            env_result = self.env.revise_safe_actions(
                revised_actions,
                avail_actions=avail_actions,
                guard_margin=getattr(
                    self.args,
                    "hrl_safe_action_guard_margin",
                    None,
                ),
                guard_horizon=getattr(
                    self.args,
                    "hrl_safe_action_guard_horizon",
                    None,
                ),
            )
            if isinstance(env_result, tuple):
                revised_actions, env_guard_flags = env_result
                guard_flags = np.maximum(
                    guard_flags,
                    np.asarray(env_guard_flags, dtype=np.float32).reshape(-1),
                )
            else:
                revised_actions = env_result

        self.last_guard_applied = guard_flags.astype(np.float32).tolist()
        return revised_actions

    def obs_state_comm(self, reward=-9999):
        obs = []
        self.state_ = []
        for agent in range(self.n_agents):
            obs.append(self.obs_[agent])
            self.state_.append(obs[agent])

        if reward != -9999:
            warning_levels = self._current_warning_levels()
            next_obs = np.asarray(self.env.get_obs(), dtype=np.float32)
            for o, c, sender, share_ratio, fresh_ratio in zip(
                self.comm_buffer["obs"],
                self.comm_buffer["action"],
                self.comm_buffer["sender"],
                self.comm_buffer["share_ratio"],
                self.comm_buffer["fresh_ratio"],
            ):
                warning_level = float(
                    warning_levels[sender] if sender < warning_levels.size else 0.0
                )
                adjusted_reward = (
                    float(np.asarray(reward, dtype=np.float32).mean())
                    - getattr(self.args, "comm_cost_penalty", 0.1) * float(share_ratio)
                    + getattr(self.args, "comm_effect_bonus", 0.1) * float(fresh_ratio)
                )
                risky = int(
                    warning_level >= getattr(self.args, "comm_warning_threshold", 0.1)
                )
                next_o = np.asarray(next_obs[sender], dtype=np.float32).reshape(-1)
                next_o = np.concatenate(
                    (
                        next_o.astype(np.float32),
                        np.array([float(share_ratio)], dtype=np.float32),
                    )
                ).astype(np.float32)
                self.comm_policy.store_transition(
                    o, c, adjusted_reward, next_o, warning=risky
                )
            self.comm_buffer = {
                "obs": [],
                "action": [],
                "sender": [],
                "aoi": [],
                "share_ratio": [],
                "fresh_ratio": [],
            }

        return np.array(obs), np.array(self.state_).ravel()

    def save_model(self, train_step):
        if hasattr(self.policy, "save_model"):
            self.policy.save_model(train_step)
        if self.safety_guard is not None:
            self.safety_guard.save_model(train_step)
        if self.use_comm_plugin:
            comm_path = (
                self.args.model_dir
                + "/"
                + self.args.alg
                + "/"
                + self.args.map
                + "/comm_policy"
            )
            self.comm_policy.save_models(comm_path)
    
    def select_action(self, o, noise_rate, epsilon):
        if np.random.uniform() < epsilon:
            u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
        else:
            inputs = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
            # import pdb; pdb.set_trace()
            pi = self.policy.actor_network(inputs).squeeze(0)
            # print('{} : {}'.format(self.name, pi))
            u = pi.cpu().numpy()
            noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
            u += noise
            u = np.clip(u, -self.args.high_action, self.args.high_action)
        return u.copy()

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

    def learnit(self, transitions):
        return self.policy.trainit(transitions)


# Agent for communication
class CommAgents:
    def __init__(self, args):
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        alg = args.alg
        if alg.find("reinforce") > -1:
            from policy.reinforce import Reinforce

            self.policy = Reinforce(args)
        elif alg.find("coma") > -1:
            from policy.coma import COMA

            self.policy = COMA(args)
        elif alg.find("central_v") > -1:
            from policy.central_v import CentralV

            self.policy = CentralV(args)
        else:
            raise Exception("No such algorithm")
        self.args = args
        print("Init CommAgents")

    # 根据weights得到概率，然后再根据epsilon选动作
    def choose_action(self, weights, avail_actions, epsilon):
        weights = weights.unsqueeze(0)
        avail_actions = torch.tensor(avail_actions, dtype=torch.float32).unsqueeze(0)
        action_num = (
            avail_actions.sum(dim=1, keepdim=True)
            .float()
            .repeat(1, avail_actions.shape[-1])
        )  # 可以选择的动作的个数
        # 先将Actor网络的输出通过softmax转换成概率分布
        prob = torch.nn.functional.softmax(weights, dim=-1)
        # 在训练的时候给概率分布添加噪音
        prob = (1 - epsilon) * prob + torch.ones_like(prob) * epsilon / action_num
        prob[avail_actions == 0] = 0.0  # 不能执行的动作概率为0

        """
        不能执行的动作概率为0之后，prob中的概率和不为1，这里不需要进行正则化，因为torch.distributions.Categorical
        会将其进行正则化。要注意在训练的过程中没有用到Categorical，所以训练时取执行的动作对应的概率需要再正则化。
        """

        action = Categorical(prob).sample().long()
        return action

    def get_action_weights(self, obs, last_action, active_agent_mask=None):
        obs = torch.tensor(obs, dtype=torch.float32)
        last_action = torch.tensor(last_action, dtype=torch.float32)
        if active_agent_mask is None:
            active_agent_mask = np.ones(self.n_agents, dtype=np.float32)
        active_agent_mask = np.asarray(active_agent_mask, dtype=np.float32).reshape(-1)
        active_indices = np.nonzero(active_agent_mask > 0.0)[0]
        weights = torch.zeros((self.args.n_agents, self.args.n_actions), dtype=torch.float32)
        if len(active_indices) == 0:
            return weights
        inputs = list()
        inputs.append(obs)
        # 给obs添加上一个动作、agent编号
        if self.args.last_action:
            inputs.append(last_action)
        if self.args.reuse_network:
            inputs.append(torch.eye(self.args.n_agents))
        inputs = torch.cat([x for x in inputs], dim=1)
        if self.args.cuda:
            inputs = inputs.cuda()
            self.policy.eval_hidden = self.policy.eval_hidden.cuda()
            weights = weights.cuda()
        active_indices_tensor = torch.tensor(
            active_indices,
            dtype=torch.long,
            device=inputs.device,
        )
        active_inputs = inputs.index_select(0, active_indices_tensor)
        active_hidden = self.policy.eval_hidden.index_select(1, active_indices_tensor)
        active_weights, active_hidden_next = self.policy.eval_rnn(
            active_inputs,
            active_hidden,
        )
        weights[active_indices_tensor] = active_weights.reshape(
            len(active_indices),
            self.args.n_actions,
        )
        self.policy.eval_hidden[:, active_indices_tensor, :] = active_hidden_next.view(
            1,
            len(active_indices),
            self.args.rnn_hidden_dim,
        )
        return weights.cpu()

    def _get_max_episode_len(self, batch):
        terminated = batch["terminated"]
        episode_num = terminated.shape[0]
        max_episode_len = 0
        for episode_idx in range(episode_num):
            for transition_idx in range(self.args.episode_limit):
                if terminated[episode_idx, transition_idx, 0] == 1:
                    if transition_idx + 1 >= max_episode_len:
                        max_episode_len = transition_idx + 1
                    break
        if max_episode_len == 0:  # 防止所有的episode都没有结束，导致terminated中没有1
            max_episode_len = self.args.episode_limit
        return max_episode_len

    def train(
        self, batch, train_step, epsilon=None
    ):  # coma在训练时也需要epsilon计算动作的执行概率
        # 每次学习时，各个episode的长度不一样，因此取其中最长的episode作为所有episode的长度
        max_episode_len = self._get_max_episode_len(batch)
        for key in batch.keys():
            batch[key] = batch[key][:, :max_episode_len]
        self.policy.learn(batch, max_episode_len, train_step, epsilon)
        if train_step > 0 and train_step % self.args.save_cycle == 0:
            self.policy.save_model(train_step)

    
