from logging import warning
import numpy as np
import torch
from torch.distributions import one_hot_categorical
import time

from common.seeding import (
    derive_episode_seed,
    preserve_rng_state,
    reset_env_with_seed,
    temporary_seed,
)

SMAC_MAPS = [
    "3m",
    "8m",
    "25m",
    "2s3z",
    "3s5z",
    "MMM",
    "5m_vs_6m",
    "8m_vs_9m",
    "10m_vs_11m",
    "27m_vs_30m",
    "3s5z_vs_3s6z",
    "MMM2",
    "2m_vs_1z",
    "2s_vs_1sc",
    "3s_vs_3z",
    "3s_vs_4z",
    "3s_vs_5z",
    "6h_vs_8z",
    "corridor",
    "bane_vs_bane",
    "so_many_banelings",
    "2c_vs_64zg",
    "1c3s5z",
]

SMAC_SAFE_DISABLED_ALGS = set()


def smac_safe_enabled(args):
    return (
        getattr(args, "map", None) in SMAC_MAPS
        and getattr(args, "alg", "") not in SMAC_SAFE_DISABLED_ALGS
        and getattr(args, "alg", "").lower().find("safe") > -1
    )


def smac_penalty_enabled(args):
    return (
        getattr(args, "map", None) in SMAC_MAPS
        and (
            getattr(args, "alg", "").lower().find("safe") > -1
            or getattr(args, "warning_penalty_weight", None) is not None
        )
    )


def _get_warning_penalty_weight(args):
    override = getattr(args, "warning_penalty_weight", None)
    if override is not None:
        return float(override)
    weights = {
        "Basic2P": 1,
        "IoV": 1 / 20,
        "3m": 1 / 20,
        "8m": 1 / 5,
        "2m_vs_1z": 1 / 10,
    }
    return float(weights.get(args.map, 1.0))


def _get_env_msg(env, args, n_agents):
    if (
        getattr(args, "alg", "").lower().find("comm") > -1
        and getattr(args, "alg", "").lower().find("rgmcomm") < 0
        and hasattr(env, "get_obs")
    ):
        obs = np.asarray(env.get_obs(), dtype=np.float32)
        msg_shape = getattr(args, "msg_shape", obs.shape[-1] if obs.ndim > 1 else 0)
        messages = []
        for receiver in range(n_agents):
            receiver_msgs = []
            for sender in range(n_agents):
                if sender == receiver:
                    continue
                sender_obs = np.asarray(obs[sender], dtype=np.float32).reshape(-1)
                if sender_obs.size > msg_shape:
                    sender_obs = sender_obs[:msg_shape]
                elif sender_obs.size < msg_shape:
                    sender_obs = np.pad(sender_obs, (0, msg_shape - sender_obs.size))
                receiver_msgs.append(sender_obs.astype(np.float32))
            if receiver_msgs:
                messages.append(np.stack(receiver_msgs, axis=0).astype(np.float32))
            else:
                messages.append(np.zeros((0, msg_shape), dtype=np.float32))
        return messages

    if hasattr(env, "get_msg"):
        return env.get_msg()

    msg_shape = getattr(args, "msg_shape", 0)
    return [
        np.zeros((max(n_agents - 1, 0), msg_shape), dtype=np.float32)
        for _ in range(n_agents)
    ]


def _get_noop_action(env, n_actions):
    if hasattr(env, "get_noop_action"):
        return env.get_noop_action()
    return min(int(n_actions) - 1, max(0, int(n_actions) // 2))


def _get_active_agent_mask(env, n_agents):
    if hasattr(env, "get_active_agent_mask"):
        mask = np.asarray(env.get_active_agent_mask(), dtype=np.float32).reshape(-1)
        if mask.size == n_agents:
            return mask
    return np.ones(n_agents, dtype=np.float32)


def _build_env_summary(env, info, step, win_tag, n_agents):
    if hasattr(env, "summary"):
        summary = env.summary()
    else:
        summary = {
            "step": step,
            "agent_health": 0.0,
            "enemy_health": 0.0,
            "agent_alive": 0.0,
            "collision_count": 0.0,
            "obstacle_collision_count": 0.0,
            "agent_collision_count": 0.0,
        }

        allies = getattr(env, "agents", None)
        enemies = getattr(env, "enemies", None)
        if allies is not None:
            ally_health = [
                max(getattr(unit, "health", 0), 0) + max(getattr(unit, "shield", 0), 0)
                for unit in allies
            ]
            summary["agent_health"] = float(np.sum(ally_health))
            summary["agent_alive"] = float(np.sum(np.array(ally_health) > 0))
        elif "dead_allies" in info:
            summary["agent_alive"] = float(n_agents - info["dead_allies"])

        if enemies is not None:
            enemy_health = [
                max(getattr(unit, "health", 0), 0) + max(getattr(unit, "shield", 0), 0)
                for unit in enemies
            ]
            summary["enemy_health"] = float(np.sum(enemy_health))

    summary["step"] = step
    summary["win_tag"] = win_tag
    summary.setdefault("collision_count", 0.0)
    summary.setdefault("obstacle_collision_count", 0.0)
    summary.setdefault("agent_collision_count", 0.0)
    return summary


class RolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.train_episode_index = 0
        # print("Init RolloutWorker")

    def _episode_seed(self, evaluate, episode_num):
        return derive_episode_seed(
            self.args,
            evaluate=evaluate,
            episode_index=episode_num if evaluate else self.train_episode_index,
        )

    def _reset_env_for_episode(self, evaluate, episode_num):
        seed = getattr(self, "_active_episode_seed", None)
        if seed is None:
            seed = self._episode_seed(evaluate, episode_num)

        if evaluate:
            return reset_env_with_seed(self.env, seed)

        with preserve_rng_state(include_torch=False):
            result = reset_env_with_seed(self.env, seed)
        self.train_episode_index += 1
        return result

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        if evaluate and not getattr(self, "_inside_seeded_eval", False):
            seed = self._episode_seed(evaluate=True, episode_num=episode_num)
            with temporary_seed(seed, include_torch=True):
                self._inside_seeded_eval = True
                self._active_episode_seed = seed
                try:
                    return self.generate_episode(episode_num=episode_num, evaluate=True)
                finally:
                    self._inside_seeded_eval = False
                    self._active_episode_seed = None

        if (
            self.args.replay_dir != "" and evaluate and episode_num == 0
        ):  # prepare for save replay of evaluation
            self.env.close()
        o, o_raw, u, r, s, avail_u, u_onehot, terminate, padded, w_signal, c, guard_applied, active_masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        u_raw, u_correct, raw_log_prob, correction_delta, intervention_masks = (
            [],
            [],
            [],
            [],
            [],
        )
        self._reset_env_for_episode(evaluate, episode_num)
        low_action_type = getattr(self.args, "low_action_type", "discrete")
        low_continuous = low_action_type == "continuous"
        low_action_dim = int(getattr(self.args, "low_action_dim", self.args.n_actions))
        if hasattr(self.agents, "reset_episode_state"):
            self.agents.reset_episode_state()
        level_training = bool(
            getattr(self.args, "is_level_training", False)
            and hasattr(self.env, "get_high_level_obs")
            and hasattr(self.env, "apply_high_level_actions")
            and int(getattr(self.args, "high_level_n_actions", 0)) > 0
            and int(getattr(self.args, "high_level_obs_shape", 0)) > 0
            and int(getattr(self.args, "high_level_state_shape", 0)) > 0
        )
        meta_period = max(1, int(getattr(self.args, "hmappo_meta_period", 5)))
        if level_training and hasattr(self.env, "set_meta_period"):
            self.env.set_meta_period(meta_period)
        if level_training and hasattr(self.env, "set_hrl_parameters"):
            self.env.set_hrl_parameters(
                reachable_subgoal_scale=getattr(
                    self.args, "hrl_reachable_subgoal_scale", None
                ),
                intrinsic_reward_scale=getattr(
                    self.args, "hrl_intrinsic_reward_scale", None
                ),
                intrinsic_success_bonus=getattr(
                    self.args, "hrl_intrinsic_success_bonus", None
                ),
                high_goal_style=getattr(self.args, "hrl_high_goal_style", None),
                high_lateral_scale=getattr(
                    self.args, "hrl_high_lateral_scale", None
                ),
                order_progress_override=getattr(
                    self.args, "hrl_order_progress_override", None
                ),
                energy_margin_reserve_ratio=getattr(
                    self.args, "hrl_energy_margin_reserve_ratio", None
                ),
                charge_energy_threshold=getattr(
                    self.args, "hrl_charge_energy_threshold", None
                ),
                charge_release_threshold=getattr(
                    self.args, "hrl_charge_release_threshold", None
                ),
                charge_queue_enabled=getattr(
                    self.args, "hrl_charge_queue_enabled", None
                ),
                charge_queue_radius=getattr(
                    self.args, "hrl_charge_queue_radius", None
                ),
            )
        high_o, high_s, high_u, high_u_raw, high_r = [], [], [], [], []
        high_avail_u, high_o_next, high_s_next = [], [], []
        high_terminate, high_padded, high_active_masks = [], [], []
        high_energy_margins, high_energy_order_masks = [], []
        high_mode_train_masks = []
        high_intervention_masks = []
        high_durations = []
        current_high_transition = None
        current_high_reward = np.zeros(self.n_agents, dtype=np.float32)

        def _agent_energy_ratios():
            if hasattr(self.env, "get_agent_energy_ratios"):
                return np.asarray(
                    self.env.get_agent_energy_ratios(), dtype=np.float32
                ).reshape(self.n_agents, 1)
            return np.zeros((self.n_agents, 1), dtype=np.float32)

        def _terminal_depleted_high_cost(step_info, terminal_flag):
            coef = float(getattr(self.args, "hrl_depleted_terminal_cost_coef", 0.0))
            if coef <= 0.0 or not terminal_flag:
                return np.zeros(self.n_agents, dtype=np.float32)
            if step_info is None:
                step_info = {}
            energy = np.asarray(
                step_info.get("agent_energy", []), dtype=np.float32
            ).reshape(-1)
            if energy.size == self.n_agents:
                depleted = (energy <= 1e-5).astype(np.float32)
            else:
                active_mask = np.asarray(
                    step_info.get("agent_active_mask", []), dtype=np.float32
                ).reshape(-1)
                if active_mask.size == self.n_agents:
                    depleted = (active_mask <= 0.0).astype(np.float32)
                else:
                    depleted = (_agent_energy_ratios().reshape(-1) <= 1e-5).astype(
                        np.float32
                    )
            return coef * depleted

        def _high_action_for_training(high_transition):
            action = np.asarray(high_transition["u"], dtype=np.float32).copy()
            if not hasattr(self.env, "relabel_high_level_actions_with_achieved"):
                return action
            if not hasattr(self.env, "get_agent_positions"):
                return action
            return self.env.relabel_high_level_actions_with_achieved(
                high_transition.get("start_positions"),
                self.env.get_agent_positions(),
                action,
                active_mask=high_transition.get("active_mask"),
            )
        terminated = False
        info = {}
        win_tag = False
        step = 0
        episode_reward = 0  # cumulative environment rewards for logging/eval
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))

        penalty_applied = False
        uses_policy_gradient_reward = any(
            alg_name in self.args.alg.lower()
            for alg_name in ("ippo", "mappo", "macpo")
        )
        apply_warning_reshape = (
            "reshape" in self.args.alg.lower()
            or getattr(self.args, "warning_penalty_weight", None) is not None
            or (
                self.args.map in SMAC_MAPS
                and self.args.alg.lower().find("safe") > -1
            )
        )
        use_constraint_cost = (
            "macpo" in self.args.alg.lower()
            or "lagrangian" in self.args.alg.lower()
        )
        reward_template = np.zeros(
            (self.n_agents,) if uses_policy_gradient_reward else (1,),
            dtype=np.float32,
        )
        apply_policy_arrival_penalty = (
            self.args.map == "Basic2P" and uses_policy_gradient_reward
        )
        penalty_value = -10.0
        
        if self.args.alg.lower() != 'rgmcomm':
            self.agents.policy.init_hidden(1)

        # epsilon
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == "episode":
            epsilon = (
                epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            )

        # sample z for maven
        if self.args.alg == "maven":
            state = self.env.get_state()
            state = torch.tensor(state, dtype=torch.float32)
            if self.args.cuda:
                state = state.cuda()
            z_prob = self.agents.policy.z_policy(state)
            maven_z = one_hot_categorical.OneHotCategorical(z_prob).sample()
            maven_z = list(maven_z.cpu())

        while not terminated and step < self.episode_limit:
            cbf_flow = self.args.alg == "hmappo_cbf_flow"
            if cbf_flow and hasattr(self.env, "prepare_cbf_flow_step"):
                self.env.prepare_cbf_flow_step()
            active_agent_mask = _get_active_agent_mask(self.env, self.n_agents)
            noop_action = _get_noop_action(self.env, self.n_actions)
            if level_training and step % meta_period == 0:
                if current_high_transition is not None:
                    high_reward = current_high_reward.copy()
                    high_o.append(current_high_transition["o"])
                    high_s.append(current_high_transition["s"])
                    high_u.append(_high_action_for_training(current_high_transition))
                    high_u_raw.append(current_high_transition["u_raw"])
                    high_avail_u.append(current_high_transition["avail_u"])
                    high_active_masks.append(current_high_transition["active_mask"])
                    high_mode_train_masks.append(
                        current_high_transition["mode_train_mask"]
                    )
                    high_intervention_masks.append(
                        current_high_transition["intervention_mask"]
                    )
                    high_durations.append(
                        [float(max(1, current_high_transition["duration"]))]
                    )
                    high_energy_margins.append(current_high_transition["energy_margin"])
                    high_energy_order_masks.append(
                        current_high_transition["energy_order_mask"]
                    )
                    high_r.append(high_reward)
                    high_o_next.append(self.env.get_high_level_obs())
                    high_s_next.append(self.env.get_high_level_state())
                    high_terminate.append([False])
                    high_padded.append([0.0])

                if hasattr(self.env, "prepare_high_level_decision"):
                    self.env.prepare_high_level_decision()
                high_obs = self.env.get_high_level_obs()
                high_state = self.env.get_high_level_state()
                high_action_dim = int(getattr(self.args, "high_level_n_actions", 0))
                if hasattr(self.env, "get_high_level_energy_margins"):
                    high_energy_margin = self.env.get_high_level_energy_margins()
                else:
                    high_energy_margin = np.zeros((self.n_agents, 1), dtype=np.float32)
                if hasattr(self.env, "get_high_level_energy_order_masks"):
                    high_energy_order_mask = (
                        self.env.get_high_level_energy_order_masks()
                    )
                else:
                    high_energy_order_mask = np.zeros(
                        (self.n_agents, 1), dtype=np.float32
                    )
                if hasattr(self.env, "get_high_level_avail_actions"):
                    high_avail = np.asarray(
                        self.env.get_high_level_avail_actions(),
                        dtype=np.float32,
                    )
                else:
                    high_avail = np.ones(
                        (self.n_agents, high_action_dim), dtype=np.float32
                    )
                high_actions = []
                use_oracle_high_level = bool(
                    getattr(self.args, "hrl_oracle_high_level", False)
                )
                oracle_high_actions = None
                if use_oracle_high_level and hasattr(
                    self.env, "get_oracle_high_level_actions"
                ):
                    oracle_high_actions = np.asarray(
                        self.env.get_oracle_high_level_actions(),
                        dtype=np.float32,
                    ).reshape(self.n_agents, high_action_dim)
                for agent_id in range(self.n_agents):
                    if active_agent_mask[agent_id] <= 0.0:
                        high_action = np.zeros(high_action_dim, dtype=np.float32)
                    elif use_oracle_high_level:
                        if oracle_high_actions is None:
                            high_action = np.zeros(high_action_dim, dtype=np.float32)
                        else:
                            high_action = oracle_high_actions[agent_id]
                    else:
                        high_action = self.agents.choose_high_level_action(
                            high_obs[agent_id],
                            agent_id,
                            high_avail[agent_id],
                            epsilon,
                        )
                    high_actions.append(np.asarray(high_action, dtype=np.float32))

                raw_high_actions = np.asarray(high_actions, dtype=np.float32).reshape(
                    self.n_agents, high_action_dim
                )
                applied_high_actions = self.env.apply_high_level_actions(high_actions)
                if applied_high_actions is not None:
                    high_actions = np.asarray(applied_high_actions, dtype=np.float32)
                executed_high_actions = np.asarray(
                    high_actions, dtype=np.float32
                ).reshape(self.n_agents, high_action_dim)
                intervention_mask = (
                    np.max(
                        np.abs(executed_high_actions - raw_high_actions),
                        axis=-1,
                        keepdims=True,
                    )
                    > 1e-5
                ).astype(np.float32)
                intervention_mask *= active_agent_mask.reshape(self.n_agents, 1)
                if hasattr(self.env, "get_high_level_mode_training_mask"):
                    high_mode_train_mask = self.env.get_high_level_mode_training_mask()
                else:
                    high_mode_train_mask = active_agent_mask.reshape(self.n_agents, 1)
                current_high_transition = {
                    "o": np.asarray(high_obs, dtype=np.float32).copy(),
                    "s": np.asarray(high_state, dtype=np.float32).copy(),
                    "u": np.asarray(high_actions, dtype=np.float32).reshape(
                        self.n_agents, high_action_dim
                    ),
                    "u_raw": raw_high_actions.copy(),
                    "avail_u": np.asarray(high_avail, dtype=np.float32).copy(),
                    "active_mask": active_agent_mask.reshape(
                        self.n_agents, 1
                    ).copy(),
                    "mode_train_mask": np.asarray(
                        high_mode_train_mask, dtype=np.float32
                    ).reshape(self.n_agents, 1),
                    "energy_margin": np.asarray(
                        high_energy_margin, dtype=np.float32
                    ).reshape(self.n_agents, 1),
                    "energy_order_mask": np.asarray(
                        high_energy_order_mask, dtype=np.float32
                    ).reshape(self.n_agents, 1),
                    "start_positions": (
                        self.env.get_agent_positions()
                        if hasattr(self.env, "get_agent_positions")
                        else None
                    ),
                    "intervention_mask": intervention_mask.copy(),
                    "duration": 0,
                }
                current_high_reward = np.zeros(self.n_agents, dtype=np.float32)

            if apply_policy_arrival_penalty:
                prev_positions = [player.pos for player in self.env.players]
                prev_succeed = [player.succeed for player in self.env.players]

            obs = self.env.get_obs()
            raw_obs = np.asarray(obs, dtype=np.float32).copy()
            if (
                getattr(self.agents, "use_comm_plugin", False)
                and self.args.alg.lower().find("rgmcomm") < 0
            ):
                obs = self.agents.prepare_comm_obs(
                    raw_obs,
                    epsilon,
                    active_agent_mask=active_agent_mask,
                )
                msg = None
            else:
                msg = _get_env_msg(self.env, self.args, self.n_agents)
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if active_agent_mask[agent_id] <= 0.0:
                    action = np.asarray(noop_action, dtype=np.float32).copy() if low_continuous else noop_action
                elif self.args.alg == "maven":
                    action = self.agents.choose_action(
                        obs[agent_id],
                        last_action[agent_id],
                        agent_id,
                        avail_action,
                        epsilon,
                        maven_z=maven_z,
                        timestep_cur=step,
                        timestep_max=self.args.n_steps,
                    )
                else:
                    # import pdb; pdb.set_trace()
                    action = self.agents.choose_action(
                        obs[agent_id],
                        last_action[agent_id],
                        agent_id,
                        avail_action,
                        epsilon,
                        timestep_cur=step,
                        timestep_max=self.args.n_steps,
                        msg=None if msg is None else msg[agent_id],
                    )
                if low_continuous:
                    action = np.asarray(action, dtype=np.float32).reshape(-1)
                    if action.size < low_action_dim:
                        action = np.pad(action, (0, low_action_dim - action.size))
                    action = np.clip(action[:low_action_dim], -1.0, 1.0).astype(np.float32)
                    action_onehot = action.copy()
                    actions.append(action)
                else:
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1
                    actions.append(np.int_(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot
            raw_actions = (
                np.asarray(actions, dtype=np.float32).reshape(self.n_agents, low_action_dim)
                if low_continuous
                else np.reshape(actions, [self.n_agents, 1])
            )
            raw_log_probs_step = None
            if cbf_flow and hasattr(self.agents, "get_low_action_log_probs"):
                raw_log_probs_step = self.agents.get_low_action_log_probs(raw_obs, raw_actions)
            if raw_log_probs_step is None:
                raw_log_probs_step = np.zeros((self.n_agents, 1), dtype=np.float32)

            if hasattr(self.agents, "revise_safe_actions"):
                revised_actions = self.agents.revise_safe_actions(
                    observations=raw_obs,
                    avail_actions=avail_actions,
                    base_actions=actions,
                )
                if revised_actions is not None:
                    if cbf_flow and low_continuous:
                        actions = [
                            np.asarray(action, dtype=np.float32).reshape(-1)[:low_action_dim]
                            for action in revised_actions
                        ]
                        actions = [
                            action if active_agent_mask[agent_id] > 0.0 else np.asarray(noop_action, dtype=np.float32)
                            for agent_id, action in enumerate(actions)
                        ]
                    elif not low_continuous:
                        actions = [int(action) for action in revised_actions]
                        actions = [
                            action if active_agent_mask[agent_id] > 0.0 else noop_action
                            for agent_id, action in enumerate(actions)
                        ]
                        actions_onehot = []
                        for action in actions:
                            action_onehot = np.zeros(self.args.n_actions)
                            action_onehot[action] = 1
                            actions_onehot.append(action_onehot)
                        for agent_id in range(self.n_agents):
                            last_action[agent_id] = actions_onehot[agent_id]
            guard_flags = np.asarray(
                getattr(self.agents, "last_guard_applied", [0 for _ in range(self.n_agents)]),
                dtype=np.float32,
            ).reshape(self.n_agents, 1)
            guard_flags *= active_agent_mask.reshape(self.n_agents, 1)
            correct_actions = (
                np.asarray(actions, dtype=np.float32).reshape(self.n_agents, low_action_dim)
                if low_continuous
                else np.reshape(actions, [self.n_agents, 1])
            )
            correction_step = correct_actions - raw_actions

            reward, terminated, info = self.env.step(actions)
            if (
                cbf_flow
                and hasattr(self.agents, "energy_records")
                and isinstance(info, dict)
            ):
                self.agents.energy_records.extend(info.get("energy_transitions", []))
            log_reward = float(np.asarray(reward, dtype=np.float32).mean())

            if self.args.alg.find("Comm") != -1:
                obs, state = self.agents.obs_state_comm(reward)

            warning_signal = info.get("warning_signal", np.zeros((self.n_agents, 1)))
            per_agent_reward = info.get("per_agent_reward", None)
            cost_signal = np.asarray(warning_signal, dtype=np.float32).reshape(-1)
            if cost_signal.size == 0:
                cost_signal = np.zeros(self.n_agents, dtype=np.float32)
            elif cost_signal.size == 1:
                cost_signal = np.repeat(cost_signal.item(), self.n_agents).astype(
                    np.float32
                )
            elif cost_signal.size != self.n_agents:
                cost_signal = np.resize(cost_signal, self.n_agents).astype(np.float32)
            
            win_tag = terminated and info.get("battle_won", False)
            boost_task = 5 if win_tag else 1
            if uses_policy_gradient_reward and per_agent_reward is not None:
                utility = boost_task * np.asarray(
                    per_agent_reward, dtype=np.float32
                ).reshape(self.n_agents)
                utility *= active_agent_mask
                if apply_warning_reshape:
                    warning_signal_weight = _get_warning_penalty_weight(self.args)
                    utility -= warning_signal_weight * np.asarray(
                        warning_signal, dtype=np.float32
                    ).reshape(self.n_agents)
                reward_for_batch = utility.astype(np.float32)
            else:
                utility = boost_task * reward
                # reward reshape via punishment function
                if apply_warning_reshape:
                    warning_signal_weight = _get_warning_penalty_weight(self.args)
                    utility -= warning_signal_weight * np.sum(warning_signal)
                reward_for_batch = np.array([utility], dtype=np.float32)

            # 针对环境修正 reward 标准化
            if self.args.map == "Basic2P":
                utility *= 0.005
                reward *= 0.005
                reward_for_batch = np.array([utility], dtype=np.float32)

            if apply_policy_arrival_penalty and not penalty_applied:
                cur_positions = [player.pos for player in self.env.players]
                cur_succeed = [player.succeed for player in self.env.players]
                arrived_indices = [
                    idx
                    for idx, (prev_s, cur_s) in enumerate(zip(prev_succeed, cur_succeed))
                    if cur_s and not prev_s
                ]
                if arrived_indices:
                    stayed_indices = [
                        idx
                        for idx, (prev_pos, cur_pos) in enumerate(
                            zip(prev_positions, cur_positions)
                        )
                        if prev_pos == cur_pos
                    ]
                    if any(
                        idx in stayed_indices
                        for idx in range(self.n_agents)
                        if idx not in arrived_indices
                    ):
                        utility += penalty_value
                        reward += penalty_value
                        penalty_applied = True
            if reward_for_batch.shape == (1,):
                reward_for_batch = np.array([utility], dtype=np.float32)

            external_reward_for_high = np.asarray(
                reward_for_batch, dtype=np.float32
            ).copy()
            if level_training:
                external_reward_for_high = np.zeros(self.n_agents, dtype=np.float32)
            if level_training and current_high_transition is not None:
                reward_values = np.asarray(
                    external_reward_for_high, dtype=np.float32
                ).reshape(-1)
                if reward_values.size == self.n_agents:
                    high_step_reward = reward_values * active_agent_mask
                else:
                    high_step_reward = (
                        np.ones(self.n_agents, dtype=np.float32)
                        * float(np.mean(reward_values))
                        * active_agent_mask
                    )
                duration = int(current_high_transition.get("duration", 0))
                current_high_reward += (
                    (float(self.args.gamma) ** duration)
                    * high_step_reward.astype(np.float32)
                )
                current_high_transition["duration"] = duration + 1

            o.append(obs)
            o_raw.append(raw_obs)
            s.append(state)
            if low_continuous:
                u.append(raw_actions if cbf_flow else correct_actions)
            else:
                u.append(np.reshape(actions, [self.n_agents, 1]))
            u_raw.append(raw_actions.copy())
            u_correct.append(correct_actions.copy())
            raw_log_prob.append(np.asarray(raw_log_probs_step, dtype=np.float32).reshape(self.n_agents, 1))
            correction_delta.append(correction_step.copy())
            intervention_masks.append(guard_flags.copy())
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            active_masks.append(active_agent_mask.reshape(self.n_agents, 1).copy())
            reward_template = reward_for_batch.copy()
            r.append(reward_for_batch.copy())  # utility 用于学习，episode_reward 单独记录环境原始回报
            if use_constraint_cost:
                c.append(cost_signal.copy())
            w_signal.append(warning_signal)
            guard_applied.append(guard_flags.copy())
            terminate.append([terminated])
            padded.append([0.0])
            episode_reward += log_reward
            step += 1
            if self.args.epsilon_anneal_scale == "step":
                epsilon = (
                    epsilon - self.anneal_epsilon
                    if epsilon > self.min_epsilon
                    else epsilon
                )
        # last obs
        obs = self.env.get_obs()
        raw_obs = np.asarray(obs, dtype=np.float32).copy()
        state = self.env.get_state()  # flattened numpy array
        if self.args.alg.find("Comm") != -1:
            obs, state = self.agents.obs_state_comm()

        if level_training and current_high_transition is not None:
            high_reward = current_high_reward.copy()
            final_high_terminal = bool(terminated or step >= self.episode_limit)
            high_reward -= _terminal_depleted_high_cost(info, final_high_terminal)
            high_o.append(current_high_transition["o"])
            high_s.append(current_high_transition["s"])
            high_u.append(_high_action_for_training(current_high_transition))
            high_u_raw.append(current_high_transition["u_raw"])
            high_avail_u.append(current_high_transition["avail_u"])
            high_active_masks.append(current_high_transition["active_mask"])
            high_mode_train_masks.append(current_high_transition["mode_train_mask"])
            high_intervention_masks.append(current_high_transition["intervention_mask"])
            high_durations.append([float(max(1, current_high_transition["duration"]))])
            high_energy_margins.append(current_high_transition["energy_margin"])
            high_energy_order_masks.append(current_high_transition["energy_order_mask"])
            high_r.append(high_reward)
            high_o_next.append(self.env.get_high_level_obs())
            high_s_next.append(self.env.get_high_level_state())
            high_terminate.append([final_high_terminal])
            high_padded.append([0.0])

        o.append(obs)
        o_raw.append(raw_obs)
        s.append(state)
        o_next = o[1:]
        o_next_raw = o_raw[1:]
        s_next = s[1:]
        o = o[:-1]
        o_raw = o_raw[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            o_raw.append(np.zeros((self.n_agents, getattr(self.args, "raw_obs_shape", self.obs_shape))))
            u.append(np.zeros((self.n_agents, low_action_dim if low_continuous else 1)))
            u_raw.append(np.zeros((self.n_agents, low_action_dim if low_continuous else 1), dtype=np.float32))
            u_correct.append(np.zeros((self.n_agents, low_action_dim if low_continuous else 1), dtype=np.float32))
            raw_log_prob.append(np.zeros((self.n_agents, 1), dtype=np.float32))
            correction_delta.append(np.zeros((self.n_agents, low_action_dim if low_continuous else 1), dtype=np.float32))
            intervention_masks.append(np.zeros((self.n_agents, 1), dtype=np.float32))
            s.append(np.zeros(self.state_shape))
            r.append(np.zeros_like(reward_template))
            if use_constraint_cost:
                c.append(np.zeros(self.n_agents, dtype=np.float32))
            w_signal.append(np.zeros((self.n_agents, 1)))
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            o_next_raw.append(np.zeros((self.n_agents, getattr(self.args, "raw_obs_shape", self.obs_shape))))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            guard_applied.append(np.zeros((self.n_agents, 1), dtype=np.float32))
            active_masks.append(np.zeros((self.n_agents, 1), dtype=np.float32))
            padded.append([1.0])
            terminate.append([1.0])

        if level_training:
            high_obs_shape = int(getattr(self.args, "high_level_obs_shape", 0))
            high_state_shape = int(getattr(self.args, "high_level_state_shape", 0))
            high_n_actions = int(getattr(self.args, "high_level_n_actions", 0))
            high_mode_n_actions = int(
                getattr(self.args, "high_level_mode_n_actions", high_n_actions)
            )
            for _ in range(len(high_o), self.episode_limit):
                high_o.append(np.zeros((self.n_agents, high_obs_shape), dtype=np.float32))
                high_s.append(np.zeros(high_state_shape, dtype=np.float32))
                high_u.append(np.zeros((self.n_agents, high_n_actions), dtype=np.float32))
                high_u_raw.append(
                    np.zeros((self.n_agents, high_n_actions), dtype=np.float32)
                )
                high_r.append(np.zeros(self.n_agents, dtype=np.float32))
                high_avail_u.append(
                    np.zeros((self.n_agents, high_mode_n_actions), dtype=np.float32)
                )
                high_o_next.append(
                    np.zeros((self.n_agents, high_obs_shape), dtype=np.float32)
                )
                high_s_next.append(np.zeros(high_state_shape, dtype=np.float32))
                high_active_masks.append(
                    np.zeros((self.n_agents, 1), dtype=np.float32)
                )
                high_mode_train_masks.append(
                    np.zeros((self.n_agents, 1), dtype=np.float32)
                )
                high_intervention_masks.append(
                    np.zeros((self.n_agents, 1), dtype=np.float32)
                )
                high_durations.append([1.0])
                high_energy_margins.append(
                    np.zeros((self.n_agents, 1), dtype=np.float32)
                )
                high_energy_order_masks.append(
                    np.zeros((self.n_agents, 1), dtype=np.float32)
                )
                high_padded.append([1.0])
                high_terminate.append([1.0])

        episode = dict(
            o=o.copy(),
            o_raw=o_raw.copy(),
            s=s.copy(),
            u=u.copy(),
            r=r.copy(),
            warning_signal=w_signal.copy(),
            avail_u=avail_u.copy(),
            o_next=o_next.copy(),
            o_next_raw=o_next_raw.copy(),
            s_next=s_next.copy(),
            avail_u_next=avail_u_next.copy(),
            u_onehot=u_onehot.copy(),
            guard_applied=guard_applied.copy(),
            agent_active_mask=active_masks.copy(),
            u_raw=u_raw.copy(),
            u_correct=u_correct.copy(),
            raw_log_prob=raw_log_prob.copy(),
            correction_delta=correction_delta.copy(),
            intervention_mask=intervention_masks.copy(),
            padded=padded.copy(),
            terminated=terminate.copy(),
        )
        if level_training:
            episode.update(
                high_o=high_o.copy(),
                high_s=high_s.copy(),
                high_u=high_u.copy(),
                high_u_raw=high_u_raw.copy(),
                high_r=high_r.copy(),
                high_avail_u=high_avail_u.copy(),
                high_o_next=high_o_next.copy(),
                high_s_next=high_s_next.copy(),
                high_agent_active_mask=high_active_masks.copy(),
                high_mode_train_mask=high_mode_train_masks.copy(),
                high_intervention_mask=high_intervention_masks.copy(),
                high_duration=high_durations.copy(),
                high_energy_margin=high_energy_margins.copy(),
                high_energy_order_mask=high_energy_order_masks.copy(),
                high_padded=high_padded.copy(),
                high_terminated=high_terminate.copy(),
            )
        if use_constraint_cost:
            episode["c"] = c.copy()
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])  # type: ignore
        if not evaluate:
            self.epsilon = epsilon
        if self.args.alg == "maven":
            episode["z"] = np.array([maven_z.copy()])  # type: ignore
        if (
            evaluate
            and episode_num == self.args.evaluate_epoch - 1
            and self.args.replay_dir != ""
            and self.args.map in SMAC_MAPS
        ):
            self.env.save_replay()
            self.env.close()

        summary = _build_env_summary(
            self.env, info, step, win_tag, self.n_agents
        )

        return episode, episode_reward, summary, step, warning_signal


# RolloutWorker for communication
class CommRolloutWorker:
    def __init__(self, env, agents, args):
        self.env = env
        self.agents = agents
        self.episode_limit = args.episode_limit
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.state_shape = args.state_shape
        self.obs_shape = args.obs_shape
        self.args = args

        self.epsilon = args.epsilon
        self.anneal_epsilon = args.anneal_epsilon
        self.min_epsilon = args.min_epsilon
        self.train_episode_index = 0
        print("Init CommRolloutWorker")

    def _episode_seed(self, evaluate, episode_num):
        return derive_episode_seed(
            self.args,
            evaluate=evaluate,
            episode_index=episode_num if evaluate else self.train_episode_index,
        )

    def _reset_env_for_episode(self, evaluate, episode_num):
        seed = getattr(self, "_active_episode_seed", None)
        if seed is None:
            seed = self._episode_seed(evaluate, episode_num)

        if evaluate:
            return reset_env_with_seed(self.env, seed)

        with preserve_rng_state(include_torch=False):
            result = reset_env_with_seed(self.env, seed)
        self.train_episode_index += 1
        return result

    @torch.no_grad()
    def generate_episode(self, episode_num=None, evaluate=False):
        if evaluate and not getattr(self, "_inside_seeded_eval", False):
            seed = self._episode_seed(evaluate=True, episode_num=episode_num)
            with temporary_seed(seed, include_torch=True):
                self._inside_seeded_eval = True
                self._active_episode_seed = seed
                try:
                    return self.generate_episode(episode_num=episode_num, evaluate=True)
                finally:
                    self._inside_seeded_eval = False
                    self._active_episode_seed = None

        if (
            self.args.replay_dir != "" and evaluate and episode_num == 0
        ):  # prepare for save replay
            self.env.close()
        o, u, r, s, avail_u, u_onehot, terminate, padded, active_masks = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        self._reset_env_for_episode(evaluate, episode_num)
        low_action_type = getattr(self.args, "low_action_type", "discrete")
        low_continuous = low_action_type == "continuous"
        low_action_dim = int(getattr(self.args, "low_action_dim", self.args.n_actions))
        terminated = False
        win_tag = False
        step = 0
        episode_reward = 0
        last_action = np.zeros((self.args.n_agents, self.args.n_actions))
        self.agents.policy.init_hidden(1)
        epsilon = 0 if evaluate else self.epsilon
        if self.args.epsilon_anneal_scale == "episode":
            epsilon = (
                epsilon - self.anneal_epsilon if epsilon > self.min_epsilon else epsilon
            )
        reward_template = np.zeros((1,), dtype=np.float32)
        while not terminated and step < self.episode_limit:
            active_agent_mask = _get_active_agent_mask(self.env, self.n_agents)
            noop_action = _get_noop_action(self.env, self.n_actions)
            obs = self.env.get_obs()
            state = self.env.get_state()
            actions, avail_actions, actions_onehot = [], [], []

            # get the weights of all actions for all agents
            weights = self.agents.get_action_weights(
                np.array(obs),
                last_action,
                active_agent_mask=active_agent_mask,
            )

            # choose action for each agent
            for agent_id in range(self.n_agents):
                avail_action = self.env.get_avail_agent_actions(agent_id)
                if active_agent_mask[agent_id] <= 0.0:
                    action = (
                        np.asarray(noop_action, dtype=np.float32).copy()
                        if low_continuous
                        else noop_action
                    )
                else:
                    action = self.agents.choose_action(
                        weights[agent_id], avail_action, epsilon, step, self.args.n_steps
                    )

                if low_continuous:
                    action = np.asarray(action, dtype=np.float32).reshape(-1)
                    if action.size < low_action_dim:
                        action = np.pad(action, (0, low_action_dim - action.size))
                    action = np.clip(action[:low_action_dim], -1.0, 1.0).astype(np.float32)
                    action_onehot = action.copy()
                    actions.append(action)
                else:
                    # generate onehot vector of the action
                    action_onehot = np.zeros(self.args.n_actions)
                    action_onehot[action] = 1
                    actions.append(np.int_(action))
                actions_onehot.append(action_onehot)
                avail_actions.append(avail_action)
                last_action[agent_id] = action_onehot

            reward, terminated, info = self.env.step(actions)
            win_tag = terminated and info.get("battle_won", False)
            o.append(obs)
            s.append(state)
            if low_continuous:
                u.append(
                    np.asarray(actions, dtype=np.float32).reshape(
                        self.n_agents, low_action_dim
                    )
                )
            else:
                u.append(np.reshape(actions, [self.n_agents, 1]))
            u_onehot.append(actions_onehot)
            avail_u.append(avail_actions)
            active_masks.append(active_agent_mask.reshape(self.n_agents, 1).copy())
            r.append([reward])
            reward_template = np.array([reward], dtype=np.float32)
            terminate.append([terminated])
            padded.append([0.0])
            episode_reward += reward
            step += 1
            if self.args.epsilon_anneal_scale == "step":
                epsilon = (
                    epsilon - self.anneal_epsilon
                    if epsilon > self.min_epsilon
                    else epsilon
                )
        # last obs
        obs = self.env.get_obs()
        state = self.env.get_state()
        o.append(obs)
        s.append(state)
        o_next = o[1:]
        s_next = s[1:]
        o = o[:-1]
        s = s[:-1]
        # get avail_action for last obs，because target_q needs avail_action in training
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_action = self.env.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_action)
        avail_u.append(avail_actions)
        avail_u_next = avail_u[1:]
        avail_u = avail_u[:-1]

        # if step < self.episode_limit，padding
        for i in range(step, self.episode_limit):
            o.append(np.zeros((self.n_agents, self.obs_shape)))
            u.append(np.zeros((self.n_agents, low_action_dim if low_continuous else 1)))
            s.append(np.zeros(self.state_shape))
            r.append(np.zeros_like(reward_template))
            o_next.append(np.zeros((self.n_agents, self.obs_shape)))
            s_next.append(np.zeros(self.state_shape))
            u_onehot.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u.append(np.zeros((self.n_agents, self.n_actions)))
            avail_u_next.append(np.zeros((self.n_agents, self.n_actions)))
            active_masks.append(np.zeros((self.n_agents, 1), dtype=np.float32))
            padded.append([1.0])
            terminate.append([1.0])

        episode = dict(
            o=o.copy(),
            s=s.copy(),
            u=u.copy(),
            r=r.copy(),
            avail_u=avail_u.copy(),
            o_next=o_next.copy(),
            s_next=s_next.copy(),
            avail_u_next=avail_u_next.copy(),
            u_onehot=u_onehot.copy(),
            agent_active_mask=active_masks.copy(),
            padded=padded.copy(),
            terminated=terminate.copy(),
        )
        # add episode dim
        for key in episode.keys():
            episode[key] = np.array([episode[key]])  # type: ignore
        if not evaluate:
            self.epsilon = epsilon
        if (
            evaluate
            and episode_num == self.args.evaluate_epoch - 1
            and self.args.replay_dir != ""
        ):
            self.env.save_replay()
            self.env.close()
        # return episode, episode_reward, win_tag, step
        summary = _build_env_summary(
            self.env, info, step, 1 if win_tag else 0, self.n_agents
        )
        warning_signal = [0.0]
        return episode, episode_reward, summary, step, warning_signal
