import numpy as np

from envs.UAVEnergyDelivery import (
    UAVEnvDiscreteWrapper as BaseUAVEnvDiscreteWrapper,
    DeliveryOrder,
    UAVEnv,
    boundary_height,
    boundary_length,
    boundary_width,
    default_charging_capacity,
    default_charging_radius,
    default_charging_station_count,
    default_energy_depletion_fraction,
    default_initial_energy,
    default_num_obstacles,
    eps,
)


class HierarchicalUAVEnv(UAVEnv):
    """Adapter over the flat UAVEnergyDelivery environment.

    The high-level interface is intentionally empty. The base geometry, order
    sampling, charging, collision handling, energy accounting, and environment
    reward terms come from UAVEnergyDelivery.py. New high-level modules can be
    added here without touching the flat environment.
    """

    def __init__(
        self,
        dim_actions,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=6,
        num_targets=1,
        episode_limit=200,
        reset_retry_limit=20,
        sample_retry_limit=100,
        obstacle_crash_penalty=10.0,
        total_orders=16,
        max_active_orders=8,
        pickup_reward=3.0,
        delivery_reward=8.0,
        initial_energy=default_initial_energy,
        energy_decay_per_step=None,
        energy_depletion_fraction=default_energy_depletion_fraction,
        charging_capacity=default_charging_capacity,
        charging_station_count=default_charging_station_count,
        charging_radius=default_charging_radius,
        charging_rate=None,
        charging_station_pos=None,
        charge_mode_fraction=0.5,
        high_goal_style="line",
        high_mode_policy="hybrid",
        high_lateral_scale=0.35,
    ):
        super().__init__(
            dim_actions=dim_actions,
            length=length,
            width=width,
            height=height,
            num_obstacle=num_obstacle,
            num_hunters=num_hunters,
            num_targets=num_targets,
            episode_limit=episode_limit,
            reset_retry_limit=reset_retry_limit,
            sample_retry_limit=sample_retry_limit,
            obstacle_crash_penalty=obstacle_crash_penalty,
            total_orders=total_orders,
            max_active_orders=max_active_orders,
            pickup_reward=pickup_reward,
            delivery_reward=delivery_reward,
            initial_energy=initial_energy,
            energy_decay_per_step=energy_decay_per_step,
            energy_depletion_fraction=energy_depletion_fraction,
            charging_capacity=charging_capacity,
            charging_station_count=charging_station_count,
            charging_radius=charging_radius,
            charging_rate=charging_rate,
            charging_station_pos=charging_station_pos,
        )
        self.meta_period = 5
        self.high_level_mode_n_actions = 0
        self.high_level_n_actions = 0
        self.high_level_obs_shape = 0
        self.high_level_state_shape = 0
        self.low_task_shape = 0

        self._last_high_mode_train_mask = np.zeros((self.num_agents, 1), dtype=np.float32)
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_reward_terms = {}
        self._init_hierarchical_agent_state()

    def _init_hierarchical_agent_state(self):
        for agent in self.agents:
            agent.reached = False

    def set_meta_period(self, meta_period):
        self.meta_period = max(1, int(meta_period))

    def set_hrl_parameters(
        self,
        reachable_subgoal_scale=None,
        intrinsic_reward_scale=None,
        intrinsic_success_bonus=None,
        high_goal_style=None,
        high_lateral_scale=None,
        order_progress_override=None,
        energy_margin_reserve_ratio=None,
        charge_energy_threshold=None,
        charge_release_threshold=None,
        charge_queue_enabled=None,
        charge_queue_radius=None,
    ):
        del (
            reachable_subgoal_scale,
            intrinsic_reward_scale,
            intrinsic_success_bonus,
            high_goal_style,
            high_lateral_scale,
            order_progress_override,
            energy_margin_reserve_ratio,
            charge_energy_threshold,
            charge_release_threshold,
            charge_queue_enabled,
            charge_queue_radius,
        )
        return None

    def reset(self, seed=None):
        obs = super().reset(seed=seed)
        self._init_hierarchical_agent_state()
        self._last_high_mode_train_mask = np.zeros((self.num_agents, 1), dtype=np.float32)
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_reward_terms = {}
        return obs

    def _assign_orders(self):
        self._activate_orders()

    def _set_agent_idle(self, agent):
        super()._set_agent_idle(agent)
        agent.reached = False
        return agent.goal.copy()

    def prepare_high_level_decision(self):
        return None

    def apply_high_level_actions(self, actions):
        del actions
        self._last_high_mode_train_mask = np.zeros((self.num_agents, 1), dtype=np.float32)
        return np.zeros((self.num_agents, 0), dtype=np.float32)

    def _agent_has_motion_task(self, agent):
        return self._agent_is_active(agent)

    def _advance_order_if_reached(self, agent, current_dist=None):
        return super()._advance_order_if_reached(agent, current_dist)

    def _consume_step_energy(self, powered_mask, actions=None):
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        for agent_idx, (is_powered, agent) in enumerate(zip(powered_mask, self.agents)):
            if not is_powered:
                continue
            step_energy = float(self.energy_decay_per_step)
            self._last_step_energy_ratio[agent_idx] = step_energy / (
                agent.initial_energy + eps
            )
            agent.consume_energy(step_energy)

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        self.current_step += 1
        self._activate_orders()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        reward_terms = {
            "progress": np.zeros(self.num_agents, dtype=np.float32),
            "velocity_toward_goal": np.zeros(self.num_agents, dtype=np.float32),
            "time": np.zeros(self.num_agents, dtype=np.float32),
            "obstacle_collision": np.zeros(self.num_agents, dtype=np.float32),
            "agent_collision": np.zeros(self.num_agents, dtype=np.float32),
            "pickup": np.zeros(self.num_agents, dtype=np.float32),
            "delivery": np.zeros(self.num_agents, dtype=np.float32),
            "all_orders_completed": np.zeros(self.num_agents, dtype=np.float32),
        }
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
        powered_mask = np.asarray(
            [self._agent_is_active(agent) for agent in self.agents], dtype=bool
        )
        prev_dists = np.array(
            [self._distance_to_goal(agent) for agent in self.agents], dtype=np.float32
        )
        order_status_before = []
        for agent in self.agents:
            if agent.assigned_order_id is None:
                order_status_before.append(None)
            else:
                order_status_before.append(self.orders[agent.assigned_order_id].status)

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if (
                not powered_mask[idx]
                or not self._agent_has_motion_task(agent)
            ):
                agent.vel[:] = 0.0
                agent.prev_pos = agent.pos.copy()
                continue
            agent.update_velocity(action, self.time_step)
            agent.preview_position(self.time_step)

        obstacle_collisions = [False] * self.num_agents
        for idx, agent in enumerate(self.agents):
            if not powered_mask[idx]:
                agent.vel[:] = 0.0
                agent.prev_pos = agent.pos.copy()
                continue
            (
                boundary_collision,
                boundary_penalty,
                boundary_reward_penalty,
            ) = self._apply_boundary_constraints(agent)
            (
                obstacle_collision,
                obstacle_penalty,
                obstacle_reward_penalty,
            ) = self._resolve_obstacle_collisions(agent)
            obstacle_collisions[idx] = boundary_collision or obstacle_collision
            self.safe_value[idx] += boundary_penalty + obstacle_penalty
            self.reward_safe_value[idx] += boundary_reward_penalty + obstacle_reward_penalty

        agent_collisions = self._resolve_agent_collisions()

        for agent in self.agents:
            agent.pos = agent.prev_pos.copy()

        self._consume_step_energy(powered_mask, actions)
        depleted_penalties = self._deactivate_depleted_agents()
        rewards += depleted_penalties
        self._charge_agents_at_station()
        self.update_lasers()

        for idx, agent in enumerate(self.agents):
            if not powered_mask[idx] or not self._agent_is_active(agent):
                agent.vel[:] = 0.0
                agent.prev_collided = agent.collided
                agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
                continue

            current_dist = self._distance_to_goal(agent)
            progress = prev_dists[idx] - current_dist
            goal_direction = agent.goal - agent.pos
            goal_direction_norm = np.linalg.norm(goal_direction)
            if goal_direction_norm > eps:
                goal_direction = goal_direction / goal_direction_norm
                velocity_toward_goal = float(np.dot(agent.vel, goal_direction)) / (
                    agent.v_max + eps
                )
            else:
                velocity_toward_goal = 0.0
            obstacle_penalty = self.obstacle_collision_penalty
            agent_penalty = self.agent_collision_penalty
            if agent.prev_collided:
                obstacle_penalty *= self.repeat_collision_scale
                agent_penalty *= self.repeat_collision_scale

            progress_reward = 2.5 * progress
            velocity_reward = self.velocity_reward_weight * max(0.0, velocity_toward_goal)
            time_penalty = -0.01
            obstacle_collision_penalty = -obstacle_penalty * float(obstacle_collisions[idx])
            agent_collision_penalty = -agent_penalty * float(agent_collisions[idx])
            reward_terms["progress"][idx] = progress_reward
            reward_terms["velocity_toward_goal"][idx] = velocity_reward
            reward_terms["time"][idx] = time_penalty
            reward_terms["obstacle_collision"][idx] = obstacle_collision_penalty
            reward_terms["agent_collision"][idx] = agent_collision_penalty
            rewards[idx] += (
                progress_reward
                + velocity_reward
                + time_penalty
                + obstacle_collision_penalty
                + agent_collision_penalty
            )

            if current_dist <= self.goal_tolerance:
                agent.reached = True

            prev_status = order_status_before[idx]
            order_reward = self._advance_order_if_reached(agent, current_dist)
            if order_reward > 0.0:
                agent.vel[:] = 0.0
                if prev_status == DeliveryOrder.ASSIGNED:
                    reward_terms["pickup"][idx] = order_reward
                elif prev_status == DeliveryOrder.PICKED:
                    reward_terms["delivery"][idx] = order_reward
                rewards[idx] += order_reward
            agent.prev_collided = agent.collided
            agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
            self.agent_paths[idx].append(agent.pos.copy())

        self._activate_orders()
        delivery_done = self._all_orders_completed()
        dones = [not self._agent_is_active(agent) for agent in self.agents]

        obstacle_collision_total = float(np.sum(np.asarray(obstacle_collisions, dtype=bool)))
        agent_collision_total = float(np.sum(np.asarray(agent_collisions, dtype=bool)))
        self.obstacle_collision_count += obstacle_collision_total
        self.agent_collision_count += agent_collision_total
        self.collision_count += obstacle_collision_total + agent_collision_total

        if delivery_done:
            active_mask = self.get_active_agent_mask()
            reward_terms["all_orders_completed"] += 5.0 * active_mask
            rewards += 5.0 * active_mask

        self._last_reward_terms = {
            name: values.astype(np.float32).copy() for name, values in reward_terms.items()
        }
        return self.get_obs(), rewards, dones, self.safe_value.copy()

    def get_obs(self):
        return super().get_obs()

    def _high_level_agent_obs(self, agent):
        del agent
        return np.zeros((0,), dtype=np.float32)

    def get_high_level_obs(self):
        return np.zeros((self.num_agents, self.high_level_obs_shape), dtype=np.float32)

    def get_high_level_state(self):
        return np.zeros((self.high_level_state_shape,), dtype=np.float32)

    def get_high_level_avail_agent_actions(self, agent_id):
        del agent_id
        return np.zeros((self.high_level_mode_n_actions,), dtype=np.float32)

    def get_high_level_avail_actions(self):
        return np.stack(
            [self.get_high_level_avail_agent_actions(i) for i in range(self.num_agents)],
            axis=0,
        )

    def get_high_level_energy_margins(self):
        return np.zeros((self.num_agents, 1), dtype=np.float32)

    def get_high_level_energy_order_masks(self):
        return np.zeros((self.num_agents, 1), dtype=np.float32)

    def get_high_level_mode_training_mask(self):
        return np.asarray(self._last_high_mode_train_mask, dtype=np.float32).copy()

    def get_oracle_high_level_actions(self):
        actions = np.zeros((self.num_agents, self.high_level_n_actions), dtype=np.float32)
        return actions

    def relabel_high_level_actions_with_achieved(
        self,
        start_positions,
        end_positions,
        actions,
        active_mask=None,
    ):
        del start_positions, end_positions, active_mask
        actions = np.asarray(actions, dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(self.num_agents, -1)
        return actions.astype(np.float32)


class UAVEnvDiscreteWrapper(BaseUAVEnvDiscreteWrapper):
    def __init__(
        self,
        dim_actions=3,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=4,
        num_targets=1,
        episode_limit=200,
        total_orders=16,
        max_active_orders=8,
        pickup_reward=3.0,
        delivery_reward=8.0,
        initial_energy=default_initial_energy,
        energy_decay_per_step=None,
        energy_depletion_fraction=default_energy_depletion_fraction,
        charging_capacity=default_charging_capacity,
        charging_station_count=default_charging_station_count,
        charging_radius=default_charging_radius,
        charging_rate=None,
        charging_station_pos=None,
        charge_mode_fraction=0.5,
        high_goal_style="line",
        high_mode_policy="hybrid",
        high_lateral_scale=0.35,
        **unused_kwargs,
    ):
        self.env = HierarchicalUAVEnv(
            dim_actions=dim_actions,
            length=length,
            width=width,
            height=height,
            num_obstacle=num_obstacle,
            num_hunters=num_hunters,
            num_targets=num_targets,
            episode_limit=episode_limit,
            total_orders=total_orders,
            max_active_orders=max_active_orders,
            pickup_reward=pickup_reward,
            delivery_reward=delivery_reward,
            initial_energy=initial_energy,
            energy_decay_per_step=energy_decay_per_step,
            energy_depletion_fraction=energy_depletion_fraction,
            charging_capacity=charging_capacity,
            charging_station_count=charging_station_count,
            charging_radius=charging_radius,
            charging_rate=charging_rate,
            charging_station_pos=charging_station_pos,
            charge_mode_fraction=charge_mode_fraction,
            high_goal_style=high_goal_style,
            high_mode_policy=high_mode_policy,
            high_lateral_scale=high_lateral_scale,
        )
        self.dim_actions = dim_actions
        self.episode_limit = episode_limit
        self.n_agents = self.env.num_agents
        self.low_action_type = "continuous"
        self.action_dim = self.dim_actions
        self.n_actions = self.action_dim
        self._episode_steps = 0
        self._last_obs = None
        self._last_reward = 0.0
        self._last_info = {}

    def __getattr__(self, name):
        if name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)

    def step(self, actions):
        reward, terminated, info = super().step(actions)
        info["per_agent_reward_terms"] = {
            name: np.asarray(values, dtype=np.float32).copy()
            for name, values in getattr(self.env, "_last_reward_terms", {}).items()
        }
        return reward, terminated, info

    def get_env_info(self):
        info = super().get_env_info()
        info.update(
            {
                "high_level_n_actions": int(self.env.high_level_n_actions),
                "high_level_mode_n_actions": int(self.env.high_level_mode_n_actions),
                "high_level_obs_shape": int(self.env.get_high_level_obs().shape[-1]),
                "high_level_state_shape": int(
                    self.env.get_high_level_state().shape[-1]
                ),
                "low_task_shape": int(self.env.low_task_shape),
                "max_active_orders": int(self.env.max_active_orders),
                "charge_action_id": -1,
            }
        )
        return info

    def apply_high_level_actions(self, actions):
        applied = self.env.apply_high_level_actions(actions)
        self._last_obs = self.env.get_obs()
        return applied

    def revise_safe_actions(
        self,
        actions,
        avail_actions=None,
        guard_margin=None,
        guard_horizon=None,
    ):
        del avail_actions, guard_margin, guard_horizon
        return list(actions), np.zeros(self.n_agents, dtype=np.float32)


def parallel_env(**kwargs):
    return UAVEnvDiscreteWrapper(**kwargs)
