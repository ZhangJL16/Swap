import numpy as np

from hierarchy.high_level_controller import HighLevelController, order_features
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
        cbf_flow_enabled=False,
        agent_entry_interval=1,
        order_max_duration=120,
        energy_reserve=0.0,
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
        self.cbf_flow_enabled = bool(cbf_flow_enabled)
        if self.cbf_flow_enabled:
            self.max_active_orders = max(1, int(max_active_orders))
        self.agent_entry_interval = max(1, int(agent_entry_interval))
        self.order_max_duration = max(1, int(order_max_duration))
        self.energy_reserve = float(max(0.0, energy_reserve))
        self.high_level_mode_n_actions = 0
        self.high_level_n_actions = 0
        self.high_level_obs_shape = 0
        self.high_level_state_shape = 0
        self.low_task_shape = 0

        self._last_high_mode_train_mask = np.zeros((self.num_agents, 1), dtype=np.float32)
        self._last_step_energy_ratio = np.zeros(self.num_agents, dtype=np.float32)
        self._last_reward_terms = {}
        self.high_controller = HighLevelController(reserve=self.energy_reserve)
        self._high_decision_mask = np.zeros(self.num_agents, dtype=np.float32)
        self._completed_option_transitions = []
        self._last_energy_transitions = []
        self._entry_prepared_step = -1
        self._next_agent_to_enter = 0
        self._init_hierarchical_agent_state()

    def _init_hierarchical_agent_state(self):
        for agent in self.agents:
            agent.reached = False
            if self.cbf_flow_enabled:
                agent.entered = False
                agent.active = False
                agent.exit_reason = "not_entered"
                agent.current_option_type = None
                agent.current_station_id = None
                agent.option_start_step = 0
                agent.option_reward = 0.0
                agent.option_start_uav_state = np.zeros(6, dtype=np.float32)
                agent.option_order_features = None

    def set_meta_period(self, meta_period):
        self.meta_period = max(1, int(meta_period))

    def set_hrl_parameters(
        self,
        reachable_subgoal_scale=None,
        intrinsic_reward_scale=None,
        intrinsic_success_bonus=None,
        intrinsic_collision_penalty=None,
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
            intrinsic_collision_penalty,
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
        self._completed_option_transitions = []
        self._last_energy_transitions = []
        self._high_decision_mask = np.zeros(self.num_agents, dtype=np.float32)
        self._entry_prepared_step = -1
        self._next_agent_to_enter = 0
        if self.cbf_flow_enabled:
            self._reset_agents_to_entry_station()
            self.update_lasers()
            obs = self.get_obs()
        return obs

    def _assign_orders(self):
        self._activate_orders()

    def _activate_orders(self):
        if not self.cbf_flow_enabled:
            return super()._activate_orders()
        while self._active_order_count() < self.max_active_orders:
            order = self._sample_delivery_order(self.next_order_id_to_activate)
            order.time_limit = float(self.order_max_duration)
            order.status = DeliveryOrder.ACTIVE
            order.assigned_agent = None
            self.orders.append(order)
            self.active_order_ids.append(order.order_id)
            self.next_order_id_to_activate += 1
        self._sync_goals_from_orders()

    def _all_orders_completed(self):
        if self.cbf_flow_enabled:
            return False
        return super()._all_orders_completed()

    def summary(self):
        summary = super().summary()
        if self.cbf_flow_enabled:
            summary["total_orders"] = float(self.next_order_id_to_activate)
            summary["win_tag"] = False
        return summary

    def _reset_agents_to_entry_station(self):
        station = self.charging_station_positions[0].astype(np.float32)
        self.agent_paths = [[] for _ in range(self.num_agents)]
        for agent in self.agents:
            agent.pos = station.copy()
            agent.spawn_pos = station.copy()
            agent.prev_pos = station.copy()
            agent.last_pos = station.copy()
            agent.goal = station.copy()
            agent.vel[:] = 0.0
            agent.energy = float(agent.initial_energy)
            agent.active = False
            agent.entered = False
            agent.exit_reason = "not_entered"
            agent.reached = False
            agent.assigned_order_id = None
            agent.carrying_order = False
            agent.current_option_type = None
            agent.current_station_id = None
            agent.option_start_step = 0
            agent.option_reward = 0.0
            agent.option_start_uav_state = np.concatenate([agent.pos, agent.vel]).astype(np.float32)
            agent.option_order_features = None
            self.agent_paths[agent.number] = [agent.pos.copy()]

    def prepare_cbf_flow_step(self):
        if not self.cbf_flow_enabled:
            return
        if self._entry_prepared_step == self.current_step:
            return
        if (
            self._next_agent_to_enter < self.num_agents
            and self.current_step % self.agent_entry_interval == 0
        ):
            agent = self.agents[self._next_agent_to_enter]
            station = self.charging_station_positions[0].astype(np.float32)
            agent.pos = station.copy()
            agent.prev_pos = station.copy()
            agent.last_pos = station.copy()
            agent.goal = station.copy()
            agent.vel[:] = 0.0
            agent.energy = float(agent.initial_energy)
            agent.active = True
            agent.entered = True
            agent.exit_reason = None
            agent.current_option_type = None
            self._high_decision_mask[agent.number] = 1.0
            self._next_agent_to_enter += 1
        self._activate_orders()
        self._process_high_decisions()
        self._entry_prepared_step = self.current_step

    def get_high_decision_mask(self):
        return self._high_decision_mask.astype(np.float32).copy()

    def get_visible_available_orders(self, agent_id):
        del agent_id
        return [
            self.orders[order_id]
            for order_id in self.active_order_ids
            if self.orders[order_id].status == DeliveryOrder.ACTIVE
            and self.orders[order_id].assigned_agent is None
        ]

    def assign_order(self, agent_id, order_id):
        agent = self.agents[int(agent_id)]
        if not self._agent_is_active(agent):
            return False
        order = self.orders[int(order_id)]
        if order.status != DeliveryOrder.ACTIVE or order.assigned_agent is not None:
            return False
        if not self._assign_order_to_agent(agent, order):
            return False
        agent.current_option_type = "order"
        agent.current_station_id = None
        agent.option_start_step = int(self.current_step)
        agent.option_reward = 0.0
        agent.option_start_uav_state = np.concatenate([agent.pos, agent.vel]).astype(np.float32)
        agent.option_order_features = order_features(order, self.order_max_duration)
        self._high_decision_mask[agent.number] = 0.0
        return True

    def assign_charge(self, agent_id, station_id):
        agent = self.agents[int(agent_id)]
        if not self._agent_is_active(agent):
            return False
        station_id = int(np.clip(station_id, 0, self.charging_station_count - 1))
        if agent.assigned_order_id is not None and not agent.carrying_order:
            order = self.orders[agent.assigned_order_id]
            if order.status == DeliveryOrder.ASSIGNED:
                order.status = DeliveryOrder.ACTIVE
                order.assigned_agent = None
            agent.assigned_order_id = None
        agent.carrying_order = False
        agent.current_option_type = "charge"
        agent.current_station_id = station_id
        agent.option_start_step = int(self.current_step)
        agent.option_reward = 0.0
        agent.option_start_uav_state = np.concatenate([agent.pos, agent.vel]).astype(np.float32)
        agent.option_order_features = None
        agent.goal = self.charging_station_positions[station_id].copy()
        agent.reached = False
        self._high_decision_mask[agent.number] = 0.0
        return True

    def get_completed_option_transitions(self):
        transitions = self._completed_option_transitions
        self._completed_option_transitions = []
        return transitions

    def _process_high_decisions(self):
        if not self.cbf_flow_enabled:
            return
        for agent_id in range(self.num_agents):
            if self._high_decision_mask[agent_id] <= 0.0:
                continue
            agent = self.agents[agent_id]
            if not self._agent_is_active(agent):
                continue
            visible_orders = self.get_visible_available_orders(agent_id)
            order = self.high_controller.select_order(
                agent,
                visible_orders,
                self.charging_station_positions,
                self.order_max_duration,
            )
            if order is not None:
                self.assign_order(agent_id, order.order_id)
            else:
                station_id = self.high_controller.select_min_energy_station(
                    agent, self.charging_station_positions
                )
                self.assign_charge(agent_id, station_id)

    def _deactivate_depleted_agents(self):
        if not self.cbf_flow_enabled:
            return super()._deactivate_depleted_agents()
        penalties = np.zeros(self.num_agents, dtype=np.float32)
        for agent in self.agents:
            if getattr(agent, "active", True) and not agent.has_energy():
                penalties[agent.number] = float(self.depleted_penalty)
                if agent.assigned_order_id is not None:
                    order = self.orders[agent.assigned_order_id]
                    if order.status != DeliveryOrder.COMPLETED:
                        order.status = DeliveryOrder.ACTIVE
                        order.assigned_agent = None
                agent.assigned_order_id = None
                agent.carrying_order = False
                agent.active = False
                agent.exit_reason = "depleted_exit"
                agent.vel[:] = 0.0
                self._high_decision_mask[agent.number] = 0.0
        return penalties

    def cbf_flow_all_entered_depleted(self):
        if not self.cbf_flow_enabled:
            return False
        entered = [bool(getattr(agent, "entered", False)) for agent in self.agents]
        if not all(entered):
            return False
        return all(
            (not entered[idx]) or (not self._agent_is_active(agent))
            for idx, agent in enumerate(self.agents)
        )

    def get_cbf_safety_state(self):
        parts = []
        for agent in self.agents:
            if self._agent_is_active(agent):
                parts.extend([agent.pos.astype(np.float32), agent.vel.astype(np.float32)])
            else:
                parts.extend(
                    [
                        np.zeros(self.dim_actions, dtype=np.float32),
                        np.zeros(self.dim_actions, dtype=np.float32),
                    ]
                )
        for obstacle in self.obstacles:
            pos = np.zeros(self.dim_actions, dtype=np.float32)
            pos[:2] = obstacle.pos.astype(np.float32)
            parts.append(pos)
            parts.append(np.asarray([obstacle.radius], dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def _agent_collision_exempt(self, agent):
        if not self.cbf_flow_enabled or not self._agent_is_active(agent):
            return False
        for station in self.charging_station_positions:
            if np.linalg.norm(agent.prev_pos - station) <= self.charging_radius:
                return True
        return False

    def _resolve_agent_collisions(self):
        if not self.cbf_flow_enabled:
            return super()._resolve_agent_collisions()
        collided = [False] * self.num_agents
        for i in range(self.num_agents):
            if not self._agent_is_active(self.agents[i]):
                continue
            for j in range(i + 1, self.num_agents):
                if not self._agent_is_active(self.agents[j]):
                    continue
                if self._agent_collision_exempt(self.agents[i]) and self._agent_collision_exempt(self.agents[j]):
                    continue
                delta = self.agents[i].prev_pos - self.agents[j].prev_pos
                dist = np.linalg.norm(delta)
                collision_min_dist = (
                    self.agents[i].safe_radius + self.agents[j].safe_radius
                )
                warning_min_dist = collision_min_dist + self.risk_warning_margin
                overlap = max(0.0, warning_min_dist - dist)
                pair_risk = self._nonlinear_risk_from_overlap(overlap, warning_min_dist)
                pair_reward_risk = self._nonlinear_risk_from_overlap(
                    max(0.0, collision_min_dist - dist), collision_min_dist
                )
                self.safe_value[i] += pair_risk
                self.safe_value[j] += pair_risk
                self.reward_safe_value[i] += pair_reward_risk
                self.reward_safe_value[j] += pair_reward_risk
                if dist < collision_min_dist:
                    collided[i] = True
                    collided[j] = True
                    if dist < eps:
                        delta = np.zeros(self.dim_actions, dtype=np.float32)
                        delta[0] = 1.0
                        dist = 1.0
                    direction = delta / dist
                    overlap = collision_min_dist - dist
                    self.agents[i].prev_pos += direction * (overlap / 2.0 + eps)
                    self.agents[j].prev_pos -= direction * (overlap / 2.0 + eps)
                    self.agents[i].vel[:] = 0.0
                    self.agents[j].vel[:] = 0.0
        return collided

    def _set_agent_idle(self, agent):
        super()._set_agent_idle(agent)
        agent.reached = False
        return agent.goal.copy()

    def prepare_high_level_decision(self):
        if self.cbf_flow_enabled:
            self._process_high_decisions()
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

        if self.cbf_flow_enabled:
            self.prepare_cbf_flow_step()
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
        energy_start_positions = [agent.pos.copy() for agent in self.agents]
        energy_start_velocities = [agent.vel.copy() for agent in self.agents]
        energy_targets = [agent.goal.copy() for agent in self.agents]
        energy_loaded_leg = [
            float(bool(getattr(agent, "carrying_order", False))) for agent in self.agents
        ]
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
                if self.cbf_flow_enabled and prev_status == DeliveryOrder.PICKED:
                    self._completed_option_transitions.append(
                        {
                            "agent_id": idx,
                            "uav_state": agent.option_start_uav_state.copy(),
                            "order": (
                                agent.option_order_features.copy()
                                if agent.option_order_features is not None
                                else np.zeros(7, dtype=np.float32)
                            ),
                            "option_return": float(agent.option_reward + order_reward),
                            "duration": float(max(1, self.current_step - agent.option_start_step)),
                            "next_uav_state": np.concatenate([agent.pos, agent.vel]).astype(np.float32),
                            "next_feasible_orders": [
                                order_features(order, self.order_max_duration)
                                for order in self.get_visible_available_orders(idx)
                            ],
                            "episode_done": False,
                        }
                    )
                    agent.current_option_type = None
                    agent.option_order_features = None
                    self._high_decision_mask[idx] = 1.0
            agent.prev_collided = agent.collided
            agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
            self.agent_paths[idx].append(agent.pos.copy())
            if self.cbf_flow_enabled:
                agent.option_reward = float(getattr(agent, "option_reward", 0.0) + rewards[idx])
                if (
                    agent.current_option_type == "charge"
                    and np.linalg.norm(agent.pos - self.charging_station_positions[int(agent.current_station_id)])
                    <= self.charging_radius
                    and agent.energy >= agent.initial_energy - eps
                ):
                    agent.current_option_type = None
                    self._high_decision_mask[idx] = 1.0
                elif (
                    agent.current_option_type is not None
                    and self.current_step - int(agent.option_start_step)
                    >= self.order_max_duration
                ):
                    if agent.assigned_order_id is not None and not agent.carrying_order:
                        order = self.orders[agent.assigned_order_id]
                        if order.status == DeliveryOrder.ASSIGNED:
                            order.status = DeliveryOrder.ACTIVE
                            order.assigned_agent = None
                        agent.assigned_order_id = None
                    agent.current_option_type = None
                    self._high_decision_mask[idx] = 1.0

        self._activate_orders()
        delivery_done = self._all_orders_completed()
        dones = [not self._agent_is_active(agent) for agent in self.agents]

        obstacle_collision_total = float(np.sum(np.asarray(obstacle_collisions, dtype=bool)))
        agent_collision_total = float(np.sum(np.asarray(agent_collisions, dtype=bool)))
        self.obstacle_collision_count += obstacle_collision_total
        self.agent_collision_count += agent_collision_total
        self.collision_count += obstacle_collision_total + agent_collision_total

        if delivery_done and not self.cbf_flow_enabled:
            active_mask = self.get_active_agent_mask()
            reward_terms["all_orders_completed"] += 5.0 * active_mask
            rewards += 5.0 * active_mask

        if self.cbf_flow_enabled:
            self._process_high_decisions()
            self._last_energy_transitions = []
            for idx, agent in enumerate(self.agents):
                if not powered_mask[idx]:
                    continue
                self._last_energy_transitions.append(
                    {
                        "position": energy_start_positions[idx],
                        "velocity": energy_start_velocities[idx],
                        "target": energy_targets[idx],
                        "step_energy": float(
                            self._last_step_energy_ratio[idx] * agent.initial_energy
                        ),
                        "next_position": agent.pos.copy(),
                        "next_velocity": agent.vel.copy(),
                        "goal_done": float(
                            np.linalg.norm(agent.pos - energy_targets[idx])
                            <= self.goal_tolerance
                        ),
                        "loaded_leg": energy_loaded_leg[idx],
                    }
                )

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
        cbf_flow_enabled=False,
        agent_entry_interval=1,
        order_max_duration=120,
        energy_reserve=0.0,
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
            cbf_flow_enabled=cbf_flow_enabled,
            agent_entry_interval=agent_entry_interval,
            order_max_duration=order_max_duration,
            energy_reserve=energy_reserve,
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
        if self.env.cbf_flow_enabled:
            terminated = bool(
                info.get("time_limit_reached", False)
                or self.env.cbf_flow_all_entered_depleted()
            )
            info["all_depleted"] = self.env.cbf_flow_all_entered_depleted()
            info["energy_transitions"] = list(
                getattr(self.env, "_last_energy_transitions", [])
            )
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
