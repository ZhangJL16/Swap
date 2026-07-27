import itertools
import math
import os
import random

import numpy as np
from gymnasium import spaces


boundary_length = 4.0
boundary_width = 4.0
boundary_height = 4.0
default_num_obstacles = 10
default_obstacle_radius_range = (0.16, 0.24)
default_initial_energy = 100.0
default_energy_depletion_fraction = 0.5
default_charging_station_count = None
default_charging_capacity = 2
default_charging_radius = 0.18
eps = 1e-6
legend_font_size_xy = 26
legend_font_size_3d = 26


def _laser_angles(num_lasers):
    return np.linspace(0.0, 2.0 * math.pi, num_lasers, endpoint=False, dtype=np.float32)


def update_lasers_to_boundary(agent_pos, l_sensor, num_lasers, length, width):
    origin = np.asarray(agent_pos, dtype=np.float32)
    distances = np.full(num_lasers, l_sensor, dtype=np.float32)

    for idx, angle in enumerate(_laser_angles(num_lasers)):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        ray_limits = []

        if abs(direction[0]) > eps:
            x_target = length if direction[0] > 0 else 0.0
            t_x = (x_target - origin[0]) / direction[0]
            if t_x >= 0:
                y_at_x = origin[1] + t_x * direction[1]
                if -eps <= y_at_x <= width + eps:
                    ray_limits.append(t_x)

        if abs(direction[1]) > eps:
            y_target = width if direction[1] > 0 else 0.0
            t_y = (y_target - origin[1]) / direction[1]
            if t_y >= 0:
                x_at_y = origin[0] + t_y * direction[0]
                if -eps <= x_at_y <= length + eps:
                    ray_limits.append(t_y)

        if ray_limits:
            distances[idx] = min(l_sensor, float(min(ray_limits)))

    return distances


def update_lasers_to_obstacle(agent_pos, obstacle_pos, radius, l_sensor, num_lasers):
    origin = np.asarray(agent_pos, dtype=np.float32)
    center = np.asarray(obstacle_pos, dtype=np.float32)
    distances = np.full(num_lasers, l_sensor, dtype=np.float32)

    oc = origin - center
    oc_dot = float(np.dot(oc, oc))
    radius_sq = float(radius * radius)

    for idx, angle in enumerate(_laser_angles(num_lasers)):
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
        b = 2.0 * float(np.dot(direction, oc))
        c = oc_dot - radius_sq
        discriminant = b * b - 4.0 * c
        if discriminant < 0.0:
            continue

        sqrt_disc = math.sqrt(discriminant)
        t0 = (-b - sqrt_disc) / 2.0
        t1 = (-b + sqrt_disc) / 2.0
        valid_ts = [t for t in (t0, t1) if t >= 0.0]
        if valid_ts:
            distances[idx] = min(l_sensor, float(min(valid_ts)))

    return distances


class UAVAgent:
    def __init__(
        self,
        number,
        pos,
        v_max,
        a_max,
        num_lasers,
        l_sensor,
        safe_radius,
        dim,
        initial_energy=default_initial_energy,
    ):
        self.number = number
        self.pos = np.asarray(pos, dtype=np.float32)
        self.spawn_pos = np.asarray(pos, dtype=np.float32)
        self.prev_pos = np.asarray(pos, dtype=np.float32)
        self.last_pos = np.asarray(pos, dtype=np.float32)
        self.goal = np.asarray(pos, dtype=np.float32)
        self.vel = np.zeros(dim, dtype=np.float32)
        self.v_max = v_max
        self.a_max = a_max
        self.num_lasers = num_lasers
        self.l_sensor = l_sensor
        self.lasers = np.full((num_lasers,), l_sensor, dtype=np.float32)
        self.safe_radius = safe_radius
        self.dim = dim
        self.reached = False
        self.collided = False
        self.prev_collided = False
        self.assigned_order_id = None
        self.carrying_order = False
        self.completed_orders = 0
        self.initial_energy = float(initial_energy)
        self.energy = float(initial_energy)
        self.active = True
        self.exit_reason = None

    def has_energy(self):
        return self.energy > eps

    def consume_energy(self, amount):
        if not self.has_energy():
            self.energy = 0.0
            return
        self.energy = round(max(0.0, self.energy - float(amount)), 1)

    def charge_energy(self, amount):
        self.energy = round(
            min(self.initial_energy, self.energy + float(amount)),
            1,
        )

    def update_velocity(self, action, time_step):
        if not self.has_energy():
            self.vel[:] = 0.0
            return
        action = np.asarray(action, dtype=np.float32)
        norm = np.linalg.norm(action)
        if norm > self.a_max:
            action = action / (norm + eps) * self.a_max
        self.vel += action * time_step
        speed = np.linalg.norm(self.vel)
        if speed > self.v_max:
            self.vel = self.vel / (speed + eps) * self.v_max

    def preview_position(self, time_step):
        self.last_pos = self.pos.copy()
        if not self.has_energy():
            self.vel[:] = 0.0
            self.prev_pos = self.pos.copy()
            return
        self.prev_pos = self.pos + self.vel * time_step


class GoalPoint:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float32)


class DeliveryOrder:
    PENDING = "pending"
    ACTIVE = "active"
    ASSIGNED = "assigned"
    PICKED = "picked"
    COMPLETED = "completed"

    def __init__(self, order_id, pickup_pos, dropoff_pos):
        self.order_id = int(order_id)
        self.pickup_pos = np.asarray(pickup_pos, dtype=np.float32)
        self.dropoff_pos = np.asarray(dropoff_pos, dtype=np.float32)
        self.status = self.PENDING
        self.assigned_agent = None


class Obstacle:
    def __init__(self, length, width, radius_range=default_obstacle_radius_range):
        self.pos = np.array(
            [
                np.random.uniform(0.45, length - 0.45),
                np.random.uniform(0.45, width - 0.45),
            ],
            dtype=np.float32,
        )
        self.radius = float(np.random.uniform(*radius_range))
        self.vel = np.zeros(2, dtype=np.float32)


class UAVEnv:
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
    ):
        if dim_actions not in (2, 3):
            raise ValueError("Dimension must be 2 or 3")
        if int(total_orders) <= 0:
            raise ValueError("total_orders must be positive")
        if int(max_active_orders) <= 0:
            raise ValueError("max_active_orders must be positive")

        self.dim_actions = dim_actions
        self.length = length
        self.width = width
        self.height = height if dim_actions == 3 else 0.0
        self.num_obstacle = num_obstacle
        self.num_agents = num_hunters
        self.episode_limit = episode_limit
        self.reset_retry_limit = reset_retry_limit
        self.sample_retry_limit = sample_retry_limit
        self.obstacle_crash_penalty = float(obstacle_crash_penalty)
        self.total_orders = int(total_orders)
        self.max_active_orders = min(int(max_active_orders), self.total_orders)
        self.pickup_reward = float(pickup_reward)
        self.delivery_reward = float(delivery_reward)
        self.depleted_penalty = -float(
            10.0
            * (
                self.total_orders * (self.pickup_reward + self.delivery_reward)
                + 5.0
                + float(episode_limit)
            )
        )
        self.initial_energy = float(initial_energy)
        self.energy_depletion_fraction = float(energy_depletion_fraction)
        if energy_decay_per_step is None:
            depletion_steps = max(
                1.0,
                float(episode_limit) * self.energy_depletion_fraction,
            )
            energy_decay_per_step = self.initial_energy / depletion_steps
        self.energy_decay_per_step = round(float(energy_decay_per_step), 1)
        if charging_station_count is None:
            charging_station_count = max(1, (int(self.num_agents) + 1) // 2)
        self.charging_station_count = int(max(1, charging_station_count))
        if charging_capacity is None:
            charging_capacity = 2
        self.charging_capacity = int(charging_capacity)
        self.charging_radius = float(charging_radius)
        self.charging_rate = round(
            float(
                4.0 * self.energy_decay_per_step
                if charging_rate is None
                else charging_rate
            ),
            1,
        )
        self.charging_station_positions = self._init_charging_station_positions(
            charging_station_pos
        )
        self.charging_station_pos = self.charging_station_positions[0].copy()
        self.charging_agent_ids = []
        self.charging_station_agent_ids = {
            station_idx: [] for station_idx in range(self.charging_station_count)
        }

        self.time_step = 0.4
        self.goal_tolerance = 0.12
        self.v_max = 0.16
        self.a_max = 0.05
        self.safe_radius = 0.05
        self.risk_warning_margin = 0.06
        self.guard_prediction_margin = 0.04
        self.velocity_reward_weight = 0.1
        self.obstacle_collision_penalty = 1.2
        self.agent_collision_penalty = 1.5
        self.repeat_collision_scale = 0.35
        self.l_sensor = 0.35
        self.num_lasers = 16
        self.msg_shape = self.dim_actions * 2 + 3

        self.max_cycles = episode_limit
        self.agents = []
        self.goals = []
        self.orders = []
        self.active_order_ids = []
        self.next_order_id_to_activate = 0
        self.completed_order_count = 0
        self.obstacles = []
        self.agent_paths = [[] for _ in range(self.num_agents)]
        self.current_step = 0
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.collision_count = 0.0
        self.obstacle_collision_count = 0.0
        self.agent_collision_count = 0.0
        self._render_fig = None
        self._render_has_shown = False

        obs_dim = 2 * self.dim_actions + self.num_lasers
        self.action_space = {
            f"agent_{i}": spaces.Box(
                low=-self.a_max,
                high=self.a_max,
                shape=(self.dim_actions,),
                dtype=np.float32,
            )
            for i in range(self.num_agents)
        }
        self.observation_space = {
            f"agent_{i}": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(obs_dim,),
                dtype=np.float32,
            )
            for i in range(self.num_agents)
        }

    def _space_scale(self):
        return np.array(
            [self.length, self.width, self.height][: self.dim_actions],
            dtype=np.float32,
        )

    def _horizontal_pos(self, pos):
        return np.asarray(pos[:2], dtype=np.float32)

    def _cylinder_distance(self, pos, obstacle):
        return np.linalg.norm(self._horizontal_pos(pos) - obstacle.pos)

    def _distance_to_goal(self, agent):
        return float(np.linalg.norm(agent.goal - agent.pos))

    @staticmethod
    def _nonlinear_risk_from_overlap(overlap, scale):
        overlap = float(max(0.0, overlap))
        if overlap <= 0.0:
            return 0.0
        normalized = overlap / (float(scale) + eps)
        return float(normalized * normalized)

    def _sample_position(self, lower_margin=0.2):
        low = np.full(self.dim_actions, lower_margin, dtype=np.float32)
        high = self._space_scale() - lower_margin
        return np.random.uniform(low, high).astype(np.float32)

    def _is_valid_spawn(self, pos, others, radius):
        for obstacle in self.obstacles:
            if self._cylinder_distance(pos, obstacle) <= radius + obstacle.radius + 0.05:
                return False
        for other in others:
            if np.linalg.norm(pos - other.pos) <= radius + other.safe_radius + 0.08:
                return False
        return True

    def _is_valid_obstacle(self, obstacle, others):
        boundary_margin = obstacle.radius + self.safe_radius + 0.05
        if obstacle.pos[0] - boundary_margin < 0 or obstacle.pos[0] + boundary_margin > self.length:
            return False
        if obstacle.pos[1] - boundary_margin < 0 or obstacle.pos[1] + boundary_margin > self.width:
            return False

        for other in others:
            min_dist = obstacle.radius + other.radius + self.safe_radius + 0.08
            if np.linalg.norm(obstacle.pos - other.pos) <= min_dist:
                return False
        return True

    def _is_valid_goal(self, goal, existing_goals, agents):
        for obstacle in self.obstacles:
            if self._cylinder_distance(goal, obstacle) <= self.goal_tolerance + obstacle.radius + 0.05:
                return False
        for existing in existing_goals:
            if np.linalg.norm(goal - existing.pos) <= 0.25:
                return False
        for agent in agents:
            if np.linalg.norm(goal - agent.pos) <= 0.35:
                return False
        return True

    def _is_valid_delivery_point(self, point, existing_points, agents):
        for obstacle in self.obstacles:
            if self._cylinder_distance(point, obstacle) <= self.goal_tolerance + obstacle.radius + 0.05:
                return False
        for existing in existing_points:
            if np.linalg.norm(point - existing) <= 0.25:
                return False
        for agent in agents:
            if np.linalg.norm(point - agent.pos) <= 0.35:
                return False
        return True

    def _initialize_obstacles(self):
        obstacles = []
        for obstacle_idx in range(self.num_obstacle):
            for _ in range(self.sample_retry_limit):
                obstacle = Obstacle(self.length, self.width)
                if self._is_valid_obstacle(obstacle, obstacles):
                    obstacles.append(obstacle)
                    break
            else:
                raise RuntimeError(
                    f'Failed to place obstacle {obstacle_idx} after {self.sample_retry_limit} attempts.'
                )
        self.obstacles = obstacles

    def _initialize_agents(self):
        self.agents = []
        for agent_idx in range(self.num_agents):
            for _ in range(self.sample_retry_limit):
                pos = self._sample_position()
                candidate = UAVAgent(
                    number=agent_idx,
                    pos=pos,
                    v_max=self.v_max,
                    a_max=self.a_max,
                    num_lasers=self.num_lasers,
                    l_sensor=self.l_sensor,
                    safe_radius=self.safe_radius,
                    dim=self.dim_actions,
                    initial_energy=self.initial_energy,
                )
                if self._is_valid_spawn(pos, self.agents, self.safe_radius):
                    self.agents.append(candidate)
                    break
            else:
                raise RuntimeError(
                    f'Failed to place UAV {agent_idx} after {self.sample_retry_limit} attempts.'
                )

    def _initialize_orders(self):
        self.orders = []
        self.goals = []
        self.active_order_ids = []
        self.next_order_id_to_activate = 0
        self.completed_order_count = 0

    def _active_delivery_points(self):
        points = []
        for order_id in self.active_order_ids:
            order = self.orders[order_id]
            if order.status == DeliveryOrder.COMPLETED:
                continue
            points.extend([order.pickup_pos, order.dropoff_pos])
        return points

    def _sample_delivery_order(self, order_id):
        existing_points = self._active_delivery_points()

        pickup = None
        for _ in range(self.sample_retry_limit):
            candidate = self._sample_position(lower_margin=0.3)
            if self._is_valid_delivery_point(candidate, existing_points, self.agents):
                pickup = candidate
                break
        if pickup is None:
            raise RuntimeError(
                f'Failed to place pickup point for order {order_id} after {self.sample_retry_limit} attempts.'
            )

        dropoff = None
        for _ in range(self.sample_retry_limit):
            candidate = self._sample_position(lower_margin=0.3)
            if self._is_valid_delivery_point(
                candidate, existing_points + [pickup], self.agents
            ):
                dropoff = candidate
                break
        if dropoff is None:
            raise RuntimeError(
                f'Failed to place dropoff point for order {order_id} after {self.sample_retry_limit} attempts.'
            )

        return DeliveryOrder(order_id, pickup, dropoff)

    def _active_order_count(self):
        active_statuses = {
            DeliveryOrder.ACTIVE,
            DeliveryOrder.ASSIGNED,
            DeliveryOrder.PICKED,
        }
        return sum(
            1
            for order_id in self.active_order_ids
            if self.orders[order_id].status in active_statuses
        )

    def _all_orders_completed(self):
        return self.completed_order_count >= self.total_orders

    def _sync_goals_from_orders(self):
        self.goals = []
        for order_id in self.active_order_ids:
            order = self.orders[order_id]
            if order.status in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED):
                self.goals.append(GoalPoint(order.pickup_pos))
            elif order.status == DeliveryOrder.PICKED:
                self.goals.append(GoalPoint(order.dropoff_pos))

    def _activate_orders(self):
        while (
            self._active_order_count() < self.max_active_orders
            and self.next_order_id_to_activate < self.total_orders
        ):
            order = self._sample_delivery_order(self.next_order_id_to_activate)
            order.status = DeliveryOrder.ACTIVE
            order.assigned_agent = None
            self.orders.append(order)
            self.active_order_ids.append(order.order_id)
            self.next_order_id_to_activate += 1
        self._sync_goals_from_orders()

    def _available_orders(self):
        return [
            self.orders[order_id]
            for order_id in self.active_order_ids
            if self.orders[order_id].status == DeliveryOrder.ACTIVE
        ]

    def _order_target(self, order):
        if order.status in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED):
            return order.pickup_pos.copy()
        if order.status == DeliveryOrder.PICKED:
            return order.dropoff_pos.copy()
        return None

    def _nearest_available_order(self, agent):
        available_orders = self._available_orders()
        if not available_orders:
            return None
        return min(
            available_orders,
            key=lambda order: float(np.linalg.norm(order.pickup_pos - agent.pos)),
        )

    def _set_agent_idle(self, agent):
        agent.assigned_order_id = None
        agent.carrying_order = False
        agent.reached = True
        agent.vel[:] = 0.0
        agent.goal = agent.pos.copy()

    def _agent_is_active(self, agent):
        return bool(getattr(agent, "active", True))

    def _deactivate_agent(self, agent, reason, release_order=True):
        if release_order and agent.assigned_order_id is not None:
            order = self.orders[agent.assigned_order_id]
            if order.status != DeliveryOrder.COMPLETED:
                order.status = DeliveryOrder.ACTIVE
                order.assigned_agent = None
        agent.assigned_order_id = None
        agent.carrying_order = False
        agent.active = False
        agent.exit_reason = str(reason)
        agent.reached = True
        agent.vel[:] = 0.0
        agent.prev_pos = agent.pos.copy()
        agent.goal = agent.pos.copy()

    def _deactivate_depleted_agents(self):
        penalties = np.zeros(self.num_agents, dtype=np.float32)
        for agent in self.agents:
            if getattr(agent, "active", True) and not agent.has_energy():
                penalties[agent.number] = self._reset_depleted_agent(agent)
        return penalties

    def _reset_depleted_agent(self, agent):
        if agent.assigned_order_id is not None:
            order = self.orders[agent.assigned_order_id]
            if order.status != DeliveryOrder.COMPLETED:
                order.status = DeliveryOrder.ACTIVE
                order.assigned_agent = None
        agent.assigned_order_id = None
        agent.carrying_order = False
        agent.energy = float(agent.initial_energy)
        agent.active = True
        agent.exit_reason = "depleted_reset"
        agent.reached = False
        agent.collided = False
        agent.prev_collided = False
        agent.pos = agent.spawn_pos.copy()
        agent.prev_pos = agent.spawn_pos.copy()
        agent.last_pos = agent.spawn_pos.copy()
        agent.vel[:] = 0.0
        nearest_order = self._nearest_available_order(agent)
        if nearest_order is not None:
            agent.goal = nearest_order.pickup_pos.copy()
        else:
            agent.goal = agent.pos.copy()
            agent.reached = True
        self.agent_paths[agent.number] = [agent.pos.copy()]
        return float(self.depleted_penalty)

    def _assign_order_to_agent(self, agent, order):
        if not self._agent_is_active(agent):
            return False
        order.status = DeliveryOrder.ASSIGNED
        order.assigned_agent = agent.number
        agent.assigned_order_id = order.order_id
        agent.carrying_order = False
        agent.reached = False
        agent.goal = order.pickup_pos.copy()
        return True

    def _mark_order_picked(self, agent, order, update_goal=True):
        order.status = DeliveryOrder.PICKED
        agent.carrying_order = True
        if update_goal:
            agent.goal = order.dropoff_pos.copy()
        return self.pickup_reward

    def _mark_order_completed(self, agent, order):
        order.status = DeliveryOrder.COMPLETED
        order.assigned_agent = None
        self._remove_active_order(order.order_id)
        self.completed_order_count += 1
        agent.completed_orders += 1
        agent.assigned_order_id = None
        agent.carrying_order = False
        agent.reached = True
        agent.vel[:] = 0.0
        agent.goal = agent.pos.copy()
        return self.delivery_reward

    def _assign_orders(self):
        self._activate_orders()
        self._sync_goals_from_orders()

    def _remove_active_order(self, order_id):
        self.active_order_ids = [
            active_id for active_id in self.active_order_ids if active_id != order_id
        ]

    def _advance_order_if_reached(self, agent, current_dist):
        del current_dist
        if not self._agent_is_active(agent):
            return 0.0
        if agent.assigned_order_id is not None:
            order = self.orders[agent.assigned_order_id]
        else:
            order = None
        if (
            order is not None
            and order.status == DeliveryOrder.ASSIGNED
            and np.linalg.norm(agent.pos - order.pickup_pos) <= self.goal_tolerance
        ):
            return self._mark_order_picked(agent, order, update_goal=True)

        if (
            order is not None
            and order.status == DeliveryOrder.PICKED
            and np.linalg.norm(agent.pos - order.dropoff_pos) <= self.goal_tolerance
        ):
            return self._mark_order_completed(agent, order)

        if agent.assigned_order_id is None and not agent.carrying_order:
            candidates = [
                order
                for order in self._available_orders()
                if np.linalg.norm(agent.pos - order.pickup_pos) <= self.goal_tolerance
            ]
            if candidates:
                order = min(
                    candidates,
                    key=lambda item: float(np.linalg.norm(agent.pos - item.pickup_pos)),
                )
                if self._assign_order_to_agent(agent, order):
                    return self._mark_order_picked(agent, order, update_goal=True)

        return 0.0

    def get_active_agent_mask(self):
        return np.asarray(
            [1.0 if self._agent_is_active(agent) else 0.0 for agent in self.agents],
            dtype=np.float32,
        )

    def get_agent_positions(self):
        positions = []
        for agent in self.agents:
            if self._agent_is_active(agent):
                positions.append(agent.pos.copy())
            else:
                positions.append(np.zeros(self.dim_actions, dtype=np.float32))
        return np.stack(positions, axis=0).astype(np.float32)

    def get_agent_energy_ratios(self):
        return np.asarray(
            [
                [
                    agent.energy / (agent.initial_energy + eps)
                    if self._agent_is_active(agent)
                    else 0.0
                ]
                for agent in self.agents
            ],
            dtype=np.float32,
        )

    def get_agent_completed_order_counts(self):
        return np.asarray(
            [[float(agent.completed_orders)] for agent in self.agents],
            dtype=np.float32,
        )

    def _init_charging_station_positions(self, charging_station_pos):
        if charging_station_pos is not None:
            positions = np.asarray(charging_station_pos, dtype=np.float32)
            if positions.ndim == 1:
                positions = positions.reshape(1, -1)
            positions = positions[:, : self.dim_actions]
            if len(positions) >= self.charging_station_count:
                return positions[: self.charging_station_count].copy()
            extra = self._default_charging_station_positions(
                self.charging_station_count - len(positions)
            )
            return np.concatenate([positions, extra], axis=0).astype(np.float32)
        return self._default_charging_station_positions(self.charging_station_count)

    def _default_charging_station_positions(self, count):
        count = int(max(1, count))
        center = np.array([self.length * 0.5, self.width * 0.5], dtype=np.float32)
        radius = min(self.length, self.width) * 0.22
        positions = []
        for idx in range(count):
            angle = 2.0 * np.pi * float(idx) / float(count)
            pos2 = center + radius * np.array(
                [np.cos(angle), np.sin(angle)],
                dtype=np.float32,
            )
            pos2[0] = np.clip(pos2[0], self.charging_radius, self.length - self.charging_radius)
            pos2[1] = np.clip(pos2[1], self.charging_radius, self.width - self.charging_radius)
            if self.dim_actions == 3:
                positions.append(np.array([pos2[0], pos2[1], self.height * 0.5], dtype=np.float32))
            else:
                positions.append(pos2.astype(np.float32))
        return np.stack(positions, axis=0).astype(np.float32)

    def _nearest_charging_station_index(self, pos):
        distances = np.linalg.norm(
            self.charging_station_positions - np.asarray(pos, dtype=np.float32)[: self.dim_actions],
            axis=1,
        )
        return int(np.argmin(distances))

    def _nearest_charging_station_pos(self, pos):
        return self.charging_station_positions[
            self._nearest_charging_station_index(pos)
        ].copy()

    def _distance_to_charging_station(self, agent):
        return float(
            np.min(np.linalg.norm(self.charging_station_positions - agent.pos, axis=1))
        )

    def _distance_to_nearest_charging_station_from_pos(self, pos):
        return float(
            np.min(
                np.linalg.norm(
                    self.charging_station_positions
                    - np.asarray(pos, dtype=np.float32)[: self.dim_actions],
                    axis=1,
                )
            )
        )

    def _consume_step_energy(self, powered_mask):
        for is_powered, agent in zip(powered_mask, self.agents):
            if is_powered:
                agent.consume_energy(self.energy_decay_per_step)

    def _charge_agents_at_station(self):
        selected = []
        station_assignments = {
            station_idx: [] for station_idx in range(self.charging_station_count)
        }
        for station_idx, station_pos in enumerate(self.charging_station_positions):
            candidates = [
                agent
                for agent in self.agents
                if self._agent_is_active(agent)
                and np.linalg.norm(agent.pos - station_pos) <= self.charging_radius
                and agent.energy < agent.initial_energy
            ]
            candidates.sort(
                key=lambda agent: (
                    float(np.linalg.norm(agent.pos - station_pos)),
                    agent.number,
                )
            )
            station_selected = candidates[: max(0, self.charging_capacity)]
            station_assignments[station_idx] = [agent.number for agent in station_selected]
            selected.extend(station_selected)
        self.charging_station_agent_ids = station_assignments
        self.charging_agent_ids = sorted({agent.number for agent in selected})
        for agent in selected:
            agent.charge_energy(self.charging_rate)

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(1, 100000)
        random.seed(seed)
        np.random.seed(seed)

        last_error = None
        for _ in range(self.reset_retry_limit):
            self.current_step = 0
            self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
            self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
            self.collision_count = 0.0
            self.obstacle_collision_count = 0.0
            self.agent_collision_count = 0.0
            self.agent_paths = [[] for _ in range(self.num_agents)]
            self.obstacles = []
            self.agents = []
            self.goals = []
            self.orders = []
            self.active_order_ids = []
            self.next_order_id_to_activate = 0
            self.completed_order_count = 0
            self.charging_agent_ids = []
            self.charging_station_agent_ids = {
                station_idx: [] for station_idx in range(self.charging_station_count)
            }

            try:
                self._initialize_obstacles()
                self._initialize_agents()
                self._initialize_orders()
                self._assign_orders()
                self.update_lasers()
                return self.get_obs()
            except RuntimeError as exc:
                last_error = exc
                continue

        raise RuntimeError(
            'UAVEnv reset failed to sample a valid scenario '
            f'after {self.reset_retry_limit} retries. Last error: {last_error}'
        )

    def _apply_boundary_constraints(self, agent):
        collided = False
        boundary_risk = 0.0
        boundary_reward_risk = 0.0
        boundaries = [self.length, self.width, self.height][: self.dim_actions]
        for dim, boundary in enumerate(boundaries):
            lower_clearance = float(agent.prev_pos[dim])
            upper_clearance = float(boundary - agent.prev_pos[dim])
            warning_clearance = agent.safe_radius + self.risk_warning_margin
            lower_overlap = max(0.0, warning_clearance - lower_clearance)
            upper_overlap = max(0.0, warning_clearance - upper_clearance)
            boundary_risk += self._nonlinear_risk_from_overlap(
                lower_overlap, warning_clearance
            )
            boundary_risk += self._nonlinear_risk_from_overlap(
                upper_overlap, warning_clearance
            )
            boundary_reward_risk += self._nonlinear_risk_from_overlap(
                max(0.0, agent.safe_radius - lower_clearance), agent.safe_radius
            )
            boundary_reward_risk += self._nonlinear_risk_from_overlap(
                max(0.0, agent.safe_radius - upper_clearance), agent.safe_radius
            )
            if agent.prev_pos[dim] - agent.safe_radius < 0:
                agent.prev_pos[dim] = agent.safe_radius
                agent.vel[dim] = 0.0
                collided = True
            elif agent.prev_pos[dim] + agent.safe_radius > boundary:
                agent.prev_pos[dim] = boundary - agent.safe_radius
                agent.vel[dim] = 0.0
                collided = True
        return collided, boundary_risk, boundary_reward_risk

    def _resolve_obstacle_collisions(self, agent):
        collided = False
        safe_penalty = 0.0
        reward_penalty = 0.0
        for obstacle in self.obstacles:
            delta_xy = agent.prev_pos[:2] - obstacle.pos
            dist_xy = np.linalg.norm(delta_xy)
            collision_min_dist = obstacle.radius + agent.safe_radius
            warning_min_dist = collision_min_dist + self.risk_warning_margin
            overlap = max(0.0, warning_min_dist - dist_xy)
            safe_penalty = max(
                safe_penalty,
                self._nonlinear_risk_from_overlap(overlap, warning_min_dist),
            )
            reward_penalty = max(
                reward_penalty,
                self._nonlinear_risk_from_overlap(
                    max(0.0, collision_min_dist - dist_xy), collision_min_dist
                ),
            )
            if dist_xy < collision_min_dist:
                collided = True
                if dist_xy < eps:
                    delta_xy = np.array([1.0, 0.0], dtype=np.float32)
                    dist_xy = 1.0
                direction_xy = delta_xy / dist_xy
                corrected_xy = obstacle.pos + direction_xy * collision_min_dist
                agent.prev_pos[:2] = corrected_xy
                agent.vel[:2] = 0.0
        return collided, safe_penalty, reward_penalty

    def _resolve_agent_collisions(self):
        collided = [False] * self.num_agents
        for i in range(self.num_agents):
            if not self._agent_is_active(self.agents[i]):
                continue
            for j in range(i + 1, self.num_agents):
                if not self._agent_is_active(self.agents[j]):
                    continue
                delta = self.agents[i].prev_pos - self.agents[j].prev_pos
                dist = np.linalg.norm(delta)
                collision_min_dist = (
                    self.agents[i].safe_radius + self.agents[j].safe_radius
                )
                warning_min_dist = collision_min_dist + self.risk_warning_margin
                overlap = max(0.0, warning_min_dist - dist)
                pair_risk = self._nonlinear_risk_from_overlap(
                    overlap, warning_min_dist
                )
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
                    i_powered = self._agent_is_active(self.agents[i])
                    j_powered = self._agent_is_active(self.agents[j])
                    if i_powered and j_powered:
                        self.agents[i].prev_pos += direction * (overlap / 2.0 + eps)
                        self.agents[j].prev_pos -= direction * (overlap / 2.0 + eps)
                    elif i_powered:
                        self.agents[i].prev_pos += direction * (overlap + eps)
                    elif j_powered:
                        self.agents[j].prev_pos -= direction * (overlap + eps)
                    self.agents[i].vel[:] = 0.0
                    self.agents[j].vel[:] = 0.0
        return collided

    def _predict_agent_kinematics(self, agent, action):
        accel = np.asarray(action, dtype=np.float32).copy()
        norm = np.linalg.norm(accel)
        if norm > agent.a_max:
            accel = accel / (norm + eps) * agent.a_max

        pred_vel = agent.vel + accel * self.time_step
        speed = np.linalg.norm(pred_vel)
        if speed > agent.v_max:
            pred_vel = pred_vel / (speed + eps) * agent.v_max

        pred_pos = agent.pos + agent.vel * self.time_step + 0.5 * accel * (
            self.time_step ** 2
        )
        return pred_pos.astype(np.float32), pred_vel.astype(np.float32)

    def _evaluate_joint_risk_batch_exact(self, actions_batch):
        actions_batch = np.asarray(actions_batch, dtype=np.float32)
        if actions_batch.ndim != 3:
            raise ValueError(
                "actions_batch must have shape (batch, n_agents, dim_actions)"
            )
        if actions_batch.shape[1] != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")
        if actions_batch.shape[2] != self.dim_actions:
            raise ValueError("Action dimension does not match UAV action dimension.")

        batch_size = actions_batch.shape[0]
        positions = np.asarray([agent.pos for agent in self.agents], dtype=np.float32)
        velocities = np.asarray([agent.vel for agent in self.agents], dtype=np.float32)
        safe_radii = np.asarray(
            [agent.safe_radius for agent in self.agents], dtype=np.float32
        )
        reached = np.asarray(
            [agent.reached or not self._agent_is_active(agent) for agent in self.agents],
            dtype=bool,
        )
        agent_a_max = np.asarray([agent.a_max for agent in self.agents], dtype=np.float32)
        agent_v_max = np.asarray([agent.v_max for agent in self.agents], dtype=np.float32)
        boundaries = np.asarray(
            [self.length, self.width, self.height][: self.dim_actions], dtype=np.float32
        )
        obstacle_pos = np.asarray(
            [obstacle.pos for obstacle in self.obstacles], dtype=np.float32
        )
        obstacle_radius = np.asarray(
            [obstacle.radius for obstacle in self.obstacles], dtype=np.float32
        )
        safe_values = np.zeros((batch_size, self.num_agents), dtype=np.float32)

        for batch_idx in range(batch_size):
            prev_pos = positions.copy()
            pred_vel = velocities.copy()
            batch_safe_value = np.zeros(self.num_agents, dtype=np.float32)

            for agent_idx in range(self.num_agents):
                if reached[agent_idx]:
                    pred_vel[agent_idx, :] = 0.0
                    prev_pos[agent_idx, :] = positions[agent_idx]
                    continue

                accel = actions_batch[batch_idx, agent_idx].copy()
                norm = np.linalg.norm(accel)
                if norm > agent_a_max[agent_idx]:
                    accel = accel / (norm + eps) * agent_a_max[agent_idx]

                pred_vel[agent_idx] = pred_vel[agent_idx] + accel * self.time_step
                speed = np.linalg.norm(pred_vel[agent_idx])
                if speed > agent_v_max[agent_idx]:
                    pred_vel[agent_idx] = (
                        pred_vel[agent_idx] / (speed + eps) * agent_v_max[agent_idx]
                    )

                prev_pos[agent_idx] = (
                    positions[agent_idx] + pred_vel[agent_idx] * self.time_step
                )

            for agent_idx in range(self.num_agents):
                if reached[agent_idx]:
                    continue

                boundary_risk = 0.0
                warning_clearance = safe_radii[agent_idx] + self.risk_warning_margin
                for dim_idx, boundary in enumerate(boundaries):
                    lower_clearance = float(prev_pos[agent_idx, dim_idx])
                    upper_clearance = float(boundary - prev_pos[agent_idx, dim_idx])
                    lower_overlap = max(0.0, warning_clearance - lower_clearance)
                    upper_overlap = max(0.0, warning_clearance - upper_clearance)
                    boundary_risk += self._nonlinear_risk_from_overlap(
                        lower_overlap, warning_clearance
                    )
                    boundary_risk += self._nonlinear_risk_from_overlap(
                        upper_overlap, warning_clearance
                    )
                    if prev_pos[agent_idx, dim_idx] - safe_radii[agent_idx] < 0:
                        prev_pos[agent_idx, dim_idx] = safe_radii[agent_idx]
                        pred_vel[agent_idx, dim_idx] = 0.0
                    elif (
                        prev_pos[agent_idx, dim_idx] + safe_radii[agent_idx] > boundary
                    ):
                        prev_pos[agent_idx, dim_idx] = boundary - safe_radii[agent_idx]
                        pred_vel[agent_idx, dim_idx] = 0.0
                batch_safe_value[agent_idx] += boundary_risk

                obstacle_risk = 0.0
                for obs_idx in range(len(self.obstacles)):
                    delta_xy = prev_pos[agent_idx, :2] - obstacle_pos[obs_idx]
                    dist_xy = np.linalg.norm(delta_xy)
                    collision_min_dist = obstacle_radius[obs_idx] + safe_radii[agent_idx]
                    warning_min_dist = collision_min_dist + self.risk_warning_margin
                    overlap = max(0.0, warning_min_dist - dist_xy)
                    obstacle_risk = max(
                        obstacle_risk,
                        self._nonlinear_risk_from_overlap(overlap, warning_min_dist),
                    )
                    if dist_xy < collision_min_dist:
                        if dist_xy < eps:
                            delta_xy = np.array([1.0, 0.0], dtype=np.float32)
                            dist_xy = 1.0
                        direction_xy = delta_xy / dist_xy
                        corrected_xy = (
                            obstacle_pos[obs_idx] + direction_xy * collision_min_dist
                        )
                        prev_pos[agent_idx, :2] = corrected_xy
                        pred_vel[agent_idx, :2] = 0.0
                batch_safe_value[agent_idx] += obstacle_risk

            for i in range(self.num_agents):
                if reached[i]:
                    continue
                for j in range(i + 1, self.num_agents):
                    if reached[j]:
                        continue
                    delta = prev_pos[i] - prev_pos[j]
                    dist = np.linalg.norm(delta)
                    collision_min_dist = safe_radii[i] + safe_radii[j]
                    warning_min_dist = collision_min_dist + self.risk_warning_margin
                    overlap = max(0.0, warning_min_dist - dist)
                    pair_risk = self._nonlinear_risk_from_overlap(
                        overlap, warning_min_dist
                    )
                    batch_safe_value[i] += pair_risk
                    batch_safe_value[j] += pair_risk
                    if dist < collision_min_dist:
                        if dist < eps:
                            delta = np.zeros(self.dim_actions, dtype=np.float32)
                            delta[0] = 1.0
                            dist = 1.0
                        direction = delta / dist
                        overlap = collision_min_dist - dist
                        prev_pos[i] += direction * (overlap / 2.0 + eps)
                        prev_pos[j] -= direction * (overlap / 2.0 + eps)
                        pred_vel[i, :] = 0.0
                        pred_vel[j, :] = 0.0

            safe_values[batch_idx] = batch_safe_value

        return safe_values

    def predict_joint_collision_flags(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        predicted_pos = []
        for agent, action in zip(self.agents, actions):
            if agent.reached or not self._agent_is_active(agent):
                predicted_pos.append(agent.pos.copy())
                continue
            pred_pos, _ = self._predict_agent_kinematics(agent, action)
            predicted_pos.append(pred_pos)

        collision_flags = np.zeros(self.num_agents, dtype=bool)

        boundaries = [self.length, self.width, self.height][: self.dim_actions]
        for idx, (agent, pos) in enumerate(zip(self.agents, predicted_pos)):
            if agent.reached or not self._agent_is_active(agent):
                continue
            guard_clearance = agent.safe_radius + self.guard_prediction_margin
            for dim, boundary in enumerate(boundaries):
                if (
                    pos[dim] - guard_clearance < 0.0
                    or pos[dim] + guard_clearance > boundary
                ):
                    collision_flags[idx] = True
                    break

        for idx, (agent, pos) in enumerate(zip(self.agents, predicted_pos)):
            if agent.reached or not self._agent_is_active(agent):
                continue
            for obstacle in self.obstacles:
                delta_xy = pos[:2] - obstacle.pos
                dist_xy = np.linalg.norm(delta_xy)
                collision_min_dist = (
                    obstacle.radius + agent.safe_radius + self.guard_prediction_margin
                )
                if dist_xy < collision_min_dist:
                    collision_flags[idx] = True
                    break

        for i in range(self.num_agents):
            if self.agents[i].reached or not self._agent_is_active(self.agents[i]):
                continue
            for j in range(i + 1, self.num_agents):
                if self.agents[j].reached or not self._agent_is_active(self.agents[j]):
                    continue
                dist = np.linalg.norm(predicted_pos[i] - predicted_pos[j])
                collision_min_dist = (
                    self.agents[i].safe_radius
                    + self.agents[j].safe_radius
                    + self.guard_prediction_margin
                )
                if dist < collision_min_dist:
                    collision_flags[i] = True
                    collision_flags[j] = True

        return collision_flags.astype(np.float32)

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        self.current_step += 1
        self._assign_orders()
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.reward_safe_value = np.zeros(self.num_agents, dtype=np.float32)
        powered_mask = np.asarray(
            [self._agent_is_active(agent) for agent in self.agents], dtype=bool
        )

        prev_dists = np.array([self._distance_to_goal(agent) for agent in self.agents], dtype=np.float32)

        for idx, (agent, action) in enumerate(zip(self.agents, actions)):
            if (
                not powered_mask[idx]
                or agent.reached
                or agent.assigned_order_id is None
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
            self.reward_safe_value[idx] += (
                boundary_reward_penalty + obstacle_reward_penalty
            )

        agent_collisions = self._resolve_agent_collisions()

        for agent in self.agents:
            agent.pos = agent.prev_pos.copy()

        self._consume_step_energy(powered_mask)
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
            rewards[idx] += 2.5 * progress
            rewards[idx] += self.velocity_reward_weight * max(0.0, velocity_toward_goal)
            rewards[idx] -= 0.01
            rewards[idx] -= obstacle_penalty * float(obstacle_collisions[idx])
            rewards[idx] -= agent_penalty * float(agent_collisions[idx])

            order_reward = self._advance_order_if_reached(agent, current_dist)
            if order_reward > 0.0:
                agent.vel[:] = 0.0
                rewards[idx] += order_reward

            agent.prev_collided = agent.collided
            agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
            if self._agent_is_active(agent):
                self.agent_paths[idx].append(agent.pos.copy())

        self._assign_orders()
        delivery_done = self._all_orders_completed()
        dones = [not self._agent_is_active(agent) for agent in self.agents]

        obstacle_collision_total = float(
            np.sum(np.asarray(obstacle_collisions, dtype=bool))
        )
        agent_collision_total = float(
            np.sum(np.asarray(agent_collisions, dtype=bool))
        )
        self.obstacle_collision_count += obstacle_collision_total
        self.agent_collision_count += agent_collision_total
        self.collision_count += obstacle_collision_total + agent_collision_total

        if delivery_done:
            rewards += 5.0 * self.get_active_agent_mask()

        return self.get_obs(), rewards, dones, self.safe_value.copy()

    def _message_from_sender(self, receiver, sender):
        if not self._agent_is_active(receiver) or not self._agent_is_active(sender):
            return np.zeros(self.msg_shape, dtype=np.float32)
        scale = self._space_scale() + eps
        goal_delta = (sender.goal - sender.pos) / scale
        velocity = sender.vel / (sender.v_max + eps)
        sender_energy = np.array(
            [sender.energy / (sender.initial_energy + eps)],
            dtype=np.float32,
        )
        sender_risk = np.array(
            [
                min(
                    1.0,
                    float(self.safe_value[sender.number]) / (sender.safe_radius + eps),
                )
            ],
            dtype=np.float32,
        )
        sender_reached = np.array([float(sender.reached)], dtype=np.float32)
        return np.concatenate(
            [
                goal_delta.astype(np.float32),
                velocity.astype(np.float32),
                sender_energy,
                sender_risk,
                sender_reached,
            ]
        )

    def get_msg(self):
        messages = []
        for receiver in self.agents:
            receiver_msgs = []
            for sender in self.agents:
                if sender is receiver:
                    continue
                receiver_msgs.append(self._message_from_sender(receiver, sender))
            if receiver_msgs:
                messages.append(np.stack(receiver_msgs, axis=0).astype(np.float32))
            else:
                messages.append(
                    np.zeros((0, self.msg_shape), dtype=np.float32)
                )
        return messages

    def get_obs(self):
        observations = []
        scale = self._space_scale() + eps
        for agent in self.agents:
            if not self._agent_is_active(agent):
                observations.append(
                    np.zeros(self.observation_space[f"agent_{agent.number}"].shape, dtype=np.float32)
                )
                continue
            own = np.concatenate([agent.pos / scale, agent.vel / (agent.v_max + eps)])
            obs = np.concatenate(
                [
                    own,
                    np.asarray(agent.lasers, dtype=np.float32),
                ]
            )
            observations.append(obs.astype(np.float32))
        return np.stack(observations, axis=0)

    def get_state(self):
        parts = []
        scale = self._space_scale() + eps
        for agent in self.agents:
            if not self._agent_is_active(agent):
                parts.append(np.zeros(self.dim_actions, dtype=np.float32))
                parts.append(np.zeros(self.dim_actions, dtype=np.float32))
                continue
            parts.append(agent.pos / scale)
            parts.append(agent.vel / (agent.v_max + eps))
        for obstacle in self.obstacles:
            parts.append(obstacle.pos / scale[:2])
            parts.append(np.array([obstacle.radius / max(self.length, self.width)], dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def update_lasers(self):
        for agent in self.agents:
            if not self._agent_is_active(agent):
                agent.lasers = np.full(agent.num_lasers, agent.l_sensor, dtype=np.float32)
                continue
            current_lasers = np.full(agent.num_lasers, agent.l_sensor, dtype=np.float32)
            for obstacle in self.obstacles:
                radius = obstacle.radius + agent.safe_radius
                obstacle_lasers = update_lasers_to_obstacle(
                    agent.pos[:2], obstacle.pos, radius, agent.l_sensor, agent.num_lasers
                )
                current_lasers = np.minimum(current_lasers, obstacle_lasers)
            boundary_lasers = update_lasers_to_boundary(
                agent.pos[:2], agent.l_sensor, agent.num_lasers, self.length, self.width
            )
            agent.lasers = np.minimum(current_lasers, boundary_lasers)

    def summary(self):
        remaining = [
            self._distance_to_goal(agent)
            for agent in self.agents
            if agent.assigned_order_id is not None
        ]
        active_orders = self._active_order_count()
        available_orders = sum(
            1 for order in self.orders if order.status == DeliveryOrder.ACTIVE
        )
        picked_orders = sum(
            1 for order in self.orders if order.status == DeliveryOrder.PICKED
        )
        active_agents = float(np.sum([self._agent_is_active(agent) for agent in self.agents]))
        healthy_agents = float(
            np.sum([self._agent_is_active(agent) and not agent.collided for agent in self.agents])
        )
        powered_agents = float(np.sum([agent.has_energy() for agent in self.agents]))
        mean_energy = (
            float(np.mean([agent.energy for agent in self.agents]))
            if self.agents
            else 0.0
        )
        mean_goal_distance = float(np.mean(remaining)) if remaining else 0.0
        completed_orders = float(self.completed_order_count)
        return {
            "step": float(self.current_step),
            "agent_health": healthy_agents,
            "enemy_health": mean_goal_distance,
            "agent_alive": completed_orders,
            "collision_count": float(self.collision_count),
            "obstacle_collision_count": float(self.obstacle_collision_count),
            "agent_collision_count": float(self.agent_collision_count),
            "orders_completed": completed_orders,
            "total_orders": float(self.total_orders),
            "active_orders": float(active_orders),
            "available_orders": float(available_orders),
            "picked_orders": float(picked_orders),
            "idle_agents": float(
                0.0
            ),
            "healthy_agents": healthy_agents,
            "active_agents": active_agents,
            "powered_agents": powered_agents,
            "depleted_agents": float(self.num_agents - powered_agents),
            "mean_energy": mean_energy,
            "energy_decay_per_step": float(self.energy_decay_per_step),
            "charging_station_count": float(self.charging_station_count),
            "charging_capacity": float(self.charging_capacity),
            "total_charging_capacity": float(
                self.charging_station_count * self.charging_capacity
            ),
            "charging_agents": float(len(self.charging_agent_ids)),
            "mean_goal_distance": mean_goal_distance,
            "episode_reward": float(0.0),
            "win_tag": bool(self._all_orders_completed()),
        }

    def _capture_runtime_state(self):
        return {
            "current_step": int(self.current_step),
            "safe_value": self.safe_value.copy(),
            "reward_safe_value": self.reward_safe_value.copy(),
            "collision_count": float(self.collision_count),
            "obstacle_collision_count": float(self.obstacle_collision_count),
            "agent_collision_count": float(self.agent_collision_count),
            "agent_paths": [[pos.copy() for pos in path] for path in self.agent_paths],
            "active_order_ids": list(self.active_order_ids),
            "next_order_id_to_activate": int(self.next_order_id_to_activate),
            "completed_order_count": int(self.completed_order_count),
            "charging_agent_ids": list(self.charging_agent_ids),
            "charging_station_agent_ids": {
                int(station_idx): list(agent_ids)
                for station_idx, agent_ids in self.charging_station_agent_ids.items()
            },
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "orders": [
                {
                    "order_id": int(order.order_id),
                    "pickup_pos": order.pickup_pos.copy(),
                    "dropoff_pos": order.dropoff_pos.copy(),
                    "status": order.status,
                    "assigned_agent": order.assigned_agent,
                }
                for order in self.orders
            ],
            "agents": [
                {
                    "pos": agent.pos.copy(),
                    "prev_pos": agent.prev_pos.copy(),
                    "last_pos": agent.last_pos.copy(),
                    "goal": agent.goal.copy(),
                    "vel": agent.vel.copy(),
                    "lasers": agent.lasers.copy(),
                    "reached": bool(agent.reached),
                    "active": bool(getattr(agent, "active", True)),
                    "exit_reason": getattr(agent, "exit_reason", None),
                    "collided": bool(agent.collided),
                    "prev_collided": bool(agent.prev_collided),
                    "assigned_order_id": agent.assigned_order_id,
                    "carrying_order": bool(agent.carrying_order),
                    "completed_orders": int(agent.completed_orders),
                    "energy": float(agent.energy),
                    "initial_energy": float(agent.initial_energy),
                }
                for agent in self.agents
            ],
        }

    def _restore_runtime_state(self, snapshot):
        self.current_step = int(snapshot["current_step"])
        self.safe_value = snapshot["safe_value"].copy()
        self.reward_safe_value = snapshot["reward_safe_value"].copy()
        self.collision_count = float(snapshot["collision_count"])
        self.obstacle_collision_count = float(
            snapshot.get("obstacle_collision_count", 0.0)
        )
        self.agent_collision_count = float(
            snapshot.get("agent_collision_count", 0.0)
        )
        self.agent_paths = [[pos.copy() for pos in path] for path in snapshot["agent_paths"]]
        self.active_order_ids = list(snapshot.get("active_order_ids", []))
        self.next_order_id_to_activate = int(
            snapshot.get("next_order_id_to_activate", 0)
        )
        self.completed_order_count = int(snapshot.get("completed_order_count", 0))
        self.charging_agent_ids = list(snapshot.get("charging_agent_ids", []))
        self.charging_station_agent_ids = {
            int(station_idx): list(agent_ids)
            for station_idx, agent_ids in snapshot.get(
                "charging_station_agent_ids",
                {idx: [] for idx in range(self.charging_station_count)},
            ).items()
        }
        if "python_random_state" in snapshot:
            random.setstate(snapshot["python_random_state"])
        if "numpy_random_state" in snapshot:
            np.random.set_state(snapshot["numpy_random_state"])
        self.orders = []
        for state in snapshot.get("orders", []):
            order = DeliveryOrder(
                state["order_id"],
                state["pickup_pos"].copy(),
                state["dropoff_pos"].copy(),
            )
            order.status = state["status"]
            order.assigned_agent = state["assigned_agent"]
            self.orders.append(order)
        self._sync_goals_from_orders()
        for agent, state in zip(self.agents, snapshot["agents"]):
            agent.pos = state["pos"].copy()
            agent.prev_pos = state["prev_pos"].copy()
            agent.last_pos = state["last_pos"].copy()
            agent.goal = state["goal"].copy()
            agent.vel = state["vel"].copy()
            agent.lasers = state["lasers"].copy()
            agent.reached = bool(state["reached"])
            agent.active = bool(state.get("active", True))
            agent.exit_reason = state.get("exit_reason", None)
            agent.collided = bool(state["collided"])
            agent.prev_collided = bool(state["prev_collided"])
            agent.assigned_order_id = state.get("assigned_order_id")
            agent.carrying_order = bool(state.get("carrying_order", False))
            agent.completed_orders = int(state.get("completed_orders", 0))
            agent.energy = float(state.get("energy", self.initial_energy))
            agent.initial_energy = float(
                state.get("initial_energy", self.initial_energy)
            )

    def estimate_joint_risk(self, actions):
        snapshot = self._capture_runtime_state()
        try:
            _, _, _, safe_value = self.step(actions)
            return np.asarray(safe_value, dtype=np.float32).copy()
        finally:
            self._restore_runtime_state(snapshot)

    def estimate_joint_risk_batch(self, actions_batch):
        return self._evaluate_joint_risk_batch_exact(actions_batch)

    def _agent_color(self, idx):
        cmap = __import__("matplotlib").cm.get_cmap("tab10", max(self.num_agents, 1))
        return cmap(idx % max(self.num_agents, 1))

    def _visible_orders(self):
        return [
            order
            for order in self.orders
            if order.status
            in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED, DeliveryOrder.PICKED)
        ]

    def _order_color(self, order):
        if order.assigned_agent is not None:
            return self._agent_color(order.assigned_agent)
        return "#2ca02c"

    def _math_label_font(self):
        from matplotlib import font_manager

        cached = getattr(self, "_axis_math_font", None)
        if cached is not None:
            return cached

        candidates = [
            os.path.join(os.getcwd(), "fonts", "latinmodern-math.ttf"),
            os.path.join(os.getcwd(), "fonts", "latinmodern-math.otf"),
            "/mnt/c/Windows/Fonts/latinmodern-math.ttf",
            "/mnt/c/Windows/Fonts/latinmodern-math.otf",
            "/usr/share/fonts/truetype/lmodern/latinmodern-math.ttf",
            "/usr/share/fonts/opentype/lmodern/latinmodern-math.otf",
            "/usr/share/texlive/texmf-dist/fonts/opentype/public/lm-math/latinmodern-math.otf",
            "/usr/share/texmf/fonts/opentype/public/lm-math/latinmodern-math.otf",
        ]
        font_prop = None
        for font_path in candidates:
            if os.path.exists(font_path):
                font_manager.fontManager.addfont(font_path)
                font_prop = font_manager.FontProperties(fname=font_path)
                break
        if font_prop is None:
            try:
                font_path = font_manager.findfont(
                    "Latin Modern Math", fallback_to_default=False
                )
                font_prop = font_manager.FontProperties(fname=font_path)
            except ValueError:
                font_prop = font_manager.FontProperties(family="Times New Roman")
        self._axis_math_font = font_prop
        return font_prop

    def _set_axis_labels(self, ax, is_3d=False):
        label_font = self._math_label_font()
        ax.set_xlabel(r"$x$", fontproperties=label_font)
        ax.set_ylabel(r"$y$", fontproperties=label_font)
        if not is_3d:
            ax.yaxis.label.set_rotation(0)
        if is_3d:
            ax.set_zlabel(r"$z$", fontproperties=label_font)

    def _style_axes(self, ax, is_3d=False):
        font = "Times New Roman"
        label_font = self._math_label_font()
        label_size = 24
        tick_size = 26
        ax.set_facecolor("white")
        ax.grid(True, color="#d9d9d9", linewidth=0.6, alpha=0.8)
        tick_pad = -2 if is_3d else 14
        ax.tick_params(axis="x", labelsize=tick_size, pad=tick_pad)
        ax.tick_params(axis="y", labelsize=tick_size, pad=tick_pad)
        ax.xaxis.label.set_fontproperties(label_font)
        ax.yaxis.label.set_fontproperties(label_font)
        ax.xaxis.label.set_fontsize(label_size)
        ax.yaxis.label.set_fontsize(label_size)
        ax.xaxis.labelpad = 7
        ax.yaxis.labelpad = 7
        if is_3d:
            ax.tick_params(axis="z", labelsize=tick_size, pad=1)
            ax.zaxis.label.set_fontproperties(label_font)
            ax.zaxis.label.set_fontsize(label_size)
            ax.zaxis.labelpad = 2
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontname(font)
            label.set_fontsize(tick_size)
        if is_3d:
            for label in ax.get_zticklabels():
                label.set_fontname(font)
                label.set_fontsize(tick_size)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                try:
                    axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
                    axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
                    axis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.8)
                    axis._axinfo["grid"]["linewidth"] = 0.6
                except AttributeError:
                    pass
            try:
                ax.set_proj_type("ortho")
            except AttributeError:
                pass

    def _add_render_legend(self, ax, is_3d=False):
        font_size = legend_font_size_3d if is_3d else legend_font_size_xy

        legend_anchor = (0.98, 0.98)
        legend = ax.legend(
            loc="upper right",
            bbox_to_anchor=legend_anchor,
            prop={"family": "Times New Roman", "size": font_size},
            markerscale=1.35,
            frameon=True,
            framealpha=0.88,
            borderpad=0.18,
            labelspacing=0.25,
            handlelength=1.2,
            handletextpad=0.35,
            borderaxespad=0.25,
            scatterpoints=1,
        )
        if legend is not None:
            frame = legend.get_frame()
            frame.set_linewidth(0.6)
            frame.set_edgecolor("#666666")

    def _set_vertical_z_view(self, ax):
        try:
            ax.view_init(elev=25, azim=45, roll=0)
        except TypeError:
            ax.view_init(elev=25, azim=45)

    def _set_integer_ticks(self, ax, is_3d=False):
        ax.set_xticks(np.arange(0, int(round(self.length)) + 1, 1))
        ax.set_yticks(np.arange(0, int(round(self.width)) + 1, 1))
        if is_3d:
            ax.set_zticks(np.arange(0, int(round(self.height)) + 1, 1))

    def _agent_trajectory(self, idx, agent):
        path = [pos.copy() for pos in self.agent_paths[idx]]
        if not path or np.linalg.norm(path[-1] - agent.pos) > eps:
            path.append(agent.pos.copy())
        return np.asarray(path, dtype=np.float32)

    def _render_2d(self, ax):
        from matplotlib.patches import Circle

        pickup_labeled = False
        dropoff_labeled = False
        for order in self._visible_orders():
            color = self._order_color(order)
            if order.status in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED):
                ax.scatter(
                    order.pickup_pos[0],
                    order.pickup_pos[1],
                    c=[color],
                    marker="^",
                    s=115,
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.9,
                    label="pickup" if not pickup_labeled else None,
                )
                pickup_labeled = True
            ax.scatter(
                order.dropoff_pos[0],
                order.dropoff_pos[1],
                c=[color],
                marker="s",
                s=105,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                label="dropoff" if not dropoff_labeled else None,
            )
            dropoff_labeled = True

        for idx, agent in enumerate(self.agents):
            if not self._agent_is_active(agent):
                continue
            color = self._agent_color(idx)
            trajectory = self._agent_trajectory(idx, agent)
            if len(trajectory) > 1:
                ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, alpha=0.9, linewidth=1.8)
            ax.scatter(
                agent.pos[0],
                agent.pos[1],
                c=[color],
                s=90,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="hunter" if idx == 0 else None,
            )

        for obstacle in self.obstacles:
            ax.add_patch(Circle(
                obstacle.pos,
                obstacle.radius,
                color="gray",
                alpha=0.5,
            ))

        for station_idx, station_pos in enumerate(self.charging_station_positions):
            ax.scatter(
                station_pos[0],
                station_pos[1],
                c=["#1f77b4"],
                marker="P",
                s=145,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="charging station" if station_idx == 0 else None,
            )

        ax.set_xlim(-0.1, self.length + 0.1)
        ax.set_ylim(-0.1, self.width + 0.1)
        ax.set_aspect("equal", adjustable="box")
        self._set_integer_ticks(ax, is_3d=False)
        self._set_axis_labels(ax, is_3d=False)
        self._style_axes(ax, is_3d=False)
        ax.yaxis.set_label_coords(-0.14, 0.5)

    def _render_3d(self, ax):
        theta = np.linspace(0.0, 2.0 * math.pi, 24)
        z_values = np.linspace(0.0, self.height, 8)

        pickup_labeled = False
        dropoff_labeled = False
        for order in self._visible_orders():
            color = self._order_color(order)
            pickup_z = order.pickup_pos[2] if self.dim_actions == 3 else 0.0
            dropoff_z = order.dropoff_pos[2] if self.dim_actions == 3 else 0.0
            if order.status in (DeliveryOrder.ACTIVE, DeliveryOrder.ASSIGNED):
                ax.scatter(
                    order.pickup_pos[0],
                    order.pickup_pos[1],
                    pickup_z,
                    c=[color],
                    marker="^",
                    s=115,
                    edgecolors="black",
                    linewidths=0.8,
                    alpha=0.9,
                    label="pickup" if not pickup_labeled else None,
                )
                pickup_labeled = True
            ax.scatter(
                order.dropoff_pos[0],
                order.dropoff_pos[1],
                dropoff_z,
                c=[color],
                marker="s",
                s=105,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                label="dropoff" if not dropoff_labeled else None,
            )
            dropoff_labeled = True

        for idx, agent in enumerate(self.agents):
            if not self._agent_is_active(agent):
                continue
            color = self._agent_color(idx)
            trajectory = self._agent_trajectory(idx, agent)
            if len(trajectory) > 1:
                ax.plot(
                    trajectory[:, 0],
                    trajectory[:, 1],
                    trajectory[:, 2],
                    color=color,
                    alpha=0.9,
                    linewidth=1.8,
                )

            ax.scatter(
                agent.pos[0],
                agent.pos[1],
                agent.pos[2],
                c=[color],
                s=70,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="hunter" if idx == 0 else None,
            )

            speed = np.linalg.norm(agent.vel)
            if speed > eps:
                ax.quiver(
                    agent.pos[0],
                    agent.pos[1],
                    agent.pos[2],
                    agent.vel[0],
                    agent.vel[1],
                    agent.vel[2],
                    length=0.4,
                    normalize=True,
                    color=color,
                )

        theta_grid, z_grid = np.meshgrid(theta, z_values)
        for obstacle in self.obstacles:
            x_grid = obstacle.pos[0] + obstacle.radius * np.cos(theta_grid)
            y_grid = obstacle.pos[1] + obstacle.radius * np.sin(theta_grid)
            ax.plot_surface(
                x_grid,
                y_grid,
                z_grid,
                color="#d8d8d8",
                alpha=0.22,
                linewidth=0,
                shade=True,
            )

        for station_idx, station_pos in enumerate(self.charging_station_positions):
            station_z = station_pos[2] if self.dim_actions == 3 else 0.0
            ax.scatter(
                station_pos[0],
                station_pos[1],
                station_z,
                c=["#1f77b4"],
                marker="P",
                s=130,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.95,
                label="charging station" if station_idx == 0 else None,
            )

        ax.set_xlim(0.0, self.length)
        ax.set_ylim(0.0, self.width)
        ax.set_zlim(0.0, self.height)
        self._set_integer_ticks(ax, is_3d=True)
        self._set_axis_labels(ax, is_3d=True)
        self._set_vertical_z_view(ax)
        self._style_axes(ax, is_3d=True)
        self._add_render_legend(ax, is_3d=True)
        try:
            ax.set_box_aspect((self.length, self.width, self.height))
        except AttributeError:
            pass

    def render(self, show=False, view=None):
        import matplotlib.backends.backend_agg as agg
        import matplotlib.pyplot as plt

        plt.rcParams["font.family"] = "Times New Roman"
        render_xy = view == "xy" or self.dim_actions != 3

        if show:
            if self._render_fig is None:
                self._render_fig = plt.figure("UAVEnv")
                self._render_has_shown = False
            fig = self._render_fig
        else:
            fig = plt.figure("UAVEnv-rgb")
        fig.set_size_inches((5.2, 5.2), forward=True)
        fig.clf()
        fig.patch.set_facecolor("white")

        if not render_xy and self.dim_actions == 3:
            ax = fig.add_subplot(111, projection="3d")
            self._render_3d(ax)
        else:
            ax = fig.add_subplot(111)
            self._render_2d(ax)

        if render_xy:
            fig.subplots_adjust(left=0.12, right=0.985, bottom=0.17, top=0.985)
        else:
            fig.subplots_adjust(left=0.02, right=1.06, bottom=0.07, top=1.02)

        if show:
            plt.ion()
            if not self._render_has_shown:
                plt.show()
                self._render_has_shown = True
            fig.canvas.draw_idle()
            try:
                fig.canvas.flush_events()
            except NotImplementedError:
                pass
            return None

        canvas = agg.FigureCanvasAgg(fig)
        canvas.draw()
        return np.asarray(canvas.buffer_rgba())

    def close(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        self._render_fig = None
        self._render_has_shown = False
        plt.close("all")


class UAVEnvDiscreteWrapper:
    def __init__(
        self,
        dim_actions=3,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=8,
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
    ):
        self.env = UAVEnv(
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

    def _build_discrete_actions(self):
        action_values = [-1.0, 0.0, 1.0]
        discrete_actions = []
        for action in itertools.product(action_values, repeat=self.dim_actions):
            vec = np.array(action, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > eps:
                vec = vec / norm
            discrete_actions.append(vec * self.env.a_max)
        return discrete_actions

    def get_noop_action(self):
        return np.zeros(self.action_dim, dtype=np.float32)

    def _continuous_agent_action(self, action, is_active=True):
        if not is_active:
            return np.zeros(self.action_dim, dtype=np.float32)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < self.action_dim:
            action = np.pad(action, (0, self.action_dim - action.size))
        action = np.clip(action[: self.action_dim], -1.0, 1.0)
        return action.astype(np.float32) * float(self.env.a_max)

    def reset(self, seed=None):
        self._episode_steps = 0
        self._last_obs = self.env.reset(seed=seed)
        self._last_reward = 0.0
        self._last_info = {
            "battle_won": False,
            "warning_signal": np.zeros((self.n_agents, 1), dtype=np.float32),
        }
        return self.get_obs()

    def step(self, actions):
        active_mask = self.get_active_agent_mask()
        agent_actions = [
            self._continuous_agent_action(action, active_mask[idx] > 0.0)
            for idx, action in enumerate(actions)
        ]
        self._last_obs, rewards, dones, safe_value = self.env.step(agent_actions)
        self._episode_steps += 1
        active_count = float(np.sum(active_mask))
        reward = (
            float(np.sum(rewards * active_mask) / active_count)
            if active_count > 0.0
            else 0.0
        )
        battle_won = bool(self.env._all_orders_completed())
        post_step_active_mask = self.get_active_agent_mask()
        all_depleted = bool(np.sum(post_step_active_mask) <= 0.0)
        time_limit_reached = self._episode_steps >= self.episode_limit
        terminated = battle_won or all_depleted or time_limit_reached
        self._last_reward = reward
        self._last_info = {
            "battle_won": battle_won,
            "all_depleted": all_depleted,
            "time_limit_reached": time_limit_reached,
            "warning_signal": np.asarray(safe_value, dtype=np.float32).reshape(self.n_agents, 1),
            "per_agent_reward": np.asarray(rewards, dtype=np.float32).copy(),
            "agent_active_mask": np.asarray(post_step_active_mask, dtype=np.float32).reshape(self.n_agents, 1),
            "agent_energy": np.asarray(
                [agent.energy for agent in self.env.agents],
                dtype=np.float32,
            ).reshape(self.n_agents, 1),
        }
        return reward, terminated, self._last_info

    def estimate_joint_short_risk(self, actions):
        active_mask = self.get_active_agent_mask()
        agent_actions = [
            self._continuous_agent_action(action, active_mask[idx] > 0.0)
            for idx, action in enumerate(actions)
        ]
        return self.env.estimate_joint_risk(agent_actions)

    def estimate_joint_short_risk_batch(self, actions_batch):
        actions_batch = np.asarray(actions_batch)
        if actions_batch.ndim != 3:
            raise ValueError("actions_batch must have shape (batch, n_agents, action_dim)")
        agent_actions_batch = np.asarray(
            [
                [
                    self._continuous_agent_action(action, True)
                    for action in joint_actions
                ]
                for joint_actions in actions_batch
            ],
            dtype=np.float32,
        )
        return self.env.estimate_joint_risk_batch(agent_actions_batch)

    def estimate_joint_collision_flags(self, actions):
        active_mask = self.get_active_agent_mask()
        agent_actions = [
            self._continuous_agent_action(action, active_mask[idx] > 0.0)
            for idx, action in enumerate(actions)
        ]
        return self.env.predict_joint_collision_flags(agent_actions)

    def get_env_info(self):
        self.reset()
        obs = self.get_obs()
        state = self.get_state()
        return {
            "n_actions": self.n_actions,
            "low_action_type": self.low_action_type,
            "low_action_dim": self.action_dim,
            "n_agents": self.n_agents,
            "state_shape": int(state.shape[0]),
            "obs_shape": int(obs.shape[-1]),
            "episode_limit": self.episode_limit,
            "msg_shape": int(self.env.msg_shape),
        }

    def get_obs(self):
        if self._last_obs is None:
            self.reset()
        return np.asarray(self._last_obs, dtype=np.float32)

    def get_obs_agent(self, agent_id):
        return self.get_obs()[agent_id]

    def get_state(self):
        return self.env.get_state()

    def get_msg(self):
        return self.env.get_msg()

    def get_avail_agent_actions(self, agent_id):
        del agent_id
        return np.ones(self.n_actions, dtype=np.float32)

    def get_active_agent_mask(self):
        return self.env.get_active_agent_mask()

    def summary(self):
        summary = self.env.summary()
        summary["episode_reward"] = self._last_reward
        return summary

    def close(self):
        self.env.close()


class UAVParallelEnv:
    metadata = {"name": "uav_delivery_env"}

    def __init__(
        self,
        dim_actions=3,
        length=boundary_length,
        width=boundary_width,
        height=boundary_height,
        num_obstacle=default_num_obstacles,
        num_hunters=4,
        num_targets=1,
        max_cycles=200,
        continuous_actions=True,
        render_mode=None,
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
    ):
        self.render_mode = render_mode
        self.continuous_actions = continuous_actions
        self.max_cycles = int(max_cycles)
        self.possible_agents = [f"agent_{i}" for i in range(num_hunters)]
        self.max_num_agents = len(self.possible_agents)
        self.agents = list(self.possible_agents)
        self._agent_index = {
            agent_name: idx for idx, agent_name in enumerate(self.possible_agents)
        }
        self.unwrapped = self

        self.base_env = UAVEnv(
            dim_actions=dim_actions,
            length=length,
            width=width,
            height=height,
            num_obstacle=num_obstacle,
            num_hunters=num_hunters,
            num_targets=num_targets,
            episode_limit=self.max_cycles,
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
        self.msg_shape = getattr(self.base_env, "msg_shape", 0)

        if continuous_actions:
            self.action_spaces = {
                agent_name: self.base_env.action_space[agent_name]
                for agent_name in self.possible_agents
            }
        else:
            discrete_actions = self._build_discrete_actions()
            self._discrete_actions = discrete_actions
            discrete_space = spaces.Discrete(len(discrete_actions))
            self.action_spaces = {
                agent_name: discrete_space
                for agent_name in self.possible_agents
            }
        self.observation_spaces = {
            agent_name: self.base_env.observation_space[agent_name]
            for agent_name in self.possible_agents
        }

    def _build_discrete_actions(self):
        action_values = [-1.0, 0.0, 1.0]
        discrete_actions = []
        for action in itertools.product(action_values, repeat=self.base_env.dim_actions):
            vec = np.array(action, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > eps:
                vec = vec / norm
            discrete_actions.append(vec * self.base_env.a_max)
        return discrete_actions

    def action_space(self, agent):
        return self.action_spaces[agent]

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def _obs_array_to_dict(self, obs_array):
        return {
            agent_name: np.asarray(obs_array[idx], dtype=np.float32)
            for idx, agent_name in enumerate(self.possible_agents)
        }

    def _info_dict(self):
        summary = self.base_env.summary()
        return {
            agent_name: dict(summary)
            for agent_name in self.possible_agents
        }

    def reset(self, seed=None, options=None):
        del options
        obs_array = self.base_env.reset(seed=seed)
        self.agents = list(self.possible_agents)
        return self._obs_array_to_dict(obs_array), self._info_dict()

    def _convert_action(self, agent_name, action):
        if self.continuous_actions:
            return np.asarray(action, dtype=np.float32)
        return np.asarray(
            self._discrete_actions[int(action)],
            dtype=np.float32,
        )

    def step(self, actions):
        active_mask = self.base_env.get_active_agent_mask()
        action_list = [
            (
                self._convert_action(agent_name, actions[agent_name])
                if active_mask[idx] > 0.0
                else np.zeros(self.base_env.dim_actions, dtype=np.float32)
            )
            for idx, agent_name in enumerate(self.possible_agents)
        ]
        obs_array, rewards, reached_flags, safe_value = self.base_env.step(action_list)
        collision_flags = [bool(agent.collided) for agent in self.base_env.agents]

        success_terminated = bool(all(reached_flags))
        collision_terminated = bool(any(collision_flags))
        post_step_active_mask = self.base_env.get_active_agent_mask()
        all_depleted = bool(np.sum(post_step_active_mask) <= 0.0)
        terminated = success_terminated or all_depleted
        truncated = bool(
            self.base_env.current_step >= self.max_cycles and not terminated
        )
        obs_dict = self._obs_array_to_dict(obs_array)
        reward_dict = {
            agent_name: float(rewards[idx])
            for idx, agent_name in enumerate(self.possible_agents)
        }
        done_dict = {
            agent_name: terminated
            for agent_name in self.possible_agents
        }
        trunc_dict = {
            agent_name: truncated
            for agent_name in self.possible_agents
        }
        info_dict = {
            agent_name: {
                "safe_value": float(safe_value[idx]),
                "reached": bool(reached_flags[idx]),
                "collided": bool(collision_flags[idx]),
                "active": bool(post_step_active_mask[idx]),
                "energy": float(self.base_env.agents[idx].energy),
                "collision_terminated": collision_terminated,
                "success_terminated": success_terminated,
                "all_depleted": all_depleted,
                **self.base_env.summary(),
            }
            for idx, agent_name in enumerate(self.possible_agents)
        }

        self.agents = [] if terminated or truncated else list(self.possible_agents)
        return obs_dict, reward_dict, done_dict, trunc_dict, info_dict

    def render(self, view=None):
        if self.render_mode == "human":
            self.base_env.render(show=True, view=view)
            return None
        return self.base_env.render(show=False, view=view)

    def get_msg(self):
        return self.base_env.get_msg()

    def close(self):
        self.base_env.close()


def parallel_env(
    dim_actions=3,
    length=boundary_length,
    width=boundary_width,
    height=boundary_height,
    num_obstacle=default_num_obstacles,
    num_hunters=4,
    num_targets=1,
    max_cycles=200,
    continuous_actions=True,
    render_mode=None,
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
):
    return UAVParallelEnv(
        dim_actions=dim_actions,
        length=length,
        width=width,
        height=height,
        num_obstacle=num_obstacle,
        num_hunters=num_hunters,
        num_targets=num_targets,
        max_cycles=max_cycles,
        continuous_actions=continuous_actions,
        render_mode=render_mode,
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
