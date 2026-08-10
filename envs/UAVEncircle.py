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
eps = 1e-6
legend_font_size_xy = 26
legend_font_size_3d = 26


def _laser_angles(num_lasers):
    return np.linspace(0.0, 2.0 * math.pi, num_lasers, endpoint=False, dtype=np.float32)


def update_lasers_to_boundary(agent_pos, l_sensor, num_lasers, length, width):
    origin = np.asarray(agent_pos, dtype=np.float32)[:2]
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
    origin = np.asarray(agent_pos, dtype=np.float32)[:2]
    center = np.asarray(obstacle_pos, dtype=np.float32)[:2]
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
    def __init__(self, number, pos, v_max, a_max, num_lasers, l_sensor, safe_radius, dim):
        self.number = number
        self.pos = np.asarray(pos, dtype=np.float32)
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

    def update_velocity(self, action, time_step):
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
        self.prev_pos = self.pos + self.vel * time_step


class GoalPoint:
    def __init__(self, pos):
        self.pos = np.asarray(pos, dtype=np.float32)


class MovingTarget:
    def __init__(
        self,
        pos,
        v_max,
        a_max,
        safe_radius,
        dim,
        max_interference=100.0,
        interference_decay_rate=0.1,
    ):
        self.pos = np.asarray(pos, dtype=np.float32)
        self.prev_pos = np.asarray(pos, dtype=np.float32)
        self.last_pos = np.asarray(pos, dtype=np.float32)
        self.vel = np.zeros(dim, dtype=np.float32)
        self.v_max = float(v_max)
        self.a_max = float(a_max)
        self.safe_radius = float(safe_radius)
        self.dim = int(dim)
        self.max_interference = float(max_interference)
        self.interference = 0.0
        self.interference_decay_rate = float(interference_decay_rate)

    def interference_decay(self):
        self.interference *= 1.0 - self.interference_decay_rate

    def update_velocity(self, action, time_step):
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
        self.prev_pos = self.pos + self.vel * time_step


class Obstacle:
    def __init__(self, length, width, radius_range=(0.12, 0.2)):
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
        num_hunters=4,
        num_targets=1,
        episode_limit=200,
        reset_retry_limit=20,
        sample_retry_limit=100,
        obstacle_crash_penalty=10.0,
    ):
        if dim_actions not in (2, 3):
            raise ValueError("Dimension must be 2 or 3")

        self.dim_actions = dim_actions
        self.length = length
        self.width = width
        self.height = height if dim_actions == 3 else 0.0
        self.num_obstacle = num_obstacle
        self.num_agents = num_hunters
        self.num_hunters = num_hunters  # compatibility with the old wrapper
        self.num_targets = max(1, int(num_targets))
        self.episode_limit = episode_limit
        self.reset_retry_limit = reset_retry_limit
        self.sample_retry_limit = sample_retry_limit
        self.obstacle_crash_penalty = float(obstacle_crash_penalty)

        self.time_step = 0.4
        self.goal_tolerance = 0.12
        self.capture_radius = 0.35
        self.v_max = 0.16
        self.a_max = 0.05
        self.target_v_max = 0.12
        self.target_a_max = 0.035
        self.safe_radius = 0.05
        self.target_safe_radius = 0.06
        self.target_escape_radius = 0.7
        self.interference_decay_rate = 0.1
        self.interference_radius = 0.7
        self.interference_strength = 2.0
        self.min_interference_hunters = min(3, self.num_hunters)
        self.max_interference = 100.0
        self.velocity_reward_weight = 0.9
        self.risk_warning_margin = 0.06
        self.guard_prediction_margin = 0.04
        self.l_sensor = 0.35
        self.num_lasers = 16
        self.msg_shape = self.dim_actions * 2 + 2

        self.max_cycles = episode_limit
        self.agents = []
        self.target = None
        self.targets = []
        self.goals = []
        self.obstacles = []
        self.agent_paths = [[] for _ in range(self.num_agents)]
        self.target_path = []
        self.current_step = 0
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self.collision_count = 0.0
        self.obstacle_collision_count = 0.0
        self.agent_collision_count = 0.0
        self._render_fig = None
        self._render_has_shown = False

        obs_dim = (
            2 * self.dim_actions
            + self.num_lasers
            + self.dim_actions * (self.num_agents - 1)
            + (self.dim_actions + 1)
        )
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
            if self._cylinder_distance(goal, obstacle) <= self.target_safe_radius + obstacle.radius + 0.05:
                return False
        for existing in existing_goals:
            if np.linalg.norm(goal - existing.pos) <= 0.25:
                return False
        for agent in agents:
            if np.linalg.norm(goal - agent.pos) <= self.target_escape_radius:
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
                )
                candidate.interference_radius = self.interference_radius
                candidate.interference_strength = self.interference_strength
                if self._is_valid_spawn(pos, self.agents, self.safe_radius):
                    self.agents.append(candidate)
                    break
            else:
                raise RuntimeError(
                    f'Failed to place UAV {agent_idx} after {self.sample_retry_limit} attempts.'
                )

    def _sync_agent_goals(self):
        if self.target is None:
            return
        for agent in self.agents:
            agent.goal = self.target.pos.copy()
        self.goals = [GoalPoint(self.target.pos.copy())]

    def _initialize_goals(self):
        self.goals = []
        self.target = None
        self.targets = []
        for _ in range(self.sample_retry_limit):
            target_pos = self._sample_position(lower_margin=0.35)
            candidate_goal = GoalPoint(target_pos)
            if self._is_valid_goal(target_pos, self.goals, self.agents):
                self.target = MovingTarget(
                    pos=target_pos,
                    v_max=self.target_v_max,
                    a_max=self.target_a_max,
                    safe_radius=self.target_safe_radius,
                    dim=self.dim_actions,
                    max_interference=self.max_interference,
                    interference_decay_rate=self.interference_decay_rate,
                )
                self.targets = [self.target]
                self.goals = [candidate_goal]
                self._sync_agent_goals()
                return
        raise RuntimeError(
            f'Failed to place moving target after {self.sample_retry_limit} attempts.'
        )

    def reset(self, seed=None):
        if seed is None:
            seed = random.randint(1, 100000)
        random.seed(seed)
        np.random.seed(seed)

        last_error = None
        for _ in range(self.reset_retry_limit):
            self.current_step = 0
            self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
            self.collision_count = 0.0
            self.obstacle_collision_count = 0.0
            self.agent_collision_count = 0.0
            self.agent_paths = [[] for _ in range(self.num_agents)]
            self.target_path = []
            self.obstacles = []
            self.agents = []
            self.target = None
            self.targets = []
            self.goals = []

            try:
                self._initialize_obstacles()
                self._initialize_agents()
                self._initialize_goals()
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
        boundaries = [self.length, self.width, self.height][: self.dim_actions]
        for dim, boundary in enumerate(boundaries):
            if agent.prev_pos[dim] - agent.safe_radius < 0:
                agent.prev_pos[dim] = agent.safe_radius
                agent.vel[dim] = 0.0
                collided = True
            elif agent.prev_pos[dim] + agent.safe_radius > boundary:
                agent.prev_pos[dim] = boundary - agent.safe_radius
                agent.vel[dim] = 0.0
                collided = True
        return collided

    def _resolve_obstacle_collisions(self, agent):
        collided = False
        safe_penalty = 0.0
        for obstacle in self.obstacles:
            delta_xy = agent.prev_pos[:2] - obstacle.pos
            dist_xy = np.linalg.norm(delta_xy)
            min_dist = obstacle.radius + agent.safe_radius
            safe_penalty = max(safe_penalty, max(0.0, min_dist - dist_xy))
            if dist_xy < min_dist:
                collided = True
                if dist_xy < eps:
                    delta_xy = np.array([1.0, 0.0], dtype=np.float32)
                    dist_xy = 1.0
                direction_xy = delta_xy / dist_xy
                corrected_xy = obstacle.pos + direction_xy * min_dist
                agent.prev_pos[:2] = corrected_xy
                agent.vel[:2] = 0.0
        return collided, safe_penalty

    def _resolve_agent_collisions(self):
        collided = [False] * self.num_agents
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                delta = self.agents[i].prev_pos - self.agents[j].prev_pos
                dist = np.linalg.norm(delta)
                min_dist = self.agents[i].safe_radius + self.agents[j].safe_radius
                self.safe_value[i] += max(0.0, min_dist - dist)
                self.safe_value[j] += max(0.0, min_dist - dist)
                if dist < min_dist:
                    collided[i] = True
                    collided[j] = True
                    if dist < eps:
                        delta = np.zeros(self.dim_actions, dtype=np.float32)
                        delta[0] = 1.0
                        dist = 1.0
                    direction = delta / dist
                    overlap = min_dist - dist
                    self.agents[i].prev_pos += direction * (overlap / 2.0 + eps)
                    self.agents[j].prev_pos -= direction * (overlap / 2.0 + eps)
                    self.agents[i].vel[:] = 0.0
                    self.agents[j].vel[:] = 0.0
        return collided

    def _target_escape_action(self):
        if self.target is None:
            return np.zeros(self.dim_actions, dtype=np.float32)

        force = np.zeros(self.dim_actions, dtype=np.float32)
        hunter_positions = [
            getattr(agent, "prev_pos", agent.pos) for agent in self.agents
        ]
        for hunter_pos in hunter_positions:
            delta = self.target.pos - hunter_pos
            dist = float(np.linalg.norm(delta))
            if dist < self.target_escape_radius:
                force += delta / (dist + eps) * (
                    (self.target_escape_radius - dist) / self.target_escape_radius
                )

        for obstacle in self.obstacles:
            delta_xy = self.target.pos[:2] - obstacle.pos
            dist_xy = float(np.linalg.norm(delta_xy))
            avoid_radius = obstacle.radius + self.target_safe_radius + 0.35
            if dist_xy < avoid_radius:
                direction_xy = delta_xy / (dist_xy + eps)
                obstacle_force = (avoid_radius - dist_xy) / avoid_radius
                force[:2] += direction_xy * obstacle_force

        boundaries = [self.length, self.width, self.height][: self.dim_actions]
        wall_margin = 0.45
        for dim_idx, boundary in enumerate(boundaries):
            lower = float(self.target.pos[dim_idx])
            upper = float(boundary - self.target.pos[dim_idx])
            if lower < wall_margin:
                force[dim_idx] += (wall_margin - lower) / wall_margin
            if upper < wall_margin:
                force[dim_idx] -= (wall_margin - upper) / wall_margin

        if np.linalg.norm(force) < eps:
            force = np.random.normal(0.0, 0.2, size=self.dim_actions).astype(np.float32)

        norm = np.linalg.norm(force)
        if norm > eps:
            force = force / norm * self.target.a_max
        return force.astype(np.float32)

    def _update_target(self):
        if self.target is None:
            return
        action = self._target_escape_action()
        self.target.update_velocity(action, self.time_step)
        self.target.preview_position(self.time_step)
        self._apply_boundary_constraints(self.target)
        self._resolve_obstacle_collisions(self.target)
        self.target.pos = self.target.prev_pos.copy()
        self.target_path.append(self.target.pos.copy())
        self._sync_agent_goals()

    def _update_interference(self):
        if self.target is None:
            return 0, 0.0
        active_hunters = 0
        interference_value = 0.0
        for hunter in self.agents:
            distance = float(np.linalg.norm(self.target.pos - hunter.pos))
            if distance <= getattr(hunter, "interference_radius", self.interference_radius):
                active_hunters += 1
                interference_value += getattr(
                    hunter, "interference_strength", self.interference_strength
                )
        if active_hunters >= self.min_interference_hunters:
            self.target.interference = min(
                self.target.max_interference,
                self.target.interference + interference_value,
            )
        elif active_hunters == 0:
            self.target.interference_decay()
        return active_hunters, interference_value

    def _predict_agent_kinematics(self, agent, action):
        accel = np.asarray(action, dtype=np.float32).copy()
        norm = np.linalg.norm(accel)
        if norm > agent.a_max:
            accel = accel / (norm + eps) * agent.a_max

        pred_vel = agent.vel + accel * self.time_step
        speed = np.linalg.norm(pred_vel)
        if speed > agent.v_max:
            pred_vel = pred_vel / (speed + eps) * agent.v_max

        pred_pos = agent.pos + pred_vel * self.time_step
        return pred_pos.astype(np.float32), pred_vel.astype(np.float32)

    def _evaluate_joint_risk_batch_exact(self, actions_batch):
        actions_batch = np.asarray(actions_batch, dtype=np.float32)
        if actions_batch.ndim != 3:
            raise ValueError("actions_batch must have shape (batch, n_agents, dim_actions)")
        if actions_batch.shape[1] != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")
        if actions_batch.shape[2] != self.dim_actions:
            raise ValueError("Action dimension does not match UAV action dimension.")

        batch_size = actions_batch.shape[0]
        positions = np.asarray([agent.pos for agent in self.agents], dtype=np.float32)
        velocities = np.asarray([agent.vel for agent in self.agents], dtype=np.float32)
        safe_radii = np.asarray([agent.safe_radius for agent in self.agents], dtype=np.float32)
        reached = np.asarray([agent.reached for agent in self.agents], dtype=bool)
        a_max = np.asarray([agent.a_max for agent in self.agents], dtype=np.float32)
        v_max = np.asarray([agent.v_max for agent in self.agents], dtype=np.float32)
        boundaries = np.asarray([self.length, self.width, self.height][: self.dim_actions], dtype=np.float32)
        obstacle_pos = np.asarray([obstacle.pos for obstacle in self.obstacles], dtype=np.float32)
        obstacle_radius = np.asarray([obstacle.radius for obstacle in self.obstacles], dtype=np.float32)

        safe_values = np.zeros((batch_size, self.num_agents), dtype=np.float32)
        for batch_idx in range(batch_size):
            pred_pos = positions.copy()
            pred_vel = velocities.copy()
            for agent_idx in range(self.num_agents):
                if reached[agent_idx]:
                    continue
                accel = actions_batch[batch_idx, agent_idx].copy()
                norm = np.linalg.norm(accel)
                if norm > a_max[agent_idx]:
                    accel = accel / (norm + eps) * a_max[agent_idx]
                pred_vel[agent_idx] = pred_vel[agent_idx] + accel * self.time_step
                speed = np.linalg.norm(pred_vel[agent_idx])
                if speed > v_max[agent_idx]:
                    pred_vel[agent_idx] = pred_vel[agent_idx] / (speed + eps) * v_max[agent_idx]
                pred_pos[agent_idx] = positions[agent_idx] + pred_vel[agent_idx] * self.time_step

            for agent_idx in range(self.num_agents):
                if reached[agent_idx]:
                    continue
                warning_clearance = safe_radii[agent_idx] + self.risk_warning_margin
                for dim_idx, boundary in enumerate(boundaries):
                    lower_overlap = max(0.0, warning_clearance - float(pred_pos[agent_idx, dim_idx]))
                    upper_overlap = max(0.0, warning_clearance - float(boundary - pred_pos[agent_idx, dim_idx]))
                    safe_values[batch_idx, agent_idx] += self._nonlinear_risk_from_overlap(lower_overlap, warning_clearance)
                    safe_values[batch_idx, agent_idx] += self._nonlinear_risk_from_overlap(upper_overlap, warning_clearance)

                obstacle_risk = 0.0
                for obs_idx in range(len(self.obstacles)):
                    dist_xy = np.linalg.norm(pred_pos[agent_idx, :2] - obstacle_pos[obs_idx])
                    collision_min_dist = obstacle_radius[obs_idx] + safe_radii[agent_idx]
                    warning_min_dist = collision_min_dist + self.risk_warning_margin
                    overlap = max(0.0, warning_min_dist - dist_xy)
                    obstacle_risk = max(
                        obstacle_risk,
                        self._nonlinear_risk_from_overlap(overlap, warning_min_dist),
                    )
                safe_values[batch_idx, agent_idx] += obstacle_risk

            for i in range(self.num_agents):
                if reached[i]:
                    continue
                for j in range(i + 1, self.num_agents):
                    if reached[j]:
                        continue
                    dist = np.linalg.norm(pred_pos[i] - pred_pos[j])
                    collision_min_dist = safe_radii[i] + safe_radii[j]
                    warning_min_dist = collision_min_dist + self.risk_warning_margin
                    pair_risk = self._nonlinear_risk_from_overlap(
                        max(0.0, warning_min_dist - dist), warning_min_dist
                    )
                    safe_values[batch_idx, i] += pair_risk
                    safe_values[batch_idx, j] += pair_risk
        return safe_values

    def estimate_joint_risk(self, actions):
        return self._evaluate_joint_risk_batch_exact(np.asarray([actions], dtype=np.float32))[0]

    def estimate_joint_risk_batch(self, actions_batch):
        return self._evaluate_joint_risk_batch_exact(actions_batch)

    def predict_joint_collision_flags(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        predicted_pos = []
        for agent, action in zip(self.agents, actions):
            if agent.reached:
                predicted_pos.append(agent.pos.copy())
                continue
            pred_pos, _ = self._predict_agent_kinematics(agent, action)
            predicted_pos.append(pred_pos)

        predicted_pos = np.asarray(predicted_pos, dtype=np.float32)
        collision_flags = np.zeros(self.num_agents, dtype=bool)
        boundaries = [self.length, self.width, self.height][: self.dim_actions]

        for idx, agent in enumerate(self.agents):
            if agent.reached:
                continue
            guard_clearance = agent.safe_radius + self.guard_prediction_margin
            for dim_idx, boundary in enumerate(boundaries):
                if (
                    predicted_pos[idx, dim_idx] - guard_clearance < 0.0
                    or predicted_pos[idx, dim_idx] + guard_clearance > boundary
                ):
                    collision_flags[idx] = True
                    break
            if collision_flags[idx]:
                continue
            for obstacle in self.obstacles:
                dist_xy = np.linalg.norm(predicted_pos[idx, :2] - obstacle.pos)
                if dist_xy < obstacle.radius + agent.safe_radius + self.guard_prediction_margin:
                    collision_flags[idx] = True
                    break

        for i in range(self.num_agents):
            if self.agents[i].reached:
                continue
            for j in range(i + 1, self.num_agents):
                if self.agents[j].reached:
                    continue
                dist = np.linalg.norm(predicted_pos[i] - predicted_pos[j])
                min_dist = (
                    self.agents[i].safe_radius
                    + self.agents[j].safe_radius
                    + self.guard_prediction_margin
                )
                if dist < min_dist:
                    collision_flags[i] = True
                    collision_flags[j] = True
        return collision_flags.astype(np.float32)

    def step(self, actions):
        if len(actions) != self.num_agents:
            raise ValueError("Action count does not match the number of UAVs.")

        self.current_step += 1
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        dones = [False] * self.num_agents
        self.safe_value = np.zeros(self.num_agents, dtype=np.float32)
        self._sync_agent_goals()
        prev_interference = 0.0 if self.target is None else float(self.target.interference)

        prev_dists = np.array([self._distance_to_goal(agent) for agent in self.agents], dtype=np.float32)

        for agent, action in zip(self.agents, actions):
            agent.reached = False
            agent.update_velocity(action, self.time_step)
            agent.preview_position(self.time_step)

        self._update_target()

        obstacle_collisions = [False] * self.num_agents
        for idx, agent in enumerate(self.agents):
            boundary_collision = self._apply_boundary_constraints(agent)
            obstacle_collision, obstacle_penalty = self._resolve_obstacle_collisions(agent)
            obstacle_collisions[idx] = boundary_collision or obstacle_collision
            self.safe_value[idx] += obstacle_penalty + float(boundary_collision or obstacle_collision)

        agent_collisions = self._resolve_agent_collisions()

        for agent in self.agents:
            agent.pos = agent.prev_pos.copy()

        active_hunters, _ = self._update_interference()
        interference_progress = (
            0.0
            if self.target is None
            else max(0.0, float(self.target.interference) - prev_interference)
            / (self.target.max_interference + eps)
        )
        interference_ratio = (
            0.0
            if self.target is None
            else float(self.target.interference) / (self.target.max_interference + eps)
        )

        self.update_lasers()

        capture_flags = []
        for idx, agent in enumerate(self.agents):
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
            rewards[idx] += 2.5 * progress
            rewards[idx] += self.velocity_reward_weight * max(0.0, velocity_toward_goal)
            rewards[idx] -= 0.01
            rewards[idx] -= 1.2 * float(obstacle_collisions[idx])
            rewards[idx] -= 1.5 * float(agent_collisions[idx])
            rewards[idx] -= 0.2 * self.safe_value[idx]
            rewards[idx] -= 0.3 * min(current_dist, 1.0)
            rewards[idx] += 4.0 * interference_progress
            rewards[idx] += 0.2 * interference_ratio

            captured = current_dist <= self.capture_radius
            capture_flags.append(captured)
            if captured:
                rewards[idx] += 2.0
            if current_dist <= self.interference_radius:
                rewards[idx] += 0.5

            dones[idx] = False
            agent.reached = captured
            agent.collided = obstacle_collisions[idx] or agent_collisions[idx]
            self.agent_paths[idx].append(agent.pos.copy())

        obstacle_collision_total = float(np.sum(np.asarray(obstacle_collisions, dtype=bool)))
        agent_collision_total = float(np.sum(np.asarray(agent_collisions, dtype=bool)))
        self.obstacle_collision_count += obstacle_collision_total
        self.agent_collision_count += agent_collision_total
        self.collision_count += obstacle_collision_total + agent_collision_total

        interference_done = (
            self.target is not None
            and self.target.interference >= self.target.max_interference
        )
        no_collision = not any(obstacle_collisions) and not any(agent_collisions)
        if interference_done and active_hunters >= self.min_interference_hunters and no_collision:
            rewards += 50.0
            dones = [True] * self.num_agents

        return self.get_obs(), rewards, dones, self.safe_value.copy()

    def _goal_features(self, agent):
        delta = agent.goal - agent.pos
        distance = np.linalg.norm(delta) / (np.linalg.norm(self._space_scale()) + eps)
        return np.concatenate([delta / (self._space_scale() + eps), np.array([distance], dtype=np.float32)])

    def _message_from_sender(self, receiver, sender):
        del receiver
        scale = self._space_scale() + eps
        goal_delta = (sender.goal - sender.pos) / scale
        velocity = sender.vel / (sender.v_max + eps)
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
        return np.concatenate([goal_delta, velocity, sender_risk, sender_reached])

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
                messages.append(np.zeros((0, self.msg_shape), dtype=np.float32))
        return messages

    def get_obs(self):
        observations = []
        scale = self._space_scale() + eps
        for agent in self.agents:
            own = np.concatenate([agent.pos / scale, agent.vel / (agent.v_max + eps)])
            ally = []
            for other in self.agents:
                if other is agent:
                    continue
                ally.append((other.pos - agent.pos) / scale)
            ally = np.concatenate(ally, axis=0) if ally else np.array([], dtype=np.float32)
            obs = np.concatenate([own, np.asarray(agent.lasers, dtype=np.float32), ally, self._goal_features(agent)])
            observations.append(obs.astype(np.float32))
        return np.stack(observations, axis=0)

    def get_state(self):
        parts = []
        scale = self._space_scale() + eps
        for agent in self.agents:
            parts.append(agent.pos / scale)
            parts.append(agent.vel / (agent.v_max + eps))
            parts.append(agent.goal / scale)
            parts.append(np.array([float(agent.reached)], dtype=np.float32))
        for obstacle in self.obstacles:
            parts.append(obstacle.pos / scale[:2])
            parts.append(np.array([obstacle.radius / max(self.length, self.width)], dtype=np.float32))
        return np.concatenate(parts, axis=0).astype(np.float32)

    def update_lasers(self):
        for agent in self.agents:
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
        remaining = [self._distance_to_goal(agent) for agent in self.agents]
        target_interference = 0.0 if self.target is None else float(self.target.interference)
        max_interference = self.max_interference if self.target is None else self.target.max_interference
        return {
            "step": float(self.current_step),
            "agent_health": float(np.sum([not agent.collided for agent in self.agents])),
            "enemy_health": float(max(0.0, 1.0 - target_interference / (max_interference + eps))),
            "agent_alive": float(np.sum([agent.reached for agent in self.agents])),
            "collision_count": float(self.collision_count),
            "obstacle_collision_count": float(self.obstacle_collision_count),
            "agent_collision_count": float(self.agent_collision_count),
            "target_interference": target_interference,
            "episode_reward": float(0.0),
            "win_tag": bool(target_interference >= max_interference),
        }

    def _agent_color(self, idx):
        cmap = __import__("matplotlib").cm.get_cmap("tab10", max(self.num_agents, 1))
        return cmap(idx % max(self.num_agents, 1))

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

    def _render_target_positions(self):
        targets = getattr(self, "targets", None)
        if targets:
            return [np.asarray(target.pos, dtype=np.float32) for target in targets]
        target = getattr(self, "target", None)
        if target is not None and hasattr(target, "pos"):
            return [np.asarray(target.pos, dtype=np.float32)]
        if self.goals:
            goal_positions = np.asarray([goal.pos for goal in self.goals], dtype=np.float32)
            return [np.mean(goal_positions, axis=0)]
        return []

    def _render_2d(self, ax):
        from matplotlib.patches import Circle

        for target_pos in self._render_target_positions():
            target_traj = np.asarray(self.target_path, dtype=np.float32)
            if target_traj.ndim == 2 and len(target_traj) > 1:
                ax.plot(target_traj[:, 0], target_traj[:, 1], color="red", alpha=0.9, linewidth=1.8)
            ax.scatter(
                target_pos[0],
                target_pos[1],
                c=["red"],
                marker="*",
                s=180,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                label="target",
            )
            ax.add_patch(Circle(target_pos[:2], 0.2, color="red", fill=False, linestyle="--", alpha=0.35))

        for idx, agent in enumerate(self.agents):
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

        for target_pos in self._render_target_positions():
            target_z = target_pos[2] if self.dim_actions == 3 else 0.0
            target_traj = np.asarray(self.target_path, dtype=np.float32)
            if target_traj.ndim == 2 and len(target_traj) > 1:
                ax.plot(
                    target_traj[:, 0],
                    target_traj[:, 1],
                    target_traj[:, 2],
                    color="red",
                    alpha=0.9,
                    linewidth=1.8,
                )
            ax.scatter(
                target_pos[0],
                target_pos[1],
                target_z,
                c=["red"],
                marker="*",
                s=180,
                edgecolors="black",
                linewidths=0.8,
                alpha=0.9,
                label="target",
            )

        for idx, agent in enumerate(self.agents):
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
        num_hunters=4,
        num_targets=1,
        episode_limit=200,
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
        )
        self.dim_actions = dim_actions
        self.episode_limit = episode_limit
        self.n_agents = self.env.num_agents
        self.discrete_actions = self._build_discrete_actions()
        self.n_actions = len(self.discrete_actions)
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
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        self._last_obs, rewards, dones, safe_value = self.env.step(agent_actions)
        self._episode_steps += 1
        reward = float(np.mean(rewards))
        battle_won = bool(all(dones))
        terminated = battle_won or self._episode_steps >= self.episode_limit
        self._last_reward = reward
        self._last_info = {
            "battle_won": battle_won,
            "warning_signal": np.asarray(safe_value, dtype=np.float32).reshape(self.n_agents, 1),
            "per_agent_reward": np.asarray(rewards, dtype=np.float32).copy(),
        }
        return reward, terminated, self._last_info

    def estimate_joint_short_risk(self, actions):
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        return self.env.estimate_joint_risk(agent_actions)

    def estimate_joint_short_risk_batch(self, actions_batch):
        actions_batch = np.asarray(actions_batch)
        if actions_batch.ndim != 2:
            raise ValueError("actions_batch must have shape (batch, n_agents)")
        agent_actions_batch = np.asarray(
            [
                [self.discrete_actions[int(action)] for action in joint_actions]
                for joint_actions in actions_batch
            ],
            dtype=np.float32,
        )
        return self.env.estimate_joint_risk_batch(agent_actions_batch)

    def estimate_joint_collision_flags(self, actions):
        agent_actions = [self.discrete_actions[int(action)] for action in actions]
        return self.env.predict_joint_collision_flags(agent_actions)

    def get_env_info(self):
        self.reset()
        obs = self.get_obs()
        state = self.get_state()
        return {
            "n_actions": self.n_actions,
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
        return np.ones(self.n_actions, dtype=np.float32)

    def summary(self):
        summary = self.env.summary()
        summary["episode_reward"] = self._last_reward
        return summary

    def close(self):
        self.env.close()


class UAVParallelEnv:
    metadata = {"name": "uav_env"}

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
        )

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
        action_list = [
            self._convert_action(agent_name, actions[agent_name])
            for agent_name in self.possible_agents
        ]
        obs_array, rewards, reached_flags, safe_value = self.base_env.step(action_list)

        terminated = bool(all(reached_flags))
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
    )
