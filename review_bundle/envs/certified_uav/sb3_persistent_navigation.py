from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .config import CertifiedUAVConfig, apply_configuration_overrides
from .dynamics import integrate_double_integrator
from .energy import EnergyModel, SimulationEnergyConfig
from .lidar import HorizontalLidarModel
from .scenario import FixedCertificationScenario, ScenarioDefinition
from .state import UAVPhysicalState, as_vec3


@dataclass(frozen=True)
class NavigationRewardConfig:
    progress_weight: float = 2.5
    velocity_toward_goal_weight: float = 0.1
    time_cost: float = 0.01
    task_completion_reward: float = 10.0
    collision_penalty: float = 1.2
    energy_cost_weight: float = 0.01
    backup_intervention_cost: float = 0.1


class PersistentNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Direct-SAC navigation environment isolated from certification internals.

    Scenario ``coverage_waypoints`` are offline certification partition seeds;
    they are never task waypoints, observations, rewards, or sampling inputs here.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario_name: str = "random_persistent_open.json",
        *,
        max_episode_steps: int = 5000,
        navigation_energy_capacity: float = 1000.0,
        goal_radius: float = 0.20,
        minimum_goal_separation: float = 0.60,
        sampling_margin: float = 0.20,
        reward_config: NavigationRewardConfig | None = None,
    ) -> None:
        super().__init__()
        self.scenario: ScenarioDefinition = FixedCertificationScenario(scenario_name).definition
        base_config = CertifiedUAVConfig(world_size=self.scenario.world_size)
        self.config = apply_configuration_overrides(base_config, self.scenario.configuration_overrides)
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if navigation_energy_capacity <= 0.0:
            raise ValueError("navigation_energy_capacity must be positive")
        if goal_radius <= 0.0 or minimum_goal_separation <= goal_radius:
            raise ValueError("goal sampling distances are invalid")
        if sampling_margin < self.config.body_radius:
            raise ValueError("sampling_margin must cover the UAV body radius")

        self.world = self.scenario.world
        self.max_episode_steps = int(max_episode_steps)
        self.navigation_energy_capacity = float(navigation_energy_capacity)
        self.goal_radius = float(goal_radius)
        self.minimum_goal_separation = float(minimum_goal_separation)
        self.sampling_margin = float(sampling_margin)
        self.reward_config = reward_config or NavigationRewardConfig()
        self.energy_model = EnergyModel(SimulationEnergyConfig())
        self.lidar_model = HorizontalLidarModel(
            self.config.num_lasers,
            self.config.lidar_range,
            "synthetic-sensor-v1",
            self.config.lidar_range_noise,
            self.config.lidar_pose_noise,
            self.config.lidar_heading_noise,
            self.config.lidar_invalid_probability,
        )

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.observation_layout: dict[str, slice] = {}
        cursor = 0
        for name, length in (
            ("absolute_position", 3),
            ("velocity", 3),
            ("absolute_goal_position", 3),
            ("absolute_station_position", 3),
            ("state_of_charge", 1),
            ("lidar_distances", self.config.num_lasers),
            ("lidar_valid", self.config.num_lasers),
        ):
            self.observation_layout[name] = slice(cursor, cursor + length)
            cursor += length
        observation_low = np.concatenate(
            (
                np.zeros(3),
                -np.ones(3),
                np.zeros(3),
                np.zeros(3),
                np.zeros(1),
                np.zeros(self.config.num_lasers),
                np.zeros(self.config.num_lasers),
            )
        )
        observation_high = np.ones(cursor)
        observation_high[self.observation_layout["velocity"]] = 1.0
        self.observation_space = gym.spaces.Box(
            observation_low.astype(np.float32),
            observation_high.astype(np.float32),
            dtype=np.float32,
        )

        self.state = UAVPhysicalState(
            self.scenario.initial_state.position,
            np.zeros(3),
            self.navigation_energy_capacity,
            0.0,
        )
        self.goal = self.scenario.task_goal.copy()
        self.last_lidar = None
        self.episode_step = 0
        self.tasks_completed = 0
        self.collision_count = 0
        self.boundary_collision_count = 0
        self.obstacle_collision_count = 0
        self.velocity_saturation_count = 0
        self.cumulative_energy_usage = 0.0
        self.minimum_goal_distance = float("inf")
        self.goal_distance_sum = 0.0
        self.goal_distance_samples = 0

    @property
    def observation_fields(self) -> tuple[str, ...]:
        return tuple(self.observation_layout)

    def normalized_to_physical_action(self, action: np.ndarray) -> np.ndarray:
        normalized = np.asarray(action, dtype=np.float64)
        if normalized.shape != (3,) or not np.all(np.isfinite(normalized)):
            raise ValueError("SAC action must be a finite (3,) array")
        if np.any(normalized < -1.0 - 1e-6) or np.any(normalized > 1.0 + 1e-6):
            raise ValueError("SAC action must lie in [-1, 1]^3")
        return np.clip(normalized, -1.0, 1.0) * self.config.a_max

    def _is_legal_position(self, position: np.ndarray) -> bool:
        point = as_vec3(position, "position")
        radius = self.config.body_radius
        if np.any(point < radius - 1e-12) or np.any(point > self.config.world_size - radius + 1e-12):
            return False
        return not self.world.swept_collision(point, point, radius)

    def _sample_free_position(self, reference: np.ndarray | None = None) -> np.ndarray:
        low = np.full(3, self.sampling_margin)
        high = self.config.world_size - self.sampling_margin
        for _ in range(10000):
            candidate = self.np_random.uniform(low, high)
            if not self._is_legal_position(candidate):
                continue
            if reference is not None and np.linalg.norm(candidate - reference) < self.minimum_goal_separation:
                continue
            return candidate
        raise RuntimeError("failed to sample a free-space task position")

    def _validate_reset_position(self, value: Any, name: str) -> np.ndarray:
        point = as_vec3(np.asarray(value, dtype=np.float64), name)
        if not self._is_legal_position(point):
            raise ValueError(f"{name} must be collision-free")
        return point

    def _observation(self) -> np.ndarray:
        if self.last_lidar is None:
            raise RuntimeError("LiDAR packet unavailable")
        observation = np.concatenate(
            (
                self.state.position / self.config.world_size,
                self.state.velocity / self.config.v_max,
                self.goal / self.config.world_size,
                self.scenario.station_position / self.config.world_size,
                np.array([self.state.energy / self.navigation_energy_capacity]),
                self.last_lidar.distances / self.config.lidar_range,
                self.last_lidar.valid.astype(np.float64),
            )
        )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("nonfinite navigation observation")
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        reset_options = {} if options is None else dict(options)
        if "start_position" in reset_options:
            start = self._validate_reset_position(reset_options["start_position"], "start_position")
        else:
            start = self._sample_free_position()
        if "goal_position" in reset_options:
            goal = self._validate_reset_position(reset_options["goal_position"], "goal_position")
        else:
            goal = self._sample_free_position(start)
        if "start_velocity" in reset_options:
            velocity = as_vec3(np.asarray(reset_options["start_velocity"], dtype=np.float64), "start_velocity")
            velocity = np.clip(velocity, -self.config.v_max, self.config.v_max)
        else:
            velocity = np.zeros(3)

        self.state = UAVPhysicalState(start, velocity, self.navigation_energy_capacity, 0.0)
        self.goal = goal
        self.last_lidar = self.lidar_model.measure(self.state, self.world, self.np_random)
        self.episode_step = 0
        self.tasks_completed = 0
        self.collision_count = 0
        self.boundary_collision_count = 0
        self.obstacle_collision_count = 0
        self.velocity_saturation_count = 0
        self.cumulative_energy_usage = 0.0
        initial_distance = float(np.linalg.norm(self.goal - self.state.position))
        self.minimum_goal_distance = initial_distance
        self.goal_distance_sum = initial_distance
        self.goal_distance_samples = 1
        return self._observation(), {
            "sampled_start": start.copy(),
            "sampled_goal": goal.copy(),
            "observation_layout": dict(self.observation_layout),
            "energy_semantics": "navigation_baseline_nonterminating_large_budget",
        }

    def _correct_collision(
        self,
        position_before: np.ndarray,
        candidate_position: np.ndarray,
        candidate_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        radius = self.config.body_radius
        safe_low = np.full(3, radius)
        safe_high = self.config.world_size - radius
        boundary_axes = (candidate_position < safe_low) | (candidate_position > safe_high)
        corrected_position = np.clip(candidate_position, safe_low, safe_high)
        corrected_velocity = candidate_velocity.copy()
        corrected_velocity[boundary_axes] = 0.0
        boundary_collision = bool(np.any(boundary_axes))

        obstacle_collision = self.world.swept_collision(position_before, corrected_position, radius)
        if obstacle_collision:
            lower_fraction = 0.0
            upper_fraction = 1.0
            displacement = corrected_position - position_before
            for _ in range(48):
                middle_fraction = 0.5 * (lower_fraction + upper_fraction)
                middle_position = position_before + middle_fraction * displacement
                if self.world.swept_collision(position_before, middle_position, radius):
                    upper_fraction = middle_fraction
                else:
                    lower_fraction = middle_fraction
            corrected_position = position_before + lower_fraction * displacement
            corrected_velocity[:] = 0.0
        if not self._is_legal_position(corrected_position):
            raise RuntimeError("collision correction failed to produce a legal state")
        return corrected_position, corrected_velocity, boundary_collision, obstacle_collision

    def step(self, action: np.ndarray):
        normalized_action = np.asarray(action, dtype=np.float64)
        physical_action = self.normalized_to_physical_action(normalized_action)
        state_before = self.state.copy()
        goal_before = self.goal.copy()
        distance_before = float(np.linalg.norm(goal_before - state_before.position))

        candidate_position, raw_velocity = integrate_double_integrator(
            state_before.position,
            state_before.velocity,
            physical_action,
            self.config.dt,
        )
        clipped_velocity = np.clip(raw_velocity, -self.config.v_max, self.config.v_max)
        velocity_saturated = bool(np.any(np.abs(raw_velocity) > self.config.v_max + 1e-12))
        position_after, velocity_after, boundary_collision, obstacle_collision = self._correct_collision(
            state_before.position,
            candidate_position,
            clipped_velocity,
        )
        collision = boundary_collision or obstacle_collision

        energy_usage = self.energy_model.realized_cost(state_before, physical_action, self.config.dt)
        energy_after = max(0.0, state_before.energy - energy_usage)
        if energy_after <= 0.0:
            raise RuntimeError("navigation baseline energy budget exhausted")
        self.state = UAVPhysicalState(
            position_after,
            velocity_after,
            energy_after,
            state_before.timestamp + self.config.dt,
        )
        if not np.all(np.isfinite(np.concatenate((self.state.position, self.state.velocity, [self.state.energy])))):
            raise FloatingPointError("nonfinite simulator state")

        distance_after = float(np.linalg.norm(goal_before - self.state.position))
        goal_progress = distance_before - distance_after
        goal_direction = goal_before - self.state.position
        goal_direction_norm = float(np.linalg.norm(goal_direction))
        if goal_direction_norm > 1e-12:
            velocity_projection = float(np.dot(self.state.velocity, goal_direction / goal_direction_norm))
            velocity_toward_goal = max(0.0, velocity_projection / float(np.linalg.norm(self.config.v_max)))
        else:
            velocity_toward_goal = 0.0
        task_completed_now = distance_after <= self.goal_radius + 1e-12

        reward_components = {
            "goal_progress_reward": self.reward_config.progress_weight * goal_progress,
            "velocity_toward_goal_reward": self.reward_config.velocity_toward_goal_weight * velocity_toward_goal,
            "time_cost": -self.reward_config.time_cost,
            "task_completion_reward": self.reward_config.task_completion_reward * float(task_completed_now),
            "collision_penalty": -self.reward_config.collision_penalty * float(collision),
            "energy_cost": -self.reward_config.energy_cost_weight * energy_usage,
            "backup_intervention_event_cost": 0.0,
        }
        reward = float(sum(reward_components.values()))
        if not np.isfinite(reward):
            raise FloatingPointError("nonfinite navigation reward")

        self.episode_step += 1
        self.collision_count += int(collision)
        self.boundary_collision_count += int(boundary_collision)
        self.obstacle_collision_count += int(obstacle_collision)
        self.velocity_saturation_count += int(velocity_saturated)
        self.cumulative_energy_usage += energy_usage
        self.minimum_goal_distance = min(self.minimum_goal_distance, distance_after)
        self.goal_distance_sum += distance_after
        self.goal_distance_samples += 1
        completed_goal = goal_before.copy() if task_completed_now else None
        if task_completed_now:
            self.tasks_completed += 1
            self.goal = self._sample_free_position(self.state.position)
        self.last_lidar = self.lidar_model.measure(self.state, self.world, self.np_random)

        terminated = False
        truncated = self.episode_step >= self.max_episode_steps
        info = {
            "normalized_action": normalized_action.astype(np.float32),
            "physical_acceleration": physical_action.astype(np.float32),
            "reward_components": reward_components,
            "goal_progress": goal_progress,
            "distance_to_goal_before": distance_before,
            "distance_to_goal_after": distance_after,
            "minimum_goal_distance": self.minimum_goal_distance,
            "mean_goal_distance": self.goal_distance_sum / self.goal_distance_samples,
            "velocity_toward_goal": velocity_toward_goal,
            "task_completed_now": task_completed_now,
            "completed_goal": completed_goal,
            "current_goal": self.goal.copy(),
            "tasks_completed": self.tasks_completed,
            "tasks_per_1000_steps": 1000.0 * self.tasks_completed / self.episode_step,
            "collision": collision,
            "boundary_collision": boundary_collision,
            "obstacle_collision": obstacle_collision,
            "collision_count": self.collision_count,
            "collision_rate": self.collision_count / self.episode_step,
            "velocity_saturated": velocity_saturated,
            "velocity_saturation_count": self.velocity_saturation_count,
            "energy_usage": energy_usage,
            "cumulative_energy_usage": self.cumulative_energy_usage,
            "kappa_takeover_count": 0,
            "fallback_count": 0,
            "physical_state": self.state.copy(),
        }
        return self._observation(), reward, terminated, truncated, info


def make_sb3_persistent_navigation_env(
    scenario_name: str = "random_persistent_open.json",
    **kwargs: Any,
) -> PersistentNavigationEnv:
    return PersistentNavigationEnv(scenario_name, **kwargs)
