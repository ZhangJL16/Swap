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
    distance_potential_scale: float = 0.25
    gamma: float = 0.99
    velocity_toward_goal_weight: float = 0.1
    time_cost: float = 0.01
    task_completion_reward: float = 10.0
    collision_penalty: float = 1.2
    energy_cost_weight: float = 0.01
    backup_intervention_cost: float = 0.1

    def __post_init__(self) -> None:
        if self.distance_potential_scale < 0.0:
            raise ValueError("distance_potential_scale must be nonnegative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")


@dataclass(frozen=True)
class EnergyNavigationConfig:
    battery_capacity: float = 30.0
    charging_rate: float = 2.0
    charging_radius: float = 0.18
    charging_velocity_limit: tuple[float, float, float] = (0.05, 0.05, 0.04)
    initial_energy_fraction_min: float = 0.30
    initial_energy_fraction_max: float = 1.00

    def __post_init__(self) -> None:
        if not np.isfinite(self.battery_capacity) or self.battery_capacity <= 0.0:
            raise ValueError("battery_capacity must be positive")
        if not np.isfinite(self.charging_rate) or self.charging_rate <= 0.0:
            raise ValueError("charging_rate must be positive")
        if not np.isfinite(self.charging_radius) or self.charging_radius <= 0.0:
            raise ValueError("charging_radius must be positive")
        velocity_limit = as_vec3(np.asarray(self.charging_velocity_limit), "charging_velocity_limit")
        if np.any(velocity_limit <= 0.0):
            raise ValueError("charging velocity limits must be positive")
        if not 0.0 <= self.initial_energy_fraction_min <= self.initial_energy_fraction_max <= 1.0:
            raise ValueError("initial energy fractions must lie in [0, 1]")


class PersistentNavigationEnv(gym.Env[np.ndarray, np.ndarray]):
    """Direct-SAC navigation environment isolated from certification internals.

    Scenario ``coverage_waypoints`` are offline certification partition seeds;
    they are never task waypoints, observations, rewards, or sampling inputs here.
    This phase uses a nonterminating large energy budget and does not establish
    charging or energy-management learnability.
    """

    metadata = {"render_modes": []}
    distance_level_boundaries = (2.0, 1.2, 0.7, 0.4)

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
        if goal_radius != 0.20:
            raise ValueError("the formal distance-potential baseline requires goal_radius=0.20")
        if minimum_goal_separation <= goal_radius:
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
        self.observation_space = gym.spaces.Box(
            observation_low.astype(np.float32),
            np.ones(cursor, dtype=np.float32),
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
        self.current_consecutive_boundary_contacts = 0
        self.maximum_consecutive_boundary_contacts = 0
        self.boundary_lock_event_count = 0
        self._episode_index = -1
        self._next_goal_index = 0
        self._episode_truncated = False
        self._goal_attempt: dict[str, Any] = {}

    @property
    def observation_fields(self) -> tuple[str, ...]:
        return tuple(self.observation_layout)

    def distance_potential(self, distance: float) -> float:
        value = float(distance)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("distance must be finite and nonnegative")
        if value > 2.0:
            return 0.0
        if value > 1.2:
            return 1.0
        if value > 0.7:
            return 2.0
        if value > 0.4:
            return 3.0
        return 4.0

    def distance_potential_shaping(self, distance_before: float, distance_after: float) -> float:
        potential_before = self.distance_potential(distance_before)
        potential_after = self.distance_potential(distance_after)
        return self.reward_config.distance_potential_scale * (
            self.reward_config.gamma * potential_after - potential_before
        )

    def signed_velocity_toward_goal(self, velocity: np.ndarray, position: np.ndarray, goal: np.ndarray) -> float:
        velocity_array = as_vec3(velocity, "velocity")
        direction = as_vec3(goal, "goal") - as_vec3(position, "position")
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-12:
            return 0.0
        projection = float(np.dot(velocity_array, direction / direction_norm))
        normalized = projection / float(np.linalg.norm(self.config.v_max))
        return float(np.clip(normalized, -1.0, 1.0))

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
                np.array([self.state.energy / self._energy_observation_capacity]),
                self.last_lidar.distances / self.config.lidar_range,
                self.last_lidar.valid.astype(np.float64),
            )
        )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError("nonfinite navigation observation")
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    @property
    def _energy_observation_capacity(self) -> float:
        return self.navigation_energy_capacity

    def _initial_energy(self, reset_options: dict[str, Any]) -> float:
        del reset_options
        return self.navigation_energy_capacity

    def _start_goal_attempt(self) -> None:
        delta = self.goal - self.state.position
        self._goal_attempt = {
            "goal_id": f"episode-{self._episode_index}-goal-{self._next_goal_index}",
            "goal_absolute_coordinates": self.goal.copy(),
            "goal_start_episode_step": self.episode_step,
            "goal_initial_distance": float(np.linalg.norm(delta)),
            "initial_xy_distance": float(np.linalg.norm(delta[:2])),
            "initial_z_distance": float(abs(delta[2])),
            "start_position": self.state.position.copy(),
            "steps": 0,
            "collisions_during_goal": 0,
            "boundary_contacts_during_goal": 0,
            "velocity_saturations_during_goal": 0,
            "signed_velocity_sum": 0.0,
            "reward_component_totals": {
                "distance_potential_shaping": 0.0,
                "signed_velocity_toward_goal_reward": 0.0,
                "time_cost": 0.0,
                "task_completion_reward": 0.0,
                "collision_penalty": 0.0,
                "energy_cost": 0.0,
                "backup_intervention_event_cost": 0.0,
            },
        }
        self._next_goal_index += 1

    def _update_goal_attempt(
        self,
        reward_components: dict[str, float],
        signed_velocity: float,
        collision: bool,
        boundary_collision: bool,
        velocity_saturated: bool,
    ) -> None:
        self._goal_attempt["steps"] += 1
        self._goal_attempt["collisions_during_goal"] += int(collision)
        self._goal_attempt["boundary_contacts_during_goal"] += int(boundary_collision)
        self._goal_attempt["velocity_saturations_during_goal"] += int(velocity_saturated)
        self._goal_attempt["signed_velocity_sum"] += signed_velocity
        for name, value in reward_components.items():
            self._goal_attempt["reward_component_totals"][name] += float(value)

    def _finalize_goal_attempt(self, *, completed: bool, final_distance: float) -> dict[str, Any]:
        steps = int(self._goal_attempt["steps"])
        record = {
            key: value
            for key, value in self._goal_attempt.items()
            if key not in {"signed_velocity_sum", "steps"}
        }
        record |= {
            "status": "completed" if completed else "unfinished_goal",
            "completed": completed,
            "completion_episode_step": self.episode_step if completed else None,
            "attempt_end_episode_step": self.episode_step,
            "steps_to_goal": steps if completed else None,
            "attempt_steps": steps,
            "final_distance": float(final_distance),
            "mean_signed_velocity_toward_goal": self._goal_attempt["signed_velocity_sum"] / max(1, steps),
        }
        return record

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        reset_options = {} if options is None else dict(options)
        start = (
            self._validate_reset_position(reset_options["start_position"], "start_position")
            if "start_position" in reset_options
            else self._sample_free_position()
        )
        goal = (
            self._validate_reset_position(reset_options["goal_position"], "goal_position")
            if "goal_position" in reset_options
            else self._sample_free_position(start)
        )
        if "start_velocity" in reset_options:
            velocity = as_vec3(np.asarray(reset_options["start_velocity"], dtype=np.float64), "start_velocity")
            velocity = np.clip(velocity, -self.config.v_max, self.config.v_max)
        else:
            velocity = np.zeros(3)

        self.state = UAVPhysicalState(start, velocity, self._initial_energy(reset_options), 0.0)
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
        self.current_consecutive_boundary_contacts = 0
        self.maximum_consecutive_boundary_contacts = 0
        self.boundary_lock_event_count = 0
        self._episode_index += 1
        self._next_goal_index = 0
        self._episode_truncated = False
        self._start_goal_attempt()
        return self._observation(), {
            "sampled_start": start.copy(),
            "sampled_goal": goal.copy(),
            "goal_id": self._goal_attempt["goal_id"],
            "observation_layout": dict(self.observation_layout),
            "energy_semantics": "navigation_baseline_nonterminating_large_budget",
            "phase_scope": "DOES_NOT_ESTABLISH_CHARGING_OR_ENERGY_MANAGEMENT_LEARNABILITY",
        }

    def _correct_collision(
        self,
        position_before: np.ndarray,
        candidate_position: np.ndarray,
        candidate_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        radius = self.config.body_radius
        safe_low = np.full(3, radius)
        safe_high = self.config.world_size - radius
        boundary_axes = (candidate_position < safe_low) | (candidate_position > safe_high)
        corrected_position = np.clip(candidate_position, safe_low, safe_high)
        corrected_velocity = candidate_velocity.copy()
        corrected_velocity[boundary_axes] = 0.0

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
        return corrected_position, corrected_velocity, boundary_axes, obstacle_collision

    def _integrate_motion(
        self,
        state_before: UAVPhysicalState,
        physical_action: np.ndarray,
    ) -> dict[str, Any]:
        candidate_position, raw_velocity = integrate_double_integrator(
            state_before.position,
            state_before.velocity,
            physical_action,
            self.config.dt,
        )
        clipped_velocity = np.clip(raw_velocity, -self.config.v_max, self.config.v_max)
        velocity_saturated = bool(np.any(np.abs(raw_velocity) > self.config.v_max + 1e-12))
        position_after, velocity_after, boundary_axes, obstacle_collision = self._correct_collision(
            state_before.position,
            candidate_position,
            clipped_velocity,
        )
        boundary_collision = bool(np.any(boundary_axes))
        return {
            "position_after": position_after,
            "velocity_after": velocity_after,
            "boundary_axes": boundary_axes,
            "boundary_collision": boundary_collision,
            "obstacle_collision": obstacle_collision,
            "collision": boundary_collision or obstacle_collision,
            "velocity_saturated": velocity_saturated,
            "executed_physical_acceleration": physical_action.copy(),
        }

    def _transition_state(
        self,
        state_before: UAVPhysicalState,
        physical_action: np.ndarray,
    ) -> dict[str, Any]:
        transition = self._integrate_motion(state_before, physical_action)
        energy_usage = self.energy_model.realized_cost(state_before, physical_action, self.config.dt)
        energy_after = max(0.0, state_before.energy - energy_usage)
        if energy_after <= 0.0:
            raise RuntimeError("navigation baseline energy budget exhausted")
        transition |= {
            "energy_after": energy_after,
            "energy_usage": energy_usage,
            "flight_energy_used": energy_usage,
            "gross_charge_received": 0.0,
            "net_energy_change": -energy_usage,
            "charging": False,
            "inside_charging_region": False,
            "energy_stranded": False,
        }
        return transition

    def step(self, action: np.ndarray):
        if self._episode_truncated:
            raise RuntimeError("step called after truncation without reset")
        normalized_action = np.asarray(action, dtype=np.float64)
        physical_action = self.normalized_to_physical_action(normalized_action)
        state_before = self.state.copy()
        goal_before = self.goal.copy()
        goal_id_before = str(self._goal_attempt["goal_id"])
        distance_before = float(np.linalg.norm(goal_before - state_before.position))

        transition = self._transition_state(state_before, physical_action)
        position_after = transition["position_after"]
        velocity_after = transition["velocity_after"]
        boundary_axes = transition["boundary_axes"]
        obstacle_collision = bool(transition["obstacle_collision"])
        boundary_collision = bool(transition["boundary_collision"])
        collision = bool(transition["collision"])
        velocity_saturated = bool(transition["velocity_saturated"])
        energy_usage = float(transition["energy_usage"])
        energy_after = float(transition["energy_after"])
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
        potential_before = self.distance_potential(distance_before)
        potential_after = self.distance_potential(distance_after)
        potential_shaping = self.distance_potential_shaping(distance_before, distance_after)
        signed_velocity = self.signed_velocity_toward_goal(self.state.velocity, self.state.position, goal_before)
        task_completed_now = distance_after <= self.goal_radius + 1e-12
        reward_components = {
            "distance_potential_shaping": potential_shaping,
            "signed_velocity_toward_goal_reward": self.reward_config.velocity_toward_goal_weight * signed_velocity,
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
        if boundary_collision:
            self.current_consecutive_boundary_contacts += 1
        else:
            self.current_consecutive_boundary_contacts = 0
        self.maximum_consecutive_boundary_contacts = max(
            self.maximum_consecutive_boundary_contacts,
            self.current_consecutive_boundary_contacts,
        )
        boundary_lock_event = None
        if self.current_consecutive_boundary_contacts == 101:
            self.boundary_lock_event_count += 1
            boundary_lock_event = {
                "event": "BOUNDARY_LOCK_EVENT",
                "episode_step": self.episode_step,
                "consecutive_boundary_contacts": self.current_consecutive_boundary_contacts,
                "boundary_axes": np.flatnonzero(boundary_axes).tolist(),
                "position": self.state.position.copy(),
            }

        self._update_goal_attempt(
            reward_components,
            signed_velocity,
            collision,
            boundary_collision,
            velocity_saturated,
        )
        goal_attempt_records: list[dict[str, Any]] = []
        completed_goal = None
        if task_completed_now:
            completed_goal = goal_before.copy()
            goal_attempt_records.append(self._finalize_goal_attempt(completed=True, final_distance=distance_after))
            self.tasks_completed += 1
            self.goal = self._sample_free_position(self.state.position)
            self._start_goal_attempt()

        truncated = self.episode_step >= self.max_episode_steps
        if truncated:
            current_distance = float(np.linalg.norm(self.goal - self.state.position))
            goal_attempt_records.append(self._finalize_goal_attempt(completed=False, final_distance=current_distance))
            self._episode_truncated = True
        self.last_lidar = self.lidar_model.measure(self.state, self.world, self.np_random)

        info = {
            "normalized_action": normalized_action.astype(np.float32),
            "physical_acceleration": physical_action.astype(np.float32),
            "executed_physical_acceleration": np.asarray(
                transition["executed_physical_acceleration"], dtype=np.float32
            ),
            "reward_components": reward_components,
            "goal_id_before": goal_id_before,
            "goal_before": goal_before,
            "goal_progress": goal_progress,
            "distance_to_goal_before": distance_before,
            "distance_to_goal_after": distance_after,
            "distance_potential_before": potential_before,
            "distance_potential_after": potential_after,
            "minimum_goal_distance": self.minimum_goal_distance,
            "mean_goal_distance": self.goal_distance_sum / self.goal_distance_samples,
            "signed_velocity_toward_goal": signed_velocity,
            "velocity_toward_goal": signed_velocity,
            "task_completed_now": task_completed_now,
            "completed_goal": completed_goal,
            "current_goal": self.goal.copy(),
            "current_goal_id": self._goal_attempt["goal_id"],
            "tasks_completed": self.tasks_completed,
            "tasks_per_1000_steps": 1000.0 * self.tasks_completed / self.episode_step,
            "goal_attempt_records": goal_attempt_records,
            "collision": collision,
            "boundary_collision": boundary_collision,
            "boundary_axes": np.flatnonzero(boundary_axes).tolist(),
            "obstacle_collision": obstacle_collision,
            "collision_count": self.collision_count,
            "collision_rate": self.collision_count / self.episode_step,
            "boundary_collision_count": self.boundary_collision_count,
            "boundary_collision_rate": self.boundary_collision_count / self.episode_step,
            "current_consecutive_boundary_contacts": self.current_consecutive_boundary_contacts,
            "maximum_consecutive_boundary_contacts": self.maximum_consecutive_boundary_contacts,
            "boundary_lock_event": boundary_lock_event,
            "boundary_lock_event_count": self.boundary_lock_event_count,
            "velocity_saturated": velocity_saturated,
            "velocity_saturation_count": self.velocity_saturation_count,
            "velocity_saturation_rate": self.velocity_saturation_count / self.episode_step,
            "energy_usage": energy_usage,
            "flight_energy_used": float(transition["flight_energy_used"]),
            "gross_charge_received": float(transition["gross_charge_received"]),
            "net_energy_change": float(transition["net_energy_change"]),
            "charging": bool(transition["charging"]),
            "inside_charging_region": bool(transition["inside_charging_region"]),
            "energy_stranded": bool(transition["energy_stranded"]),
            "state_of_charge": self.state.energy / self._energy_observation_capacity,
            "cumulative_energy_usage": self.cumulative_energy_usage,
            "kappa_takeover_count": 0,
            "fallback_count": 0,
            "physical_state": self.state.copy(),
        }
        return self._observation(), reward, False, truncated, info


class PersistentEnergyNavigationEnv(PersistentNavigationEnv):
    """Finite-energy persistent navigation with continuous station charging."""

    def __init__(
        self,
        scenario_name: str = "random_persistent_open.json",
        *,
        energy_config: EnergyNavigationConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.energy_navigation_config = energy_config or EnergyNavigationConfig()
        kwargs.pop("navigation_energy_capacity", None)
        super().__init__(
            scenario_name,
            navigation_energy_capacity=self.energy_navigation_config.battery_capacity,
            **kwargs,
        )
        self.station_visit_count = 0
        self.charging_session_count = 0
        self.successful_charging_session_count = 0
        self.successful_resume_count = 0
        self.energy_stranded_count = 0
        self.minimum_soc = 1.0
        self.soc_sum = 0.0
        self.soc_samples = 0
        self._was_inside_station = False
        self._was_charging = False
        self._energy_stranded_active = False
        self._active_charging_session: dict[str, Any] | None = None
        self._tasks_since_last_charge = 0

    @property
    def _energy_observation_capacity(self) -> float:
        return self.energy_navigation_config.battery_capacity

    @property
    def charging_velocity_limit(self) -> np.ndarray:
        return np.asarray(self.energy_navigation_config.charging_velocity_limit, dtype=np.float64)

    def _initial_energy(self, reset_options: dict[str, Any]) -> float:
        if "initial_energy_fraction" in reset_options:
            fraction = float(reset_options["initial_energy_fraction"])
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("initial_energy_fraction must lie in [0, 1]")
        elif "initial_energy_fraction_range" in reset_options:
            low, high = (float(value) for value in reset_options["initial_energy_fraction_range"])
            if not 0.0 <= low <= high <= 1.0:
                raise ValueError("initial_energy_fraction_range must lie in [0, 1]")
            fraction = float(self.np_random.uniform(low, high))
        else:
            fraction = float(
                self.np_random.uniform(
                    self.energy_navigation_config.initial_energy_fraction_min,
                    self.energy_navigation_config.initial_energy_fraction_max,
                )
            )
        return fraction * self.energy_navigation_config.battery_capacity

    def _inside_station(self, position: np.ndarray) -> bool:
        return bool(
            np.linalg.norm(as_vec3(position, "position") - self.scenario.station_position)
            <= self.energy_navigation_config.charging_radius + 1e-12
        )

    def _charging_admissible(self, position: np.ndarray, velocity: np.ndarray) -> bool:
        return self._inside_station(position) and bool(
            np.all(np.abs(as_vec3(velocity, "velocity")) <= self.charging_velocity_limit + 1e-12)
        )

    def _transition_state(
        self,
        state_before: UAVPhysicalState,
        physical_action: np.ndarray,
    ) -> dict[str, Any]:
        if state_before.energy <= 0.0:
            zero = np.zeros(3, dtype=np.float64)
            transition = {
                "position_after": state_before.position.copy(),
                "velocity_after": zero.copy(),
                "boundary_axes": np.zeros(3, dtype=bool),
                "boundary_collision": False,
                "obstacle_collision": False,
                "collision": False,
                "velocity_saturated": False,
                "executed_physical_acceleration": zero.copy(),
            }
            flight_energy_demand = 0.0
        else:
            transition = self._integrate_motion(state_before, physical_action)
            flight_energy_demand = self.energy_model.realized_cost(
                state_before,
                physical_action,
                self.config.dt,
            )
        flight_energy_used = min(state_before.energy, flight_energy_demand)
        energy_after_flight = max(0.0, state_before.energy - flight_energy_used)
        charging = self._charging_admissible(
            transition["position_after"],
            transition["velocity_after"],
        )
        gross_charge_received = 0.0
        if charging:
            gross_charge_received = min(
                self.energy_navigation_config.charging_rate * self.config.dt,
                self.energy_navigation_config.battery_capacity - energy_after_flight,
            )
        energy_after = float(
            np.clip(
                energy_after_flight + gross_charge_received,
                0.0,
                self.energy_navigation_config.battery_capacity,
            )
        )
        transition |= {
            "energy_after": energy_after,
            "energy_usage": flight_energy_used,
            "flight_energy_used": flight_energy_used,
            "flight_energy_demand": flight_energy_demand,
            "energy_after_flight": energy_after_flight,
            "gross_charge_received": gross_charge_received,
            "net_energy_change": energy_after - state_before.energy,
            "charging": charging,
            "inside_charging_region": self._inside_station(transition["position_after"]),
            "energy_stranded": energy_after <= 0.0 and not charging,
        }
        return transition

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        observation, info = super().reset(seed=seed, options=options)
        initial_soc = self.state.energy / self.energy_navigation_config.battery_capacity
        self.station_visit_count = int(self._inside_station(self.state.position))
        self.charging_session_count = 0
        self.successful_charging_session_count = 0
        self.successful_resume_count = 0
        self.energy_stranded_count = 0
        self.minimum_soc = initial_soc
        self.soc_sum = initial_soc
        self.soc_samples = 1
        self._was_inside_station = self._inside_station(self.state.position)
        self._was_charging = False
        self._energy_stranded_active = False
        self._active_charging_session = None
        self._tasks_since_last_charge = 0
        info |= {
            "initial_energy": self.state.energy,
            "initial_soc": initial_soc,
            "energy_mode": "finite_charging",
        }
        return observation, info

    def _finalize_charging_session(self, status: str, current_goal_id: str) -> dict[str, Any]:
        if self._active_charging_session is None:
            raise RuntimeError("charging session accounting is inconsistent")
        session = dict(self._active_charging_session)
        interrupted_goal_id = str(session["interrupted_pending_goal_id"])
        resumed = status == "departed" and current_goal_id == interrupted_goal_id
        session |= {
            "status": status,
            "charge_end_episode_step": self.episode_step,
            "charge_end_soc": float(session["last_charging_soc"]),
            "departure_soc": self.state.energy / self.energy_navigation_config.battery_capacity,
            "resumed_pending_goal_id": current_goal_id if status == "departed" else None,
            "pending_goal_preserved": resumed,
            "successful_resume": resumed,
            "successful_charge": status == "departed" and session["energy_received"] > 0.0,
        }
        session.pop("last_charging_soc")
        if session["successful_charge"]:
            self.successful_charging_session_count += 1
        if resumed:
            self.successful_resume_count += 1
        return session

    def step(self, action: np.ndarray):
        observation, reward, terminated, truncated, info = super().step(action)
        soc = self.state.energy / self.energy_navigation_config.battery_capacity
        self.minimum_soc = min(self.minimum_soc, soc)
        self.soc_sum += soc
        self.soc_samples += 1
        if info["task_completed_now"]:
            self._tasks_since_last_charge += 1

        inside_station = bool(info["inside_charging_region"])
        charging = bool(info["charging"])
        station_visit_now = inside_station and not self._was_inside_station
        charging_session_started_now = charging and not self._was_charging
        if inside_station and not self._was_inside_station:
            self.station_visit_count += 1

        charging_session_records: list[dict[str, Any]] = []
        if charging_session_started_now:
            self.charging_session_count += 1
            start_soc = float(
                (self.state.energy - info["gross_charge_received"])
                / self.energy_navigation_config.battery_capacity
            )
            self._active_charging_session = {
                "charging_session_id": f"episode-{self._episode_index}-charge-{self.charging_session_count - 1}",
                "charge_start_episode_step": self.episode_step,
                "charge_start_soc": start_soc,
                "interrupted_pending_goal_id": info["goal_id_before"],
                "tasks_between_charges": self._tasks_since_last_charge,
                "charging_duration_steps": 0,
                "energy_received": 0.0,
                "last_charging_soc": soc,
                "voluntary": True,
            }
            self._tasks_since_last_charge = 0
        if charging:
            if self._active_charging_session is None:
                raise RuntimeError("charging active without a charging session")
            self._active_charging_session["charging_duration_steps"] += 1
            self._active_charging_session["energy_received"] += float(info["gross_charge_received"])
            self._active_charging_session["last_charging_soc"] = soc
        charging_session_ended_now = not charging and self._was_charging
        if charging_session_ended_now:
            charging_session_records.append(
                self._finalize_charging_session("departed", str(info["goal_id_before"]))
            )
            self._active_charging_session = None

        stranding_event = None
        if info["energy_stranded"] and not self._energy_stranded_active:
            self.energy_stranded_count += 1
            stranding_event = {
                "event": "ENERGY_STRANDED",
                "episode_step": self.episode_step,
                "goal_id": info["goal_id_before"],
                "position": self.state.position.copy(),
                "state_of_charge": soc,
            }
        self._energy_stranded_active = bool(info["energy_stranded"])

        if truncated and self._active_charging_session is not None:
            charging_session_records.append(
                self._finalize_charging_session("active_at_truncation", str(info["current_goal_id"]))
            )
            self._active_charging_session = None

        self._was_inside_station = inside_station
        self._was_charging = charging and not truncated
        info |= {
            "station_visit_count": self.station_visit_count,
            "station_visit_now": station_visit_now,
            "charging_session_count": self.charging_session_count,
            "charging_session_started_now": charging_session_started_now,
            "charging_session_ended_now": charging_session_ended_now,
            "voluntary_charging_session_count": self.charging_session_count,
            "successful_charging_session_count": self.successful_charging_session_count,
            "successful_resume_count": self.successful_resume_count,
            "energy_stranded_count": self.energy_stranded_count,
            "minimum_soc": self.minimum_soc,
            "mean_soc": self.soc_sum / self.soc_samples,
            "charging_session_records": charging_session_records,
            "stranding_event": stranding_event,
            "energy_stranded_now": stranding_event is not None,
            "energy_mode": "finite_charging",
        }
        return observation, reward, terminated, truncated, info


def make_sb3_persistent_navigation_env(
    scenario_name: str = "random_persistent_open.json",
    **kwargs: Any,
) -> PersistentNavigationEnv:
    return PersistentNavigationEnv(scenario_name, **kwargs)


def make_sb3_persistent_energy_navigation_env(
    scenario_name: str = "random_persistent_open.json",
    **kwargs: Any,
) -> PersistentEnergyNavigationEnv:
    return PersistentEnergyNavigationEnv(scenario_name, **kwargs)
