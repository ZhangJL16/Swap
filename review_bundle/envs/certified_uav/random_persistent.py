from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .persistent_task import PersistentMissionMode, PersistentRewardConfig
from .plant_env import CertifiedSingleUAVPlantEnv
from .recovery_atlas import CertifiedRecoverabilityAtlas
from .state import as_vec3


@dataclass(slots=True)
class RandomGoalTask:
    task_id: str
    goal_position: np.ndarray
    reward: float
    assignment_step: int
    completion_step: int | None = None
    interrupted_by_charge: bool = False

    def __post_init__(self) -> None:
        self.goal_position = as_vec3(self.goal_position, "random persistent goal")

    @property
    def goal_node(self) -> str:
        return self.task_id


class RandomPersistentGoalManager:
    """Seeded continuous goals sampled from certified atlas interiors."""

    def __init__(
        self,
        atlas: CertifiedRecoverabilityAtlas,
        goal_radius: float,
        minimum_goal_separation: float,
        task_reward: float,
    ) -> None:
        self.atlas = atlas
        self.goal_radius = float(goal_radius)
        self.minimum_goal_separation = float(minimum_goal_separation)
        self.task_reward = float(task_reward)
        self.rng = np.random.default_rng(0)
        self.current_task: RandomGoalTask | None = None
        self.tasks_completed = 0
        self.task_interruption_count = 0
        self.task_resume_count = 0
        self._next_task_id = 0
        self.completed_tasks: list[RandomGoalTask] = []
        self.goal_sequence: list[np.ndarray] = []

    def reset(self, seed: int | None, position: np.ndarray, assignment_step: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.current_task = None
        self.tasks_completed = 0
        self.task_interruption_count = 0
        self.task_resume_count = 0
        self._next_task_id = 0
        self.completed_tasks = []
        self.goal_sequence = []
        self.assign_next_goal(position, assignment_step)

    @property
    def navigation_target(self) -> np.ndarray:
        if self.current_task is None:
            raise RuntimeError("random persistent goal is unavailable")
        return self.current_task.goal_position

    def assign_next_goal(self, position: np.ndarray, assignment_step: int) -> RandomGoalTask:
        goal = self.atlas.sample_goal(self.rng, position, self.minimum_goal_separation)
        task = RandomGoalTask(
            f"random-goal-{self._next_task_id}",
            goal,
            self.task_reward,
            int(assignment_step),
        )
        self._next_task_id += 1
        self.current_task = task
        self.goal_sequence.append(goal.copy())
        return task

    def interrupt_for_charge(self) -> None:
        if self.current_task is None or self.current_task.interrupted_by_charge:
            return
        self.current_task.interrupted_by_charge = True
        self.task_interruption_count += 1

    def mark_station_arrival(self) -> None:
        return None

    def resume_from_station(self) -> None:
        if self.current_task is None:
            raise RuntimeError("cannot resume without a pending random goal")
        self.task_resume_count += 1

    def advance(self, position: np.ndarray, step: int) -> dict[str, Any]:
        events = {
            "task_completed": False,
            "task_assigned": False,
            "completed_task_id": None,
            "new_goal_id": None,
        }
        task = self.current_task
        point = as_vec3(position, "position")
        if task is None or float(np.linalg.norm(point - task.goal_position)) > self.goal_radius:
            return events
        task.completion_step = int(step)
        self.completed_tasks.append(task)
        self.tasks_completed += 1
        events["task_completed"] = True
        events["completed_task_id"] = task.task_id
        next_task = self.assign_next_goal(point, step)
        events["task_assigned"] = True
        events["new_goal_id"] = next_task.task_id
        return events


class RandomPersistentTaskWrapper(gym.Wrapper):
    """Task-independent persistent goal stream over the certified atlas."""

    multi_step_mission = False
    persistent_goal_stream = True
    task_edge_dependency = False
    task_waypoint_dependency = False

    def __init__(
        self,
        plant: CertifiedSingleUAVPlantEnv,
        atlas: CertifiedRecoverabilityAtlas | None = None,
        reward_config: PersistentRewardConfig | None = None,
        goal_radius: float = 0.20,
        minimum_goal_separation: float = 0.60,
        task_reward: float = 10.0,
        battery_capacity: float = 30.0,
    ) -> None:
        super().__init__(plant)
        self.plant = plant
        self.atlas = atlas
        self._manager_config = (float(goal_radius), float(minimum_goal_separation), float(task_reward))
        self.manager = None if atlas is None else RandomPersistentGoalManager(atlas, *self._manager_config)
        self.reward_config = reward_config or PersistentRewardConfig(task_completion_reward=task_reward)
        self.battery_capacity = float(battery_capacity)
        self.mode = PersistentMissionMode.TASK_RL
        self.phase = self.mode
        self.required_return_energy = 0.0
        self.energy_margin = 0.0
        self.episode_step = 0
        self.time_since_last_charge = 0
        self.voluntary_station_approach = False
        self.episode_seed: int | None = None
        self.sampled_start = plant.state.copy()
        self.observation_layout: dict[str, slice] = {}
        cursor = 0
        for name, length in (
            ("position", 3),
            ("velocity", 3),
            ("energy", 1),
            ("goal_delta", 3),
            ("station_delta", 3),
            ("required_return_energy", 1),
            ("energy_margin", 1),
            ("mission_mode", len(PersistentMissionMode)),
            ("charging", 1),
            ("state_of_charge", 1),
            ("tasks_completed", 1),
            ("distance_to_goal", 1),
            ("distance_to_station", 1),
            ("time_since_last_charge", 1),
            ("lidar_distances", plant.config.num_lasers),
            ("lidar_valid", plant.config.num_lasers),
            ("local_map_crop", plant.config.local_map_encoding_size),
            ("recovery_corridor", plant.config.corridor_encoding_size),
        ):
            self.observation_layout[name] = slice(cursor, cursor + length)
            cursor += length
        self.observation_space = gym.spaces.Box(
            np.full(cursor, -2.0, dtype=np.float32),
            np.full(cursor, 2.0, dtype=np.float32),
            dtype=np.float32,
        )

    def attach_atlas(self, atlas: CertifiedRecoverabilityAtlas) -> None:
        if self.atlas is not None and self.atlas is not atlas:
            raise RuntimeError("random persistent task wrapper already has a recovery atlas")
        self.atlas = atlas
        if self.manager is None:
            self.manager = RandomPersistentGoalManager(atlas, *self._manager_config)

    def _require_components(self) -> tuple[CertifiedRecoverabilityAtlas, RandomPersistentGoalManager]:
        if self.atlas is None or self.manager is None:
            raise RuntimeError("random persistent recovery atlas is not attached")
        return self.atlas, self.manager

    @property
    def active_goal(self) -> np.ndarray:
        _, manager = self._require_components()
        if self.mode == PersistentMissionMode.BACKUP_RECOVERY:
            return self.plant.scenario.station_position
        return manager.navigation_target

    def set_certificate_quantities(self, required_return_energy: float, energy_margin: float) -> None:
        self.required_return_energy = float(required_return_energy)
        self.energy_margin = float(energy_margin)

    def set_time_since_last_charge(self, steps: int) -> None:
        self.time_since_last_charge = max(0, int(steps))

    def on_runtime_recovery(self, reason: str) -> None:
        self.begin_backup_recovery(reason)

    def begin_backup_recovery(self, reason: str) -> None:
        del reason
        if self.mode != PersistentMissionMode.BACKUP_RECOVERY:
            self.manager.interrupt_for_charge()
        self.mode = PersistentMissionMode.BACKUP_RECOVERY
        self.phase = self.mode

    def enter_charging(self, *, voluntary: bool) -> None:
        self.manager.mark_station_arrival()
        if voluntary:
            self.manager.interrupt_for_charge()
        self.voluntary_station_approach = voluntary
        self.mode = PersistentMissionMode.CHARGING_RL
        self.phase = self.mode
        self.time_since_last_charge = 0

    def leave_station(self) -> None:
        self.manager.resume_from_station()
        self.mode = PersistentMissionMode.TASK_RL
        self.phase = self.mode
        self.voluntary_station_approach = False

    def build_observation(
        self,
        local_map_crop_encoding: np.ndarray | None = None,
        corridor_encoding: np.ndarray | None = None,
    ) -> np.ndarray:
        _, manager = self._require_components()
        lidar = self.plant.last_lidar
        task = manager.current_task
        if lidar is None or task is None:
            raise RuntimeError("random persistent observation is unavailable")
        local_map = np.zeros(self.plant.config.local_map_encoding_size) if local_map_crop_encoding is None else np.asarray(local_map_crop_encoding)
        recovery_corridor = np.zeros(self.plant.config.corridor_encoding_size) if corridor_encoding is None else np.asarray(corridor_encoding)
        state = self.plant.state
        mode = np.eye(len(PersistentMissionMode), dtype=np.float64)[int(self.mode)]
        scale = float(np.linalg.norm(self.plant.config.world_size))
        observation = np.concatenate((
            state.position / self.plant.config.world_size,
            state.velocity / self.plant.config.v_max,
            np.array([state.energy / self.battery_capacity]),
            (task.goal_position - state.position) / self.plant.config.world_size,
            (self.plant.scenario.station_position - state.position) / self.plant.config.world_size,
            np.array([self.required_return_energy / self.battery_capacity, self.energy_margin / self.battery_capacity]),
            mode,
            np.array([
                float(self.mode == PersistentMissionMode.CHARGING_RL),
                state.energy / self.battery_capacity,
                manager.tasks_completed / 100.0,
                np.linalg.norm(task.goal_position - state.position) / scale,
                np.linalg.norm(self.plant.scenario.station_position - state.position) / scale,
                self.time_since_last_charge / max(1, self.plant.config.episode_limit),
            ]),
            lidar.distances / self.plant.config.lidar_range,
            lidar.valid.astype(np.float64),
            local_map,
            recovery_corridor,
        ))
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        atlas, manager = self._require_components()
        _, info = self.plant.reset(seed=seed, options=options)
        self.episode_seed = seed
        self.plant.state = atlas.sample_initial_state(seed, self.battery_capacity)
        self.sampled_start = self.plant.state.copy()
        self.plant.last_lidar = self.plant.lidar_model.measure(self.plant.state, self.plant.world, self.plant.np_random)
        manager.reset(seed, self.plant.state.position)
        self.mode = PersistentMissionMode.TASK_RL
        self.phase = self.mode
        self.episode_step = 0
        self.time_since_last_charge = 0
        self.voluntary_station_approach = False
        return self.build_observation(), info | {
            "episode_seed": seed,
            "sampled_start": self.sampled_start.copy(),
            "sampled_goal": manager.current_task.goal_position.copy(),
            "recovery_atlas_hash": atlas.atlas_hash,
            "observation_layout": dict(self.observation_layout),
        }

    def step(self, action):
        atlas, manager = self._require_components()
        task_before = manager.current_task
        if task_before is None:
            raise RuntimeError("random persistent task is unavailable")
        goal_before = task_before.goal_position.copy()
        distance_before = float(np.linalg.norm(self.plant.state.position - goal_before))
        _, _, terminated, truncated, info = self.plant.step(action)
        telemetry = info["telemetry"]
        self.episode_step += 1
        events = {
            "task_completed": False,
            "task_assigned": False,
            "completed_task_id": None,
            "new_goal_id": None,
        }
        if self.mode == PersistentMissionMode.TASK_RL:
            events = manager.advance(self.plant.state.position, self.episode_step)
        elif self.mode == PersistentMissionMode.BACKUP_RECOVERY and telemetry.terminal_admissible:
            self.enter_charging(voluntary=False)
        if terminated or info.get("failure_reason"):
            self.mode = PersistentMissionMode.FAILURE
            self.phase = self.mode
        distance_after = float(np.linalg.norm(self.plant.state.position - goal_before))
        reward = (
            self.reward_config.goal_progress_weight * (distance_before - distance_after)
            + self.reward_config.task_completion_reward * float(events["task_completed"])
            - self.reward_config.elapsed_time_cost
            - self.reward_config.flight_energy_cost * telemetry.energy_cost
        )
        task = manager.current_task
        return self.build_observation(), reward, terminated, truncated, info | {
            "persistent_mode": self.mode.name,
            "task_id": None if task is None else task.task_id,
            "current_goal_id": None if task is None else task.task_id,
            "current_goal": None if task is None else task.goal_position.copy(),
            "task_completed_now": events["task_completed"],
            "task_assigned_now": events["task_assigned"],
            "completed_task_id": events["completed_task_id"],
            "tasks_completed": manager.tasks_completed,
            "episode_step": self.episode_step,
            "recovery_atlas_hash": atlas.atlas_hash,
        }
