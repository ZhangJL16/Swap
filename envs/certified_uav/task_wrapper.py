from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

from .plant_env import CertifiedSingleUAVPlantEnv
from .state import UAVPhysicalState


@dataclass(frozen=True)
class TaskRewardConfig:
    progress_weight: float = 1.0
    goal_reward: float = 5.0
    step_cost: float = 0.01
    collision_cost: float = 10.0
    goal_radius: float = 0.08


class CertifiedTaskWrapper(gym.Wrapper):
    """Task semantics only; rewards and features are not certificate evidence."""

    def __init__(self, plant: CertifiedSingleUAVPlantEnv, reward_config: TaskRewardConfig | None = None) -> None:
        super().__init__(plant)
        self.plant = plant
        self.reward_config = TaskRewardConfig() if reward_config is None else reward_config
        self.goal = plant.scenario.task_goal.copy()
        config = plant.config
        self.observation_layout: dict[str, slice] = {}
        cursor = 0
        for name, length in (
            ("position", 3),
            ("velocity", 3),
            ("energy", 1),
            ("goal_delta", 3),
            ("station_delta", 3),
            ("lidar_distances", config.num_lasers),
            ("lidar_valid", config.num_lasers),
            ("local_map_crop", config.local_map_encoding_size),
            ("corridor", config.corridor_encoding_size),
        ):
            self.observation_layout[name] = slice(cursor, cursor + length)
            cursor += length
        low = np.full(cursor, -2.0, dtype=np.float32)
        high = np.full(cursor, 2.0, dtype=np.float32)
        for name in ("energy", "lidar_distances", "lidar_valid"):
            low[self.observation_layout[name]] = 0.0
            high[self.observation_layout[name]] = 1.0
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset_task(self, scenario) -> None:
        self.goal = scenario.task_goal.copy()

    def reward(
        self,
        state_before: UAVPhysicalState,
        state_after: UAVPhysicalState,
        collision: bool,
        terminal: bool,
    ) -> float:
        del terminal
        previous_distance = float(np.linalg.norm(state_before.position - self.goal))
        current_distance = float(np.linalg.norm(state_after.position - self.goal))
        reached_goal = current_distance <= self.reward_config.goal_radius
        return (
            self.reward_config.progress_weight * (previous_distance - current_distance)
            + self.reward_config.goal_reward * float(reached_goal)
            - self.reward_config.step_cost
            - self.reward_config.collision_cost * float(collision)
        )

    def build_observation(
        self,
        local_map_crop_encoding: np.ndarray | None = None,
        corridor_encoding: np.ndarray | None = None,
    ) -> np.ndarray:
        lidar = self.plant.last_lidar
        if lidar is None:
            raise RuntimeError("LiDAR packet is unavailable")
        config = self.plant.config
        local_map = np.zeros(config.local_map_encoding_size) if local_map_crop_encoding is None else np.asarray(local_map_crop_encoding, dtype=np.float64)
        corridor = np.zeros(config.corridor_encoding_size) if corridor_encoding is None else np.asarray(corridor_encoding, dtype=np.float64)
        if local_map.shape != (config.local_map_encoding_size,) or corridor.shape != (config.corridor_encoding_size,):
            raise ValueError("task map or corridor encoding has the wrong shape")
        state = self.plant.state
        observation = np.concatenate(
            (
                state.position / config.world_size,
                state.velocity / config.v_max,
                np.array([state.energy / config.initial_energy]),
                (self.goal - state.position) / config.world_size,
                (self.plant.scenario.station_position - state.position) / config.world_size,
                lidar.distances / config.lidar_range,
                lidar.valid.astype(np.float64),
                local_map,
                corridor,
            )
        )
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = self.plant.reset(seed=seed, options=options)
        self.reset_task(self.plant.scenario)
        return self.build_observation(), info | {"observation_layout": dict(self.observation_layout)}

    def step(self, action):
        _, _, terminated, truncated, info = self.plant.step(action)
        telemetry = info["telemetry"]
        reward = self.reward(
            telemetry.state_before,
            telemetry.state_after,
            telemetry.collision,
            telemetry.terminal_admissible,
        )
        task_goal_reached = float(np.linalg.norm(self.plant.state.position - self.goal)) <= self.reward_config.goal_radius
        info = info | {"task_goal_reached": task_goal_reached}
        return self.build_observation(), reward, terminated, truncated, info
