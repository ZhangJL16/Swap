from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, IntEnum
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
    return_success_reward: float = 10.0
    energy_cost_weight: float = 0.05


class MissionPhase(IntEnum):
    OUTBOUND = 0
    RETURN = 1
    SUCCESS = 2
    FAILURE = 3


class MissionTerminationReason(str, Enum):
    TASK_AND_RETURN_SUCCESS = "TASK_AND_RETURN_SUCCESS"
    COLLISION = "COLLISION"
    ENERGY_DEPLETED = "ENERGY_DEPLETED"
    VELOCITY_LIMIT = "VELOCITY_LIMIT"
    PREMATURE_TERMINAL = "PREMATURE_TERMINAL"
    CORRIDOR_EXIT = "CORRIDOR_EXIT"
    RECOVERY_CERTIFICATE_INVALID = "RECOVERY_CERTIFICATE_INVALID"
    CERTIFICATE_EXPIRED = "CERTIFICATE_EXPIRED"
    DEADLINE = "DEADLINE"
    EPISODE_TIMEOUT = "EPISODE_TIMEOUT"
    OTHER_FAILURE = "OTHER_FAILURE"


@dataclass(frozen=True)
class RewardBreakdown:
    progress_reward: float
    task_completion_reward: float
    return_success_reward: float
    step_penalty: float
    energy_penalty: float
    collision_penalty: float

    @property
    def total(self) -> float:
        return float(
            self.progress_reward
            + self.task_completion_reward
            + self.return_success_reward
            - self.step_penalty
            - self.energy_penalty
            - self.collision_penalty
        )


class CertifiedTaskWrapper(gym.Wrapper):
    """Task semantics only; rewards and features are not certificate evidence."""

    def __init__(self, plant: CertifiedSingleUAVPlantEnv, reward_config: TaskRewardConfig | None = None) -> None:
        super().__init__(plant)
        self.plant = plant
        self.reward_config = TaskRewardConfig() if reward_config is None else reward_config
        self.goal = plant.scenario.task_goal.copy()
        self.multi_step_mission = bool(plant.scenario.mission_config.get("enabled", False))
        if self.multi_step_mission:
            self.reward_config = replace(
                self.reward_config,
                goal_radius=float(plant.scenario.mission_config.get("goal_radius", self.reward_config.goal_radius)),
                progress_weight=float(plant.scenario.mission_config.get("progress_weight", self.reward_config.progress_weight)),
                goal_reward=float(plant.scenario.mission_config.get("task_reward", self.reward_config.goal_reward)),
                return_success_reward=float(plant.scenario.mission_config.get("return_reward", self.reward_config.return_success_reward)),
                energy_cost_weight=float(plant.scenario.mission_config.get("energy_cost_weight", self.reward_config.energy_cost_weight)),
            )
        self.phase = MissionPhase.OUTBOUND
        self.episode_step = 0
        self.task_completed = False
        self.return_triggered = False
        self.termination_reason: MissionTerminationReason | None = None
        self.last_reward_breakdown = RewardBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
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
            ("mission_phase", 4),
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
        self.phase = MissionPhase.OUTBOUND
        self.episode_step = 0
        self.task_completed = False
        self.return_triggered = False
        self.termination_reason = None

    @property
    def active_goal(self) -> np.ndarray:
        if self.multi_step_mission and self.phase == MissionPhase.RETURN:
            return self.plant.scenario.station_position
        return self.goal

    def reward(
        self,
        state_before: UAVPhysicalState,
        state_after: UAVPhysicalState,
        collision: bool,
        terminal: bool,
        energy_cost: float = 0.0,
        task_completed_now: bool = False,
        target: np.ndarray | None = None,
    ) -> float:
        target = self.active_goal if target is None else target
        previous_distance = float(np.linalg.norm(state_before.position - target))
        current_distance = float(np.linalg.norm(state_after.position - target))
        breakdown = RewardBreakdown(
            self.reward_config.progress_weight * (previous_distance - current_distance),
            self.reward_config.goal_reward * float(task_completed_now),
            self.reward_config.return_success_reward * float(terminal and self.task_completed),
            self.reward_config.step_cost,
            self.reward_config.energy_cost_weight * energy_cost,
            self.reward_config.collision_cost * float(collision),
        )
        self.last_reward_breakdown = breakdown
        return breakdown.total

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
                (self.active_goal - state.position) / config.world_size,
                (self.plant.scenario.station_position - state.position) / config.world_size,
                lidar.distances / config.lidar_range,
                lidar.valid.astype(np.float64),
                local_map,
                corridor,
                np.eye(4, dtype=np.float64)[int(self.phase)],
            )
        )
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = self.plant.reset(seed=seed, options=options)
        self.reset_task(self.plant.scenario)
        return self.build_observation(), info | {"observation_layout": dict(self.observation_layout)}

    def step(self, action):
        reward_target = self.active_goal.copy()
        _, _, terminated, truncated, info = self.plant.step(action)
        telemetry = info["telemetry"]
        self.episode_step += 1
        reached_task = float(np.linalg.norm(self.plant.state.position - self.goal)) <= self.reward_config.goal_radius
        task_completed_now = bool(self.multi_step_mission and self.phase == MissionPhase.OUTBOUND and reached_task)
        if task_completed_now:
            self.task_completed = True
            self.return_triggered = True
            self.phase = MissionPhase.RETURN
        terminal_return = bool(
            telemetry.terminal_admissible
            and (not self.multi_step_mission or self.phase == MissionPhase.RETURN)
        )
        failure = bool(telemetry.collision or info.get("failure_reason") in {"energy_depleted", "velocity_limit_exceeded"})
        if terminal_return:
            self.phase = MissionPhase.SUCCESS
        elif failure or truncated or (terminated and not terminal_return):
            self.phase = MissionPhase.FAILURE
        termination_reason = None
        if terminal_return and self.task_completed:
            termination_reason = MissionTerminationReason.TASK_AND_RETURN_SUCCESS
        elif terminal_return:
            termination_reason = MissionTerminationReason.PREMATURE_TERMINAL
        elif telemetry.collision:
            termination_reason = MissionTerminationReason.COLLISION
        elif info.get("failure_reason") == "energy_depleted":
            termination_reason = MissionTerminationReason.ENERGY_DEPLETED
        elif info.get("failure_reason") == "velocity_limit_exceeded":
            termination_reason = MissionTerminationReason.VELOCITY_LIMIT
        elif terminated and telemetry.terminal_admissible and not self.task_completed:
            termination_reason = MissionTerminationReason.PREMATURE_TERMINAL
        elif truncated:
            termination_reason = MissionTerminationReason.EPISODE_TIMEOUT
        elif terminated:
            termination_reason = MissionTerminationReason.OTHER_FAILURE
        if terminated or truncated:
            self.termination_reason = termination_reason or MissionTerminationReason.OTHER_FAILURE
        reward = self.reward(
            telemetry.state_before,
            telemetry.state_after,
            telemetry.collision,
            terminal_return,
            telemetry.energy_cost,
            task_completed_now,
            reward_target,
        )
        info = info | {
            "task_goal_reached": reached_task,
            "task_completed": self.task_completed,
            "task_completed_now": task_completed_now,
            "return_triggered": self.return_triggered,
            "terminal_return_success": terminal_return,
            "mission_phase": self.phase.name,
            "episode_step": self.episode_step,
            "mission_termination_reason": None if self.termination_reason is None else self.termination_reason.value,
            "reward_components": self.last_reward_breakdown.__dict__.copy() | {"total_reward": reward},
        }
        return self.build_observation(), reward, terminated, truncated, info
