from __future__ import annotations

from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np

from .actuator import ActionTrace, ActuatorTrackingModel, validate_action_box
from .config import CertifiedUAVConfig
from .dynamics import integrate_double_integrator
from .energy import EnergyModel
from .lidar import HorizontalLidarModel, LidarPacket
from .scenario import ScenarioDefinition
from .state import UAVPhysicalState
from .telemetry import CalibrationLogRecord, CalibrationRecordLogger, StepTelemetry, tuple3


class CertifiedSingleUAVPlantEnv(gym.Env[np.ndarray, np.ndarray]):
    """Physical simulator that accepts only the final published acceleration."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: CertifiedUAVConfig,
        scenario: ScenarioDefinition,
        actuator_model: ActuatorTrackingModel | None = None,
        energy_model: EnergyModel | None = None,
        lidar_model: HorizontalLidarModel | None = None,
        calibration_versions: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        if not np.allclose(config.world_size, scenario.world_size):
            raise ValueError("configuration and scenario world dimensions differ")
        self.config = config
        self.scenario = scenario
        self.world = scenario.world
        self.terminal = scenario.terminal
        self.scenario_consistency_failures = scenario.consistency_failures(config)
        self.actuator_model = actuator_model or ActuatorTrackingModel(config.tracking_error_bound)
        self.energy_model = energy_model or EnergyModel()
        self.lidar_model = lidar_model or HorizontalLidarModel(
            config.num_lasers,
            config.lidar_range,
            "synthetic-sensor-v1",
            config.lidar_range_noise,
            config.lidar_pose_noise,
            config.lidar_heading_noise,
            config.lidar_invalid_probability,
        )
        self.calibration_versions = {
            "sensor": self.lidar_model.sensor_version,
            "dynamics": "synthetic-dynamics-v1",
            "tracking": "synthetic-tracking-v1",
            "energy": "synthetic-energy-v1",
            "terminal": self.terminal.version,
        } | (calibration_versions or {})
        self.action_space = gym.spaces.Box(-config.a_max.astype(np.float32), config.a_max.astype(np.float32), dtype=np.float32)
        observation_low = np.concatenate((np.zeros(3), -config.v_max, np.array([-config.initial_energy]), np.zeros(config.num_lasers), np.zeros(2 * config.num_lasers)))
        observation_high = np.concatenate((config.world_size, config.v_max, np.array([config.initial_energy]), np.full(config.num_lasers, config.lidar_range), np.ones(2 * config.num_lasers)))
        self.observation_space = gym.spaces.Box(observation_low.astype(np.float32), observation_high.astype(np.float32), dtype=np.float32)
        self.state = scenario.initial_state.copy()
        self.last_lidar: LidarPacket | None = None
        self.last_telemetry: StepTelemetry | None = None
        self.failure_reason: str | None = None
        self.step_count = 0
        self.calibration_logger = CalibrationRecordLogger()

    def _observation(self, lidar: LidarPacket) -> np.ndarray:
        return np.concatenate(
            (
                self.state.position,
                self.state.velocity,
                np.array([self.state.energy]),
                lidar.distances,
                lidar.valid.astype(np.float64),
                lidar.hit.astype(np.float64),
            )
        ).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options
        self.state = self.scenario.initial_state.copy()
        self.step_count = 0
        self.failure_reason = None
        self.last_telemetry = None
        self.last_lidar = self.lidar_model.measure(self.state, self.world, self.np_random)
        return self._observation(self.last_lidar), {"lidar_packet": self.last_lidar, "physical_state": self.state.copy()}

    def step(self, executed_action: np.ndarray):
        state_before = self.state.copy()
        published = validate_action_box(executed_action, self.config.a_max)
        measured = self.actuator_model.apply(published, state_before, self.np_random)
        position_next, velocity_next = integrate_double_integrator(
            state_before.position,
            state_before.velocity,
            measured,
            self.config.dt,
        )
        collision = self.world.swept_collision(state_before.position, position_next, self.config.body_radius)
        velocity_violation = bool(np.any(np.abs(velocity_next) > self.config.v_max + 1e-12))
        energy_cost = self.energy_model.realized_cost(state_before, measured, self.config.dt)
        energy_next = state_before.energy - energy_cost
        self.state = UAVPhysicalState(position_next, velocity_next, energy_next, state_before.timestamp + self.config.dt)
        self.last_lidar = self.lidar_model.measure(self.state, self.world, self.np_random)
        terminal_admissible = self.terminal.is_admissible(self.state)
        self.step_count += 1
        if collision:
            self.failure_reason = "collision"
        elif energy_next <= 0.0:
            self.failure_reason = "energy_depleted"
        elif velocity_violation:
            self.failure_reason = "velocity_limit_exceeded"
        else:
            self.failure_reason = None
        terminated = collision or energy_next <= 0.0 or velocity_violation or terminal_admissible
        truncated = self.step_count >= self.config.episode_limit and not terminated
        plant_trace = ActionTrace(None, None, published, published, measured, False, "plant-only", "")
        self.last_telemetry = StepTelemetry(
            state_before,
            self.state.copy(),
            plant_trace,
            energy_cost,
            collision,
            terminal_admissible,
            self.last_lidar,
            None,
            None,
            None,
        )
        self._record_calibration(self.last_telemetry)
        info = {
            "telemetry": self.last_telemetry,
            "failure_reason": self.failure_reason,
            "physical_state": self.state.copy(),
            "lidar_packet": self.last_lidar,
            "velocity_limit_exceeded": velocity_violation,
        }
        return self._observation(self.last_lidar), 0.0, terminated, truncated, info

    def attach_runtime_trace(
        self,
        action_trace: ActionTrace,
        certificate_version: str | None,
        geometry_version: str | None,
        corridor_version: str | None,
    ) -> StepTelemetry:
        if self.last_telemetry is None:
            raise RuntimeError("no plant transition is available")
        self.last_telemetry = replace(
            self.last_telemetry,
            action_trace=action_trace,
            certificate_version=certificate_version,
            geometry_version=geometry_version,
            corridor_version=corridor_version,
        )
        if self.calibration_logger.records:
            commanded = action_trace.candidate if action_trace.candidate is not None else action_trace.fallback
            self.calibration_logger.records[-1] = replace(
                self.calibration_logger.records[-1],
                commanded_action=tuple3(commanded),
                published_action=tuple3(action_trace.published),
                measured_action=tuple3(action_trace.measured),
            )
        return self.last_telemetry

    def _record_calibration(self, telemetry: StepTelemetry) -> None:
        trace = telemetry.action_trace
        self.calibration_logger.append(
            CalibrationLogRecord(
                telemetry.state_before.timestamp,
                tuple3(telemetry.state_before.position),
                tuple3(telemetry.state_before.velocity),
                None if trace.nominal is None else tuple3(trace.nominal),
                tuple3(trace.published),
                tuple3(trace.measured),
                tuple3(telemetry.state_after.position),
                tuple3(telemetry.state_after.velocity),
                tuple(float(value) for value in telemetry.lidar_packet.distances),
                tuple(bool(value) for value in telemetry.lidar_packet.valid),
                tuple(bool(value) for value in telemetry.lidar_packet.hit),
                None,
                None,
                None,
                telemetry.state_before.energy,
                telemetry.state_after.energy,
                self.calibration_versions["sensor"],
                self.calibration_versions["dynamics"],
                self.calibration_versions["tracking"],
                self.calibration_versions["energy"],
                self.calibration_versions["terminal"],
            )
        )

    def export_calibration_record(self) -> tuple[dict[str, Any], ...]:
        return self.calibration_logger.export()

    def synthetic_bootstrap_lidar_packets(self) -> tuple[LidarPacket, ...]:
        """Plant-generated sensor replay for deterministic synthetic fixtures.

        The caller receives only packets; this does not expose the obstacle list.
        """

        packets = []
        for pose in self.scenario.bootstrap_lidar_poses:
            synthetic_state = UAVPhysicalState(pose, np.zeros(3), self.state.energy, self.state.timestamp)
            packets.append(self.lidar_model.measure(synthetic_state, self.world, self.np_random))
        return tuple(packets)
