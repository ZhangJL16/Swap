from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import monotonic
from typing import Any, Sequence

import gymnasium as gym
import numpy as np

from calibration.confidence import BoundEstimate
from calibration.dynamics import DynamicsSample, build_dynamics_contract
from calibration.energy import EnergySample, build_energy_contract
from calibration.schema import ConfidenceSemantics, DataSplit, RawCalibrationRecord
from calibration.sensor import build_sensor_contract
from calibration.synthetic import build_synthetic_calibration_bundle, synthetic_metadata
from calibration.terminal import build_terminal_contract
from calibration.tracking import TrackingSample, build_tracking_contract
from cert_runtime import (
    AABB2,
    AtomicCommandPublisher,
    CalibrationBundle,
    CertificateConfig,
    CertificateReplay,
    CertificateState,
    CorridorCell,
    CorridorRecoveryVerifier,
    FrozenRecoveryPolicy,
    Interval,
    Interval3,
    RecoveryConfig,
    RecoveryEnergySolver,
    ReturnCorridor,
    RollingLocalGeometry,
    RuntimeCertifier,
    SensorBounds,
    SimulatedWatchdog,
    SingleCorridorClosurePipeline,
    StateCellBounds,
    SuccessorEnvelopeBuilder,
    WCETContract,
    ZonotopeConstructor,
)
from cert_runtime.recovery import RecoveryDecision
from cert_runtime.trainer import CertificateEpoch
from cert_runtime.watchdog import PublishedCommand

from .actuator import ActionTrace
from .config import CertifiedUAVConfig
from .dynamics import integrate_double_integrator
from .plant_env import CertifiedSingleUAVPlantEnv
from .mission_certificate import SyntheticMissionCertificateProvider
from .scenario import ScenarioDefinition
from .task_wrapper import CertifiedTaskWrapper


class ProvidedActor:
    def __init__(self) -> None:
        self.output = np.zeros(3, dtype=np.float64)
        self.calls = 0

    def set_output(self, output: Sequence[float]) -> None:
        array = np.asarray(output, dtype=np.float64)
        if array.shape != (3,):
            raise ValueError("actor output must have shape (3,)")
        self.output = array.copy()

    def sample_u(self, observation: Sequence[float]) -> tuple[float, float, float]:
        del observation
        self.calls += 1
        return tuple(float(value) for value in self.output)


@dataclass(frozen=True)
class RuntimeCyclePreparation:
    state: CertificateState
    task_observation: np.ndarray
    closure_result: Any
    recovery: Any
    failure_reason: str | None


def _operating_point(config: CertifiedUAVConfig, state) -> dict[str, float | str]:
    return {
        "speed": float(np.linalg.norm(state.velocity)),
        "acceleration": float(np.linalg.norm(config.a_max)),
        "payload": 0.5,
        "temperature": 20.0,
        "voltage": 20.0,
        "flight_mode": "hover",
    }


def _build_synthetic_calibration(config: CertifiedUAVConfig, scenario: ScenarioDefinition):
    contracts, reports = build_synthetic_calibration_bundle()
    _, _, _, energy, _ = contracts
    mission_bounds = scenario.mission_config.get("certificate_bounds", {})
    point = tuple(sorted(_operating_point(config, scenario.initial_state).items()))
    sensor_metadata = synthetic_metadata("cert-uav-sensor-evidence", "synthetic-sensor-v2")
    sensor_records = tuple(
        RawCalibrationRecord(
            f"{channel}-{index}",
            float(index),
            channel,
            residual,
            0.0,
            split,
            sensor_metadata.device_version,
            sensor_metadata.firmware_version,
            point,
        )
        for channel, residual in (("position", 0.001), ("attitude", 0.001), ("range", 0.002), ("time_sync", 0.001))
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
    )
    estimates = {
        channel: BoundEstimate(value, ConfidenceSemantics.SIMULTANEOUS_CONFIDENCE, 0.01, ("synthetic",), False)
        for channel, value in (("position", 0.002), ("attitude", 0.002), ("range", 0.004), ("time_sync", 0.002))
    }
    sensor, sensor_report = build_sensor_contract(
        sensor_records,
        sensor_metadata,
        "synthetic-sensor-v2",
        estimates,
        beam_half_angle_radians=0.5,
        footprint_radius=config.body_radius,
        map_discretization_error=config.grid_resolution,
        maximum_range=max(config.lidar_range + 0.1, 6.1),
        maximum_speed=float(np.max(config.v_max)),
        evidence_max_age_seconds=10.0,
        minimum_free_observations=1,
    )
    tracking_metadata = synthetic_metadata("cert-uav-tracking-evidence", "synthetic-tracking-v2")
    tracking_samples = tuple(
        TrackingSample(
            f"tracking-{index}",
            float(index),
            float(index) + 0.001,
            float(index) + 0.002,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            tuple(float(value) for value in config.tracking_error_bound * 0.5),
            split,
            point,
        )
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
    )
    tracking, tracking_report = build_tracking_contract(
        tracking_samples,
        tracking_metadata,
        "synthetic-tracking-v2",
        tuple(float(value) for value in config.tracking_error_bound),
        0.005,
    )
    dynamics_metadata = synthetic_metadata("cert-uav-dynamics-evidence", "synthetic-dynamics-v2")
    sample_position = np.array([0.8, 0.8, 1.0])
    sample_velocity = np.array([-0.1, 0.0, 0.0])
    sample_action = np.array([0.02, 0.0, 0.0])
    next_position, next_velocity = integrate_double_integrator(
        sample_position,
        sample_velocity,
        sample_action,
        config.dt,
    )
    dynamics_samples = tuple(
        DynamicsSample(
            f"dynamics-{index}",
            float(index),
            float(index) + config.dt,
            tuple(sample_position),
            tuple(sample_velocity),
            tuple(next_position),
            tuple(next_velocity),
            tuple(sample_action),
            tuple(sample_action),
            tuple(sample_action),
            split,
            point,
        )
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
    )
    dynamics, dynamics_report = build_dynamics_contract(
        dynamics_samples,
        dynamics_metadata,
        "synthetic-dynamics-v2",
        tracking,
        initial_position_radius=(0.001, 0.001, 0.001),
        initial_velocity_radius=(0.001, 0.001, 0.001),
        control_period=config.dt,
        control_period_error=float(mission_bounds.get("control_period_error", 0.001)),
        sensor_latency_upper=config.total_latency / 3.0,
        compute_latency_upper=config.total_latency / 3.0,
        switch_latency_upper=config.total_latency / 3.0,
        position_residual_radius=tuple(mission_bounds.get("position_residual_radius", (0.0002, 0.0002, 0.0002))),
        velocity_residual_radius=tuple(mission_bounds.get("velocity_residual_radius", (0.0002, 0.0002, 0.0002))),
        wind_acceleration_radius=tuple(mission_bounds.get("wind_acceleration_radius", (0.001, 0.001, 0.001))),
    )
    if scenario.mission_config.get("enabled", False):
        energy_metadata = synthetic_metadata(
            f"{scenario.name}-energy-evidence",
            "synthetic-mission-energy-v2",
        )
        energy_samples = tuple(
            EnergySample(
                f"mission-energy-{index}",
                float(index),
                float(index) + config.dt,
                20.0,
                20.0,
                0.1,
                0.1,
                0.013,
                (0.1, 0.1, 0.0),
                (0.05, 0.05, 0.0),
                True,
                1.0,
                20.0,
                0.5,
                split,
            )
            for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
        )
        energy, energy_report = build_energy_contract(
            energy_samples,
            energy_metadata,
            "synthetic-mission-energy-v2",
            avionics_cost=0.006,
            hover_cost=0.004,
            velocity_coefficients=(0.001, 0.001, 0.001),
            action_coefficients=(0.001, 0.001, 0.0015),
            communication_cost=0.001,
            computation_cost=0.001,
            measurement_error=0.0005,
            underestimation_margin=0.001,
        )
        reports = reports + (energy_report,)
    terminal_metadata = synthetic_metadata("cert-uav-terminal-evidence", scenario.terminal.version)
    terminal = build_terminal_contract(
        terminal_metadata,
        scenario.terminal.version,
        horizontal_position=(
            float(scenario.terminal.position_low[0]),
            float(scenario.terminal.position_low[1]),
            float(scenario.terminal.position_high[0]),
            float(scenario.terminal.position_high[1]),
        ),
        altitude=(float(scenario.terminal.position_low[2]), float(scenario.terminal.position_high[2])),
        velocity_low=tuple(float(-value) for value in scenario.terminal.velocity_abs_max),
        velocity_high=tuple(float(value) for value in scenario.terminal.velocity_abs_max),
        minimum_energy=scenario.terminal.minimum_energy,
        continuation_evidence=tuple((mode, f"synthetic-{mode}-evidence") for mode in scenario.terminal.continuation_modes),
    )
    return CalibrationBundle(sensor, dynamics, tracking, energy, terminal), reports + (sensor_report, tracking_report, dynamics_report)


class CertifiedRuntimeWrapper(gym.Env[np.ndarray, np.ndarray]):
    """Certificate-first execution wrapper around the isolated plant/task layers.

    ``wall_clock`` is the fail-closed runtime default. ``functional`` preserves
    certificate semantics while removing host scheduling from deterministic
    algorithm tests; it is not deployment timing evidence.
    """

    def __init__(
        self,
        task_env: CertifiedTaskWrapper,
        *,
        allow_synthetic_certificates: bool = True,
        freeze_certificate_epoch: bool = False,
        generator_center_mode: str = "task_oriented",
        timing_mode: str = "wall_clock",
    ) -> None:
        super().__init__()
        self.task_env = task_env
        self.plant: CertifiedSingleUAVPlantEnv = task_env.plant
        self.config = self.plant.config
        self.scenario = self.plant.scenario
        self.allow_synthetic_certificates = allow_synthetic_certificates
        self.freeze_certificate_epoch = freeze_certificate_epoch
        if timing_mode not in {"wall_clock", "functional"}:
            raise ValueError("timing_mode must be 'wall_clock' or 'functional'")
        self.timing_mode = timing_mode
        self.action_space = gym.spaces.Box(np.full(3, -10.0, dtype=np.float32), np.full(3, 10.0, dtype=np.float32), dtype=np.float32)
        self.observation_space = task_env.observation_space
        self.calibration, self.calibration_reports = _build_synthetic_calibration(self.config, self.scenario)
        self.plant.calibration_versions.update(dict(self.calibration.versions))
        self.plant.lidar_model.sensor_version = self.calibration.sensor.version
        self.actor = ProvidedActor()
        self.replay = CertificateReplay()
        self.last_preparation: RuntimeCyclePreparation | None = None
        self.last_manifest = None
        self.last_fallback_reason: str | None = None
        self.current_epoch: CertificateEpoch | None = None
        self._frozen_preparation: RuntimeCyclePreparation | None = None
        self.last_bundle = None
        self.last_publisher: AtomicCommandPublisher | None = None
        self.last_stage_timings: dict[str, float] = {}
        self._prepare_stage_timings: dict[str, float] = {}
        self._build_certificate_objects()
        self.mission_provider = (
            SyntheticMissionCertificateProvider(self, generator_center_mode)
            if self.scenario.mission_config.get("certificate_mode") == "synthetic_preverified"
            else None
        )

    def _build_certificate_objects(self) -> None:
        width = int(ceil(self.config.world_size[0] / self.config.grid_resolution))
        height = int(ceil(self.config.world_size[1] / self.config.grid_resolution))
        self.geometry = RollingLocalGeometry(0.0, 0.0, width, height, self.config.grid_resolution)
        self.sensor_bounds = SensorBounds.from_calibration_contract(
            self.calibration.sensor,
            allow_synthetic=self.allow_synthetic_certificates,
        )
        self.corridor = ReturnCorridor(transfer_radius=self.config.body_radius, geometry_margin=self.config.geometry_margin)
        runtime_dynamics = self.calibration.dynamics.to_runtime_bounds(self.calibration.tracking)
        runtime_energy = self.calibration.energy.to_runtime_bounds()
        self.envelope_builder = SuccessorEnvelopeBuilder(runtime_dynamics, runtime_energy)
        self.recovery_policy = FrozenRecoveryPolicy(
            RecoveryConfig(
                0.0,
                0.5,
                tuple(float(value) for value in self.config.a_max),
                0.2,
                self.scenario.terminal.minimum_energy,
                0.5,
                braking_deceleration=self.config.braking_deceleration,
                update_latency=runtime_dynamics.latency_upper,
                geometry_margin=self.config.geometry_margin,
                parameter_version="synthetic-kappa-v2",
            )
        )
        self.recovery_verifier = CorridorRecoveryVerifier(
            self.recovery_policy,
            self.envelope_builder,
            self.calibration.terminal.to_runtime_condition(),
            100.0,
            self.calibration.sensor.version,
            self.calibration.tracking.version,
            self.calibration.sensor.contract_hash,
            self.calibration.tracking.contract_hash,
        )
        self.energy_solver = RecoveryEnergySolver(runtime_energy, 100.0)
        self.constructor = ZonotopeConstructor(
            self.envelope_builder,
            CertificateConfig(
                tuple(float(-value) for value in self.config.a_max),
                tuple(float(value) for value in self.config.a_max),
                self.config.minimum_generator_sigma,
                self.config.maximum_generator_condition,
                self.scenario.terminal.minimum_energy,
                0.5,
                self.config.braking_deceleration,
                runtime_dynamics.latency_upper,
                self.config.geometry_margin,
                1e-9,
                self.config.certification_deadline,
                self.config.generator_bisection_iterations,
            ),
            self.recovery_policy.config.parameter_version,
            clock=monotonic,
            enforce_wall_clock_deadline=self.timing_mode == "wall_clock",
        )
        self.closure_pipeline = SingleCorridorClosurePipeline(
            self.calibration,
            self.recovery_policy,
            self.recovery_verifier,
            self.energy_solver,
            self.constructor,
        )
        self.runtime_certifier = RuntimeCertifier(self.actor, self.recovery_policy, self.constructor, self.replay)
        self.watchdog = SimulatedWatchdog(
            self.config.certification_deadline,
            WCETContract(control_period_seconds=self.config.dt),
            enforce_wall_clock_deadline=self.timing_mode == "wall_clock",
        )

    def _corridor_cells(self) -> tuple[CorridorCell, ...]:
        cells = []
        for spec in self.scenario.corridor_cells:
            cells.append(
                CorridorCell(
                    spec.cell_id,
                    AABB2(*spec.region_low_xy, *spec.region_high_xy),
                    float(np.max(self.config.v_max[:2]) + 0.1),
                    self.geometry.version,
                    StateCellBounds(
                        Interval3(spec.state_position_low, spec.state_position_high),
                        Interval3(spec.state_velocity_low, spec.state_velocity_high),
                        Interval(spec.energy_low, spec.energy_high),
                    ),
                )
            )
        return tuple(cells)

    def _certificate_state(self) -> CertificateState:
        state = self.plant.state
        certificate_state = CertificateState(
            tuple(float(value) for value in state.position),
            tuple(float(value) for value in state.velocity),
            float(max(state.energy, 0.0)),
            tuple(float(value) for value in self.scenario.station_position),
            self.geometry,
            self.corridor,
            explicit_task_state={
                "scenario": self.scenario.name,
                "mission_phase": self.task_env.phase.name,
            },
            position_error_radius=self.calibration.dynamics.initial_position_radius,
            velocity_error_radius=self.calibration.dynamics.initial_velocity_radius,
            energy_error_radius=self.calibration.energy.measurement_error,
        )
        certificate_state.bound_versions = dict(self.calibration.versions + self.calibration.fingerprints) | {
            "kappa": self.recovery_policy.config.parameter_version,
        }
        return certificate_state

    def _mission_preparation(self) -> RuntimeCyclePreparation:
        state = self._certificate_state()
        state.explicit_task_state["mission_phase"] = self.task_env.phase.name
        context = self.mission_provider.evaluate(state, self.plant.state.timestamp)
        if self.task_env.multi_step_mission and self.task_env.phase.name == "OUTBOUND" and state.energy <= context.required_energy + self.mission_provider.trigger_margin:
            from .task_wrapper import MissionPhase

            self.task_env.phase = MissionPhase.RETURN
            self.task_env.return_triggered = True
            state.explicit_task_state["mission_phase"] = self.task_env.phase.name
            context = self.mission_provider.evaluate(state, self.plant.state.timestamp)
        observation = self.task_env.build_observation(self._map_encoding(), self._corridor_encoding())
        failure = (
            None
            if context.recovery.certified and self.mission_provider.gate_pass
            else context.closure.status
        )
        preparation = RuntimeCyclePreparation(state, observation, context.closure, context.recovery, failure)
        self.last_preparation = preparation
        self._prepare_stage_timings = {name: 0.0 for name in ("T_sensor", "T_update", "T_snapshot", "T_corridor", "T_energy", "T_set")}
        return preparation

    def _map_encoding(self) -> np.ndarray:
        values: list[float] = []
        position = self.plant.state.position
        column = int((position[0] - self.geometry.origin_x) / self.geometry.resolution)
        row = int((position[1] - self.geometry.origin_y) / self.geometry.resolution)
        radius = 0
        while len(values) < self.config.local_map_encoding_size:
            for row_offset in range(-radius, radius + 1):
                for column_offset in range(-radius, radius + 1):
                    if max(abs(row_offset), abs(column_offset)) != radius:
                        continue
                    selected_row, selected_column = row + row_offset, column + column_offset
                    if 0 <= selected_row < self.geometry.height and 0 <= selected_column < self.geometry.width:
                        values.append(float(int(self.geometry.state_at(selected_row, selected_column)) - 1))
                        if len(values) == self.config.local_map_encoding_size:
                            break
                if len(values) == self.config.local_map_encoding_size:
                    break
            radius += 1
        return np.asarray(values, dtype=np.float64)

    def _corridor_encoding(self) -> np.ndarray:
        encoded: list[float] = []
        for cell in self.corridor.cells:
            encoded.extend(
                (
                    cell.region.low_x / self.config.world_size[0],
                    cell.region.low_y / self.config.world_size[1],
                    cell.region.high_x / self.config.world_size[0],
                    cell.region.high_y / self.config.world_size[1],
                    cell.maximum_speed / float(np.max(self.config.v_max)),
                    float(cell.valid),
                )
            )
        result = np.zeros(self.config.corridor_encoding_size, dtype=np.float64)
        result[: min(len(result), len(encoded))] = encoded[: len(result)]
        return result

    def prepare_certificate_cycle(self) -> RuntimeCyclePreparation:
        if self.mission_provider is not None:
            return self._mission_preparation()
        prepare_timings: dict[str, float] = {}
        sensor_started = monotonic()
        if self.freeze_certificate_epoch and self._frozen_preparation is not None:
            preparation = self._frozen_preparation
            current_state = self._certificate_state()
            if current_state.snapshot() != preparation.state.snapshot():
                observation = self.task_env.build_observation(self._map_encoding(), self._corridor_encoding())
                return RuntimeCyclePreparation(
                    current_state,
                    observation,
                    None,
                    None,
                    "FROZEN_CERTIFICATE_STATE_MISMATCH",
                )
            self._prepare_stage_timings = {
                "T_sensor": 0.0,
                "T_update": 0.0,
                "T_snapshot": monotonic() - sensor_started,
                "T_corridor": 0.0,
                "T_energy": 0.0,
                "T_set": 0.0,
            }
            return preparation
        packet = self.plant.last_lidar
        if packet is None:
            raise RuntimeError("LiDAR packet is unavailable")
        initialization_failure = self.plant.scenario_consistency_failures[0] if self.plant.scenario_consistency_failures else None
        if self.plant.state.energy < self.scenario.terminal.minimum_energy + 0.5:
            initialization_failure = initialization_failure or "INSUFFICIENT_RECOVERY_RESERVE"
        if initialization_failure is not None or not self.config.certified_sensing_valid:
            state = self._certificate_state()
            observation = self.task_env.build_observation(self._map_encoding(), self._corridor_encoding())
            reason = initialization_failure or "INSUFFICIENT_SENSING_FOR_BRAKING_TUBE"
            preparation = RuntimeCyclePreparation(state, observation, None, None, reason)
            self.last_preparation = preparation
            return preparation
        prepare_timings["T_sensor"] = monotonic() - sensor_started
        update_started = monotonic()
        self.geometry.update_lidar(
            (float(packet.pose_position[0]), float(packet.pose_position[1])),
            packet.to_certificate_rays(),
            self.sensor_bounds,
            packet.timestamp,
        )
        prepare_timings["T_update"] = monotonic() - update_started
        snapshot_started = monotonic()
        state = self._certificate_state()
        prepare_timings["T_snapshot"] = monotonic() - snapshot_started
        closure = self.closure_pipeline.close(
            state,
            self.geometry,
            self.corridor,
            self._corridor_cells(),
            _operating_point(self.config, self.plant.state),
            self.calibration.sensor.metadata.device_version,
            self.calibration.sensor.metadata.firmware_version,
            self.plant.state.timestamp,
            allow_synthetic=self.allow_synthetic_certificates,
        )
        observation = self.task_env.build_observation(self._map_encoding(), self._corridor_encoding())
        recovery = self.runtime_certifier.recovery_decision(state, self.plant.state.timestamp)
        prepare_timings.update(self.closure_pipeline.last_stage_timings)
        self._prepare_stage_timings = prepare_timings
        failure = None if closure.closed else closure.failure_witness.failed_predicate if closure.failure_witness else closure.status
        if closure.closed:
            self.last_manifest = closure.manifest
        preparation = RuntimeCyclePreparation(state, observation, closure, recovery, failure)
        self.last_preparation = preparation
        return preparation

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = self.task_env.reset(seed=seed, options=options)
        if not (self.freeze_certificate_epoch and self._frozen_preparation is not None):
            self._build_certificate_objects()
            if not self.plant.scenario_consistency_failures and self.config.certified_sensing_valid:
                for packet in self.plant.synthetic_bootstrap_lidar_packets():
                    self.geometry.update_lidar(
                        (float(packet.pose_position[0]), float(packet.pose_position[1])),
                        packet.to_certificate_rays("bootstrap"),
                        self.sensor_bounds,
                        packet.timestamp,
                    )
        if self.mission_provider is not None:
            self.mission_provider.reset()
        preparation = self.prepare_certificate_cycle()
        if self.freeze_certificate_epoch and preparation.failure_reason is None:
            self._frozen_preparation = preparation
        self.current_epoch = CertificateEpoch.from_snapshot(preparation.state.snapshot())
        return preparation.task_observation, info | {
            "certificate_ready": preparation.failure_reason is None,
            "certificate_failure_reason": preparation.failure_reason,
            "synthetic_certificate_evidence": True,
        }

    def step(self, actor_output: np.ndarray):
        total_started = monotonic()
        timings: dict[str, float] = {}
        stage_started = monotonic()
        pre_state = self._certificate_state()
        pre_snapshot = pre_state.snapshot()
        pre_recovery = (
            self.mission_provider.evaluate(pre_state, self.plant.state.timestamp).recovery
            if self.mission_provider is not None
            else self.runtime_certifier.recovery_decision(pre_state, self.plant.state.timestamp)
        )
        pre_fallback = (
            pre_recovery.action
            if self.mission_provider is not None or pre_recovery.certified
            else self.recovery_policy.emergency_brake(pre_state.velocity)
        )
        timings["T_kappa"] = monotonic() - stage_started
        publisher = AtomicCommandPublisher()
        self.last_publisher = publisher
        publisher.stage_default(
            PublishedCommand(pre_fallback, "kappa", "STAGED_BEFORE_CERTIFICATION", pre_snapshot.certificate_version)
        )
        cycle_started = monotonic()
        preparation = self.prepare_certificate_cycle()
        cycle_elapsed = monotonic() - cycle_started
        timings["T_certificate"] = cycle_elapsed
        timings.update(self._prepare_stage_timings)
        state = preparation.state
        snapshot = state.snapshot()
        self.current_epoch = CertificateEpoch.from_snapshot(snapshot)
        closure = preparation.closure_result
        recovery = preparation.recovery
        bundle_holder: dict[str, Any] = {}
        self.last_bundle = None
        cycle_deadline_missed = (
            self.timing_mode == "wall_clock"
            and self.watchdog.wcet_contract.status == "implemented"
            and cycle_elapsed > self.config.certification_deadline
        )
        if cycle_deadline_missed or preparation.failure_reason is not None or recovery is None or not recovery.certified or closure is None or not closure.closed:
            command_action = pre_fallback
            command_source = "kappa"
            command_reason = (
                "CERTIFICATION_CYCLE_DEADLINE"
                if cycle_deadline_missed
                else preparation.failure_reason
                or (closure.status if closure is not None and not closure.closed else None)
                or (recovery.reason if recovery is not None else "CERTIFICATE_UNAVAILABLE")
            )
            publisher.publish_once(
                PublishedCommand(pre_fallback, "kappa", command_reason, pre_snapshot.certificate_version)
            )
            nominal = None
            candidate = None
            execution_snapshot = pre_snapshot
            execution_recovery = pre_recovery
        else:
            self.actor.set_output(actor_output)

            def producer():
                actor_started = monotonic()
                bundle = self.runtime_certifier.prepare_candidate_from_certificate(
                    state,
                    preparation.task_observation,
                    recovery,
                    closure.zonotope_certificate,
                    self.plant.state.timestamp,
                    snapshot,
                )
                timings["T_actor"] = monotonic() - actor_started
                bundle_holder["bundle"] = bundle
                return bundle

            watchdog_started = monotonic()
            command = self.watchdog.execute(
                snapshot,
                recovery.action,
                producer,
                state.snapshot,
                publisher,
            )
            watchdog_elapsed = monotonic() - watchdog_started
            timings["T_publish"] = publisher.last_publish_elapsed
            timings["T_recheck"] = max(
                0.0,
                watchdog_elapsed - timings.get("T_actor", 0.0) - timings["T_publish"],
            )
            command_action = command.action
            command_source = command.source
            command_reason = command.reason
            self.last_bundle = bundle_holder.get("bundle")
            bundle = bundle_holder.get("bundle") if command.source == "task" else None
            nominal = np.asarray(bundle.nominal_u, dtype=np.float64) if bundle is not None else None
            candidate = np.asarray(bundle.final_action, dtype=np.float64) if bundle is not None else None
            execution_snapshot = snapshot
            execution_recovery = recovery
        plant_started = monotonic()
        mission_context = self.mission_provider.last_context if self.mission_provider is not None else None
        if self.mission_provider is not None and command_source != "task" and execution_recovery.certified:
            from .task_wrapper import MissionPhase

            self.task_env.phase = MissionPhase.RETURN
            self.task_env.return_triggered = True
        observation, reward, terminated, truncated, info = self.task_env.step(np.asarray(command_action, dtype=np.float64))
        if self.mission_provider is not None and not execution_recovery.certified:
            from .task_wrapper import MissionPhase, MissionTerminationReason

            terminated = True
            self.task_env.phase = MissionPhase.FAILURE
            self.task_env.termination_reason = MissionTerminationReason.RECOVERY_CERTIFICATE_INVALID
            info = info | {
                "failure_reason": "recovery_certificate_invalid",
                "mission_phase": MissionPhase.FAILURE.name,
                "mission_termination_reason": MissionTerminationReason.RECOVERY_CERTIFICATE_INVALID.value,
            }
        timings["T_plant"] = monotonic() - plant_started
        measured = self.plant.last_telemetry.action_trace.measured
        fallback_action = np.asarray(recovery.action if recovery is not None else command_action, dtype=np.float64)
        trace = ActionTrace(
            nominal,
            candidate,
            fallback_action,
            np.asarray(command_action, dtype=np.float64),
            measured,
            command_source == "task",
            None if command_source == "task" else command_reason,
            str(execution_snapshot.certificate_version),
        )
        telemetry = self.plant.attach_runtime_trace(
            trace,
            str(execution_snapshot.certificate_version),
            str(self.geometry.version),
            str(self.corridor.version),
        )
        bundle = bundle_holder.get("bundle") if command_source == "task" else None
        recorded_recovery = execution_recovery or RecoveryDecision(
            tuple(float(value) for value in fallback_action),
            False,
            None,
            command_reason,
        )
        self.runtime_certifier.record_published_command(
            execution_snapshot,
            preparation.task_observation,
            recorded_recovery,
            tuple(float(value) for value in command_action),
            command_source,
            command_reason,
            self.plant.state.timestamp,
            bundle,
            tuple(float(value) for value in measured),
        )
        if self.mission_provider is not None and mission_context is not None:
            self.mission_provider.commit_execution(mission_context, command_source == "task")
        timings["T_log"] = monotonic() - plant_started - timings["T_plant"]
        self.last_fallback_reason = trace.fallback_reason
        next_observation = self.task_env.build_observation(self._map_encoding(), self._corridor_encoding())
        info = info | {
            "telemetry": telemetry,
            "certificate_manifest": self.last_manifest,
            "certificate_epoch": self.current_epoch,
            "accepted": trace.accepted,
            "fallback_reason": trace.fallback_reason,
            "critic_action": trace.published.copy(),
            "actor_called": self.actor.calls,
            "certificate_cycle_profile_seconds": cycle_elapsed,
            "wcet_status": self.watchdog.wcet_contract.status,
            "timing_mode": self.timing_mode,
            "candidate_bundle": self.last_bundle,
            "publication_count": publisher.publication_count,
            "action_context": self.action_context(preparation),
        }
        timings.setdefault("T_actor", 0.0)
        timings.setdefault("T_recheck", 0.0)
        timings.setdefault("T_publish", publisher.last_publish_elapsed)
        timings["T_total"] = monotonic() - total_started
        self.last_stage_timings = timings
        info["stage_timings"] = dict(timings)
        return next_observation, reward, terminated, truncated, info

    def action_context(self, preparation: RuntimeCyclePreparation | None = None) -> dict[str, Any]:
        selected = self.last_preparation if preparation is None else preparation
        if selected is None:
            return {"certificate_valid": False, "generator_available": False}
        closure = selected.closure_result
        certificate = None if closure is None else closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        recovery = selected.recovery
        mission_context = self.mission_provider.last_context if self.mission_provider is not None else None
        return {
            "certificate_valid": bool(recovery is not None and recovery.certified),
            "generator_available": bool(certificate is not None and certificate.verified and zonotope is not None),
            "c": None if zonotope is None else np.asarray(zonotope.center, dtype=np.float32),
            "G": None if zonotope is None else np.asarray(zonotope.generators, dtype=np.float32),
            "kappa": None if recovery is None else np.asarray(recovery.action, dtype=np.float32),
            "certificate_epoch": (
                self.mission_provider.manifest.manifest_hash
                if self.mission_provider is not None
                else None if self.current_epoch is None else self.current_epoch.epoch_id
            ),
            "certificate_version": selected.state.certificate_version,
            "geometry_version": self.geometry.version,
            "corridor_version": self.corridor.version,
            "energy_version": self.calibration.energy.version,
            "recovery_hash": None if recovery is None else recovery.certificate_hash,
            "zonotope_hash": None if certificate is None else certificate.complete_set_inclusion_hash,
            "failure_reason": selected.failure_reason,
            "generator_status": None if closure is None else closure.status,
            "mission_certificate_gate": (
                None
                if self.mission_provider is None
                else "PASS" if self.mission_provider.gate_pass else "blocked-by-mission-certificate"
            ),
            "recovery_level": None if mission_context is None else mission_context.recovery_level,
            "recovery_energy_required": None if mission_context is None else mission_context.required_energy,
            "energy_margin": None if mission_context is None else mission_context.current_energy_margin,
            "recovery_cell_id": None if mission_context is None else mission_context.recovery_cell_id,
        }

    def preview_next_action_context(self) -> dict[str, Any]:
        preparation = self.prepare_certificate_cycle()
        self.current_epoch = CertificateEpoch.from_snapshot(preparation.state.snapshot())
        return self.action_context(preparation)

    def step_nominal_action(self, nominal_action: np.ndarray):
        """B2 control: execute a nominal box action only when it belongs to C_run."""

        total_started = monotonic()
        pre_state = self._certificate_state()
        context = self.mission_provider.evaluate(pre_state, self.plant.state.timestamp) if self.mission_provider is not None else None
        recovery = context.recovery if context is not None else self.runtime_certifier.recovery_decision(pre_state, self.plant.state.timestamp)
        fallback = np.asarray(recovery.action if recovery.certified else self.recovery_policy.emergency_brake(pre_state.velocity), dtype=np.float64)
        publisher = AtomicCommandPublisher()
        self.last_publisher = publisher
        snapshot = pre_state.snapshot()
        publisher.stage_default(PublishedCommand(tuple(fallback), "kappa", "STAGED_BEFORE_NOMINAL_CHECK", snapshot.certificate_version))
        certificate_started = monotonic()
        preparation = self.prepare_certificate_cycle()
        certificate_elapsed = monotonic() - certificate_started
        selected = np.asarray(nominal_action, dtype=np.float64)
        closure = preparation.closure_result
        certificate = None if closure is None else closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        point_verified = bool(
            self.mission_provider is not None
            and self.mission_provider.verify_task_action(preparation.state, selected)
        )
        accepted = bool(
            preparation.failure_reason is None
            and preparation.recovery is not None
            and preparation.recovery.certified
            and (
                point_verified
                if self.mission_provider is not None
                else certificate is not None and certificate.verified and zonotope is not None and zonotope.contains(selected)
            )
        )
        action = selected if accepted else fallback
        reason = "VERIFIED_NOMINAL_MEMBERSHIP" if accepted else (preparation.failure_reason or "NOMINAL_OUTSIDE_CERTIFIED_SET")
        publish_started = monotonic()
        publisher.publish_once(PublishedCommand(tuple(float(value) for value in action), "task" if accepted else "kappa", reason, snapshot.certificate_version))
        publish_elapsed = monotonic() - publish_started
        if self.mission_provider is not None and not accepted and preparation.recovery is not None and preparation.recovery.certified:
            from .task_wrapper import MissionPhase

            self.task_env.phase = MissionPhase.RETURN
            self.task_env.return_triggered = True
        plant_started = monotonic()
        observation, reward, terminated, truncated, info = self.task_env.step(action)
        plant_elapsed = monotonic() - plant_started
        measured = self.plant.last_telemetry.action_trace.measured
        trace = ActionTrace(selected, selected if accepted else None, fallback, action, measured, accepted, None if accepted else reason, str(snapshot.certificate_version))
        telemetry = self.plant.attach_runtime_trace(trace, str(snapshot.certificate_version), str(self.geometry.version), str(self.corridor.version))
        if self.mission_provider is not None and self.mission_provider.last_context is not None:
            self.mission_provider.commit_execution(self.mission_provider.last_context, accepted)
        self.last_stage_timings = {
            "T_certificate": certificate_elapsed,
            "T_recheck": 0.0,
            "T_publish": publish_elapsed,
            "T_plant": plant_elapsed,
            "T_total": monotonic() - total_started,
        }
        return observation, reward, terminated, truncated, info | {
            "telemetry": telemetry,
            "accepted": accepted,
            "fallback_reason": trace.fallback_reason,
            "critic_action": action.copy(),
            "certificate_epoch": CertificateEpoch.from_snapshot(preparation.state.snapshot()),
            "publication_count": publisher.publication_count,
            "action_context": self.action_context(preparation),
            "stage_timings": dict(self.last_stage_timings),
        }

    def export_calibration_record(self):
        return self.plant.export_calibration_record()
