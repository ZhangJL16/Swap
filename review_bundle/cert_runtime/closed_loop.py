from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite
from time import sleep
from typing import Mapping, Sequence

from .adapters import (
    ActuatorCommandSink,
    LidarSource,
    LogRecorder,
    PowerEnergySource,
    StateSource,
    TimestampSource,
    ActionTrackingSource,
    WatchdogAdapter,
)
from .closure import SingleCorridorClosurePipeline
from .corridor import CorridorCell
from .geometry import SensorBounds
from .runtime import RuntimeCertifier
from .watchdog import PublishedCommand


class FailureInjection(str, Enum):
    CALIBRATION_MISSING = "calibration-missing"
    CALIBRATION_EXPIRED = "calibration-expired"
    POSE_JUMP = "pose-jump"
    STALE_LIDAR = "stale-lidar"
    INVALID_BEAM = "invalid-beam"
    CORRIDOR_SUFFIX_INVALID = "corridor-suffix-invalidation"
    ENERGY_VERSION_CHANGE = "energy-version-change"
    DYNAMICS_VERSION_CHANGE = "dynamics-version-change"
    RECOVERY_CERTIFICATE_TAMPER = "recovery-certificate-tamper"
    E3_RESIDUAL_FAILURE = "e3-residual-failure"
    NO_GENERATOR_SET = "no-generator-set"
    ACTOR_NAN = "actor-nan"
    ACTOR_TIMEOUT = "actor-timeout"
    CERTIFIER_BLOCK = "certifier-block"
    WORKER_EXCEPTION = "worker-exception"
    CERTIFICATE_MUTATION = "certificate-mutation"
    PUBLISH_TIMEOUT = "publish-timeout"


@dataclass(frozen=True)
class ClosedLoopAuditRecord:
    timestamp: float
    accepted: bool
    command_source: str
    executed_action: tuple[float, float, float]
    fallback_reason: str
    certificate_version: tuple[int, int, int]
    bound_versions: tuple[tuple[str, str], ...]
    injection: str | None
    closure_status: str


class DeterministicClosedLoopHarness:
    """Synthetic/HIL coordinator; it does not constitute flight evidence."""

    def __init__(
        self,
        closure: SingleCorridorClosurePipeline,
        runtime: RuntimeCertifier,
        watchdog: WatchdogAdapter,
        state_source: StateSource,
        lidar_source: LidarSource,
        energy_source: PowerEnergySource,
        command_sink: ActuatorCommandSink,
        timestamp_source: TimestampSource,
        recorder: LogRecorder,
        fixed_cells: Sequence[CorridorCell],
        operating_point: Mapping[str, float | str],
        device_version: str,
        firmware_version: str,
        sensor_bounds: SensorBounds,
        tracking_source: ActionTrackingSource | None = None,
    ) -> None:
        self.closure = closure
        self.runtime = runtime
        self.watchdog = watchdog
        self.state_source = state_source
        self.lidar_source = lidar_source
        self.energy_source = energy_source
        self.command_sink = command_sink
        self.timestamp_source = timestamp_source
        self.recorder = recorder
        self.fixed_cells = tuple(fixed_cells)
        self.operating_point = operating_point
        self.device_version = device_version
        self.firmware_version = firmware_version
        self.sensor_bounds = sensor_bounds
        self.tracking_source = tracking_source

    def run_cycle(
        self,
        task_observation: Sequence[float],
        injection: FailureInjection | None = None,
    ) -> ClosedLoopAuditRecord:
        timestamp = self.timestamp_source.now()
        state = self.state_source.read_state()
        state.energy = self.energy_source.read_energy()
        if injection in {FailureInjection.CALIBRATION_MISSING, FailureInjection.CALIBRATION_EXPIRED}:
            return self._emergency(state, timestamp, injection.value, injection, "calibration-invalid")
        calibration_valid, calibration_reason = self.closure.calibration.validate(
            timestamp,
            self.operating_point,
            self.device_version,
            self.firmware_version,
            allow_synthetic=True,
        )
        if not calibration_valid:
            return self._emergency(
                state,
                timestamp,
                calibration_reason,
                injection,
                "calibration-invalid",
            )
        if injection == FailureInjection.POSE_JUMP:
            return self._emergency(state, timestamp, "pose-jump", injection, "state-estimate-invalid")
        rays = self.lidar_source.read_rays()
        if injection == FailureInjection.STALE_LIDAR:
            rays = tuple(replace(ray, timestamp=timestamp - 10.0) for ray in rays)
        elif injection == FailureInjection.INVALID_BEAM:
            rays = tuple(replace(ray, valid=False) for ray in rays)
        state.local_geometry.update_lidar(
            (state.position[0], state.position[1]),
            rays,
            self.sensor_bounds,
            timestamp,
        )
        if injection in {FailureInjection.STALE_LIDAR, FailureInjection.INVALID_BEAM}:
            return self._emergency(state, timestamp, injection.value, injection, "geometry-update-invalid")
        if injection == FailureInjection.CORRIDOR_SUFFIX_INVALID:
            return self._emergency(state, timestamp, injection.value, injection, "corridor-invalid")
        closure_result = self.closure.close(
            state,
            state.local_geometry,
            state.return_corridor,
            self.fixed_cells,
            self.operating_point,
            self.device_version,
            self.firmware_version,
            timestamp,
            allow_synthetic=True,
        )
        if not closure_result.closed:
            reason = closure_result.failure_witness.failed_predicate if closure_result.failure_witness else closure_result.status
            return self._emergency(state, timestamp, reason, injection, closure_result.status)
        if injection == FailureInjection.ENERGY_VERSION_CHANGE:
            state.bound_versions["energy"] = "mutated-energy-version"
        elif injection == FailureInjection.DYNAMICS_VERSION_CHANGE:
            state.bound_versions["dynamics"] = "mutated-dynamics-version"
        elif injection == FailureInjection.RECOVERY_CERTIFICATE_TAMPER:
            level = state.return_corridor.locate_level((state.position[0], state.position[1]))
            cell = next(cell for cell in state.return_corridor.cells if cell.cell_id == level)
            cell_index = state.return_corridor.cells.index(cell)
            state.return_corridor.cells[cell_index] = replace(
                cell,
                recovery_certificate=replace(cell.recovery_certificate, transit_cost_upper=0.0),
            )
        elif injection == FailureInjection.E3_RESIDUAL_FAILURE:
            level = state.return_corridor.locate_level((state.position[0], state.position[1]))
            cell = next(cell for cell in state.return_corridor.cells if cell.cell_id == level)
            cell_index = state.return_corridor.cells.index(cell)
            state.return_corridor.cells[cell_index] = replace(
                cell,
                energy_certificate=replace(cell.energy_certificate, residual_lower=-1.0),
            )
        recovery = self.runtime.recovery_decision(state, timestamp)
        if not recovery.certified:
            return self._emergency(state, timestamp, recovery.reason, injection, closure_result.status, recovery.action)
        snapshot = state.snapshot()
        bundle_holder = {}

        def producer():
            if injection == FailureInjection.CERTIFIER_BLOCK:
                sleep(self.watchdog.deadline_seconds * 5.0 + 0.01)
            if injection == FailureInjection.WORKER_EXCEPTION:
                raise RuntimeError("injected worker exception")
            if injection == FailureInjection.NO_GENERATOR_SET:
                raise RuntimeError("NO_GENERATOR_SET")
            if injection == FailureInjection.ACTOR_TIMEOUT:
                sleep(self.watchdog.deadline_seconds * 5.0 + 0.01)
            bundle = self.runtime.prepare_candidate_from_certificate(
                state,
                task_observation,
                recovery,
                closure_result.zonotope_certificate,
                timestamp,
                snapshot,
            )
            if injection == FailureInjection.ACTOR_NAN:
                bundle = replace(bundle, nominal_u=(float("nan"), 0.0, 0.0))
            if injection == FailureInjection.CERTIFICATE_MUTATION:
                state.return_corridor.certificate_epoch += 1
            bundle_holder["bundle"] = bundle
            return bundle

        command = self.watchdog.execute(snapshot, recovery.action, producer, state.snapshot)
        if injection == FailureInjection.PUBLISH_TIMEOUT and hasattr(self.command_sink, "fail_normal_publish"):
            self.command_sink.fail_normal_publish = True
        published = self.command_sink.publish(command)
        if not published:
            self.command_sink.publish_emergency(recovery.action, "PUBLISH_TIMEOUT")
            command = PublishedCommand(
                recovery.action,
                "certificate-loss-emergency",
                "PUBLISH_TIMEOUT",
                state.certificate_version,
            )
        measured_action = (
            self.tracking_source.read_measured_action(command.action)
            if self.tracking_source is not None
            else None
        )
        self.runtime.record_published_command(
            snapshot,
            task_observation,
            recovery,
            command.action,
            command.source,
            command.reason,
            timestamp,
            bundle_holder.get("bundle"),
            measured_action,
        )
        accepted = command.source == "task"
        record = ClosedLoopAuditRecord(
            timestamp,
            accepted,
            command.source,
            command.action,
            command.reason,
            state.certificate_version,
            tuple(sorted(state.bound_versions.items())),
            injection.value if injection else None,
            closure_result.status,
        )
        self.recorder.record(record)
        return record

    def _emergency(
        self,
        state,
        timestamp: float,
        reason: str,
        injection: FailureInjection | None,
        closure_status: str,
        action=None,
    ) -> ClosedLoopAuditRecord:
        fallback = action or self.runtime.recovery_policy.emergency_brake(state.velocity)
        self.command_sink.publish_emergency(fallback, reason)
        record = ClosedLoopAuditRecord(
            timestamp,
            False,
            "certificate-loss-emergency",
            fallback,
            reason,
            state.certificate_version,
            tuple(sorted(state.bound_versions.items())),
            injection.value if injection else None,
            closure_status,
        )
        self.recorder.record(record)
        return record
