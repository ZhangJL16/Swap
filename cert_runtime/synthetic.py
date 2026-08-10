from __future__ import annotations

from dataclasses import dataclass

from calibration.synthetic import build_synthetic_calibration_bundle, synthetic_operating_domain

from .adapters import (
    FixedTimestampSource,
    InMemoryLogRecorder,
    MockActuatorSink,
    MockEnergySource,
    MockLidarSource,
    MockStateSource,
    MockTrackingSource,
)
from .certificates import StateCellBounds
from .closure import CalibrationBundle, SingleCorridorClosurePipeline
from .contracts import WCETContract
from .corridor import CorridorCell, ReturnCorridor
from .energy import RecoveryEnergySolver
from .envelope import SuccessorEnvelopeBuilder
from .geometry import RollingLocalGeometry, SensorBounds
from .interval import Interval
from .recovery import CorridorRecoveryVerifier, FrozenRecoveryPolicy, RecoveryConfig
from .runtime import CertificateReplay, RuntimeCertifier
from .state import CertificateState
from .types import AABB2, Interval3
from .watchdog import SimulatedWatchdog
from .zonotope import CertificateConfig, ZonotopeConstructor


class SyntheticActor:
    def __init__(self, output: tuple[float, float, float] = (0.2, -0.3, 0.5)) -> None:
        self.output = output
        self.calls = 0

    def sample_u(self, observation):
        self.calls += 1
        return self.output


@dataclass
class SyntheticClosureFixture:
    calibration: CalibrationBundle
    reports: tuple[object, ...]
    geometry: RollingLocalGeometry
    corridor: ReturnCorridor
    cells: tuple[CorridorCell, ...]
    state: CertificateState
    policy: FrozenRecoveryPolicy
    recovery_verifier: CorridorRecoveryVerifier
    energy_solver: RecoveryEnergySolver
    constructor: ZonotopeConstructor
    closure: SingleCorridorClosurePipeline
    actor: SyntheticActor
    runtime: RuntimeCertifier
    replay: CertificateReplay
    sensor_bounds: SensorBounds
    operating_point: dict[str, float | str]
    device_version: str
    firmware_version: str
    timestamp: float


def build_synthetic_closure_fixture() -> SyntheticClosureFixture:
    contracts, reports = build_synthetic_calibration_bundle()
    sensor, tracking, dynamics_contract, energy_contract, terminal_contract = contracts
    calibration = CalibrationBundle(sensor, dynamics_contract, tracking, energy_contract, terminal_contract)
    timestamp = 20.0
    geometry = RollingLocalGeometry(-6.0, -6.0, 24, 24, 0.5)
    geometry.mark_free_from_certificate(
        AABB2(-5.5, -5.5, 5.5, 5.5),
        "synthetic-continuous-free-proof",
        timestamp,
        sensor.version,
    )
    terminal_bounds = StateCellBounds(
        Interval3((-3.0, -3.0, -0.3), (3.0, 3.0, 0.3)),
        Interval3((-1.5, -1.5, -0.5), (1.5, 1.5, 0.5)),
        Interval(10.0, 101.0),
    )
    outer_bounds = StateCellBounds(
        Interval3((0.9, -0.1, -0.1), (1.1, 0.1, 0.1)),
        Interval3((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05)),
        Interval(20.0, 100.0),
    )
    cells = (
        CorridorCell(0, AABB2(-4.0, -4.0, 4.0, 4.0), 2.5, geometry.version, terminal_bounds),
        CorridorCell(1, AABB2(0.5, -0.5, 1.5, 0.5), 2.0, geometry.version, outer_bounds),
    )
    corridor = ReturnCorridor(transfer_radius=0.1, geometry_margin=0.02)
    dynamics = dynamics_contract.to_runtime_bounds(tracking)
    energy = energy_contract.to_runtime_bounds()
    envelope = SuccessorEnvelopeBuilder(dynamics, energy)
    policy = FrozenRecoveryPolicy(
        RecoveryConfig(
            1.0,
            0.5,
            (2.0, 2.0, 2.0),
            0.25,
            terminal_contract.minimum_energy,
            0.5,
            braking_deceleration=10.0,
            update_latency=dynamics.latency_upper,
            geometry_margin=0.02,
            parameter_version="synthetic-kappa-v1",
        )
    )
    recovery_verifier = CorridorRecoveryVerifier(
        policy,
        envelope,
        terminal_contract.to_runtime_condition(),
        100.0,
        sensor.version,
        tracking.version,
        sensor.contract_hash,
        tracking.contract_hash,
    )
    energy_solver = RecoveryEnergySolver(energy, 100.0)
    constructor = ZonotopeConstructor(
        envelope,
        CertificateConfig(
            (-2.0, -2.0, -2.0),
            (2.0, 2.0, 2.0),
            0.05,
            20.0,
            terminal_contract.minimum_energy,
            0.5,
            10.0,
            dynamics.latency_upper,
            0.02,
            1e-8,
            1.0,
            8,
        ),
        policy.config.parameter_version,
        clock=lambda: timestamp,
    )
    closure = SingleCorridorClosurePipeline(
        calibration,
        policy,
        recovery_verifier,
        energy_solver,
        constructor,
    )
    state = CertificateState(
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        50.0,
        (0.0, 0.0, 0.0),
        geometry,
        corridor,
        position_error_radius=dynamics_contract.initial_position_radius,
        velocity_error_radius=dynamics_contract.initial_velocity_radius,
        energy_error_radius=energy_contract.measurement_error,
    )
    actor = SyntheticActor()
    replay = CertificateReplay()
    runtime = RuntimeCertifier(actor, policy, constructor, replay)
    sensor_bounds = SensorBounds.from_calibration_contract(sensor, allow_synthetic=True)
    operating_point = {
        "speed": 0.5,
        "acceleration": 0.5,
        "payload": 0.5,
        "temperature": 20.0,
        "voltage": 20.0,
        "flight_mode": "hover",
    }
    return SyntheticClosureFixture(
        calibration,
        reports,
        geometry,
        corridor,
        cells,
        state,
        policy,
        recovery_verifier,
        energy_solver,
        constructor,
        closure,
        actor,
        runtime,
        replay,
        sensor_bounds,
        operating_point,
        sensor.metadata.device_version,
        sensor.metadata.firmware_version,
        timestamp,
    )


def synthetic_watchdog(deadline_seconds: float = 0.02) -> SimulatedWatchdog:
    return SimulatedWatchdog(deadline_seconds, WCETContract(control_period_seconds=0.5))


def synthetic_adapters(fixture: SyntheticClosureFixture):
    return {
        "state": MockStateSource(fixture.state),
        "lidar": MockLidarSource(tuple()),
        "energy": MockEnergySource(fixture.state.energy),
        "sink": MockActuatorSink(),
        "time": FixedTimestampSource(fixture.timestamp),
        "recorder": InMemoryLogRecorder(),
        "tracking": MockTrackingSource(),
    }
