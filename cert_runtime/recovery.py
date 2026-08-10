from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from time import monotonic

from .certificates import (
    ProofMetadata,
    RecoveryCellCertificate,
    StateCellBounds,
    TerminalCondition,
    make_recovery_hash_payload,
)
from .corridor import CorridorCell, ReturnCorridor
from .envelope import SuccessorEnvelope, SuccessorEnvelopeBuilder
from .interval import Interval, round_down, round_up
from .state import CertificateState
from .types import AABB2, Interval3, Vec3


@dataclass(frozen=True)
class RecoveryConfig:
    position_gain: float
    velocity_gain: float
    maximum_acceleration: Vec3
    terminal_radius: float
    terminal_energy: float
    energy_reserve: float
    braking_deceleration: float = 1.0
    update_latency: float = 0.0
    geometry_margin: float = 0.0
    numerical_tolerance: float = 1e-9
    parameter_version: str = "unversioned-kappa"


@dataclass(frozen=True)
class RecoveryDecision:
    action: Vec3
    certified: bool
    certificate_hash: str | None
    reason: str


@dataclass(frozen=True)
class RecoveryVerificationResult:
    verified: bool
    certificates: dict[int, RecoveryCellCertificate]
    failed_cell_id: int | None
    reason: str
    witness: "RecoveryFailureWitness | None" = None


@dataclass(frozen=True)
class RecoveryFailureWitness:
    failed_cell_id: int
    failed_predicate: str
    interval_residual: str
    involved_versions: tuple[tuple[str, str], ...]
    required_margin: float | None
    actual_margin: float | None


class FrozenRecoveryPolicy:
    """Explicit corridor controller whose runtime authority comes from cell certificates."""

    def __init__(self, config: RecoveryConfig) -> None:
        self.config = config
        self.calls = 0

    def action(self, state: CertificateState) -> Vec3:
        return self._point_action(state)

    def certified_action(
        self,
        state: CertificateState,
        dynamics_bound_version: str,
        energy_bound_version: str,
        timestamp: float,
    ) -> RecoveryDecision:
        self.calls += 1
        level = state.return_corridor.locate_level((state.position[0], state.position[1]))
        if level is None:
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "outside-corridor")
        cell = next(cell for cell in state.return_corridor.cells if cell.cell_id == level)
        certificate = cell.recovery_certificate
        energy_certificate = cell.energy_certificate
        if certificate is None or energy_certificate is None:
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "missing-cell-certificate")
        metadata = certificate.proof_metadata
        expected_versions = {
            "sensor": metadata.sensor_version if metadata else None,
            "dynamics": metadata.dynamics_version if metadata else None,
            "tracking": metadata.tracking_version if metadata else None,
            "energy": metadata.energy_version if metadata else None,
            "terminal": metadata.terminal_version if metadata else None,
            "kappa": metadata.kappa_version if metadata else None,
        }
        if metadata is None or any(
            key in state.bound_versions and state.bound_versions[key] != value
            for key, value in expected_versions.items()
        ):
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "bound-version-mismatch")
        if not self._state_belongs_to_certificate(state, certificate.state_bounds):
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "state-outside-cell-certificate")
        if not certificate.is_valid(
            state.local_geometry.version,
            state.return_corridor.version,
            self.config.parameter_version,
            dynamics_bound_version,
            energy_bound_version,
            timestamp,
        ):
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "stale-cell-certificate")
        if not energy_certificate.is_valid(
            certificate.certificate_hash,
            energy_bound_version,
            state.return_corridor.version,
            timestamp,
        ):
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "stale-energy-certificate")
        energy_lower = round_down(state.energy - state.energy_error_radius)
        required_energy = round_up(
            energy_certificate.transit_energy_upper
            + self.config.terminal_energy
            + self.config.energy_reserve
        )
        if energy_lower < required_energy:
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "insufficient-recovery-reserve")
        action = self._point_action(state)
        if not certificate.action_interval.contains_point(action, self.config.numerical_tolerance):
            return RecoveryDecision(self.emergency_brake(state.velocity), False, None, "action-outside-cell-proof")
        return RecoveryDecision(action, True, certificate.certificate_hash, "certified")

    def _state_belongs_to_certificate(
        self,
        state: CertificateState,
        bounds: StateCellBounds,
    ) -> bool:
        tolerance = self.config.numerical_tolerance
        return (
            bounds.position.contains_point(state.position, tolerance)
            and bounds.velocity.contains_point(state.velocity, tolerance)
            and bounds.energy.contains(state.energy, tolerance)
        )

    def _point_action(self, state: CertificateState) -> Vec3:
        corridor = state.return_corridor
        level = corridor.locate_level((state.position[0], state.position[1]))
        if level is None or level == 0:
            return self.emergency_brake(state.velocity)
        current_cell = next(cell for cell in corridor.cells if cell.cell_id == level)
        if hypot(state.velocity[0], state.velocity[1]) > current_cell.maximum_speed:
            return self.emergency_brake(state.velocity)
        target_cell = corridor.lower_level_cell(level)
        if target_cell is None:
            return self.emergency_brake(state.velocity)
        target_x, target_y = target_cell.region.center
        target_z = (
            target_cell.state_bounds.position.low[2]
            + target_cell.state_bounds.position.high[2]
        ) / 2.0
        raw = (
            self.config.position_gain * (target_x - state.position[0])
            - self.config.velocity_gain * state.velocity[0],
            self.config.position_gain * (target_y - state.position[1])
            - self.config.velocity_gain * state.velocity[1],
            self.config.position_gain * (target_z - state.position[2])
            - self.config.velocity_gain * state.velocity[2],
        )
        return tuple(
            min(max(raw[index], -self.config.maximum_acceleration[index]), self.config.maximum_acceleration[index])
            for index in range(3)
        )  # type: ignore[return-value]

    def action_interval_for_cell(
        self,
        cell: CorridorCell,
        target: CorridorCell | None,
    ) -> Interval3:
        intervals = []
        if target is None:
            for index in range(3):
                raw = cell.state_bounds.velocity.components[index].scale(-self.config.velocity_gain)
                intervals.append(
                    raw.saturate(
                        -self.config.maximum_acceleration[index],
                        self.config.maximum_acceleration[index],
                    )
                )
            return Interval3.from_intervals(intervals)
        target_coordinates = (
            target.region.center[0],
            target.region.center[1],
            (target.state_bounds.position.low[2] + target.state_bounds.position.high[2]) / 2.0,
        )
        for index in range(3):
            position_term = (
                Interval.point(target_coordinates[index])
                - cell.state_bounds.position.components[index]
            ).scale(self.config.position_gain)
            velocity_term = cell.state_bounds.velocity.components[index].scale(-self.config.velocity_gain)
            raw = position_term + velocity_term
            intervals.append(
                raw.saturate(
                    -self.config.maximum_acceleration[index],
                    self.config.maximum_acceleration[index],
                )
            )
        return Interval3.from_intervals(intervals)

    def emergency_brake(self, velocity: Vec3) -> Vec3:
        return tuple(
            min(
                max(-self.config.velocity_gain * velocity[index], -self.config.maximum_acceleration[index]),
                self.config.maximum_acceleration[index],
            )
            for index in range(3)
        )  # type: ignore[return-value]


class CorridorRecoveryVerifier:
    """Offline/epoch-level one-step verifier over complete corridor state cells."""

    def __init__(
        self,
        policy: FrozenRecoveryPolicy,
        envelope_builder: SuccessorEnvelopeBuilder,
        terminal_condition: TerminalCondition,
        certificate_lifetime_seconds: float,
        sensor_version: str = "software-unbound-sensor",
        tracking_version: str = "software-unbound-tracking",
        sensor_contract_hash: str = "software-unbound-sensor-contract",
        tracking_contract_hash: str = "software-unbound-tracking-contract",
    ) -> None:
        self.policy = policy
        self.envelope_builder = envelope_builder
        self.terminal_condition = terminal_condition
        self.certificate_lifetime_seconds = certificate_lifetime_seconds
        self.sensor_version = sensor_version
        self.tracking_version = tracking_version
        self.sensor_contract_hash = sensor_contract_hash
        self.tracking_contract_hash = tracking_contract_hash
        self.last_failure: RecoveryFailureWitness | None = None

    def verify(
        self,
        corridor: ReturnCorridor,
        geometry,
        timestamp: float | None = None,
    ) -> RecoveryVerificationResult:
        now = monotonic() if timestamp is None else timestamp
        self.last_failure = None
        certificates: dict[int, RecoveryCellCertificate] = {}
        for cell in corridor.cells:
            result = self._verify_cell(cell, corridor, geometry, now, certificates)
            if result is None:
                predicate = self.last_failure.failed_predicate if self.last_failure else "cell-proof-failed"
                return RecoveryVerificationResult(False, {}, cell.cell_id, predicate, self.last_failure)
            certificates[cell.cell_id] = result
        if not corridor.install_recovery_certificates(certificates):
            return RecoveryVerificationResult(False, {}, None, "certificate-install-failed", self.last_failure)
        return RecoveryVerificationResult(True, certificates, None, "verified")

    def _verify_cell(
        self,
        cell: CorridorCell,
        corridor: ReturnCorridor,
        geometry,
        timestamp: float,
        lower_certificates: dict[int, RecoveryCellCertificate],
    ) -> RecoveryCellCertificate | None:
        tolerance = self.policy.config.numerical_tolerance
        target = corridor.lower_level_cell(cell.cell_id)
        action_interval = self.policy.action_interval_for_cell(cell, target)
        if any(
            action_interval.low[index] < -self.policy.config.maximum_acceleration[index] - tolerance
            or action_interval.high[index] > self.policy.config.maximum_acceleration[index] + tolerance
            for index in range(3)
        ):
            return self._fail(
                cell,
                corridor,
                geometry,
                "actuator-inclusion",
                repr(action_interval),
                max(self.policy.config.maximum_acceleration),
                max(max(abs(value) for value in action_interval.low), max(abs(value) for value in action_interval.high)),
            )
        envelope = self.envelope_builder.propagate_interval_state(
            cell.state_bounds.position,
            cell.state_bounds.velocity,
            cell.state_bounds.energy,
            action_interval,
            geometry.version,
            corridor.version,
        )
        speed = envelope.velocity.max_abs()
        horizontal_speed = hypot(speed[0], speed[1])
        stopping_margin = round_up(
            round_up(horizontal_speed * self.policy.config.update_latency)
            + round_up(
                horizontal_speed
                * horizontal_speed
                / (2.0 * self.policy.config.braking_deceleration)
            )
            + self.policy.config.geometry_margin
        )
        if not geometry.box_is_verified_free(
            envelope.position.horizontal_box().expanded(stopping_margin)
        ):
            return self._fail(
                cell,
                corridor,
                geometry,
                "stopping-tube-free",
                repr(envelope.position.horizontal_box().expanded(stopping_margin)),
                0.0,
                -stopping_margin,
            )
        if max(speed[0], speed[1]) > cell.maximum_speed + tolerance:
            return self._fail(
                cell,
                corridor,
                geometry,
                "successor-speed",
                repr(envelope.velocity),
                cell.maximum_speed,
                max(speed[0], speed[1]),
            )
        if envelope.energy_low + tolerance < self.policy.config.terminal_energy + self.policy.config.energy_reserve:
            return self._fail(
                cell,
                corridor,
                geometry,
                "energy-floor",
                f"[{envelope.energy_low},{envelope.energy_high}]",
                self.policy.config.terminal_energy + self.policy.config.energy_reserve,
                envelope.energy_low,
            )
        if cell.cell_id == 0:
            successor_state = StateCellBounds(
                envelope.position,
                envelope.velocity,
                Interval(envelope.energy_low, envelope.energy_high),
            )
            if not self.terminal_condition.contains_state_cell(successor_state, tolerance):
                return self._fail(
                    cell,
                    corridor,
                    geometry,
                    "terminal-condition",
                    repr(successor_state),
                    self.terminal_condition.minimum_energy,
                    successor_state.energy.low,
                )
            successor_ids: tuple[int, ...] = ()
            predecessors: tuple[int, ...] = ()
            progress = "terminal"
        else:
            lower_cells = [candidate for candidate in corridor.cells if candidate.cell_id < cell.cell_id]
            containing = [
                candidate
                for candidate in lower_cells
                if self._state_bounds_contain_envelope(candidate.state_bounds, envelope, tolerance)
            ]
            if not containing:
                return self._fail(
                    cell,
                    corridor,
                    geometry,
                    "one-step-lower-level",
                    repr(envelope),
                    float(cell.cell_id - 1),
                    float(cell.cell_id),
                )
            successor_ids = tuple(candidate.cell_id for candidate in containing)
            predecessors = successor_ids
            progress = "one-step-lower-level"
        transit_cost_upper = self.envelope_builder.energy.cost_upper(
            action_interval,
            cell.state_bounds.velocity,
        )
        proof_metadata = ProofMetadata(
            f"recovery-cell-{corridor.version}-{cell.cell_id}",
            corridor.certificate_epoch,
            self.sensor_version,
            self.envelope_builder.dynamics.version,
            self.tracking_version,
            self.envelope_builder.energy.version,
            self.terminal_condition.parameter_version,
            geometry.version,
            corridor.version,
            self.policy.config.parameter_version,
            timestamp,
            timestamp + self.certificate_lifetime_seconds,
            (
                self.sensor_contract_hash,
                self.envelope_builder.dynamics.contract_hash,
                self.tracking_contract_hash,
                self.envelope_builder.energy.contract_hash,
                self.terminal_condition.contract_hash,
                geometry.certificate_digest(),
            )
            + tuple(
                lower_certificates[identifier].certificate_hash
                for identifier in successor_ids
                if identifier in lower_certificates
            ),
        )
        payload = {
            "cell_id": cell.cell_id,
            "level": cell.cell_id,
            "state_bounds": repr(cell.state_bounds),
            "predecessors": predecessors,
            "successors": successor_ids,
            "kappa_version": self.policy.config.parameter_version,
            "dynamics_version": self.envelope_builder.dynamics.version,
            "geometry_version": geometry.version,
            "corridor_version": corridor.version,
            "energy_version": self.envelope_builder.energy.version,
            "action_interval": repr(action_interval),
            "successor": repr(envelope),
            "progress": progress,
            "transit_cost_upper": transit_cost_upper,
            "valid_from": timestamp,
            "valid_until": timestamp + self.certificate_lifetime_seconds,
            "proof_metadata": proof_metadata,
        }
        return RecoveryCellCertificate(
            cell.cell_id,
            cell.cell_id,
            cell.state_bounds,
            predecessors,
            successor_ids,
            self.policy.config.parameter_version,
            self.envelope_builder.dynamics.version,
            geometry.version,
            corridor.version,
            self.envelope_builder.energy.version,
            action_interval,
            envelope,
            progress,
            transit_cost_upper,
            timestamp,
            timestamp + self.certificate_lifetime_seconds,
            make_recovery_hash_payload(payload),
            proof_metadata,
        )

    def _fail(
        self,
        cell: CorridorCell,
        corridor: ReturnCorridor,
        geometry,
        predicate: str,
        interval_residual: str,
        required_margin: float | None,
        actual_margin: float | None,
    ) -> None:
        self.last_failure = RecoveryFailureWitness(
            cell.cell_id,
            predicate,
            interval_residual,
            (
                ("sensor", self.sensor_version),
                ("dynamics", self.envelope_builder.dynamics.version),
                ("tracking", self.tracking_version),
                ("energy", self.envelope_builder.energy.version),
                ("terminal", self.terminal_condition.parameter_version),
                ("geometry", str(geometry.version)),
                ("corridor", str(corridor.version)),
                ("kappa", self.policy.config.parameter_version),
            ),
            required_margin,
            actual_margin,
        )
        return None

    @staticmethod
    def _state_bounds_contain_envelope(
        bounds: StateCellBounds,
        envelope: SuccessorEnvelope,
        tolerance: float,
    ) -> bool:
        return (
            bounds.position.contains_box(envelope.position, tolerance)
            and bounds.velocity.contains_box(envelope.velocity, tolerance)
            and bounds.energy.contains( envelope.energy_low, tolerance)
            and bounds.energy.contains(envelope.energy_high, tolerance)
        )
