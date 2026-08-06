from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from typing import Any

from .interval import Interval
from .types import AABB2, Interval3


def certificate_hash(payload: dict[str, Any]) -> str:
    encoded = dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ProofMetadata:
    object_id: str
    certificate_epoch: int
    sensor_version: str
    dynamics_version: str
    tracking_version: str
    energy_version: str
    terminal_version: str
    geometry_version: int
    corridor_version: int
    kappa_version: str
    creation_timestamp: float
    expires_at: float
    predecessor_certificates: tuple[str, ...]
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        required = (
            self.object_id,
            self.sensor_version,
            self.dynamics_version,
            self.tracking_version,
            self.energy_version,
            self.terminal_version,
            self.kappa_version,
        )
        if any(not value for value in required):
            raise ValueError("proof metadata requires all version identifiers")
        if self.certificate_epoch < 0 or self.creation_timestamp > self.expires_at:
            raise ValueError("invalid proof epoch or validity interval")

    @property
    def valid(self) -> bool:
        return self.invalidation_reason is None


@dataclass(frozen=True)
class StateCellBounds:
    position: Interval3
    velocity: Interval3
    energy: Interval


@dataclass(frozen=True)
class TerminalCondition:
    horizontal_position: AABB2
    altitude: Interval
    velocity: Interval3
    minimum_energy: float
    hover_continuation_admissible: bool
    descent_continuation_admissible: bool
    docking_continuation_admissible: bool
    parameter_version: str
    charging_handoff_admissible: bool = False
    continuation_evidence: tuple[tuple[str, str], ...] = ()
    contract_hash: str = "unversioned-terminal-contract"
    physical_status: str = "blocked-by-calibration"

    def __post_init__(self) -> None:
        if self.minimum_energy < 0.0:
            raise ValueError("terminal energy must be nonnegative")
        if not self.charge_admissible:
            raise ValueError("at least one terminal continuation mode is required")
        if not self.parameter_version or not self.contract_hash:
            raise ValueError("terminal-condition version and evidence hash are required")

    @property
    def charge_admissible(self) -> bool:
        return (
            self.hover_continuation_admissible
            or self.descent_continuation_admissible
            or self.docking_continuation_admissible
            or self.charging_handoff_admissible
        )

    def contains_state_cell(self, state: StateCellBounds, tolerance: float = 0.0) -> bool:
        return (
            self.horizontal_position.contains_box(state.position.horizontal_box(), tolerance)
            and self.altitude.contains_interval(state.position.components[2], tolerance)
            and self.velocity.contains_box(state.velocity, tolerance)
            and state.energy.low + tolerance >= self.minimum_energy
            and self.charge_admissible
        )


@dataclass(frozen=True)
class RecoveryCellCertificate:
    cell_id: int
    level: int
    state_bounds: StateCellBounds
    predecessor_cell_ids: tuple[int, ...]
    successor_cell_ids: tuple[int, ...]
    kappa_parameter_version: str
    dynamics_bound_version: str
    geometry_version: int
    corridor_version: int
    energy_bound_version: str
    action_interval: Interval3
    verified_successor_envelope: Any
    progress_result: str
    transit_cost_upper: float
    valid_from: float
    valid_until: float
    certificate_hash: str
    proof_metadata: ProofMetadata | None = None

    @property
    def expected_hash(self) -> str:
        return certificate_hash(
            {
                "cell_id": self.cell_id,
                "level": self.level,
                "state_bounds": repr(self.state_bounds),
                "predecessors": self.predecessor_cell_ids,
                "successors": self.successor_cell_ids,
                "kappa_version": self.kappa_parameter_version,
                "dynamics_version": self.dynamics_bound_version,
                "geometry_version": self.geometry_version,
                "corridor_version": self.corridor_version,
                "energy_version": self.energy_bound_version,
                "action_interval": repr(self.action_interval),
                "successor": repr(self.verified_successor_envelope),
                "progress": self.progress_result,
                "transit_cost_upper": self.transit_cost_upper,
                "valid_from": self.valid_from,
                "valid_until": self.valid_until,
                "proof_metadata": self.proof_metadata,
            }
        )

    def is_valid(
        self,
        geometry_version: int,
        corridor_version: int,
        kappa_parameter_version: str,
        dynamics_bound_version: str,
        energy_bound_version: str,
        timestamp: float,
    ) -> bool:
        return (
            self.geometry_version == geometry_version
            and self.corridor_version == corridor_version
            and self.kappa_parameter_version == kappa_parameter_version
            and self.dynamics_bound_version == dynamics_bound_version
            and self.energy_bound_version == energy_bound_version
            and self.valid_from <= timestamp <= self.valid_until
            and self.progress_result in {"terminal", "one-step-lower-level"}
            and self.proof_metadata is not None
            and self.proof_metadata.valid
            and self.certificate_hash == self.expected_hash
        )


@dataclass(frozen=True)
class RecoveryEnergyCertificate:
    cell_id: int
    level: int
    transit_energy_upper: float
    one_step_cost_upper: float
    successor_cell_ids: tuple[int, ...]
    residual_lower: float
    recovery_certificate_hash: str
    energy_bound_version: str
    corridor_version: int
    valid_from: float
    valid_until: float
    certificate_hash: str
    proof_metadata: ProofMetadata | None = None

    @property
    def expected_hash(self) -> str:
        return certificate_hash(
            {
                "cell_id": self.cell_id,
                "level": self.level,
                "value": self.transit_energy_upper,
                "one_step_cost": self.one_step_cost_upper,
                "successors": self.successor_cell_ids,
                "residual": self.residual_lower,
                "recovery_hash": self.recovery_certificate_hash,
                "energy_version": self.energy_bound_version,
                "corridor_version": self.corridor_version,
                "valid_from": self.valid_from,
                "valid_until": self.valid_until,
                "proof_metadata": self.proof_metadata,
            }
        )

    def is_valid(
        self,
        recovery_certificate_hash: str,
        energy_bound_version: str,
        corridor_version: int,
        timestamp: float,
    ) -> bool:
        return (
            self.recovery_certificate_hash == recovery_certificate_hash
            and self.energy_bound_version == energy_bound_version
            and self.corridor_version == corridor_version
            and self.residual_lower >= 0.0
            and self.valid_from <= timestamp <= self.valid_until
            and self.proof_metadata is not None
            and self.proof_metadata.valid
            and self.certificate_hash == self.expected_hash
        )


def make_recovery_hash_payload(certificate_fields: dict[str, Any]) -> str:
    return certificate_hash(certificate_fields)


def make_energy_hash_payload(certificate_fields: dict[str, Any]) -> str:
    return certificate_hash(certificate_fields)
