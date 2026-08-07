from __future__ import annotations

from dataclasses import asdict, dataclass
from time import monotonic, time
from typing import Mapping, Sequence

from calibration.dynamics import DynamicsCalibrationContract
from calibration.energy import EnergyCalibrationContract
from calibration.schema import evidence_hash
from calibration.sensor import SensorCalibrationContract
from calibration.terminal import TerminalCalibrationContract
from calibration.tracking import TrackingCalibrationContract

from .certificates import ProofMetadata
from .corridor import CorridorCell, ReturnCorridor
from .energy import RecoveryEnergySolver
from .geometry import RollingLocalGeometry
from .recovery import CorridorRecoveryVerifier, FrozenRecoveryPolicy
from .state import CertificateState
from .zonotope import ZonotopeCertificate, ZonotopeConstructor


@dataclass(frozen=True)
class CalibrationBundle:
    sensor: SensorCalibrationContract
    dynamics: DynamicsCalibrationContract
    tracking: TrackingCalibrationContract
    energy: EnergyCalibrationContract
    terminal: TerminalCalibrationContract

    @property
    def versions(self) -> tuple[tuple[str, str], ...]:
        return (
            ("sensor", self.sensor.version),
            ("dynamics", self.dynamics.version),
            ("tracking", self.tracking.version),
            ("energy", self.energy.version),
            ("terminal", self.terminal.version),
        )

    @property
    def fingerprints(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (f"{name}_hash", getattr(self, name).contract_hash)
            for name in ("sensor", "dynamics", "tracking", "energy", "terminal")
        )

    @property
    def physical_status(self) -> str:
        return (
            "implemented"
            if all(contract.status == "implemented" for contract in (
                self.sensor, self.dynamics, self.tracking, self.energy, self.terminal
            ))
            else "blocked-by-calibration"
        )

    def validate(
        self,
        timestamp: float,
        operating_point: Mapping[str, float | str],
        device_version: str,
        firmware_version: str,
        *,
        allow_synthetic: bool,
    ) -> tuple[bool, str]:
        contracts = (self.sensor, self.tracking, self.dynamics, self.energy, self.terminal)
        if self.dynamics.tracking_version != self.tracking.version:
            return False, "tracking-dynamics-version-mismatch"
        for contract in contracts:
            if contract.status != "implemented" and not allow_synthetic:
                return False, f"{type(contract).__name__}-blocked-by-calibration"
            if not contract.is_applicable(timestamp, operating_point, device_version, firmware_version):
                return False, f"{type(contract).__name__}-expired-or-out-of-domain"
        return True, "valid"


@dataclass(frozen=True)
class FailureWitness:
    failed_cell: int | None
    failed_predicate: str
    interval_residual: str
    involved_versions: tuple[tuple[str, str], ...]
    required_margin: float | None
    actual_margin: float | None


@dataclass(frozen=True)
class ProofManifestEntry:
    object_id: str
    object_type: str
    certificate_epoch: int
    versions: tuple[tuple[str, str], ...]
    creation_timestamp: float
    expiry: float
    predecessor_certificates: tuple[str, ...]
    proof_hash: str
    status: str
    invalidation_reason: str | None


@dataclass(frozen=True)
class ProofManifest:
    manifest_id: str
    certificate_epoch: int
    entries: tuple[ProofManifestEntry, ...]
    dependency_edges: tuple[tuple[str, str], ...]
    physical_status: str
    complete: bool
    manifest_hash: str

    @property
    def expected_hash(self) -> str:
        return evidence_hash(
            {
                "manifest_id": self.manifest_id,
                "certificate_epoch": self.certificate_epoch,
                "entries": self.entries,
                "edges": self.dependency_edges,
                "physical_status": self.physical_status,
                "complete": self.complete,
            }
        )

    def verify_integrity(self) -> bool:
        if not self.complete or self.manifest_hash != self.expected_hash:
            return False
        object_ids = tuple(entry.object_id for entry in self.entries)
        if len(object_ids) != len(set(object_ids)):
            return False
        proof_hashes = {entry.proof_hash for entry in self.entries}
        return all(
            predecessor in proof_hashes
            for entry in self.entries
            for predecessor in entry.predecessor_certificates
        ) and all(left in proof_hashes and right in proof_hashes for left, right in self.dependency_edges)


@dataclass(frozen=True)
class CorridorClosureResult:
    closed: bool
    status: str
    manifest: ProofManifest | None
    failure_witness: FailureWitness | None
    zonotope_certificate: ZonotopeCertificate | None


class SingleCorridorClosurePipeline:
    """Fail-closed T4a closure for one fixed proof-carrying AABB chain."""

    def __init__(
        self,
        calibration: CalibrationBundle,
        recovery_policy: FrozenRecoveryPolicy,
        recovery_verifier: CorridorRecoveryVerifier,
        energy_solver: RecoveryEnergySolver,
        zonotope_constructor: ZonotopeConstructor,
    ) -> None:
        self.calibration = calibration
        self.recovery_policy = recovery_policy
        self.recovery_verifier = recovery_verifier
        self.energy_solver = energy_solver
        self.zonotope_constructor = zonotope_constructor
        self.last_stage_timings: dict[str, float] = {}

    def close(
        self,
        state: CertificateState,
        geometry: RollingLocalGeometry,
        corridor: ReturnCorridor,
        cells: Sequence[CorridorCell],
        operating_point: Mapping[str, float | str],
        device_version: str,
        firmware_version: str,
        timestamp: float | None = None,
        *,
        allow_synthetic: bool = False,
    ) -> CorridorClosureResult:
        total_started = monotonic()
        timings: dict[str, float] = {}
        now = time() if timestamp is None else timestamp
        valid, reason = self.calibration.validate(
            now,
            operating_point,
            device_version,
            firmware_version,
            allow_synthetic=allow_synthetic,
        )
        if not valid:
            return self._failure(None, reason, "calibration-bundle", corridor, geometry)
        if geometry.active_calibration_version != self.calibration.sensor.version:
            return self._failure(
                None,
                "sensor-geometry-version-mismatch",
                str(geometry.active_calibration_version),
                corridor,
                geometry,
            )
        if self.zonotope_constructor.envelope_builder.dynamics.version != self.calibration.dynamics.version:
            return self._failure(None, "dynamics-version-mismatch", "runtime bounds", corridor, geometry)
        if self.zonotope_constructor.envelope_builder.energy.version != self.calibration.energy.version:
            return self._failure(None, "energy-version-mismatch", "runtime bounds", corridor, geometry)
        if (
            "kappa" in state.bound_versions
            and self.recovery_policy.config.parameter_version != state.bound_versions["kappa"]
        ):
            return self._failure(None, "kappa-version-mismatch", "state bound versions", corridor, geometry)
        if self.recovery_verifier.terminal_condition.parameter_version != self.calibration.terminal.version:
            return self._failure(None, "terminal-version-mismatch", "terminal condition", corridor, geometry)
        corridor_started = monotonic()
        if not corridor.create(list(cells), geometry):
            return self._failure(None, "corridor-geometry-or-overlap", "AABB chain", corridor, geometry)
        state.bound_versions = dict(self.calibration.versions + self.calibration.fingerprints) | {
            "kappa": self.recovery_policy.config.parameter_version,
        }
        recovery_result = self.recovery_verifier.verify(corridor, geometry, now)
        timings["T_corridor"] = monotonic() - corridor_started
        if not recovery_result.verified:
            witness = recovery_result.witness
            return CorridorClosureResult(
                False,
                "corridor-recovery-proof-failed",
                None,
                FailureWitness(
                    witness.failed_cell_id if witness else recovery_result.failed_cell_id,
                    witness.failed_predicate if witness else recovery_result.reason,
                    witness.interval_residual if witness else "unavailable",
                    witness.involved_versions if witness else self._versions(corridor, geometry),
                    witness.required_margin if witness else None,
                    witness.actual_margin if witness else None,
                ),
                None,
            )
        energy_started = monotonic()
        energy_result = self.energy_solver.solve(corridor, now)
        if not energy_result.verified:
            return self._failure(
                energy_result.failed_cell_id,
                energy_result.status,
                "E3/backward recursion",
                corridor,
                geometry,
            )
        if not self.energy_solver.verify_residuals(corridor, energy_result.certificates):
            return self._failure(None, "e3-residual-failed", "energy certificates", corridor, geometry)
        timings["T_energy"] = monotonic() - energy_started
        set_started = monotonic()
        recovery = self.recovery_policy.certified_action(
            state,
            self.zonotope_constructor.envelope_builder.dynamics.version,
            self.zonotope_constructor.envelope_builder.energy.version,
            now,
        )
        if not recovery.certified:
            return self._failure(None, recovery.reason, "runtime recovery authorization", corridor, geometry)
        zonotope = self.zonotope_constructor.construct(state, recovery, now)
        timings["T_set"] = monotonic() - set_started
        if not zonotope.verified:
            return self._failure(None, zonotope.reason, "generator construction", corridor, geometry, zonotope)
        manifest = self._manifest(corridor, geometry, zonotope)
        if not manifest.verify_integrity():
            return self._failure(None, "manifest-hash-mismatch", "proof manifest", corridor, geometry)
        status = (
            "software-verified"
            if self.calibration.physical_status == "implemented"
            else "conditionally-verified-blocked-by-calibration"
        )
        timings["T_total_closure"] = monotonic() - total_started
        self.last_stage_timings = timings
        return CorridorClosureResult(True, status, manifest, None, zonotope)

    def _manifest(
        self,
        corridor: ReturnCorridor,
        geometry: RollingLocalGeometry,
        zonotope: ZonotopeCertificate,
    ) -> ProofManifest:
        entries: list[ProofManifestEntry] = []
        edges: list[tuple[str, str]] = []
        calibration_hashes = {
            "sensor": self.calibration.sensor.contract_hash,
            "dynamics": self.calibration.dynamics.contract_hash,
            "tracking": self.calibration.tracking.contract_hash,
            "energy": self.calibration.energy.contract_hash,
            "terminal": self.calibration.terminal.contract_hash,
        }
        for name, proof_hash in calibration_hashes.items():
            entries.append(
                ProofManifestEntry(
                    f"calibration-{name}-{dict(self.calibration.versions)[name]}",
                    "calibration-contract",
                    corridor.certificate_epoch,
                    self.calibration.versions,
                    0.0,
                    getattr(self.calibration, name).metadata.expires_at,
                    (),
                    proof_hash,
                    getattr(self.calibration, name).status,
                    None,
                )
            )
        geometry_id = f"geometry-{geometry.version}"
        entries.append(
            ProofManifestEntry(
                geometry_id,
                "proof-carrying-grid",
                corridor.certificate_epoch,
                self._versions(corridor, geometry),
                geometry.current_timestamp,
                self.calibration.sensor.metadata.expires_at,
                (calibration_hashes["sensor"],),
                geometry.certificate_digest(),
                self.calibration.sensor.status,
                None,
            )
        )
        for cell in corridor.cells:
            for object_type, certificate in (
                ("recovery-cell", cell.recovery_certificate),
                ("recovery-energy", cell.energy_certificate),
            ):
                if certificate is None or certificate.proof_metadata is None:
                    raise RuntimeError("manifest cannot omit a corridor proof object")
                metadata: ProofMetadata = certificate.proof_metadata
                entries.append(self._entry(object_type, metadata, certificate.certificate_hash))
        if zonotope.proof_metadata is None or zonotope.complete_set_inclusion_hash is None:
            raise RuntimeError("manifest cannot omit zonotope proof metadata")
        entries.append(self._entry("zonotope", zonotope.proof_metadata, zonotope.complete_set_inclusion_hash))
        proof_hashes = {entry.proof_hash for entry in entries}
        edges = sorted({
            (predecessor, entry.proof_hash)
            for entry in entries
            for predecessor in entry.predecessor_certificates
            if predecessor in proof_hashes
        })
        payload = {
            "manifest_id": f"single-corridor-{corridor.version}-{corridor.certificate_epoch}",
            "certificate_epoch": corridor.certificate_epoch,
            "entries": tuple(entries),
            "edges": tuple(edges),
            "physical_status": self.calibration.physical_status,
            "complete": True,
        }
        return ProofManifest(
            payload["manifest_id"],
            corridor.certificate_epoch,
            tuple(entries),
            tuple(edges),
            self.calibration.physical_status,
            True,
            evidence_hash(payload),
        )

    def _entry(self, object_type: str, metadata: ProofMetadata, proof_hash: str) -> ProofManifestEntry:
        versions = (
            ("sensor", metadata.sensor_version),
            ("dynamics", metadata.dynamics_version),
            ("tracking", metadata.tracking_version),
            ("energy", metadata.energy_version),
            ("terminal", metadata.terminal_version),
            ("geometry", str(metadata.geometry_version)),
            ("corridor", str(metadata.corridor_version)),
            ("kappa", metadata.kappa_version),
        )
        return ProofManifestEntry(
            metadata.object_id,
            object_type,
            metadata.certificate_epoch,
            versions,
            metadata.creation_timestamp,
            metadata.expires_at,
            metadata.predecessor_certificates,
            proof_hash,
            "valid" if metadata.valid else "invalid",
            metadata.invalidation_reason,
        )

    def _failure(
        self,
        cell: int | None,
        predicate: str,
        residual: str,
        corridor: ReturnCorridor,
        geometry: RollingLocalGeometry,
        zonotope: ZonotopeCertificate | None = None,
    ) -> CorridorClosureResult:
        return CorridorClosureResult(
            False,
            "failed-closed",
            None,
            FailureWitness(cell, predicate, residual, self._versions(corridor, geometry), None, None),
            zonotope,
        )

    def _versions(self, corridor: ReturnCorridor, geometry: RollingLocalGeometry) -> tuple[tuple[str, str], ...]:
        return self.calibration.versions + (
            ("geometry", str(geometry.version)),
            ("corridor", str(corridor.version)),
            ("kappa", self.recovery_policy.config.parameter_version),
        )
