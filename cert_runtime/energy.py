from __future__ import annotations

from dataclasses import dataclass
from time import monotonic

from .certificates import ProofMetadata, RecoveryEnergyCertificate, make_energy_hash_payload
from .corridor import ReturnCorridor
from .envelope import EnergyBounds
from .interval import round_down, round_up


@dataclass(frozen=True)
class EnergySolveResult:
    verified: bool
    status: str
    certificates: dict[int, RecoveryEnergyCertificate]
    failed_cell_id: int | None = None


class RecoveryEnergySolver:
    """Outward-rounded T4a backward recursion and E3 residual verifier."""

    def __init__(self, energy_bounds: EnergyBounds, certificate_lifetime_seconds: float) -> None:
        self.energy_bounds = energy_bounds
        self.certificate_lifetime_seconds = certificate_lifetime_seconds

    def solve(
        self,
        corridor: ReturnCorridor,
        timestamp: float | None = None,
    ) -> EnergySolveResult:
        if not self.energy_bounds.calibration_complete:
            return EnergySolveResult(False, "blocked-by-calibration", {})
        now = monotonic() if timestamp is None else timestamp
        certificates: dict[int, RecoveryEnergyCertificate] = {}
        for cell in sorted(corridor.cells, key=lambda item: item.cell_id):
            recovery = cell.recovery_certificate
            if recovery is None:
                return EnergySolveResult(False, "missing-recovery-certificate", {}, cell.cell_id)
            if cell.cell_id == 0:
                value = 0.0
                one_step_cost = 0.0
                successors: tuple[int, ...] = ()
                residual = 0.0
            else:
                successors = recovery.successor_cell_ids
                if not successors or any(identifier not in certificates for identifier in successors):
                    return EnergySolveResult(False, "invalid-successor-level", {}, cell.cell_id)
                one_step_cost = recovery.transit_cost_upper
                successor_upper = max(
                    certificates[identifier].transit_energy_upper for identifier in successors
                )
                rhs = round_up(one_step_cost + successor_upper)
                value = round_up(rhs)
                residual = round_down(value - rhs)
                if residual < 0.0:
                    value = round_up(round_up(value))
                    residual = round_down(value - rhs)
            payload = {
                "cell_id": cell.cell_id,
                "level": cell.cell_id,
                "value": value,
                "one_step_cost": one_step_cost,
                "successors": successors,
                "residual": residual,
                "recovery_hash": recovery.certificate_hash,
                "energy_version": self.energy_bounds.version,
                "corridor_version": corridor.version,
                "valid_from": now,
                "valid_until": now + self.certificate_lifetime_seconds,
            }
            recovery_metadata = recovery.proof_metadata
            if recovery_metadata is None:
                return EnergySolveResult(False, "missing-proof-metadata", {}, cell.cell_id)
            proof_metadata = ProofMetadata(
                f"recovery-energy-{corridor.version}-{cell.cell_id}",
                corridor.certificate_epoch,
                recovery_metadata.sensor_version,
                recovery_metadata.dynamics_version,
                recovery_metadata.tracking_version,
                self.energy_bounds.version,
                recovery_metadata.terminal_version,
                recovery_metadata.geometry_version,
                corridor.version,
                recovery_metadata.kappa_version,
                now,
                now + self.certificate_lifetime_seconds,
                (recovery.certificate_hash,)
                + tuple(
                    certificates[identifier].certificate_hash
                    for identifier in successors
                ),
            )
            payload["proof_metadata"] = proof_metadata
            certificates[cell.cell_id] = RecoveryEnergyCertificate(
                cell.cell_id,
                cell.cell_id,
                value,
                one_step_cost,
                successors,
                residual,
                recovery.certificate_hash,
                self.energy_bounds.version,
                corridor.version,
                now,
                now + self.certificate_lifetime_seconds,
                make_energy_hash_payload(payload),
                proof_metadata,
            )
        if not self.verify_residuals(corridor, certificates):
            return EnergySolveResult(False, "e3-residual-failed", {}, None)
        if not corridor.install_energy_certificates(certificates):
            return EnergySolveResult(False, "energy-certificate-install-failed", {}, None)
        return EnergySolveResult(True, "verified", certificates)

    def verify_residuals(
        self,
        corridor: ReturnCorridor,
        certificates: dict[int, RecoveryEnergyCertificate],
    ) -> bool:
        for cell in corridor.cells:
            certificate = certificates.get(cell.cell_id)
            recovery = cell.recovery_certificate
            if certificate is None or recovery is None:
                return False
            if certificate.certificate_hash != certificate.expected_hash:
                return False
            if recovery.certificate_hash != recovery.expected_hash:
                return False
            if cell.cell_id == 0:
                if certificate.transit_energy_upper < 0.0:
                    return False
                continue
            if not recovery.successor_cell_ids:
                return False
            successor_upper = max(
                certificates[identifier].transit_energy_upper
                for identifier in recovery.successor_cell_ids
            )
            rhs = round_up(recovery.transit_cost_upper + successor_upper)
            residual = round_down(certificate.transit_energy_upper - rhs)
            if residual < 0.0:
                return False
        return True
