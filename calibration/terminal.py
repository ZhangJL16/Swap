from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .schema import EvidenceMetadata, evidence_hash


@dataclass(frozen=True)
class TerminalCalibrationContract:
    metadata: EvidenceMetadata
    version: str
    horizontal_position: tuple[float, float, float, float]
    altitude: tuple[float, float]
    velocity_low: tuple[float, float, float]
    velocity_high: tuple[float, float, float]
    minimum_energy: float
    continuation_evidence: tuple[tuple[str, str], ...]
    contract_hash: str

    def __post_init__(self) -> None:
        if not self.version or not self.contract_hash or self.minimum_energy < 0.0:
            raise ValueError("invalid terminal calibration contract")
        modes = dict(self.continuation_evidence)
        allowed = {"hover", "descent", "docking", "charging_handoff"}
        if not modes or any(mode not in allowed or not evidence for mode, evidence in modes.items()):
            raise ValueError("terminal continuation claims require explicit evidence")
        if self.contract_hash != self.expected_hash:
            raise ValueError("terminal calibration hash mismatch")

    @property
    def expected_hash(self) -> str:
        return evidence_hash(
            {
                "metadata": self.metadata.canonical_payload(),
                "version": self.version,
                "horizontal": self.horizontal_position,
                "altitude": self.altitude,
                "velocity_low": self.velocity_low,
                "velocity_high": self.velocity_high,
                "minimum_energy": self.minimum_energy,
                "continuation": self.continuation_evidence,
            }
        )

    @property
    def status(self) -> str:
        return self.metadata.physical_status

    def is_applicable(self, timestamp: float, point: Mapping[str, float | str], device: str, firmware: str) -> bool:
        return self.contract_hash == self.expected_hash and self.metadata.is_applicable(timestamp, point, device, firmware)

    def charge_admissible(self, mode: str) -> bool:
        return mode in dict(self.continuation_evidence)

    def to_runtime_condition(self):
        from cert_runtime.certificates import TerminalCondition
        from cert_runtime.interval import Interval
        from cert_runtime.types import AABB2, Interval3

        modes = dict(self.continuation_evidence)
        return TerminalCondition(
            AABB2(*self.horizontal_position),
            Interval(*self.altitude),
            Interval3(self.velocity_low, self.velocity_high),
            self.minimum_energy,
            "hover" in modes,
            "descent" in modes,
            "docking" in modes,
            self.version,
            "charging_handoff" in modes,
            self.continuation_evidence,
            self.contract_hash,
            self.status,
        )


def build_terminal_contract(
    metadata: EvidenceMetadata,
    version: str,
    *,
    horizontal_position: tuple[float, float, float, float],
    altitude: tuple[float, float],
    velocity_low: tuple[float, float, float],
    velocity_high: tuple[float, float, float],
    minimum_energy: float,
    continuation_evidence: tuple[tuple[str, str], ...],
) -> TerminalCalibrationContract:
    payload = {
        "metadata": metadata.canonical_payload(),
        "version": version,
        "horizontal": horizontal_position,
        "altitude": altitude,
        "velocity_low": velocity_low,
        "velocity_high": velocity_high,
        "minimum_energy": minimum_energy,
        "continuation": continuation_evidence,
    }
    return TerminalCalibrationContract(
        metadata,
        version,
        horizontal_position,
        altitude,
        velocity_low,
        velocity_high,
        minimum_energy,
        continuation_evidence,
        evidence_hash(payload),
    )
