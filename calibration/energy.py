from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from .reports import CalibrationReportBase, ChannelCalibrationResult
from .schema import DataSplit, EvidenceMetadata, evidence_hash
from .tracking import Vec3


@dataclass(frozen=True)
class EnergySample:
    sample_id: str
    start_timestamp: float
    end_timestamp: float
    voltage_start: float
    voltage_end: float
    current_start: float
    current_end: float
    measured_energy: float
    velocity: Vec3
    action: Vec3
    communication_active: bool
    compute_load: float
    temperature: float
    payload: float
    split: DataSplit

    def __post_init__(self) -> None:
        values = (
            self.start_timestamp,
            self.end_timestamp,
            self.voltage_start,
            self.voltage_end,
            self.current_start,
            self.current_end,
            self.measured_energy,
            *self.velocity,
            *self.action,
            self.compute_load,
            self.temperature,
            self.payload,
        )
        if not self.sample_id or any(not isfinite(value) for value in values):
            raise ValueError("invalid energy sample")
        if self.end_timestamp <= self.start_timestamp or self.measured_energy < 0.0:
            raise ValueError("energy samples require positive duration and nonnegative energy")
        if not 0.0 <= self.compute_load <= 1.0:
            raise ValueError("compute load must be in [0,1]")

    @property
    def integrated_electrical_energy(self) -> float:
        duration = self.end_timestamp - self.start_timestamp
        mean_power = 0.5 * (
            self.voltage_start * self.current_start
            + self.voltage_end * self.current_end
        )
        return max(0.0, duration * mean_power)


@dataclass(frozen=True)
class EnergyCalibrationContract:
    metadata: EvidenceMetadata
    version: str
    avionics_cost: float
    hover_cost: float
    velocity_coefficients: Vec3
    action_coefficients: Vec3
    communication_cost: float
    computation_cost: float
    measurement_error: float
    underestimation_margin: float
    validation_passed: bool
    contract_hash: str

    def __post_init__(self) -> None:
        values = (
            self.avionics_cost,
            self.hover_cost,
            *self.velocity_coefficients,
            *self.action_coefficients,
            self.communication_cost,
            self.computation_cost,
            self.measurement_error,
            self.underestimation_margin,
        )
        if not self.version or not self.contract_hash or any(not isfinite(value) or value < 0.0 for value in values):
            raise ValueError("invalid energy calibration contract")
        if self.contract_hash != self.expected_hash:
            raise ValueError("energy calibration hash mismatch")

    @property
    def expected_hash(self) -> str:
        return evidence_hash(
            {
                "metadata": self.metadata.canonical_payload(),
                "version": self.version,
                "avionics": self.avionics_cost,
                "hover": self.hover_cost,
                "velocity": self.velocity_coefficients,
                "action": self.action_coefficients,
                "communication": self.communication_cost,
                "computation": self.computation_cost,
                "measurement_error": self.measurement_error,
                "underestimation_margin": self.underestimation_margin,
                "validation_passed": self.validation_passed,
            }
        )

    @property
    def status(self) -> str:
        return self.metadata.physical_status if self.validation_passed else "blocked-by-calibration"

    def is_applicable(self, timestamp: float, point: Mapping[str, float | str], device: str, firmware: str) -> bool:
        return self.validation_passed and self.contract_hash == self.expected_hash and self.metadata.is_applicable(timestamp, point, device, firmware)

    def to_runtime_bounds(self):
        from cert_runtime.envelope import EnergyBounds

        return EnergyBounds(
            self.avionics_cost + self.hover_cost,
            self.action_coefficients,
            self.communication_cost + self.computation_cost,
            self.measurement_error + self.underestimation_margin,
            self.version,
            self.validation_passed,
            velocity_coefficients=self.velocity_coefficients,
            contract_hash=self.contract_hash,
            physical_status=self.status,
        )

    def upper_cost(self, velocity: Vec3, action: Vec3) -> float:
        return (
            self.avionics_cost
            + self.hover_cost
            + sum(self.velocity_coefficients[i] * abs(velocity[i]) for i in range(3))
            + sum(self.action_coefficients[i] * abs(action[i]) for i in range(3))
            + self.communication_cost
            + self.computation_cost
            + self.measurement_error
            + self.underestimation_margin
        )


@dataclass(frozen=True)
class EnergyCalibrationReport(CalibrationReportBase):
    maximum_underestimation: float
    electrical_alignment_error_max: float


def build_energy_contract(
    samples: Iterable[EnergySample],
    metadata: EvidenceMetadata,
    version: str,
    *,
    avionics_cost: float,
    hover_cost: float,
    velocity_coefficients: Vec3,
    action_coefficients: Vec3,
    communication_cost: float,
    computation_cost: float,
    measurement_error: float,
    underestimation_margin: float,
) -> tuple[EnergyCalibrationContract, EnergyCalibrationReport]:
    samples_tuple = tuple(samples)
    if not samples_tuple:
        raise ValueError("energy calibration requires samples")
    counts = {split: sum(sample.split == split for sample in samples_tuple) for split in DataSplit}
    if counts[DataSplit.CALIBRATION] == 0 or counts[DataSplit.VALIDATION] == 0:
        raise ValueError("energy calibration and validation splits are required")
    provisional_fixed = avionics_cost + hover_cost + communication_cost + computation_cost + measurement_error + underestimation_margin
    validation = tuple(sample for sample in samples_tuple if sample.split == DataSplit.VALIDATION)
    provisional_underestimations = tuple(
        max(
            0.0,
            sample.measured_energy
            - (
                provisional_fixed
                + sum(velocity_coefficients[i] * abs(sample.velocity[i]) for i in range(3))
                + sum(action_coefficients[i] * abs(sample.action[i]) for i in range(3))
            ),
        )
        for sample in validation
    )
    validation_passed = not any(value > 0.0 for value in provisional_underestimations)
    payload = {
        "metadata": metadata.canonical_payload(),
        "version": version,
        "avionics": avionics_cost,
        "hover": hover_cost,
        "velocity": velocity_coefficients,
        "action": action_coefficients,
        "communication": communication_cost,
        "computation": computation_cost,
        "measurement_error": measurement_error,
        "underestimation_margin": underestimation_margin,
        "validation_passed": validation_passed,
    }
    contract = EnergyCalibrationContract(
        metadata,
        version,
        avionics_cost,
        hover_cost,
        velocity_coefficients,
        action_coefficients,
        communication_cost,
        computation_cost,
        measurement_error,
        underestimation_margin,
        validation_passed,
        evidence_hash(payload),
    )
    underestimations = tuple(
        max(0.0, sample.measured_energy - contract.upper_cost(sample.velocity, sample.action))
        for sample in validation
    )
    alignment_errors = tuple(
        abs(sample.integrated_electrical_energy - sample.measured_energy) for sample in samples_tuple
    )
    channel = ChannelCalibrationResult(
        "one_step_energy",
        len(samples_tuple),
        max(sample.measured_energy for sample in samples_tuple),
        max(contract.upper_cost(sample.velocity, sample.action) for sample in samples_tuple),
        sum(value > 0.0 for value in underestimations),
        len(validation),
        metadata.confidence_semantics,
    )
    report = EnergyCalibrationReport(
        metadata.evidence_id,
        version,
        len(samples_tuple),
        counts[DataSplit.TRAIN],
        counts[DataSplit.CALIBRATION],
        counts[DataSplit.VALIDATION],
        metadata.applicable_domain,
        (channel,),
        metadata.confidence_semantics,
        validation_passed,
        metadata.physical_status,
        (() if metadata.physical_status == "implemented" else ("synthetic energy data",)),
        metadata.evidence_digest,
        max(underestimations, default=0.0),
        max(alignment_errors, default=0.0),
    )
    return contract, report
