from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import Iterable, Mapping

from .confidence import BoundEstimate
from .reports import ChannelCalibrationResult, SensorCalibrationReport
from .schema import (
    ConfidenceSemantics,
    DataSplit,
    EvidenceMetadata,
    OutlierPolicy,
    RawCalibrationRecord,
    evidence_hash,
)
from .validation import retained_records, validate_split_separation, validation_exceedances


SensorResidualSample = RawCalibrationRecord


@dataclass(frozen=True)
class SensorCalibrationContract:
    metadata: EvidenceMetadata
    version: str
    position_error: float
    attitude_error_radians: float
    range_error: float
    beam_half_angle_radians: float
    time_sync_error: float
    footprint_radius: float
    map_discretization_error: float
    maximum_range: float
    maximum_speed: float
    evidence_max_age_seconds: float
    minimum_free_observations: int
    calibration_version: str
    validation_passed: bool
    contract_hash: str

    def __post_init__(self) -> None:
        numeric = (
            self.position_error,
            self.attitude_error_radians,
            self.range_error,
            self.beam_half_angle_radians,
            self.time_sync_error,
            self.footprint_radius,
            self.map_discretization_error,
            self.maximum_range,
            self.maximum_speed,
            self.evidence_max_age_seconds,
        )
        if not self.version or not self.calibration_version or not self.contract_hash:
            raise ValueError("sensor contract requires versions and evidence hash")
        if any(not isfinite(value) or value < 0.0 for value in numeric):
            raise ValueError("sensor bounds must be finite and nonnegative")
        if self.maximum_range <= 0.0 or self.minimum_free_observations <= 0:
            raise ValueError("invalid sensor range or free-evidence count")
        if self.attitude_error_radians + self.beam_half_angle_radians >= pi:
            raise ValueError("combined sensor angular uncertainty must be below pi")
        if self.contract_hash != self.expected_hash:
            raise ValueError("sensor calibration hash mismatch")

    @property
    def expected_hash(self) -> str:
        return evidence_hash(
            {
                "metadata": self.metadata.canonical_payload(),
                "version": self.version,
                "theta": self.theta,
                "calibration_version": self.calibration_version,
                "validation_passed": self.validation_passed,
            }
        )

    @property
    def theta(self) -> tuple[object, ...]:
        return (
            self.position_error,
            self.attitude_error_radians,
            self.range_error,
            self.beam_half_angle_radians,
            self.time_sync_error,
            self.footprint_radius,
            self.map_discretization_error,
            self.maximum_range,
            self.maximum_speed,
            self.evidence_max_age_seconds,
            self.minimum_free_observations,
        )

    @property
    def status(self) -> str:
        return self.metadata.physical_status if self.validation_passed else "blocked-by-calibration"

    def is_applicable(
        self,
        timestamp: float,
        operating_point: Mapping[str, float | str],
        device_version: str,
        firmware_version: str,
    ) -> bool:
        return self.validation_passed and self.contract_hash == self.expected_hash and self.metadata.is_applicable(
            timestamp, operating_point, device_version, firmware_version
        )


def build_sensor_contract(
    records: Iterable[SensorResidualSample],
    metadata: EvidenceMetadata,
    version: str,
    estimates: Mapping[str, BoundEstimate],
    *,
    beam_half_angle_radians: float,
    footprint_radius: float,
    map_discretization_error: float,
    maximum_range: float,
    maximum_speed: float,
    evidence_max_age_seconds: float,
    minimum_free_observations: int,
    outlier_policy: OutlierPolicy = OutlierPolicy.RETAIN_ALL,
) -> tuple[SensorCalibrationContract, SensorCalibrationReport]:
    retained = retained_records(records, outlier_policy)
    counts = validate_split_separation(retained)
    required = ("position", "attitude", "range", "time_sync")
    if any(channel not in estimates for channel in required):
        raise ValueError("sensor estimates must cover position, attitude, range, and time_sync")
    if metadata.confidence_semantics == ConfidenceSemantics.DETERMINISTIC_ENGINEERING and any(
        not estimates[channel].deterministic for channel in required
    ):
        raise ValueError("empirical or probabilistic estimates cannot be labeled deterministic")
    channel_results = []
    for channel in required:
        channel_records = tuple(record for record in retained if record.channel == channel)
        validation = tuple(record for record in channel_records if record.split == DataSplit.VALIDATION)
        if not channel_records or not validation:
            raise ValueError(f"sensor channel {channel} lacks independent validation data")
        estimate = estimates[channel]
        channel_results.append(
            ChannelCalibrationResult(
                channel,
                len(channel_records),
                max(record.absolute_residual for record in channel_records),
                estimate.value,
                validation_exceedances(validation, estimate.value),
                len(validation),
                estimate.semantics,
            )
        )
    theta = (
        estimates["position"].value,
        estimates["attitude"].value,
        estimates["range"].value,
        estimates["time_sync"].value,
        beam_half_angle_radians,
        footprint_radius,
        map_discretization_error,
        maximum_range,
        maximum_speed,
        evidence_max_age_seconds,
        minimum_free_observations,
    )
    validation_passed = all(result.validation_exceedances == 0 for result in channel_results)
    contract_payload = {
        "metadata": metadata.canonical_payload(),
        "version": version,
        "theta": (
            theta[0], theta[1], theta[2], theta[4], theta[3], theta[5], theta[6],
            theta[7], theta[8], theta[9], theta[10]
        ),
        "calibration_version": version,
        "validation_passed": validation_passed,
    }
    contract = SensorCalibrationContract(
        metadata,
        version,
        estimates["position"].value,
        estimates["attitude"].value,
        estimates["range"].value,
        beam_half_angle_radians,
        estimates["time_sync"].value,
        footprint_radius,
        map_discretization_error,
        maximum_range,
        maximum_speed,
        evidence_max_age_seconds,
        minimum_free_observations,
        version,
        validation_passed,
        evidence_hash(contract_payload),
    )
    report = SensorCalibrationReport(
        metadata.evidence_id,
        version,
        len(retained),
        counts[DataSplit.TRAIN],
        counts[DataSplit.CALIBRATION],
        counts[DataSplit.VALIDATION],
        metadata.applicable_domain,
        tuple(channel_results),
        metadata.confidence_semantics,
        validation_passed,
        metadata.physical_status,
        (() if metadata.physical_status == "implemented" else ("synthetic/replay evidence is not flight calibration",)),
        metadata.evidence_digest,
    )
    return contract, report
