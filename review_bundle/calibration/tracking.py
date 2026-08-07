from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from .reports import ChannelCalibrationResult, TrackingCalibrationReport
from .schema import DataSplit, EvidenceMetadata, OutlierPolicy, evidence_hash


Vec3 = tuple[float, float, float]


@dataclass(frozen=True)
class TrackingSample:
    sample_id: str
    command_timestamp: float
    publish_timestamp: float
    measurement_timestamp: float
    commanded_action: Vec3
    published_action: Vec3
    measured_action: Vec3
    split: DataSplit
    operating_point: tuple[tuple[str, float | str], ...]

    def __post_init__(self) -> None:
        values = (
            self.command_timestamp,
            self.publish_timestamp,
            self.measurement_timestamp,
            *self.commanded_action,
            *self.published_action,
            *self.measured_action,
        )
        if not self.sample_id or any(not isfinite(value) for value in values):
            raise ValueError("invalid tracking sample")
        if not self.command_timestamp <= self.publish_timestamp <= self.measurement_timestamp:
            raise ValueError("tracking timestamps are not aligned in causal order")

    @property
    def residual(self) -> Vec3:
        return tuple(abs(self.measured_action[i] - self.published_action[i]) for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class TrackingCalibrationContract:
    metadata: EvidenceMetadata
    version: str
    action_tracking_radius: Vec3
    alignment_tolerance_seconds: float
    validation_passed: bool
    contract_hash: str

    def __post_init__(self) -> None:
        values = (*self.action_tracking_radius, self.alignment_tolerance_seconds)
        if not self.version or not self.contract_hash or any(not isfinite(v) or v < 0.0 for v in values):
            raise ValueError("invalid tracking calibration contract")
        if self.contract_hash != self.expected_hash:
            raise ValueError("tracking calibration hash mismatch")

    @property
    def expected_hash(self) -> str:
        return evidence_hash(
            {
                "metadata": self.metadata.canonical_payload(),
                "version": self.version,
                "radius": self.action_tracking_radius,
                "alignment": self.alignment_tolerance_seconds,
                "validation_passed": self.validation_passed,
            }
        )

    @property
    def status(self) -> str:
        return self.metadata.physical_status if self.validation_passed else "blocked-by-calibration"

    def is_applicable(self, timestamp: float, point: Mapping[str, float | str], device: str, firmware: str) -> bool:
        return self.validation_passed and self.contract_hash == self.expected_hash and self.metadata.is_applicable(timestamp, point, device, firmware)


def build_tracking_contract(
    samples: Iterable[TrackingSample],
    metadata: EvidenceMetadata,
    version: str,
    bound: Vec3,
    alignment_tolerance_seconds: float,
    outlier_policy: OutlierPolicy = OutlierPolicy.RETAIN_ALL,
) -> tuple[TrackingCalibrationContract, TrackingCalibrationReport]:
    samples_tuple = tuple(samples)
    if outlier_policy == OutlierPolicy.WINSORIZE_EXPLORATORY:
        raise ValueError("winsorized tracking data cannot certify an outer bound")
    if not samples_tuple:
        raise ValueError("tracking calibration requires samples")
    split_counts = {split: sum(sample.split == split for sample in samples_tuple) for split in DataSplit}
    if split_counts[DataSplit.CALIBRATION] == 0 or split_counts[DataSplit.VALIDATION] == 0:
        raise ValueError("tracking calibration and validation splits are required")
    validation = tuple(sample for sample in samples_tuple if sample.split == DataSplit.VALIDATION)
    channels = tuple(
        ChannelCalibrationResult(
            f"action_{axis}",
            len(samples_tuple),
            max(sample.residual[axis] for sample in samples_tuple),
            bound[axis],
            sum(sample.residual[axis] > bound[axis] for sample in validation),
            len(validation),
            metadata.confidence_semantics,
        )
        for axis in range(3)
    )
    validation_passed = all(channel.validation_exceedances == 0 for channel in channels)
    payload = {
        "metadata": metadata.canonical_payload(),
        "version": version,
        "radius": bound,
        "alignment": alignment_tolerance_seconds,
        "validation_passed": validation_passed,
    }
    contract = TrackingCalibrationContract(
        metadata, version, bound, alignment_tolerance_seconds, validation_passed, evidence_hash(payload)
    )
    report = TrackingCalibrationReport(
        metadata.evidence_id,
        version,
        len(samples_tuple),
        split_counts[DataSplit.TRAIN],
        split_counts[DataSplit.CALIBRATION],
        split_counts[DataSplit.VALIDATION],
        metadata.applicable_domain,
        channels,
        metadata.confidence_semantics,
        validation_passed,
        metadata.physical_status,
        (() if metadata.physical_status == "implemented" else ("synthetic tracking data",)),
        metadata.evidence_digest,
        alignment_tolerance_seconds,
    )
    return contract, report
