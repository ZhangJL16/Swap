from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from json import dumps
from math import isfinite
from typing import Mapping


class ConfidenceSemantics(str, Enum):
    EMPIRICAL_QUANTILE = "empirical-quantile"
    POINTWISE_CONFIDENCE = "pointwise-confidence"
    SIMULTANEOUS_CONFIDENCE = "simultaneous-confidence"
    DETERMINISTIC_ENGINEERING = "deterministic-engineering-bound"


class DataSplit(str, Enum):
    TRAIN = "train"
    CALIBRATION = "calibration"
    VALIDATION = "validation"


class SourceKind(str, Enum):
    REAL = "real"
    SYNTHETIC = "synthetic"
    REPLAY = "replay"


class OutlierPolicy(str, Enum):
    RETAIN_ALL = "retain-all-valid-measurements"
    REJECT_INSTRUMENT_FAULTS = "reject-only-documented-instrument-faults"
    WINSORIZE_EXPLORATORY = "winsorize-exploratory-only"


def evidence_hash(payload: object) -> str:
    encoded = dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class OperatingDomain:
    speed_range: tuple[float, float]
    acceleration_range: tuple[float, float]
    payload_range: tuple[float, float]
    temperature_range: tuple[float, float]
    voltage_range: tuple[float, float]
    flight_modes: tuple[str, ...]

    def __post_init__(self) -> None:
        ranges = (
            self.speed_range,
            self.acceleration_range,
            self.payload_range,
            self.temperature_range,
            self.voltage_range,
        )
        if any(
            not all(isfinite(value) for value in bounds) or bounds[0] > bounds[1]
            for bounds in ranges
        ):
            raise ValueError("invalid calibration operating domain")
        if not self.flight_modes or any(not mode for mode in self.flight_modes):
            raise ValueError("at least one nonempty flight mode is required")

    def contains(self, point: Mapping[str, float | str]) -> bool:
        numeric = (
            ("speed", self.speed_range),
            ("acceleration", self.acceleration_range),
            ("payload", self.payload_range),
            ("temperature", self.temperature_range),
            ("voltage", self.voltage_range),
        )
        for name, bounds in numeric:
            value = point.get(name)
            if value is None or not isinstance(value, (int, float)) or not isfinite(float(value)):
                return False
            if not bounds[0] <= float(value) <= bounds[1]:
                return False
        mode = point.get("flight_mode")
        return isinstance(mode, str) and mode in self.flight_modes


@dataclass(frozen=True)
class EvidenceMetadata:
    evidence_id: str
    data_start_time: float
    data_end_time: float
    device_version: str
    firmware_version: str
    estimation_method: str
    applicable_domain: OperatingDomain
    confidence_semantics: ConfidenceSemantics
    source_kind: SourceKind
    expires_at: float
    recalibration_rule: str
    evidence_digest: str
    confidence_delta: float | None = None
    simultaneous_family_size: int | None = None

    def __post_init__(self) -> None:
        required = (
            self.evidence_id,
            self.device_version,
            self.firmware_version,
            self.estimation_method,
            self.recalibration_rule,
            self.evidence_digest,
        )
        if any(not value for value in required):
            raise ValueError("calibration evidence metadata cannot use empty identifiers")
        times = (self.data_start_time, self.data_end_time, self.expires_at)
        if not all(isfinite(value) for value in times):
            raise ValueError("calibration times must be finite")
        if self.data_start_time > self.data_end_time or self.expires_at <= self.data_end_time:
            raise ValueError("invalid calibration data or expiry interval")
        probabilistic = self.confidence_semantics in {
            ConfidenceSemantics.POINTWISE_CONFIDENCE,
            ConfidenceSemantics.SIMULTANEOUS_CONFIDENCE,
        }
        if probabilistic and (
            self.confidence_delta is None or not 0.0 < self.confidence_delta < 1.0
        ):
            raise ValueError("probabilistic confidence requires delta in (0,1)")
        if self.confidence_semantics == ConfidenceSemantics.SIMULTANEOUS_CONFIDENCE:
            if self.simultaneous_family_size is None or self.simultaneous_family_size <= 0:
                raise ValueError("simultaneous confidence requires a positive family size")

    @property
    def physical_status(self) -> str:
        return "implemented" if self.source_kind == SourceKind.REAL else "blocked-by-calibration"

    def is_applicable(
        self,
        timestamp: float,
        operating_point: Mapping[str, float | str],
        device_version: str,
        firmware_version: str,
    ) -> bool:
        return (
            isfinite(timestamp)
            and timestamp <= self.expires_at
            and device_version == self.device_version
            and firmware_version == self.firmware_version
            and self.applicable_domain.contains(operating_point)
        )

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RawCalibrationRecord:
    record_id: str
    timestamp: float
    channel: str
    observed: float
    reference: float
    split: DataSplit
    device_version: str
    firmware_version: str
    operating_point: tuple[tuple[str, float | str], ...]
    valid_measurement: bool = True
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id or not self.channel or not self.device_version or not self.firmware_version:
            raise ValueError("raw calibration records require identifiers and versions")
        if not all(isfinite(value) for value in (self.timestamp, self.observed, self.reference)):
            raise ValueError("raw calibration record contains nonfinite data")
        if not self.valid_measurement and not self.invalid_reason:
            raise ValueError("invalid measurements require a documented reason")

    @property
    def absolute_residual(self) -> float:
        return abs(self.observed - self.reference)

    @property
    def operating_domain_point(self) -> dict[str, float | str]:
        return dict(self.operating_point)
