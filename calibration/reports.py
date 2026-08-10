from __future__ import annotations

from dataclasses import dataclass

from .schema import ConfidenceSemantics, OperatingDomain


@dataclass(frozen=True)
class ChannelCalibrationResult:
    channel: str
    sample_count: int
    maximum_observed_residual: float
    selected_bound: float
    validation_exceedances: int
    validation_sample_count: int
    confidence_semantics: ConfidenceSemantics


@dataclass(frozen=True)
class CalibrationReportBase:
    evidence_id: str
    contract_version: str
    sample_count: int
    train_count: int
    calibration_count: int
    validation_count: int
    operating_domain: OperatingDomain
    channels: tuple[ChannelCalibrationResult, ...]
    confidence_semantics: ConfidenceSemantics
    valid: bool
    physical_status: str
    unresolved_notes: tuple[str, ...]
    evidence_hash: str


@dataclass(frozen=True)
class SensorCalibrationReport(CalibrationReportBase):
    pass


@dataclass(frozen=True)
class TrackingCalibrationReport(CalibrationReportBase):
    alignment_tolerance_seconds: float = 0.0
