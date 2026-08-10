from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Mapping

from .reports import CalibrationReportBase, ChannelCalibrationResult
from .schema import DataSplit, EvidenceMetadata, evidence_hash
from .tracking import TrackingCalibrationContract, Vec3


@dataclass(frozen=True)
class DynamicsSample:
    sample_id: str
    state_timestamp: float
    next_state_timestamp: float
    position: Vec3
    velocity: Vec3
    next_position: Vec3
    next_velocity: Vec3
    commanded_action: Vec3
    published_action: Vec3
    measured_action: Vec3
    split: DataSplit
    operating_point: tuple[tuple[str, float | str], ...]

    def __post_init__(self) -> None:
        values = (
            self.state_timestamp,
            self.next_state_timestamp,
            *self.position,
            *self.velocity,
            *self.next_position,
            *self.next_velocity,
            *self.commanded_action,
            *self.published_action,
            *self.measured_action,
        )
        if not self.sample_id or any(not isfinite(value) for value in values):
            raise ValueError("invalid dynamics sample")
        if self.next_state_timestamp <= self.state_timestamp:
            raise ValueError("dynamics sample requires a positive time step")

    @property
    def duration(self) -> float:
        return self.next_state_timestamp - self.state_timestamp

    @property
    def position_residual(self) -> Vec3:
        from envs.certified_uav.dynamics import integrate_double_integrator

        predicted_position, _ = integrate_double_integrator(
            self.position,
            self.velocity,
            self.measured_action,
            self.duration,
        )
        return tuple(abs(self.next_position[i] - predicted_position[i]) for i in range(3))  # type: ignore[return-value]

    @property
    def velocity_residual(self) -> Vec3:
        from envs.certified_uav.dynamics import integrate_double_integrator

        _, predicted_velocity = integrate_double_integrator(
            self.position,
            self.velocity,
            self.measured_action,
            self.duration,
        )
        return tuple(abs(self.next_velocity[i] - predicted_velocity[i]) for i in range(3))  # type: ignore[return-value]


@dataclass(frozen=True)
class DynamicsCalibrationContract:
    metadata: EvidenceMetadata
    version: str
    initial_position_radius: Vec3
    initial_velocity_radius: Vec3
    control_period: float
    control_period_error: float
    sensor_latency_upper: float
    compute_latency_upper: float
    switch_latency_upper: float
    position_residual_radius: Vec3
    velocity_residual_radius: Vec3
    wind_acceleration_radius: Vec3
    tracking_version: str
    validation_passed: bool
    contract_hash: str

    def __post_init__(self) -> None:
        values = (
            *self.initial_position_radius,
            *self.initial_velocity_radius,
            self.control_period,
            self.control_period_error,
            self.sensor_latency_upper,
            self.compute_latency_upper,
            self.switch_latency_upper,
            *self.position_residual_radius,
            *self.velocity_residual_radius,
            *self.wind_acceleration_radius,
        )
        if not self.version or not self.tracking_version or not self.contract_hash:
            raise ValueError("dynamics contract requires versions and evidence hash")
        if any(not isfinite(value) or value < 0.0 for value in values) or self.control_period <= 0.0:
            raise ValueError("invalid dynamics calibration bounds")
        if self.control_period_error >= self.control_period:
            raise ValueError("control-period error must be smaller than the period")
        if self.contract_hash != self.expected_hash:
            raise ValueError("dynamics calibration hash mismatch")

    @property
    def expected_hash(self) -> str:
        return evidence_hash(
            {
                "metadata": self.metadata.canonical_payload(),
                "version": self.version,
                "initial_position": self.initial_position_radius,
                "initial_velocity": self.initial_velocity_radius,
                "period": self.control_period,
                "period_error": self.control_period_error,
                "latencies": (
                    self.sensor_latency_upper,
                    self.compute_latency_upper,
                    self.switch_latency_upper,
                ),
                "position_residual": self.position_residual_radius,
                "velocity_residual": self.velocity_residual_radius,
                "wind": self.wind_acceleration_radius,
                "tracking_version": self.tracking_version,
                "validation_passed": self.validation_passed,
            }
        )

    @property
    def status(self) -> str:
        return self.metadata.physical_status if self.validation_passed else "blocked-by-calibration"

    @property
    def latency_upper(self) -> float:
        return self.sensor_latency_upper + self.compute_latency_upper + self.switch_latency_upper

    def is_applicable(self, timestamp: float, point: Mapping[str, float | str], device: str, firmware: str) -> bool:
        return self.validation_passed and self.contract_hash == self.expected_hash and self.metadata.is_applicable(timestamp, point, device, firmware)

    def to_runtime_bounds(self, tracking: TrackingCalibrationContract):
        if tracking.version != self.tracking_version:
            raise ValueError("tracking and dynamics contract versions do not match")
        from cert_runtime.envelope import DynamicsBounds

        return DynamicsBounds(
            self.control_period,
            self.position_residual_radius,
            self.velocity_residual_radius,
            tracking.action_tracking_radius,
            self.control_period_error,
            self.latency_upper,
            self.wind_acceleration_radius,
            self.version,
            self.validation_passed,
            sensor_latency_upper=self.sensor_latency_upper,
            compute_latency_upper=self.compute_latency_upper,
            switch_latency_upper=self.switch_latency_upper,
            tracking_version=tracking.version,
            contract_hash=self.contract_hash,
            physical_status=(
                "implemented"
                if self.status == "implemented" and tracking.status == "implemented"
                else "blocked-by-calibration"
            ),
        )


@dataclass(frozen=True)
class DynamicsCalibrationReport(CalibrationReportBase):
    strata: tuple[tuple[str, int], ...]
    tracking_version: str


def build_dynamics_contract(
    samples: Iterable[DynamicsSample],
    metadata: EvidenceMetadata,
    version: str,
    tracking: TrackingCalibrationContract,
    *,
    initial_position_radius: Vec3,
    initial_velocity_radius: Vec3,
    control_period: float,
    control_period_error: float,
    sensor_latency_upper: float,
    compute_latency_upper: float,
    switch_latency_upper: float,
    position_residual_radius: Vec3,
    velocity_residual_radius: Vec3,
    wind_acceleration_radius: Vec3,
) -> tuple[DynamicsCalibrationContract, DynamicsCalibrationReport]:
    samples_tuple = tuple(samples)
    if not samples_tuple:
        raise ValueError("dynamics calibration requires samples")
    counts = {split: sum(sample.split == split for sample in samples_tuple) for split in DataSplit}
    if counts[DataSplit.CALIBRATION] == 0 or counts[DataSplit.VALIDATION] == 0:
        raise ValueError("dynamics calibration and validation splits are required")
    validation = tuple(sample for sample in samples_tuple if sample.split == DataSplit.VALIDATION)
    channels = []
    for name, selected, accessor in (
        ("position", position_residual_radius, lambda sample: sample.position_residual),
        ("velocity", velocity_residual_radius, lambda sample: sample.velocity_residual),
    ):
        for axis in range(3):
            residuals = tuple(accessor(sample)[axis] for sample in samples_tuple)
            validation_residuals = tuple(accessor(sample)[axis] for sample in validation)
            channels.append(
                ChannelCalibrationResult(
                    f"{name}_{axis}",
                    len(samples_tuple),
                    max(residuals),
                    selected[axis],
                    sum(value > selected[axis] for value in validation_residuals),
                    len(validation),
                    metadata.confidence_semantics,
                )
            )
    validation_passed = all(channel.validation_exceedances == 0 for channel in channels)
    payload = {
        "metadata": metadata.canonical_payload(),
        "version": version,
        "initial_position": initial_position_radius,
        "initial_velocity": initial_velocity_radius,
        "period": control_period,
        "period_error": control_period_error,
        "latencies": (sensor_latency_upper, compute_latency_upper, switch_latency_upper),
        "position_residual": position_residual_radius,
        "velocity_residual": velocity_residual_radius,
        "wind": wind_acceleration_radius,
        "tracking_version": tracking.version,
        "validation_passed": validation_passed,
    }
    contract = DynamicsCalibrationContract(
        metadata,
        version,
        initial_position_radius,
        initial_velocity_radius,
        control_period,
        control_period_error,
        sensor_latency_upper,
        compute_latency_upper,
        switch_latency_upper,
        position_residual_radius,
        velocity_residual_radius,
        wind_acceleration_radius,
        tracking.version,
        validation_passed,
        evidence_hash(payload),
    )
    strata_counts: dict[str, int] = {}
    for sample in samples_tuple:
        point = dict(sample.operating_point)
        key = f"{point.get('flight_mode','unknown')}|payload={point.get('payload','unknown')}"
        strata_counts[key] = strata_counts.get(key, 0) + 1
    report = DynamicsCalibrationReport(
        metadata.evidence_id,
        version,
        len(samples_tuple),
        counts[DataSplit.TRAIN],
        counts[DataSplit.CALIBRATION],
        counts[DataSplit.VALIDATION],
        metadata.applicable_domain,
        tuple(channels),
        metadata.confidence_semantics,
        validation_passed,
        metadata.physical_status,
        (() if metadata.physical_status == "implemented" else ("synthetic dynamics data",)),
        metadata.evidence_digest,
        tuple(sorted(strata_counts.items())),
        tracking.version,
    )
    return contract, report
