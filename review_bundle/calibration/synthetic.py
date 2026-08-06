from __future__ import annotations

from .confidence import BoundEstimate
from .dynamics import DynamicsSample, build_dynamics_contract
from .energy import EnergySample, build_energy_contract
from .schema import (
    ConfidenceSemantics,
    DataSplit,
    EvidenceMetadata,
    OperatingDomain,
    RawCalibrationRecord,
    SourceKind,
    evidence_hash,
)
from .sensor import build_sensor_contract
from .terminal import build_terminal_contract
from .tracking import TrackingSample, build_tracking_contract


def synthetic_operating_domain() -> OperatingDomain:
    return OperatingDomain((0.0, 3.0), (0.0, 3.0), (0.0, 2.0), (0.0, 45.0), (10.0, 30.0), ("hover",))


def synthetic_metadata(evidence_id: str, version: str, start: float = 0.0, end: float = 10.0) -> EvidenceMetadata:
    payload = {"evidence_id": evidence_id, "version": version, "synthetic": True}
    return EvidenceMetadata(
        evidence_id,
        start,
        end,
        "synthetic-device-v1",
        "synthetic-firmware-v1",
        "deterministic synthetic fixture",
        synthetic_operating_domain(),
        ConfidenceSemantics.SIMULTANEOUS_CONFIDENCE,
        SourceKind.SYNTHETIC,
        1000.0,
        "invalidate on any fixture or version change",
        evidence_hash(payload),
        confidence_delta=0.01,
        simultaneous_family_size=16,
    )


def build_synthetic_calibration_bundle():
    point = tuple(sorted({
        "speed": 0.5,
        "acceleration": 0.5,
        "payload": 0.5,
        "temperature": 20.0,
        "voltage": 20.0,
        "flight_mode": "hover",
    }.items()))
    sensor_metadata = synthetic_metadata("sensor-evidence", "sensor-v1")
    sensor_records = []
    for channel, residual in (("position", 0.01), ("attitude", 0.005), ("range", 0.01), ("time_sync", 0.005)):
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION)):
            sensor_records.append(
                RawCalibrationRecord(
                    f"{channel}-{index}",
                    float(index + 1),
                    channel,
                    residual,
                    0.0,
                    split,
                    sensor_metadata.device_version,
                    sensor_metadata.firmware_version,
                    point,
                )
            )
    estimates = {
        channel: BoundEstimate(value, ConfidenceSemantics.SIMULTANEOUS_CONFIDENCE, 0.01, ("synthetic",), False)
        for channel, value in (("position", 0.02), ("attitude", 0.01), ("range", 0.02), ("time_sync", 0.01))
    }
    sensor, sensor_report = build_sensor_contract(
        sensor_records,
        sensor_metadata,
        "sensor-v1",
        estimates,
        beam_half_angle_radians=0.6,
        footprint_radius=0.03,
        map_discretization_error=0.01,
        maximum_range=8.0,
        maximum_speed=3.0,
        evidence_max_age_seconds=10.0,
        minimum_free_observations=1,
    )
    tracking_metadata = synthetic_metadata("tracking-evidence", "tracking-v1")
    tracking_samples = tuple(
        TrackingSample(
            f"tracking-{index}",
            float(index),
            float(index) + 0.001,
            float(index) + 0.002,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.005, -0.005, 0.0),
            split,
            point,
        )
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
    )
    tracking, tracking_report = build_tracking_contract(
        tracking_samples, tracking_metadata, "tracking-v1", (0.01, 0.01, 0.01), 0.01
    )
    dynamics_metadata = synthetic_metadata("dynamics-evidence", "dynamics-v1")
    dynamics_samples = tuple(
        DynamicsSample(
            f"dynamics-{index}",
            float(index),
            float(index) + 0.5,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.001, 0.0, 0.0),
            (0.001, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            split,
            point,
        )
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
    )
    dynamics, dynamics_report = build_dynamics_contract(
        dynamics_samples,
        dynamics_metadata,
        "dynamics-v1",
        tracking,
        initial_position_radius=(0.01, 0.01, 0.01),
        initial_velocity_radius=(0.01, 0.01, 0.01),
        control_period=0.5,
        control_period_error=0.01,
        sensor_latency_upper=0.005,
        compute_latency_upper=0.005,
        switch_latency_upper=0.005,
        position_residual_radius=(0.01, 0.01, 0.01),
        velocity_residual_radius=(0.01, 0.01, 0.01),
        wind_acceleration_radius=(0.01, 0.01, 0.01),
    )
    energy_metadata = synthetic_metadata("energy-evidence", "energy-v1")
    energy_samples = tuple(
        EnergySample(
            f"energy-{index}",
            float(index),
            float(index) + 0.5,
            20.0,
            20.0,
            0.1,
            0.1,
            0.2,
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            True,
            1.0,
            20.0,
            0.5,
            split,
        )
        for index, split in enumerate((DataSplit.CALIBRATION, DataSplit.VALIDATION), start=1)
    )
    energy, energy_report = build_energy_contract(
        energy_samples,
        energy_metadata,
        "energy-v1",
        avionics_cost=0.05,
        hover_cost=0.05,
        velocity_coefficients=(0.01, 0.01, 0.01),
        action_coefficients=(0.01, 0.01, 0.02),
        communication_cost=0.02,
        computation_cost=0.02,
        measurement_error=0.02,
        underestimation_margin=0.05,
    )
    terminal_metadata = synthetic_metadata("terminal-evidence", "terminal-v1")
    terminal = build_terminal_contract(
        terminal_metadata,
        "terminal-v1",
        horizontal_position=(-5.0, -5.0, 5.0, 5.0),
        altitude=(-1.0, 1.0),
        velocity_low=(-3.0, -3.0, -2.0),
        velocity_high=(3.0, 3.0, 2.0),
        minimum_energy=1.0,
        continuation_evidence=(("hover", "synthetic-hover-evidence"),),
    )
    return (
        (sensor, tracking, dynamics, energy, terminal),
        (sensor_report, tracking_report, dynamics_report, energy_report),
    )
