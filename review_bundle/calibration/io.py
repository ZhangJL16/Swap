from __future__ import annotations

import csv
import json
from pathlib import Path

from .dynamics import DynamicsSample
from .energy import EnergySample
from .schema import DataSplit, RawCalibrationRecord
from .tracking import TrackingSample


def _vec3(value: str) -> tuple[float, float, float]:
    parsed = tuple(float(item) for item in value.split(";"))
    if len(parsed) != 3:
        raise ValueError("vector CSV fields require three semicolon-separated values")
    return parsed  # type: ignore[return-value]


def _operating_point(value: str) -> tuple[tuple[str, float | str], ...]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise ValueError("operating_point must be a JSON object")
    return tuple(sorted((str(key), item) for key, item in payload.items()))


def load_raw_calibration_csv(path: str | Path) -> tuple[RawCalibrationRecord, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(
            RawCalibrationRecord(
                row["record_id"],
                float(row["timestamp"]),
                row["channel"],
                float(row["observed"]),
                float(row["reference"]),
                DataSplit(row["split"]),
                row["device_version"],
                row["firmware_version"],
                _operating_point(row["operating_point"]),
                row.get("valid_measurement", "true").lower() == "true",
                row.get("invalid_reason") or None,
            )
            for row in csv.DictReader(handle)
        )


def load_tracking_csv(path: str | Path) -> tuple[TrackingSample, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(
            TrackingSample(
                row["sample_id"],
                float(row["command_timestamp"]),
                float(row["publish_timestamp"]),
                float(row["measurement_timestamp"]),
                _vec3(row["commanded_action"]),
                _vec3(row["published_action"]),
                _vec3(row["measured_action"]),
                DataSplit(row["split"]),
                _operating_point(row["operating_point"]),
            )
            for row in csv.DictReader(handle)
        )


def load_dynamics_csv(path: str | Path) -> tuple[DynamicsSample, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(
            DynamicsSample(
                row["sample_id"],
                float(row["state_timestamp"]),
                float(row["next_state_timestamp"]),
                _vec3(row["position"]),
                _vec3(row["velocity"]),
                _vec3(row["next_position"]),
                _vec3(row["next_velocity"]),
                _vec3(row["commanded_action"]),
                _vec3(row["published_action"]),
                _vec3(row["measured_action"]),
                DataSplit(row["split"]),
                _operating_point(row["operating_point"]),
            )
            for row in csv.DictReader(handle)
        )


def load_energy_csv(path: str | Path) -> tuple[EnergySample, ...]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return tuple(
            EnergySample(
                row["sample_id"],
                float(row["start_timestamp"]),
                float(row["end_timestamp"]),
                float(row["voltage_start"]),
                float(row["voltage_end"]),
                float(row["current_start"]),
                float(row["current_end"]),
                float(row["measured_energy"]),
                _vec3(row["velocity"]),
                _vec3(row["action"]),
                row["communication_active"].lower() == "true",
                float(row["compute_load"]),
                float(row["temperature"]),
                float(row["payload"]),
                DataSplit(row["split"]),
            )
            for row in csv.DictReader(handle)
        )
