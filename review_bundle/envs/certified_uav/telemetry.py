from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .actuator import ActionTrace
from .lidar import LidarPacket
from .state import UAVPhysicalState


@dataclass(frozen=True)
class StepTelemetry:
    state_before: UAVPhysicalState
    state_after: UAVPhysicalState
    action_trace: ActionTrace
    energy_cost: float
    collision: bool
    terminal_admissible: bool
    lidar_packet: LidarPacket
    certificate_version: str | None
    geometry_version: str | None
    corridor_version: str | None


@dataclass(frozen=True)
class CalibrationLogRecord:
    timestamp: float
    position: tuple[float, float, float]
    velocity: tuple[float, float, float]
    commanded_action: tuple[float, float, float] | None
    published_action: tuple[float, float, float]
    measured_action: tuple[float, float, float]
    next_position: tuple[float, float, float]
    next_velocity: tuple[float, float, float]
    lidar_distances: tuple[float, ...]
    lidar_valid: tuple[bool, ...]
    lidar_hit: tuple[bool, ...]
    battery_voltage: float | None
    battery_current: float | None
    power: float | None
    energy_before: float
    energy_after: float
    sensor_version: str
    dynamics_version: str
    tracking_version: str
    energy_version: str
    terminal_version: str
    evidence_kind: str = "synthetic-simulator"


class CalibrationRecordLogger:
    def __init__(self) -> None:
        self.records: list[CalibrationLogRecord] = []

    def append(self, record: CalibrationLogRecord) -> None:
        self.records.append(record)

    def export(self) -> tuple[dict[str, Any], ...]:
        return tuple(asdict(record) for record in self.records)


def tuple3(value: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(component) for component in value)
