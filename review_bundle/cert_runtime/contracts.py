from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite, pi
from typing import Literal


ContractStatus = Literal[
    "implemented",
    "blocked-by-calibration",
    "blocked-by-deployment-evidence",
]


@dataclass(frozen=True)
class RuntimeSensorBoundsContract:
    position_error: float | None = None
    attitude_error_radians: float | None = None
    range_error: float | None = None
    beam_half_angle_radians: float | None = None
    time_sync_error: float | None = None
    footprint_radius: float | None = None
    map_discretization_error: float | None = None
    maximum_range: float | None = None
    maximum_speed: float | None = None
    evidence_max_age_seconds: float | None = None
    minimum_free_observations: int | None = None
    calibration_version: str | None = None

    @property
    def status(self) -> ContractStatus:
        return (
            "implemented"
            if not self.missing_parameters and not self.invalid_parameters
            else "blocked-by-calibration"
        )

    @property
    def missing_parameters(self) -> tuple[str, ...]:
        return tuple(field.name for field in fields(self) if getattr(self, field.name) is None)

    @property
    def invalid_parameters(self) -> tuple[str, ...]:
        if self.missing_parameters:
            return tuple()
        numeric_names = (
            "position_error",
            "attitude_error_radians",
            "range_error",
            "beam_half_angle_radians",
            "time_sync_error",
            "footprint_radius",
            "map_discretization_error",
            "maximum_range",
            "maximum_speed",
            "evidence_max_age_seconds",
        )
        invalid = [
            name
            for name in numeric_names
            if not isfinite(float(getattr(self, name))) or float(getattr(self, name)) < 0.0
        ]
        if float(self.maximum_range) <= 0.0:
            invalid.append("maximum_range")
        if int(self.minimum_free_observations) <= 0:
            invalid.append("minimum_free_observations")
        if not str(self.calibration_version):
            invalid.append("calibration_version")
        total_angle = float(self.attitude_error_radians) + float(self.beam_half_angle_radians)
        if not 0.0 <= total_angle < pi:
            invalid.extend(("attitude_error_radians", "beam_half_angle_radians"))
        return tuple(sorted(set(invalid)))

    def require_complete(self) -> None:
        if self.missing_parameters:
            raise CalibrationError(
                "sensor calibration is incomplete: " + ", ".join(self.missing_parameters)
            )
        if self.invalid_parameters:
            raise CalibrationError(
                "sensor calibration is invalid: " + ", ".join(self.invalid_parameters)
            )


@dataclass(frozen=True)
class WCETContract:
    sensor_seconds: float | None = None
    update_seconds: float | None = None
    kappa_seconds: float | None = None
    corridor_seconds: float | None = None
    energy_seconds: float | None = None
    set_construction_seconds: float | None = None
    actor_seconds: float | None = None
    recheck_seconds: float | None = None
    publish_seconds: float | None = None
    margin_seconds: float | None = None
    control_period_seconds: float | None = None
    deployment_evidence_id: str | None = None

    @property
    def status(self) -> ContractStatus:
        return "implemented" if self.is_satisfied else "blocked-by-deployment-evidence"

    @property
    def total_seconds(self) -> float | None:
        components = (
            self.sensor_seconds,
            self.update_seconds,
            self.kappa_seconds,
            self.corridor_seconds,
            self.energy_seconds,
            self.set_construction_seconds,
            self.actor_seconds,
            self.recheck_seconds,
            self.publish_seconds,
            self.margin_seconds,
        )
        if any(value is None for value in components):
            return None
        return sum(value for value in components if value is not None)

    @property
    def is_satisfied(self) -> bool:
        total = self.total_seconds
        components = (
            self.sensor_seconds,
            self.update_seconds,
            self.kappa_seconds,
            self.corridor_seconds,
            self.energy_seconds,
            self.set_construction_seconds,
            self.actor_seconds,
            self.recheck_seconds,
            self.publish_seconds,
            self.margin_seconds,
        )
        return (
            total is not None
            and self.control_period_seconds is not None
            and isfinite(total)
            and all(value is not None and isfinite(value) and value >= 0.0 for value in components)
            and isfinite(self.control_period_seconds)
            and self.control_period_seconds > 0.0
            and bool(self.deployment_evidence_id)
            and total < self.control_period_seconds
        )


class CalibrationError(RuntimeError):
    pass
