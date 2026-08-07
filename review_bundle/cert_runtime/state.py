from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .corridor import ReturnCorridor
from .geometry import RollingLocalGeometry
from .types import Vec3, vec3


@dataclass(frozen=True)
class CertificateStateSnapshot:
    position: Vec3
    velocity: Vec3
    energy: float
    charging_position: Vec3
    certificate_version: tuple[int, int, int]
    local_geometry_digest: str
    return_corridor_digest: str
    bound_versions: tuple[tuple[str, str], ...]
    explicit_task_state: tuple[tuple[str, str], ...]


@dataclass
class CertificateState:
    position: Vec3
    velocity: Vec3
    energy: float
    charging_position: Vec3
    local_geometry: RollingLocalGeometry
    return_corridor: ReturnCorridor
    explicit_task_state: dict[str, Any] = field(default_factory=dict)
    position_error_radius: Vec3 = (0.0, 0.0, 0.0)
    velocity_error_radius: Vec3 = (0.0, 0.0, 0.0)
    energy_error_radius: float = 0.0
    bound_versions: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.position = vec3(self.position)
        self.velocity = vec3(self.velocity)
        self.charging_position = vec3(self.charging_position)
        if self.energy < 0.0:
            raise ValueError("remaining energy must be nonnegative")
        if any(value < 0.0 for value in self.position_error_radius + self.velocity_error_radius):
            raise ValueError("state error radii must be nonnegative")
        if self.energy_error_radius < 0.0:
            raise ValueError("energy error radius must be nonnegative")

    @property
    def certificate_version(self) -> tuple[int, int, int]:
        return (
            self.local_geometry.version,
            self.return_corridor.version,
            self.return_corridor.certificate_epoch,
        )

    def snapshot(self) -> CertificateStateSnapshot:
        return CertificateStateSnapshot(
            self.position,
            self.velocity,
            self.energy,
            self.charging_position,
            self.certificate_version,
            self.local_geometry.certificate_digest(),
            self.return_corridor.certificate_digest(),
            tuple(sorted((str(key), str(value)) for key, value in self.bound_versions.items())),
            tuple(sorted((str(key), repr(value)) for key, value in self.explicit_task_state.items())),
        )
