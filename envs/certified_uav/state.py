from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def as_vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (3,) array")
    return array


@dataclass
class UAVPhysicalState:
    position: np.ndarray
    velocity: np.ndarray
    energy: float
    timestamp: float

    def __post_init__(self) -> None:
        self.position = as_vec3(self.position, "position")
        self.velocity = as_vec3(self.velocity, "velocity")
        if not np.isfinite(self.energy) or not np.isfinite(self.timestamp):
            raise ValueError("energy and timestamp must be finite")

    def copy(self) -> "UAVPhysicalState":
        return UAVPhysicalState(self.position.copy(), self.velocity.copy(), float(self.energy), float(self.timestamp))
