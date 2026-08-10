from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .lidar import LidarPacket
from .plant_env import CertifiedSingleUAVPlantEnv
from .state import UAVPhysicalState


class StateSource(Protocol):
    def read_state(self) -> UAVPhysicalState: ...


class LidarSource(Protocol):
    def read_lidar(self) -> LidarPacket: ...


class ActuatorCommandSink(Protocol):
    def publish(self, action: np.ndarray) -> None: ...


@dataclass
class PlantAdapter:
    """Synthetic/HIL-shaped adapter exposing measurements, not world geometry."""

    plant: CertifiedSingleUAVPlantEnv

    def read_state(self) -> UAVPhysicalState:
        return self.plant.state.copy()

    def read_lidar(self) -> LidarPacket:
        if self.plant.last_lidar is None:
            raise RuntimeError("LiDAR is unavailable")
        return self.plant.last_lidar


class ReplayLidarSource:
    def __init__(self, packets: tuple[LidarPacket, ...]) -> None:
        self._packets = iter(packets)

    def read_lidar(self) -> LidarPacket:
        return next(self._packets)
