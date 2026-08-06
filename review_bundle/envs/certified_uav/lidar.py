from __future__ import annotations

from dataclasses import dataclass
from math import pi

import numpy as np

from cert_runtime.geometry import LidarRay

from .obstacles import StaticWorld
from .state import UAVPhysicalState, as_vec3


@dataclass(frozen=True)
class LidarPacket:
    distances: np.ndarray
    valid: np.ndarray
    hit: np.ndarray
    angles: np.ndarray
    timestamp: float
    pose_position: np.ndarray
    pose_heading: float
    sensor_version: str

    def __post_init__(self) -> None:
        distances = np.asarray(self.distances, dtype=np.float64).copy()
        valid = np.asarray(self.valid, dtype=bool).copy()
        hit = np.asarray(self.hit, dtype=bool).copy()
        angles = np.asarray(self.angles, dtype=np.float64).copy()
        if distances.ndim != 1 or distances.shape != valid.shape or distances.shape != hit.shape or distances.shape != angles.shape:
            raise ValueError("LiDAR arrays must be aligned one-dimensional arrays")
        if not np.all(np.isfinite(distances)) or not np.all(np.isfinite(angles)):
            raise ValueError("LiDAR distances and angles must be finite")
        if np.any(hit & ~valid):
            raise ValueError("invalid LiDAR rays cannot be marked as hits")
        object.__setattr__(self, "distances", distances)
        object.__setattr__(self, "valid", valid)
        object.__setattr__(self, "hit", hit)
        object.__setattr__(self, "angles", angles)
        object.__setattr__(self, "pose_position", as_vec3(self.pose_position, "pose_position"))
        if not np.isfinite(self.timestamp) or not np.isfinite(self.pose_heading):
            raise ValueError("LiDAR timestamp and heading must be finite")

    def to_certificate_rays(self, frame_prefix: str = "lidar") -> tuple[LidarRay, ...]:
        rays: list[LidarRay] = []
        for index, angle in enumerate(self.angles):
            rays.append(
                LidarRay(
                    float(np.cos(angle)),
                    float(np.sin(angle)),
                    float(self.distances[index]),
                    bool(self.valid[index]),
                    bool(self.hit[index]),
                    f"{frame_prefix}-{index}",
                    float(self.timestamp),
                )
            )
        return tuple(rays)


class HorizontalLidarModel:
    def __init__(
        self,
        num_lasers: int,
        maximum_range: float,
        sensor_version: str,
        range_noise: float = 0.0,
        pose_noise: float = 0.0,
        heading_noise: float = 0.0,
        invalid_probability: float = 0.0,
    ) -> None:
        if num_lasers != 32:
            raise ValueError("the first certified environment requires 32 rays")
        if maximum_range <= 0.0 or not 0.0 <= invalid_probability <= 1.0:
            raise ValueError("invalid LiDAR configuration")
        self.num_lasers = num_lasers
        self.maximum_range = float(maximum_range)
        self.sensor_version = sensor_version
        self.range_noise = float(range_noise)
        self.pose_noise = float(pose_noise)
        self.heading_noise = float(heading_noise)
        self.invalid_probability = float(invalid_probability)
        self.forced_invalid_indices: set[int] = set()
        self.timestamp_offset = 0.0

    def measure(
        self,
        state: UAVPhysicalState,
        world: StaticWorld,
        rng: np.random.Generator,
    ) -> LidarPacket:
        pose = state.position + rng.uniform(-self.pose_noise, self.pose_noise, size=3)
        heading = float(rng.uniform(-self.heading_noise, self.heading_noise))
        angles = heading + np.arange(self.num_lasers, dtype=np.float64) * (2.0 * pi / self.num_lasers)
        distances = np.empty(self.num_lasers, dtype=np.float64)
        hits = np.empty(self.num_lasers, dtype=bool)
        valid = rng.random(self.num_lasers) >= self.invalid_probability
        for index, angle in enumerate(angles):
            distance, hit = world.ray_distance(state.position, float(angle), self.maximum_range)
            noisy = distance + float(rng.uniform(-self.range_noise, self.range_noise))
            distances[index] = float(np.clip(noisy, 0.0, self.maximum_range))
            hits[index] = bool(hit)
        for index in self.forced_invalid_indices:
            if 0 <= index < self.num_lasers:
                valid[index] = False
        hits &= valid
        return LidarPacket(
            distances,
            valid,
            hits,
            angles,
            state.timestamp + self.timestamp_offset,
            pose,
            heading,
            self.sensor_version,
        )
