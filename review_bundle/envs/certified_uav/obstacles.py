from __future__ import annotations

from dataclasses import dataclass
from math import cos, inf, sin, sqrt

import numpy as np

from .state import as_vec3


@dataclass(frozen=True)
class AABBObstacle:
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self) -> None:
        low = as_vec3(self.low, "obstacle low")
        high = as_vec3(self.high, "obstacle high")
        if np.any(high <= low):
            raise ValueError("AABB high must exceed low")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True)
class CylinderObstacle:
    center_xy: np.ndarray
    radius: float
    z_low: float
    z_high: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_xy, dtype=np.float64).copy()
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            raise ValueError("cylinder center must be finite (2,)")
        if self.radius <= 0.0 or self.z_high <= self.z_low:
            raise ValueError("invalid cylinder dimensions")
        object.__setattr__(self, "center_xy", center)


def _segment_aabb_intersection(start: np.ndarray, end: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    direction = end - start
    lower_time = 0.0
    upper_time = 1.0
    for axis in range(3):
        if abs(direction[axis]) <= 1e-15:
            if start[axis] < low[axis] or start[axis] > high[axis]:
                return False
            continue
        first = (low[axis] - start[axis]) / direction[axis]
        second = (high[axis] - start[axis]) / direction[axis]
        entry, exit_ = min(first, second), max(first, second)
        lower_time = max(lower_time, entry)
        upper_time = min(upper_time, exit_)
        if lower_time > upper_time:
            return False
    return True


def _segment_point_distance_xy(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    delta = end[:2] - start[:2]
    denominator = float(np.dot(delta, delta))
    if denominator <= 1e-18:
        return float(np.linalg.norm(start[:2] - point))
    fraction = float(np.clip(np.dot(point - start[:2], delta) / denominator, 0.0, 1.0))
    closest = start[:2] + fraction * delta
    return float(np.linalg.norm(closest - point))


class StaticWorld:
    """Ground-truth plant geometry, never passed to the certificate pathway."""

    def __init__(
        self,
        world_size: np.ndarray,
        aabbs: tuple[AABBObstacle, ...] = (),
        cylinders: tuple[CylinderObstacle, ...] = (),
    ) -> None:
        self.world_size = as_vec3(world_size, "world_size")
        if np.any(self.world_size <= 0.0):
            raise ValueError("world_size must be positive")
        self.aabbs = tuple(aabbs)
        self.cylinders = tuple(cylinders)

    def swept_collision(
        self,
        position_start: np.ndarray,
        position_end: np.ndarray,
        body_radius: float,
    ) -> bool:
        start = as_vec3(position_start, "position_start")
        end = as_vec3(position_end, "position_end")
        if body_radius < 0.0 or not np.isfinite(body_radius):
            raise ValueError("body_radius must be finite and nonnegative")
        safe_low = np.full(3, body_radius)
        safe_high = self.world_size - body_radius
        if np.any(start < safe_low) or np.any(start > safe_high) or np.any(end < safe_low) or np.any(end > safe_high):
            return True
        inflation = np.full(3, body_radius)
        for obstacle in self.aabbs:
            if _segment_aabb_intersection(start, end, obstacle.low - inflation, obstacle.high + inflation):
                return True
        for obstacle in self.cylinders:
            segment_z_low = min(start[2], end[2])
            segment_z_high = max(start[2], end[2])
            if segment_z_high < obstacle.z_low - body_radius or segment_z_low > obstacle.z_high + body_radius:
                continue
            if _segment_point_distance_xy(start, end, obstacle.center_xy) <= obstacle.radius + body_radius:
                return True
        return False

    def ray_distance(self, origin: np.ndarray, angle: float, maximum_range: float) -> tuple[float, bool]:
        point = as_vec3(origin, "ray origin")
        direction = np.array([cos(angle), sin(angle)], dtype=np.float64)
        nearest = inf
        for axis in range(2):
            component = direction[axis]
            if abs(component) <= 1e-15:
                continue
            for boundary in (0.0, self.world_size[axis]):
                distance = (boundary - point[axis]) / component
                if distance >= 0.0:
                    cross = point[:2] + distance * direction
                    other = 1 - axis
                    if -1e-12 <= cross[other] <= self.world_size[other] + 1e-12:
                        nearest = min(nearest, distance)
        for obstacle in self.aabbs:
            low, high = obstacle.low[:2], obstacle.high[:2]
            lower_time, upper_time = 0.0, inf
            valid = True
            for axis in range(2):
                if abs(direction[axis]) <= 1e-15:
                    if point[axis] < low[axis] or point[axis] > high[axis]:
                        valid = False
                        break
                    continue
                first = (low[axis] - point[axis]) / direction[axis]
                second = (high[axis] - point[axis]) / direction[axis]
                lower_time = max(lower_time, min(first, second))
                upper_time = min(upper_time, max(first, second))
            if valid and upper_time >= max(lower_time, 0.0):
                nearest = min(nearest, max(lower_time, 0.0))
        for obstacle in self.cylinders:
            relative = point[:2] - obstacle.center_xy
            linear = 2.0 * float(np.dot(relative, direction))
            constant = float(np.dot(relative, relative) - obstacle.radius * obstacle.radius)
            discriminant = linear * linear - 4.0 * constant
            if discriminant >= 0.0:
                root = sqrt(discriminant)
                candidates = [(-linear - root) / 2.0, (-linear + root) / 2.0]
                positive = [value for value in candidates if value >= 0.0]
                if positive:
                    nearest = min(nearest, min(positive))
        if nearest <= maximum_range:
            return float(max(nearest, 0.0)), True
        return float(maximum_range), False
