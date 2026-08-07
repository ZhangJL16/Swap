from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np


def _positive_vec3(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.shape != (3,) or not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError(f"{name} must be a finite positive (3,) array")
    return array


@dataclass(frozen=True)
class CertifiedUAVConfig:
    """Synthetic experiment parameters, not flight-calibrated constants."""

    world_size: np.ndarray = field(default_factory=lambda: np.array([4.0, 4.0, 2.0]))
    dt: float = 0.2
    v_max: np.ndarray = field(default_factory=lambda: np.array([0.12, 0.12, 0.08]))
    a_max: np.ndarray = field(default_factory=lambda: np.array([0.08, 0.08, 0.05]))
    body_radius: float = 0.05
    lidar_range: float = 6.0
    num_lasers: int = 32
    initial_energy: float = 100.0
    episode_limit: int = 300
    tracking_error_bound: np.ndarray = field(default_factory=lambda: np.array([0.002, 0.002, 0.001]))
    lidar_range_noise: float = 0.0
    lidar_pose_noise: float = 0.0
    lidar_heading_noise: float = 0.0
    lidar_invalid_probability: float = 0.0
    total_latency: float = 0.02
    certification_deadline: float = 0.1
    braking_deceleration: float = 1.0
    geometry_margin: float = 0.02
    grid_resolution: float = 0.05
    local_map_encoding_size: int = 16
    corridor_encoding_size: int = 24
    minimum_generator_sigma: float = 0.005
    maximum_generator_condition: float = 20.0
    generator_bisection_iterations: int = 8
    synthetic_fixture: bool = True
    terminate_on_terminal: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_size", _positive_vec3(self.world_size, "world_size"))
        object.__setattr__(self, "v_max", _positive_vec3(self.v_max, "v_max"))
        object.__setattr__(self, "a_max", _positive_vec3(self.a_max, "a_max"))
        tracking = np.asarray(self.tracking_error_bound, dtype=np.float64).copy()
        if tracking.shape != (3,) or not np.all(np.isfinite(tracking)) or np.any(tracking < 0.0):
            raise ValueError("tracking_error_bound must be a finite nonnegative (3,) array")
        object.__setattr__(self, "tracking_error_bound", tracking)
        positive_scalars = {
            "dt": self.dt,
            "body_radius": self.body_radius,
            "lidar_range": self.lidar_range,
            "initial_energy": self.initial_energy,
            "certification_deadline": self.certification_deadline,
            "braking_deceleration": self.braking_deceleration,
            "grid_resolution": self.grid_resolution,
            "minimum_generator_sigma": self.minimum_generator_sigma,
        }
        if any(not np.isfinite(value) or value <= 0.0 for value in positive_scalars.values()):
            raise ValueError(f"positive finite parameters required: {positive_scalars}")
        if self.num_lasers != 32:
            raise ValueError("the first certified environment version requires exactly 32 LiDAR rays")
        if self.episode_limit <= 0 or self.generator_bisection_iterations <= 0:
            raise ValueError("episode and bisection limits must be positive")
        if self.total_latency < 0.0 or self.total_latency >= self.dt:
            raise ValueError("total_latency must lie in [0, dt)")
        if self.certification_deadline >= self.dt:
            raise ValueError("certification_deadline must be smaller than the control period")

    @property
    def sensing_braking_requirement(self) -> float:
        horizontal_speed = float(np.max(self.v_max[:2]))
        return (
            horizontal_speed * (self.dt + self.total_latency)
            + horizontal_speed * horizontal_speed / (2.0 * self.braking_deceleration)
            + self.geometry_margin
        )

    @property
    def certified_sensing_valid(self) -> bool:
        return self.lidar_range > self.sensing_braking_requirement


def apply_configuration_overrides(
    config: CertifiedUAVConfig,
    overrides: dict[str, object],
) -> CertifiedUAVConfig:
    if not overrides:
        return config
    allowed = {field_name for field_name in config.__dataclass_fields__}
    unknown = set(overrides) - allowed
    if unknown:
        raise ValueError(f"unsupported scenario configuration overrides: {sorted(unknown)}")
    converted = {
        key: np.asarray(value, dtype=np.float64) if key in {"world_size", "v_max", "a_max", "tracking_error_bound"} else value
        for key, value in overrides.items()
    }
    return replace(config, **converted)
