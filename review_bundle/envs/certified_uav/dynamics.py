from __future__ import annotations

import numpy as np

from .state import as_vec3


def integrate_double_integrator(
    position: np.ndarray,
    velocity: np.ndarray,
    measured_acceleration: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Authoritative state propagation used by plant, tests, and calibration."""

    position_array = as_vec3(position, "position")
    velocity_array = as_vec3(velocity, "velocity")
    acceleration_array = as_vec3(measured_acceleration, "measured_acceleration")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    position_next = position_array + dt * velocity_array + 0.5 * dt * dt * acceleration_array
    velocity_next = velocity_array + dt * acceleration_array
    return position_next, velocity_next
