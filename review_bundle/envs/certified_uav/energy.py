from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .state import UAVPhysicalState, as_vec3


@dataclass(frozen=True)
class SimulationEnergyConfig:
    base_power: float = 0.05
    velocity_coefficients: np.ndarray = field(default_factory=lambda: np.full(3, 0.005))
    acceleration_coefficients: np.ndarray = field(default_factory=lambda: np.full(3, 0.005))
    compute_power: float = 0.005
    communication_power: float = 0.005
    simulation_error: float = 0.0

    def __post_init__(self) -> None:
        velocity = as_vec3(self.velocity_coefficients, "velocity_coefficients")
        acceleration = as_vec3(self.acceleration_coefficients, "acceleration_coefficients")
        if np.any(velocity < 0.0) or np.any(acceleration < 0.0):
            raise ValueError("energy coefficients must be nonnegative")
        object.__setattr__(self, "velocity_coefficients", velocity)
        object.__setattr__(self, "acceleration_coefficients", acceleration)
        scalars = (self.base_power, self.compute_power, self.communication_power, self.simulation_error)
        if any(not np.isfinite(value) or value < 0.0 for value in scalars):
            raise ValueError("energy scalar parameters must be finite and nonnegative")


class EnergyModel:
    def __init__(self, config: SimulationEnergyConfig | None = None) -> None:
        self.config = SimulationEnergyConfig() if config is None else config

    def realized_cost(
        self,
        state: UAVPhysicalState,
        measured_action: np.ndarray,
        dt: float,
    ) -> float:
        action = as_vec3(measured_action, "measured_action")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")
        power = (
            self.config.base_power
            + float(np.sum(self.config.velocity_coefficients * np.abs(state.velocity)))
            + float(np.sum(self.config.acceleration_coefficients * action * action))
            + self.config.compute_power
            + self.config.communication_power
        )
        return float(dt * power + self.config.simulation_error)
