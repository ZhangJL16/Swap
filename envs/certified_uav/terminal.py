from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .state import UAVPhysicalState, as_vec3


@dataclass(frozen=True)
class TerminalSpec:
    position_low: np.ndarray
    position_high: np.ndarray
    velocity_abs_max: np.ndarray
    minimum_energy: float
    continuation_modes: tuple[str, ...]
    version: str

    def __post_init__(self) -> None:
        low = as_vec3(self.position_low, "terminal position_low")
        high = as_vec3(self.position_high, "terminal position_high")
        velocity = as_vec3(self.velocity_abs_max, "terminal velocity_abs_max")
        if np.any(high <= low) or np.any(velocity < 0.0):
            raise ValueError("invalid terminal position or velocity bounds")
        if not np.isfinite(self.minimum_energy) or self.minimum_energy < 0.0:
            raise ValueError("minimum terminal energy must be finite and nonnegative")
        if any(mode not in {"hover", "descent", "docking", "charging_handoff"} for mode in self.continuation_modes):
            raise ValueError("unsupported terminal continuation mode")
        object.__setattr__(self, "position_low", low)
        object.__setattr__(self, "position_high", high)
        object.__setattr__(self, "velocity_abs_max", velocity)

    def is_charge_admissible(self, state: UAVPhysicalState) -> bool:
        continuation_evidenced = "hover" in self.continuation_modes
        return bool(
            np.all(state.position >= self.position_low)
            and np.all(state.position <= self.position_high)
            and np.all(np.abs(state.velocity) <= self.velocity_abs_max)
            and state.energy >= self.minimum_energy
            and continuation_evidenced
        )

    def is_admissible(self, state: UAVPhysicalState) -> bool:
        return self.is_charge_admissible(state)
