from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .state import UAVPhysicalState, as_vec3


def validate_action_box(action: np.ndarray, maximum: np.ndarray, tolerance: float = 1e-12) -> np.ndarray:
    action_array = as_vec3(action, "action")
    maximum_array = as_vec3(maximum, "maximum")
    if np.any(maximum_array <= 0.0):
        raise ValueError("maximum must be positive")
    if np.any(action_array < -maximum_array - tolerance) or np.any(action_array > maximum_array + tolerance):
        raise ValueError("executed action lies outside the actuator box")
    return action_array


@dataclass(frozen=True)
class ActionTrace:
    nominal: np.ndarray | None
    candidate: np.ndarray | None
    fallback: np.ndarray
    published: np.ndarray
    measured: np.ndarray
    accepted: bool
    fallback_reason: str | None
    certificate_epoch: str

    def __post_init__(self) -> None:
        for name in ("fallback", "published", "measured"):
            object.__setattr__(self, name, as_vec3(getattr(self, name), name))
        for name in ("nominal", "candidate"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, as_vec3(value, name))


class ActuatorTrackingModel:
    """Bounded synthetic action-tracking fixture or deterministic replay source."""

    def __init__(
        self,
        error_bound: np.ndarray,
        deterministic_bias: np.ndarray | None = None,
        replay_actions: Iterable[np.ndarray] | None = None,
    ) -> None:
        self.error_bound = as_vec3(error_bound, "error_bound")
        if np.any(self.error_bound < 0.0):
            raise ValueError("error_bound must be nonnegative")
        self.deterministic_bias = as_vec3(
            np.zeros(3) if deterministic_bias is None else deterministic_bias,
            "deterministic_bias",
        )
        if np.any(np.abs(self.deterministic_bias) > self.error_bound):
            raise ValueError("deterministic bias exceeds the declared tracking bound")
        self._replay = iter(replay_actions) if replay_actions is not None else None

    def apply(
        self,
        command: np.ndarray,
        state: UAVPhysicalState,
        rng: np.random.Generator,
    ) -> np.ndarray:
        del state, rng
        command_array = as_vec3(command, "command")
        if self._replay is not None:
            measured = as_vec3(next(self._replay), "replay measured action")
            if np.any(np.abs(measured - command_array) > self.error_bound + 1e-12):
                raise ValueError("replay tracking residual exceeds the declared bound")
            return measured
        return command_array + self.deterministic_bias
