from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
import json
from pathlib import Path
from typing import Any

import numpy as np

from .obstacles import AABBObstacle, CylinderObstacle, StaticWorld
from .state import UAVPhysicalState, as_vec3
from .terminal import TerminalSpec


@dataclass(frozen=True)
class CorridorCellSpec:
    cell_id: int
    level: int
    region_low_xy: np.ndarray
    region_high_xy: np.ndarray
    altitude_limit: float
    state_position_low: np.ndarray
    state_position_high: np.ndarray
    state_velocity_low: np.ndarray
    state_velocity_high: np.ndarray
    energy_low: float
    energy_high: float

    def __post_init__(self) -> None:
        for name in ("region_low_xy", "region_high_xy"):
            array = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if array.shape != (2,) or not np.all(np.isfinite(array)):
                raise ValueError(f"{name} must be finite (2,)")
            object.__setattr__(self, name, array)
        for name in ("state_position_low", "state_position_high", "state_velocity_low", "state_velocity_high"):
            object.__setattr__(self, name, as_vec3(getattr(self, name), name))
        if np.any(self.region_high_xy <= self.region_low_xy):
            raise ValueError("corridor region must have positive area")
        if np.any(self.state_position_high <= self.state_position_low):
            raise ValueError("state position interval must have positive width")
        if np.any(self.state_velocity_high <= self.state_velocity_low):
            raise ValueError("state velocity interval must have positive width")
        if self.energy_high <= self.energy_low:
            raise ValueError("state energy interval must have positive width")


@dataclass(frozen=True)
class ScenarioDefinition:
    name: str
    world_size: np.ndarray
    initial_state: UAVPhysicalState
    station_position: np.ndarray
    task_goal: np.ndarray
    terminal: TerminalSpec
    world: StaticWorld
    corridor_cells: tuple[CorridorCellSpec, ...]
    bootstrap_lidar_poses: tuple[np.ndarray, ...]
    configuration_overrides: dict[str, Any]
    expected_certificate_outcome: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "world_size", as_vec3(self.world_size, "world_size"))
        object.__setattr__(self, "station_position", as_vec3(self.station_position, "station_position"))
        object.__setattr__(self, "task_goal", as_vec3(self.task_goal, "task_goal"))

    def consistency_failures(self, config) -> tuple[str, ...]:
        failures: list[str] = []
        if not self.corridor_cells[-1].region_low_xy[0] <= self.initial_state.position[0] <= self.corridor_cells[-1].region_high_xy[0] or not self.corridor_cells[-1].region_low_xy[1] <= self.initial_state.position[1] <= self.corridor_cells[-1].region_high_xy[1]:
            failures.append("INITIAL_STATE_OUTSIDE_CORRIDOR_SUFFIX")
        first = self.corridor_cells[0]
        if not (
            first.region_low_xy[0] <= self.station_position[0] <= first.region_high_xy[0]
            and first.region_low_xy[1] <= self.station_position[1] <= first.region_high_xy[1]
        ):
            failures.append("CORRIDOR_START_DOES_NOT_CONTAIN_TERMINAL")
        for left, right in zip(self.corridor_cells, self.corridor_cells[1:]):
            overlap_low = np.maximum(left.region_low_xy, right.region_low_xy)
            overlap_high = np.minimum(left.region_high_xy, right.region_high_xy)
            if np.any(overlap_high - overlap_low < 2.0 * config.body_radius - 1e-12):
                failures.append("INSUFFICIENT_CORRIDOR_TRANSFER_OVERLAP")
                break
        terminal_low = self.terminal.position_low
        terminal_high = self.terminal.position_high
        for obstacle in self.world.aabbs:
            if np.all(terminal_high >= obstacle.low - config.body_radius) and np.all(terminal_low <= obstacle.high + config.body_radius):
                failures.append("TERMINAL_INTERSECTS_OBSTACLE")
                break
        for obstacle in self.world.cylinders:
            closest = np.minimum(np.maximum(obstacle.center_xy, terminal_low[:2]), terminal_high[:2])
            vertical_overlap = terminal_high[2] >= obstacle.z_low and terminal_low[2] <= obstacle.z_high
            if vertical_overlap and np.linalg.norm(closest - obstacle.center_xy) <= obstacle.radius + config.body_radius:
                failures.append("TERMINAL_INTERSECTS_OBSTACLE")
                break
        if np.any(np.abs(self.initial_state.velocity) > config.v_max + 1e-12):
            failures.append("INITIAL_VELOCITY_OUTSIDE_PLANT_LIMIT")
        return tuple(dict.fromkeys(failures))


def _parse_scenario(payload: dict[str, Any]) -> ScenarioDefinition:
    world_size = np.asarray(payload["world_size"], dtype=np.float64)
    aabbs = tuple(
        AABBObstacle(np.asarray(item["low"], dtype=np.float64), np.asarray(item["high"], dtype=np.float64))
        for item in payload.get("aabb_obstacles", ())
    )
    cylinders = tuple(
        CylinderObstacle(
            np.asarray(item["center_xy"], dtype=np.float64),
            float(item["radius"]),
            float(item.get("z_low", 0.0)),
            float(item.get("z_high", world_size[2])),
        )
        for item in payload.get("cylinder_obstacles", ())
    )
    terminal_payload = payload["terminal"]
    terminal = TerminalSpec(
        np.asarray(terminal_payload["position_low"], dtype=np.float64),
        np.asarray(terminal_payload["position_high"], dtype=np.float64),
        np.asarray(terminal_payload["velocity_abs_max"], dtype=np.float64),
        float(terminal_payload["minimum_energy"]),
        tuple(terminal_payload["continuation_modes"]),
        str(terminal_payload["version"]),
    )
    initial_payload = payload["initial_state"]
    initial_state = UAVPhysicalState(
        np.asarray(initial_payload["position"], dtype=np.float64),
        np.asarray(initial_payload["velocity"], dtype=np.float64),
        float(initial_payload["energy"]),
        float(initial_payload.get("timestamp", 0.0)),
    )
    corridor_cells = tuple(
        CorridorCellSpec(
            int(item["cell_id"]),
            int(item["level"]),
            np.asarray(item["region_low_xy"], dtype=np.float64),
            np.asarray(item["region_high_xy"], dtype=np.float64),
            float(item["altitude_limit"]),
            np.asarray(item["state_position_low"], dtype=np.float64),
            np.asarray(item["state_position_high"], dtype=np.float64),
            np.asarray(item["state_velocity_low"], dtype=np.float64),
            np.asarray(item["state_velocity_high"], dtype=np.float64),
            float(item["energy_low"]),
            float(item["energy_high"]),
        )
        for item in payload["corridor_cells"]
    )
    return ScenarioDefinition(
        str(payload["name"]),
        world_size,
        initial_state,
        np.asarray(payload["station_position"], dtype=np.float64),
        np.asarray(payload["task_goal"], dtype=np.float64),
        terminal,
        StaticWorld(world_size, aabbs, cylinders),
        corridor_cells,
        tuple(as_vec3(value, "bootstrap_lidar_pose") for value in payload.get("bootstrap_lidar_poses", ())),
        dict(payload.get("configuration_overrides", {})),
        str(payload.get("expected_certificate_outcome", "unspecified")),
    )


def load_scenario(path_or_name: str | Path) -> ScenarioDefinition:
    path = Path(path_or_name)
    if not path.exists():
        path = Path(str(files("envs.certified_uav.scenarios").joinpath(str(path_or_name))))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "base" in payload:
        base_path = Path(str(files("envs.certified_uav.scenarios").joinpath(str(payload["base"]))))
        with base_path.open("r", encoding="utf-8") as handle:
            merged = json.load(handle)
        merged["name"] = payload["name"]
        merged["expected_certificate_outcome"] = payload.get("expected_certificate_outcome", "unspecified")
        merged["configuration_overrides"] = payload.get("configuration_overrides", {})
        if "initial_energy" in payload:
            merged["initial_state"]["energy"] = payload["initial_energy"]
        if payload.get("invalidate_last_corridor_overlap"):
            merged["corridor_cells"][-1]["region_low_xy"] = [1.4, 0.72]
            merged["corridor_cells"][-1]["region_high_xy"] = [1.6, 0.78]
            merged["corridor_cells"][-1]["state_position_low"] = [1.42, 0.73, 0.98]
            merged["corridor_cells"][-1]["state_position_high"] = [1.58, 0.77, 1.025]
        payload = merged
    return _parse_scenario(payload)


class FixedCertificationScenario:
    def __init__(self, name: str = "open_corridor.json") -> None:
        self.definition = load_scenario(name)


class RandomTrainingScenario:
    """Reserved interface; strict testing does not generate random maps."""

    def sample(self, seed: int | None = None) -> ScenarioDefinition:
        del seed
        raise NotImplementedError("random training scenarios are intentionally not implemented in version one")
