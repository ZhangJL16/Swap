from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from envs.certified_uav.scenario import ScenarioDefinition, load_scenario


@dataclass(frozen=True)
class ScenarioFamilyRecord:
    scenario_id: str
    family: str
    split: str
    seed: int
    path: str
    scenario_hash: str
    geometry_hash: str
    certificate_manifest_hash: str | None = None
    certificate_gate: str = "not_evaluated"


def scenario_file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry_hash(payload: dict) -> str:
    geometry = {
        "world_size": payload["world_size"],
        "aabb_obstacles": payload.get("aabb_obstacles", []),
        "cylinder_obstacles": payload.get("cylinder_obstacles", []),
        "free_boxes": payload["mission"].get("free_boxes", []),
        "occupied_boxes": payload["mission"].get("occupied_boxes", []),
    }
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _serialize(definition: ScenarioDefinition) -> dict:
    return {
        "name": definition.name,
        "world_size": definition.world_size.tolist(),
        "initial_state": {
            "position": definition.initial_state.position.tolist(),
            "velocity": definition.initial_state.velocity.tolist(),
            "energy": definition.initial_state.energy,
            "timestamp": definition.initial_state.timestamp,
        },
        "station_position": definition.station_position.tolist(),
        "task_goal": definition.task_goal.tolist(),
        "terminal": {
            "position_low": definition.terminal.position_low.tolist(),
            "position_high": definition.terminal.position_high.tolist(),
            "velocity_abs_max": definition.terminal.velocity_abs_max.tolist(),
            "minimum_energy": definition.terminal.minimum_energy,
            "continuation_modes": list(definition.terminal.continuation_modes),
            "version": definition.terminal.version,
        },
        "aabb_obstacles": [
            {"low": obstacle.low.tolist(), "high": obstacle.high.tolist()}
            for obstacle in definition.world.aabbs
        ],
        "cylinder_obstacles": [
            {
                "center_xy": obstacle.center_xy.tolist(),
                "radius": obstacle.radius,
                "z_low": obstacle.z_low,
                "z_high": obstacle.z_high,
            }
            for obstacle in definition.world.cylinders
        ],
        "corridor_cells": [
            {
                "cell_id": cell.cell_id,
                "level": cell.level,
                "region_low_xy": cell.region_low_xy.tolist(),
                "region_high_xy": cell.region_high_xy.tolist(),
                "altitude_limit": cell.altitude_limit,
                "state_position_low": cell.state_position_low.tolist(),
                "state_position_high": cell.state_position_high.tolist(),
                "state_velocity_low": cell.state_velocity_low.tolist(),
                "state_velocity_high": cell.state_velocity_high.tolist(),
                "energy_low": cell.energy_low,
                "energy_high": cell.energy_high,
            }
            for cell in definition.corridor_cells
        ],
        "bootstrap_lidar_poses": [pose.tolist() for pose in definition.bootstrap_lidar_poses],
        "configuration_overrides": dict(definition.configuration_overrides),
        "mission": dict(definition.mission_config),
        "expected_certificate_outcome": definition.expected_certificate_outcome,
    }


def _inside_obstacle(point: np.ndarray, payload: dict) -> bool:
    for obstacle in payload.get("aabb_obstacles", ()):
        low = np.asarray(obstacle["low"], dtype=np.float64)
        high = np.asarray(obstacle["high"], dtype=np.float64)
        if np.all(point >= low) and np.all(point <= high):
            return True
    return False


def _perturb_obstacles_inside_certified_boxes(payload: dict, rng: np.random.Generator) -> None:
    certified = payload["mission"].get("occupied_boxes", ())
    if len(certified) != len(payload.get("aabb_obstacles", ())):
        return
    for obstacle, certificate_box in zip(payload["aabb_obstacles"], certified):
        certificate_low = np.asarray([certificate_box[0], certificate_box[1]], dtype=np.float64)
        certificate_high = np.asarray([certificate_box[2], certificate_box[3]], dtype=np.float64)
        inset = rng.uniform(0.005, 0.025, size=4)
        obstacle["low"][:2] = (certificate_low + inset[:2]).tolist()
        obstacle["high"][:2] = (certificate_high - inset[2:]).tolist()


def _variant(base_family: str, split: str, index: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    payload = _serialize(load_scenario(f"mission_{base_family}.json"))
    scenario_id = f"{base_family}_{split}_{index:03d}"
    payload["name"] = scenario_id
    initial = np.asarray(payload["initial_state"]["position"], dtype=np.float64)
    if base_family != "narrow":
        initial[:2] += rng.uniform([-0.06, -0.035], [0.06, 0.035])
    payload["initial_state"]["position"] = initial.tolist()
    payload["initial_state"]["velocity"] = rng.uniform([-0.008, -0.008, -0.003], [0.008, 0.008, 0.003]).tolist()
    if base_family == "energy_tight":
        payload["initial_state"]["energy"] = float(rng.uniform(5.35, 5.65))
    else:
        payload["initial_state"]["energy"] = float(rng.uniform(28.0, 31.0))

    goal = np.asarray(payload["task_goal"], dtype=np.float64)
    if base_family != "narrow":
        goal[:2] += rng.uniform([-0.08, -0.08], [0.08, 0.08])
    if _inside_obstacle(goal, payload):
        goal = np.asarray(payload["task_goal"], dtype=np.float64)
    payload["task_goal"] = goal.tolist()
    mission = payload["mission"]
    mission["task_waypoints"][0] = initial.tolist()
    mission["task_waypoints"][-1] = goal.tolist()
    mission.update({
        "scenario_family": base_family,
        "scenario_split": split,
        "scenario_seed": seed,
        "synthetic_disturbance_fraction": float(rng.choice([0.0, 0.25, 0.5, 0.75, 1.0])),
        "evidence_scope": "synthetic scenario-family fixture",
    })
    _perturb_obstacles_inside_certified_boxes(payload, rng)
    return payload


def generate_scenario_splits(
    output_root: str | Path,
    *,
    split_sizes: dict[str, int] | None = None,
    master_seed: int = 20260807,
) -> list[ScenarioFamilyRecord]:
    sizes = {"training": 20, "validation": 10, "heldout": 20} if split_sizes is None else dict(split_sizes)
    root = Path(output_root)
    records: list[ScenarioFamilyRecord] = []
    families = ("open", "obstacle", "narrow", "energy_tight")
    for split_offset, (split, count) in enumerate(sizes.items()):
        split_root = root / split
        split_root.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            family = families[index % len(families)]
            seed = master_seed + split_offset * 10000 + index
            payload = _variant(family, split, index, seed)
            path = split_root / f"{payload['name']}.json"
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            records.append(ScenarioFamilyRecord(
                payload["name"], family, split, seed, str(path), scenario_file_hash(path), _geometry_hash(payload)
            ))
    (root / "scenario_index.json").write_text(
        json.dumps([asdict(record) for record in records], indent=2), encoding="utf-8"
    )
    return records
