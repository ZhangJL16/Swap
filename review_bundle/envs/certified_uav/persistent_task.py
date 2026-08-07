from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from hashlib import sha256
import json
from typing import Any, Iterable

import gymnasium as gym
import numpy as np

from .plant_env import CertifiedSingleUAVPlantEnv
from .state import as_vec3


class PersistentTaskStatus(str, Enum):
    PENDING = "PENDING"
    TO_PICKUP = "TO_PICKUP"
    CARRYING = "CARRYING"
    COMPLETED = "COMPLETED"
    PAUSED_FOR_CHARGE = "PAUSED_FOR_CHARGE"


class PersistentMissionMode(IntEnum):
    TO_PICKUP = 0
    TO_DROPOFF = 1
    VOLUNTARY_RETURN = 2
    FORCED_RETURN = 3
    CHARGING = 4
    FAILURE = 5


@dataclass(frozen=True, slots=True)
class ServiceNode:
    node_id: str
    position: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", as_vec3(self.position, "service-node position"))


@dataclass(frozen=True, slots=True)
class ServiceEdge:
    edge_id: str
    source: str
    target: str
    task_waypoints: tuple[np.ndarray, ...]
    return_waypoints: tuple[np.ndarray, ...]
    energy_upper: float

    def __post_init__(self) -> None:
        task = tuple(as_vec3(point, "task waypoint") for point in self.task_waypoints)
        recovery = tuple(as_vec3(point, "return waypoint") for point in self.return_waypoints)
        if len(task) < 2 or len(recovery) < 2:
            raise ValueError("service edges require task and recovery waypoint chains")
        if not np.isfinite(self.energy_upper) or self.energy_upper <= 0.0:
            raise ValueError("service-edge energy upper bound must be positive")
        object.__setattr__(self, "task_waypoints", task)
        object.__setattr__(self, "return_waypoints", recovery)


class CertifiedServiceNetwork:
    """Finite directed service graph whose edges carry independent certificate profiles."""

    def __init__(
        self,
        nodes: Iterable[ServiceNode],
        edges: Iterable[ServiceEdge],
        charging_station: str,
        task_edge_ids: Iterable[str],
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = {edge.edge_id: edge for edge in edges}
        self.charging_station = charging_station
        self.task_edge_ids = tuple(task_edge_ids)
        if charging_station not in self.nodes:
            raise ValueError("charging station must be a service node")
        if not self.nodes or not self.edges or not self.task_edge_ids:
            raise ValueError("service network cannot be empty")
        for edge in self.edges.values():
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            if not np.allclose(edge.task_waypoints[0], self.nodes[edge.source].position):
                raise ValueError(f"edge {edge.edge_id} does not start at its source node")
            if not np.allclose(edge.task_waypoints[-1], self.nodes[edge.target].position):
                raise ValueError(f"edge {edge.edge_id} does not end at its target node")
        if any(edge_id not in self.edges for edge_id in self.task_edge_ids):
            raise ValueError("task edge list contains an unknown edge")
        self._outgoing: dict[str, list[ServiceEdge]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            self._outgoing[edge.source].append(edge)
        for outgoing in self._outgoing.values():
            outgoing.sort(key=lambda edge: edge.edge_id)
        self.network_hash = self._hash()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "CertifiedServiceNetwork":
        nodes = tuple(ServiceNode(str(item["node_id"]), np.asarray(item["position"], dtype=np.float64)) for item in payload["service_nodes"])
        edges = tuple(
            ServiceEdge(
                str(item["edge_id"]),
                str(item["source"]),
                str(item["target"]),
                tuple(np.asarray(point, dtype=np.float64) for point in item["task_waypoints"]),
                tuple(np.asarray(point, dtype=np.float64) for point in item["return_waypoints"]),
                float(item["energy_upper"]),
            )
            for item in payload["service_edges"]
        )
        return cls(nodes, edges, str(payload["charging_station"]), tuple(payload["task_edge_ids"]))

    def _hash(self) -> str:
        payload = {
            "nodes": tuple((key, tuple(value.position)) for key, value in sorted(self.nodes.items())),
            "edges": tuple(
                (
                    key,
                    edge.source,
                    edge.target,
                    tuple(tuple(point) for point in edge.task_waypoints),
                    tuple(tuple(point) for point in edge.return_waypoints),
                    edge.energy_upper,
                )
                for key, edge in sorted(self.edges.items())
            ),
            "station": self.charging_station,
            "task_edges": self.task_edge_ids,
        }
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def nearest_node(self, position: np.ndarray) -> str:
        point = as_vec3(position, "service position")
        return min(self.nodes, key=lambda node_id: float(np.linalg.norm(point - self.nodes[node_id].position)))

    def shortest_path(self, source: str, target: str) -> tuple[ServiceEdge, ...]:
        if source == target:
            return ()
        frontier: list[tuple[float, str, tuple[str, ...]]] = [(0.0, source, ())]
        best = {source: 0.0}
        while frontier:
            frontier.sort(key=lambda item: (item[0], item[1], item[2]))
            cost, node_id, edge_ids = frontier.pop(0)
            if node_id == target:
                return tuple(self.edges[edge_id] for edge_id in edge_ids)
            if cost > best.get(node_id, float("inf")) + 1e-12:
                continue
            for edge in self._outgoing[node_id]:
                candidate = cost + edge.energy_upper
                if candidate + 1e-12 < best.get(edge.target, float("inf")):
                    best[edge.target] = candidate
                    frontier.append((candidate, edge.target, edge_ids + (edge.edge_id,)))
        raise ValueError(f"no certified service path from {source} to {target}")

    def path_energy_upper(self, source: str, target: str) -> float:
        return float(sum(edge.energy_upper for edge in self.shortest_path(source, target)))


@dataclass(slots=True)
class PersistentTask:
    task_id: str
    pickup_node: str
    dropoff_node: str
    pickup_position: np.ndarray
    dropoff_position: np.ndarray
    reward: float
    deadline_optional: int | None
    status: PersistentTaskStatus = PersistentTaskStatus.PENDING

    def __post_init__(self) -> None:
        self.pickup_position = as_vec3(self.pickup_position, "pickup position")
        self.dropoff_position = as_vec3(self.dropoff_position, "dropoff position")


class PersistentTaskManager:
    def __init__(self, network: CertifiedServiceNetwork, goal_radius: float, task_reward: float) -> None:
        if goal_radius <= 0.0 or task_reward <= 0.0:
            raise ValueError("task radius and reward must be positive")
        self.network = network
        self.goal_radius = float(goal_radius)
        self.task_reward = float(task_reward)
        self.rng = np.random.default_rng(0)
        self.current_task: PersistentTask | None = None
        self.current_node = network.charging_station
        self.active_route: tuple[ServiceEdge, ...] = ()
        self.route_index = 0
        self.tasks_completed = 0
        self.pickup_count = 0
        self.delivery_count = 0
        self.task_pause_count = 0
        self.task_resume_count = 0
        self._next_task_id = 0
        self._paused_status: PersistentTaskStatus | None = None
        self.decision_required = False

    def reset(self, seed: int | None, position: np.ndarray) -> None:
        self.rng = np.random.default_rng(seed)
        self.current_node = self.network.nearest_node(position)
        self.current_task = None
        self.active_route = ()
        self.route_index = 0
        self.tasks_completed = self.pickup_count = self.delivery_count = 0
        self.task_pause_count = self.task_resume_count = 0
        self._next_task_id = 0
        self._paused_status = None
        self.assign_next_task()

    @property
    def active_edge(self) -> ServiceEdge | None:
        return None if self.route_index >= len(self.active_route) else self.active_route[self.route_index]

    @property
    def paused_status(self) -> PersistentTaskStatus | None:
        return self._paused_status

    def accept_service_decision(self) -> None:
        self.decision_required = False

    @property
    def navigation_target(self) -> np.ndarray:
        edge = self.active_edge
        if edge is not None:
            return self.network.nodes[edge.target].position
        if self.current_task is None:
            return self.network.nodes[self.current_node].position
        if self.current_task.status in {PersistentTaskStatus.TO_PICKUP, PersistentTaskStatus.PAUSED_FOR_CHARGE}:
            return self.current_task.pickup_position
        return self.current_task.dropoff_position

    def _plan(self, target_node: str) -> None:
        self.active_route = self.network.shortest_path(self.current_node, target_node)
        self.route_index = 0

    def assign_next_task(self) -> PersistentTask:
        candidates = [
            self.network.edges[edge_id]
            for edge_id in self.network.task_edge_ids
            if self.network.edges[edge_id].source == self.current_node
        ]
        if not candidates:
            candidates = [self.network.edges[edge_id] for edge_id in self.network.task_edge_ids]
        edge = candidates[int(self.rng.integers(0, len(candidates)))]
        task = PersistentTask(
            f"task-{self._next_task_id}",
            edge.source,
            edge.target,
            self.network.nodes[edge.source].position.copy(),
            self.network.nodes[edge.target].position.copy(),
            self.task_reward,
            None,
            PersistentTaskStatus.TO_PICKUP,
        )
        self._next_task_id += 1
        self.current_task = task
        self._plan(task.pickup_node)
        if self.current_node == task.pickup_node:
            task.status = PersistentTaskStatus.CARRYING
            self.pickup_count += 1
            self._plan(task.dropoff_node)
        self.decision_required = True
        return task

    def pause_for_charge(self) -> None:
        if self.current_task is None or self.current_task.status == PersistentTaskStatus.PAUSED_FOR_CHARGE:
            return
        self._paused_status = self.current_task.status
        self.current_task.status = PersistentTaskStatus.PAUSED_FOR_CHARGE
        self.task_pause_count += 1
        self.decision_required = False

    def resume_from_station(self) -> None:
        if self.current_task is None or self.current_task.status != PersistentTaskStatus.PAUSED_FOR_CHARGE:
            return
        self.current_node = self.network.charging_station
        restored = self._paused_status or PersistentTaskStatus.TO_PICKUP
        self.current_task.status = restored
        self._paused_status = None
        self.task_resume_count += 1
        self.decision_required = False
        if restored == PersistentTaskStatus.TO_PICKUP and self.current_node == self.current_task.pickup_node:
            self.current_task.status = PersistentTaskStatus.CARRYING
            self.pickup_count += 1
            self._plan(self.current_task.dropoff_node)
        elif restored == PersistentTaskStatus.CARRYING and self.current_node == self.current_task.dropoff_node:
            self.current_task.status = PersistentTaskStatus.COMPLETED
            self.delivery_count += 1
            self.tasks_completed += 1
            self.assign_next_task()
        else:
            target = self.current_task.pickup_node if restored == PersistentTaskStatus.TO_PICKUP else self.current_task.dropoff_node
            self._plan(target)

    def advance(self, position: np.ndarray) -> dict[str, bool]:
        events = {"pickup": False, "delivery": False, "task_assigned": False}
        edge = self.active_edge
        if edge is None:
            return events
        target = self.network.nodes[edge.target].position
        if float(np.linalg.norm(as_vec3(position, "position") - target)) > self.goal_radius:
            return events
        self.current_node = edge.target
        self.route_index += 1
        if self.route_index < len(self.active_route):
            return events
        task = self.current_task
        if task is None:
            return events
        if task.status == PersistentTaskStatus.TO_PICKUP:
            task.status = PersistentTaskStatus.CARRYING
            self.pickup_count += 1
            events["pickup"] = True
            self._plan(task.dropoff_node)
        elif task.status == PersistentTaskStatus.CARRYING:
            task.status = PersistentTaskStatus.COMPLETED
            self.delivery_count += 1
            self.tasks_completed += 1
            events["delivery"] = True
            self.assign_next_task()
            events["task_assigned"] = True
        return events


@dataclass(frozen=True, slots=True)
class PersistentRewardConfig:
    pickup_reward: float = 1.0
    delivery_reward: float = 10.0
    elapsed_time_cost: float = 0.01
    flight_energy_cost: float = 0.1
    charging_dwell_cost: float = 0.01
    forced_return_interruption_cost: float = 1.0


class PersistentTaskWrapper(gym.Wrapper):
    """Persistent task semantics; certificate quantities remain explicit inputs."""

    multi_step_mission = False

    def __init__(
        self,
        plant: CertifiedSingleUAVPlantEnv,
        network: CertifiedServiceNetwork,
        reward_config: PersistentRewardConfig | None = None,
        goal_radius: float = 0.20,
        task_reward: float = 10.0,
    ) -> None:
        super().__init__(plant)
        self.plant = plant
        self.network = network
        self.manager = PersistentTaskManager(network, goal_radius, task_reward)
        self.reward_config = reward_config or PersistentRewardConfig()
        self.mode = PersistentMissionMode.TO_PICKUP
        self.phase = self.mode
        self.required_return_energy = 0.0
        self.energy_margin = 0.0
        self.last_events: dict[str, bool] = {}
        self.episode_step = 0
        self.observation_layout: dict[str, slice] = {}
        cursor = 0
        for name, length in (
            ("position", 3), ("velocity", 3), ("energy", 1),
            ("pickup_delta", 3), ("dropoff_delta", 3), ("station_delta", 3),
            ("energy_margin", 1), ("required_return_energy", 1),
            ("task_stage", len(PersistentTaskStatus)), ("mission_mode", len(PersistentMissionMode)),
            ("charging", 1), ("state_of_charge", 1), ("tasks_completed", 1),
            ("lidar_distances", plant.config.num_lasers), ("lidar_valid", plant.config.num_lasers),
            ("local_map_crop", plant.config.local_map_encoding_size),
            ("corridor", plant.config.corridor_encoding_size),
        ):
            self.observation_layout[name] = slice(cursor, cursor + length)
            cursor += length
        self.observation_space = gym.spaces.Box(
            np.full(cursor, -2.0, dtype=np.float32),
            np.full(cursor, 2.0, dtype=np.float32),
            dtype=np.float32,
        )

    @property
    def active_goal(self) -> np.ndarray:
        if self.mode in {PersistentMissionMode.VOLUNTARY_RETURN, PersistentMissionMode.FORCED_RETURN, PersistentMissionMode.CHARGING}:
            return self.plant.scenario.station_position
        return self.manager.navigation_target

    def set_certificate_quantities(self, required_return_energy: float, energy_margin: float) -> None:
        self.required_return_energy = float(required_return_energy)
        self.energy_margin = float(energy_margin)

    def on_runtime_recovery(self, reason: str) -> None:
        del reason
        if self.mode in {PersistentMissionMode.TO_PICKUP, PersistentMissionMode.TO_DROPOFF}:
            self.manager.pause_for_charge()
            self.mode = PersistentMissionMode.FORCED_RETURN
            self.phase = self.mode

    def request_return(self, forced: bool) -> None:
        self.manager.pause_for_charge()
        self.mode = PersistentMissionMode.FORCED_RETURN if forced else PersistentMissionMode.VOLUNTARY_RETURN
        self.phase = self.mode

    def enter_charging(self) -> None:
        self.mode = PersistentMissionMode.CHARGING
        self.phase = self.mode

    def leave_station(self) -> None:
        self.manager.resume_from_station()
        task = self.manager.current_task
        self.mode = PersistentMissionMode.TO_PICKUP if task is None or task.status == PersistentTaskStatus.TO_PICKUP else PersistentMissionMode.TO_DROPOFF
        self.phase = self.mode

    def build_observation(self, local_map_crop_encoding: np.ndarray | None = None, corridor_encoding: np.ndarray | None = None) -> np.ndarray:
        lidar = self.plant.last_lidar
        if lidar is None:
            raise RuntimeError("LiDAR packet is unavailable")
        task = self.manager.current_task
        pickup = self.plant.state.position if task is None else task.pickup_position
        dropoff = self.plant.state.position if task is None else task.dropoff_position
        local_map = np.zeros(self.plant.config.local_map_encoding_size) if local_map_crop_encoding is None else np.asarray(local_map_crop_encoding, dtype=np.float64)
        corridor = np.zeros(self.plant.config.corridor_encoding_size) if corridor_encoding is None else np.asarray(corridor_encoding, dtype=np.float64)
        task_status = PersistentTaskStatus.PENDING if task is None else task.status
        status_vector = np.zeros(len(PersistentTaskStatus), dtype=np.float64)
        status_vector[list(PersistentTaskStatus).index(task_status)] = 1.0
        mode_vector = np.eye(len(PersistentMissionMode), dtype=np.float64)[int(self.mode)]
        state = self.plant.state
        capacity = float(self.plant.config.initial_energy)
        observation = np.concatenate((
            state.position / self.plant.config.world_size,
            state.velocity / self.plant.config.v_max,
            np.array([state.energy / capacity]),
            (pickup - state.position) / self.plant.config.world_size,
            (dropoff - state.position) / self.plant.config.world_size,
            (self.plant.scenario.station_position - state.position) / self.plant.config.world_size,
            np.array([self.energy_margin / capacity, self.required_return_energy / capacity]),
            status_vector,
            mode_vector,
            np.array([float(self.mode == PersistentMissionMode.CHARGING), state.energy / capacity, self.manager.tasks_completed / 100.0]),
            lidar.distances / self.plant.config.lidar_range,
            lidar.valid.astype(np.float64),
            local_map,
            corridor,
        ))
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = self.plant.reset(seed=seed, options=options)
        self.manager.reset(seed, self.plant.state.position)
        task = self.manager.current_task
        self.mode = PersistentMissionMode.TO_PICKUP if task is None or task.status == PersistentTaskStatus.TO_PICKUP else PersistentMissionMode.TO_DROPOFF
        self.phase = self.mode
        self.episode_step = 0
        self.last_events = {}
        return self.build_observation(), info | {"observation_layout": dict(self.observation_layout)}

    def step(self, action):
        _, _, terminated, truncated, info = self.plant.step(action)
        telemetry = info["telemetry"]
        self.episode_step += 1
        events = {"pickup": False, "delivery": False, "task_assigned": False}
        if self.mode in {PersistentMissionMode.TO_PICKUP, PersistentMissionMode.TO_DROPOFF}:
            events = self.manager.advance(self.plant.state.position)
            task = self.manager.current_task
            self.mode = PersistentMissionMode.TO_PICKUP if task is None or task.status == PersistentTaskStatus.TO_PICKUP else PersistentMissionMode.TO_DROPOFF
            self.phase = self.mode
        elif self.mode in {PersistentMissionMode.VOLUNTARY_RETURN, PersistentMissionMode.FORCED_RETURN} and telemetry.terminal_admissible:
            self.enter_charging()
        if terminated or info.get("failure_reason"):
            self.mode = PersistentMissionMode.FAILURE
            self.phase = self.mode
        reward = (
            self.reward_config.pickup_reward * float(events["pickup"])
            + self.reward_config.delivery_reward * float(events["delivery"])
            - self.reward_config.elapsed_time_cost
            - self.reward_config.flight_energy_cost * telemetry.energy_cost
        )
        self.last_events = events
        return self.build_observation(), reward, terminated, truncated, info | {
            "persistent_mode": self.mode.name,
            "task_status": None if self.manager.current_task is None else self.manager.current_task.status.value,
            "task_id": None if self.manager.current_task is None else self.manager.current_task.task_id,
            "pickup_now": events["pickup"],
            "delivery_now": events["delivery"],
            "task_assigned_now": events["task_assigned"],
            "tasks_completed": self.manager.tasks_completed,
            "episode_step": self.episode_step,
        }
