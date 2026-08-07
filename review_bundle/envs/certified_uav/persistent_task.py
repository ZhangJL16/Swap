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


class PersistentMissionMode(IntEnum):
    TASK = 0
    VOLUNTARY_RETURN = 1
    FORCED_RETURN = 2
    CHARGING = 3
    FAILURE = 4


class GoalEdgeType(str, Enum):
    TASK_EDGE = "TASK_EDGE"
    RECOVERY_EDGE = "RECOVERY_EDGE"
    DEPARTURE_EDGE = "DEPARTURE_EDGE"


@dataclass(frozen=True, slots=True)
class GoalNode:
    node_id: str
    position: np.ndarray

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("goal-node ID cannot be empty")
        object.__setattr__(self, "position", as_vec3(self.position, "goal-node position"))


@dataclass(frozen=True, slots=True)
class GoalEdge:
    edge_id: str
    source: str
    target: str
    edge_type: GoalEdgeType
    waypoints: tuple[np.ndarray, ...]
    energy_upper: float

    def __post_init__(self) -> None:
        points = tuple(as_vec3(point, "goal-edge waypoint") for point in self.waypoints)
        if len(points) < 2:
            raise ValueError("goal edges require at least two waypoints")
        if not np.isfinite(self.energy_upper) or self.energy_upper <= 0.0:
            raise ValueError("goal-edge energy upper bound must be positive")
        object.__setattr__(self, "edge_type", GoalEdgeType(self.edge_type))
        object.__setattr__(self, "waypoints", points)


class CertifiedGoalNetwork:
    """Finite typed graph separating service, recovery, and departure semantics."""

    def __init__(
        self,
        nodes: Iterable[GoalNode],
        edges: Iterable[GoalEdge],
        charging_station: str,
        goal_node_ids: Iterable[str],
    ) -> None:
        self.nodes = {node.node_id: node for node in nodes}
        self.edges = {edge.edge_id: edge for edge in edges}
        self.charging_station = str(charging_station)
        self.goal_node_ids = tuple(str(node_id) for node_id in goal_node_ids)
        if self.charging_station not in self.nodes:
            raise ValueError("charging station must be a declared network node")
        if self.charging_station in self.goal_node_ids:
            raise ValueError("charging station cannot be a normal task goal")
        if len(self.goal_node_ids) < 2 or len(set(self.goal_node_ids)) != len(self.goal_node_ids):
            raise ValueError("at least two distinct certified goal nodes are required")
        if any(node_id not in self.nodes for node_id in self.goal_node_ids):
            raise ValueError("goal-node list contains an unknown node")
        if not self.edges:
            raise ValueError("certified goal network cannot be empty")
        self._outgoing: dict[str, list[GoalEdge]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges.values():
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"edge {edge.edge_id} references an unknown node")
            if not np.allclose(edge.waypoints[0], self.nodes[edge.source].position):
                raise ValueError(f"edge {edge.edge_id} does not start at its source node")
            if not np.allclose(edge.waypoints[-1], self.nodes[edge.target].position):
                raise ValueError(f"edge {edge.edge_id} does not end at its target node")
            if edge.edge_type == GoalEdgeType.TASK_EDGE and (
                edge.source == self.charging_station or edge.target == self.charging_station
            ):
                raise ValueError("TASK_EDGE cannot use the charging station")
            if edge.edge_type == GoalEdgeType.RECOVERY_EDGE and edge.target != self.charging_station:
                raise ValueError("RECOVERY_EDGE must terminate at the charging station")
            if edge.edge_type == GoalEdgeType.DEPARTURE_EDGE and edge.source != self.charging_station:
                raise ValueError("DEPARTURE_EDGE must start at the charging station")
            self._outgoing[edge.source].append(edge)
        for outgoing in self._outgoing.values():
            outgoing.sort(key=lambda edge: edge.edge_id)
        self._validate_route_closure()
        self.network_hash = self._hash()

    @classmethod
    def from_config(cls, payload: dict[str, Any]) -> "CertifiedGoalNetwork":
        nodes = tuple(
            GoalNode(str(item["node_id"]), np.asarray(item["position"], dtype=np.float64))
            for item in payload["goal_nodes"]
        )
        edges = tuple(
            GoalEdge(
                str(item["edge_id"]),
                str(item["source"]),
                str(item["target"]),
                GoalEdgeType(str(item["edge_type"])),
                tuple(np.asarray(point, dtype=np.float64) for point in item["waypoints"]),
                float(item["energy_upper"]),
            )
            for item in payload["goal_edges"]
        )
        return cls(nodes, edges, str(payload["charging_station"]), tuple(payload["goal_node_ids"]))

    def _hash(self) -> str:
        payload = {
            "nodes": tuple((key, tuple(value.position)) for key, value in sorted(self.nodes.items())),
            "edges": tuple(
                (
                    key,
                    edge.source,
                    edge.target,
                    edge.edge_type.value,
                    tuple(tuple(point) for point in edge.waypoints),
                    edge.energy_upper,
                )
                for key, edge in sorted(self.edges.items())
            ),
            "station": self.charging_station,
            "goals": self.goal_node_ids,
        }
        return sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _validate_route_closure(self) -> None:
        for source in self.goal_node_ids:
            self.shortest_path(source, self.charging_station, {GoalEdgeType.RECOVERY_EDGE})
            self.shortest_path(self.charging_station, source, {GoalEdgeType.DEPARTURE_EDGE})
            for target in self.goal_node_ids:
                if source != target:
                    self.shortest_path(source, target, {GoalEdgeType.TASK_EDGE})

    def nearest_node(self, position: np.ndarray, *, include_station: bool = True) -> str:
        point = as_vec3(position, "network position")
        candidates = tuple(self.nodes) if include_station else self.goal_node_ids
        return min(candidates, key=lambda node_id: float(np.linalg.norm(point - self.nodes[node_id].position)))

    def edges_of_type(self, edge_type: GoalEdgeType) -> tuple[GoalEdge, ...]:
        return tuple(edge for _, edge in sorted(self.edges.items()) if edge.edge_type == edge_type)

    def shortest_path(
        self,
        source: str,
        target: str,
        allowed_types: set[GoalEdgeType] | None = None,
    ) -> tuple[GoalEdge, ...]:
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
                if allowed_types is not None and edge.edge_type not in allowed_types:
                    continue
                candidate = cost + edge.energy_upper
                if candidate + 1e-12 < best.get(edge.target, float("inf")):
                    best[edge.target] = candidate
                    frontier.append((candidate, edge.target, edge_ids + (edge.edge_id,)))
        types = "ANY" if allowed_types is None else ",".join(sorted(item.value for item in allowed_types))
        raise ValueError(f"no certified {types} path from {source} to {target}")

    def path_energy_upper(
        self,
        source: str,
        target: str,
        allowed_types: set[GoalEdgeType] | None = None,
    ) -> float:
        return float(sum(edge.energy_upper for edge in self.shortest_path(source, target, allowed_types)))


@dataclass(slots=True)
class PersistentGoalTask:
    task_id: str
    goal_position: np.ndarray
    reward: float
    deadline_optional: int | None = None
    goal_node: str = ""
    previous_goal: str | None = None
    assignment_step: int = 0
    completion_step: int | None = None
    interrupted_by_charge: bool = False

    def __post_init__(self) -> None:
        self.goal_position = as_vec3(self.goal_position, "persistent goal position")
        if not self.task_id or not self.goal_node:
            raise ValueError("persistent goal task requires task and goal IDs")
        if not np.isfinite(self.reward) or self.reward <= 0.0:
            raise ValueError("persistent goal reward must be positive")


class PersistentGoalTaskManager:
    """Seeded continuous goal stream; it never chooses charging decisions."""

    def __init__(self, network: CertifiedGoalNetwork, goal_radius: float, task_reward: float) -> None:
        if goal_radius <= 0.0 or task_reward <= 0.0:
            raise ValueError("goal radius and task reward must be positive")
        self.network = network
        self.goal_radius = float(goal_radius)
        self.task_reward = float(task_reward)
        self.rng = np.random.default_rng(0)
        self.current_task: PersistentGoalTask | None = None
        self.current_node = network.goal_node_ids[0]
        self.active_route: tuple[GoalEdge, ...] = ()
        self.route_index = 0
        self.tasks_completed = 0
        self.task_interruption_count = 0
        self.task_resume_count = 0
        self._next_task_id = 0
        self.decision_required = False
        self.completed_tasks: list[PersistentGoalTask] = []

    def reset(self, seed: int | None, position: np.ndarray, assignment_step: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.current_node = self.network.nearest_node(position)
        self.current_task = None
        self.active_route = ()
        self.route_index = 0
        self.tasks_completed = 0
        self.task_interruption_count = 0
        self.task_resume_count = 0
        self._next_task_id = 0
        self.decision_required = False
        self.completed_tasks = []
        self.assign_next_goal(assignment_step)

    @property
    def active_edge(self) -> GoalEdge | None:
        return None if self.route_index >= len(self.active_route) else self.active_route[self.route_index]

    @property
    def navigation_target(self) -> np.ndarray:
        edge = self.active_edge
        if edge is not None:
            return self.network.nodes[edge.target].position
        if self.current_task is not None:
            return self.current_task.goal_position
        return self.network.nodes[self.current_node].position

    def accept_service_decision(self) -> None:
        self.decision_required = False

    def _plan_to_pending_goal(self) -> None:
        if self.current_task is None:
            raise RuntimeError("cannot plan without a pending goal")
        edge_type = GoalEdgeType.DEPARTURE_EDGE if self.current_node == self.network.charging_station else GoalEdgeType.TASK_EDGE
        self.active_route = self.network.shortest_path(self.current_node, self.current_task.goal_node, {edge_type})
        self.route_index = 0

    def assign_next_goal(self, assignment_step: int) -> PersistentGoalTask:
        excluded = self.current_node if self.current_node in self.network.goal_node_ids else None
        candidates = [node_id for node_id in self.network.goal_node_ids if node_id != excluded]
        goal_node = candidates[int(self.rng.integers(0, len(candidates)))]
        task = PersistentGoalTask(
            task_id=f"goal-{self._next_task_id}",
            goal_position=self.network.nodes[goal_node].position.copy(),
            reward=self.task_reward,
            deadline_optional=None,
            goal_node=goal_node,
            previous_goal=None if excluded is None else excluded,
            assignment_step=int(assignment_step),
        )
        self._next_task_id += 1
        self.current_task = task
        self._plan_to_pending_goal()
        self.decision_required = True
        return task

    def interrupt_for_charge(self) -> None:
        if self.current_task is None or self.current_task.interrupted_by_charge:
            return
        self.current_task.interrupted_by_charge = True
        self.task_interruption_count += 1
        self.decision_required = False

    def mark_station_arrival(self) -> None:
        self.current_node = self.network.charging_station
        self.active_route = ()
        self.route_index = 0

    def resume_from_station(self) -> None:
        if self.current_task is None:
            raise RuntimeError("cannot leave station without a pending goal")
        self.current_node = self.network.charging_station
        self._plan_to_pending_goal()
        self.task_resume_count += 1
        self.decision_required = False

    def advance(self, position: np.ndarray, step: int) -> dict[str, Any]:
        events: dict[str, Any] = {
            "task_completed": False,
            "task_assigned": False,
            "completed_task_id": None,
            "new_goal_id": None,
        }
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
        if task is None or self.current_node != task.goal_node:
            return events
        task.completion_step = int(step)
        self.completed_tasks.append(task)
        self.tasks_completed += 1
        events["task_completed"] = True
        events["completed_task_id"] = task.task_id
        next_task = self.assign_next_goal(step)
        events["task_assigned"] = True
        events["new_goal_id"] = next_task.goal_node
        return events


@dataclass(frozen=True, slots=True)
class PersistentRewardConfig:
    task_completion_reward: float = 10.0
    elapsed_time_cost: float = 0.01
    flight_energy_cost: float = 0.1
    charging_dwell_cost: float = 0.01
    forced_return_interruption_cost: float = 1.0


class PersistentGoalWrapper(gym.Wrapper):
    """Continuous goal-stream semantics; certificate quantities remain explicit inputs."""

    multi_step_mission = False

    def __init__(
        self,
        plant: CertifiedSingleUAVPlantEnv,
        network: CertifiedGoalNetwork,
        reward_config: PersistentRewardConfig | None = None,
        goal_radius: float = 0.20,
        task_reward: float = 10.0,
    ) -> None:
        super().__init__(plant)
        self.plant = plant
        self.network = network
        self.manager = PersistentGoalTaskManager(network, goal_radius, task_reward)
        self.reward_config = reward_config or PersistentRewardConfig(task_completion_reward=task_reward)
        self.mode = PersistentMissionMode.TASK
        self.phase = self.mode
        self.required_return_energy = 0.0
        self.energy_margin = 0.0
        self.last_events: dict[str, Any] = {}
        self.episode_step = 0
        self.observation_layout: dict[str, slice] = {}
        cursor = 0
        for name, length in (
            ("position", 3),
            ("velocity", 3),
            ("energy", 1),
            ("goal_delta", 3),
            ("station_delta", 3),
            ("required_return_energy", 1),
            ("energy_margin", 1),
            ("mission_mode", len(PersistentMissionMode)),
            ("charging", 1),
            ("state_of_charge", 1),
            ("tasks_completed", 1),
            ("lidar_distances", plant.config.num_lasers),
            ("lidar_valid", plant.config.num_lasers),
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
        if self.mode in {
            PersistentMissionMode.VOLUNTARY_RETURN,
            PersistentMissionMode.FORCED_RETURN,
            PersistentMissionMode.CHARGING,
        }:
            return self.plant.scenario.station_position
        return self.manager.navigation_target

    def set_certificate_quantities(self, required_return_energy: float, energy_margin: float) -> None:
        self.required_return_energy = float(required_return_energy)
        self.energy_margin = float(energy_margin)

    def on_runtime_recovery(self, reason: str) -> None:
        del reason
        if self.mode == PersistentMissionMode.TASK:
            self.request_return(forced=True)

    def request_return(self, forced: bool) -> None:
        if self.mode == PersistentMissionMode.TASK:
            self.manager.interrupt_for_charge()
        self.mode = PersistentMissionMode.FORCED_RETURN if forced else PersistentMissionMode.VOLUNTARY_RETURN
        self.phase = self.mode

    def enter_charging(self) -> None:
        self.manager.mark_station_arrival()
        self.mode = PersistentMissionMode.CHARGING
        self.phase = self.mode

    def leave_station(self) -> None:
        self.manager.resume_from_station()
        self.mode = PersistentMissionMode.TASK
        self.phase = self.mode

    def build_observation(
        self,
        local_map_crop_encoding: np.ndarray | None = None,
        corridor_encoding: np.ndarray | None = None,
    ) -> np.ndarray:
        lidar = self.plant.last_lidar
        if lidar is None:
            raise RuntimeError("LiDAR packet is unavailable")
        task = self.manager.current_task
        goal = self.plant.state.position if task is None else task.goal_position
        local_map = (
            np.zeros(self.plant.config.local_map_encoding_size)
            if local_map_crop_encoding is None
            else np.asarray(local_map_crop_encoding, dtype=np.float64)
        )
        corridor = (
            np.zeros(self.plant.config.corridor_encoding_size)
            if corridor_encoding is None
            else np.asarray(corridor_encoding, dtype=np.float64)
        )
        mode_vector = np.eye(len(PersistentMissionMode), dtype=np.float64)[int(self.mode)]
        state = self.plant.state
        capacity = float(self.plant.config.initial_energy)
        observation = np.concatenate((
            state.position / self.plant.config.world_size,
            state.velocity / self.plant.config.v_max,
            np.array([state.energy / capacity]),
            (goal - state.position) / self.plant.config.world_size,
            (self.plant.scenario.station_position - state.position) / self.plant.config.world_size,
            np.array([self.required_return_energy / capacity, self.energy_margin / capacity]),
            mode_vector,
            np.array([
                float(self.mode == PersistentMissionMode.CHARGING),
                state.energy / capacity,
                self.manager.tasks_completed / 100.0,
            ]),
            lidar.distances / self.plant.config.lidar_range,
            lidar.valid.astype(np.float64),
            local_map,
            corridor,
        ))
        return np.clip(observation, self.observation_space.low, self.observation_space.high).astype(np.float32)

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _, info = self.plant.reset(seed=seed, options=options)
        self.manager.reset(seed, self.plant.state.position)
        self.mode = PersistentMissionMode.TASK
        self.phase = self.mode
        self.episode_step = 0
        self.last_events = {}
        return self.build_observation(), info | {"observation_layout": dict(self.observation_layout)}

    def step(self, action):
        _, _, terminated, truncated, info = self.plant.step(action)
        telemetry = info["telemetry"]
        self.episode_step += 1
        events = {
            "task_completed": False,
            "task_assigned": False,
            "completed_task_id": None,
            "new_goal_id": None,
        }
        if self.mode == PersistentMissionMode.TASK:
            events = self.manager.advance(self.plant.state.position, self.episode_step)
        elif self.mode in {PersistentMissionMode.VOLUNTARY_RETURN, PersistentMissionMode.FORCED_RETURN} and telemetry.terminal_admissible:
            self.enter_charging()
        if terminated or info.get("failure_reason"):
            self.mode = PersistentMissionMode.FAILURE
            self.phase = self.mode
        reward = (
            self.reward_config.task_completion_reward * float(events["task_completed"])
            - self.reward_config.elapsed_time_cost
            - self.reward_config.flight_energy_cost * telemetry.energy_cost
        )
        self.last_events = events
        task = self.manager.current_task
        return self.build_observation(), reward, terminated, truncated, info | {
            "persistent_mode": self.mode.name,
            "task_id": None if task is None else task.task_id,
            "current_goal_id": None if task is None else task.goal_node,
            "task_completed_now": events["task_completed"],
            "task_assigned_now": events["task_assigned"],
            "completed_task_id": events["completed_task_id"],
            "tasks_completed": self.manager.tasks_completed,
            "episode_step": self.episode_step,
        }


# Compatibility names are aliases only; the persistent path has no pickup/dropoff state.
PersistentTask = PersistentGoalTask
PersistentTaskManager = PersistentGoalTaskManager
PersistentTaskWrapper = PersistentGoalWrapper
CertifiedServiceNetwork = CertifiedGoalNetwork
ServiceNode = GoalNode
ServiceEdge = GoalEdge
