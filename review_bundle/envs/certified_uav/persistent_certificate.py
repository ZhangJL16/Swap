from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from cert_runtime.certificates import certificate_hash
from cert_runtime.interval import round_up
from cert_runtime.types import Interval3

from .mission_certificate import MissionActionContext, MultiStepSyntheticMissionCertificateProvider
from .persistent_task import CertifiedServiceNetwork, PersistentTask, PersistentTaskStatus, ServiceEdge


@dataclass(frozen=True, slots=True)
class PersistentServiceEdgeCertificate:
    edge_id: str
    source: str
    target: str
    recovery_manifest_hash: str
    recovery_gate_pass: bool
    task_support_complete: bool
    departure_energy_upper: float
    dependency_hashes: tuple[str, ...]
    certificate_hash: str


@dataclass(frozen=True, slots=True)
class PersistentCertificateManifest:
    scenario_id: str
    service_network_hash: str
    edge_certificates: tuple[PersistentServiceEdgeCertificate, ...]
    docking_valid: bool
    energy_recursion_valid: bool
    departure_gate_valid: bool
    task_switching_valid: bool
    charging_separated_from_return_energy: bool
    version_consistent: bool
    gate_pass: bool
    failure_reasons: tuple[str, ...]
    manifest_hash: str


class PersistentCertificateProvider:
    """Finite persistent service certificate composed from edge-level T4a manifests."""

    version = "persistent-service-certificate-v1"

    def __init__(self, runtime: Any, network: CertifiedServiceNetwork, battery_capacity: float) -> None:
        self.runtime = runtime
        self.network = network
        self.battery_capacity = float(battery_capacity)
        self.providers: dict[str, MultiStepSyntheticMissionCertificateProvider] = {}
        self.edge_energy_upper: dict[str, float] = {}
        self.active_edge_id = sorted(network.edges)[0]
        failures: list[str] = []
        certificates: list[PersistentServiceEdgeCertificate] = []
        for edge_id, edge in sorted(network.edges.items()):
            provider = self._build_edge_provider(edge)
            self.providers[edge_id] = provider
            certified_energy_upper = self._edge_energy_bound(provider)
            self.edge_energy_upper[edge_id] = certified_energy_upper
            complete = bool(provider.manifest.task_transition_verified and all(provider.manifest.task_transition_verified))
            certificate_payload = {
                "edge": edge_id,
                "source": edge.source,
                "target": edge.target,
                "manifest": provider.manifest.manifest_hash,
                "gate": provider.gate_pass,
                "task_support": complete,
                "energy": certified_energy_upper,
                "network": network.network_hash,
            }
            edge_hash = certificate_hash(certificate_payload)
            certificates.append(
                PersistentServiceEdgeCertificate(
                    edge_id,
                    edge.source,
                    edge.target,
                    provider.manifest.manifest_hash,
                    provider.gate_pass,
                    complete,
                    certified_energy_upper,
                    (provider.manifest.manifest_hash, network.network_hash),
                    edge_hash,
                )
            )
            if not provider.gate_pass:
                failures.append(f"EDGE_RECOVERY_GATE_FAILED:{edge_id}")
            if not complete:
                failures.append(f"EDGE_TASK_SUPPORT_INCOMPLETE:{edge_id}")
        docking_valid = bool(
            runtime.scenario.terminal.is_charge_admissible(
                replace(runtime.scenario.initial_state, position=runtime.scenario.station_position.copy(), velocity=np.zeros(3), energy=max(runtime.scenario.terminal.minimum_energy, 1.0))
            )
        )
        energy_valid = all(
            cell.e3_residual >= -1e-12 and cell.energy_upper >= 0.0
            for provider in self.providers.values()
            for cell in provider.manifest.cells
        )
        departure_valid = self._all_departures_fit_capacity()
        task_switching_valid = self._task_switching_valid()
        version_consistent = len({provider.runtime.recovery_policy.config.parameter_version for provider in self.providers.values()}) == 1
        checks = {
            "DOCKING_INVALID": docking_valid,
            "ENERGY_RECURSION_INVALID": energy_valid,
            "DEPARTURE_GATE_INVALID": departure_valid,
            "TASK_SWITCH_GRAPH_INVALID": task_switching_valid,
            "VERSION_MISMATCH": version_consistent,
        }
        failures.extend(name for name, valid in checks.items() if not valid)
        gate_pass = not failures
        manifest_payload = {
            "scenario": runtime.scenario.name,
            "network": network.network_hash,
            "edges": tuple(certificate.certificate_hash for certificate in certificates),
            "docking": docking_valid,
            "energy": energy_valid,
            "departure": departure_valid,
            "switching": task_switching_valid,
            "charging_separated": True,
            "versions": version_consistent,
            "failures": tuple(failures),
        }
        self.persistent_manifest = PersistentCertificateManifest(
            runtime.scenario.name,
            network.network_hash,
            tuple(certificates),
            docking_valid,
            energy_valid,
            departure_valid,
            task_switching_valid,
            True,
            version_consistent,
            gate_pass,
            tuple(failures),
            certificate_hash(manifest_payload),
        )
        self.last_context: MissionActionContext | None = None

    def _edge_energy_bound(self, provider: MultiStepSyntheticMissionCertificateProvider) -> float:
        action = Interval3(-self.runtime.config.a_max, self.runtime.config.a_max)
        velocity = Interval3(-self.runtime.config.v_max, self.runtime.config.v_max)
        task_step_upper = self.runtime.envelope_builder.energy.cost_upper(action, velocity)
        task_transition_count = max(0, len(provider.task_reference) - 1)
        task_upper = 0.0
        for _ in range(task_transition_count):
            task_upper = round_up(task_upper + task_step_upper)
        recovery_upper = max((cell.energy_upper for cell in provider.root_cells), default=float("inf"))
        return round_up(task_upper + recovery_upper)

    def path_energy_upper(self, source: str, target: str) -> float:
        if source == target:
            return 0.0
        frontier: list[tuple[float, str]] = [(0.0, source)]
        best = {source: 0.0}
        while frontier:
            frontier.sort(key=lambda item: (item[0], item[1]))
            cost, node_id = frontier.pop(0)
            if node_id == target:
                return float(cost)
            if cost > best.get(node_id, float("inf")) + 1e-12:
                continue
            for edge in self.network._outgoing[node_id]:
                candidate = round_up(cost + self.edge_energy_upper[edge.edge_id])
                if candidate + 1e-12 < best.get(edge.target, float("inf")):
                    best[edge.target] = candidate
                    frontier.append((candidate, edge.target))
        raise ValueError(f"no certified energy path from {source} to {target}")

    def _all_departures_fit_capacity(self) -> bool:
        station = self.network.charging_station
        try:
            for edge_id in self.network.task_edge_ids:
                edge = self.network.edges[edge_id]
                required = (
                    self.path_energy_upper(station, edge.source)
                    + self.edge_energy_upper[edge_id]
                    + self.path_energy_upper(edge.target, station)
                    + self.runtime.scenario.terminal.minimum_energy
                )
                if not np.isfinite(required) or required > self.battery_capacity:
                    return False
            return True
        except ValueError:
            return False

    def _build_edge_provider(self, edge: ServiceEdge) -> MultiStepSyntheticMissionCertificateProvider:
        profile = dict(self.runtime.scenario.mission_config)
        profile["task_waypoints"] = [point.tolist() for point in edge.task_waypoints]
        profile["return_waypoints"] = [point.tolist() for point in edge.return_waypoints]
        profile["persistent_edge_id"] = edge.edge_id
        scenario = replace(
            self.runtime.scenario,
            name=f"{self.runtime.scenario.name}:{edge.edge_id}",
            initial_state=replace(
                self.runtime.scenario.initial_state,
                position=self.network.nodes[edge.source].position.copy(),
                velocity=np.zeros(3, dtype=np.float64),
                energy=self.battery_capacity,
                timestamp=0.0,
            ),
            task_goal=self.network.nodes[edge.target].position.copy(),
            mission_config=profile,
        )
        proxy = SimpleNamespace(
            scenario=scenario,
            config=self.runtime.config,
            calibration=self.runtime.calibration,
            envelope_builder=self.runtime.envelope_builder,
            recovery_policy=self.runtime.recovery_policy,
        )
        return MultiStepSyntheticMissionCertificateProvider(proxy, self.runtime.generator_center_mode)

    def _task_switching_valid(self) -> bool:
        station = self.network.charging_station
        try:
            for edge_id in self.network.task_edge_ids:
                edge = self.network.edges[edge_id]
                self.network.shortest_path(station, edge.source)
                self.network.shortest_path(edge.target, station)
            return True
        except ValueError:
            return False

    @property
    def manifest(self):
        return SimpleNamespace(manifest_hash=self.persistent_manifest.manifest_hash)

    @property
    def gate_pass(self) -> bool:
        return self.persistent_manifest.gate_pass

    @property
    def trigger_margin(self) -> float:
        return self.providers[self.active_edge_id].trigger_margin

    @property
    def recovery_active(self) -> bool:
        return self.providers[self.active_edge_id].recovery_active

    def activate_edge(self, edge_id: str) -> None:
        if edge_id not in self.providers:
            raise ValueError(f"edge {edge_id} has no persistent certificate")
        self.active_edge_id = edge_id

    def reset(self) -> None:
        for provider in self.providers.values():
            provider.reset()
        self.last_context = None

    def evaluate(self, state, timestamp: float | None = None) -> MissionActionContext:
        context = self.providers[self.active_edge_id].evaluate(state, timestamp)
        self.last_context = context
        return context

    def commit_execution(self, context: MissionActionContext, task_action_executed: bool) -> None:
        self.providers[self.active_edge_id].commit_execution(context, task_action_executed)
        self.last_context = self.providers[self.active_edge_id].last_context

    def verify_task_action(self, state, action: np.ndarray) -> bool:
        return self.providers[self.active_edge_id].verify_task_action(state, action)

    def required_departure_energy(self, task: PersistentTask | None, paused_status: PersistentTaskStatus | None = None) -> float:
        station = self.network.charging_station
        if task is None:
            return 0.0
        status = task.status if paused_status is None else paused_status
        if status == PersistentTaskStatus.CARRYING and task.dropoff_node == station:
            next_task_requirements = []
            for edge_id in self.network.task_edge_ids:
                edge = self.network.edges[edge_id]
                if edge.source != station:
                    continue
                next_task_requirements.append(
                    self.edge_energy_upper[edge_id]
                    + self.path_energy_upper(edge.target, station)
                )
            route = max(next_task_requirements, default=0.0)
            return float(route + self.runtime.scenario.terminal.minimum_energy)
        if status in {PersistentTaskStatus.TO_PICKUP, PersistentTaskStatus.PAUSED_FOR_CHARGE}:
            route = (
                self.path_energy_upper(station, task.pickup_node)
                + self.path_energy_upper(task.pickup_node, task.dropoff_node)
                + self.path_energy_upper(task.dropoff_node, station)
            )
        else:
            route = (
                self.path_energy_upper(station, task.dropoff_node)
                + self.path_energy_upper(task.dropoff_node, station)
            )
        return float(route + self.runtime.scenario.terminal.minimum_energy)
