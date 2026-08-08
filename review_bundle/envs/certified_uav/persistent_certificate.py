from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any

import numpy as np

from cert_runtime.certificates import certificate_hash
from cert_runtime.interval import round_up
from cert_runtime.types import Interval3

from .mission_certificate import (
    MissionActionContext,
    MissionClosureResult,
    MissionFailureWitness,
    MultiStepSyntheticMissionCertificateProvider,
)
from .persistent_task import (
    CertifiedGoalNetwork,
    GoalEdge,
    GoalEdgeType,
    PersistentGoalTask,
)
from .recoverability import (
    RECOVERABILITY_ACTION_RULE_VERSION,
    RECOVERABLE_SET_VERSION,
    PolicyAuthorityCertificate,
    RecoverabilityActionCertificate,
    RecoverabilityVerifier,
    RecoverableSetCertificate,
)


@dataclass(frozen=True, slots=True)
class PersistentGoalEdgeCertificate:
    edge_id: str
    source: str
    target: str
    edge_type: str
    recovery_manifest_hash: str
    recovery_gate_pass: bool
    complete_successor_support: bool
    energy_upper: float
    dependency_hashes: tuple[str, ...]
    certificate_hash: str


@dataclass(frozen=True, slots=True)
class PersistentGoalCertificateManifest:
    scenario_id: str
    goal_network_hash: str
    edge_certificates: tuple[PersistentGoalEdgeCertificate, ...]
    task_routes_valid: bool
    recovery_routes_valid: bool
    departure_routes_valid: bool
    docking_valid: bool
    energy_recursion_valid: bool
    departure_gate_valid: bool
    interruption_resume_valid: bool
    charging_separated_from_return_energy: bool
    recoverable_set_valid: bool
    recoverability_action_rule_valid: bool
    complete_generator_recoverability_required: bool
    version_consistent: bool
    recoverable_set_version: str
    recoverability_action_rule_version: str
    energy_field_version: str
    kappa_version: str
    geometry_version: str
    tracking_version: str
    dynamics_version: str
    gate_pass: bool
    failure_reasons: tuple[str, ...]
    manifest_hash: str


class PersistentGoalCertificateProvider:
    """Persistent gate composed from typed edge certificates and the frozen κ chain."""

    version = "persistent-goal-certificate-v2"

    def __init__(self, runtime: Any, network: CertifiedGoalNetwork, battery_capacity: float) -> None:
        self.runtime = runtime
        self.network = network
        self.battery_capacity = float(battery_capacity)
        self.providers: dict[str, MultiStepSyntheticMissionCertificateProvider] = {}
        self.recoverability_verifiers: dict[str, RecoverabilityVerifier] = {}
        self.edge_energy_upper: dict[str, float] = {}
        self.active_edge_id = sorted(network.edges)[0]
        failures: list[str] = []
        certificates: list[PersistentGoalEdgeCertificate] = []
        for edge_id, edge in sorted(network.edges.items()):
            provider = self._build_edge_provider(edge)
            self.providers[edge_id] = provider
            self.recoverability_verifiers[edge_id] = RecoverabilityVerifier(runtime, provider)
            energy_upper = self._edge_energy_bound(provider)
            self.edge_energy_upper[edge_id] = energy_upper
            complete = bool(
                provider.manifest.task_transition_verified
                and all(provider.manifest.task_transition_verified)
            )
            payload = {
                "edge": edge_id,
                "source": edge.source,
                "target": edge.target,
                "type": edge.edge_type.value,
                "manifest": provider.manifest.manifest_hash,
                "gate": provider.gate_pass,
                "support": complete,
                "energy": energy_upper,
                "network": network.network_hash,
            }
            edge_hash = certificate_hash(payload)
            certificates.append(PersistentGoalEdgeCertificate(
                edge_id,
                edge.source,
                edge.target,
                edge.edge_type.value,
                provider.manifest.manifest_hash,
                provider.gate_pass,
                complete,
                energy_upper,
                (provider.manifest.manifest_hash, network.network_hash),
                edge_hash,
            ))
            if not provider.gate_pass:
                failures.append(f"EDGE_RECOVERY_GATE_FAILED:{edge_id}")
            if not complete:
                failures.append(f"EDGE_SUCCESSOR_SUPPORT_INCOMPLETE:{edge_id}")

        task_routes_valid = self._all_task_routes_valid()
        recovery_routes_valid = self._all_recovery_routes_valid()
        departure_routes_valid = self._all_departure_routes_valid()
        docking_valid = bool(runtime.scenario.terminal.is_charge_admissible(replace(
            runtime.scenario.initial_state,
            position=runtime.scenario.station_position.copy(),
            velocity=np.zeros(3),
            energy=max(runtime.scenario.terminal.minimum_energy, 1.0),
        )))
        energy_valid = all(
            cell.e3_residual >= -1e-12 and cell.energy_upper >= 0.0
            for provider in self.providers.values()
            for cell in provider.manifest.cells
        )
        departure_valid = self._all_departures_fit_capacity()
        interruption_resume_valid = recovery_routes_valid and departure_routes_valid
        recoverable_set_valid = bool(
            energy_valid
            and all(
                cell.hash_valid
                and cell.complete_successor_containment
                and cell.energy_upper >= 0.0
                and cell.state_bounds.energy.low
                >= cell.energy_upper + runtime.scenario.terminal.minimum_energy + provider.energy_reserve - 1e-12
                for provider in self.providers.values()
                for cell in provider.manifest.cells
            )
        )
        recoverability_action_rule_valid = bool(task_routes_valid and recovery_routes_valid and recoverable_set_valid)
        complete_generator_recoverability_required = True
        version_values = tuple(self.providers.values())[0]._versions()
        version_consistent = len({provider._versions() for provider in self.providers.values()}) == 1
        checks = {
            "TASK_ROUTE_GRAPH_INVALID": task_routes_valid,
            "RECOVERY_ROUTE_GRAPH_INVALID": recovery_routes_valid,
            "DEPARTURE_ROUTE_GRAPH_INVALID": departure_routes_valid,
            "DOCKING_INVALID": docking_valid,
            "ENERGY_RECURSION_INVALID": energy_valid,
            "DEPARTURE_GATE_INVALID": departure_valid,
            "INTERRUPTION_RESUME_INVALID": interruption_resume_valid,
            "RECOVERABLE_SET_INVALID": recoverable_set_valid,
            "RECOVERABILITY_ACTION_RULE_INVALID": recoverability_action_rule_valid,
            "VERSION_MISMATCH": version_consistent,
        }
        failures.extend(name for name, valid in checks.items() if not valid)
        gate_pass = not failures
        manifest_payload = {
            "scenario": runtime.scenario.name,
            "network": network.network_hash,
            "edges": tuple(certificate.certificate_hash for certificate in certificates),
            "task_routes": task_routes_valid,
            "recovery_routes": recovery_routes_valid,
            "departure_routes": departure_routes_valid,
            "docking": docking_valid,
            "energy": energy_valid,
            "departure_gate": departure_valid,
            "interruption_resume": interruption_resume_valid,
            "charging_separated": True,
            "recoverable_set": recoverable_set_valid,
            "recoverability_action_rule": recoverability_action_rule_valid,
            "complete_generator_recoverability_required": complete_generator_recoverability_required,
            "recoverable_set_version": RECOVERABLE_SET_VERSION,
            "recoverability_action_rule_version": RECOVERABILITY_ACTION_RULE_VERSION,
            "bound_versions": version_values,
            "versions": version_consistent,
            "failures": tuple(failures),
        }
        self.persistent_manifest = PersistentGoalCertificateManifest(
            scenario_id=runtime.scenario.name,
            goal_network_hash=network.network_hash,
            edge_certificates=tuple(certificates),
            task_routes_valid=task_routes_valid,
            recovery_routes_valid=recovery_routes_valid,
            departure_routes_valid=departure_routes_valid,
            docking_valid=docking_valid,
            energy_recursion_valid=energy_valid,
            departure_gate_valid=departure_valid,
            interruption_resume_valid=interruption_resume_valid,
            charging_separated_from_return_energy=True,
            recoverable_set_valid=recoverable_set_valid,
            recoverability_action_rule_valid=recoverability_action_rule_valid,
            complete_generator_recoverability_required=complete_generator_recoverability_required,
            version_consistent=version_consistent,
            recoverable_set_version=RECOVERABLE_SET_VERSION,
            recoverability_action_rule_version=RECOVERABILITY_ACTION_RULE_VERSION,
            energy_field_version=version_values[3],
            kappa_version=version_values[5],
            geometry_version=version_values[0],
            tracking_version=version_values[2],
            dynamics_version=version_values[1],
            gate_pass=gate_pass,
            failure_reasons=tuple(failures),
            manifest_hash=certificate_hash(manifest_payload),
        )
        self.last_context: MissionActionContext | None = None
        self.last_recoverable_set_certificate: RecoverableSetCertificate | None = None
        self.last_recoverability_action_certificate: RecoverabilityActionCertificate | None = None
        self.last_policy_authority_certificate: PolicyAuthorityCertificate | None = None
        self.charging_support_required = False
        self.last_charging_support_verified = False
        self.last_charging_support_hash: str | None = None

    def _edge_energy_bound(self, provider: MultiStepSyntheticMissionCertificateProvider) -> float:
        action = Interval3(-self.runtime.config.a_max, self.runtime.config.a_max)
        velocity = Interval3(-self.runtime.config.v_max, self.runtime.config.v_max)
        task_step_upper = self.runtime.envelope_builder.energy.cost_upper(action, velocity)
        transition_count = max(0, len(provider.task_reference) - 1)
        task_upper = 0.0
        for _ in range(transition_count):
            task_upper = round_up(task_upper + task_step_upper)
        recovery_upper = max((cell.energy_upper for cell in provider.root_cells), default=float("inf"))
        return round_up(task_upper + recovery_upper)

    def _typed_path_energy(self, source: str, target: str, edge_type: GoalEdgeType) -> float:
        path = self.network.shortest_path(source, target, {edge_type})
        return float(sum(self.edge_energy_upper[edge.edge_id] for edge in path))

    def path_energy_upper(
        self,
        source: str,
        target: str,
        edge_type: GoalEdgeType | None = None,
    ) -> float:
        if source == target:
            return 0.0
        allowed = None if edge_type is None else {edge_type}
        path = self.network.shortest_path(source, target, allowed)
        return float(sum(self.edge_energy_upper[edge.edge_id] for edge in path))

    def _all_task_routes_valid(self) -> bool:
        try:
            for source in self.network.goal_node_ids:
                for target in self.network.goal_node_ids:
                    if source != target:
                        self.network.shortest_path(source, target, {GoalEdgeType.TASK_EDGE})
            return True
        except ValueError:
            return False

    def _all_recovery_routes_valid(self) -> bool:
        try:
            for goal in self.network.goal_node_ids:
                self.network.shortest_path(goal, self.network.charging_station, {GoalEdgeType.RECOVERY_EDGE})
            return True
        except ValueError:
            return False

    def _all_departure_routes_valid(self) -> bool:
        try:
            for goal in self.network.goal_node_ids:
                self.network.shortest_path(self.network.charging_station, goal, {GoalEdgeType.DEPARTURE_EDGE})
            return True
        except ValueError:
            return False

    def _all_departures_fit_capacity(self) -> bool:
        station = self.network.charging_station
        terminal_energy = self.runtime.scenario.terminal.minimum_energy
        try:
            for goal in self.network.goal_node_ids:
                required = (
                    self._typed_path_energy(station, goal, GoalEdgeType.DEPARTURE_EDGE)
                    + self._typed_path_energy(goal, station, GoalEdgeType.RECOVERY_EDGE)
                    + terminal_energy
                )
                if not np.isfinite(required) or required > self.battery_capacity:
                    return False
            return True
        except ValueError:
            return False

    def _recovery_waypoints(self, source: str) -> list[list[float]]:
        station = self.network.charging_station
        path = self.network.shortest_path(source, station, {GoalEdgeType.RECOVERY_EDGE})
        points: list[np.ndarray] = []
        for edge in path:
            segment = list(edge.waypoints)
            if points and np.allclose(points[-1], segment[0]):
                segment = segment[1:]
            points.extend(segment)
        return [point.tolist() for point in points]

    def _build_edge_provider(self, edge: GoalEdge) -> MultiStepSyntheticMissionCertificateProvider:
        profile = dict(self.runtime.scenario.mission_config)
        profile["task_waypoints"] = [point.tolist() for point in edge.waypoints]
        if edge.edge_type == GoalEdgeType.RECOVERY_EDGE:
            recovery = [point.tolist() for point in edge.waypoints]
        else:
            recovery = self._recovery_waypoints(edge.target)
        profile["return_waypoints"] = recovery
        profile["persistent_edge_id"] = edge.edge_id
        profile["persistent_edge_type"] = edge.edge_type.value
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
        self.last_recoverable_set_certificate = None
        self.last_recoverability_action_certificate = None
        self.last_policy_authority_certificate = None
        self.charging_support_required = False
        self.last_charging_support_verified = False
        self.last_charging_support_hash = None

    def configure_charging_support(self, required: bool) -> None:
        self.charging_support_required = bool(required)

    def evaluate(self, state, timestamp: float | None = None) -> MissionActionContext:
        provider = self.providers[self.active_edge_id]
        verifier = self.recoverability_verifiers[self.active_edge_id]
        context = provider.evaluate(state, timestamp)
        self.last_charging_support_verified = False
        self.last_charging_support_hash = None
        membership = verifier.membership(state, context)
        self.last_recoverable_set_certificate = membership
        certificate = context.closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        if zonotope is not None:
            action_certificate = verifier.certify_action_set(state, zonotope, context)
            self.last_recoverability_action_certificate = action_certificate
            if not action_certificate.verified:
                closure = MissionClosureResult(
                    False,
                    None,
                    action_certificate.reason,
                    MissionFailureWitness(
                        action_certificate.reason,
                        action_certificate.target_recovery_cell_id,
                        action_certificate.successor_required_energy,
                        action_certificate.successor_energy_lower,
                    ),
                    context.closure.manifest,
                )
                context = MissionActionContext(
                    context.recovery,
                    closure,
                    context.required_energy,
                    context.current_energy_margin,
                    context.recovery_cell_id,
                    context.successor_cell_id,
                    context.recovery_level,
                    context.root_index,
                    context.task_successor_cell_id,
                )
            elif self.charging_support_required:
                restricted, restricted_certificate = verifier.restrict_to_charging_set(
                    state,
                    zonotope,
                    context,
                )
                if restricted is None or restricted_certificate is None:
                    self.last_charging_support_verified = False
                    self.last_charging_support_hash = None
                    closure = MissionClosureResult(
                        False,
                        None,
                        "NO_CHARGING_GENERATOR_SET",
                        MissionFailureWitness("NO_CHARGING_GENERATOR_SET", context.recovery_cell_id),
                        context.closure.manifest,
                    )
                    context = replace(context, closure=closure)
                    self.last_recoverability_action_certificate = None
                else:
                    original = context.closure.zonotope_certificate
                    support_hash = certificate_hash({
                        "base": original.complete_set_inclusion_hash,
                        "restricted": repr(restricted),
                        "a_rec": restricted_certificate.certificate_hash,
                        "charging": True,
                    })
                    restricted_zonotope_certificate = replace(
                        original,
                        reason="VERIFIED_CHARGING_SUPPORT",
                        zonotope=restricted,
                        successor_envelope=self.runtime.envelope_builder.propagate_zonotope(state, restricted),
                        complete_set_inclusion_hash=support_hash,
                    )
                    context = replace(
                        context,
                        closure=replace(
                            context.closure,
                            zonotope_certificate=restricted_zonotope_certificate,
                            status="VERIFIED_CHARGING_SUPPORT",
                        ),
                    )
                    self.last_recoverability_action_certificate = restricted_certificate
                    self.last_charging_support_verified = True
                    self.last_charging_support_hash = support_hash
            else:
                self.last_charging_support_verified = False
                self.last_charging_support_hash = None
        else:
            self.last_recoverability_action_certificate = None
            self.last_charging_support_verified = False
            self.last_charging_support_hash = None
        self.last_context = context
        return context

    def commit_execution(self, context: MissionActionContext, task_action_executed: bool) -> None:
        self.providers[self.active_edge_id].commit_execution(context, task_action_executed)
        self.last_context = self.providers[self.active_edge_id].last_context

    def verify_task_action(self, state, action: np.ndarray) -> bool:
        context = self.evaluate(state)
        result = self.recoverability_verifiers[self.active_edge_id].certify_point_action(
            state,
            np.asarray(action, dtype=np.float64),
            context,
        )
        self.last_recoverability_action_certificate = result
        return result.verified

    def recoverable_set_membership(self, state, timestamp: float | None = None) -> RecoverableSetCertificate:
        context = self.evaluate(state, timestamp)
        result = self.recoverability_verifiers[self.active_edge_id].membership(state, context)
        self.last_recoverable_set_certificate = result
        return result

    def certify_recoverability_action_set(
        self,
        state,
        zonotope,
        context: MissionActionContext | None = None,
    ) -> RecoverabilityActionCertificate:
        selected = self.evaluate(state) if context is None else context
        result = self.recoverability_verifiers[self.active_edge_id].certify_action_set(state, zonotope, selected)
        self.last_recoverability_action_certificate = result
        return result

    def policy_authority_gate(
        self,
        state,
        goal_position: np.ndarray,
        station_position: np.ndarray,
    ) -> PolicyAuthorityCertificate:
        context = self.evaluate(state)
        result = self.recoverability_verifiers[self.active_edge_id].policy_authority(
            state,
            context,
            goal_position,
            station_position,
        )
        self.last_policy_authority_certificate = result
        return result

    def successor_stays_in_charging_set(self, state, action: np.ndarray) -> bool:
        return self.recoverability_verifiers[self.active_edge_id].successor_stays_in_charging_set(state, action)

    def certified_station_hold(self, state) -> bool:
        return self.recoverability_verifiers[self.active_edge_id].certified_station_hold(state)

    def required_departure_energy(self, task: PersistentGoalTask | None) -> float:
        if task is None:
            return 0.0
        station = self.network.charging_station
        route = (
            self._typed_path_energy(station, task.goal_node, GoalEdgeType.DEPARTURE_EDGE)
            + self._typed_path_energy(task.goal_node, station, GoalEdgeType.RECOVERY_EDGE)
        )
        return float(route + self.runtime.scenario.terminal.minimum_energy)


# Compatibility names; the manifest now represents a goal network, not logistics.
PersistentServiceEdgeCertificate = PersistentGoalEdgeCertificate
PersistentCertificateManifest = PersistentGoalCertificateManifest
PersistentCertificateProvider = PersistentGoalCertificateProvider
