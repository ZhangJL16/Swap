from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from cert_runtime.certificates import certificate_hash
from cert_runtime.recovery import RecoveryDecision
from cert_runtime.state import CertificateState

from .mission_certificate import (
    MissionActionContext,
    MissionClosureResult,
    MissionFailureWitness,
    MultiStepSyntheticMissionCertificateProvider,
)
from .recoverability import (
    RECOVERABILITY_ACTION_RULE_VERSION,
    RECOVERABLE_SET_VERSION,
    RecoverabilityVerifier,
)
from .state import UAVPhysicalState


@dataclass(frozen=True, slots=True)
class RecoverabilityAtlasManifest:
    scenario_id: str
    atlas_version: str
    recovery_manifest_hash: str
    recoverable_set_version: str
    recoverability_action_rule_version: str
    geometry_version: str
    dynamics_version: str
    tracking_version: str
    energy_version: str
    terminal_version: str
    kappa_version: str
    number_of_cells: int
    number_of_rl_authority_cells: int
    number_of_kappa_only_cells: int
    certified_coverage_fraction: float
    rl_authority_coverage_fraction: float
    rl_authority_fraction_within_recoverable_domain: float
    terminal_recovery_certificate_hash: str
    gate_pass: bool
    failure_reasons: tuple[str, ...]
    manifest_hash: str


@dataclass(frozen=True, slots=True)
class TerminalRecoveryCertificate:
    cell_id: str
    terminal: bool
    level: int
    successor: None
    recovery_energy_upper: float
    position_low: tuple[float, float, float]
    position_high: tuple[float, float, float]
    velocity_abs_max: tuple[float, float, float]
    minimum_terminal_energy: float
    terminal_geometry_hash: str
    geometry_version: str
    dynamics_version: str
    tracking_version: str
    energy_version: str
    terminal_version: str
    kappa_version: str
    atlas_hash: str
    certificate_hash: str

    @property
    def expected_hash(self) -> str:
        return certificate_hash({
            "cell_id": self.cell_id,
            "terminal": self.terminal,
            "level": self.level,
            "successor": self.successor,
            "recovery_energy_upper": self.recovery_energy_upper,
            "position_low": self.position_low,
            "position_high": self.position_high,
            "velocity_abs_max": self.velocity_abs_max,
            "minimum_terminal_energy": self.minimum_terminal_energy,
            "terminal_geometry_hash": self.terminal_geometry_hash,
            "versions": (
                self.geometry_version,
                self.dynamics_version,
                self.tracking_version,
                self.energy_version,
                self.terminal_version,
                self.kappa_version,
            ),
            "atlas_hash": self.atlas_hash,
        })

    @property
    def valid(self) -> bool:
        return bool(self.terminal and self.level == 0 and self.successor is None and self.certificate_hash == self.expected_hash)


class CertifiedRecoverabilityAtlas(MultiStepSyntheticMissionCertificateProvider):
    """Task-independent certified coverage of states with a frozen-kappa return.

    Coverage waypoints are offline proof-partition seeds, not task waypoints.  The
    manifest and the resulting safe action support never include a task goal,
    task seed, route index, or task-edge identity.
    """

    atlas_version = "task-independent-recoverability-atlas-v1"
    task_independent = True
    consumes_task_edges = False
    consumes_task_waypoints = False

    def __init__(self, runtime: Any) -> None:
        if "coverage_waypoints" not in runtime.scenario.mission_config:
            raise ValueError("random persistent scenarios require coverage_waypoints")
        if "task_waypoints" in runtime.scenario.mission_config.get("random_persistent", {}):
            raise ValueError("random persistent configuration cannot contain task waypoints")
        super().__init__(runtime, center_mode="safety_neutral")
        self.active_edge_id = "recovery_atlas"
        self.verifier = RecoverabilityVerifier(runtime, self)
        self.recoverability_verifiers = {self.active_edge_id: self.verifier}
        self.last_recoverable_set_certificate = None
        self.last_recoverability_action_certificate = None
        self.last_policy_authority_certificate = None
        self.charging_support_required = False
        self.last_charging_support_verified = False
        self.last_charging_support_hash: str | None = None
        self.last_continuation_verified = False
        self.last_continuation_target_cell_id: str | None = None
        versions = self._versions()
        self._rl_authority_cell_ids, self._rl_successor_ids = self._build_rl_authority_domain()
        failures = tuple(witness.reason for witness in self.manifest.failure_witnesses)
        atlas_core_payload = {
            "scenario": runtime.scenario.name,
            "atlas_version": self.atlas_version,
            "recovery_manifest": self.manifest.manifest_hash,
            "recoverable_set_version": RECOVERABLE_SET_VERSION,
            "recoverability_action_rule_version": RECOVERABILITY_ACTION_RULE_VERSION,
            "versions": versions,
            "calibration_hashes": tuple(runtime.calibration.fingerprints),
            "free_boxes": tuple(tuple(float(value) for value in box) for box in self.free_boxes),
            "occupied_boxes": tuple(tuple(float(value) for value in box) for box in self.occupied_boxes),
            "number_of_cells": len(self.manifest.cells),
        }
        atlas_core_hash = certificate_hash(atlas_core_payload)
        self.terminal_recovery_certificate = self._build_terminal_recovery_certificate(atlas_core_hash)
        payload = atlas_core_payload | {
            "rl_authority_cells": tuple(sorted(self._rl_authority_cell_ids)),
            "rl_successors": tuple(sorted(self._rl_successor_ids.items())),
            "terminal_recovery": self.terminal_recovery_certificate.certificate_hash,
        }
        atlas_hash = certificate_hash(payload)
        coverage = self._coverage_fraction()
        rl_coverage = self._coverage_fraction(self.rl_authority_cells)
        self.persistent_manifest = RecoverabilityAtlasManifest(
            scenario_id=runtime.scenario.name,
            atlas_version=self.atlas_version,
            recovery_manifest_hash=self.manifest.manifest_hash,
            recoverable_set_version=RECOVERABLE_SET_VERSION,
            recoverability_action_rule_version=RECOVERABILITY_ACTION_RULE_VERSION,
            geometry_version=versions[0],
            dynamics_version=versions[1],
            tracking_version=versions[2],
            energy_version=versions[3],
            terminal_version=versions[4],
            kappa_version=versions[5],
            number_of_cells=len(self.manifest.cells),
            number_of_rl_authority_cells=len(self._rl_authority_cell_ids),
            number_of_kappa_only_cells=len(self.manifest.cells) - len(self._rl_authority_cell_ids),
            certified_coverage_fraction=coverage,
            rl_authority_coverage_fraction=rl_coverage,
            rl_authority_fraction_within_recoverable_domain=0.0 if coverage <= 0.0 else rl_coverage / coverage,
            terminal_recovery_certificate_hash=self.terminal_recovery_certificate.certificate_hash,
            gate_pass=self.gate_pass,
            failure_reasons=failures,
            manifest_hash=atlas_hash,
        )

    @property
    def atlas_hash(self) -> str:
        return self.persistent_manifest.manifest_hash

    @property
    def gate_pass(self) -> bool:
        base_valid = super().gate_pass
        terminal = getattr(self, "terminal_recovery_certificate", None)
        authority_cells = getattr(self, "_rl_authority_cell_ids", None)
        return bool(
            base_valid
            and (terminal is None or terminal.valid)
            and (authority_cells is None or len(authority_cells) > 0)
        )

    @property
    def rl_authority_cells(self):
        return tuple(cell for cell in self.root_cells if cell.cell_id in self._rl_authority_cell_ids)

    def _coverage_fraction(self, cells=None, resolution: int = 48) -> float:
        selected_cells = self.root_cells if cells is None else tuple(cells)
        xs = np.linspace(0.0, self.runtime.config.world_size[0], resolution)
        ys = np.linspace(0.0, self.runtime.config.world_size[1], resolution)
        altitude = float(np.median([cell.reference_position[2] for cell in self.root_cells]))
        covered = 0
        for x_value in xs:
            for y_value in ys:
                point = np.array((x_value, y_value, altitude), dtype=np.float64)
                if any(cell.state_bounds.position.contains_point(point, 1e-12) for cell in selected_cells):
                    covered += 1
        return float(covered / (resolution * resolution))

    def _build_terminal_recovery_certificate(self, atlas_core_hash: str) -> TerminalRecoveryCertificate:
        terminal = self.runtime.scenario.terminal
        versions = self._versions()
        geometry_hash = certificate_hash({
            "position_low": tuple(float(value) for value in terminal.position_low),
            "position_high": tuple(float(value) for value in terminal.position_high),
            "velocity_abs_max": tuple(float(value) for value in terminal.velocity_abs_max),
            "geometry_version": versions[0],
            "terminal_version": versions[4],
        })
        values = {
            "cell_id": f"{self.runtime.scenario.name}-terminal-recovery-level-0",
            "terminal": True,
            "level": 0,
            "successor": None,
            "recovery_energy_upper": 0.0,
            "position_low": tuple(float(value) for value in terminal.position_low),
            "position_high": tuple(float(value) for value in terminal.position_high),
            "velocity_abs_max": tuple(float(value) for value in terminal.velocity_abs_max),
            "minimum_terminal_energy": float(terminal.minimum_energy),
            "terminal_geometry_hash": geometry_hash,
            "geometry_version": versions[0],
            "dynamics_version": versions[1],
            "tracking_version": versions[2],
            "energy_version": versions[3],
            "terminal_version": versions[4],
            "kappa_version": versions[5],
            "atlas_hash": atlas_core_hash,
        }
        provisional = TerminalRecoveryCertificate(**values, certificate_hash="")
        return replace(provisional, certificate_hash=provisional.expected_hash)

    def _terminal_certificate_valid_for_state(self, state: CertificateState) -> bool:
        certificate = self.terminal_recovery_certificate
        versions = self._versions()
        position = np.asarray(state.position)
        velocity = np.asarray(state.velocity)
        position_error = np.asarray(state.position_error_radius)
        velocity_error = np.asarray(state.velocity_error_radius)
        return bool(
            certificate.valid
            and certificate.terminal_geometry_hash == certificate_hash({
                "position_low": certificate.position_low,
                "position_high": certificate.position_high,
                "velocity_abs_max": certificate.velocity_abs_max,
                "geometry_version": certificate.geometry_version,
                "terminal_version": certificate.terminal_version,
            })
            and np.all(position - position_error >= np.asarray(certificate.position_low) - 1e-12)
            and np.all(position + position_error <= np.asarray(certificate.position_high) + 1e-12)
            and np.all(np.abs(velocity) + velocity_error <= np.asarray(certificate.velocity_abs_max) + 1e-12)
            and state.energy - state.energy_error_radius >= certificate.minimum_terminal_energy + self.energy_reserve - 1e-12
            and (
                certificate.geometry_version,
                certificate.dynamics_version,
                certificate.tracking_version,
                certificate.energy_version,
                certificate.terminal_version,
                certificate.kappa_version,
            ) == versions
            and state.bound_versions.get("dynamics") == self.runtime.calibration.dynamics.version
            and state.bound_versions.get("tracking") == self.runtime.calibration.tracking.version
            and state.bound_versions.get("energy") == self.runtime.calibration.energy.version
            and state.bound_versions.get("terminal") == self.runtime.calibration.terminal.version
        )

    def _build_rl_authority_domain(self) -> tuple[frozenset[str], dict[str, str]]:
        roots = self.root_cells
        if not roots:
            return frozenset(), {}
        task_radii = self._task_tube_radii()
        candidates = {
            cell.cell_id
            for cell in roots
            if cell.hash_valid
            and cell.complete_successor_containment
            and cell.minimum_geometry_slack >= -1e-12
            and np.all(self.base_scales >= self.runtime.config.minimum_generator_sigma - 1e-12)
        }
        transitions: dict[str, tuple[str, ...]] = {}
        for index, root in enumerate(roots):
            options: list[str] = []
            preferred = tuple(dict.fromkeys((min(index + 1, len(roots) - 1), index, max(index - 1, 0))))
            for target_index in preferred:
                if self._task_transition_valid(root, self.coverage_reference[target_index], task_radii, self.base_scales):
                    options.append(roots[target_index].cell_id)
            transitions[root.cell_id] = tuple(options)
        viable = set(candidates)
        while True:
            reduced = {
                cell_id
                for cell_id in viable
                if any(target_id in viable for target_id in transitions.get(cell_id, ()))
            }
            if reduced == viable:
                break
            viable = reduced
        successors = {
            cell_id: next(target_id for target_id in transitions[cell_id] if target_id in viable)
            for cell_id in viable
        }
        return frozenset(viable), successors

    def _center(self, state: CertificateState, root) -> np.ndarray:
        if self.center_mode != "safety_neutral":
            return super()._center(state, root)
        root_index = self._chain_by_root[root.cell_id].root_index
        reference = self.coverage_reference[root_index]
        position_error = np.asarray(state.position) - reference.position
        if self._rl_successor_ids.get(root.cell_id) == root.cell_id:
            raw = -self.position_gain * position_error - self.velocity_gain * np.asarray(state.velocity)
        else:
            velocity_error = np.asarray(state.velocity) - reference.velocity
            raw = reference.action - self.position_gain * position_error - self.velocity_gain * velocity_error
        room = np.maximum(self.runtime.config.a_max - self.runtime.config.minimum_generator_sigma, 0.0)
        return np.clip(raw, -room, room)

    def _recoverable_successor_candidates(self, cell):
        successor_id = self._rl_successor_ids.get(cell.cell_id)
        if successor_id is None:
            return ()
        successor = self._cells_by_id.get(successor_id)
        return () if successor is None else (successor,)

    def _locate_recovery_cell(self, state: CertificateState):
        if self.active_cell_id is None:
            return self._locate_recoverable_cell(state)
        return super()._locate_recovery_cell(state)

    def _locate_recoverable_cell(self, state: CertificateState):
        viable_roots = [
            cell
            for cell in self.root_cells
            if cell.cell_id in getattr(self, "_rl_authority_cell_ids", ())
            and self._cell_contains_state(cell, state)
        ]
        if viable_roots:
            return min(
                viable_roots,
                key=lambda cell: sum(
                    np.array((state.position[axis] - cell.reference_position[axis], state.velocity[axis] - cell.reference_velocity[axis]))
                    @ self._matrix
                    @ np.array((state.position[axis] - cell.reference_position[axis], state.velocity[axis] - cell.reference_velocity[axis]))
                    / (cell.ellipsoid_radii[axis] ** 2)
                    for axis in range(3)
                ),
            )
        return super()._locate_recoverable_cell(state)

    def reset(self) -> None:
        super().reset()
        self.last_recoverable_set_certificate = None
        self.last_recoverability_action_certificate = None
        self.last_policy_authority_certificate = None
        self.charging_support_required = False
        self.last_charging_support_verified = False
        self.last_charging_support_hash = None
        self.last_continuation_verified = False
        self.last_continuation_target_cell_id = None

    def configure_charging_support(self, required: bool) -> None:
        self.charging_support_required = bool(required)

    def evaluate(self, state, timestamp: float | None = None) -> MissionActionContext:
        terminal_complete = self._terminal_certificate_valid_for_state(state)
        if terminal_complete:
            self.recovery_active = False
            self.active_cell_id = None
        context = super().evaluate(state, timestamp)
        if terminal_complete:
            required = float(self.runtime.scenario.terminal.minimum_energy + self.energy_reserve)
            context = replace(
                context,
                recovery=RecoveryDecision(
                    (0.0, 0.0, 0.0),
                    True,
                    self.terminal_recovery_certificate.certificate_hash,
                    "terminal-recovery-complete-zero-step",
                ),
                required_energy=required,
                current_energy_margin=float(state.energy - state.energy_error_radius - required),
                recovery_cell_id=self.terminal_recovery_certificate.cell_id,
                recovery_level=0,
            )
        self.last_charging_support_verified = False
        self.last_charging_support_hash = None
        self.last_continuation_verified = False
        self.last_continuation_target_cell_id = None
        membership = self.verifier.membership(state, context)
        self.last_recoverable_set_certificate = membership
        certificate = context.closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        if zonotope is None:
            self.last_recoverability_action_certificate = None
            self.last_context = context
            return context
        action_certificate = self.verifier.certify_action_set(state, zonotope, context)
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
            context = replace(context, closure=closure)
        elif self.charging_support_required:
            restricted, restricted_certificate = self.verifier.restrict_to_charging_set(state, zonotope, context)
            if restricted is None or restricted_certificate is None:
                context = replace(
                    context,
                    closure=MissionClosureResult(
                        False,
                        None,
                        "NO_CHARGING_GENERATOR_SET",
                        MissionFailureWitness("NO_CHARGING_GENERATOR_SET", context.recovery_cell_id),
                        context.closure.manifest,
                    ),
                )
                self.last_recoverability_action_certificate = None
            else:
                original = context.closure.zonotope_certificate
                support_hash = certificate_hash({
                    "base": original.complete_set_inclusion_hash,
                    "restricted": repr(restricted),
                    "a_rec": restricted_certificate.certificate_hash,
                    "charging": True,
                })
                context = replace(
                    context,
                    closure=replace(
                        context.closure,
                        status="VERIFIED_CHARGING_SUPPORT",
                        zonotope_certificate=replace(
                            original,
                            reason="VERIFIED_CHARGING_SUPPORT",
                            zonotope=restricted,
                            successor_envelope=self.runtime.envelope_builder.propagate_zonotope(state, restricted),
                            complete_set_inclusion_hash=support_hash,
                        ),
                    ),
                )
                self.last_recoverability_action_certificate = restricted_certificate
                self.last_charging_support_verified = True
                self.last_charging_support_hash = support_hash
        target_id = context.task_successor_cell_id
        self.last_continuation_verified = bool(
            self.last_recoverability_action_certificate is not None
            and self.last_recoverability_action_certificate.verified
            and (
                target_id in self._rl_authority_cell_ids
                or (
                    terminal_complete
                    and self.last_charging_support_verified
                )
            )
        )
        self.last_continuation_target_cell_id = target_id if self.last_continuation_verified else None
        self.last_context = context
        return context

    def policy_authority_gate(self, state, goal_position: np.ndarray, station_position: np.ndarray):
        context = self.evaluate(state)
        result = self.verifier.policy_authority(state, context, goal_position, station_position)
        self.last_policy_authority_certificate = result
        return result

    def certified_station_hold(self, state) -> bool:
        return self.verifier.certified_station_hold(state)

    def required_departure_energy(self, task=None) -> float:
        del task
        return float(self.runtime.scenario.terminal.minimum_energy + self.energy_reserve)

    def authority_domain_report(self) -> dict[str, Any]:
        manifest = self.persistent_manifest
        return {
            "number_of_recoverable_cells": manifest.number_of_cells,
            "number_of_rl_authority_cells": manifest.number_of_rl_authority_cells,
            "number_of_kappa_only_cells": manifest.number_of_kappa_only_cells,
            "recoverable_coverage_fraction": manifest.certified_coverage_fraction,
            "rl_authority_coverage_fraction": manifest.rl_authority_coverage_fraction,
            "rl_authority_fraction_within_recoverable_domain": manifest.rl_authority_fraction_within_recoverable_domain,
            "terminal_recovery_certificate_hash": manifest.terminal_recovery_certificate_hash,
            "terminal_recovery_energy_upper": self.terminal_recovery_certificate.recovery_energy_upper,
        }

    def contains_certificate_state(self, state) -> bool:
        return self._terminal_certificate_valid_for_state(state) or self._locate_recoverable_cell(state) is not None

    def contains_rl_authority_state(self, state) -> bool:
        cell = self._locate_recoverable_cell(state)
        return bool(cell is not None and cell.cell_id in self._rl_authority_cell_ids)

    def sample_initial_state(self, seed: int | None, battery_capacity: float) -> UAVPhysicalState:
        rng = np.random.default_rng(seed)
        candidates = [
            cell
            for cell in self.root_cells
            if cell.cell_id in self._rl_authority_cell_ids
            and cell.hash_valid
            and cell.complete_successor_containment
            and cell.minimum_geometry_slack >= -1e-12
            and cell.energy_upper + self.runtime.scenario.terminal.minimum_energy + self.energy_reserve < battery_capacity
        ]
        if not candidates:
            raise RuntimeError("RECOVERY_ATLAS_HAS_NO_INITIAL_STATE_CELL")
        cell = candidates[int(rng.integers(0, len(candidates)))]
        position_radius, velocity_radius = self._coordinate_radii(np.asarray(cell.ellipsoid_radii))
        position = np.asarray(cell.reference_position) + rng.uniform(-0.12, 0.12, 3) * position_radius
        velocity = np.asarray(cell.reference_velocity) + rng.uniform(-0.05, 0.05, 3) * velocity_radius
        random_config = dict(self.profile.get("random_persistent", {}))
        requested_low, requested_high = random_config.get("initial_energy_range", (0.75 * battery_capacity, battery_capacity))
        required = cell.energy_upper + self.runtime.scenario.terminal.minimum_energy + self.energy_reserve + 0.25
        low = max(float(requested_low), float(required))
        high = min(float(requested_high), float(battery_capacity))
        if low > high:
            raise RuntimeError("RANDOM_INITIAL_ENERGY_RANGE_OUTSIDE_RECOVERABLE_DOMAIN")
        return UAVPhysicalState(position, velocity, float(rng.uniform(low, high)), 0.0)

    def sample_goal(
        self,
        rng: np.random.Generator,
        current_position: np.ndarray,
        minimum_separation: float,
    ) -> np.ndarray:
        candidates = [
            cell
            for cell in self.root_cells
            if cell.hash_valid and cell.complete_successor_containment and cell.minimum_geometry_slack >= -1e-12
        ]
        if not candidates:
            raise RuntimeError("RECOVERY_ATLAS_HAS_NO_GOAL_CELL")
        terminal = self.runtime.scenario.terminal
        current = np.asarray(current_position, dtype=np.float64)
        for _ in range(512):
            cell = candidates[int(rng.integers(0, len(candidates)))]
            position_radius, _ = self._coordinate_radii(np.asarray(cell.ellipsoid_radii))
            point = np.asarray(cell.reference_position) + rng.uniform(-0.18, 0.18, 3) * position_radius
            point[2] = float(cell.reference_position[2])
            in_terminal = bool(np.all(point >= terminal.position_low) and np.all(point <= terminal.position_high))
            if not in_terminal and np.linalg.norm(point - current) >= minimum_separation:
                return point
        raise RuntimeError("RANDOM_GOAL_SAMPLER_COULD_NOT_SATISFY_SEPARATION")
