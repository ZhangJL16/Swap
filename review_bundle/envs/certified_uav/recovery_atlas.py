from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from cert_runtime.certificates import certificate_hash

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
    certified_coverage_fraction: float
    gate_pass: bool
    failure_reasons: tuple[str, ...]
    manifest_hash: str


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
        versions = self._versions()
        failures = tuple(witness.reason for witness in self.manifest.failure_witnesses)
        payload = {
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
        atlas_hash = certificate_hash(payload)
        coverage = self._coverage_fraction()
        self.persistent_manifest = RecoverabilityAtlasManifest(
            runtime.scenario.name,
            self.atlas_version,
            self.manifest.manifest_hash,
            RECOVERABLE_SET_VERSION,
            RECOVERABILITY_ACTION_RULE_VERSION,
            versions[0],
            versions[1],
            versions[2],
            versions[3],
            versions[4],
            versions[5],
            len(self.manifest.cells),
            coverage,
            self.gate_pass,
            failures,
            atlas_hash,
        )

    @property
    def atlas_hash(self) -> str:
        return self.persistent_manifest.manifest_hash

    def _coverage_fraction(self, resolution: int = 48) -> float:
        xs = np.linspace(0.0, self.runtime.config.world_size[0], resolution)
        ys = np.linspace(0.0, self.runtime.config.world_size[1], resolution)
        altitude = float(np.median([cell.reference_position[2] for cell in self.root_cells]))
        covered = 0
        for x_value in xs:
            for y_value in ys:
                point = np.array((x_value, y_value, altitude), dtype=np.float64)
                if any(cell.state_bounds.position.contains_point(point, 1e-12) for cell in self.root_cells):
                    covered += 1
        return float(covered / (resolution * resolution))

    def reset(self) -> None:
        super().reset()
        self.last_recoverable_set_certificate = None
        self.last_recoverability_action_certificate = None
        self.last_policy_authority_certificate = None
        self.charging_support_required = False
        self.last_charging_support_verified = False
        self.last_charging_support_hash = None

    def configure_charging_support(self, required: bool) -> None:
        self.charging_support_required = bool(required)

    def evaluate(self, state, timestamp: float | None = None) -> MissionActionContext:
        context = super().evaluate(state, timestamp)
        self.last_charging_support_verified = False
        self.last_charging_support_hash = None
        membership = self.verifier.membership(state, context)
        self.last_recoverable_set_certificate = membership
        certificate = context.closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        if zonotope is None:
            self.last_recoverability_action_certificate = None
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

    def contains_certificate_state(self, state) -> bool:
        return self._locate_recoverable_cell(state) is not None

    def sample_initial_state(self, seed: int | None, battery_capacity: float) -> UAVPhysicalState:
        rng = np.random.default_rng(seed)
        candidates = [
            cell
            for cell in self.root_cells
            if cell.hash_valid
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
