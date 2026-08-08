from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING

import numpy as np

from cert_runtime.certificates import certificate_hash
from cert_runtime.state import CertificateState
from cert_runtime.types import Zonotope3

if TYPE_CHECKING:
    from .mission_certificate import MissionActionContext, MultiStepSyntheticMissionCertificateProvider
    from .runtime_wrapper import CertifiedRuntimeWrapper


RECOVERABLE_SET_VERSION = "recoverable-set-v1"
RECOVERABILITY_ACTION_RULE_VERSION = "recoverability-action-rule-v1"


@dataclass(frozen=True, slots=True)
class RecoverableSetCertificate:
    recoverable: bool
    required_recovery_energy: float
    energy_margin: float
    recovery_cell_id: str | None
    recovery_certificate_hash: str | None
    bound_versions: tuple[tuple[str, str], ...]
    recoverable_set_version: str
    certificate_hash: str
    reason: str


@dataclass(frozen=True, slots=True)
class RecoverabilityActionCertificate:
    verified: bool
    reason: str
    current_membership_hash: str
    target_recovery_cell_id: str | None
    successor_energy_lower: float
    successor_required_energy: float
    actuator_inclusion: bool
    collision_successor_inclusion: bool
    velocity_successor_inclusion: bool
    energy_successor_inclusion: bool
    version_consistent: bool
    recoverability_action_rule_version: str
    certificate_hash: str


@dataclass(frozen=True, slots=True)
class PolicyAuthorityCertificate:
    passed: bool
    output_dimension: int
    neutral_center: bool
    full_rank: bool
    nondegenerate: bool
    goal_direction_available: bool
    station_direction_available: bool
    complete_set_recoverable: bool
    sigma_min: float
    condition_number: float
    zonotope_volume: float
    reason: str


class RecoverabilityVerifier:
    """Theorem-facing verifier for R and A_rec using complete successor envelopes."""

    def __init__(self, runtime: "CertifiedRuntimeWrapper", provider: "MultiStepSyntheticMissionCertificateProvider") -> None:
        self.runtime = runtime
        self.provider = provider

    def membership(self, state: CertificateState, context: "MissionActionContext") -> RecoverableSetCertificate:
        margin = float(context.current_energy_margin)
        recoverable = bool(
            context.recovery.certified
            and context.recovery_cell_id is not None
            and isfinite(context.required_energy)
            and isfinite(margin)
            and margin >= -1e-12
        )
        payload = {
            "state": state.snapshot(),
            "required": context.required_energy,
            "margin": margin,
            "cell": context.recovery_cell_id,
            "recovery": context.recovery.certificate_hash,
            "bounds": tuple(sorted(state.bound_versions.items())),
            "version": RECOVERABLE_SET_VERSION,
        }
        return RecoverableSetCertificate(
            recoverable,
            float(context.required_energy),
            margin,
            context.recovery_cell_id,
            context.recovery.certificate_hash,
            tuple(sorted(state.bound_versions.items())),
            RECOVERABLE_SET_VERSION,
            certificate_hash(payload),
            "RECOVERABLE" if recoverable else "RECOVERABLE_SET_MEMBERSHIP_FAILED",
        )

    def _target_cell(self, context: "MissionActionContext"):
        task_successor_cell_id = getattr(context, "task_successor_cell_id", None)
        if task_successor_cell_id is not None:
            return self.provider._cells_by_id.get(task_successor_cell_id)
        if context.root_index is None:
            return None
        return self.provider._target_root(context.root_index)

    def certify_action_set(
        self,
        state: CertificateState,
        action_set: Zonotope3,
        context: "MissionActionContext",
    ) -> RecoverabilityActionCertificate:
        current = self.membership(state, context)
        target = self._target_cell(context)
        actuator = bool(
            np.all(np.asarray(action_set.action_bounds.low) >= -self.runtime.config.a_max - 1e-12)
            and np.all(np.asarray(action_set.action_bounds.high) <= self.runtime.config.a_max + 1e-12)
        )
        if target is None:
            successor_energy_lower = float("-inf")
            successor_required = float("inf")
            collision = velocity = energy = versions = False
        else:
            envelope = self.runtime.envelope_builder.propagate_zonotope(state, action_set)
            successor_energy_lower = float(envelope.energy_low)
            successor_required = float(
                target.energy_upper
                + self.runtime.scenario.terminal.minimum_energy
                + self.provider.energy_reserve
            )
            collision = bool(
                target.hash_valid
                and target.complete_successor_containment
                and target.minimum_geometry_slack >= -1e-12
                and self.provider._candidate_successor_in_root(state, action_set, target)
            )
            velocity = target.state_bounds.velocity.contains_box(envelope.velocity, 1e-12)
            energy = successor_energy_lower >= successor_required - 1e-12
            expected_versions = self.provider._versions()
            versions = bool(
                target.dynamics_version == expected_versions[1]
                and target.tracking_version == expected_versions[2]
                and target.energy_version == expected_versions[3]
                and target.terminal_version == expected_versions[4]
                and target.kappa_version == expected_versions[5]
                and state.bound_versions.get("dynamics") == self.runtime.calibration.dynamics.version
                and state.bound_versions.get("tracking") == self.runtime.calibration.tracking.version
                and state.bound_versions.get("energy") == self.runtime.calibration.energy.version
                and state.bound_versions.get("terminal") == self.runtime.calibration.terminal.version
                and envelope.dynamics_bound_version == self.runtime.envelope_builder.dynamics.version
                and envelope.energy_bound_version == self.runtime.envelope_builder.energy.version
            )
        verified = bool(current.recoverable and actuator and collision and velocity and energy and versions)
        reason = "VERIFIED_A_REC" if verified else next(
            name
            for name, condition in (
                ("CURRENT_STATE_NOT_RECOVERABLE", current.recoverable),
                ("NO_RECOVERABLE_SUCCESSOR_CELL", target is not None),
                ("ACTUATOR_SET_EXCLUSION", actuator),
                ("COLLISION_SUCCESSOR_EXCLUSION", collision),
                ("VELOCITY_SUCCESSOR_EXCLUSION", velocity),
                ("ENERGY_SUCCESSOR_EXCLUSION", energy),
                ("RECOVERABILITY_VERSION_MISMATCH", versions),
            )
            if not condition
        )
        payload = {
            "state": state.snapshot(),
            "action_set": repr(action_set),
            "current": current.certificate_hash,
            "target": None if target is None else target.recovery_certificate_hash,
            "energy_lower": successor_energy_lower,
            "energy_required": successor_required,
            "checks": (actuator, collision, velocity, energy, versions),
            "rule": RECOVERABILITY_ACTION_RULE_VERSION,
        }
        return RecoverabilityActionCertificate(
            verified,
            reason,
            current.certificate_hash,
            None if target is None else target.cell_id,
            successor_energy_lower,
            successor_required,
            actuator,
            collision,
            velocity,
            energy,
            versions,
            RECOVERABILITY_ACTION_RULE_VERSION,
            certificate_hash(payload),
        )

    def certify_point_action(
        self,
        state: CertificateState,
        action: np.ndarray,
        context: "MissionActionContext",
    ) -> RecoverabilityActionCertificate:
        selected = np.asarray(action, dtype=np.float64)
        return self.certify_action_set(state, Zonotope3.diagonal(selected, (0.0, 0.0, 0.0)), context)

    def successor_stays_in_charging_set(self, state: CertificateState, action: np.ndarray) -> bool:
        envelope = self.runtime.envelope_builder.propagate_point_action(state, tuple(np.asarray(action, dtype=np.float64)))
        return self._envelope_stays_in_charging_set(envelope)

    def _envelope_stays_in_charging_set(self, envelope) -> bool:
        terminal = self.runtime.scenario.terminal
        return bool(
            np.all(np.asarray(envelope.position.low) >= terminal.position_low - 1e-12)
            and np.all(np.asarray(envelope.position.high) <= terminal.position_high + 1e-12)
            and np.all(np.asarray(envelope.velocity.low) >= -terminal.velocity_abs_max - 1e-12)
            and np.all(np.asarray(envelope.velocity.high) <= terminal.velocity_abs_max + 1e-12)
        )

    def action_set_stays_in_charging_set(self, state: CertificateState, action_set: Zonotope3) -> bool:
        envelope = self.runtime.envelope_builder.propagate_zonotope(state, action_set)
        return self._envelope_stays_in_charging_set(envelope)

    def restrict_to_charging_set(
        self,
        state: CertificateState,
        action_set: Zonotope3,
        context: "MissionActionContext",
    ) -> tuple[Zonotope3 | None, RecoverabilityActionCertificate | None]:
        """Largest deterministic uniform scaling that stays in G_charge and A_rec."""

        minimum_sigma = float(self.runtime.config.minimum_generator_sigma)
        current_sigma = float(action_set.sigma_min_lower_bound)
        if current_sigma < minimum_sigma - 1e-12:
            return None, None
        minimum_factor = minimum_sigma / current_sigma
        center = np.asarray(action_set.center, dtype=np.float64)
        generators = np.asarray(action_set.generators, dtype=np.float64)

        def scaled(factor: float) -> Zonotope3:
            matrix = generators * factor
            return Zonotope3(
                tuple(float(value) for value in center),
                tuple(tuple(float(value) for value in row) for row in matrix),
            )

        minimum = scaled(minimum_factor)
        minimum_certificate = self.certify_action_set(state, minimum, context)
        if not minimum_certificate.verified or not self.action_set_stays_in_charging_set(state, minimum):
            return None, None
        if self.action_set_stays_in_charging_set(state, action_set):
            full_certificate = self.certify_action_set(state, action_set, context)
            return (action_set, full_certificate) if full_certificate.verified else (None, None)
        low, high = minimum_factor, 1.0
        accepted = minimum
        accepted_certificate = minimum_certificate
        for _ in range(self.runtime.config.generator_bisection_iterations):
            factor = (low + high) / 2.0
            candidate = scaled(factor)
            candidate_certificate = self.certify_action_set(state, candidate, context)
            if candidate_certificate.verified and self.action_set_stays_in_charging_set(state, candidate):
                accepted = candidate
                accepted_certificate = candidate_certificate
                low = factor
            else:
                high = factor
        return accepted, accepted_certificate

    def certified_station_hold(self, state: CertificateState) -> bool:
        zero = np.zeros(3, dtype=np.float64)
        return bool(
            self.runtime.scenario.terminal.is_charge_admissible(self.runtime.plant.state)
            and self.successor_stays_in_charging_set(state, zero)
        )

    def policy_authority(
        self,
        state: CertificateState,
        context: "MissionActionContext",
        goal_position: np.ndarray,
        station_position: np.ndarray,
    ) -> PolicyAuthorityCertificate:
        certificate = context.closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        if zonotope is None:
            return PolicyAuthorityCertificate(False, 3, False, False, False, False, False, False, 0.0, float("inf"), 0.0, "NO_GENERATOR_SET")
        action_certificate = self.certify_action_set(state, zonotope, context)
        center = np.asarray(zonotope.center)
        generators = np.asarray(zonotope.generators)
        sigma = float(zonotope.sigma_min_lower_bound)
        condition = float(zonotope.condition_number_upper_bound)
        full_rank = bool(abs(zonotope.determinant) > 0.0 and np.linalg.matrix_rank(generators) == 3)
        nondegenerate = bool(sigma >= self.runtime.config.minimum_generator_sigma - 1e-12)
        neutral = bool(np.linalg.norm(center) <= 1e-10)

        def direction_available(target: np.ndarray) -> bool:
            direction = np.asarray(target, dtype=np.float64) - np.asarray(state.position, dtype=np.float64)
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12 or not full_rank:
                return True
            desired = 0.5 * sigma * direction / norm
            eta = np.linalg.solve(generators, desired)
            return bool(np.max(np.abs(eta)) <= 1.0 + 1e-12 and float(desired @ direction) > 0.0)

        goal = direction_available(goal_position)
        station = direction_available(station_position)
        passed = bool(neutral and full_rank and nondegenerate and goal and station and action_certificate.verified)
        return PolicyAuthorityCertificate(
            passed,
            3,
            neutral,
            full_rank,
            nondegenerate,
            goal,
            station,
            action_certificate.verified,
            sigma,
            condition,
            float(8.0 * abs(zonotope.determinant)),
            "PASS" if passed else "POLICY_AUTHORITY_INSUFFICIENT",
        )
