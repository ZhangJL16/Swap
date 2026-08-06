from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from json import dumps
from math import isfinite
from time import monotonic
from typing import Any

import numpy as np

from cert_runtime.certificates import ProofMetadata, certificate_hash
from cert_runtime.recovery import RecoveryDecision
from cert_runtime.state import CertificateState
from cert_runtime.types import Zonotope3
from cert_runtime.zonotope import ZonotopeCertificate


@dataclass(frozen=True)
class MissionFailureWitness:
    failed_predicate: str


@dataclass(frozen=True)
class MissionClosureResult:
    closed: bool
    zonotope_certificate: ZonotopeCertificate | None
    status: str
    failure_witness: MissionFailureWitness | None = None
    manifest: None = None


@dataclass(frozen=True)
class MissionActionContext:
    recovery: RecoveryDecision
    closure: MissionClosureResult
    required_energy: float
    current_energy_margin: float

    @property
    def generator_available(self) -> bool:
        return bool(
            self.closure.closed
            and self.closure.zonotope_certificate is not None
            and self.closure.zonotope_certificate.verified
            and self.closure.zonotope_certificate.zonotope is not None
        )


class SyntheticMissionCertificateProvider:
    """Fast synthetic action-set fixture.

    It uses only explicit free/occupied boxes from the scenario's certificate
    profile.  It is software experiment evidence, not calibrated flight proof.
    Complete diagonal action intervals are propagated analytically for one step.
    """

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        self.profile = runtime.scenario.mission_config
        self.free_boxes = tuple(np.asarray(box, dtype=np.float64) for box in self.profile["free_boxes"])
        self.occupied_boxes = tuple(np.asarray(box, dtype=np.float64) for box in self.profile.get("occupied_boxes", ()))
        self.waypoints = tuple(np.asarray(point, dtype=np.float64) for point in self.profile["return_waypoints"])
        self.base_scales = np.asarray(self.profile["generator_scale"], dtype=np.float64)
        self.narrow_regions = tuple(np.asarray(box, dtype=np.float64) for box in self.profile.get("narrow_regions", ()))
        self.reserve_per_meter = float(self.profile.get("energy_reserve_per_meter", 0.1))
        self.trigger_margin = float(self.profile.get("return_trigger_margin", 1.0))
        self.version = "synthetic-mission-provider-v1"
        self.last_context: MissionActionContext | None = None

    def _path_distance(self, position: np.ndarray) -> float:
        nearest = min(range(len(self.waypoints)), key=lambda index: float(np.linalg.norm(position - self.waypoints[index])))
        distance = float(np.linalg.norm(position - self.waypoints[nearest]))
        for left, right in zip(self.waypoints[nearest:], self.waypoints[nearest + 1 :]):
            distance += float(np.linalg.norm(left - right))
        return distance

    def required_energy(self, position: np.ndarray) -> float:
        terminal = self.runtime.scenario.terminal.minimum_energy
        return terminal + self.trigger_margin + self.reserve_per_meter * self._path_distance(position)

    def recovery_action(self, state: CertificateState) -> tuple[float, float, float]:
        position = np.asarray(state.position)
        velocity = np.asarray(state.velocity)
        nearest = min(range(len(self.waypoints)), key=lambda index: float(np.linalg.norm(position - self.waypoints[index])))
        target_index = min(nearest + 1, len(self.waypoints) - 1)
        if float(np.linalg.norm(position - self.waypoints[target_index])) < 0.12 and target_index < len(self.waypoints) - 1:
            target_index += 1
        target = self.waypoints[target_index]
        raw = 0.9 * (target - position) - 1.4 * velocity
        maximum = self.runtime.config.a_max
        lower_velocity_action = (-self.runtime.config.v_max - velocity) / self.runtime.config.dt
        upper_velocity_action = (self.runtime.config.v_max - velocity) / self.runtime.config.dt
        bounded = np.minimum(np.maximum(raw, np.maximum(-maximum, lower_velocity_action)), np.minimum(maximum, upper_velocity_action))
        return tuple(float(value) for value in bounded)

    @staticmethod
    def _box_contains(box: np.ndarray, low: np.ndarray, high: np.ndarray, margin: float = 0.0) -> bool:
        return bool(np.all(low[:2] >= box[:2] + margin) and np.all(high[:2] <= box[2:] - margin))

    @staticmethod
    def _boxes_overlap(left_low: np.ndarray, left_high: np.ndarray, box: np.ndarray, margin: float) -> bool:
        return bool(
            left_high[0] >= box[0] - margin
            and left_low[0] <= box[2] + margin
            and left_high[1] >= box[1] - margin
            and left_low[1] <= box[3] + margin
        )

    def _inside_narrow_region(self, position: np.ndarray) -> bool:
        return any(box[0] <= position[0] <= box[2] and box[1] <= position[1] <= box[3] for box in self.narrow_regions)

    def _continuous_one_step_valid(self, state: CertificateState, zonotope: Zonotope3) -> bool:
        config = self.runtime.config
        action = zonotope.action_bounds
        action_low = np.asarray(action.low)
        action_high = np.asarray(action.high)
        if np.any(action_low < -config.a_max - 1e-12) or np.any(action_high > config.a_max + 1e-12):
            return False
        tracking = config.tracking_error_bound
        dt_low = config.dt
        dt_high = config.dt + config.total_latency
        position = np.asarray(state.position)
        velocity = np.asarray(state.velocity)
        acceleration_low = action_low - tracking
        acceleration_high = action_high + tracking
        candidates = []
        for duration in (dt_low, dt_high):
            candidates.append(position + duration * velocity + 0.5 * duration * duration * acceleration_low)
            candidates.append(position + duration * velocity + 0.5 * duration * duration * acceleration_high)
        candidates.append(position - np.asarray(state.position_error_radius))
        candidates.append(position + np.asarray(state.position_error_radius))
        position_low = np.min(np.stack(candidates), axis=0) - np.asarray(state.position_error_radius)
        position_high = np.max(np.stack(candidates), axis=0) + np.asarray(state.position_error_radius)
        velocity_low = velocity + dt_high * acceleration_low - np.asarray(state.velocity_error_radius)
        velocity_high = velocity + dt_high * acceleration_high + np.asarray(state.velocity_error_radius)
        if np.any(velocity_low < -config.v_max - 1e-12) or np.any(velocity_high > config.v_max + 1e-12):
            return False
        margin = config.body_radius + config.geometry_margin
        if position_low[2] < margin or position_high[2] > config.world_size[2] - margin:
            return False
        if not any(self._box_contains(box, position_low, position_high, margin) for box in self.free_boxes):
            return False
        if any(self._boxes_overlap(position_low, position_high, box, margin) for box in self.occupied_boxes):
            return False
        return bool(np.all(np.isfinite(position_low)) and np.all(np.isfinite(position_high)))

    def _construct_zonotope(self, state: CertificateState) -> Zonotope3 | None:
        position = np.asarray(state.position)
        if self._inside_narrow_region(position):
            return None
        center = np.clip(-0.25 * np.asarray(state.velocity), -0.25 * self.runtime.config.a_max, 0.25 * self.runtime.config.a_max)
        scales = np.minimum(self.base_scales, self.runtime.config.a_max - np.abs(center))
        minimum = self.runtime.config.minimum_generator_sigma
        if np.any(scales < minimum):
            return None
        # Deterministic common-factor bisection; every accepted factor is checked
        # using the complete action interval, not samples or vertices.
        low, high = 0.0, 1.0
        accepted: Zonotope3 | None = None
        for _ in range(self.runtime.config.generator_bisection_iterations + 1):
            factor = high if accepted is None and low == 0.0 else (low + high) / 2.0
            candidate_scales = np.maximum(minimum, scales * factor)
            candidate = Zonotope3.diagonal(center, candidate_scales)
            if (
                candidate.sigma_min_lower_bound >= minimum
                and candidate.condition_number_upper_bound <= self.runtime.config.maximum_generator_condition
                and self._continuous_one_step_valid(state, candidate)
            ):
                accepted = candidate
                low = factor
            else:
                high = factor
        return accepted

    def evaluate(self, state: CertificateState, timestamp: float | None = None) -> MissionActionContext:
        now = monotonic() if timestamp is None else timestamp
        required = self.required_energy(np.asarray(state.position))
        recovery_action = self.recovery_action(state)
        inside_free = any(self._box_contains(box, np.asarray(state.position), np.asarray(state.position)) for box in self.free_boxes)
        recovery_hash = certificate_hash({
            "provider": self.version,
            "state_version": state.certificate_version,
            "action": recovery_action,
            "required_energy": required,
        })
        recovery_set = Zonotope3.diagonal(recovery_action, (0.0, 0.0, 0.0))
        recovery_valid = (
            inside_free
            and state.energy >= required
            and all(isfinite(value) for value in recovery_action)
            and self._continuous_one_step_valid(state, recovery_set)
        )
        recovery = RecoveryDecision(recovery_action, recovery_valid, recovery_hash if recovery_valid else None, "certified" if recovery_valid else "synthetic-recovery-invalid")
        zonotope = self._construct_zonotope(state) if recovery_valid else None
        reason = "VERIFIED" if zonotope is not None else ("NO_GENERATOR_SET" if recovery_valid else "INSUFFICIENT_RECOVERY_RESERVE")
        metadata = None
        inclusion_hash = None
        certificate = None
        if zonotope is not None:
            versions = dict(state.bound_versions)
            metadata = ProofMetadata(
                f"mission-zonotope-{state.certificate_version}",
                state.certificate_version[2],
                versions.get("sensor", "synthetic-sensor"),
                versions.get("dynamics", "synthetic-dynamics"),
                versions.get("tracking", "synthetic-tracking"),
                versions.get("energy", "synthetic-energy"),
                versions.get("terminal", "synthetic-terminal"),
                state.certificate_version[0],
                state.certificate_version[1],
                versions.get("kappa", self.version),
                now,
                now + 1000.0,
                (recovery_hash,),
            )
            inclusion_hash = sha256(dumps({"z": repr(zonotope), "recovery": recovery_hash, "version": state.certificate_version}, sort_keys=True).encode()).hexdigest()
            certificate = ZonotopeCertificate(
                True,
                "VERIFIED",
                zonotope,
                None,
                state.certificate_version,
                recovery_hash,
                inclusion_hash,
                tuple(sorted(state.bound_versions.items())),
                metadata,
                0.0,
                now + self.runtime.config.certification_deadline,
                1,
                self.runtime.config.generator_bisection_iterations,
            )
        closure = MissionClosureResult(zonotope is not None, certificate, reason, None if zonotope is not None else MissionFailureWitness(reason))
        context = MissionActionContext(recovery, closure, required, state.energy - required)
        self.last_context = context
        return context
