from __future__ import annotations

from dataclasses import dataclass
from math import hypot, isfinite
from time import monotonic
from typing import Callable

from .certificates import ProofMetadata, certificate_hash
from .envelope import SuccessorEnvelope, SuccessorEnvelopeBuilder
from .interval import round_up
from .recovery import RecoveryDecision
from .state import CertificateState
from .types import Vec3, Zonotope3, vec3


@dataclass(frozen=True)
class CertificateConfig:
    actuator_low: Vec3
    actuator_high: Vec3
    minimum_sigma: float
    maximum_condition_number: float
    terminal_energy: float
    energy_reserve: float
    braking_deceleration: float
    update_latency: float
    geometry_margin: float
    numerical_tolerance: float
    deadline_seconds: float
    bisection_iterations: int = 12

    def __post_init__(self) -> None:
        if any(low >= high for low, high in zip(self.actuator_low, self.actuator_high)):
            raise ValueError("invalid actuator interval")
        if (
            self.minimum_sigma <= 0.0
            or self.maximum_condition_number < 1.0
            or self.braking_deceleration <= 0.0
        ):
            raise ValueError("invalid rank, conditioning, or braking bound")
        if self.deadline_seconds < 0.0 or self.bisection_iterations < 0:
            raise ValueError("invalid deadline or iteration count")


@dataclass(frozen=True)
class ZonotopeCertificate:
    verified: bool
    reason: str
    zonotope: Zonotope3 | None
    successor_envelope: SuccessorEnvelope | None
    certificate_version: tuple[int, int, int]
    recovery_certificate_hash: str | None
    complete_set_inclusion_hash: str | None
    bound_versions: tuple[tuple[str, str], ...]
    proof_metadata: ProofMetadata | None
    elapsed_seconds: float
    deadline_at: float
    verifier_calls: int = 0
    bisection_steps: int = 0


class ZonotopeConstructor:
    """Deterministic lexicographic inner-zonotope constructor and verifier."""

    def __init__(
        self,
        envelope_builder: SuccessorEnvelopeBuilder,
        config: CertificateConfig,
        kappa_parameter_version: str,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        self.envelope_builder = envelope_builder
        self.config = config
        self.kappa_parameter_version = kappa_parameter_version
        self.clock = clock
        self.calls = 0

    @property
    def worst_case_verifier_calls(self) -> int:
        return 2 + 3 * self.config.bisection_iterations

    def construct(
        self,
        state: CertificateState,
        recovery: RecoveryDecision,
        timestamp: float | None = None,
    ) -> ZonotopeCertificate:
        self.calls += 1
        started = self.clock()
        now = started if timestamp is None else timestamp
        version = state.certificate_version
        bound_versions = tuple(sorted(state.bound_versions.items()))
        if not recovery.certified or recovery.certificate_hash is None:
            return self._failure("RECOVERY_NOT_CERTIFIED", version, recovery, started)
        if self._expired(started):
            return self._failure("DEADLINE", version, recovery, started)
        center = vec3(recovery.action)
        maximum_scales = tuple(
            min(center[index] - self.config.actuator_low[index], self.config.actuator_high[index] - center[index])
            for index in range(3)
        )
        if any(scale + self.config.numerical_tolerance < self.config.minimum_sigma for scale in maximum_scales):
            return self._failure("NO_GENERATOR_SET", version, recovery, started)
        scales = [self.config.minimum_sigma] * 3
        initial = Zonotope3.diagonal(center, scales)
        if self.verify_complete(state, initial, recovery, now) is None:
            return self._failure("NO_GENERATOR_SET", version, recovery, started)
        for axis in range(3):
            low = scales[axis]
            high = maximum_scales[axis]
            for _ in range(self.config.bisection_iterations):
                if state.certificate_version != version or tuple(sorted(state.bound_versions.items())) != bound_versions:
                    return self._failure("CERTIFICATE_VERSION_CHANGED", version, recovery, started)
                if self._expired(started):
                    return self._failure("DEADLINE", version, recovery, started)
                middle = (low + high) / 2.0
                candidate_scales = list(scales)
                candidate_scales[axis] = middle
                candidate = Zonotope3.diagonal(center, candidate_scales)
                if self.verify_complete(state, candidate, recovery, now) is not None:
                    low = middle
                else:
                    high = middle
            scales[axis] = low
        zonotope = Zonotope3.diagonal(center, scales)
        envelope = self.verify_complete(state, zonotope, recovery, now)
        if envelope is None:
            return self._failure("FINAL_SET_VERIFICATION_FAILED", version, recovery, started)
        if state.certificate_version != version or tuple(sorted(state.bound_versions.items())) != bound_versions:
            return self._failure("CERTIFICATE_VERSION_CHANGED", version, recovery, started)
        if self._expired(started):
            return self._failure("DEADLINE", version, recovery, started)
        current_level = state.return_corridor.locate_level((state.position[0], state.position[1]))
        recovery_metadata = None
        if current_level is not None:
            current_cell = next(
                cell for cell in state.return_corridor.cells if cell.cell_id == current_level
            )
            if current_cell.recovery_certificate is not None:
                recovery_metadata = current_cell.recovery_certificate.proof_metadata
        if recovery_metadata is None:
            return self._failure("MISSING_PROOF_METADATA", version, recovery, started)
        proof_metadata = ProofMetadata(
            f"zonotope-{version[1]}-{version[2]}",
            version[2],
            recovery_metadata.sensor_version,
            recovery_metadata.dynamics_version,
            recovery_metadata.tracking_version,
            recovery_metadata.energy_version,
            recovery_metadata.terminal_version,
            version[0],
            version[1],
            recovery_metadata.kappa_version,
            now,
            started + self.config.deadline_seconds,
            (recovery.certificate_hash,),
        )
        inclusion_hash = certificate_hash(
            {
                "version": version,
                "recovery_hash": recovery.certificate_hash,
                "center": zonotope.center,
                "generators": zonotope.generators,
                "successor": repr(envelope),
                "sigma_lower": zonotope.sigma_min_lower_bound,
                "condition_upper": zonotope.condition_number_upper_bound,
                "bound_versions": bound_versions,
                "proof_metadata": proof_metadata,
            }
        )
        return ZonotopeCertificate(
            True,
            "VERIFIED",
            zonotope,
            envelope,
            version,
            recovery.certificate_hash,
            inclusion_hash,
            bound_versions,
            proof_metadata,
            self.clock() - started,
            started + self.config.deadline_seconds,
            self.worst_case_verifier_calls,
            3 * self.config.bisection_iterations,
        )

    def verify_complete(
        self,
        state: CertificateState,
        zonotope: Zonotope3,
        recovery: RecoveryDecision,
        timestamp: float,
    ) -> SuccessorEnvelope | None:
        tolerance = self.config.numerical_tolerance
        if not recovery.certified or recovery.certificate_hash is None:
            return None
        if zonotope.sigma_min_lower_bound + tolerance < self.config.minimum_sigma:
            return None
        if zonotope.condition_number_upper_bound > self.config.maximum_condition_number + tolerance:
            return None
        action_bounds = zonotope.action_bounds
        if any(
            action_bounds.low[index] < self.config.actuator_low[index] - tolerance
            or action_bounds.high[index] > self.config.actuator_high[index] + tolerance
            for index in range(3)
        ):
            return None
        envelope = self.envelope_builder.propagate_zonotope(state, zonotope)
        if not all(
            isfinite(value)
            for value in envelope.position.low
            + envelope.position.high
            + envelope.velocity.low
            + envelope.velocity.high
            + (envelope.energy_low, envelope.energy_high)
        ):
            return None
        speed_abs = envelope.velocity.max_abs()
        horizontal_speed = hypot(speed_abs[0], speed_abs[1])
        stopping_margin = round_up(
            round_up(horizontal_speed * self.config.update_latency)
            + round_up(
                horizontal_speed
                * horizontal_speed
                / (2.0 * self.config.braking_deceleration)
            )
            + self.config.geometry_margin
        )
        stopping_box = envelope.position.horizontal_box().expanded(stopping_margin)
        if not state.local_geometry.box_is_verified_free(stopping_box):
            return None
        successor_cell = state.return_corridor.containing_cell_for_envelope(
            envelope.position,
            envelope.velocity,
            envelope.energy_low,
            self.config.energy_reserve,
            self.config.terminal_energy,
            tolerance,
            state.local_geometry.version,
            self.kappa_parameter_version,
            self.envelope_builder.dynamics.version,
            self.envelope_builder.energy.version,
            timestamp,
        )
        if successor_cell is None:
            return None
        return envelope

    def _expired(self, started: float) -> bool:
        return self.clock() - started > self.config.deadline_seconds

    def _failure(
        self,
        reason: str,
        version: tuple[int, int, int],
        recovery: RecoveryDecision,
        started: float,
    ) -> ZonotopeCertificate:
        return ZonotopeCertificate(
            False,
            reason,
            None,
            None,
            version,
            recovery.certificate_hash,
            None,
            tuple(),
            None,
            self.clock() - started,
            started + self.config.deadline_seconds,
        )
