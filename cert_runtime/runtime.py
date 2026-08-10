from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from typing import Protocol, Sequence

from .recovery import FrozenRecoveryPolicy, RecoveryDecision
from .state import CertificateState, CertificateStateSnapshot
from .types import Vec3, Zonotope3, vec3
from .watchdog import CandidateBundle
from .zonotope import ZonotopeCertificate, ZonotopeConstructor


class TaskActor(Protocol):
    def sample_u(self, observation: Sequence[float]) -> Sequence[float]: ...


@dataclass(frozen=True)
class ReplayRecord:
    certificate_state: CertificateStateSnapshot
    task_observation: tuple[float, ...]
    nominal_pre_squash_u: Vec3 | None
    squashed_eta: Vec3 | None
    candidate_action: Vec3 | None
    executed_action: Vec3
    recovery_action: Vec3
    accepted: bool
    fallback_reason: str | None
    zonotope_center: Vec3 | None
    zonotope_generators: tuple[Vec3, Vec3, Vec3] | None
    recovery_certificate_hash: str | None
    inclusion_certificate_hash: str | None
    timestamp: float = 0.0
    bound_versions: tuple[tuple[str, str], ...] = ()
    measured_tracking_action: Vec3 | None = None

    @property
    def certificate_version(self) -> tuple[int, int, int]:
        return self.certificate_state.certificate_version

    @property
    def critic_action(self) -> Vec3:
        return self.executed_action


class CertificateReplay:
    def __init__(self) -> None:
        self.records: list[ReplayRecord] = []

    def append(self, record: ReplayRecord) -> None:
        self.records.append(record)


@dataclass(frozen=True)
class RuntimeDecision:
    action: Vec3
    accepted: bool
    fallback_reason: str | None
    certificate: ZonotopeCertificate
    recovery_certified: bool


class RuntimeCertifier:
    """State-level set acceptance followed by affine-tanh membership mapping."""

    def __init__(
        self,
        actor: TaskActor,
        recovery_policy: FrozenRecoveryPolicy,
        constructor: ZonotopeConstructor,
        replay: CertificateReplay,
    ) -> None:
        self.actor = actor
        self.recovery_policy = recovery_policy
        self.constructor = constructor
        self.replay = replay

    def recovery_decision(self, state: CertificateState, timestamp: float) -> RecoveryDecision:
        return self.recovery_policy.certified_action(
            state,
            self.constructor.envelope_builder.dynamics.version,
            self.constructor.envelope_builder.energy.version,
            timestamp,
        )

    def prepare_candidate_bundle(
        self,
        state: CertificateState,
        task_observation: Sequence[float],
        recovery: RecoveryDecision,
        timestamp: float,
    ) -> CandidateBundle:
        snapshot = state.snapshot()
        certificate = self.constructor.construct(state, recovery, timestamp)
        if not certificate.verified or certificate.zonotope is None:
            raise RuntimeError(certificate.reason)
        return self.prepare_candidate_from_certificate(
            state, task_observation, recovery, certificate, timestamp, snapshot
        )

    def prepare_candidate_from_certificate(
        self,
        state: CertificateState,
        task_observation: Sequence[float],
        recovery: RecoveryDecision,
        certificate: ZonotopeCertificate,
        timestamp: float,
        snapshot: CertificateStateSnapshot | None = None,
    ) -> CandidateBundle:
        snapshot = state.snapshot() if snapshot is None else snapshot
        if not certificate.verified or certificate.zonotope is None:
            raise RuntimeError(certificate.reason)
        if certificate.certificate_version != snapshot.certificate_version:
            raise RuntimeError("CERTIFICATE_VERSION_CHANGED")
        raw_u = tuple(float(value) for value in self.actor.sample_u(task_observation))
        if len(raw_u) != 3 or not all(isfinite(value) for value in raw_u):
            raise RuntimeError("ACTOR_NONFINITE")
        nominal_u = vec3(raw_u)
        eta = vec3(tanh(value) for value in nominal_u)
        candidate = certificate.zonotope.map_eta(eta)
        if not certificate.zonotope.contains(candidate, self.constructor.config.numerical_tolerance):
            raise RuntimeError("MEMBERSHIP_NUMERICAL_FAILURE")
        if state.snapshot() != snapshot:
            raise RuntimeError("CERTIFICATE_VERSION_CHANGED")
        return CandidateBundle(snapshot, certificate, nominal_u, eta, certificate.zonotope, candidate, True)

    def record_published_command(
        self,
        snapshot: CertificateStateSnapshot,
        task_observation: Sequence[float],
        recovery: RecoveryDecision,
        command_action: Vec3,
        command_source: str,
        reason: str,
        timestamp: float,
        bundle: CandidateBundle | None = None,
        measured_tracking_action: Vec3 | None = None,
    ) -> None:
        accepted = command_source == "task"
        if accepted and (bundle is None or not bundle.complete):
            raise ValueError("task publication requires a complete candidate bundle")
        self.replay.append(
            ReplayRecord(
                snapshot,
                tuple(float(value) for value in task_observation),
                bundle.nominal_u if accepted and bundle else None,
                bundle.eta if accepted and bundle else None,
                bundle.final_action if accepted and bundle else None,
                command_action,
                recovery.action,
                accepted,
                None if accepted else reason,
                bundle.zonotope.center if accepted and bundle else None,
                bundle.zonotope.generators if accepted and bundle else None,
                recovery.certificate_hash,
                bundle.certificate.complete_set_inclusion_hash if accepted and bundle else None,
                timestamp,
                snapshot.bound_versions,
                measured_tracking_action,
            )
        )

    def step(self, state: CertificateState, task_observation: Sequence[float]) -> RuntimeDecision:
        timestamp = self.constructor.clock()
        snapshot = state.snapshot()
        recovery = self.recovery_decision(state, timestamp)
        certificate = self.constructor.construct(state, recovery, timestamp)
        if not certificate.verified or certificate.zonotope is None:
            self._record_fallback(snapshot, task_observation, recovery, certificate.reason, certificate)
            return RuntimeDecision(recovery.action, False, certificate.reason, certificate, recovery.certified)
        try:
            raw_u = tuple(float(value) for value in self.actor.sample_u(task_observation))
        except Exception:
            self._record_fallback(snapshot, task_observation, recovery, "ACTOR_FAILURE", certificate)
            return RuntimeDecision(recovery.action, False, "ACTOR_FAILURE", certificate, recovery.certified)
        if len(raw_u) != 3 or not all(isfinite(value) for value in raw_u):
            self._record_fallback(snapshot, task_observation, recovery, "ACTOR_NONFINITE", certificate)
            return RuntimeDecision(recovery.action, False, "ACTOR_NONFINITE", certificate, recovery.certified)
        nominal_u = vec3(raw_u)
        if state.snapshot() != snapshot:
            refreshed = self.recovery_decision(state, self.constructor.clock())
            self._record_fallback(state.snapshot(), task_observation, refreshed, "CERTIFICATE_VERSION_CHANGED", certificate)
            return RuntimeDecision(
                refreshed.action,
                False,
                "CERTIFICATE_VERSION_CHANGED",
                certificate,
                refreshed.certified,
            )
        if self.constructor.clock() > certificate.deadline_at:
            self._record_fallback(snapshot, task_observation, recovery, "DEADLINE_AFTER_ACTOR", certificate)
            return RuntimeDecision(recovery.action, False, "DEADLINE_AFTER_ACTOR", certificate, recovery.certified)
        eta = vec3(tanh(value) for value in nominal_u)
        candidate = certificate.zonotope.map_eta(eta)
        if not certificate.zonotope.contains(candidate, self.constructor.config.numerical_tolerance):
            self._record_fallback(snapshot, task_observation, recovery, "MEMBERSHIP_NUMERICAL_FAILURE", certificate)
            return RuntimeDecision(
                recovery.action,
                False,
                "MEMBERSHIP_NUMERICAL_FAILURE",
                certificate,
                recovery.certified,
            )
        self.replay.append(
            ReplayRecord(
                snapshot,
                tuple(float(value) for value in task_observation),
                nominal_u,
                eta,
                candidate,
                candidate,
                recovery.action,
                True,
                None,
                certificate.zonotope.center,
                certificate.zonotope.generators,
                certificate.recovery_certificate_hash,
                certificate.complete_set_inclusion_hash,
                timestamp,
                snapshot.bound_versions,
            )
        )
        return RuntimeDecision(candidate, True, None, certificate, recovery.certified)

    def _record_fallback(
        self,
        snapshot: CertificateStateSnapshot,
        task_observation: Sequence[float],
        recovery: RecoveryDecision,
        reason: str,
        certificate: ZonotopeCertificate,
    ) -> None:
        zonotope: Zonotope3 | None = certificate.zonotope
        self.replay.append(
            ReplayRecord(
                snapshot,
                tuple(float(value) for value in task_observation),
                None,
                None,
                None,
                recovery.action,
                recovery.action,
                False,
                reason,
                zonotope.center if zonotope else None,
                zonotope.generators if zonotope else None,
                recovery.certificate_hash,
                certificate.complete_set_inclusion_hash,
                self.constructor.clock(),
                snapshot.bound_versions,
            )
        )
