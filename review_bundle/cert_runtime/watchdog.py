from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, tanh
from queue import Empty, Queue
from threading import Lock, Thread
from time import monotonic
from typing import Callable

from .contracts import WCETContract
from .state import CertificateStateSnapshot
from .types import Vec3, Zonotope3
from .zonotope import ZonotopeCertificate


@dataclass(frozen=True)
class CandidateBundle:
    snapshot: CertificateStateSnapshot
    certificate: ZonotopeCertificate
    nominal_u: Vec3
    eta: Vec3
    zonotope: Zonotope3
    final_action: Vec3
    atomic_acceptance: bool

    @property
    def complete(self) -> bool:
        finite = all(
            isfinite(value)
            for value in self.nominal_u + self.eta + self.final_action
        )
        eta_matches = all(
            abs(self.eta[index] - tanh(self.nominal_u[index])) <= 1e-12
            for index in range(3)
        )
        mapped = self.zonotope.map_eta(self.eta) if finite and all(abs(value) <= 1.0 for value in self.eta) else None
        action_matches = mapped is not None and all(
            abs(mapped[index] - self.final_action[index]) <= 1e-12
            for index in range(3)
        )
        return (
            self.atomic_acceptance
            and finite
            and eta_matches
            and action_matches
            and self.certificate.verified
            and self.certificate.complete_set_inclusion_hash is not None
            and self.certificate.zonotope == self.zonotope
            and self.zonotope.contains(self.final_action)
            and self.certificate.certificate_version == self.snapshot.certificate_version
            and self.certificate.bound_versions == self.snapshot.bound_versions
            and self.certificate.proof_metadata is not None
        )


@dataclass(frozen=True)
class PublishedCommand:
    action: Vec3
    source: str
    reason: str
    certificate_version: tuple[int, int, int]


@dataclass(frozen=True)
class WatchdogTrace:
    kappa_staged_before_worker: bool
    staged_kappa_action: Vec3
    outcome: str
    elapsed_seconds: float
    publication_count: int


class AtomicCommandPublisher:
    """Single-assignment command register used by the watchdog."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._command: PublishedCommand | None = None
        self._staged_default: PublishedCommand | None = None
        self.publication_count = 0
        self.last_publish_elapsed = 0.0

    def stage_default(self, command: PublishedCommand) -> bool:
        with self._lock:
            if self._command is not None or self._staged_default is not None:
                return False
            self._staged_default = command
            return True

    def publish_once(self, command: PublishedCommand) -> bool:
        started = monotonic()
        with self._lock:
            if self._command is not None:
                self.last_publish_elapsed = monotonic() - started
                return False
            self._command = command
            self.publication_count += 1
            self.last_publish_elapsed = monotonic() - started
            return True

    @property
    def command(self) -> PublishedCommand | None:
        with self._lock:
            return self._command

    @property
    def staged_default(self) -> PublishedCommand | None:
        with self._lock:
            return self._staged_default


class SimulatedWatchdog:
    """Independent fail-default state machine; not an RTOS/WCET certificate."""

    def __init__(
        self,
        deadline_seconds: float,
        wcet_contract: WCETContract,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if deadline_seconds < 0.0:
            raise ValueError("deadline must be nonnegative")
        self.deadline_seconds = deadline_seconds
        self.wcet_contract = wcet_contract
        self.clock = clock
        self.last_trace: WatchdogTrace | None = None

    def execute(
        self,
        snapshot: CertificateStateSnapshot,
        recovery_action: Vec3,
        producer: Callable[[], CandidateBundle],
        current_version: Callable[[], tuple[int, int, int] | CertificateStateSnapshot],
        publisher: AtomicCommandPublisher | None = None,
    ) -> PublishedCommand:
        output = publisher or AtomicCommandPublisher()
        staged_kappa = PublishedCommand(
            recovery_action,
            "kappa",
            "STAGED_FAIL_DEFAULT",
            snapshot.certificate_version,
        )
        if not output.stage_default(staged_kappa):
            existing = output.command
            if existing is not None:
                self.last_trace = WatchdogTrace(True, recovery_action, existing.reason, 0.0, output.publication_count)
                return existing
            staged = output.staged_default
            if staged is None or staged.source != "kappa" or staged.action != recovery_action:
                fallback = PublishedCommand(
                    recovery_action,
                    "kappa",
                    "PUBLISHER_STAGE_CONFLICT",
                    snapshot.certificate_version,
                )
                output.publish_once(fallback)
                self.last_trace = WatchdogTrace(False, recovery_action, fallback.reason, 0.0, output.publication_count)
                return output.command  # type: ignore[return-value]
        queue: Queue[tuple[str, object]] = Queue(maxsize=1)

        def worker() -> None:
            try:
                queue.put_nowait(("candidate", producer()))
            except BaseException as error:
                try:
                    queue.put_nowait(("exception", error))
                except Exception:
                    pass

        started = self.clock()
        thread = Thread(target=worker, daemon=True, name="certificate-producer")
        thread.start()
        remaining = max(0.0, self.deadline_seconds - (self.clock() - started))
        thread.join(remaining)
        if thread.is_alive():
            command = PublishedCommand(staged_kappa.action, "kappa", "WATCHDOG_DEADLINE", snapshot.certificate_version)
            output.publish_once(command)
            self.last_trace = WatchdogTrace(True, recovery_action, command.reason, self.clock() - started, output.publication_count)
            return output.command  # type: ignore[return-value]
        try:
            result_type, payload = queue.get_nowait()
        except Empty:
            result_type, payload = "exception", RuntimeError("producer returned no bundle")
        current = current_version()
        current_matches = (
            current == snapshot
            if isinstance(current, CertificateStateSnapshot)
            else current == snapshot.certificate_version
        )
        if result_type == "candidate":
            bundle = payload
            if (
                isinstance(bundle, CandidateBundle)
                and bundle.complete
                and bundle.snapshot == snapshot
                and current_matches
                and self.clock() - started <= self.deadline_seconds
            ):
                command = PublishedCommand(
                    bundle.final_action,
                    "task",
                    "VERIFIED_BUNDLE",
                    snapshot.certificate_version,
                )
                output.publish_once(command)
                self.last_trace = WatchdogTrace(True, recovery_action, command.reason, self.clock() - started, output.publication_count)
                return output.command  # type: ignore[return-value]
            reason = "STALE_OR_INCOMPLETE_BUNDLE"
        else:
            reason = "CERTIFIER_EXCEPTION"
        command = PublishedCommand(recovery_action, "kappa", reason, snapshot.certificate_version)
        output.publish_once(command)
        self.last_trace = WatchdogTrace(True, recovery_action, command.reason, self.clock() - started, output.publication_count)
        return output.command  # type: ignore[return-value]
