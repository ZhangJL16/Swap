from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from .geometry import LidarRay
from .state import CertificateState
from .types import Vec3
from .watchdog import PublishedCommand
from .state import CertificateStateSnapshot
from .watchdog import CandidateBundle


class StateSource(Protocol):
    def read_state(self) -> CertificateState: ...


class LidarSource(Protocol):
    def read_rays(self) -> tuple[LidarRay, ...]: ...


class ActuatorCommandSink(Protocol):
    def publish(self, command: PublishedCommand) -> bool: ...
    def publish_emergency(self, action: Vec3, reason: str) -> bool: ...


class PowerEnergySource(Protocol):
    def read_energy(self) -> float: ...


class TimestampSource(Protocol):
    def now(self) -> float: ...


class ActionTrackingSource(Protocol):
    def read_measured_action(self, published_action: Vec3) -> Vec3 | None: ...


class LogRecorder(Protocol):
    def record(self, payload: object) -> None: ...


class WatchdogAdapter(Protocol):
    deadline_seconds: float

    def execute(
        self,
        snapshot: CertificateStateSnapshot,
        recovery_action: Vec3,
        producer: Callable[[], CandidateBundle],
        current_version: Callable[[], tuple[int, int, int] | CertificateStateSnapshot],
        publisher=None,
    ) -> PublishedCommand: ...


@dataclass
class MockStateSource:
    state: CertificateState

    def read_state(self) -> CertificateState:
        return self.state


@dataclass
class MockLidarSource:
    rays: tuple[LidarRay, ...]

    def read_rays(self) -> tuple[LidarRay, ...]:
        return self.rays


class MockActuatorSink:
    def __init__(self, fail_normal_publish: bool = False) -> None:
        self.fail_normal_publish = fail_normal_publish
        self.commands: list[PublishedCommand] = []
        self.emergencies: list[tuple[Vec3, str]] = []

    def publish(self, command: PublishedCommand) -> bool:
        if self.fail_normal_publish:
            return False
        self.commands.append(command)
        return True

    def publish_emergency(self, action: Vec3, reason: str) -> bool:
        self.emergencies.append((action, reason))
        return True


@dataclass
class MockEnergySource:
    energy: float

    def read_energy(self) -> float:
        return self.energy


@dataclass
class FixedTimestampSource:
    timestamp: float

    def now(self) -> float:
        return self.timestamp


@dataclass
class MockTrackingSource:
    measured_action: Vec3 | None = None

    def read_measured_action(self, published_action: Vec3) -> Vec3 | None:
        return self.measured_action if self.measured_action is not None else published_action


class InMemoryLogRecorder:
    def __init__(self) -> None:
        self.records: list[object] = []

    def record(self, payload: object) -> None:
        self.records.append(payload)


class ReplayLogAdapter:
    """Deterministic source adapter over previously recorded states/rays."""

    def __init__(self, entries: Iterable[tuple[CertificateState, tuple[LidarRay, ...]]]) -> None:
        self.entries = list(entries)
        self.index = 0

    def next(self) -> tuple[CertificateState, tuple[LidarRay, ...]]:
        if self.index >= len(self.entries):
            raise StopIteration("replay log exhausted")
        value = self.entries[self.index]
        self.index += 1
        return value
