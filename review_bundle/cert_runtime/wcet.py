from __future__ import annotations

from dataclasses import dataclass
from os import sched_getaffinity, sched_setaffinity
from platform import machine, platform, python_implementation, python_version, processor, system
from statistics import median
from time import perf_counter
from typing import Callable, Mapping, Sequence

from .contracts import WCETContract


@dataclass(frozen=True)
class StageTiming:
    stage: str
    input_size: int
    sample_count: int
    median_seconds: float
    p99_seconds: float
    maximum_seconds: float


@dataclass(frozen=True)
class WCETEvidenceReport:
    platform: str
    operating_system: str
    cpu: str
    interpreter: str
    thread_policy: str
    affinity: tuple[int, ...] | None
    sample_count: int
    timings: tuple[StageTiming, ...]
    per_stage_maxima: tuple[tuple[str, float], ...]
    total_maximum_seconds: float
    deadline_seconds: float
    margin_seconds: float
    deployment_qualified: bool
    status: str


class WCETBenchmarkHarness:
    """Repeatable desktop profiler; never self-certifies hard WCET."""

    def __init__(self, contract: WCETContract, hard_realtime_platform: bool = False) -> None:
        self.contract = contract
        self.hard_realtime_platform = hard_realtime_platform

    def run(
        self,
        stages: Mapping[str, Callable[[int], object]],
        input_sizes: Sequence[int],
        *,
        warmup_runs: int,
        measured_runs: int,
        cpu_affinity: Sequence[int] | None = None,
    ) -> WCETEvidenceReport:
        if warmup_runs < 0 or measured_runs <= 0 or not stages or not input_sizes:
            raise ValueError("invalid WCET benchmark configuration")
        previous_affinity = None
        applied_affinity = None
        if cpu_affinity is not None:
            try:
                previous_affinity = sched_getaffinity(0)
                sched_setaffinity(0, set(cpu_affinity))
                applied_affinity = tuple(sorted(sched_getaffinity(0)))
            except (AttributeError, OSError):
                applied_affinity = None
        timings: list[StageTiming] = []
        try:
            for stage_name, stage in stages.items():
                for size in input_sizes:
                    for _ in range(warmup_runs):
                        stage(size)
                    samples = []
                    for _ in range(measured_runs):
                        started = perf_counter()
                        stage(size)
                        samples.append(perf_counter() - started)
                    ordered = sorted(samples)
                    p99_index = min(len(ordered) - 1, max(0, int(0.99 * len(ordered))))
                    timings.append(
                        StageTiming(
                            stage_name,
                            size,
                            len(samples),
                            median(samples),
                            ordered[p99_index],
                            max(samples),
                        )
                    )
        finally:
            if previous_affinity is not None:
                try:
                    sched_setaffinity(0, previous_affinity)
                except OSError:
                    pass
        maxima = {
            stage: max(item.maximum_seconds for item in timings if item.stage == stage)
            for stage in stages
        }
        total = sum(maxima.values())
        deadline = self.contract.control_period_seconds or 0.0
        margin = self.contract.margin_seconds or 0.0
        qualified = (
            self.hard_realtime_platform
            and self.contract.is_satisfied
            and total + margin < deadline
        )
        return WCETEvidenceReport(
            platform(),
            system(),
            processor() or machine(),
            f"{python_implementation()} {python_version()}",
            "Python threads; no RTOS priority guarantee",
            applied_affinity,
            measured_runs * len(input_sizes),
            tuple(timings),
            tuple(sorted(maxima.items())),
            total,
            deadline,
            margin,
            qualified,
            "implemented" if qualified else "blocked-by-deployment-evidence",
        )
