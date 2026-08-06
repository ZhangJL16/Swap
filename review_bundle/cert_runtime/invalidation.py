from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .corridor import ReturnCorridor


DEPENDENCY_CLOSURE = {
    "sensor": ("geometry", "corridor", "recovery", "recovery-energy", "zonotope"),
    "dynamics": ("successor", "recovery", "recovery-energy", "zonotope"),
    "tracking": ("successor", "recovery", "recovery-energy", "zonotope"),
    "energy": ("recovery-energy", "zonotope"),
    "terminal": ("recovery", "recovery-energy", "zonotope"),
    "kappa": ("recovery", "recovery-energy", "zonotope"),
    "geometry": ("corridor", "recovery", "recovery-energy", "zonotope"),
    "corridor": ("recovery", "recovery-energy", "zonotope"),
}


@dataclass(frozen=True)
class InvalidationPlan:
    changed_versions: tuple[tuple[str, str, str], ...]
    invalidated_objects: tuple[str, ...]
    reason: str


def dependency_invalidation_plan(
    previous: Mapping[str, str],
    current: Mapping[str, str],
) -> InvalidationPlan:
    changed = tuple(
        (name, previous.get(name, "<missing>"), current.get(name, "<missing>"))
        for name in sorted(set(previous) | set(current))
        if previous.get(name) != current.get(name)
    )
    invalidated = sorted({item for name, _, _ in changed for item in DEPENDENCY_CLOSURE.get(name, ())})
    return InvalidationPlan(changed, tuple(invalidated), "version-change" if changed else "unchanged")


def apply_corridor_invalidation(corridor: ReturnCorridor, plan: InvalidationPlan) -> None:
    if any(name in plan.invalidated_objects for name in ("recovery", "recovery-energy", "zonotope")):
        corridor.invalidate_certificates()
