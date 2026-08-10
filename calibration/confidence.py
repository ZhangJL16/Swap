from __future__ import annotations

from dataclasses import dataclass
from math import ceil, isfinite
from typing import Iterable

from .schema import ConfidenceSemantics


@dataclass(frozen=True)
class BoundEstimate:
    value: float
    semantics: ConfidenceSemantics
    confidence_delta: float | None
    assumptions: tuple[str, ...]
    deterministic: bool


def _validated_residuals(residuals: Iterable[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in residuals)
    if not values or any(not isfinite(value) or value < 0.0 for value in values):
        raise ValueError("residuals must be a nonempty finite nonnegative sequence")
    return tuple(sorted(values))


def estimate_empirical_quantile(residuals: Iterable[float], quantile: float) -> BoundEstimate:
    values = _validated_residuals(residuals)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0,1]")
    index = min(len(values) - 1, max(0, ceil(quantile * len(values)) - 1))
    return BoundEstimate(
        values[index],
        ConfidenceSemantics.EMPIRICAL_QUANTILE,
        None,
        ("descriptive calibration-split quantile only",),
        False,
    )


def estimate_simultaneous_bound(
    residuals: Iterable[float],
    confidence_delta: float,
    family_size: int,
) -> BoundEstimate:
    values = _validated_residuals(residuals)
    if not 0.0 < confidence_delta < 1.0 or family_size <= 0:
        raise ValueError("invalid simultaneous-confidence parameters")
    adjusted_tail = confidence_delta / family_size
    rank = ceil((len(values) + 1) * (1.0 - adjusted_tail)) - 1
    value = values[min(len(values) - 1, max(0, rank))]
    return BoundEstimate(
        value,
        ConfidenceSemantics.SIMULTANEOUS_CONFIDENCE,
        confidence_delta,
        (
            "exchangeable calibration and deployment residuals",
            f"Bonferroni family size {family_size}",
            "coverage applies only to the declared family and operating domain",
        ),
        False,
    )


def estimate_deterministic_bound(
    observed_residuals: Iterable[float],
    engineering_bound: float,
    engineering_evidence_id: str,
) -> BoundEstimate:
    values = _validated_residuals(observed_residuals)
    if not isfinite(engineering_bound) or engineering_bound < max(values):
        raise ValueError("engineering bound must cover every retained residual")
    if not engineering_evidence_id:
        raise ValueError("deterministic bounds require engineering evidence")
    return BoundEstimate(
        engineering_bound,
        ConfidenceSemantics.DETERMINISTIC_ENGINEERING,
        None,
        (f"engineering evidence {engineering_evidence_id}",),
        True,
    )
