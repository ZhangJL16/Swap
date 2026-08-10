from __future__ import annotations

from dataclasses import dataclass
from math import inf, isfinite, nextafter


def round_down(value: float) -> float:
    if not isfinite(value):
        raise ValueError("interval arithmetic produced NaN or infinity")
    return nextafter(value, -inf)


def round_up(value: float) -> float:
    if not isfinite(value):
        raise ValueError("interval arithmetic produced NaN or infinity")
    return nextafter(value, inf)


@dataclass(frozen=True)
class Interval:
    """Closed scalar interval with outward-rounded elementary operations."""

    low: float
    high: float

    def __post_init__(self) -> None:
        if not isfinite(self.low) or not isfinite(self.high) or self.low > self.high:
            raise ValueError("invalid finite interval")

    @classmethod
    def point(cls, value: float) -> "Interval":
        if not isfinite(value):
            raise ValueError("interval point must be finite")
        return cls(float(value), float(value))

    @classmethod
    def radius(cls, center: float, radius: float) -> "Interval":
        if radius < 0.0:
            raise ValueError("radius must be nonnegative")
        return cls(round_down(center - radius), round_up(center + radius))

    def __add__(self, other: "Interval") -> "Interval":
        return Interval(round_down(self.low + other.low), round_up(self.high + other.high))

    def __sub__(self, other: "Interval") -> "Interval":
        return Interval(round_down(self.low - other.high), round_up(self.high - other.low))

    def __mul__(self, other: "Interval") -> "Interval":
        products = (
            self.low * other.low,
            self.low * other.high,
            self.high * other.low,
            self.high * other.high,
        )
        return Interval(round_down(min(products)), round_up(max(products)))

    def scale(self, scalar: float) -> "Interval":
        return self * Interval.point(scalar)

    def inflate(self, radius: float) -> "Interval":
        if radius < 0.0:
            raise ValueError("inflation radius must be nonnegative")
        return Interval(round_down(self.low - radius), round_up(self.high + radius))

    def clip(self, low: float, high: float) -> "Interval":
        if low > high:
            raise ValueError("invalid clipping range")
        clipped_low = max(self.low, low)
        clipped_high = min(self.high, high)
        if clipped_low > clipped_high:
            raise ValueError("interval does not intersect clipping range")
        return Interval(clipped_low, clipped_high)

    def saturate(self, low: float, high: float) -> "Interval":
        """Monotone image of this interval under scalar clipping."""

        if low > high:
            raise ValueError("invalid saturation range")
        return Interval(
            min(max(self.low, low), high),
            min(max(self.high, low), high),
        )

    def contains(self, value: float, tolerance: float = 0.0) -> bool:
        return self.low - tolerance <= value <= self.high + tolerance

    def contains_interval(self, other: "Interval", tolerance: float = 0.0) -> bool:
        return self.low - tolerance <= other.low and self.high + tolerance >= other.high

    @property
    def absolute_upper(self) -> float:
        return round_up(max(abs(self.low), abs(self.high)))

    @property
    def width(self) -> float:
        return round_up(self.high - self.low)


ZERO_INTERVAL = Interval.point(0.0)
