from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence

from .interval import Interval, round_down, round_up


Vec3 = tuple[float, float, float]
Matrix3 = tuple[Vec3, Vec3, Vec3]


def vec3(values: Iterable[float]) -> Vec3:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError("expected exactly three values")
    if not all(isfinite(value) for value in result):
        raise ValueError("vector entries must be finite")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class AABB2:
    """Closed world-frame axis-aligned horizontal box."""

    low_x: float
    low_y: float
    high_x: float
    high_y: float

    def __post_init__(self) -> None:
        if self.low_x > self.high_x or self.low_y > self.high_y:
            raise ValueError("invalid AABB bounds")

    def expanded(self, margin: float) -> "AABB2":
        if margin < 0:
            raise ValueError("margin must be nonnegative")
        return AABB2(
            round_down(self.low_x - margin),
            round_down(self.low_y - margin),
            round_up(self.high_x + margin),
            round_up(self.high_y + margin),
        )

    def contains_point(self, point: Sequence[float], tolerance: float = 0.0) -> bool:
        return (
            self.low_x - tolerance <= point[0] <= self.high_x + tolerance
            and self.low_y - tolerance <= point[1] <= self.high_y + tolerance
        )

    def contains_box(self, other: "AABB2", tolerance: float = 0.0) -> bool:
        return (
            self.low_x - tolerance <= other.low_x
            and self.low_y - tolerance <= other.low_y
            and self.high_x + tolerance >= other.high_x
            and self.high_y + tolerance >= other.high_y
        )

    def intersection(self, other: "AABB2") -> "AABB2 | None":
        low_x = max(self.low_x, other.low_x)
        low_y = max(self.low_y, other.low_y)
        high_x = min(self.high_x, other.high_x)
        high_y = min(self.high_y, other.high_y)
        if low_x > high_x or low_y > high_y:
            return None
        return AABB2(low_x, low_y, high_x, high_y)

    @property
    def width(self) -> float:
        return self.high_x - self.low_x

    @property
    def height(self) -> float:
        return self.high_y - self.low_y

    @property
    def center(self) -> tuple[float, float]:
        return ((self.low_x + self.high_x) / 2.0, (self.low_y + self.high_y) / 2.0)


@dataclass(frozen=True)
class Interval3:
    components: tuple[Interval, Interval, Interval]

    def __init__(self, low: Sequence[float], high: Sequence[float]) -> None:
        low3 = vec3(low)
        high3 = vec3(high)
        object.__setattr__(
            self,
            "components",
            tuple(Interval(low3[index], high3[index]) for index in range(3)),
        )

    @classmethod
    def from_intervals(cls, components: Sequence[Interval]) -> "Interval3":
        if len(components) != 3:
            raise ValueError("expected three intervals")
        return cls(
            tuple(component.low for component in components),
            tuple(component.high for component in components),
        )

    @property
    def low(self) -> Vec3:
        return tuple(component.low for component in self.components)  # type: ignore[return-value]

    @property
    def high(self) -> Vec3:
        return tuple(component.high for component in self.components)  # type: ignore[return-value]

    @classmethod
    def point(cls, value: Sequence[float]) -> "Interval3":
        point = vec3(value)
        return cls(point, point)

    def horizontal_box(self) -> AABB2:
        return AABB2(self.low[0], self.low[1], self.high[0], self.high[1])

    def max_abs(self) -> Vec3:
        return tuple(component.absolute_upper for component in self.components)  # type: ignore[return-value]

    def contains_point(self, value: Sequence[float], tolerance: float = 0.0) -> bool:
        value3 = vec3(value)
        return all(
            component.contains(value3[index], tolerance)
            for index, component in enumerate(self.components)
        )

    def contains_box(self, other: "Interval3", tolerance: float = 0.0) -> bool:
        return all(
            component.contains_interval(other.components[index], tolerance)
            for index, component in enumerate(self.components)
        )


def _det3(matrix: Matrix3) -> float:
    a, b, c = matrix
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _transpose_times_self(matrix: Matrix3) -> Matrix3:
    return tuple(
        tuple(sum(matrix[k][i] * matrix[k][j] for k in range(3)) for j in range(3))
        for i in range(3)
    )  # type: ignore[return-value]


@dataclass(frozen=True)
class Zonotope3:
    """Three-generator affine zonotope c + G[-1,1]^3."""

    center: Vec3
    generators: Matrix3

    def __post_init__(self) -> None:
        if len(self.generators) != 3 or any(len(row) != 3 for row in self.generators):
            raise ValueError("G must be 3 by 3")
        if not all(isfinite(value) for row in self.generators for value in row):
            raise ValueError("generator entries must be finite")

    @classmethod
    def diagonal(cls, center: Sequence[float], scales: Sequence[float]) -> "Zonotope3":
        center3 = vec3(center)
        scale3 = vec3(scales)
        return cls(
            center3,
            (
                (scale3[0], 0.0, 0.0),
                (0.0, scale3[1], 0.0),
                (0.0, 0.0, scale3[2]),
            ),
        )

    @property
    def determinant(self) -> float:
        return _det3(self.generators)

    @property
    def action_bounds(self) -> Interval3:
        radii = []
        for row in self.generators:
            radius = 0.0
            for value in row:
                radius = round_up(radius + abs(value))
            radii.append(radius)
        low = tuple(round_down(center - radius) for center, radius in zip(self.center, radii))
        high = tuple(round_up(center + radius) for center, radius in zip(self.center, radii))
        return Interval3(low, high)  # type: ignore[arg-type]

    @property
    def sigma_min_lower_bound(self) -> float:
        """Certified Gershgorin lower bound on sigma_min(G)."""

        gram = _transpose_times_self(self.generators)
        eigen_lower = min(
            gram[index][index]
            - sum(abs(gram[index][other]) for other in range(3) if other != index)
            for index in range(3)
        )
        return max(0.0, round_down(sqrt(max(0.0, eigen_lower))))

    @property
    def condition_number_upper_bound(self) -> float:
        sigma_lower = self.sigma_min_lower_bound
        if sigma_lower <= 0.0:
            return float("inf")
        if all(
            self.generators[row][column] == 0.0
            for row in range(3)
            for column in range(3)
            if row != column
        ):
            diagonal = [abs(self.generators[index][index]) for index in range(3)]
            return round_up(max(diagonal) / min(diagonal))
        frobenius = sqrt(sum(value * value for row in self.generators for value in row))
        return round_up(frobenius / sigma_lower)

    def map_eta(self, eta: Sequence[float]) -> Vec3:
        eta3 = vec3(eta)
        if any(abs(value) > 1.0 for value in eta3):
            raise ValueError("eta must belong to [-1,1]^3")
        return tuple(
            self.center[row]
            + sum(self.generators[row][column] * eta3[column] for column in range(3))
            for row in range(3)
        )  # type: ignore[return-value]

    def contains(self, action: Sequence[float], tolerance: float = 1e-9) -> bool:
        determinant = self.determinant
        if abs(determinant) <= tolerance:
            return False
        difference = tuple(float(action[i]) - self.center[i] for i in range(3))
        columns = tuple(tuple(self.generators[row][column] for row in range(3)) for column in range(3))
        eta = []
        for column in range(3):
            replaced_columns = list(columns)
            replaced_columns[column] = difference
            replaced_matrix = tuple(
                tuple(replaced_columns[col][row] for col in range(3)) for row in range(3)
            )
            eta.append(_det3(replaced_matrix) / determinant)  # type: ignore[arg-type]
        return all(abs(value) <= 1.0 + tolerance for value in eta)
