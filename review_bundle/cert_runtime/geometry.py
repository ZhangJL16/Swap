from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
from json import dumps
from math import atan2, floor, hypot, pi, sin
from typing import Iterable

from calibration.sensor import SensorCalibrationContract

from .contracts import RuntimeSensorBoundsContract
from .interval import Interval, round_down, round_up
from .types import AABB2


class CellState(IntEnum):
    UNKNOWN = 0
    FREE = 1
    OCCUPIED = 2


@dataclass(frozen=True)
class LidarRay:
    direction_x: float
    direction_y: float
    distance: float
    valid: bool
    hit: bool = True
    frame_id: str = "unidentified-frame"
    timestamp: float = 0.0

    def unit_direction(self) -> tuple[float, float]:
        norm = hypot(self.direction_x, self.direction_y)
        if norm <= 0.0:
            raise ValueError("LiDAR direction must be nonzero")
        return (self.direction_x / norm, self.direction_y / norm)


@dataclass(frozen=True)
class SensorBounds:
    position_error: float
    attitude_error_radians: float
    range_error: float
    beam_half_angle_radians: float
    time_sync_error: float
    footprint_radius: float
    map_discretization_error: float
    maximum_range: float
    maximum_speed: float
    evidence_max_age_seconds: float
    minimum_free_observations: int
    calibration_version: str

    @classmethod
    def from_contract(cls, contract: RuntimeSensorBoundsContract) -> "SensorBounds":
        contract.require_complete()
        return cls(
            position_error=float(contract.position_error),
            attitude_error_radians=float(contract.attitude_error_radians),
            range_error=float(contract.range_error),
            beam_half_angle_radians=float(contract.beam_half_angle_radians),
            time_sync_error=float(contract.time_sync_error),
            footprint_radius=float(contract.footprint_radius),
            map_discretization_error=float(contract.map_discretization_error),
            maximum_range=float(contract.maximum_range),
            maximum_speed=float(contract.maximum_speed),
            evidence_max_age_seconds=float(contract.evidence_max_age_seconds),
            minimum_free_observations=int(contract.minimum_free_observations),
            calibration_version=str(contract.calibration_version),
        )

    @classmethod
    def from_calibration_contract(
        cls,
        contract,
        *,
        allow_synthetic: bool = False,
    ) -> "SensorBounds":
        if not contract.validation_passed:
            raise ValueError("sensor contract failed independent validation")
        if contract.status != "implemented" and not allow_synthetic:
            raise ValueError("sensor contract is blocked-by-calibration")
        if contract.contract_hash != contract.expected_hash:
            raise ValueError("sensor calibration hash mismatch")
        return cls(
            contract.position_error,
            contract.attitude_error_radians,
            contract.range_error,
            contract.beam_half_angle_radians,
            contract.time_sync_error,
            contract.footprint_radius,
            contract.map_discretization_error,
            contract.maximum_range,
            contract.maximum_speed,
            contract.evidence_max_age_seconds,
            contract.minimum_free_observations,
            contract.calibration_version,
        )

    def __post_init__(self) -> None:
        numeric = (
            self.position_error,
            self.attitude_error_radians,
            self.range_error,
            self.beam_half_angle_radians,
            self.time_sync_error,
            self.footprint_radius,
            self.map_discretization_error,
            self.maximum_range,
            self.maximum_speed,
            self.evidence_max_age_seconds,
        )
        if any(value < 0.0 for value in numeric):
            raise ValueError("sensor bounds must be nonnegative")
        if self.maximum_range <= 0.0 or self.minimum_free_observations <= 0:
            raise ValueError("invalid range or evidence count")
        if self.attitude_error_radians + self.beam_half_angle_radians >= pi:
            raise ValueError("combined angular uncertainty must be below pi")
        if not self.calibration_version:
            raise ValueError("calibration version is required")

    @property
    def motion_margin(self) -> float:
        return round_up(self.time_sync_error * self.maximum_speed)


@dataclass(frozen=True)
class EvidenceProvenance:
    sensor_frame: str
    timestamp: float
    pose_interval: AABB2
    range_interval: Interval
    beam_angle_interval: Interval
    calibration_version: str
    certificate_version: int
    evidence_kind: str


@dataclass(frozen=True)
class GeometrySoundnessContract:
    calibration_version: str
    obstacle_cover_statement: str = "O_true intersect W subset O_grid union U_grid"
    free_exclusion_statement: str = "F_grid intersect (O_true union U_grid) is empty"
    status: str = "blocked-by-calibration"


class RollingLocalGeometry:
    """Finite world-aligned ternary grid with proof-carrying FREE cells."""

    def __init__(
        self,
        origin_x: float,
        origin_y: float,
        width: int,
        height: int,
        resolution: float,
    ) -> None:
        if width <= 0 or height <= 0 or resolution <= 0.0:
            raise ValueError("invalid grid dimensions")
        self.origin_x = float(origin_x)
        self.origin_y = float(origin_y)
        self.width = int(width)
        self.height = int(height)
        self.resolution = float(resolution)
        self.version = 0
        self.current_timestamp = 0.0
        self.active_calibration_version: str | None = None
        self._states = [[CellState.UNKNOWN for _ in range(width)] for _ in range(height)]
        self._evidence: list[list[tuple[EvidenceProvenance, ...]]] = [
            [tuple() for _ in range(width)] for _ in range(height)
        ]
        self._pending: list[list[tuple[EvidenceProvenance, ...]]] = [
            [tuple() for _ in range(width)] for _ in range(height)
        ]

    @property
    def bounds(self) -> AABB2:
        return AABB2(
            self.origin_x,
            self.origin_y,
            self.origin_x + self.width * self.resolution,
            self.origin_y + self.height * self.resolution,
        )

    def soundness_contract(self, calibration: SensorCalibrationContract) -> GeometrySoundnessContract:
        return GeometrySoundnessContract(
            calibration.calibration_version or "missing",
            status="implemented" if calibration.status == "implemented" else "blocked-by-calibration",
        )

    def certificate_digest(self) -> str:
        payload = {
            "origin": (self.origin_x, self.origin_y),
            "shape": (self.width, self.height, self.resolution),
            "version": self.version,
            "active_calibration_version": self.active_calibration_version,
            "states": [[int(value) for value in row] for row in self._states],
            "evidence": [
                [
                    [
                        (
                            item.sensor_frame,
                            item.timestamp,
                            item.calibration_version,
                            item.certificate_version,
                            item.evidence_kind,
                        )
                        for item in cell
                    ]
                    for cell in row
                ]
                for row in self._evidence
            ],
        }
        return sha256(dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def cell_box(self, row: int, column: int) -> AABB2:
        low_x = self.origin_x + column * self.resolution
        low_y = self.origin_y + row * self.resolution
        return AABB2(low_x, low_y, low_x + self.resolution, low_y + self.resolution)

    def state_at(self, row: int, column: int) -> CellState:
        return self._states[row][column]

    def evidence_at(self, row: int, column: int) -> tuple[EvidenceProvenance, ...]:
        return self._evidence[row][column]

    def recenter(self, center_x: float, center_y: float) -> None:
        """Shift by whole cells; retain evidence only for identical world cells."""

        target_origin_x = floor((center_x - self.width * self.resolution / 2.0) / self.resolution) * self.resolution
        target_origin_y = floor((center_y - self.height * self.resolution / 2.0) / self.resolution) * self.resolution
        shift_columns = round((target_origin_x - self.origin_x) / self.resolution)
        shift_rows = round((target_origin_y - self.origin_y) / self.resolution)
        if shift_columns == 0 and shift_rows == 0:
            return
        old_states = self._states
        old_evidence = self._evidence
        old_pending = self._pending
        self._states = [[CellState.UNKNOWN for _ in range(self.width)] for _ in range(self.height)]
        self._evidence = [[tuple() for _ in range(self.width)] for _ in range(self.height)]
        self._pending = [[tuple() for _ in range(self.width)] for _ in range(self.height)]
        for row in range(self.height):
            for column in range(self.width):
                old_row = row + shift_rows
                old_column = column + shift_columns
                if 0 <= old_row < self.height and 0 <= old_column < self.width:
                    self._states[row][column] = old_states[old_row][old_column]
                    self._evidence[row][column] = old_evidence[old_row][old_column]
                    self._pending[row][column] = old_pending[old_row][old_column]
        self.origin_x = target_origin_x
        self.origin_y = target_origin_y
        self.version += 1

    def update_lidar(
        self,
        pose_xy: tuple[float, float],
        rays: Iterable[LidarRay],
        bounds: SensorBounds,
        update_timestamp: float,
    ) -> None:
        """Apply conservative full-cell beam sectors and inflated hit sets."""

        self.current_timestamp = float(update_timestamp)
        if self.active_calibration_version not in (None, bounds.calibration_version):
            self._states = [[CellState.UNKNOWN for _ in range(self.width)] for _ in range(self.height)]
            self._evidence = [[tuple() for _ in range(self.width)] for _ in range(self.height)]
            self._pending = [[tuple() for _ in range(self.width)] for _ in range(self.height)]
            self.version += 1
        self.active_calibration_version = bounds.calibration_version
        self.expire_stale(update_timestamp, bounds.evidence_max_age_seconds)
        free_evidence: dict[tuple[int, int], list[EvidenceProvenance]] = {}
        occupied_evidence: dict[tuple[int, int], EvidenceProvenance] = {}
        next_version = self.version + 1
        for ray in rays:
            if not self._ray_can_certify(ray, bounds, update_timestamp):
                continue
            direction_x, direction_y = ray.unit_direction()
            range_interval = Interval(
                max(0.0, round_down(ray.distance - bounds.range_error)),
                round_up(ray.distance + bounds.range_error),
            )
            beam_interval = Interval(
                round_down(-bounds.beam_half_angle_radians - bounds.attitude_error_radians),
                round_up(bounds.beam_half_angle_radians + bounds.attitude_error_radians),
            )
            pose_radius = round_up(bounds.position_error + bounds.motion_margin)
            pose_interval = AABB2(
                round_down(pose_xy[0] - pose_radius),
                round_down(pose_xy[1] - pose_radius),
                round_up(pose_xy[0] + pose_radius),
                round_up(pose_xy[1] + pose_radius),
            )
            provenance = EvidenceProvenance(
                ray.frame_id,
                ray.timestamp,
                pose_interval,
                range_interval,
                beam_interval,
                bounds.calibration_version,
                next_version,
                "beam-free",
            )
            free_length = max(
                0.0,
                round_down(
                    range_interval.low
                    - bounds.motion_margin
                    - bounds.footprint_radius
                    - bounds.map_discretization_error
                ),
            )
            guaranteed_half_angle = max(
                0.0,
                round_down(bounds.beam_half_angle_radians - bounds.attitude_error_radians),
            )
            obstacle_radius = round_up(
                bounds.position_error
                + bounds.motion_margin
                + bounds.range_error
                + bounds.footprint_radius
                + bounds.map_discretization_error
                + 2.0
                * ray.distance
                * sin((bounds.attitude_error_radians + bounds.beam_half_angle_radians) / 2.0)
            )
            endpoint_x = pose_xy[0] + direction_x * ray.distance
            endpoint_y = pose_xy[1] + direction_y * ray.distance
            for row in range(self.height):
                for column in range(self.width):
                    expanded_cell = self.cell_box(row, column).expanded(
                        bounds.footprint_radius + bounds.map_discretization_error
                    )
                    corners = (
                        (expanded_cell.low_x, expanded_cell.low_y),
                        (expanded_cell.low_x, expanded_cell.high_y),
                        (expanded_cell.high_x, expanded_cell.low_y),
                        (expanded_cell.high_x, expanded_cell.high_y),
                    )
                    boundary_cell = (
                        row == 0
                        or row == self.height - 1
                        or column == 0
                        or column == self.width - 1
                    )
                    if not boundary_cell and guaranteed_half_angle > 0.0 and all(
                        self._corner_inside_guaranteed_sector(
                            corner,
                            pose_xy,
                            direction_x,
                            direction_y,
                            free_length,
                            guaranteed_half_angle,
                            pose_radius,
                        )
                        for corner in corners
                    ):
                        free_evidence.setdefault((row, column), []).append(provenance)
                    if self._box_intersects_disk(
                        self.cell_box(row, column),
                        endpoint_x,
                        endpoint_y,
                        obstacle_radius,
                    ):
                        occupied_evidence[(row, column)] = EvidenceProvenance(
                            ray.frame_id,
                            ray.timestamp,
                            pose_interval,
                            range_interval,
                            beam_interval,
                            bounds.calibration_version,
                            next_version,
                            "inflated-hit",
                        )
        for row in range(self.height):
            for column in range(self.width):
                key = (row, column)
                if key in occupied_evidence:
                    self._states[row][column] = CellState.OCCUPIED
                    self._evidence[row][column] = (occupied_evidence[key],)
                    self._pending[row][column] = tuple()
                    continue
                if self._states[row][column] == CellState.OCCUPIED:
                    self._pending[row][column] = tuple()
                    continue
                additions = free_evidence.get(key, [])
                pending = tuple(
                    evidence
                    for evidence in self._pending[row][column] + tuple(additions)
                    if update_timestamp - evidence.timestamp <= bounds.evidence_max_age_seconds
                )
                unique_observations = {
                    (evidence.sensor_frame, evidence.timestamp) for evidence in pending
                }
                self._pending[row][column] = pending
                if len(unique_observations) >= bounds.minimum_free_observations:
                    self._states[row][column] = CellState.FREE
                    self._evidence[row][column] = pending
        self.version = next_version

    def expire_stale(self, current_timestamp: float, maximum_age_seconds: float) -> None:
        changed = False
        for row in range(self.height):
            for column in range(self.width):
                evidence = tuple(
                    item
                    for item in self._evidence[row][column]
                    if current_timestamp - item.timestamp <= maximum_age_seconds
                )
                pending = tuple(
                    item
                    for item in self._pending[row][column]
                    if current_timestamp - item.timestamp <= maximum_age_seconds
                )
                self._evidence[row][column] = evidence
                self._pending[row][column] = pending
                if self._states[row][column] == CellState.FREE and not evidence:
                    self._states[row][column] = CellState.UNKNOWN
                    changed = True
        if changed:
            self.version += 1

    def mark_free_from_certificate(
        self,
        box: AABB2,
        certificate_id: str,
        timestamp: float = 0.0,
        calibration_version: str = "external-continuous-certificate",
    ) -> None:
        if not certificate_id or not calibration_version:
            raise ValueError("free-space promotion requires certificate and calibration ids")
        next_version = self.version + 1
        provenance = EvidenceProvenance(
            certificate_id,
            timestamp,
            box,
            Interval.point(0.0),
            Interval.point(0.0),
            calibration_version,
            next_version,
            "external-free",
        )
        for row, column in self._covered_indices(box):
            cell = self.cell_box(row, column)
            if box.contains_box(cell):
                self._states[row][column] = CellState.FREE
                self._evidence[row][column] = (provenance,)
                self._pending[row][column] = (provenance,)
        self.version = next_version
        self.active_calibration_version = calibration_version

    def mark_occupied_from_certificate(
        self,
        box: AABB2,
        certificate_id: str,
        timestamp: float = 0.0,
    ) -> None:
        if not certificate_id:
            raise ValueError("obstacle promotion requires a certificate id")
        next_version = self.version + 1
        provenance = EvidenceProvenance(
            certificate_id,
            timestamp,
            box,
            Interval.point(0.0),
            Interval.point(0.0),
            "external-continuous-certificate",
            next_version,
            "external-obstacle",
        )
        for row, column in self._covered_indices(box):
            self._states[row][column] = CellState.OCCUPIED
            self._evidence[row][column] = (provenance,)
            self._pending[row][column] = tuple()
        self.version = next_version

    def box_is_verified_free(self, box: AABB2, margin: float = 0.0) -> bool:
        query = box.expanded(margin)
        if not self.bounds.contains_box(query):
            return False
        indices = list(self._covered_indices(query))
        return bool(indices) and all(
            self._states[row][column] == CellState.FREE
            and bool(self._evidence[row][column])
            for row, column in indices
        )

    def _covered_indices(self, box: AABB2):
        low_column = floor((box.low_x - self.origin_x) / self.resolution)
        high_column = floor((box.high_x - self.origin_x) / self.resolution)
        low_row = floor((box.low_y - self.origin_y) / self.resolution)
        high_row = floor((box.high_y - self.origin_y) / self.resolution)
        for row in range(max(0, low_row), min(self.height - 1, high_row) + 1):
            for column in range(max(0, low_column), min(self.width - 1, high_column) + 1):
                yield row, column

    @staticmethod
    def _ray_can_certify(ray: LidarRay, bounds: SensorBounds, update_timestamp: float) -> bool:
        return (
            ray.valid
            and ray.hit
            and 0.0 <= ray.distance < bounds.maximum_range
            and abs(update_timestamp - ray.timestamp) <= bounds.time_sync_error
            and bool(ray.frame_id)
        )

    @staticmethod
    def _corner_inside_guaranteed_sector(
        corner: tuple[float, float],
        origin: tuple[float, float],
        direction_x: float,
        direction_y: float,
        free_length: float,
        guaranteed_half_angle: float,
        position_error: float,
    ) -> bool:
        relative_x = corner[0] - origin[0]
        relative_y = corner[1] - origin[1]
        longitudinal = relative_x * direction_x + relative_y * direction_y
        lateral = abs(relative_x * direction_y - relative_y * direction_x)
        radial_upper = hypot(relative_x, relative_y) + position_error
        angle_upper = atan2(lateral + position_error, max(0.0, longitudinal - position_error))
        return longitudinal > position_error and radial_upper < free_length and angle_upper < guaranteed_half_angle

    @staticmethod
    def _box_intersects_disk(box: AABB2, center_x: float, center_y: float, radius: float) -> bool:
        closest_x = min(max(center_x, box.low_x), box.high_x)
        closest_y = min(max(center_y, box.low_y), box.high_y)
        return (closest_x - center_x) ** 2 + (closest_y - center_y) ** 2 <= radius**2
