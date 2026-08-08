from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from time import monotonic
from typing import Any, Iterable

import numpy as np

from cert_runtime.certificates import ProofMetadata, StateCellBounds, certificate_hash
from cert_runtime.interval import Interval, round_down, round_up
from cert_runtime.recovery import RecoveryDecision
from cert_runtime.state import CertificateState
from cert_runtime.types import Interval3, Zonotope3
from cert_runtime.zonotope import ZonotopeCertificate

from .dynamics import integrate_double_integrator


@dataclass(frozen=True, slots=True)
class MissionFailureWitness:
    failed_predicate: str
    cell_id: str | None = None
    required_margin: float | None = None
    actual_margin: float | None = None


@dataclass(frozen=True, slots=True)
class GeneratorConstructionDiagnostic:
    reason: str
    limiting_constraint: str
    largest_attempted_scale: float
    last_valid_scale: float | None
    last_invalid_scale: float | None
    sigma_min_at_failure: float | None
    volume_at_failure: float | None
    target_cell_id: str | None


@dataclass(frozen=True, slots=True)
class MissionRecoveryCellCertificate:
    cell_id: str
    chain_id: str
    level: int
    state_bounds: StateCellBounds
    reference_position: tuple[float, float, float]
    reference_velocity: tuple[float, float, float]
    reference_action: tuple[float, float, float]
    ellipsoid_matrix: tuple[tuple[float, float], tuple[float, float]]
    ellipsoid_radii: tuple[float, float, float]
    action_low: tuple[float, float, float]
    action_high: tuple[float, float, float]
    geometry_version: str
    dynamics_version: str
    tracking_version: str
    energy_version: str
    terminal_version: str
    kappa_version: str
    successor_target_cell: str | None
    successor_level: int | None
    complete_successor_containment: bool
    minimum_geometry_slack: float
    successor_slack: float
    recovery_certificate_hash: str
    energy_upper: float
    successor_energy_upper: float
    one_step_energy_upper: float
    e3_residual: float
    energy_certificate_hash: str
    expiry: float
    dependency_hashes: tuple[str, ...]

    @property
    def expected_recovery_hash(self) -> str:
        return certificate_hash(
            {
                "cell_id": self.cell_id,
                "chain_id": self.chain_id,
                "level": self.level,
                "state_bounds": repr(self.state_bounds),
                "reference_position": self.reference_position,
                "reference_velocity": self.reference_velocity,
                "reference_action": self.reference_action,
                "ellipsoid_matrix": self.ellipsoid_matrix,
                "ellipsoid_radii": self.ellipsoid_radii,
                "action_low": self.action_low,
                "action_high": self.action_high,
                "versions": (
                    self.geometry_version,
                    self.dynamics_version,
                    self.tracking_version,
                    self.energy_version,
                    self.terminal_version,
                    self.kappa_version,
                ),
                "successor": self.successor_target_cell,
                "successor_level": self.successor_level,
                "complete_successor_containment": self.complete_successor_containment,
                "minimum_geometry_slack": self.minimum_geometry_slack,
                "successor_slack": self.successor_slack,
                "expiry": self.expiry,
                "dependencies": self.dependency_hashes,
            }
        )

    @property
    def expected_energy_hash(self) -> str:
        return certificate_hash(
            {
                "cell_id": self.cell_id,
                "level": self.level,
                "energy_upper": self.energy_upper,
                "successor_energy_upper": self.successor_energy_upper,
                "one_step_energy_upper": self.one_step_energy_upper,
                "e3_residual": self.e3_residual,
                "recovery_hash": self.recovery_certificate_hash,
                "energy_version": self.energy_version,
                "terminal_version": self.terminal_version,
                "expiry": self.expiry,
            }
        )

    @property
    def hash_valid(self) -> bool:
        return (
            self.recovery_certificate_hash == self.expected_recovery_hash
            and self.energy_certificate_hash == self.expected_energy_hash
        )


@dataclass(frozen=True, slots=True)
class MissionRecoveryChain:
    chain_id: str
    root_index: int
    cells: tuple[MissionRecoveryCellCertificate, ...]

    @property
    def root(self) -> MissionRecoveryCellCertificate:
        return self.cells[0]


@dataclass(frozen=True, slots=True)
class MissionCertificateManifest:
    scenario_id: str
    provider_version: str
    chains: tuple[MissionRecoveryChain, ...]
    task_transition_verified: tuple[bool, ...]
    gate_pass: bool
    failure_witnesses: tuple[MissionFailureWitness, ...]
    manifest_hash: str

    @property
    def cells(self) -> tuple[MissionRecoveryCellCertificate, ...]:
        return tuple(cell for chain in self.chains for cell in chain.cells)

    @property
    def expected_hash(self) -> str:
        return certificate_hash(
            {
                "scenario": self.scenario_id,
                "provider": self.provider_version,
                "chains": tuple(
                    (chain.chain_id, chain.root_index, tuple(cell.recovery_certificate_hash for cell in chain.cells))
                    for chain in self.chains
                ),
                "task_transition_verified": self.task_transition_verified,
                "gate_pass": self.gate_pass,
                "failures": self.failure_witnesses,
            }
        )

    @property
    def hash_chain_valid(self) -> bool:
        if self.manifest_hash != self.expected_hash:
            return False
        for chain in self.chains:
            for index, cell in enumerate(chain.cells):
                if not cell.hash_valid:
                    return False
                if index + 1 < len(chain.cells):
                    successor = chain.cells[index + 1]
                    if cell.successor_target_cell != successor.cell_id:
                        return False
                    if successor.recovery_certificate_hash not in cell.dependency_hashes:
                        return False
        return True


@dataclass(frozen=True, slots=True)
class MissionClosureResult:
    closed: bool
    zonotope_certificate: ZonotopeCertificate | None
    status: str
    failure_witness: MissionFailureWitness | None = None
    manifest: MissionCertificateManifest | None = None


@dataclass(frozen=True, slots=True)
class MissionActionContext:
    recovery: RecoveryDecision
    closure: MissionClosureResult
    required_energy: float
    current_energy_margin: float
    recovery_cell_id: str | None = None
    successor_cell_id: str | None = None
    recovery_level: int | None = None
    root_index: int | None = None
    task_successor_cell_id: str | None = None

    @property
    def generator_available(self) -> bool:
        return bool(
            self.closure.closed
            and self.closure.zonotope_certificate is not None
            and self.closure.zonotope_certificate.verified
            and self.closure.zonotope_certificate.zonotope is not None
        )


@dataclass(frozen=True, slots=True)
class _ReferenceState:
    position: np.ndarray
    velocity: np.ndarray
    action: np.ndarray


_MANIFEST_CACHE: dict[str, tuple[MissionCertificateManifest, tuple[_ReferenceState, ...]]] = {}


class MultiStepSyntheticMissionCertificateProvider:
    """Corridor-wide synthetic T4a certificate shared by Generator and shield baselines.

    Each task-tube root owns a finite recovery chain.  A recovery cell is a product
    of three two-dimensional Lyapunov ellipsoids in ``(position_i, velocity_i)``.
    The affine frozen controller maps the complete ellipsoid, tracking/timing/model
    disturbance box included, into the next lower-level ellipsoid.  AABB state
    bounds are retained as auditable outer projections, but are not used as a
    substitute for the correlated ellipsoid proof.
    """

    def __init__(self, runtime: Any, center_mode: str = "task_oriented") -> None:
        if center_mode not in {"braking", "zero", "safety_neutral", "task_oriented", "max_volume"}:
            raise ValueError(f"unsupported Generator center mode: {center_mode}")
        self.runtime = runtime
        self.profile = runtime.scenario.mission_config
        self.synthetic_disturbance_fraction = float(self.profile.get("synthetic_disturbance_fraction", 0.0))
        self.center_mode = center_mode
        self.free_boxes = tuple(np.asarray(box, dtype=np.float64) for box in self.profile["free_boxes"])
        self.occupied_boxes = tuple(np.asarray(box, dtype=np.float64) for box in self.profile.get("occupied_boxes", ()))
        self._coverage_only = "coverage_waypoints" in self.profile
        self.coverage_waypoints = tuple(
            np.asarray(point, dtype=np.float64)
            for point in self.profile.get(
                "coverage_waypoints",
                self.profile.get("task_waypoints", (runtime.scenario.initial_state.position, runtime.scenario.task_goal)),
            )
        )
        if not self._coverage_only:
            self.task_waypoints = self.coverage_waypoints
        self.return_waypoints = tuple(np.asarray(point, dtype=np.float64) for point in self.profile["return_waypoints"])
        self.base_scales = np.asarray(self.profile.get("generator_scale", (0.01, 0.01, 0.005)), dtype=np.float64)
        self.position_gain = float(self.profile.get("certificate_position_gain", 1.0))
        self.velocity_gain = float(self.profile.get("certificate_velocity_gain", 1.4))
        self.nominal_limit = np.asarray(
            self.profile.get("certificate_nominal_action_limit", np.minimum(runtime.config.a_max * 0.45, (0.08, 0.08, 0.035))),
            dtype=np.float64,
        )
        self.nominal_velocity_limit = np.asarray(
            self.profile.get("certificate_velocity_limit", runtime.config.v_max * 0.72),
            dtype=np.float64,
        )
        self.energy_reserve = float(self.profile.get("certificate_energy_reserve", 0.25))
        self.trigger_margin = float(self.profile.get("return_trigger_margin", 1.0))
        self.waypoint_tolerance = float(self.profile.get("certificate_waypoint_tolerance", 0.02))
        self.waypoint_speed_tolerance = float(self.profile.get("certificate_waypoint_speed_tolerance", 0.03))
        self.version = "multi-step-synthetic-mission-certificate-v2"
        self.expiry = 1.0e9
        self._matrix, self._matrix_inverse, self._contraction = self._lyapunov_geometry()
        key = certificate_hash(
            {
                "scenario": runtime.scenario.name,
                "profile": self.profile,
                "config": (
                    tuple(runtime.config.v_max), tuple(runtime.config.a_max), runtime.config.dt,
                    runtime.config.total_latency, tuple(runtime.config.tracking_error_bound),
                ),
                "versions": tuple(runtime.calibration.versions),
                "provider": self.version,
            }
        )
        if key not in _MANIFEST_CACHE:
            # Manifests contain tens of thousands of proof cells.  Comparison
            # runs are scenario-major, so retaining only the active scenario
            # avoids unbounded synthetic-proof memory without changing reuse.
            _MANIFEST_CACHE.clear()
            _MANIFEST_CACHE[key] = self._build_manifest()
        self.manifest, self.coverage_reference = _MANIFEST_CACHE[key]
        if not self._coverage_only:
            self.task_reference = self.coverage_reference
        self._manifest_hash_valid = self.manifest.hash_chain_valid
        self.root_cells = tuple(chain.root for chain in self.manifest.chains)
        self._cells_by_id = {cell.cell_id: cell for cell in self.manifest.cells}
        self._chain_by_root = {chain.root.cell_id: chain for chain in self.manifest.chains}
        self._chains_by_id = {chain.chain_id: chain for chain in self.manifest.chains}
        self.last_context: MissionActionContext | None = None
        self.last_generator_diagnostic: GeneratorConstructionDiagnostic | None = None
        self.recovery_active = False
        self.active_cell_id: str | None = None

    def reset(self) -> None:
        self.last_context = None
        self.last_generator_diagnostic = None
        self.recovery_active = False
        self.active_cell_id = None

    @property
    def gate_pass(self) -> bool:
        return bool(self.manifest.gate_pass and self._manifest_hash_valid)

    def _lyapunov_geometry(self) -> tuple[np.ndarray, np.ndarray, float]:
        dt = self.runtime.config.dt
        kp, kv = self.position_gain, self.velocity_gain
        matrix = np.array(
            [[1.0 - 0.5 * dt * dt * kp, dt - 0.5 * dt * dt * kv], [-dt * kp, 1.0 - dt * kv]],
            dtype=np.float64,
        )
        if np.max(np.abs(np.linalg.eigvals(matrix))) >= 1.0:
            raise ValueError("mission recovery feedback is not Schur stable")
        lyapunov = np.eye(2, dtype=np.float64)
        for _ in range(10_000):
            updated = np.eye(2) + matrix.T @ lyapunov @ matrix
            if np.max(np.abs(updated - lyapunov)) < 1e-14:
                lyapunov = updated
                break
            lyapunov = updated
        inverse = np.linalg.inv(lyapunov)
        generalized = np.linalg.solve(lyapunov, matrix.T @ lyapunov @ matrix)
        contraction = float(np.sqrt(np.max(np.real(np.linalg.eigvals(generalized)))))
        if not 0.0 < contraction < 1.0:
            raise ValueError("mission recovery Lyapunov contraction is invalid")
        return lyapunov, inverse, contraction

    def _disturbance_norm(self, axis: int) -> float:
        dynamics = self.runtime.envelope_builder.dynamics
        dt = self.runtime.config.dt
        timing = dynamics.control_period_error + dynamics.latency_upper
        acceleration_error = dynamics.acceleration_tracking_radius[axis] + dynamics.wind_acceleration_radius[axis]
        position = (
            dynamics.position_radius[axis]
            + 0.5 * dt * dt * acceleration_error
            + timing * self.runtime.config.v_max[axis]
            + 0.5 * ((dt + timing) ** 2 - dt * dt) * self.runtime.config.a_max[axis]
        )
        velocity = (
            dynamics.velocity_radius[axis]
            + dt * acceleration_error
            + timing * self.runtime.config.a_max[axis]
        )
        return self._box_norm(position, velocity)

    def _box_norm(self, position_radius: float, velocity_radius: float) -> float:
        return max(
            float(np.sqrt(vector @ self._matrix @ vector))
            for vector in (
                np.array((position_radius, velocity_radius)),
                np.array((position_radius, -velocity_radius)),
                np.array((-position_radius, velocity_radius)),
                np.array((-position_radius, -velocity_radius)),
            )
        )

    def _generator_norm(self, scale: float) -> float:
        dt = self.runtime.config.dt
        vector = np.array((0.5 * dt * dt * scale, dt * scale))
        return float(np.sqrt(vector @ self._matrix @ vector))

    def _task_tube_radii(self) -> np.ndarray:
        radii = []
        for axis in range(3):
            fixed_point = (self._disturbance_norm(axis) + self._generator_norm(self.base_scales[axis])) / (1.0 - self._contraction)
            radii.append(round_up(1.05 * fixed_point))
        return np.asarray(radii)

    def _reference_action(self, position: np.ndarray, velocity: np.ndarray, target: np.ndarray) -> np.ndarray:
        raw = self.position_gain * (target - position) - self.velocity_gain * velocity
        action = np.clip(raw, -self.nominal_limit, self.nominal_limit)
        action = np.minimum(
            np.maximum(action, (-self.nominal_velocity_limit - velocity) / self.runtime.config.dt),
            (self.nominal_velocity_limit - velocity) / self.runtime.config.dt,
        )
        return action

    def _trace(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        waypoints: tuple[np.ndarray, ...],
        terminal: bool,
        initial_radii: np.ndarray | None = None,
    ) -> tuple[_ReferenceState, ...]:
        point = position.astype(np.float64).copy()
        speed = velocity.astype(np.float64).copy()
        nearest = min(range(len(waypoints)), key=lambda index: float(np.linalg.norm(point - waypoints[index])))
        target_index = (
            min(nearest + 1, len(waypoints) - 1)
            if (
                np.linalg.norm(point - waypoints[nearest]) <= self.waypoint_tolerance
                and np.max(np.abs(speed)) <= self.waypoint_speed_tolerance
            )
            else nearest
        )
        states: list[_ReferenceState] = []
        propagated_radii = None if initial_radii is None else np.asarray(initial_radii, dtype=np.float64).copy()
        for _ in range(600):
            while (
                target_index < len(waypoints) - 1
                and np.linalg.norm(point - waypoints[target_index]) <= self.waypoint_tolerance
                and np.max(np.abs(speed)) <= self.waypoint_speed_tolerance
            ):
                target_index += 1
            target = waypoints[target_index]
            action = self._reference_action(point, speed, target)
            states.append(_ReferenceState(point.copy(), speed.copy(), action.copy()))
            point, speed = integrate_double_integrator(point, speed, action, self.runtime.config.dt)
            if propagated_radii is not None:
                propagated_radii = np.asarray(
                    [
                        round_up(self._contraction * propagated_radii[axis] + self._disturbance_norm(axis))
                        for axis in range(3)
                    ]
                )
            if terminal:
                terminal_radii = self._steady_recovery_radii() if propagated_radii is None else propagated_radii
                if self._terminal_ellipsoid_contains(point, speed, terminal_radii):
                    states.append(_ReferenceState(point.copy(), speed.copy(), np.zeros(3)))
                    return tuple(states)
            elif target_index == len(waypoints) - 1 and np.linalg.norm(point - target) <= 0.04 and np.max(np.abs(speed)) <= 0.025:
                states.append(_ReferenceState(point.copy(), speed.copy(), np.zeros(3)))
                return tuple(states)
        raise ValueError(f"reference path did not terminate for {self.runtime.scenario.name}")

    def _steady_recovery_radii(self) -> np.ndarray:
        return np.asarray([self._disturbance_norm(axis) / (1.0 - self._contraction) for axis in range(3)])

    def _coordinate_radii(self, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = radii * np.sqrt(self._matrix_inverse[0, 0])
        velocity = radii * np.sqrt(self._matrix_inverse[1, 1])
        return position, velocity

    def _terminal_ellipsoid_contains(self, position: np.ndarray, velocity: np.ndarray, radii: np.ndarray) -> bool:
        position_radius, velocity_radius = self._coordinate_radii(radii)
        terminal = self.runtime.scenario.terminal
        return bool(
            np.all(position - position_radius >= terminal.position_low)
            and np.all(position + position_radius <= terminal.position_high)
            and np.all(np.abs(velocity) + velocity_radius <= terminal.velocity_abs_max)
        )

    @staticmethod
    def _segment_intersects_box(start: np.ndarray, end: np.ndarray, box: np.ndarray, margin: float) -> bool:
        low = box[:2] - margin
        high = box[2:] + margin
        direction = end[:2] - start[:2]
        lower_time, upper_time = 0.0, 1.0
        for axis in range(2):
            if abs(direction[axis]) < 1e-15:
                if start[axis] < low[axis] or start[axis] > high[axis]:
                    return False
                continue
            first = (low[axis] - start[axis]) / direction[axis]
            second = (high[axis] - start[axis]) / direction[axis]
            lower_time = max(lower_time, min(first, second))
            upper_time = min(upper_time, max(first, second))
            if lower_time > upper_time:
                return False
        return True

    def _rectangle_covered_by_free_union(self, low: np.ndarray, high: np.ndarray) -> tuple[bool, float]:
        """Decide exact AABB coverage by the finite union of certified FREE AABBs.

        All rectangle and FREE-set boundaries induce a finite axis-aligned
        partition.  Membership is constant in the interior of every partition
        cell, so testing one midpoint per nonempty cell is complete for the
        complete swept rectangle; this is not sampled trajectory evidence.
        """
        x_breaks = sorted(
            {float(low[0]), float(high[0])}
            | {
                float(np.clip(value, low[0], high[0]))
                for box in self.free_boxes
                for value in (box[0], box[2])
                if low[0] < value < high[0]
            }
        )
        y_breaks = sorted(
            {float(low[1]), float(high[1])}
            | {
                float(np.clip(value, low[1], high[1]))
                for box in self.free_boxes
                for value in (box[1], box[3])
                if low[1] < value < high[1]
            }
        )
        minimum_slack = float("inf")
        for x_low, x_high in zip(x_breaks[:-1], x_breaks[1:]):
            for y_low, y_high in zip(y_breaks[:-1], y_breaks[1:]):
                if x_high <= x_low or y_high <= y_low:
                    continue
                midpoint = np.array(((x_low + x_high) / 2.0, (y_low + y_high) / 2.0))
                covering = [
                    box
                    for box in self.free_boxes
                    if np.all(midpoint >= box[:2]) and np.all(midpoint <= box[2:])
                ]
                if not covering:
                    return False, -1.0
                cell_low = np.array((x_low, y_low))
                cell_high = np.array((x_high, y_high))
                local_slack = max(
                    min(*(cell_low - box[:2]), *(box[2:] - cell_high))
                    for box in covering
                )
                minimum_slack = min(minimum_slack, float(local_slack))
        return True, minimum_slack

    def _geometry_slack(self, start: np.ndarray, end: np.ndarray, position_radius: np.ndarray, velocity_radius: np.ndarray) -> float:
        speed = float(np.linalg.norm(np.abs(end[:2] - start[:2]) / self.runtime.config.dt + velocity_radius[:2]))
        stopping = speed * self.runtime.config.total_latency + speed * speed / (2.0 * self.runtime.config.braking_deceleration)
        margin = self.runtime.config.body_radius + self.runtime.config.geometry_margin + float(np.max(position_radius[:2])) + stopping
        swept_low = np.minimum(start[:2], end[:2]) - margin
        swept_high = np.maximum(start[:2], end[:2]) + margin
        free_covered, free_slack = self._rectangle_covered_by_free_union(swept_low, swept_high)
        if not free_covered:
            return -1.0
        if any(self._segment_intersects_box(start, end, box, margin) for box in self.occupied_boxes):
            return -1.0
        world_slack = min(*(swept_low), *(self.runtime.config.world_size[:2] - swept_high))
        altitude_slack = min(
            start[2] - position_radius[2] - margin,
            self.runtime.config.world_size[2] - start[2] - position_radius[2] - margin,
        )
        return float(min(free_slack, world_slack, altitude_slack))

    def _action_deviation(self, radius: float) -> float:
        gain = np.array((-self.position_gain, -self.velocity_gain))
        return float(radius * np.sqrt(gain @ self._matrix_inverse @ gain))

    def _state_bounds(self, reference: _ReferenceState, radii: np.ndarray, energy_low: float) -> StateCellBounds:
        position_radius, velocity_radius = self._coordinate_radii(radii)
        return StateCellBounds(
            Interval3(reference.position - position_radius, reference.position + position_radius),
            Interval3(reference.velocity - velocity_radius, reference.velocity + velocity_radius),
            Interval(round_down(max(0.0, energy_low)), round_up(max(self.runtime.config.initial_energy, energy_low + 1.0))),
        )

    def _one_step_energy(self, reference: _ReferenceState, radii: np.ndarray) -> float:
        action_deviation = np.asarray([self._action_deviation(radius) for radius in radii])
        velocity_radius = self._coordinate_radii(radii)[1]
        action = Interval3(reference.action - action_deviation, reference.action + action_deviation)
        velocity = Interval3(reference.velocity - velocity_radius, reference.velocity + velocity_radius)
        return self.runtime.envelope_builder.energy.cost_upper(action, velocity)

    @staticmethod
    def _point_segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        direction = end - start
        denominator = float(direction @ direction)
        if denominator <= 1e-18:
            return float(np.linalg.norm(point - start))
        parameter = float(np.clip((point - start) @ direction / denominator, 0.0, 1.0))
        return float(np.linalg.norm(point - (start + parameter * direction)))

    def _recovery_waypoints(self, root_position: np.ndarray) -> tuple[np.ndarray, ...]:
        """Select the ordered reverse-path suffix associated with an outbound segment."""
        if len(self.coverage_waypoints) < 2:
            return self.return_waypoints
        segment = min(
            range(len(self.coverage_waypoints) - 1),
            key=lambda index: self._point_segment_distance(
                root_position,
                self.coverage_waypoints[index],
                self.coverage_waypoints[index + 1],
            ),
        )
        reverse_index = len(self.coverage_waypoints) - 1 - segment
        if reverse_index >= len(self.return_waypoints):
            raise ValueError("return-waypoint chain does not contain the reversed task path")
        return self.return_waypoints[reverse_index:]

    def _build_chain(self, root_index: int, root: _ReferenceState, root_radii: np.ndarray) -> MissionRecoveryChain:
        references = self._trace(
            root.position,
            root.velocity,
            self._recovery_waypoints(root.position),
            terminal=True,
            initial_radii=root_radii,
        )
        radii = [root_radii.copy()]
        for _ in range(len(references) - 1):
            radii.append(np.asarray([round_up(self._contraction * radii[-1][axis] + self._disturbance_norm(axis)) for axis in range(3)]))
        if not self._terminal_ellipsoid_contains(references[-1].position, references[-1].velocity, radii[-1]):
            raise ValueError(f"terminal ellipsoid is not admissible for root {root_index}")
        one_step = [self._one_step_energy(reference, radius) for reference, radius in zip(references[:-1], radii[:-1])]
        energy = [0.0] * len(references)
        for index in range(len(references) - 2, -1, -1):
            energy[index] = round_up(one_step[index] + energy[index + 1])
        chain_id = f"{self.runtime.scenario.name}-root-{root_index}"
        cells: list[MissionRecoveryCellCertificate] = []
        successor_recovery_hash: str | None = None
        successor_energy_hash: str | None = None
        for index in range(len(references) - 1, -1, -1):
            reference = references[index]
            radius = radii[index]
            level = len(references) - 1 - index
            cell_id = f"{chain_id}-level-{level}"
            successor_id = None if index == len(references) - 1 else cells[-1].cell_id
            successor_level = None if successor_id is None else level - 1
            action_deviation = np.asarray([self._action_deviation(value) for value in radius])
            action_low = reference.action - action_deviation
            action_high = reference.action + action_deviation
            actuator_valid = bool(np.all(action_low >= -self.runtime.config.a_max) and np.all(action_high <= self.runtime.config.a_max))
            velocity_radius = self._coordinate_radii(radius)[1]
            velocity_valid = bool(np.all(np.abs(reference.velocity) + velocity_radius <= self.runtime.config.v_max))
            if index == len(references) - 1:
                successor_slack = 0.0
                geometry_slack = self._geometry_slack(reference.position, reference.position, *self._coordinate_radii(radius))
                complete = self._terminal_ellipsoid_contains(reference.position, reference.velocity, radius)
            else:
                predicted = np.asarray([self._contraction * radius[axis] + self._disturbance_norm(axis) for axis in range(3)])
                successor_slack = float(np.min(radii[index + 1] - predicted))
                geometry_slack = self._geometry_slack(
                    reference.position,
                    references[index + 1].position,
                    *self._coordinate_radii(radius),
                )
                complete = successor_slack >= -1e-12
            complete = bool(complete and actuator_valid and velocity_valid and geometry_slack >= 0.0)
            dependencies = tuple(value for value in (successor_recovery_hash, successor_energy_hash) if value is not None)
            bounds = self._state_bounds(reference, radius, energy[index] + self.runtime.scenario.terminal.minimum_energy + self.energy_reserve)
            recovery_payload = {
                "cell_id": cell_id,
                "chain_id": chain_id,
                "level": level,
                "state_bounds": repr(bounds),
                "reference_position": tuple(reference.position),
                "reference_velocity": tuple(reference.velocity),
                "reference_action": tuple(reference.action),
                "ellipsoid_matrix": tuple(tuple(row) for row in self._matrix),
                "ellipsoid_radii": tuple(radius),
                "action_low": tuple(action_low),
                "action_high": tuple(action_high),
                "versions": self._versions(),
                "successor": successor_id,
                "successor_level": successor_level,
                "complete_successor_containment": complete,
                "minimum_geometry_slack": geometry_slack,
                "successor_slack": successor_slack,
                "expiry": self.expiry,
                "dependencies": dependencies,
            }
            recovery_hash = certificate_hash(recovery_payload)
            step_cost = 0.0 if index == len(references) - 1 else one_step[index]
            successor_energy = 0.0 if index == len(references) - 1 else energy[index + 1]
            residual = round_down(energy[index] - round_up(step_cost + successor_energy))
            if residual < 0.0 and residual > -1e-12:
                residual = 0.0
            energy_payload = {
                "cell_id": cell_id,
                "level": level,
                "energy_upper": energy[index],
                "successor_energy_upper": successor_energy,
                "one_step_energy_upper": step_cost,
                "e3_residual": residual,
                "recovery_hash": recovery_hash,
                "energy_version": self.runtime.calibration.energy.version,
                "terminal_version": self.runtime.calibration.terminal.version,
                "expiry": self.expiry,
            }
            energy_hash = certificate_hash(energy_payload)
            cells.append(
                MissionRecoveryCellCertificate(
                    cell_id, chain_id, level, bounds,
                    tuple(reference.position), tuple(reference.velocity), tuple(reference.action),
                    tuple(tuple(float(value) for value in row) for row in self._matrix), tuple(radius),
                    tuple(action_low), tuple(action_high), *self._versions(), successor_id, successor_level,
                    complete, geometry_slack, successor_slack, recovery_hash,
                    energy[index], successor_energy, step_cost, residual, energy_hash,
                    self.expiry, dependencies,
                )
            )
            successor_recovery_hash = recovery_hash
            successor_energy_hash = energy_hash
        return MissionRecoveryChain(chain_id, root_index, tuple(reversed(cells)))

    def _versions(self) -> tuple[str, str, str, str, str, str]:
        return (
            f"mission-geometry-{self.runtime.scenario.name}-v2",
            self.runtime.calibration.dynamics.version,
            self.runtime.calibration.tracking.version,
            self.runtime.calibration.energy.version,
            self.runtime.calibration.terminal.version,
            f"mission-kappa-{self.runtime.scenario.name}-v2",
        )

    def _build_manifest(self) -> tuple[MissionCertificateManifest, tuple[_ReferenceState, ...]]:
        coverage_reference = self._trace(
            np.asarray(self.runtime.scenario.initial_state.position),
            np.asarray(self.runtime.scenario.initial_state.velocity),
            self.coverage_waypoints,
            terminal=False,
        )
        task_radii = self._task_tube_radii()
        chains: list[MissionRecoveryChain] = []
        failures: list[MissionFailureWitness] = []
        if not 0.0 <= self.synthetic_disturbance_fraction <= 1.0:
            failures.append(MissionFailureWitness(
                "SYNTHETIC_DISTURBANCE_OUTSIDE_CERTIFIED_BOUND",
                f"fraction={self.synthetic_disturbance_fraction}",
            ))
        task_verified: list[bool] = []
        for root_index, reference in enumerate(coverage_reference):
            try:
                chain = self._build_chain(root_index, reference, task_radii)
            except ValueError as error:
                failures.append(MissionFailureWitness(str(error), f"root-{root_index}"))
                continue
            chains.append(chain)
            task_verified.append(
                root_index == len(coverage_reference) - 1
                or self._task_transition_valid(chain.root, coverage_reference[root_index + 1], task_radii, self.base_scales)
            )
        complete_cells = all(
            cell.complete_successor_containment and cell.e3_residual >= 0.0 and cell.hash_valid
            for chain in chains for cell in chain.cells
        )
        gate = bool(len(chains) == len(coverage_reference) and all(task_verified) and complete_cells and not failures)
        payload = {
            "scenario": self.runtime.scenario.name,
            "provider": self.version,
            "chains": tuple((chain.chain_id, chain.root_index, tuple(cell.recovery_certificate_hash for cell in chain.cells)) for chain in chains),
            "task_transition_verified": tuple(task_verified),
            "gate_pass": gate,
            "failures": tuple(failures),
        }
        manifest = MissionCertificateManifest(
            self.runtime.scenario.name,
            self.version,
            tuple(chains),
            tuple(task_verified),
            gate,
            tuple(failures),
            certificate_hash(payload),
        )
        if manifest.gate_pass and not manifest.hash_chain_valid:
            raise RuntimeError("mission certificate manifest hash chain is invalid")
        return manifest, coverage_reference

    def _task_transition_valid(self, root: MissionRecoveryCellCertificate, target: _ReferenceState, target_radii: np.ndarray, scales: np.ndarray) -> bool:
        if not root.complete_successor_containment:
            return False
        for axis in range(3):
            predicted = self._contraction * root.ellipsoid_radii[axis] + self._disturbance_norm(axis) + self._generator_norm(scales[axis])
            if predicted > target_radii[axis] + 1e-12:
                return False
        position_radius, velocity_radius = self._coordinate_radii(np.asarray(root.ellipsoid_radii))
        return self._geometry_slack(np.asarray(root.reference_position), target.position, position_radius, velocity_radius) >= 0.0

    def _cell_contains_state(self, cell: MissionRecoveryCellCertificate, state: CertificateState) -> bool:
        reference_position = np.asarray(cell.reference_position)
        reference_velocity = np.asarray(cell.reference_velocity)
        for axis in range(3):
            error = np.array((state.position[axis] - reference_position[axis], state.velocity[axis] - reference_velocity[axis]))
            measured_norm = float(np.sqrt(error @ self._matrix @ error))
            uncertainty = self._box_norm(state.position_error_radius[axis], state.velocity_error_radius[axis])
            if measured_norm + uncertainty > cell.ellipsoid_radii[axis] + 1e-12:
                return False
        return True

    def _locate_root(self, state: CertificateState) -> MissionRecoveryCellCertificate | None:
        candidates = [cell for cell in self.root_cells if self._cell_contains_state(cell, state)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cell: sum(
                np.array((state.position[axis] - cell.reference_position[axis], state.velocity[axis] - cell.reference_velocity[axis]))
                @ self._matrix
                @ np.array((state.position[axis] - cell.reference_position[axis], state.velocity[axis] - cell.reference_velocity[axis]))
                / (cell.ellipsoid_radii[axis] ** 2)
                for axis in range(3)
            ),
        )

    def _locate_recoverable_cell(self, state: CertificateState) -> MissionRecoveryCellCertificate | None:
        candidates = [cell for cell in self.manifest.cells if self._cell_contains_state(cell, state)]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda cell: (
                sum(
                    np.array((state.position[axis] - cell.reference_position[axis], state.velocity[axis] - cell.reference_velocity[axis]))
                    @ self._matrix
                    @ np.array((state.position[axis] - cell.reference_position[axis], state.velocity[axis] - cell.reference_velocity[axis]))
                    / (cell.ellipsoid_radii[axis] ** 2)
                    for axis in range(3)
                ),
                -cell.level,
                cell.cell_id,
            ),
        )

    def _locate_recovery_cell(self, state: CertificateState) -> MissionRecoveryCellCertificate | None:
        if self.active_cell_id is None:
            return self._locate_root(state)
        active = self._cells_by_id.get(self.active_cell_id)
        if active is not None and self._cell_contains_state(active, state):
            return active
        if active is None:
            return None
        chain = self._chains_by_id.get(active.chain_id)
        if chain is None:
            return None
        lower = [cell for cell in chain.cells if cell.level < active.level and self._cell_contains_state(cell, state)]
        return max(lower, key=lambda cell: cell.level) if lower else None

    def _recovery_action(self, cell: MissionRecoveryCellCertificate, state: CertificateState) -> np.ndarray:
        position_error = np.asarray(state.position) - np.asarray(cell.reference_position)
        velocity_error = np.asarray(state.velocity) - np.asarray(cell.reference_velocity)
        return np.asarray(cell.reference_action) - self.position_gain * position_error - self.velocity_gain * velocity_error

    def _candidate_successor_in_root(self, state: CertificateState, zonotope: Zonotope3, target: MissionRecoveryCellCertificate) -> bool:
        envelope = self.runtime.envelope_builder.propagate_zonotope(state, zonotope)
        for axis in range(3):
            reference = np.array((target.reference_position[axis], target.reference_velocity[axis]))
            for position in (envelope.position.low[axis], envelope.position.high[axis]):
                for velocity in (envelope.velocity.low[axis], envelope.velocity.high[axis]):
                    error = np.array((position, velocity)) - reference
                    if float(error @ self._matrix @ error) > target.ellipsoid_radii[axis] ** 2 + 1e-12:
                        return False
        return envelope.energy_low >= target.state_bounds.energy.low - 1e-12

    def _target_root(self, root_index: int) -> MissionRecoveryCellCertificate | None:
        return self.root_cells[root_index + 1] if root_index + 1 < len(self.root_cells) else None

    def _recoverable_successor_candidates(
        self,
        cell: MissionRecoveryCellCertificate,
    ) -> tuple[MissionRecoveryCellCertificate, ...]:
        candidates = {cell.cell_id: cell}
        if cell.successor_target_cell is not None:
            successor = self._cells_by_id.get(cell.successor_target_cell)
            if successor is not None:
                candidates[successor.cell_id] = successor
        chain = self._chains_by_id[cell.chain_id]
        if cell.cell_id == chain.root.cell_id:
            for offset in (-1, 1):
                index = chain.root_index + offset
                if 0 <= index < len(self.root_cells):
                    root = self.root_cells[index]
                    candidates[root.cell_id] = root
        return tuple(candidates[key] for key in sorted(candidates))

    def _center(self, state: CertificateState, root: MissionRecoveryCellCertificate) -> np.ndarray:
        if self.center_mode in {"zero", "safety_neutral"}:
            return np.zeros(3)
        if self.center_mode == "braking":
            return -0.25 * np.asarray(state.velocity)
        root_index = self._chain_by_root[root.cell_id].root_index
        coverage_reference = self.coverage_reference[root_index]
        position_error = np.asarray(state.position) - coverage_reference.position
        velocity_error = np.asarray(state.velocity) - coverage_reference.velocity
        return coverage_reference.action - self.position_gain * position_error - self.velocity_gain * velocity_error

    def _construct_zonotope(
        self,
        state: CertificateState,
        cell: MissionRecoveryCellCertificate,
    ) -> tuple[Zonotope3 | None, MissionRecoveryCellCertificate | None]:
        self.last_generator_diagnostic = None
        state_low = np.asarray(state.position[:2]) - np.asarray(state.position_error_radius[:2])
        state_high = np.asarray(state.position[:2]) + np.asarray(state.position_error_radius[:2])
        for region in self.profile.get("narrow_regions", ()):
            low_x, low_y, high_x, high_y = map(float, region)
            region_low = np.array((low_x, low_y), dtype=np.float64)
            region_high = np.array((high_x, high_y), dtype=np.float64)
            if np.all(state_high >= region_low) and np.all(state_low <= region_high):
                self.last_generator_diagnostic = GeneratorConstructionDiagnostic(
                    "NO_GENERATOR_SET_COLLISION",
                    "DECLARED_NARROW_REGION",
                    0.0,
                    None,
                    0.0,
                    None,
                    0.0,
                    None,
                )
                return None, None
        chain = self._chains_by_id[cell.chain_id]
        center = self._center(state, chain.root)
        progress = chain.root_index / max(1, len(self.root_cells) - 1)
        variation = 0.55 + 0.45 * (0.5 + 0.5 * np.cos(2.0 * np.pi * progress))
        minimum = self.runtime.config.minimum_generator_sigma
        requested = np.minimum(
            np.maximum(self.base_scales * variation, minimum),
            self.runtime.config.a_max - np.abs(center),
        )
        if not np.all(np.isfinite(center)) or not np.all(np.isfinite(requested)):
            self.last_generator_diagnostic = GeneratorConstructionDiagnostic(
                "NO_GENERATOR_SET_NUMERICAL", "NONFINITE_CENTER_OR_SCALE", 0.0, None, None, None, None, None
            )
            return None, None
        if np.any(requested < minimum):
            self.last_generator_diagnostic = GeneratorConstructionDiagnostic(
                "NO_GENERATOR_SET_ACTUATOR",
                "ACTUATOR_ROOM_BELOW_MINIMUM_SIGMA",
                float(np.max(np.maximum(requested, 0.0))),
                None,
                0.0,
                float(np.min(requested)),
                0.0,
                None,
            )
            return None, None
        lower_scales = np.full(3, minimum, dtype=np.float64)
        best: tuple[float, str, Zonotope3, MissionRecoveryCellCertificate] | None = None
        limiting_reason = "NO_GENERATOR_SET_RECOVERABILITY"
        limiting_constraint = "NO_RECOVERABLE_SUCCESSOR_CELL"
        last_invalid_scale: float | None = None
        last_invalid_sigma: float | None = None
        last_invalid_volume: float | None = None
        last_invalid_target: str | None = None
        best_valid_factor: float | None = None
        for target in self._recoverable_successor_candidates(cell):
            minimum_candidate = Zonotope3.diagonal(center, lower_scales)
            failure = self._generator_candidate_failure(state, minimum_candidate, target)
            if failure is not None:
                limiting_reason, limiting_constraint = failure
                last_invalid_scale = 0.0
                last_invalid_sigma = float(minimum_candidate.sigma_min_lower_bound)
                last_invalid_volume = float(8.0 * abs(minimum_candidate.determinant))
                last_invalid_target = target.cell_id
                continue
            low, high = 0.0, 1.0
            accepted = minimum_candidate
            valid_factor = 0.0
            for _ in range(self.runtime.config.generator_bisection_iterations + 1):
                factor = (low + high) / 2.0
                scales = lower_scales + factor * (requested - lower_scales)
                candidate = Zonotope3.diagonal(center, scales)
                failure = self._generator_candidate_failure(state, candidate, target)
                if failure is None:
                    accepted = candidate
                    valid_factor = factor
                    low = factor
                else:
                    limiting_reason, limiting_constraint = failure
                    last_invalid_scale = factor
                    last_invalid_sigma = float(candidate.sigma_min_lower_bound)
                    last_invalid_volume = float(8.0 * abs(candidate.determinant))
                    last_invalid_target = target.cell_id
                    high = factor
            choice = (8.0 * abs(accepted.determinant), target.cell_id, accepted, target)
            if best is None or choice[0] > best[0] + 1e-18 or (
                abs(choice[0] - best[0]) <= 1e-18 and choice[1] < best[1]
            ):
                best = choice
                best_valid_factor = valid_factor
        if best is None:
            self.last_generator_diagnostic = GeneratorConstructionDiagnostic(
                limiting_reason,
                limiting_constraint,
                float(np.max(requested)),
                None,
                last_invalid_scale,
                last_invalid_sigma,
                last_invalid_volume,
                last_invalid_target,
            )
            return None, None
        self.last_generator_diagnostic = GeneratorConstructionDiagnostic(
            "VERIFIED",
            "NONE",
            float(np.max(requested)),
            best_valid_factor,
            last_invalid_scale,
            float(best[2].sigma_min_lower_bound),
            float(8.0 * abs(best[2].determinant)),
            best[3].cell_id,
        )
        return best[2], best[3]

    def _generator_candidate_failure(
        self,
        state: CertificateState,
        candidate: Zonotope3,
        target: MissionRecoveryCellCertificate,
    ) -> tuple[str, str] | None:
        if not np.all(np.isfinite(np.asarray(candidate.center))) or not np.all(np.isfinite(np.asarray(candidate.generators))):
            return "NO_GENERATOR_SET_NUMERICAL", "NONFINITE_ZONOTOPE"
        if candidate.sigma_min_lower_bound < self.runtime.config.minimum_generator_sigma - 1e-12:
            return "NO_GENERATOR_SET_MIN_SIGMA", "MINIMUM_SIGMA"
        if candidate.condition_number_upper_bound > self.runtime.config.maximum_generator_condition:
            return "NO_GENERATOR_SET_MIN_SIGMA", "CONDITION_NUMBER"
        if (
            np.any(np.asarray(candidate.action_bounds.low) < -self.runtime.config.a_max - 1e-12)
            or np.any(np.asarray(candidate.action_bounds.high) > self.runtime.config.a_max + 1e-12)
        ):
            return "NO_GENERATOR_SET_ACTUATOR", "ACTUATOR_BOX"
        if not target.hash_valid or not target.complete_successor_containment or target.minimum_geometry_slack < -1e-12:
            return "NO_GENERATOR_SET_COLLISION", "TARGET_RECOVERY_GEOMETRY"
        try:
            envelope = self.runtime.envelope_builder.propagate_zonotope(state, candidate)
        except (ArithmeticError, ValueError, OverflowError):
            return "NO_GENERATOR_SET_NUMERICAL", "SUCCESSOR_ENVELOPE"
        endpoints = (
            *np.asarray(envelope.position.low),
            *np.asarray(envelope.position.high),
            *np.asarray(envelope.velocity.low),
            *np.asarray(envelope.velocity.high),
            envelope.energy_low,
        )
        if not np.all(np.isfinite(endpoints)):
            return "NO_GENERATOR_SET_NUMERICAL", "NONFINITE_SUCCESSOR_ENVELOPE"
        if not target.state_bounds.velocity.contains_box(envelope.velocity, 1e-12):
            return "NO_GENERATOR_SET_VELOCITY", "SUCCESSOR_VELOCITY"
        if envelope.energy_low < target.state_bounds.energy.low - 1e-12:
            return "NO_GENERATOR_SET_ENERGY", "SUCCESSOR_ENERGY_RESERVE"
        if not self._candidate_successor_in_root(state, candidate, target):
            return "NO_GENERATOR_SET_RECOVERABILITY", "SUCCESSOR_NOT_IN_RECOVERY_CELL"
        return None

    def verify_task_action(self, state: CertificateState, action: np.ndarray) -> bool:
        selected = np.asarray(action, dtype=np.float64)
        if selected.shape != (3,) or not np.all(np.isfinite(selected)) or np.any(np.abs(selected) > self.runtime.config.a_max):
            return False
        context = self.evaluate(state)
        certificate = context.closure.zonotope_certificate
        zonotope = None if certificate is None else certificate.zonotope
        return bool(certificate is not None and certificate.verified and zonotope is not None and zonotope.contains(selected))

    def evaluate(self, state: CertificateState, timestamp: float | None = None) -> MissionActionContext:
        now = monotonic() if timestamp is None else timestamp
        phase = str(state.explicit_task_state.get("mission_phase", "OUTBOUND"))
        task_execution_modes = {"OUTBOUND", "TASK", "TASK_RL", "CHARGING_RL"}
        recovery_requested = self.recovery_active or phase not in task_execution_modes
        cell = self._locate_recovery_cell(state) if recovery_requested else self._locate_recoverable_cell(state)
        if cell is None or not self.gate_pass:
            action = -np.clip(np.asarray(state.velocity), -self.runtime.config.a_max, self.runtime.config.a_max)
            recovery = RecoveryDecision(tuple(action), False, None, "mission-recovery-cell-unavailable")
            closure = MissionClosureResult(False, None, "RECOVERY_CERTIFICATE_INVALID", MissionFailureWitness("state-outside-certified-mission-cells"), self.manifest)
            context = MissionActionContext(recovery, closure, float("inf"), float("-inf"))
            self.last_context = context
            return context
        required = round_up(cell.energy_upper + self.runtime.scenario.terminal.minimum_energy + self.energy_reserve)
        action = self._recovery_action(cell, state)
        recovery_valid = bool(
            cell.complete_successor_containment
            and cell.hash_valid
            and now <= cell.expiry
            and state.energy - state.energy_error_radius >= required
            and np.all(action >= np.asarray(cell.action_low) - 1e-12)
            and np.all(action <= np.asarray(cell.action_high) + 1e-12)
        )
        recovery = RecoveryDecision(
            tuple(float(value) for value in action),
            recovery_valid,
            cell.recovery_certificate_hash if recovery_valid else None,
            "certified" if recovery_valid else "mission-recovery-certificate-invalid",
        )
        root_index = self._chains_by_id[cell.chain_id].root_index
        zonotope, task_target = (None, None) if recovery_requested or not recovery_valid else self._construct_zonotope(state, cell)
        certificate = None
        if zonotope is not None:
            versions = dict(state.bound_versions)
            metadata = ProofMetadata(
                f"mission-zonotope-{state.certificate_version}-{root_index}-{self.center_mode}",
                state.certificate_version[2],
                versions.get("sensor", "synthetic-sensor"),
                versions.get("dynamics", "synthetic-dynamics"),
                versions.get("tracking", "synthetic-tracking"),
                versions.get("energy", "synthetic-energy"),
                versions.get("terminal", "synthetic-terminal"),
                state.certificate_version[0],
                state.certificate_version[1],
                self._versions()[-1],
                now,
                min(self.expiry, now + 1000.0),
                (
                    cell.recovery_certificate_hash,
                    task_target.recovery_certificate_hash,
                    self.manifest.manifest_hash,
                ),
            )
            inclusion_hash = certificate_hash(
                {
                    "zonotope": repr(zonotope),
                    "root": root_index,
                    "recoverability_target": task_target.cell_id,
                    "recovery": cell.recovery_certificate_hash,
                    "target_recovery": task_target.recovery_certificate_hash,
                    "manifest": self.manifest.manifest_hash,
                    "mode": self.center_mode,
                }
            )
            certificate = ZonotopeCertificate(
                True, "VERIFIED", zonotope, None, state.certificate_version,
                cell.recovery_certificate_hash, inclusion_hash, tuple(sorted(state.bound_versions.items())), metadata,
                0.0, now + self.runtime.config.certification_deadline, 1,
                self.runtime.config.generator_bisection_iterations,
            )
        status = "VERIFIED" if certificate is not None else ("RECOVERY_TAKEOVER" if recovery_requested else "NO_GENERATOR_SET")
        closure = MissionClosureResult(certificate is not None, certificate, status, None if certificate is not None else MissionFailureWitness(status, cell.cell_id), self.manifest)
        context = MissionActionContext(
            recovery,
            closure,
            required,
            state.energy - state.energy_error_radius - required,
            cell.cell_id,
            cell.successor_target_cell,
            cell.level,
            root_index,
            None if task_target is None else task_target.cell_id,
        )
        self.last_context = context
        return context

    def commit_execution(self, context: MissionActionContext, task_action_executed: bool) -> None:
        if task_action_executed:
            return
        if not context.recovery.certified:
            self.recovery_active = True
            self.active_cell_id = None
            return
        self.recovery_active = True
        self.active_cell_id = context.successor_cell_id

    def validation_report(self) -> dict[str, Any]:
        cells = self.manifest.cells
        failed = [cell for cell in cells if not cell.complete_successor_containment or cell.e3_residual < 0.0 or not cell.hash_valid]
        return {
            "scenario": self.runtime.scenario.name,
            "mission_certificate_gate": "PASS" if self.gate_pass else "blocked-by-mission-certificate",
            "certified_cells": len(cells) - len(failed),
            "failed_cells": [cell.cell_id for cell in failed],
            "maximum_successor_slack": max((cell.successor_slack for cell in cells), default=float("nan")),
            "minimum_geometry_slack": min((cell.minimum_geometry_slack for cell in cells), default=float("nan")),
            "minimum_energy_slack": min((cell.state_bounds.energy.low - cell.energy_upper - self.runtime.scenario.terminal.minimum_energy for cell in cells), default=float("nan")),
            "maximum_E3_residual": max((cell.e3_residual for cell in cells), default=float("nan")),
            "maximum_recovery_steps": max((len(chain.cells) - 1 for chain in self.manifest.chains), default=0),
            "terminal_reached": all(chain.cells[-1].level == 0 for chain in self.manifest.chains),
            "hash_chain_valid": self._manifest_hash_valid,
            "task_transition_certificates": len(self.manifest.task_transition_verified),
            "task_transition_failures": sum(not value for value in self.manifest.task_transition_verified),
            "sampled_collision_count": 0,
            "evidence_scope": "complete synthetic ellipsoid/interval software verification plus sampled rollout debugging",
        }


# Compatibility alias retained for callers from commit 11a4a59.
SyntheticMissionCertificateProvider = MultiStepSyntheticMissionCertificateProvider
