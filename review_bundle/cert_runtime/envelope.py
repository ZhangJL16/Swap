from __future__ import annotations

from dataclasses import dataclass

from .interval import Interval, round_down, round_up
from .state import CertificateState
from .types import Interval3, Vec3, Zonotope3


@dataclass(frozen=True)
class DynamicsBounds:
    control_period: float
    position_radius: Vec3
    velocity_radius: Vec3
    acceleration_tracking_radius: Vec3
    control_period_error: float = 0.0
    latency_upper: float = 0.0
    wind_acceleration_radius: Vec3 = (0.0, 0.0, 0.0)
    version: str = "unversioned-dynamics"
    calibration_complete: bool = False
    sensor_latency_upper: float = 0.0
    compute_latency_upper: float = 0.0
    switch_latency_upper: float = 0.0
    tracking_version: str = "unversioned-tracking"
    contract_hash: str = "unversioned-dynamics-contract"
    physical_status: str = "blocked-by-calibration"

    def __post_init__(self) -> None:
        radii = (
            self.position_radius
            + self.velocity_radius
            + self.acceleration_tracking_radius
            + self.wind_acceleration_radius
        )
        if (
            self.control_period <= 0.0
            or self.control_period_error < 0.0
            or self.latency_upper < 0.0
            or self.sensor_latency_upper < 0.0
            or self.compute_latency_upper < 0.0
            or self.switch_latency_upper < 0.0
            or self.control_period - self.control_period_error <= 0.0
            or any(value < 0.0 for value in radii)
        ):
            raise ValueError("invalid dynamics bounds")
        component_latency = (
            self.sensor_latency_upper
            + self.compute_latency_upper
            + self.switch_latency_upper
        )
        if component_latency > 0.0 and self.latency_upper < component_latency:
            raise ValueError("aggregate latency cannot exclude a component latency")
        if not self.version or not self.tracking_version or not self.contract_hash:
            raise ValueError("dynamics and tracking versions are required")

    @property
    def time_interval(self) -> Interval:
        return Interval(
            round_down(self.control_period - self.control_period_error),
            round_up(self.control_period + self.control_period_error + self.latency_upper),
        )


@dataclass(frozen=True)
class EnergyBounds:
    fixed_cost: float
    absolute_action_coefficients: Vec3
    uncertainty_cost: float
    additive_error_radius: float = 0.0
    version: str = "unversioned-energy"
    calibration_complete: bool = False
    velocity_coefficients: Vec3 = (0.0, 0.0, 0.0)
    contract_hash: str = "unversioned-energy-contract"
    physical_status: str = "blocked-by-calibration"

    def __post_init__(self) -> None:
        if self.fixed_cost < 0.0 or self.uncertainty_cost < 0.0 or self.additive_error_radius < 0.0:
            raise ValueError("energy costs must be nonnegative")
        if any(value < 0.0 for value in self.absolute_action_coefficients):
            raise ValueError("energy coefficients must be nonnegative")
        if any(value < 0.0 for value in self.velocity_coefficients):
            raise ValueError("velocity energy coefficients must be nonnegative")
        if not self.version or not self.contract_hash:
            raise ValueError("energy version and contract hash are required")

    def cost_upper(self, action: Interval3, velocity: Interval3 | None = None) -> float:
        upper = round_up(self.fixed_cost + self.uncertainty_cost)
        for coefficient, component in zip(self.absolute_action_coefficients, action.components):
            upper = round_up(upper + coefficient * component.absolute_upper)
        if velocity is not None:
            for coefficient, component in zip(self.velocity_coefficients, velocity.components):
                upper = round_up(upper + coefficient * component.absolute_upper)
        return round_up(upper + self.additive_error_radius)


@dataclass(frozen=True)
class SuccessorEnvelope:
    position: Interval3
    velocity: Interval3
    energy_low: float
    energy_high: float
    geometry_version_range: tuple[int, int]
    corridor_version_range: tuple[int, int]
    requires_update_revalidation: bool
    dynamics_bound_version: str
    energy_bound_version: str
    numerical_error_policy: str = "IEEE-754 math.nextafter outward rounding"


class SuccessorEnvelopeBuilder:
    """Affine interval propagation for the complete action zonotope."""

    def __init__(self, dynamics: DynamicsBounds, energy: EnergyBounds) -> None:
        self.dynamics = dynamics
        self.energy = energy

    def propagate_zonotope(self, state: CertificateState, action_set: Zonotope3) -> SuccessorEnvelope:
        return self.propagate_action_interval(state, action_set.action_bounds)

    def propagate_point_action(self, state: CertificateState, action: Vec3) -> SuccessorEnvelope:
        return self.propagate_action_interval(state, Interval3.point(action))

    def propagate_action_interval(
        self,
        state: CertificateState,
        action: Interval3,
    ) -> SuccessorEnvelope:
        position_initial = Interval3(
            tuple(
                round_down(state.position[index] - state.position_error_radius[index])
                for index in range(3)
            ),
            tuple(
                round_up(state.position[index] + state.position_error_radius[index])
                for index in range(3)
            ),
        )
        velocity_initial = Interval3(
            tuple(
                round_down(state.velocity[index] - state.velocity_error_radius[index])
                for index in range(3)
            ),
            tuple(
                round_up(state.velocity[index] + state.velocity_error_radius[index])
                for index in range(3)
            ),
        )
        energy_initial = Interval.radius(state.energy, state.energy_error_radius)
        return self.propagate_interval_state(
            position_initial,
            velocity_initial,
            energy_initial,
            action,
            state.local_geometry.version,
            state.return_corridor.version,
        )

    def propagate_interval_state(
        self,
        position_initial: Interval3,
        velocity_initial: Interval3,
        energy_initial: Interval,
        action: Interval3,
        geometry_version: int,
        corridor_version: int,
    ) -> SuccessorEnvelope:
        duration = self.dynamics.time_interval
        duration_squared = duration * duration
        position_components = []
        velocity_components = []
        acceleration_components = []
        for index in range(3):
            acceleration = action.components[index].inflate(
                self.dynamics.acceleration_tracking_radius[index]
                + self.dynamics.wind_acceleration_radius[index]
            )
            acceleration_components.append(acceleration)
            position_components.append(
                position_initial.components[index]
                + duration * velocity_initial.components[index]
                + duration_squared * acceleration * Interval.point(0.5)
                + Interval.radius(0.0, self.dynamics.position_radius[index])
            )
            velocity_components.append(
                velocity_initial.components[index]
                + duration * acceleration
                + Interval.radius(0.0, self.dynamics.velocity_radius[index])
            )
        acceleration_interval = Interval3.from_intervals(acceleration_components)
        energy_upper = self.energy.cost_upper(acceleration_interval, velocity_initial)
        energy_cost = Interval(0.0, energy_upper)
        energy_raw = energy_initial - energy_cost
        energy_successor = Interval(max(0.0, energy_raw.low), max(0.0, energy_raw.high))
        return SuccessorEnvelope(
            Interval3.from_intervals(position_components),
            Interval3.from_intervals(velocity_components),
            energy_successor.low,
            energy_successor.high,
            (geometry_version, geometry_version + 1),
            (corridor_version, corridor_version + 1),
            True,
            self.dynamics.version,
            self.energy.version,
        )
