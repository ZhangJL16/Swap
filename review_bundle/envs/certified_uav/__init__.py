"""Synthetic single-UAV environment for corridor-conditional certification experiments."""

import numpy as np

from .actuator import ActionTrace, ActuatorTrackingModel
from .config import CertifiedUAVConfig, apply_configuration_overrides
from .dynamics import integrate_double_integrator
from .energy import EnergyModel, SimulationEnergyConfig
from .lidar import HorizontalLidarModel, LidarPacket
from .plant_env import CertifiedSingleUAVPlantEnv
from .runtime_wrapper import CertifiedRuntimeWrapper
from .scenario import FixedCertificationScenario, RandomTrainingScenario, ScenarioDefinition, load_scenario
from .state import UAVPhysicalState
from .task_wrapper import CertifiedTaskWrapper, MissionPhase, MissionTerminationReason, RewardBreakdown, TaskRewardConfig
from .telemetry import StepTelemetry
from .terminal import TerminalSpec

__all__ = [
    "ActionTrace",
    "ActuatorTrackingModel",
    "CertifiedRuntimeWrapper",
    "CertifiedSingleUAVPlantEnv",
    "CertifiedTaskWrapper",
    "MissionPhase",
    "MissionTerminationReason",
    "RewardBreakdown",
    "TaskRewardConfig",
    "CertifiedUAVConfig",
    "FixedCertificationScenario",
    "EnergyModel",
    "HorizontalLidarModel",
    "LidarPacket",
    "RandomTrainingScenario",
    "ScenarioDefinition",
    "StepTelemetry",
    "TerminalSpec",
    "UAVPhysicalState",
    "integrate_double_integrator",
    "load_scenario",
    "make_certified_uav_env",
]


def make_certified_uav_env(
    scenario_name: str = "open_corridor.json",
    config: CertifiedUAVConfig | None = None,
    *,
    freeze_certificate_epoch: bool = False,
    generator_center_mode: str = "task_oriented",
    timing_mode: str = "wall_clock",
) -> CertifiedRuntimeWrapper:
    scenario = FixedCertificationScenario(scenario_name).definition
    base = CertifiedUAVConfig(world_size=scenario.world_size) if config is None else config
    configured = apply_configuration_overrides(base, scenario.configuration_overrides)
    disturbance_fraction = float(scenario.mission_config.get("synthetic_disturbance_fraction", 0.0))
    bounded_fraction = min(max(disturbance_fraction, 0.0), 1.0)
    scenario_seed = int(scenario.mission_config.get("scenario_seed", 0))
    signs = np.where((np.arange(3) + scenario_seed) % 2 == 0, 1.0, -1.0)
    actuator = ActuatorTrackingModel(
        configured.tracking_error_bound,
        deterministic_bias=bounded_fraction * configured.tracking_error_bound * signs,
    )
    lidar = HorizontalLidarModel(
        configured.num_lasers,
        configured.lidar_range,
        "synthetic-sensor-v1",
        range_noise=0.004 * bounded_fraction,
        pose_noise=0.002 * bounded_fraction,
        heading_noise=0.002 * bounded_fraction,
        invalid_probability=configured.lidar_invalid_probability,
    )
    energy = EnergyModel(SimulationEnergyConfig(simulation_error=0.0005 * bounded_fraction))
    plant = CertifiedSingleUAVPlantEnv(
        configured,
        scenario,
        actuator_model=actuator,
        energy_model=energy,
        lidar_model=lidar,
    )
    return CertifiedRuntimeWrapper(
        CertifiedTaskWrapper(plant),
        freeze_certificate_epoch=freeze_certificate_epoch,
        generator_center_mode=generator_center_mode,
        timing_mode=timing_mode,
    )
