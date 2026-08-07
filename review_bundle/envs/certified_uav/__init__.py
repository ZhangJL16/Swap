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
from .charging import ChargingConfig, ChargingDynamics, DepartureGateResult, verify_departure_energy
from .persistent_task import (
    CertifiedServiceNetwork,
    PersistentMissionMode,
    PersistentTask,
    PersistentTaskManager,
    PersistentTaskStatus,
    PersistentTaskWrapper,
)
from .persistent_wrapper import PersistentRuntimeWrapper
from cert_runtime.charging_scheduler import make_scheduler

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
    "make_persistent_uav_env",
    "ChargingConfig",
    "ChargingDynamics",
    "DepartureGateResult",
    "verify_departure_energy",
    "CertifiedServiceNetwork",
    "PersistentMissionMode",
    "PersistentRuntimeWrapper",
    "PersistentTask",
    "PersistentTaskManager",
    "PersistentTaskStatus",
    "PersistentTaskWrapper",
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


def make_persistent_uav_env(
    scenario_name: str = "persistent_open.json",
    *,
    scheduler_name: str = "reserve_only",
    seed: int = 0,
    timing_mode: str = "functional",
    deterministic_scheduler: bool = False,
) -> PersistentRuntimeWrapper:
    scenario = FixedCertificationScenario(scenario_name).definition
    persistent = dict(scenario.mission_config.get("persistent", {}))
    if not persistent:
        raise ValueError(f"scenario {scenario.name} does not define a persistent service network")
    base = CertifiedUAVConfig(world_size=scenario.world_size)
    configured = apply_configuration_overrides(base, scenario.configuration_overrides)
    network = CertifiedServiceNetwork.from_config(persistent)
    actuator = ActuatorTrackingModel(configured.tracking_error_bound)
    lidar = HorizontalLidarModel(
        configured.num_lasers,
        configured.lidar_range,
        "synthetic-sensor-v1",
        configured.lidar_range_noise,
        configured.lidar_pose_noise,
        configured.lidar_heading_noise,
        configured.lidar_invalid_probability,
    )
    plant = CertifiedSingleUAVPlantEnv(
        configured,
        scenario,
        actuator_model=actuator,
        energy_model=EnergyModel(SimulationEnergyConfig()),
        lidar_model=lidar,
    )
    task = PersistentTaskWrapper(
        plant,
        network,
        goal_radius=float(persistent.get("goal_radius", 0.20)),
        task_reward=float(persistent.get("task_reward", 10.0)),
    )
    runtime = CertifiedRuntimeWrapper(task, generator_center_mode="task_oriented", timing_mode=timing_mode)
    charging = ChargingConfig(
        battery_capacity=float(persistent.get("battery_capacity", 30.0)),
        charging_rate=float(persistent.get("charging_rate", 2.0)),
        checkpoint_steps=int(persistent.get("charging_checkpoint_steps", 5)),
        departure_energy_margin=float(persistent.get("departure_energy_margin", 0.5)),
        forced_return_margin=float(persistent.get("forced_return_margin", 1.0)),
    )
    scheduler = make_scheduler(scheduler_name, observation_dim=15, seed=seed)
    return PersistentRuntimeWrapper(runtime, network, scheduler, charging, deterministic_scheduler)
