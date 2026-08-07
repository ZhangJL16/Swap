"""Synthetic single-UAV environment for corridor-conditional certification experiments."""

from .actuator import ActionTrace, ActuatorTrackingModel
from .config import CertifiedUAVConfig, apply_configuration_overrides
from .dynamics import integrate_double_integrator
from .energy import EnergyModel
from .lidar import HorizontalLidarModel, LidarPacket
from .plant_env import CertifiedSingleUAVPlantEnv
from .runtime_wrapper import CertifiedRuntimeWrapper
from .scenario import FixedCertificationScenario, RandomTrainingScenario, ScenarioDefinition, load_scenario
from .state import UAVPhysicalState
from .task_wrapper import CertifiedTaskWrapper, MissionPhase, RewardBreakdown, TaskRewardConfig
from .telemetry import StepTelemetry
from .terminal import TerminalSpec

__all__ = [
    "ActionTrace",
    "ActuatorTrackingModel",
    "CertifiedRuntimeWrapper",
    "CertifiedSingleUAVPlantEnv",
    "CertifiedTaskWrapper",
    "MissionPhase",
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
) -> CertifiedRuntimeWrapper:
    scenario = FixedCertificationScenario(scenario_name).definition
    base = CertifiedUAVConfig(world_size=scenario.world_size) if config is None else config
    configured = apply_configuration_overrides(base, scenario.configuration_overrides)
    plant = CertifiedSingleUAVPlantEnv(configured, scenario)
    return CertifiedRuntimeWrapper(
        CertifiedTaskWrapper(plant),
        freeze_certificate_epoch=freeze_certificate_epoch,
    )
