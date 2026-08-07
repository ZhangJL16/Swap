"""Deprecated compatibility names for the energy-management API."""

from .energy_management import (
    EnergyDecision,
    EnergyManagementPolicy,
    EnergyManagementReplayBuffer,
    EnergyManagementSAC,
    EnergyManagementSACConfig,
    EnergyManagementTransition,
    FixedThresholdEnergyPolicy,
    FullChargeEnergyPolicy,
    ReserveOnlyEnergyPolicy,
    make_energy_management_policy,
)

SchedulerBinaryDecision = EnergyDecision
ChargingScheduler = EnergyManagementPolicy
SchedulerTransition = EnergyManagementTransition
SchedulerReplayBuffer = EnergyManagementReplayBuffer
ReserveOnlyScheduler = ReserveOnlyEnergyPolicy
FixedThresholdScheduler = FixedThresholdEnergyPolicy
FullChargeScheduler = FullChargeEnergyPolicy
ChargingSchedulerSACConfig = EnergyManagementSACConfig
ChargingSchedulerSAC = EnergyManagementSAC
make_scheduler = make_energy_management_policy

__all__ = [
    "SchedulerBinaryDecision",
    "ChargingScheduler",
    "SchedulerTransition",
    "SchedulerReplayBuffer",
    "ReserveOnlyScheduler",
    "FixedThresholdScheduler",
    "FullChargeScheduler",
    "ChargingSchedulerSACConfig",
    "ChargingSchedulerSAC",
    "make_scheduler",
]
