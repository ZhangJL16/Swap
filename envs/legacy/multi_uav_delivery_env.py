"""Compatibility alias for the legacy multi-UAV delivery environment.

This environment retains empirical reset, recharge, collision-resolution, order,
and multi-agent semantics. It is not used by strict certificate experiments.
"""

from envs.UAVEnergyDelivery import UAVEnv, UAVEnvDiscreteWrapper, UAVParallelEnv, parallel_env

__all__ = ["UAVEnv", "UAVEnvDiscreteWrapper", "UAVParallelEnv", "parallel_env"]
