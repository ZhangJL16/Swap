"""Legacy empirical multi-UAV environments.

The original import paths remain authoritative for backwards compatibility.
This package only provides an explicit legacy namespace.
"""

from .multi_uav_delivery_env import UAVEnv, UAVEnvDiscreteWrapper, UAVParallelEnv, parallel_env

__all__ = ["UAVEnv", "UAVEnvDiscreteWrapper", "UAVParallelEnv", "parallel_env"]
