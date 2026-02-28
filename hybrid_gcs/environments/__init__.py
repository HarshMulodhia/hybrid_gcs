"""
Environments Module for Hybrid-GCS.

Task environments for the three application domains:
    - grasping_env: YCB object grasping (single & dual-arm)
    - drone_nav_env: Autonomous drone navigation (single & multi-agent)
    - manipulation_env: Complex manipulation (reach, pick, push, stack)

All environments follow a gym-like step/reset interface without requiring gym.
"""

from .base_env import BaseEnvironment, EnvConfig, StepResult
from .drone_nav_env import DroneNavConfig, DroneNavEnv
from .grasping_env import GraspingConfig, GraspingEnv
from .manipulation_env import ManipulationConfig, ManipulationEnv, ManipulationTask

__all__ = [
    "BaseEnvironment",
    "EnvConfig",
    "StepResult",
    "GraspingEnv",
    "GraspingConfig",
    "DroneNavEnv",
    "DroneNavConfig",
    "ManipulationEnv",
    "ManipulationConfig",
    "ManipulationTask",
]
