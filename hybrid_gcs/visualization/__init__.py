"""
Visualization Module for Hybrid-GCS.

Provides Foxglove Studio integration via MCAP recording and PyBullet
3-D simulation rendering.

Modules:
    - foxglove_recorder: MCAP scene recorder for Foxglove Studio
    - pybullet_renderer: PyBullet physics-based 3-D replay
"""

from .foxglove_recorder import FoxgloveRecorder, record_trajectory_scene
from .pybullet_renderer import PyBulletRenderer

__all__ = [
    "FoxgloveRecorder",
    "record_trajectory_scene",
    "PyBulletRenderer",
]
