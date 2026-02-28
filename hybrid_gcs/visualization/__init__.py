"""
Visualization Module for Hybrid-GCS.

Provides Foxglove Studio integration via MCAP recording. Writes
robot scenes, trajectories, and planning data to MCAP files that
can be opened directly in Foxglove Studio.

Modules:
    - foxglove_recorder: MCAP scene recorder for Foxglove
"""

from .foxglove_recorder import FoxgloveRecorder, record_trajectory_scene

__all__ = [
    "FoxgloveRecorder",
    "record_trajectory_scene",
]
