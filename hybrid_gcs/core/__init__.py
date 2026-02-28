"""
Hybrid-GCS Core Module

Main GCS planning components including configuration spaces, trajectories,
IRIS decomposition, and MICP optimization.

Modules:
    - config_space: Configuration/action space definitions
    - trajectory: Trajectory representations (splines, Bezier)
    - iris_decomposer: IRIS region decomposition algorithm
    - micp_solver: Mixed-integer convex programming solver
    - collision_checker: Collision detection
    - kinematics: Robot kinematics (FK/IK)
"""

from .config_space import ConfigSpace
from .iris_decomposer import Ellipsoid, IRISDecomposer, SimpleBoxObstacle
from .micp_solver import GCSGraph, MICPSolver
from .trajectory import BezierTrajectory, Trajectory

__all__ = [
    "ConfigSpace",
    "Trajectory",
    "BezierTrajectory",
    "IRISDecomposer",
    "Ellipsoid",
    "SimpleBoxObstacle",
    "MICPSolver",
    "GCSGraph",
]

__version__ = "0.1.0"
