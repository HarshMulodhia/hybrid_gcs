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
from .trajectory import Trajectory, BezierTrajectory
from .iris_decomposer import IRISDecomposer, Ellipsoid, SimpleBoxObstacle
from .micp_solver import MICPSolver, GCSGraph

__all__ = [
    'ConfigSpace',
    'Trajectory',
    'BezierTrajectory',
    'IRISDecomposer',
    'Ellipsoid',
    'SimpleBoxObstacle',
    'MICPSolver',
    'GCSGraph',
]

__version__ = '0.1.0'
