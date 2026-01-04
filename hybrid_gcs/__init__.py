"""
Main hybrid_gcs package.

A production-grade system combining Graph of Convex Sets (GCS) trajectory
planning with Deep Reinforcement Learning for autonomous robotics.

Modules:
    core: GCS core algorithms (ConfigSpace, Trajectory, IRIS, MICP)
    training: Deep RL training (PolicyNetwork, PPO, reward shaping)
    integration: Hybrid GCS+RL integration
    environments: Task environments (grasping, navigation, manipulation)
    evaluation: Performance metrics and analysis
    utils: Utilities and configuration
    cli: Command-line interfaces

References:
    - Marcucci et al. (2023): Motion Planning around Obstacles
    - Schulman et al. (2017): PPO Algorithms
    - Deits & Tedrake (2015): IRIS Decomposition
"""

__version__ = '0.1.0'
__author__ = 'Hybrid-GCS Contributors'

# Import core modules
try:
    from .core import (
        ConfigSpace,
        Trajectory,
        BezierTrajectory,
        IRISDecomposer,
        Ellipsoid,
        SimpleBoxObstacle,
        MICPSolver,
        GCSGraph,
    )
except ImportError as e:
    print(f"Warning: Could not import core module: {e}")

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
