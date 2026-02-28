"""
Main hybrid_gcs package.

A production-grade system combining Graph of Convex Sets (GCS) trajectory
planning with Deep Reinforcement Learning for autonomous robotics.

Modules:
    core: GCS core algorithms (ConfigSpace, Trajectory, IRIS, MICP)
    training: Deep RL training (PolicyNetwork, PPO, reward shaping)
    integration: Hybrid GCS+RL integration (blending, safety filter, features)
    environments: Task environments (grasping, navigation, manipulation)
    multi_agent: Multi-agent RL communication and Space-Time GCS

References:
    - Marcucci et al. (2023): Motion Planning around Obstacles
    - Schulman et al. (2017): PPO Algorithms
    - Deits & Tedrake (2015): IRIS Decomposition
"""

__version__ = "0.1.0"
__author__ = "Hybrid-GCS Contributors"

# Import core modules
try:
    from .core import (
        BezierTrajectory,
        ConfigSpace,
        Ellipsoid,
        GCSGraph,
        IRISDecomposer,
        MICPSolver,
        SimpleBoxObstacle,
        Trajectory,
    )
except ImportError as e:
    print(f"Warning: Could not import core module: {e}")

# Import integration modules
try:
    from .integration import (
        ConflictResolutionBlender,
        ControlBarrierFilter,
        DualPathwayExtractor,
        FeatureExtractorConfig,
        HierarchicalBlender,
        PriorityNetworkBlender,
        SafetyFilter,
        SafetyFilterConfig,
        WeightedBlender,
    )
except ImportError as e:
    print(f"Warning: Could not import integration module: {e}")

# Import environment modules
try:
    from .environments import (
        BaseEnvironment,
        DroneNavConfig,
        DroneNavEnv,
        GraspingConfig,
        GraspingEnv,
        ManipulationConfig,
        ManipulationEnv,
        ManipulationTask,
    )
except ImportError as e:
    print(f"Warning: Could not import environments module: {e}")

# Import multi-agent modules
try:
    from .multi_agent import (
        AttentionComm,
        CentralizedCritic,
        MultiAgentPolicy,
        SpaceTimeEdge,
        SpaceTimeGCS,
        SpaceTimeVertex,
    )
except ImportError as e:
    print(f"Warning: Could not import multi_agent module: {e}")

__all__ = [
    # Core
    "ConfigSpace",
    "Trajectory",
    "BezierTrajectory",
    "IRISDecomposer",
    "Ellipsoid",
    "SimpleBoxObstacle",
    "MICPSolver",
    "GCSGraph",
    # Integration
    "WeightedBlender",
    "HierarchicalBlender",
    "ConflictResolutionBlender",
    "PriorityNetworkBlender",
    "SafetyFilter",
    "SafetyFilterConfig",
    "ControlBarrierFilter",
    "DualPathwayExtractor",
    "FeatureExtractorConfig",
    # Environments
    "BaseEnvironment",
    "GraspingEnv",
    "GraspingConfig",
    "DroneNavEnv",
    "DroneNavConfig",
    "ManipulationEnv",
    "ManipulationConfig",
    "ManipulationTask",
    # Multi-Agent
    "AttentionComm",
    "MultiAgentPolicy",
    "CentralizedCritic",
    "SpaceTimeVertex",
    "SpaceTimeEdge",
    "SpaceTimeGCS",
]
