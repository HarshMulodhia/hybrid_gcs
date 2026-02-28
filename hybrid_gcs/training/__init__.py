"""
Training Module for Hybrid-GCS.

Implements Deep Reinforcement Learning (PPO) for robotics control.

Modules:
    - policy_network: Actor-Critic policy networks
    - ppo_trainer: PPO algorithm implementation
    - reward_shaper: Reward composition and shaping
    - curriculum_scheduler: Progressive difficulty scheduling
    - experience_buffer: Trajectory storage and sampling
"""

from .curriculum_scheduler import (
    CurriculumConfig,
    CurriculumManager,
    CurriculumType,
    ExponentialCurriculum,
    LinearCurriculum,
    PerformanceCurriculum,
    SigmoidCurriculum,
    StepCurriculum,
)
from .experience_buffer import ExperienceBuffer, PrioritizedExperienceBuffer, TrajectoryBuffer
from .policy_network import CNNEncoder, PolicyNetwork, PolicyNetworkConfig, PolicyNetworkWithLSTM
from .ppo_trainer import (
    PPOConfig,
    PPOTrainer,
)
from .reward_shaper import (
    ActionPenalty,
    CollisionPenalty,
    DistanceReward,
    EfficiencyReward,
    GoalReachReward,
    RewardComposer,
    RewardConfig,
    RewardStrategy,
    SmoothnessReward,
    create_reward_composer,
)

__all__ = [
    # Policy Network
    "PolicyNetwork",
    "PolicyNetworkConfig",
    "PolicyNetworkWithLSTM",
    "CNNEncoder",
    # PPO Trainer
    "PPOTrainer",
    "PPOConfig",
    # Reward Shaping
    "RewardComposer",
    "RewardStrategy",
    "DistanceReward",
    "GoalReachReward",
    "ActionPenalty",
    "CollisionPenalty",
    "SmoothnessReward",
    "EfficiencyReward",
    "RewardConfig",
    "create_reward_composer",
    # Curriculum Learning
    "CurriculumManager",
    "LinearCurriculum",
    "ExponentialCurriculum",
    "StepCurriculum",
    "SigmoidCurriculum",
    "PerformanceCurriculum",
    "CurriculumConfig",
    "CurriculumType",
    # Experience Buffer
    "ExperienceBuffer",
    "PrioritizedExperienceBuffer",
    "TrajectoryBuffer",
]

__version__ = "0.1.0"
