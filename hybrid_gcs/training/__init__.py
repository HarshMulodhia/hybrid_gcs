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

from .policy_network import (
    PolicyNetwork,
    PolicyNetworkConfig,
    PolicyNetworkWithLSTM,
    CNNEncoder
)

from .ppo_trainer import (
    PPOTrainer,
    PPOConfig,
)

from .reward_shaper import (
    RewardComposer,
    RewardStrategy,
    DistanceReward,
    GoalReachReward,
    ActionPenalty,
    CollisionPenalty,
    SmoothnessReward,
    EfficiencyReward,
    RewardConfig,
    create_reward_composer
)

from .curriculum_scheduler import (
    CurriculumManager,
    LinearCurriculum,
    ExponentialCurriculum,
    StepCurriculum,
    SigmoidCurriculum,
    PerformanceCurriculum,
    CurriculumConfig,
    CurriculumType
)

from .experience_buffer import (
    ExperienceBuffer,
    PrioritizedExperienceBuffer,
    TrajectoryBuffer
)

__all__ = [
    # Policy Network
    'PolicyNetwork',
    'PolicyNetworkConfig',
    'PolicyNetworkWithLSTM',
    'CNNEncoder',
    
    # PPO Trainer
    'PPOTrainer',
    'PPOConfig',
    
    # Reward Shaping
    'RewardComposer',
    'RewardStrategy',
    'DistanceReward',
    'GoalReachReward',
    'ActionPenalty',
    'CollisionPenalty',
    'SmoothnessReward',
    'EfficiencyReward',
    'RewardConfig',
    'create_reward_composer',
    
    # Curriculum Learning
    'CurriculumManager',
    'LinearCurriculum',
    'ExponentialCurriculum',
    'StepCurriculum',
    'SigmoidCurriculum',
    'PerformanceCurriculum',
    'CurriculumConfig',
    'CurriculumType',
    
    # Experience Buffer
    'ExperienceBuffer',
    'PrioritizedExperienceBuffer',
    'TrajectoryBuffer'
]

__version__ = '0.1.0'
