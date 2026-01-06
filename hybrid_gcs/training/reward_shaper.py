"""
Reward Shaping and Composition for Hybrid-GCS.

Implements multiple reward strategies that can be composed together:
- Distance-based rewards
- Goal reaching rewards
- Action penalties
- Collision penalties
- Smoothness rewards
"""

from dataclasses import dataclass
from typing import Dict, List, Callable
from abc import ABC, abstractmethod
import numpy as np


class RewardStrategy(ABC):
    """Base class for reward strategies."""
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize reward strategy.
        
        Args:
            weight: Weight for this reward component
        """
        self.weight = weight
    
    @abstractmethod
    def compute(self, **kwargs) -> float:
        """Compute reward. Must be implemented by subclasses."""
        pass
    
    def __call__(self, **kwargs) -> float:
        """Compute weighted reward."""
        return self.weight * self.compute(**kwargs)


class DistanceReward(RewardStrategy):
    """
    Reward based on distance to goal.
    
    Encourages the agent to move closer to the goal.
    R = -α * ||current_pos - goal_pos||
    """
    
    def __init__(self, weight: float = 1.0, scale: float = 0.1):
        """
        Initialize distance reward.
        
        Args:
            weight: Weight for this reward component
            scale: Scaling factor for distance
        """
        super().__init__(weight)
        self.scale = scale
    
    def compute(self, current_position: np.ndarray, goal_position: np.ndarray) -> float:
        """
        Compute distance-based reward.
        
        Args:
            current_position: Current position [x, y, ...] or [x, y, z]
            goal_position: Goal position
        
        Returns:
            Negative distance reward
        """
        distance = np.linalg.norm(current_position - goal_position)
        return -self.scale * distance


class GoalReachReward(RewardStrategy):
    """
    Bonus reward for reaching the goal.
    
    R = bonus if distance < threshold, else 0
    """
    
    def __init__(self, weight: float = 10.0, threshold: float = 0.5, bonus: float = 1.0):
        """
        Initialize goal reach reward.
        
        Args:
            weight: Weight for this reward component
            threshold: Distance threshold for goal reaching
            bonus: Bonus reward when goal is reached
        """
        super().__init__(weight)
        self.threshold = threshold
        self.bonus = bonus
    
    def compute(self, current_position: np.ndarray, goal_position: np.ndarray) -> float:
        """
        Compute goal reaching reward.
        
        Args:
            current_position: Current position
            goal_position: Goal position
        
        Returns:
            Bonus if goal reached, else 0
        """
        distance = np.linalg.norm(current_position - goal_position)
        if distance < self.threshold:
            return self.bonus
        return 0.0


class ActionPenalty(RewardStrategy):
    """
    Penalty for large actions.
    
    Encourages smooth, low-magnitude actions.
    R = -α * ||action||²
    """
    
    def __init__(self, weight: float = 0.01, scale: float = 1.0):
        """
        Initialize action penalty.
        
        Args:
            weight: Weight for this reward component
            scale: Scaling factor for action magnitude
        """
        super().__init__(weight)
        self.scale = scale
    
    def compute(self, action: np.ndarray) -> float:
        """
        Compute action penalty.
        
        Args:
            action: Action vector
        
        Returns:
            Negative action magnitude squared
        """
        action_magnitude = np.sum(action ** 2)
        return -self.scale * action_magnitude


class CollisionPenalty(RewardStrategy):
    """
    Penalty for collisions.
    
    R = -penalty if collision, else 0
    """
    
    def __init__(self, weight: float = 1.0, penalty: float = 1.0):
        """
        Initialize collision penalty.
        
        Args:
            weight: Weight for this reward component
            penalty: Penalty value for collision
        """
        super().__init__(weight)
        self.penalty = penalty
    
    def compute(self, collision: bool = False) -> float:
        """
        Compute collision penalty.
        
        Args:
            collision: Whether collision occurred
        
        Returns:
            Negative penalty if collision, else 0
        """
        if collision:
            return -self.penalty
        return 0.0


class SmoothnessReward(RewardStrategy):
    """
    Reward for smooth trajectories.
    
    Encourages low acceleration/jerk.
    R = -α * ||acceleration||
    """
    
    def __init__(self, weight: float = 0.01, scale: float = 1.0):
        """
        Initialize smoothness reward.
        
        Args:
            weight: Weight for this reward component
            scale: Scaling factor for acceleration
        """
        super().__init__(weight)
        self.scale = scale
    
    def compute(self, current_velocity: np.ndarray, previous_velocity: np.ndarray) -> float:
        """
        Compute smoothness reward.
        
        Args:
            current_velocity: Current velocity
            previous_velocity: Previous velocity
        
        Returns:
            Negative acceleration magnitude
        """
        acceleration = current_velocity - previous_velocity
        accel_magnitude = np.linalg.norm(acceleration)
        return -self.scale * accel_magnitude


class EfficiencyReward(RewardStrategy):
    """
    Reward for efficient paths.
    
    R = -α * path_length (encourages shorter paths)
    """
    
    def __init__(self, weight: float = 0.1, scale: float = 0.01):
        """
        Initialize efficiency reward.
        
        Args:
            weight: Weight for this reward component
            scale: Scaling factor for path length
        """
        super().__init__(weight)
        self.scale = scale
    
    def compute(self, position_delta: np.ndarray) -> float:
        """
        Compute efficiency reward.
        
        Args:
            position_delta: Change in position this step
        
        Returns:
            Negative step distance
        """
        step_distance = np.linalg.norm(position_delta)
        return -self.scale * step_distance


class RewardComposer:
    """
    Composes multiple reward strategies.
    
    Allows flexible reward design by combining different strategies.
    """
    
    def __init__(self):
        """Initialize reward composer."""
        self.strategies: Dict[str, RewardStrategy] = {}
    
    def add_strategy(self, strategy: RewardStrategy, name: str = None):
        """
        Add reward strategy.
        
        Args:
            strategy: RewardStrategy instance
            name: Name for this strategy (default: class name)
        """
        if name is None:
            name = strategy.__class__.__name__
        
        self.strategies[name] = strategy
    
    def remove_strategy(self, name: str):
        """Remove reward strategy."""
        if name in self.strategies:
            del self.strategies[name]
    
    def compute_reward(self, **kwargs) -> float:
        """
        Compute total reward from all strategies.
        
        Args:
            **kwargs: Arguments passed to all strategies
        
        Returns:
            Total composed reward
        """
        total_reward = 0.0
        
        for strategy in self.strategies.values():
            try:
                reward = strategy(**kwargs)
                total_reward += reward
            except TypeError:
                # Strategy doesn't use these kwargs, skip it
                pass
        
        return total_reward
    
    def compute_reward_breakdown(self, **kwargs) -> Dict[str, float]:
        """
        Compute reward breakdown by strategy.
        
        Useful for analysis and debugging.
        
        Args:
            **kwargs: Arguments passed to all strategies
        
        Returns:
            Dictionary of rewards by strategy name
        """
        breakdown = {}
        
        for name, strategy in self.strategies.items():
            try:
                reward = strategy(**kwargs)
                breakdown[name] = reward
            except TypeError:
                pass
        
        return breakdown
    
    def get_strategy(self, name: str) -> RewardStrategy:
        """Get strategy by name."""
        return self.strategies.get(name)
    
    def set_strategy_weight(self, name: str, weight: float):
        """Set weight for a strategy."""
        if name in self.strategies:
            self.strategies[name].weight = weight
    
    def get_strategy_weight(self, name: str) -> float:
        """Get weight for a strategy."""
        if name in self.strategies:
            return self.strategies[name].weight
        return 0.0
    
    def __repr__(self) -> str:
        """String representation."""
        lines = ["RewardComposer:"]
        for name, strategy in self.strategies.items():
            lines.append(f"  {name}: weight={strategy.weight}")
        return "\n".join(lines)


@dataclass
class RewardConfig:
    """Configuration for reward composition."""
    
    # Distance reward
    use_distance_reward: bool = True
    distance_weight: float = 1.0
    distance_scale: float = 0.1
    
    # Goal reaching
    use_goal_reward: bool = True
    goal_weight: float = 10.0
    goal_threshold: float = 0.5
    goal_bonus: float = 1.0
    
    # Action penalty
    use_action_penalty: bool = True
    action_weight: float = 0.01
    action_scale: float = 1.0
    
    # Collision penalty
    use_collision_penalty: bool = True
    collision_weight: float = 1.0
    collision_penalty: float = 1.0
    
    # Smoothness
    use_smoothness_reward: bool = False
    smoothness_weight: float = 0.01
    smoothness_scale: float = 1.0
    
    # Efficiency
    use_efficiency_reward: bool = False
    efficiency_weight: float = 0.1
    efficiency_scale: float = 0.01


def create_reward_composer(config: RewardConfig) -> RewardComposer:
    """
    Create reward composer from configuration.
    
    Args:
        config: RewardConfig instance
    
    Returns:
        Configured RewardComposer
    """
    composer = RewardComposer()
    
    if config.use_distance_reward:
        composer.add_strategy(
            DistanceReward(config.distance_weight, config.distance_scale),
            'distance'
        )
    
    if config.use_goal_reward:
        composer.add_strategy(
            GoalReachReward(config.goal_weight, config.goal_threshold, config.goal_bonus),
            'goal_reach'
        )
    
    if config.use_action_penalty:
        composer.add_strategy(
            ActionPenalty(config.action_weight, config.action_scale),
            'action_penalty'
        )
    
    if config.use_collision_penalty:
        composer.add_strategy(
            CollisionPenalty(config.collision_weight, config.collision_penalty),
            'collision'
        )
    
    if config.use_smoothness_reward:
        composer.add_strategy(
            SmoothnessReward(config.smoothness_weight, config.smoothness_scale),
            'smoothness'
        )
    
    if config.use_efficiency_reward:
        composer.add_strategy(
            EfficiencyReward(config.efficiency_weight, config.efficiency_scale),
            'efficiency'
        )
    
    return composer
