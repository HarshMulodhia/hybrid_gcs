"""
Unit tests for Reward Shaper module.

Tests reward strategies and composition.
"""

import pytest
import numpy as np
from hybrid_gcs.training import (
    RewardComposer,
    DistanceReward,
    GoalReachReward,
    ActionPenalty,
    CollisionPenalty,
    SmoothnessReward,
    EfficiencyReward,
    RewardConfig,
    create_reward_composer,
)


class TestDistanceReward:
    """Test DistanceReward strategy."""

    def test_zero_distance(self):
        """Test reward when at goal."""
        reward = DistanceReward(weight=1.0, scale=0.1)
        pos = np.array([1.0, 2.0])
        r = reward(current_position=pos, goal_position=pos)
        assert r == 0.0

    def test_positive_distance(self):
        """Test reward is negative when away from goal."""
        reward = DistanceReward(weight=1.0, scale=0.1)
        r = reward(
            current_position=np.array([0.0, 0.0]),
            goal_position=np.array([3.0, 4.0]),
        )
        assert r < 0
        assert abs(r - (-0.5)) < 1e-10  # -0.1 * 5.0 = -0.5


class TestGoalReachReward:
    """Test GoalReachReward strategy."""

    def test_goal_reached(self):
        """Test bonus when goal is reached."""
        reward = GoalReachReward(weight=10.0, threshold=0.5, bonus=1.0)
        r = reward(
            current_position=np.array([1.0, 1.0]),
            goal_position=np.array([1.0, 1.0]),
        )
        assert r == 10.0

    def test_goal_not_reached(self):
        """Test no bonus when goal is not reached."""
        reward = GoalReachReward(weight=10.0, threshold=0.5, bonus=1.0)
        r = reward(
            current_position=np.array([0.0, 0.0]),
            goal_position=np.array([5.0, 5.0]),
        )
        assert r == 0.0


class TestActionPenalty:
    """Test ActionPenalty strategy."""

    def test_zero_action(self):
        """Test no penalty for zero action."""
        penalty = ActionPenalty(weight=0.01, scale=1.0)
        r = penalty(action=np.array([0.0, 0.0]))
        assert r == 0.0

    def test_nonzero_action(self):
        """Test penalty for nonzero action."""
        penalty = ActionPenalty(weight=1.0, scale=1.0)
        r = penalty(action=np.array([1.0, 0.0]))
        assert r < 0
        assert abs(r - (-1.0)) < 1e-10


class TestCollisionPenalty:
    """Test CollisionPenalty strategy."""

    def test_no_collision(self):
        """Test no penalty when no collision."""
        penalty = CollisionPenalty(weight=1.0, penalty=5.0)
        r = penalty(collision=False)
        assert r == 0.0

    def test_collision(self):
        """Test penalty on collision."""
        penalty = CollisionPenalty(weight=1.0, penalty=5.0)
        r = penalty(collision=True)
        assert abs(r - (-5.0)) < 1e-10


class TestSmoothnessReward:
    """Test SmoothnessReward strategy."""

    def test_constant_velocity(self):
        """Test no penalty for constant velocity."""
        reward = SmoothnessReward(weight=1.0, scale=1.0)
        v = np.array([1.0, 0.0])
        r = reward(current_velocity=v, previous_velocity=v)
        assert r == 0.0

    def test_changing_velocity(self):
        """Test penalty for velocity change."""
        reward = SmoothnessReward(weight=1.0, scale=1.0)
        r = reward(
            current_velocity=np.array([1.0, 0.0]),
            previous_velocity=np.array([0.0, 0.0]),
        )
        assert r < 0


class TestEfficiencyReward:
    """Test EfficiencyReward strategy."""

    def test_no_movement(self):
        """Test no penalty for zero movement."""
        reward = EfficiencyReward(weight=1.0, scale=1.0)
        r = reward(position_delta=np.array([0.0, 0.0]))
        assert r == 0.0

    def test_movement(self):
        """Test penalty for movement."""
        reward = EfficiencyReward(weight=1.0, scale=1.0)
        r = reward(position_delta=np.array([1.0, 0.0]))
        assert r < 0


class TestRewardComposer:
    """Test RewardComposer class."""

    def test_empty_composer(self):
        """Test composer with no strategies."""
        composer = RewardComposer()
        r = composer.compute_reward()
        assert r == 0.0

    def test_add_strategy(self):
        """Test adding strategies."""
        composer = RewardComposer()
        composer.add_strategy(DistanceReward(), "distance")
        assert "distance" in composer.strategies

    def test_remove_strategy(self):
        """Test removing strategies."""
        composer = RewardComposer()
        composer.add_strategy(DistanceReward(), "distance")
        composer.remove_strategy("distance")
        assert "distance" not in composer.strategies

    def test_combined_reward(self):
        """Test combined reward computation with matching kwargs."""
        composer = RewardComposer()
        composer.add_strategy(
            DistanceReward(weight=1.0, scale=0.1), "distance"
        )

        # Only pass kwargs that the distance strategy accepts
        r = composer.compute_reward(
            current_position=np.array([0.0, 0.0]),
            goal_position=np.array([3.0, 4.0]),
        )
        assert r < 0

    def test_reward_breakdown(self):
        """Test reward breakdown."""
        composer = RewardComposer()
        composer.add_strategy(
            CollisionPenalty(weight=1.0, penalty=5.0), "collision"
        )

        breakdown = composer.compute_reward_breakdown(
            collision=False,
        )
        assert "collision" in breakdown
        assert breakdown["collision"] == 0.0

    def test_set_strategy_weight(self):
        """Test setting strategy weight."""
        composer = RewardComposer()
        composer.add_strategy(DistanceReward(weight=1.0), "distance")
        composer.set_strategy_weight("distance", 2.0)
        assert composer.get_strategy_weight("distance") == 2.0


class TestCreateRewardComposer:
    """Test create_reward_composer factory function."""

    def test_default_config(self):
        """Test creating composer with default config."""
        config = RewardConfig()
        composer = create_reward_composer(config)
        assert len(composer.strategies) > 0

    def test_minimal_config(self):
        """Test creating composer with minimal config."""
        config = RewardConfig(
            use_distance_reward=True,
            use_goal_reward=False,
            use_action_penalty=False,
            use_collision_penalty=False,
            use_smoothness_reward=False,
            use_efficiency_reward=False,
        )
        composer = create_reward_composer(config)
        assert len(composer.strategies) == 1
