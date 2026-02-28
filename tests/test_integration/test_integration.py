"""
Unit tests for Integration module.

Tests blending methods, safety filter, and feature extractor.
"""

import numpy as np
import pytest
import torch

from hybrid_gcs.integration import (
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


class TestWeightedBlender:
    """Test WeightedBlender class."""

    def test_pure_gcs(self):
        """Test alpha=1.0 returns pure GCS action."""
        blender = WeightedBlender(alpha=1.0)
        gcs = np.array([1.0, 0.0, 0.5])
        rl = np.array([0.0, 1.0, -0.5])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, gcs)

    def test_pure_rl(self):
        """Test alpha=0.0 returns pure RL action."""
        blender = WeightedBlender(alpha=0.0)
        gcs = np.array([1.0, 0.0, 0.5])
        rl = np.array([0.0, 1.0, -0.5])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, rl)

    def test_equal_blend(self):
        """Test alpha=0.5 returns average."""
        blender = WeightedBlender(alpha=0.5)
        gcs = np.array([2.0, 0.0])
        rl = np.array([0.0, 2.0])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_alpha_property(self):
        """Test alpha getter and setter."""
        blender = WeightedBlender(alpha=0.3)
        assert blender.alpha == 0.3
        blender.alpha = 0.7
        assert blender.alpha == 0.7

    def test_invalid_alpha(self):
        """Test invalid alpha raises error."""
        with pytest.raises(AssertionError):
            WeightedBlender(alpha=1.5)
        with pytest.raises(AssertionError):
            WeightedBlender(alpha=-0.1)


class TestHierarchicalBlender:
    """Test HierarchicalBlender class."""

    def test_safe_gcs_action(self):
        """Test that safe GCS action is used when safe."""
        blender = HierarchicalBlender(is_safe=lambda a: True, transition_zone=0.0)
        gcs = np.array([1.0, 0.0])
        rl = np.array([0.0, 1.0])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, gcs)

    def test_unsafe_gcs_action(self):
        """Test that RL action is used when GCS is unsafe."""
        blender = HierarchicalBlender(is_safe=lambda a: False, transition_zone=0.0)
        gcs = np.array([1.0, 0.0])
        rl = np.array([0.0, 1.0])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, rl)

    def test_transition_zone(self):
        """Test smooth blending in transition zone."""
        blender = HierarchicalBlender(is_safe=lambda a: True, transition_zone=10.0)
        gcs = np.array([1.0, 0.0])
        rl = np.array([1.0, 0.0])
        # When gcs and rl are identical, diff=0 < transition_zone,
        # so t=0 and result = 0*gcs + 1*rl = rl
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, rl)


class TestConflictResolutionBlender:
    """Test ConflictResolutionBlender class."""

    def test_no_conflict(self):
        """Test that RL action is used when no conflict."""
        blender = ConflictResolutionBlender(angle_threshold=np.pi / 4, magnitude_threshold=2.0)
        gcs = np.array([1.0, 0.0])
        rl = np.array([1.1, 0.0])  # Same direction, similar magnitude
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, rl)

    def test_angle_conflict(self):
        """Test conflict detection on opposing actions."""
        blender = ConflictResolutionBlender(angle_threshold=np.pi / 4, magnitude_threshold=10.0)
        gcs = np.array([1.0, 0.0])
        rl = np.array([0.0, 1.0])  # 90 degrees apart > pi/4
        result = blender.blend(gcs, rl)
        # RL projected onto GCS direction should be zero (dot=0)
        assert np.linalg.norm(result) < 1e-6 or np.allclose(result, gcs)

    def test_magnitude_conflict(self):
        """Test conflict detection on magnitude divergence."""
        blender = ConflictResolutionBlender(angle_threshold=np.pi, magnitude_threshold=1.5)
        gcs = np.array([1.0, 0.0])
        rl = np.array([3.0, 0.0])  # ratio 3.0 > 1.5
        result = blender.blend(gcs, rl)
        # Should project RL onto GCS direction
        assert result.shape == gcs.shape

    def test_zero_gcs_action(self):
        """Test handling of zero GCS action."""
        blender = ConflictResolutionBlender()
        gcs = np.array([0.0, 0.0])
        rl = np.array([1.0, 1.0])
        result = blender.blend(gcs, rl)
        # No conflict since gcs_norm=0
        np.testing.assert_array_almost_equal(result, rl)


class TestSafetyFilter:
    """Test SafetyFilter class."""

    def test_check_bounds_valid(self):
        """Test valid state passes bounds check."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([0.0, 0.0]),
            position_bounds_upper=np.array([10.0, 10.0]),
        )
        sf = SafetyFilter(config)
        assert sf.check_bounds(np.array([5.0, 5.0]))

    def test_check_bounds_invalid(self):
        """Test out-of-bounds state fails bounds check."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([0.0, 0.0]),
            position_bounds_upper=np.array([10.0, 10.0]),
        )
        sf = SafetyFilter(config)
        assert not sf.check_bounds(np.array([15.0, 5.0]))

    def test_check_velocity_valid(self):
        """Test valid velocity passes check."""
        config = SafetyFilterConfig(max_velocity=2.0)
        sf = SafetyFilter(config)
        assert sf.check_velocity(np.array([1.0, 0.0, 0.0]))

    def test_check_velocity_invalid(self):
        """Test excessive velocity fails check."""
        config = SafetyFilterConfig(max_velocity=1.0)
        sf = SafetyFilter(config)
        assert not sf.check_velocity(np.array([1.0, 1.0, 1.0]))

    def test_check_collision_no_obstacles(self):
        """Test collision check with no obstacles."""
        config = SafetyFilterConfig()
        sf = SafetyFilter(config)
        assert sf.check_collision(np.array([0.0, 0.0, 0.0]))

    def test_check_collision_with_obstacles(self):
        """Test collision detection with obstacles."""
        config = SafetyFilterConfig(
            obstacle_positions=[np.array([0.0, 0.0, 0.0])],
            obstacle_radii=[1.0],
            collision_margin=0.1,
        )
        sf = SafetyFilter(config)
        # Inside obstacle
        assert not sf.check_collision(np.array([0.5, 0.0, 0.0]))
        # Outside obstacle
        assert sf.check_collision(np.array([5.0, 0.0, 0.0]))

    def test_is_safe(self):
        """Test comprehensive safety check."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([0.0, 0.0, 0.0]),
            position_bounds_upper=np.array([10.0, 10.0, 10.0]),
            max_velocity=2.0,
        )
        sf = SafetyFilter(config)
        # position=5,5,5, velocity=0,0,0 -> safe
        state = np.array([5.0, 5.0, 5.0, 0.0, 0.0, 0.0])
        assert sf.is_safe(state)

    def test_filter_action_clips_acceleration(self):
        """Test that filter clips excessive acceleration."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([-100.0, -100.0, -100.0]),
            position_bounds_upper=np.array([100.0, 100.0, 100.0]),
            max_acceleration=1.0,
            max_velocity=100.0,
        )
        sf = SafetyFilter(config)
        action = np.array([10.0, 0.0, 0.0])
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        safe = sf.filter_action(action, state, dt=0.01)
        assert np.linalg.norm(safe) <= 1.0 + 1e-6

    def test_filter_action_returns_array(self):
        """Test that filter returns numpy array."""
        config = SafetyFilterConfig()
        sf = SafetyFilter(config)
        action = np.array([0.1, 0.0, 0.0])
        state = np.zeros(6)
        result = sf.filter_action(action, state)
        assert isinstance(result, np.ndarray)


class TestDualPathwayExtractor:
    """Test DualPathwayExtractor class."""

    def test_forward(self):
        """Test forward pass produces correct shapes."""
        config = FeatureExtractorConfig(input_dim=10, gcs_feature_dim=4, rl_feature_dim=16)
        extractor = DualPathwayExtractor(config)
        x = torch.randn(8, 10)
        gcs_feats, rl_feats = extractor(x)
        assert gcs_feats.shape == (8, 4)
        assert rl_feats.shape == (8, 16)

    def test_extract_gcs_features(self):
        """Test GCS-only feature extraction."""
        config = FeatureExtractorConfig(input_dim=6, gcs_feature_dim=3)
        extractor = DualPathwayExtractor(config)
        x = torch.randn(4, 6)
        gcs_feats = extractor.extract_gcs_features(x)
        assert gcs_feats.shape == (4, 3)

    def test_extract_rl_features(self):
        """Test RL-only feature extraction."""
        config = FeatureExtractorConfig(input_dim=6, rl_feature_dim=32)
        extractor = DualPathwayExtractor(config)
        x = torch.randn(4, 6)
        rl_feats = extractor.extract_rl_features(x)
        assert rl_feats.shape == (4, 32)

    def test_gradient_flow(self):
        """Test that gradients flow through both pathways."""
        config = FeatureExtractorConfig(input_dim=5, gcs_feature_dim=2, rl_feature_dim=8)
        extractor = DualPathwayExtractor(config)
        x = torch.randn(2, 5, requires_grad=True)
        gcs_feats, rl_feats = extractor(x)
        loss = gcs_feats.sum() + rl_feats.sum()
        loss.backward()
        assert x.grad is not None

    def test_config_validation(self):
        """Test config validation."""
        with pytest.raises(AssertionError):
            FeatureExtractorConfig(input_dim=0)


class TestPriorityNetworkBlender:
    """Test PriorityNetworkBlender class."""

    def test_high_priority_gcs(self):
        """Test that high priority returns GCS action."""
        blender = PriorityNetworkBlender(
            priority_fn=lambda **kwargs: 0.9, high_threshold=0.7, low_threshold=0.3
        )
        gcs = np.array([1.0, 0.0])
        rl = np.array([0.0, 1.0])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, gcs)

    def test_low_priority_rl(self):
        """Test that low priority returns RL action."""
        blender = PriorityNetworkBlender(
            priority_fn=lambda **kwargs: 0.1, high_threshold=0.7, low_threshold=0.3
        )
        gcs = np.array([1.0, 0.0])
        rl = np.array([0.0, 1.0])
        result = blender.blend(gcs, rl)
        np.testing.assert_array_almost_equal(result, rl)

    def test_mid_priority_blends(self):
        """Test that mid-range priority blends actions."""
        blender = PriorityNetworkBlender(
            priority_fn=lambda **kwargs: 0.5, high_threshold=0.7, low_threshold=0.3
        )
        gcs = np.array([2.0, 0.0])
        rl = np.array([0.0, 2.0])
        result = blender.blend(gcs, rl)
        # p=0.5: 0.5*gcs + 0.5*rl = [1.0, 1.0]
        np.testing.assert_array_almost_equal(result, [1.0, 1.0])

    def test_priority_fn_receives_kwargs(self):
        """Test that kwargs are passed to priority function."""
        received = {}

        def mock_priority_fn(**kwargs):
            received.update(kwargs)
            return 0.5

        blender = PriorityNetworkBlender(priority_fn=mock_priority_fn)
        gcs = np.array([1.0])
        rl = np.array([0.0])
        blender.blend(gcs, rl, state=np.array([1.0, 2.0]))
        assert "state" in received
        assert "gcs_action" in received
        assert "rl_action" in received

    def test_invalid_thresholds(self):
        """Test invalid threshold configuration."""
        with pytest.raises(AssertionError):
            PriorityNetworkBlender(
                priority_fn=lambda **kwargs: 0.5, high_threshold=0.3, low_threshold=0.7
            )


class TestSafetyFilterCheckAcceleration:
    """Test check_acceleration method of SafetyFilter."""

    def test_valid_acceleration(self):
        """Test valid acceleration passes check."""
        config = SafetyFilterConfig(max_acceleration=5.0)
        sf = SafetyFilter(config)
        assert sf.check_acceleration(np.array([1.0, 0.0, 0.0]))

    def test_invalid_acceleration(self):
        """Test excessive acceleration fails check."""
        config = SafetyFilterConfig(max_acceleration=1.0)
        sf = SafetyFilter(config)
        assert not sf.check_acceleration(np.array([1.0, 1.0, 1.0]))


class TestControlBarrierFilter:
    """Test ControlBarrierFilter class."""

    def test_creation(self):
        """Test CBF filter creation."""
        config = SafetyFilterConfig()
        cbf = ControlBarrierFilter(config, alpha=1.0)
        assert cbf.alpha == 1.0

    def test_barrier_bounds_safe(self):
        """Test barrier values for safe position."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([0.0, 0.0, 0.0]),
            position_bounds_upper=np.array([10.0, 10.0, 10.0]),
        )
        cbf = ControlBarrierFilter(config)
        barriers = cbf.barrier_bounds(np.array([5.0, 5.0, 5.0]))
        # All should be negative (safe)
        assert np.all(barriers < 0)

    def test_barrier_bounds_unsafe(self):
        """Test barrier values for unsafe position."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([0.0, 0.0, 0.0]),
            position_bounds_upper=np.array([10.0, 10.0, 10.0]),
        )
        cbf = ControlBarrierFilter(config)
        barriers = cbf.barrier_bounds(np.array([11.0, 5.0, 5.0]))
        # First element should be positive (upper bound violated)
        assert barriers[0] > 0

    def test_barrier_obstacles_safe(self):
        """Test obstacle barriers for safe position."""
        config = SafetyFilterConfig(
            obstacle_positions=[np.array([5.0, 5.0, 5.0])],
            obstacle_radii=[1.0],
            collision_margin=0.1,
        )
        cbf = ControlBarrierFilter(config)
        barriers = cbf.barrier_obstacles(np.array([10.0, 5.0, 5.0]))
        # Should be negative (safe, far from obstacle)
        assert np.all(barriers < 0)

    def test_barrier_obstacles_no_obstacles(self):
        """Test obstacle barriers when no obstacles configured."""
        config = SafetyFilterConfig()
        cbf = ControlBarrierFilter(config)
        barriers = cbf.barrier_obstacles(np.array([0.0, 0.0, 0.0]))
        assert len(barriers) == 0

    def test_filter_action_returns_array(self):
        """Test that filter returns numpy array."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([-10.0, -10.0, -10.0]),
            position_bounds_upper=np.array([10.0, 10.0, 10.0]),
        )
        cbf = ControlBarrierFilter(config)
        action = np.array([0.1, 0.0, 0.0])
        state = np.zeros(6)
        result = cbf.filter_action(action, state, dt=0.01)
        assert isinstance(result, np.ndarray)
        assert result.shape == action.shape

    def test_filter_action_clips_acceleration(self):
        """Test that CBF filter clips excessive acceleration."""
        config = SafetyFilterConfig(
            position_bounds_lower=np.array([-100.0, -100.0, -100.0]),
            position_bounds_upper=np.array([100.0, 100.0, 100.0]),
            max_acceleration=1.0,
            max_velocity=100.0,
        )
        cbf = ControlBarrierFilter(config, alpha=1.0)
        action = np.array([10.0, 0.0, 0.0])
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        safe = cbf.filter_action(action, state, dt=0.01)
        assert np.linalg.norm(safe) <= 1.0 + 1e-6

    def test_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        config = SafetyFilterConfig()
        with pytest.raises(AssertionError):
            ControlBarrierFilter(config, alpha=0.0)
        with pytest.raises(AssertionError):
            ControlBarrierFilter(config, alpha=-1.0)
