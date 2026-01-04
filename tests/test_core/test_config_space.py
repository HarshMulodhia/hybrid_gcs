"""
Unit tests for ConfigSpace module.

Tests configuration space validation, sampling, and common operations.
"""

import pytest
import numpy as np
from hybrid_gcs.core import ConfigSpace


class TestConfigSpaceBasics:
    """Test basic ConfigSpace functionality."""
    
    @pytest.fixture
    def simple_2d_space(self):
        """Create simple 2D configuration space."""
        return ConfigSpace(
            dim=2,
            bounds_lower=np.array([0.0, 0.0]),
            bounds_upper=np.array([10.0, 10.0]),
            names=['x', 'y']
        )
    
    def test_creation(self, simple_2d_space):
        """Test ConfigSpace creation."""
        assert simple_2d_space.dim == 2
        assert len(simple_2d_space.names) == 2
    
    def test_valid_config(self, simple_2d_space):
        """Test valid configuration check."""
        assert simple_2d_space.is_valid(np.array([5.0, 5.0]))
        assert simple_2d_space.is_valid(np.array([0.0, 0.0]))
        assert simple_2d_space.is_valid(np.array([10.0, 10.0]))
    
    def test_invalid_config(self, simple_2d_space):
        """Test invalid configuration check."""
        assert not simple_2d_space.is_valid(np.array([-1.0, 5.0]))
        assert not simple_2d_space.is_valid(np.array([5.0, 11.0]))
        assert not simple_2d_space.is_valid(np.array([15.0, 15.0]))
    
    def test_projection(self, simple_2d_space):
        """Test configuration projection."""
        projected = simple_2d_space.project(np.array([-1.0, 5.0]))
        assert simple_2d_space.is_valid(projected)
        assert projected[0] == 0.0
        assert projected[1] == 5.0


class TestConfigSpaceSampling:
    """Test ConfigSpace sampling."""
    
    @pytest.fixture
    def space(self):
        return ConfigSpace(
            dim=3,
            bounds_lower=np.array([0.0, 0.0, 0.0]),
            bounds_upper=np.array([1.0, 1.0, 1.0]),
            names=['x', 'y', 'z']
        )
    
    def test_single_sample(self, space):
        """Test single random sample."""
        sample = space.random_sample()
        assert sample.shape == (3,)
        assert space.is_valid(sample)
    
    def test_multiple_samples(self, space):
        """Test multiple random samples."""
        samples = space.random_samples(100)
        assert samples.shape == (100, 3)
        for sample in samples:
            assert space.is_valid(sample)
    
    def test_samples_in_range(self, space):
        """Test that samples are uniformly distributed."""
        samples = space.random_samples(1000)
        assert np.mean(samples) > 0.4  # Should be near 0.5
        assert np.mean(samples) < 0.6


class TestConfigSpaceDistance:
    """Test distance and interpolation."""
    
    @pytest.fixture
    def space(self):
        return ConfigSpace(
            dim=2,
            bounds_lower=np.array([0.0, 0.0]),
            bounds_upper=np.array([10.0, 10.0]),
            names=['x', 'y']
        )
    
    def test_distance(self, space):
        """Test distance computation."""
        q1 = np.array([0.0, 0.0])
        q2 = np.array([3.0, 4.0])
        dist = space.distance(q1, q2)
        assert abs(dist - 5.0) < 1e-10
    
    def test_interpolation(self, space):
        """Test linear interpolation."""
        q1 = np.array([0.0, 0.0])
        q2 = np.array([10.0, 10.0])
        
        q_mid = space.interpolate(q1, q2, 0.5)
        assert np.allclose(q_mid, [5.0, 5.0])
        
        q_quarter = space.interpolate(q1, q2, 0.25)
        assert np.allclose(q_quarter, [2.5, 2.5])
    
    def test_interpolation_extrapolation(self, space):
        """Test that interpolation clips to [0, 1]."""
        q1 = np.array([0.0, 0.0])
        q2 = np.array([10.0, 10.0])
        
        # t > 1 should clip to 1
        q = space.interpolate(q1, q2, 2.0)
        assert np.allclose(q, q2)
        
        # t < 0 should clip to 0
        q = space.interpolate(q1, q2, -1.0)
        assert np.allclose(q, q1)


class TestConfigSpaceVelocityLimits:
    """Test velocity constraints."""
    
    @pytest.fixture
    def space_with_limits(self):
        return ConfigSpace(
            dim=2,
            bounds_lower=np.array([0.0, 0.0]),
            bounds_upper=np.array([1.0, 1.0]),
            names=['x', 'y'],
            velocity_limits=np.array([1.0, 2.0])
        )
    
    def test_velocity_normalization(self, space_with_limits):
        """Test velocity normalization."""
        # Velocity within limits
        v = np.array([0.5, 1.0])
        v_norm = space_with_limits.normalize_velocity(v)
        assert np.allclose(v_norm, v)
        
        # Velocity exceeding limits
        v = np.array([2.0, 4.0])
        v_norm = space_with_limits.normalize_velocity(v)
        # Should be scaled down
        assert np.linalg.norm(v_norm) <= np.linalg.norm(v)


class TestConfigSpaceProperties:
    """Test ConfigSpace properties and utilities."""
    
    @pytest.fixture
    def space(self):
        return ConfigSpace(
            dim=3,
            bounds_lower=np.array([0.0, -5.0, 10.0]),
            bounds_upper=np.array([10.0, 5.0, 20.0]),
            names=['x', 'y', 'z']
        )
    
    def test_get_bounds(self, space):
        """Test getting bounds."""
        lower, upper = space.get_bounds()
        assert np.allclose(lower, [0.0, -5.0, 10.0])
        assert np.allclose(upper, [10.0, 5.0, 20.0])
    
    def test_get_range(self, space):
        """Test getting range."""
        range_vals = space.get_range()
        assert np.allclose(range_vals, [10.0, 10.0, 10.0])
    
    def test_get_center(self, space):
        """Test getting center."""
        center = space.get_center()
        assert np.allclose(center, [5.0, 0.0, 15.0])


class TestConfigSpaceFromRobot:
    """Test creating ConfigSpace from robot parameters."""
    
    def test_from_robot_creation(self):
        """Test creating ConfigSpace from robot specs."""
        space = ConfigSpace.from_robot(
            num_dofs=7,
            joint_lower_limits=np.array([-3.14]*7),
            joint_upper_limits=np.array([3.14]*7),
            joint_names=['j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7'],
            velocity_limits=np.array([1.0]*7)
        )
        
        assert space.dim == 7
        assert len(space.names) == 7
        assert space.velocity_limits is not None


class TestConfigSpaceErrors:
    """Test error handling."""
    
    def test_dimension_mismatch_lower(self):
        """Test dimension mismatch in lower bounds."""
        with pytest.raises(AssertionError):
            ConfigSpace(
                dim=3,
                bounds_lower=np.array([0.0, 0.0]),  # Wrong size
                bounds_upper=np.array([1.0, 1.0, 1.0]),
                names=['x', 'y', 'z']
            )
    
    def test_dimension_mismatch_names(self):
        """Test dimension mismatch in names."""
        with pytest.raises(AssertionError):
            ConfigSpace(
                dim=3,
                bounds_lower=np.array([0.0, 0.0, 0.0]),
                bounds_upper=np.array([1.0, 1.0, 1.0]),
                names=['x', 'y']  # Wrong size
            )
    
    def test_invalid_bounds(self):
        """Test that lower > upper raises error."""
        with pytest.raises(AssertionError):
            ConfigSpace(
                dim=2,
                bounds_lower=np.array([10.0, 10.0]),
                bounds_upper=np.array([0.0, 0.0]),  # Invalid
                names=['x', 'y']
            )
