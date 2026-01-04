"""
Unit tests for Trajectory module.

Tests trajectory creation, evaluation, and Bezier curves.
"""

import pytest
import numpy as np
from hybrid_gcs.core import Trajectory, BezierTrajectory


class TestTrajectoryBasics:
    """Test basic Trajectory functionality."""
    
    @pytest.fixture
    def simple_trajectory(self):
        """Create simple 2D trajectory."""
        waypoints = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0]
        ])
        return Trajectory(waypoints)
    
    def test_creation(self, simple_trajectory):
        """Test trajectory creation."""
        assert len(simple_trajectory) == 3
        assert simple_trajectory.dim == 2
    
    def test_at_time(self, simple_trajectory):
        """Test getting configuration at time."""
        # At start
        config = simple_trajectory.at_time(0.0)
        assert np.allclose(config, [0.0, 0.0])
        
        # At end
        config = simple_trajectory.at_time(1.0)
        assert np.allclose(config, [2.0, 0.0])
        
        # At middle (should interpolate)
        config = simple_trajectory.at_time(0.5)
        assert config.shape == (2,)
    
    def test_length(self, simple_trajectory):
        """Test trajectory length computation."""
        length = simple_trajectory.length()
        # Should be approximately distance from [0,0] to [1,1] + [1,1] to [2,0]
        # ≈ sqrt(2) + sqrt(2) ≈ 2.83
        assert length > 2.0
        assert length < 3.0
    
    def test_duration(self, simple_trajectory):
        """Test trajectory duration."""
        duration = simple_trajectory.duration()
        assert duration == 1.0


class TestTrajectoryInterpolation:
    """Test trajectory interpolation."""
    
    @pytest.fixture
    def trajectory(self):
        waypoints = np.array([
            [0.0, 0.0],
            [5.0, 5.0],
            [10.0, 0.0]
        ])
        return Trajectory(waypoints)
    
    def test_velocity(self, trajectory):
        """Test velocity computation."""
        v = trajectory.velocity_at_time(0.5)
        assert v.shape == (2,)
    
    def test_acceleration(self, trajectory):
        """Test acceleration computation."""
        a = trajectory.acceleration_at_time(0.5)
        assert a.shape == (2,)


class TestTrajectoryResampling:
    """Test trajectory resampling."""
    
    @pytest.fixture
    def trajectory(self):
        waypoints = np.array([
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 0.0]
        ])
        return Trajectory(waypoints)
    
    def test_resample_increase(self, trajectory):
        """Test resampling to more waypoints."""
        resampled = trajectory.resample(10)
        assert len(resampled) == 10
        assert resampled.dim == trajectory.dim
    
    def test_resample_decrease(self, trajectory):
        """Test resampling to fewer waypoints."""
        resampled = trajectory.resample(2)
        assert len(resampled) == 2
    
    def test_resample_preserves_endpoints(self, trajectory):
        """Test that resampling preserves endpoints."""
        resampled = trajectory.resample(10)
        assert np.allclose(resampled.at_time(0.0), trajectory.at_time(0.0))
        assert np.allclose(resampled.at_time(1.0), trajectory.at_time(1.0))


class TestBezierTrajectory:
    """Test Bezier trajectory functionality."""
    
    @pytest.fixture
    def bezier(self):
        """Create simple Bezier curve."""
        control_points = np.array([
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 0.0]
        ])
        return BezierTrajectory(control_points)
    
    def test_creation(self, bezier):
        """Test Bezier creation."""
        assert bezier.degree == 2
        assert bezier.dim == 2
        assert bezier.n_points == 3
    
    def test_evaluation(self, bezier):
        """Test Bezier evaluation."""
        # At t=0, should be at first control point
        config = bezier.eval(0.0)
        assert np.allclose(config, [0.0, 0.0])
        
        # At t=1, should be at last control point
        config = bezier.eval(1.0)
        assert np.allclose(config, [2.0, 0.0])
    
    def test_derivative(self, bezier):
        """Test Bezier derivative."""
        deriv = bezier.derivative(0.5)
        assert deriv.shape == (2,)
    
    def test_to_trajectory(self, bezier):
        """Test converting Bezier to trajectory."""
        traj = bezier.to_trajectory(n_samples=50)
        assert len(traj) == 50
        assert traj.dim == 2
    
    def test_bezier_clipping(self):
        """Test that evaluation clips parameter t."""
        control_points = np.array([[0.0], [1.0], [2.0]])
        bezier = BezierTrajectory(control_points)
        
        # t < 0 should behave as t = 0
        val_at_neg = bezier.eval(-1.0)
        val_at_zero = bezier.eval(0.0)
        assert np.allclose(val_at_neg, val_at_zero)
        
        # t > 1 should behave as t = 1
        val_at_over = bezier.eval(2.0)
        val_at_one = bezier.eval(1.0)
        assert np.allclose(val_at_over, val_at_one)


class TestTrajectorySmoothing:
    """Test trajectory smoothing."""
    
    @pytest.fixture
    def noisy_trajectory(self):
        """Create trajectory with noise."""
        # Create a zigzag pattern
        waypoints = np.array([
            [0.0, 0.0],
            [1.0, 0.5],
            [2.0, 0.0],
            [3.0, 0.5],
            [4.0, 0.0]
        ])
        return Trajectory(waypoints)
    
    def test_smoothing(self, noisy_trajectory):
        """Test trajectory smoothing."""
        smoothed = noisy_trajectory.smooth(smoothing_factor=0.2)
        assert len(smoothed) == len(noisy_trajectory)
        assert smoothed.dim == noisy_trajectory.dim


class TestTrajectoryErrors:
    """Test error handling."""
    
    def test_single_waypoint_error(self):
        """Test that single waypoint raises error."""
        with pytest.raises(ValueError):
            Trajectory(np.array([[0.0, 0.0]]))
    
    def test_invalid_dimension(self):
        """Test that 1D waypoints raise error."""
        with pytest.raises(ValueError):
            Trajectory(np.array([0.0, 1.0, 2.0]))
    
    def test_timestamp_mismatch(self):
        """Test that mismatched timestamps raise error."""
        waypoints = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        with pytest.raises(ValueError):
            Trajectory(waypoints, timestamps=np.array([0.0, 1.0]))  # Only 2 timestamps


class TestBezierErrors:
    """Test Bezier error handling."""
    
    def test_single_point_error(self):
        """Test that single control point raises error."""
        with pytest.raises(ValueError):
            BezierTrajectory(np.array([[0.0, 0.0]]))
    
    def test_degree_mismatch_error(self):
        """Test that incorrect degree raises error."""
        control_points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
        with pytest.raises(ValueError):
            BezierTrajectory(control_points, degree=5)  # Should be 2
