"""
Unit tests for IRIS Decomposer module.

Tests IRIS decomposition, Ellipsoid, and SimpleBoxObstacle.
"""

import pytest
import numpy as np
from hybrid_gcs.core import ConfigSpace
from hybrid_gcs.core.iris_decomposer import (
    IRISDecomposer,
    Ellipsoid,
    SimpleBoxObstacle,
)


class TestEllipsoid:
    """Test Ellipsoid class."""

    def test_creation(self):
        """Test basic ellipsoid creation."""
        center = np.array([0.0, 0.0])
        Q = np.eye(2)
        ellipsoid = Ellipsoid(center, Q)
        assert ellipsoid.dim == 2

    def test_contains_center(self):
        """Test that center is inside ellipsoid."""
        center = np.array([1.0, 2.0])
        Q = np.eye(2)
        ellipsoid = Ellipsoid(center, Q)
        assert ellipsoid.contains(center)

    def test_contains_boundary(self):
        """Test point on boundary."""
        center = np.array([0.0, 0.0])
        Q = np.eye(2)
        ellipsoid = Ellipsoid(center, Q)
        # Point at distance 1 from center
        assert ellipsoid.contains(np.array([1.0, 0.0]))

    def test_not_contains_outside(self):
        """Test point outside ellipsoid."""
        center = np.array([0.0, 0.0])
        Q = np.eye(2)
        ellipsoid = Ellipsoid(center, Q)
        assert not ellipsoid.contains(np.array([2.0, 0.0]))

    def test_volume_unit_sphere(self):
        """Test volume of unit sphere (2D = unit circle)."""
        center = np.array([0.0, 0.0])
        Q = np.eye(2)
        ellipsoid = Ellipsoid(center, Q)
        # Volume of unit circle = pi
        assert abs(ellipsoid.volume() - np.pi) < 1e-10

    def test_volume_scaled(self):
        """Test volume of scaled ellipsoid."""
        center = np.array([0.0, 0.0])
        # Q = I/4 → ellipsoid has radius 2, area = 4*pi
        Q = np.eye(2) * 0.25
        ellipsoid = Ellipsoid(center, Q)
        assert abs(ellipsoid.volume() - 4 * np.pi) < 1e-10

    def test_invalid_shape_matrix(self):
        """Test that non-positive-definite matrix raises error."""
        center = np.array([0.0, 0.0])
        Q = np.array([[1.0, 0.0], [0.0, -1.0]])  # Not positive definite
        with pytest.raises(ValueError):
            Ellipsoid(center, Q)

    def test_shape_dimension_mismatch(self):
        """Test shape matrix dimension mismatch."""
        center = np.array([0.0, 0.0])
        Q = np.eye(3)
        with pytest.raises(ValueError):
            Ellipsoid(center, Q)


class TestSimpleBoxObstacle:
    """Test SimpleBoxObstacle class."""

    def test_contains_inside(self):
        """Test point inside box."""
        box = SimpleBoxObstacle(
            lower=np.array([0.0, 0.0]),
            upper=np.array([1.0, 1.0]),
        )
        assert box.contains(np.array([0.5, 0.5]))

    def test_contains_outside(self):
        """Test point outside box."""
        box = SimpleBoxObstacle(
            lower=np.array([0.0, 0.0]),
            upper=np.array([1.0, 1.0]),
        )
        assert not box.contains(np.array([2.0, 0.5]))

    def test_signed_distance_inside(self):
        """Test signed distance for point inside box (should be negative)."""
        box = SimpleBoxObstacle(
            lower=np.array([0.0, 0.0]),
            upper=np.array([4.0, 4.0]),
        )
        # Center of box
        d = box.signed_distance(np.array([2.0, 2.0]))
        assert d < 0, "Distance should be negative for interior points"
        assert abs(d - (-2.0)) < 1e-10

    def test_signed_distance_outside_one_axis(self):
        """Test signed distance for point outside on one axis."""
        box = SimpleBoxObstacle(
            lower=np.array([3.0, 3.0]),
            upper=np.array([7.0, 7.0]),
        )
        # Point is 2 units left of box, aligned vertically
        d = box.signed_distance(np.array([1.0, 5.0]))
        assert d > 0, "Distance should be positive for exterior points"
        assert abs(d - 2.0) < 1e-10

    def test_signed_distance_outside_corner(self):
        """Test signed distance for point outside on two axes (corner)."""
        box = SimpleBoxObstacle(
            lower=np.array([3.0, 3.0]),
            upper=np.array([7.0, 7.0]),
        )
        # Point is 2 units from box in both x and y
        d = box.signed_distance(np.array([1.0, 1.0]))
        expected = np.sqrt(2**2 + 2**2)
        assert d > 0
        assert abs(d - expected) < 1e-10

    def test_signed_distance_on_boundary(self):
        """Test signed distance on boundary."""
        box = SimpleBoxObstacle(
            lower=np.array([0.0, 0.0]),
            upper=np.array([1.0, 1.0]),
        )
        d = box.signed_distance(np.array([1.0, 0.5]))
        assert abs(d) < 1e-10


class TestIRISDecomposer:
    """Test IRISDecomposer class."""

    @pytest.fixture
    def config_space_2d(self):
        """Create 2D configuration space."""
        return ConfigSpace(
            dim=2,
            bounds_lower=np.array([0.0, 0.0]),
            bounds_upper=np.array([10.0, 10.0]),
            names=["x", "y"],
        )

    def test_creation(self, config_space_2d):
        """Test decomposer creation."""
        decomposer = IRISDecomposer(config_space_2d)
        assert decomposer.max_iterations == 20

    def test_decompose_no_obstacles(self, config_space_2d):
        """Test decomposition without obstacles."""
        decomposer = IRISDecomposer(config_space_2d, max_iterations=5)
        seeds = [np.array([5.0, 5.0])]
        regions = decomposer.decompose(seeds)
        assert len(regions) >= 1

    def test_decompose_with_obstacles(self, config_space_2d):
        """Test decomposition with obstacles."""
        decomposer = IRISDecomposer(config_space_2d, max_iterations=5)
        obstacles = [
            SimpleBoxObstacle(
                lower=np.array([4.0, 4.0]), upper=np.array([6.0, 6.0])
            )
        ]
        seeds = [np.array([1.0, 1.0]), np.array([8.0, 8.0])]
        regions = decomposer.decompose(seeds, obstacles)
        assert len(regions) >= 1

    def test_seed_in_collision_skipped(self, config_space_2d):
        """Test that seeds inside obstacles are skipped."""
        decomposer = IRISDecomposer(config_space_2d, max_iterations=5)
        obstacles = [
            SimpleBoxObstacle(
                lower=np.array([0.0, 0.0]), upper=np.array([10.0, 10.0])
            )
        ]
        seeds = [np.array([5.0, 5.0])]  # Inside obstacle
        regions = decomposer.decompose(seeds, obstacles)
        assert len(regions) == 0

    def test_max_regions_limit(self, config_space_2d):
        """Test that max_regions is respected."""
        decomposer = IRISDecomposer(config_space_2d, max_iterations=5)
        seeds = [np.array([i, i]) for i in range(1, 10)]
        regions = decomposer.decompose(seeds, max_regions=3)
        assert len(regions) <= 3

    def test_regions_have_volume(self, config_space_2d):
        """Test that returned regions have positive volume."""
        decomposer = IRISDecomposer(config_space_2d, max_iterations=5)
        seeds = [np.array([5.0, 5.0])]
        regions = decomposer.decompose(seeds)
        for region in regions:
            assert region.volume() > 0
