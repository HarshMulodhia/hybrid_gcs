"""
Unit tests for MICP Solver module.

Tests GCSGraph and MICPSolver.
"""

import pytest
import numpy as np
from hybrid_gcs.core import ConfigSpace, Trajectory
from hybrid_gcs.core.micp_solver import GCSGraph, MICPSolver
from hybrid_gcs.core.iris_decomposer import Ellipsoid


class TestGCSGraph:
    """Test GCSGraph class."""

    def test_creation(self):
        """Test graph creation."""
        graph = GCSGraph()
        assert graph.num_vertices() == 0
        assert graph.num_edges() == 0

    def test_add_vertex(self):
        """Test adding vertices."""
        graph = GCSGraph()
        graph.add_vertex(0, label="start")
        graph.add_vertex(1, label="end")
        assert graph.num_vertices() == 2

    def test_add_edge(self):
        """Test adding edges."""
        graph = GCSGraph()
        graph.add_vertex(0)
        graph.add_vertex(1)
        graph.add_edge(0, 1, weight=1.0)
        assert graph.num_edges() == 1

    def test_vertex_data(self):
        """Test that vertex metadata is stored."""
        graph = GCSGraph()
        graph.add_vertex(0, label="start", cost=0.0)
        assert graph.vertex_data[0]["label"] == "start"
        assert graph.vertex_data[0]["cost"] == 0.0

    def test_edge_data(self):
        """Test that edge metadata is stored."""
        graph = GCSGraph()
        graph.add_vertex(0)
        graph.add_vertex(1)
        graph.add_edge(0, 1, weight=2.5)
        assert graph.edge_data[(0, 1)]["weight"] == 2.5

    def test_non_sequential_vertex_ids(self):
        """Test adding vertices with non-sequential IDs."""
        graph = GCSGraph()
        graph.add_vertex(5, label="v5")
        assert graph.num_vertices() == 1
        assert graph.vertex_data[5]["label"] == "v5"


class TestMICPSolver:
    """Test MICPSolver class."""

    @pytest.fixture
    def simple_setup(self):
        """Create simple graph and config space for testing."""
        config_space = ConfigSpace(
            dim=2,
            bounds_lower=np.array([0.0, 0.0]),
            bounds_upper=np.array([10.0, 10.0]),
            names=["x", "y"],
        )

        graph = GCSGraph()
        graph.add_vertex(0)
        graph.add_vertex(1)
        graph.add_edge(0, 1)
        graph.add_edge(1, 0)

        return graph, config_space

    def test_creation(self, simple_setup):
        """Test solver creation."""
        graph, config_space = simple_setup
        solver = MICPSolver(graph, config_space, solver_type="scs")
        assert solver.solver_type == "scs"

    def test_solve_basic(self, simple_setup):
        """Test basic trajectory solving."""
        graph, config_space = simple_setup
        solver = MICPSolver(graph, config_space, solver_type="scs")

        start = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        traj = solver.solve(start, goal)

        assert traj is not None
        assert isinstance(traj, Trajectory)
        assert traj.length() > 0

    def test_solve_start_equals_goal(self, simple_setup):
        """Test solving when start equals goal."""
        graph, config_space = simple_setup
        solver = MICPSolver(graph, config_space, solver_type="scs")

        point = np.array([5.0, 5.0])
        traj = solver.solve(point, point)

        assert traj is not None
        assert traj.length() < 1e-10

    def test_invalid_start(self, simple_setup):
        """Test that invalid start raises error."""
        graph, config_space = simple_setup
        solver = MICPSolver(graph, config_space, solver_type="scs")

        with pytest.raises(ValueError):
            solver.solve(np.array([-1.0, -1.0]), np.array([5.0, 5.0]))

    def test_invalid_goal(self, simple_setup):
        """Test that invalid goal raises error."""
        graph, config_space = simple_setup
        solver = MICPSolver(graph, config_space, solver_type="scs")

        with pytest.raises(ValueError):
            solver.solve(np.array([5.0, 5.0]), np.array([15.0, 15.0]))

    def test_invalid_solver_type(self, simple_setup):
        """Test that unknown solver type raises error."""
        graph, config_space = simple_setup
        with pytest.raises(ValueError):
            MICPSolver(graph, config_space, solver_type="invalid")

    def test_trajectory_endpoints(self, simple_setup):
        """Test that trajectory starts and ends at correct points."""
        graph, config_space = simple_setup
        solver = MICPSolver(graph, config_space, solver_type="scs")

        start = np.array([1.0, 1.0])
        goal = np.array([9.0, 9.0])
        traj = solver.solve(start, goal)

        assert np.allclose(traj.at_time(0.0), start, atol=1e-6)
        assert np.allclose(traj.at_time(1.0), goal, atol=1e-6)
