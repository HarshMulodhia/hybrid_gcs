"""
Unit tests for Space-Time GCS module.

Tests SpaceTimeVertex, SpaceTimeEdge, and SpaceTimeGCS.
"""

import numpy as np
import pytest

from hybrid_gcs.multi_agent.st_gcs import SpaceTimeEdge, SpaceTimeGCS, SpaceTimeVertex


class TestSpaceTimeVertex:
    """Test SpaceTimeVertex class."""

    def test_creation(self):
        """Test basic vertex creation."""
        v = SpaceTimeVertex(region_id=0, time_step=1)
        assert v.region_id == 0
        assert v.time_step == 1
        assert v.convex_set is None

    def test_id_property(self):
        """Test id property returns correct tuple."""
        v = SpaceTimeVertex(region_id=2, time_step=3)
        assert v.id == (2, 3)

    def test_with_convex_set(self):
        """Test vertex with convex set data."""
        bounds = np.array([[0.0, 1.0], [0.0, 1.0]])
        v = SpaceTimeVertex(region_id=0, time_step=0, convex_set=bounds)
        np.testing.assert_array_equal(v.convex_set, bounds)


class TestSpaceTimeEdge:
    """Test SpaceTimeEdge class."""

    def test_spatial_edge(self):
        """Test spatial edge creation."""
        edge = SpaceTimeEdge(source=(0, 0), target=(1, 0), edge_type="spatial")
        assert edge.edge_type == "spatial"

    def test_temporal_edge(self):
        """Test temporal edge creation."""
        edge = SpaceTimeEdge(source=(0, 0), target=(0, 1), edge_type="temporal")
        assert edge.edge_type == "temporal"

    def test_spatio_temporal_edge(self):
        """Test spatio-temporal edge creation."""
        edge = SpaceTimeEdge(source=(0, 0), target=(1, 1), edge_type="spatio_temporal")
        assert edge.edge_type == "spatio_temporal"

    def test_invalid_edge_type(self):
        """Test invalid edge type raises error."""
        with pytest.raises(ValueError):
            SpaceTimeEdge(source=(0, 0), target=(1, 0), edge_type="invalid")


class TestSpaceTimeGCS:
    """Test SpaceTimeGCS class."""

    def test_creation(self):
        """Test basic graph creation."""
        stg = SpaceTimeGCS(num_regions=3, num_time_steps=5)
        assert stg.num_regions == 3
        assert stg.num_time_steps == 5

    def test_build_graph(self):
        """Test graph building from spatial adjacency."""
        stg = SpaceTimeGCS(num_regions=3, num_time_steps=3)
        adjacency = [(0, 1), (1, 2)]
        stg.build_graph(adjacency)

        # Expect 3*3 = 9 vertices
        assert stg.num_vertices() == 9
        # Edges: many types
        assert stg.num_edges() > 0

    def test_build_graph_vertices(self):
        """Test that all (region, time) vertices are created."""
        stg = SpaceTimeGCS(num_regions=2, num_time_steps=2)
        stg.build_graph([(0, 1)])

        vertices = stg.get_vertices()
        vertex_ids = {v.id for v in vertices}
        assert (0, 0) in vertex_ids
        assert (0, 1) in vertex_ids
        assert (1, 0) in vertex_ids
        assert (1, 1) in vertex_ids

    def test_build_graph_edge_types(self):
        """Test that all three edge types are present."""
        stg = SpaceTimeGCS(num_regions=2, num_time_steps=2)
        stg.build_graph([(0, 1)])

        edges = stg.get_edges()
        edge_types = {e.edge_type for e in edges}
        assert "spatial" in edge_types
        assert "temporal" in edge_types
        assert "spatio_temporal" in edge_types

    def test_reservation(self):
        """Test agent reservation system."""
        stg = SpaceTimeGCS(num_regions=3, num_time_steps=3)
        stg.build_graph([(0, 1), (1, 2)])

        # Reserve (1, 1) for agent 0
        stg.add_agent_reservation(agent_id=0, region_id=1, time_step=1)

        # Available for agent 0
        assert stg.is_available(1, 1, agent_id=0)

        # Not available for agent 1
        assert not stg.is_available(1, 1, agent_id=1)

        # Unreserved slot is available for anyone
        assert stg.is_available(0, 0, agent_id=1)

    def test_find_path_simple(self):
        """Test simple path finding."""
        stg = SpaceTimeGCS(num_regions=3, num_time_steps=5)
        stg.build_graph([(0, 1), (1, 2)])

        path = stg.find_path(start_region=0, goal_region=2, agent_id=0)
        assert path is not None
        # Path should start at region 0 and end at region 2
        assert path[0][0] == 0
        assert path[-1][0] == 2

    def test_find_path_with_reservation(self):
        """Test path finding avoids reserved slots."""
        stg = SpaceTimeGCS(num_regions=3, num_time_steps=5)
        stg.build_graph([(0, 1), (1, 2)])

        # Reserve middle region at time 0 for agent 1
        stg.add_agent_reservation(agent_id=1, region_id=1, time_step=0)

        # Agent 0 should still find a path (going through time first)
        path = stg.find_path(start_region=0, goal_region=2, agent_id=0)
        assert path is not None
        # The path should not go through (1, 0)
        assert (1, 0) not in path

    def test_find_path_no_path(self):
        """Test path finding when no path exists."""
        stg = SpaceTimeGCS(num_regions=2, num_time_steps=1)
        stg.build_graph([])  # No spatial edges

        # No spatial edges and only 1 time step - no path from 0 to 1
        path = stg.find_path(start_region=0, goal_region=1, agent_id=0)
        assert path is None

    def test_find_path_start_is_goal(self):
        """Test path when start equals goal."""
        stg = SpaceTimeGCS(num_regions=2, num_time_steps=3)
        stg.build_graph([(0, 1)])

        path = stg.find_path(start_region=0, goal_region=0, agent_id=0)
        assert path is not None
        assert len(path) == 1
        assert path[0] == (0, 0)

    def test_invalid_parameters(self):
        """Test invalid constructor parameters."""
        with pytest.raises(AssertionError):
            SpaceTimeGCS(num_regions=0, num_time_steps=3)
        with pytest.raises(AssertionError):
            SpaceTimeGCS(num_regions=3, num_time_steps=0)
