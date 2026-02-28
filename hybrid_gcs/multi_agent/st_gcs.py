"""
Space-Time Graphs of Convex Sets for Multi-Agent Collision Avoidance
(Section 4.2: ST-GCS Extension).

Extends GCS to the space-time domain for conflict-free multi-agent planning:
- Vertices: (Region, TimeInterval) pairs
  V = {(C_v, [t_k, t_{k+1}]) | C_v convex region, t_k discrete time}
- Edges:
  - Spatial: (C_v, t_k) -> (C_w, t_k) if C_v, C_w adjacent
  - Temporal: (C_v, t_k) -> (C_v, t_{k+1}) (wait in same region)
  - Spatio-temporal: (C_v, t_k) -> (C_w, t_{k+1})
- Constraint: Agents must not occupy the same (region, time) simultaneously

This module provides a simplified BFS-based path finder through the space-time
graph that respects agent reservations, demonstrating the core concept of the
full MICP formulation described in the paper.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class SpaceTimeVertex:
    """
    A vertex in the space-time graph representing a (region, time) pair.

    Each vertex corresponds to a convex region C_v at a discrete time step t_k,
    forming the pair (C_v, [t_k, t_{k+1}]).

    Attributes:
        region_id: Index of the convex region in the spatial graph.
        time_step: Discrete time step index.
        convex_set: Optional convex set representation (e.g., bounding box as
            ndarray of shape [spatial_dim, 2] with [lower, upper] bounds).
    """

    region_id: int
    time_step: int
    convex_set: Optional[np.ndarray] = None

    @property
    def id(self) -> Tuple[int, int]:
        """Return unique identifier as (region_id, time_step)."""
        return (self.region_id, self.time_step)


@dataclass
class SpaceTimeEdge:
    """
    An edge in the space-time graph connecting two (region, time) vertices.

    Edge types encode the transition semantics:
    - spatial: move between adjacent regions at the same time step
    - temporal: wait in the same region across consecutive time steps
    - spatio_temporal: move to an adjacent region at the next time step

    Attributes:
        source: Source vertex identifier (region_id, time_step).
        target: Target vertex identifier (region_id, time_step).
        edge_type: One of "spatial", "temporal", or "spatio_temporal".
    """

    source: Tuple[int, int]
    target: Tuple[int, int]
    edge_type: str

    def __post_init__(self) -> None:
        """Validate edge_type."""
        valid_types = {"spatial", "temporal", "spatio_temporal"}
        if self.edge_type not in valid_types:
            raise ValueError(f"Invalid edge_type '{self.edge_type}'. Must be one of {valid_types}")


class SpaceTimeGCS:
    """
    Space-Time Graph of Convex Sets for multi-agent collision avoidance.

    Constructs a space-time graph where vertices are (region, time) pairs and
    edges encode spatial, temporal, and spatio-temporal transitions. Agent
    reservations prevent multiple agents from occupying the same (region, time),
    enabling conflict-free path planning via BFS.

    Attributes:
        num_regions: Number of convex regions in the spatial graph.
        num_time_steps: Number of discrete time steps.
        spatial_dim: Dimensionality of the spatial workspace.
    """

    def __init__(self, num_regions: int, num_time_steps: int, spatial_dim: int = 3) -> None:
        """
        Initialize the space-time GCS.

        Args:
            num_regions: Number of convex regions in the spatial graph.
            num_time_steps: Number of discrete time steps.
            spatial_dim: Dimensionality of the spatial workspace.
        """
        assert num_regions > 0, "num_regions must be positive"
        assert num_time_steps > 0, "num_time_steps must be positive"
        assert spatial_dim > 0, "spatial_dim must be positive"

        self.num_regions = num_regions
        self.num_time_steps = num_time_steps
        self.spatial_dim = spatial_dim

        self._vertices: Dict[Tuple[int, int], SpaceTimeVertex] = {}
        self._edges: List[SpaceTimeEdge] = []
        self._adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        # Reservations: maps (region_id, time_step) -> agent_id
        self._reservations: Dict[Tuple[int, int], int] = {}

    def build_graph(self, adjacency: List[Tuple[int, int]]) -> None:
        """
        Build the space-time graph from a spatial adjacency list.

        Creates vertices for all (region, time) combinations and adds three
        types of edges:
        - Spatial: between adjacent regions at the same time step
        - Temporal: same region across consecutive time steps
        - Spatio-temporal: adjacent region at the next time step

        Args:
            adjacency: List of (region_i, region_j) pairs indicating spatial
                adjacency. Edges are treated as undirected.
        """
        self._vertices.clear()
        self._edges.clear()
        self._adjacency.clear()

        # Create vertices for all (region, time) pairs
        for r in range(self.num_regions):
            for t in range(self.num_time_steps):
                vertex = SpaceTimeVertex(region_id=r, time_step=t)
                self._vertices[vertex.id] = vertex
                self._adjacency[vertex.id] = []

        # Build undirected spatial neighbor lookup
        spatial_neighbors: Dict[int, Set[int]] = {r: set() for r in range(self.num_regions)}
        for r_i, r_j in adjacency:
            if 0 <= r_i < self.num_regions and 0 <= r_j < self.num_regions:
                spatial_neighbors[r_i].add(r_j)
                spatial_neighbors[r_j].add(r_i)

        # Create edges
        for r in range(self.num_regions):
            for t in range(self.num_time_steps):
                source = (r, t)

                # Spatial edges: move to adjacent region at same time
                for neighbor in spatial_neighbors[r]:
                    target = (neighbor, t)
                    edge = SpaceTimeEdge(source=source, target=target, edge_type="spatial")
                    self._edges.append(edge)
                    self._adjacency[source].append(target)

                # Temporal edges: wait in same region at next time step
                if t + 1 < self.num_time_steps:
                    target = (r, t + 1)
                    edge = SpaceTimeEdge(source=source, target=target, edge_type="temporal")
                    self._edges.append(edge)
                    self._adjacency[source].append(target)

                    # Spatio-temporal edges: move to adjacent region at next time step
                    for neighbor in spatial_neighbors[r]:
                        target = (neighbor, t + 1)
                        edge = SpaceTimeEdge(
                            source=source, target=target, edge_type="spatio_temporal"
                        )
                        self._edges.append(edge)
                        self._adjacency[source].append(target)

    def add_agent_reservation(self, agent_id: int, region_id: int, time_step: int) -> None:
        """
        Reserve a (region, time) slot for an agent.

        Prevents other agents from occupying the same (region, time_step),
        enforcing the collision avoidance constraint.

        Args:
            agent_id: Unique identifier for the agent.
            region_id: Region index to reserve.
            time_step: Time step index to reserve.
        """
        key = (region_id, time_step)
        self._reservations[key] = agent_id

    def is_available(self, region_id: int, time_step: int, agent_id: Optional[int] = None) -> bool:
        """
        Check if a (region, time) slot is available.

        A slot is available if it is unreserved, or if it is reserved by the
        querying agent itself.

        Args:
            region_id: Region index to check.
            time_step: Time step index to check.
            agent_id: If provided, the slot is considered available when
                reserved by this agent.

        Returns:
            True if the slot is available for use.
        """
        key = (region_id, time_step)
        if key not in self._reservations:
            return True
        if agent_id is not None and self._reservations[key] == agent_id:
            return True
        return False

    def find_path(
        self,
        start_region: int,
        goal_region: int,
        agent_id: int,
        start_time: int = 0,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Find a collision-free path through the space-time graph using BFS.

        Searches for the shortest (fewest-hops) path from (start_region,
        start_time) to (goal_region, any_time) that avoids all slots reserved
        by other agents. This is a simplified version of the full MICP
        formulation but demonstrates conflict-based search in ST-GCS.

        Args:
            start_region: Starting region index.
            goal_region: Goal region index.
            agent_id: Agent identifier (used to respect own reservations).
            start_time: Starting time step (default 0).

        Returns:
            List of (region_id, time_step) pairs forming the path, or None
            if no feasible path exists.
        """
        start = (start_region, start_time)
        if start not in self._vertices:
            return None

        if not self.is_available(start_region, start_time, agent_id):
            return None

        # BFS through the space-time graph
        queue: deque[Tuple[int, int]] = deque([start])
        visited: Set[Tuple[int, int]] = {start}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}

        while queue:
            current = queue.popleft()
            current_region, current_time = current

            # Goal reached at any time step
            if current_region == goal_region:
                path: List[Tuple[int, int]] = []
                node: Optional[Tuple[int, int]] = current
                while node is not None:
                    path.append(node)
                    node = parent[node]
                path.reverse()
                return path

            # Explore neighbors
            for neighbor in self._adjacency.get(current, []):
                if neighbor in visited:
                    continue
                n_region, n_time = neighbor
                if not self.is_available(n_region, n_time, agent_id):
                    continue
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

        return None

    def get_vertices(self) -> List[SpaceTimeVertex]:
        """
        Return all vertices in the space-time graph.

        Returns:
            List of SpaceTimeVertex instances.
        """
        return list(self._vertices.values())

    def get_edges(self) -> List[SpaceTimeEdge]:
        """
        Return all edges in the space-time graph.

        Returns:
            List of SpaceTimeEdge instances.
        """
        return list(self._edges)

    def num_vertices(self) -> int:
        """
        Return the number of vertices in the space-time graph.

        Returns:
            Total vertex count (num_regions * num_time_steps).
        """
        return len(self._vertices)

    def num_edges(self) -> int:
        """
        Return the number of edges in the space-time graph.

        Returns:
            Total edge count.
        """
        return len(self._edges)
