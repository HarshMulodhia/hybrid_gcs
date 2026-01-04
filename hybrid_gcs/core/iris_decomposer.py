"""
IRIS Decomposition: Iterative Regional Inflation by Semidefinite Programming

Decomposes configuration space into convex regions for fast trajectory optimization.
Implements the IRIS algorithm for maximum volume ellipsoid growing.

References:
    - Deits & Tedrake (2015). Computing large convex regions of obstacle-free space
"""

from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class Ellipsoid:
    """
    Convex set represented as ellipsoid in configuration space.
    
    An ellipsoid E is defined as: E = {x : (x-c)^T Q (x-c) <= 1}
    where c is center and Q is the shape matrix (positive definite).
    
    Attributes:
        center: Center of ellipsoid [dim]
        shape_matrix: Positive definite shape matrix [dim, dim]
    """
    
    center: np.ndarray
    shape_matrix: np.ndarray
    
    def __post_init__(self):
        """Validate ellipsoid parameters."""
        self.center = np.asarray(self.center, dtype=np.float64)
        self.shape_matrix = np.asarray(self.shape_matrix, dtype=np.float64)
        self.dim = len(self.center)
        
        if self.shape_matrix.shape != (self.dim, self.dim):
            raise ValueError(
                f"shape_matrix shape {self.shape_matrix.shape} != ({self.dim}, {self.dim})")
        
        # Check positive definiteness via eigenvalues
        eigenvalues = np.linalg.eigvalsh(self.shape_matrix)
        if np.any(eigenvalues <= 1e-10):
            raise ValueError("shape_matrix must be positive definite")
    
    def contains(self, point: np.ndarray) -> bool:
        """
        Check if point is inside ellipsoid.
        
        Args:
            point: Configuration [dim]
        
        Returns:
            True if point inside, False otherwise
        """
        point = np.asarray(point, dtype=np.float64)
        diff = point - self.center
        quadratic_form = float(diff @ self.shape_matrix @ diff)
        return quadratic_form <= 1.0 + 1e-9  # Small numerical tolerance
    
    def volume(self) -> float:
        """
        Compute volume of ellipsoid.
        
        Volume = (π^(d/2) / Gamma(d/2 + 1)) * sqrt(det(Q^-1))
               = (π^(d/2) / Gamma(d/2 + 1)) / sqrt(det(Q))
        
        Returns:
            Volume in configuration space
        """
        import math
        det_Q = float(np.linalg.det(self.shape_matrix))
        unit_sphere_volume = (np.pi ** (self.dim / 2.0)) / math.gamma(self.dim / 2.0 + 1)
        return unit_sphere_volume / np.sqrt(det_Q)
    
    def __repr__(self) -> str:
        """String representation."""
        vol = self.volume()
        return f"Ellipsoid(dim={self.dim}, volume={vol:.4f})"


class IRISDecomposer:
    """
    IRIS Decomposition Algorithm.
    
    Decomposes configuration space into maximum volume ellipsoids avoiding obstacles.
    Uses iterative inflation of ellipsoids starting from seed points.
    
    Algorithm:
        1. Start with small ellipsoid at seed point
        2. Expand ellipsoid to maximum volume without hitting obstacles
        3. Repeat until convergence or max iterations
        4. Return final ellipsoid as convex region
    
    References:
        Deits & Tedrake (2015)
    """
    
    def __init__(self, 
                 config_space,
                 max_iterations: int = 20,
                 termination_threshold: float = 0.001,
                 verbose: bool = False):
        """
        Initialize IRIS decomposer.
        
        Args:
            config_space: ConfigSpace object
            max_iterations: Max IRIS iterations per seed point
            termination_threshold: Stop when volume growth < threshold
            verbose: Print iteration progress
        """
        self.config_space = config_space
        self.max_iterations = max_iterations
        self.termination_threshold = termination_threshold
        self.verbose = verbose
    
    def decompose(self,
                  seed_points: List[np.ndarray],
                  obstacles: Optional[List] = None,
                  max_regions: int = 50) -> List[Ellipsoid]:
        """
        Decompose space from multiple seed points.
        
        Args:
            seed_points: Initial points for region growth
            obstacles: List of obstacle objects with contains() method
            max_regions: Maximum number of regions to create
        
        Returns:
            List of Ellipsoid regions
        
        Example:
            >>> decomposer = IRISDecomposer(config_space)
            >>> seeds = [np.array([0., 0.]), np.array([5., 5.])]
            >>> regions = decomposer.decompose(seeds)
        """
        if obstacles is None:
            obstacles = []
        
        regions = []
        
        for i, seed in enumerate(seed_points):
            if len(regions) >= max_regions:
                break
            
            seed = np.asarray(seed, dtype=np.float64)
            
            # Check if seed is in collision
            if self._in_collision(seed, obstacles):
                if self.verbose:
                    print(f"Seed {i}: In collision, skipping")
                continue
            
            # Grow ellipsoid from seed
            ellipsoid = self._grow_ellipsoid(seed, obstacles)
            
            if ellipsoid is not None and ellipsoid.volume() > 1e-6:
                regions.append(ellipsoid)
                if self.verbose:
                    print(f"Seed {i}: Created region with volume {ellipsoid.volume():.6f}")
            else:
                if self.verbose:
                    print(f"Seed {i}: Failed to create region")
        
        return regions
    
    def _grow_ellipsoid(self, 
                       center: np.ndarray,
                       obstacles: List) -> Optional[Ellipsoid]:
        """
        Grow maximum volume ellipsoid from center.
        
        Args:
            center: Center point [dim]
            obstacles: List of obstacles
        
        Returns:
            Ellipsoid or None if growth fails
        """
        dim = len(center)
        
        # Initialize with small ellipsoid
        Q = np.eye(dim, dtype=np.float64)
        best_volume = 1.0
        
        for iteration in range(self.max_iterations):
            # Find minimum distance to obstacles
            min_dist = self._min_distance_to_obstacles(center, Q, obstacles)
            
            if min_dist <= 0:
                # Hit an obstacle, stop growing
                if self.verbose:
                    print(f"  Iteration {iteration}: Hit obstacle, stopping")
                break
            
            # Scale ellipsoid by distance (conservative scaling)
            scale_factor = 0.95 * min_dist
            Q_new = Q / (scale_factor ** 2)
            
            # Compute new volume
            try:
                volume_new = Ellipsoid(center, Q_new).volume()
            except (np.linalg.LinAlgError, ValueError):
                # Numerical issues, stop growing
                if self.verbose:
                    print(f"  Iteration {iteration}: Numerical issues")
                break
            
            # Check convergence
            growth_rate = (volume_new - best_volume) / (best_volume + 1e-12)
            
            if growth_rate < self.termination_threshold:
                if self.verbose:
                    print(f"  Iteration {iteration}: Converged (growth={growth_rate:.6f})")
                break
            
            Q = Q_new
            best_volume = volume_new
        
        # Return ellipsoid if it has reasonable volume
        if best_volume > 1e-6:
            return Ellipsoid(center, Q)
        return None
    
    def _min_distance_to_obstacles(self,
                                  center: np.ndarray,
                                  Q: np.ndarray,
                                  obstacles: List) -> float:
        """
        Find minimum distance from ellipsoid to obstacles.
        
        Simplified: Returns distance as if expanding uniformly.
        In practice, would solve optimization problem for exact distance.
        
        Args:
            center: Ellipsoid center [dim]
            Q: Shape matrix [dim, dim]
            obstacles: List of obstacles
        
        Returns:
            Minimum distance to any obstacle (or infinity if none)
        """
        if not obstacles:
            return float('inf')
        
        min_dist = float('inf')
        
        for obstacle in obstacles:
            # Sample points on ellipsoid and check distance
            # This is a simplified version - full implementation would
            # solve optimization problem
            
            # Use eigenvalues to estimate ellipsoid expansion
            eigenvalues = np.linalg.eigvalsh(Q)
            # Conservative estimate: minimum eigenvalue
            expansion = 1.0 / np.sqrt(np.min(eigenvalues))
            
            # Check if center approaches obstacle
            try:
                dist_to_obstacle = obstacle.signed_distance(center)
            except (AttributeError, TypeError):
                # Obstacle doesn't have signed_distance, use large value
                dist_to_obstacle = float('inf')
            
            # Distance to inflation threshold
            if dist_to_obstacle < float('inf'):
                min_dist = min(min_dist, dist_to_obstacle)
        
        return min_dist
    
    def _in_collision(self, 
                     point: np.ndarray,
                     obstacles: List) -> bool:
        """
        Check if point is in collision with any obstacle.
        
        Args:
            point: Configuration [dim]
            obstacles: List of obstacles with contains() method
        
        Returns:
            True if in collision, False otherwise
        """
        for obstacle in obstacles:
            try:
                if obstacle.contains(point):
                    return True
            except (AttributeError, TypeError):
                # Obstacle doesn't have contains method, skip
                pass
        
        return False


class SimpleBoxObstacle:
    """
    Simple box-shaped obstacle for testing.
    
    Defined by lower and upper bounds.
    """
    
    def __init__(self, lower: np.ndarray, upper: np.ndarray):
        """
        Initialize box obstacle.
        
        Args:
            lower: Lower corner [dim]
            upper: Upper corner [dim]
        """
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside box."""
        point = np.asarray(point, dtype=np.float64)
        return (np.all(point >= self.lower) and 
                np.all(point <= self.upper))
    
    def signed_distance(self, point: np.ndarray) -> float:
        """
        Compute signed distance to box.
        
        Positive if outside, negative if inside.
        """
        point = np.asarray(point, dtype=np.float64)
        
        # Distance along each dimension
        dist_lower = point - self.lower
        dist_upper = self.upper - point
        
        # Minimum distance to boundary in each dimension
        min_dist_per_dim = np.minimum(dist_lower, dist_upper)
        
        if np.any(min_dist_per_dim < 0):
            # Inside box
            return float(np.max(min_dist_per_dim))
        else:
            # Outside box
            return float(np.linalg.norm(np.maximum(0, -min_dist_per_dim)))
