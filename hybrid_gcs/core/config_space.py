"""
Configuration Space Management

Defines state and action spaces with bounds, validation, and common operations.
Provides ConfigSpace abstraction for working with robot configuration/action spaces.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class ConfigSpace:
    """
    Represents a configuration/action space with bounds and properties.
    
    Provides methods for validation, sampling, interpolation, and common operations
    on configurations within the space.
    
    Attributes:
        dim: Dimension of configuration space
        bounds_lower: Lower bounds [dim]
        bounds_upper: Upper bounds [dim]
        names: Human-readable dimension names
        velocity_limits: Max velocities per dimension (optional)
        acceleration_limits: Max accelerations per dimension (optional)
    
    Examples:
        >>> # Create 2D workspace
        >>> space = ConfigSpace(
        ...     dim=2,
        ...     bounds_lower=np.array([0.0, 0.0]),
        ...     bounds_upper=np.array([10.0, 10.0]),
        ...     names=['x', 'y']
        ... )
        >>> point = space.random_sample()
        >>> is_valid = space.is_valid(point)
    """
    
    dim: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    names: List[str]
    velocity_limits: Optional[np.ndarray] = None
    acceleration_limits: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate configuration space after initialization."""
        # Type and shape validation
        self.bounds_lower = np.asarray(self.bounds_lower, dtype=np.float64)
        self.bounds_upper = np.asarray(self.bounds_upper, dtype=np.float64)
        
        if self.velocity_limits is not None:
            self.velocity_limits = np.asarray(self.velocity_limits, dtype=np.float64)
        if self.acceleration_limits is not None:
            self.acceleration_limits = np.asarray(self.acceleration_limits, dtype=np.float64)
        
        # Dimension consistency checks
        assert self.bounds_lower.shape == (self.dim,), \
            f"bounds_lower shape {self.bounds_lower.shape} != ({self.dim},)"
        assert self.bounds_upper.shape == (self.dim,), \
            f"bounds_upper shape {self.bounds_upper.shape} != ({self.dim},)"
        assert len(self.names) == self.dim, \
            f"names length {len(self.names)} != dimension {self.dim}"
        
        # Bound validity checks
        assert np.all(self.bounds_lower < self.bounds_upper), \
            "Lower bounds must be strictly less than upper bounds"
        
        # Optional constraint validation
        if self.velocity_limits is not None:
            assert self.velocity_limits.shape == (self.dim,), \
                f"velocity_limits shape {self.velocity_limits.shape} != ({self.dim},)"
            assert np.all(self.velocity_limits > 0), \
                "All velocity limits must be positive"
        
        if self.acceleration_limits is not None:
            assert self.acceleration_limits.shape == (self.dim,), \
                f"acceleration_limits shape {self.acceleration_limits.shape} != ({self.dim},)"
            assert np.all(self.acceleration_limits > 0), \
                "All acceleration limits must be positive"
    
    def is_valid(self, q: np.ndarray) -> bool:
        """
        Check if configuration is within bounds.
        
        Args:
            q: Configuration [dim]
        
        Returns:
            True if all elements within bounds, False otherwise
        """
        q = np.asarray(q, dtype=np.float64)
        return (q.shape == (self.dim,) and 
                np.all(q >= self.bounds_lower) and 
                np.all(q <= self.bounds_upper))
    
    def project(self, q: np.ndarray) -> np.ndarray:
        """
        Project configuration to valid space via clipping.
        
        Args:
            q: Configuration [dim]
        
        Returns:
            Projected configuration within bounds
        """
        q = np.asarray(q, dtype=np.float64)
        return np.clip(q, self.bounds_lower, self.bounds_upper)
    
    def random_sample(self) -> np.ndarray:
        """
        Sample random configuration uniformly from space.
        
        Returns:
            Random configuration [dim]
        """
        return np.random.uniform(self.bounds_lower, self.bounds_upper)
    
    def random_samples(self, n: int) -> np.ndarray:
        """
        Sample multiple random configurations.
        
        Args:
            n: Number of samples
        
        Returns:
            Random configurations [n, dim]
        """
        return np.random.uniform(
            self.bounds_lower, self.bounds_upper, size=(n, self.dim))
    
    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """
        Compute Euclidean distance between configurations.
        
        Args:
            q1: Configuration 1 [dim]
            q2: Configuration 2 [dim]
        
        Returns:
            L2 distance
        """
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        return float(np.linalg.norm(q1 - q2))
    
    def interpolate(self, q1: np.ndarray, q2: np.ndarray, 
                   t: float) -> np.ndarray:
        """
        Linear interpolation between configurations.
        
        Args:
            q1: Start configuration [dim]
            q2: End configuration [dim]
            t: Interpolation parameter in [0, 1]
        
        Returns:
            Interpolated configuration [dim]
        """
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        t = float(np.clip(t, 0.0, 1.0))
        return q1 + t * (q2 - q1)
    
    def difference(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """
        Compute difference vector from q1 to q2.
        
        Args:
            q1: Configuration 1 [dim]
            q2: Configuration 2 [dim]
        
        Returns:
            Difference vector [dim]
        """
        q1 = np.asarray(q1, dtype=np.float64)
        q2 = np.asarray(q2, dtype=np.float64)
        return q2 - q1
    
    def normalize_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        Normalize velocity to satisfy limits.
        
        Args:
            velocity: Velocity vector [dim]
        
        Returns:
            Normalized velocity respecting limits
        """
        if self.velocity_limits is None:
            return np.asarray(velocity, dtype=np.float64)
        
        velocity = np.asarray(velocity, dtype=np.float64)
        violation_factors = np.abs(velocity) / self.velocity_limits
        max_violation = np.max(violation_factors)
        
        if max_violation > 1.0:
            return velocity / max_violation
        return velocity
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get configuration space bounds.
        
        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        return self.bounds_lower.copy(), self.bounds_upper.copy()
    
    def get_range(self) -> np.ndarray:
        """
        Get range of each dimension.
        
        Returns:
            Range per dimension [dim]
        """
        return self.bounds_upper - self.bounds_lower
    
    def get_center(self) -> np.ndarray:
        """
        Get center point of configuration space.
        
        Returns:
            Center configuration [dim]
        """
        return (self.bounds_lower + self.bounds_upper) / 2.0
    
    @classmethod
    def from_robot(cls, 
                  num_dofs: int,
                  joint_lower_limits: np.ndarray,
                  joint_upper_limits: np.ndarray,
                  joint_names: List[str],
                  velocity_limits: Optional[np.ndarray] = None,
                  acceleration_limits: Optional[np.ndarray] = None) -> 'ConfigSpace':
        """
        Create ConfigSpace from robot specifications.
        
        Args:
            num_dofs: Number of degrees of freedom
            joint_lower_limits: Lower joint limits [num_dofs]
            joint_upper_limits: Upper joint limits [num_dofs]
            joint_names: Names of joints
            velocity_limits: Velocity limits per joint (optional)
            acceleration_limits: Acceleration limits per joint (optional)
        
        Returns:
            ConfigSpace instance
        
        Example:
            >>> space = ConfigSpace.from_robot(
            ...     num_dofs=7,
            ...     joint_lower_limits=np.array([-3.14]*7),
            ...     joint_upper_limits=np.array([3.14]*7),
            ...     joint_names=['shoulder_pan', 'shoulder_lift', ...]
            ... )
        """
        return cls(
            dim=num_dofs,
            bounds_lower=joint_lower_limits,
            bounds_upper=joint_upper_limits,
            names=joint_names,
            velocity_limits=velocity_limits,
            acceleration_limits=acceleration_limits
        )
    
    def __repr__(self) -> str:
        """String representation of configuration space."""
        bounds_str = f"[{self.bounds_lower[0]:.2f}~{self.bounds_upper[0]:.2f}]"
        if self.dim > 1:
            bounds_str += f" x ... x [{self.bounds_lower[-1]:.2f}~{self.bounds_upper[-1]:.2f}]"
        return f"ConfigSpace(dim={self.dim}, bounds={bounds_str})"
