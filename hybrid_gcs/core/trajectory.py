"""
Trajectory Representation and Utilities

Provides trajectory abstractions supporting multiple parameterizations:
- Waypoints with cubic spline interpolation
- Bezier curves with control points
- Smooth trajectory operations

"""

from typing import List, Optional
import numpy as np
from scipy.interpolate import CubicSpline


class Trajectory:
    """
    Trajectory represented as waypoints with cubic spline interpolation.
    
    Provides smooth evaluation, derivatives, resampling, and length computation.
    Can be converted to/from Bezier parameterization.
    
    Attributes:
        waypoints: Trajectory waypoints [n_waypoints, dim]
        timestamps: Time at each waypoint [n_waypoints]
        spline: Cubic spline interpolator
    
    Examples:
        >>> waypoints = np.array([[0, 0], [1, 1], [2, 0]])
        >>> traj = Trajectory(waypoints)
        >>> config_at_t = traj.at_time(0.5)
        >>> length = traj.length()
    """
    
    def __init__(self, 
                 waypoints: np.ndarray,
                 timestamps: Optional[np.ndarray] = None):
        """
        Initialize trajectory from waypoints.
        
        Args:
            waypoints: Trajectory waypoints [n_waypoints, dim]
            timestamps: Time at each waypoint [n_waypoints], 
                       defaults to linspace(0, 1, n_waypoints)
        """
        self.waypoints = np.asarray(waypoints, dtype=np.float64)
        
        if self.waypoints.ndim != 2:
            raise ValueError(f"waypoints must be 2D, got shape {self.waypoints.shape}")
        
        n_waypoints, self.dim = self.waypoints.shape
        
        if n_waypoints < 2:
            raise ValueError(f"Need at least 2 waypoints, got {n_waypoints}")
        
        # Initialize timestamps
        if timestamps is None:
            self.timestamps = np.linspace(0, 1.0, n_waypoints)
        else:
            self.timestamps = np.asarray(timestamps, dtype=np.float64)
            if len(self.timestamps) != n_waypoints:
                raise ValueError(
                    f"timestamps length {len(self.timestamps)} != "
                    f"waypoints {n_waypoints}")
        
        # Build cubic spline for smooth interpolation
        self.spline = CubicSpline(self.timestamps, self.waypoints, 
                                  extrapolate='clip')
    
    def at_time(self, t: float) -> np.ndarray:
        """
        Get configuration at time t.
        
        Args:
            t: Time in [timestamps[0], timestamps[-1]]
        
        Returns:
            Configuration [dim]
        """
        return self.spline(t)
    
    def derivatives_at_time(self, t: float, n: int = 1) -> np.ndarray:
        """
        Get n-th derivative at time t.
        
        Args:
            t: Time parameter
            n: Derivative order (1=velocity, 2=acceleration, ...)
        
        Returns:
            n-th derivative [dim]
        """
        return self.spline(t, n)
    
    def velocity_at_time(self, t: float) -> np.ndarray:
        """
        Get velocity (first derivative) at time t.
        
        Args:
            t: Time parameter
        
        Returns:
            Velocity [dim]
        """
        return self.derivatives_at_time(t, n=1)
    
    def acceleration_at_time(self, t: float) -> np.ndarray:
        """
        Get acceleration (second derivative) at time t.
        
        Args:
            t: Time parameter
        
        Returns:
            Acceleration [dim]
        """
        return self.derivatives_at_time(t, n=2)
    
    def length(self) -> float:
        """
        Approximate trajectory length via piecewise linear approximation.
        
        Uses waypoints to estimate total arc length.
        
        Returns:
            Total trajectory length
        """
        diffs = np.diff(self.waypoints, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return float(np.sum(distances))
    
    def duration(self) -> float:
        """
        Get total trajectory time span.
        
        Returns:
            t_end - t_start
        """
        return float(self.timestamps[-1] - self.timestamps[0])
    
    def resample(self, n_waypoints: int) -> 'Trajectory':
        """
        Resample trajectory to different number of waypoints.
        
        Args:
            n_waypoints: New number of waypoints
        
        Returns:
            Resampled Trajectory object
        """
        if n_waypoints < 2:
            raise ValueError(f"Need at least 2 waypoints, got {n_waypoints}")
        
        new_times = np.linspace(
            self.timestamps[0], self.timestamps[-1], n_waypoints)
        new_waypoints = self.spline(new_times)
        
        return Trajectory(new_waypoints, new_times)
    
    def smooth(self, smoothing_factor: float = 0.1) -> 'Trajectory':
        """
        Apply smoothing to trajectory via moving average.
        
        Args:
            smoothing_factor: Smoothing window as fraction of trajectory length
        
        Returns:
            Smoothed Trajectory object
        """
        window_size = max(2, int(smoothing_factor * len(self.waypoints)))
        
        # Apply moving average per dimension
        smoothed_waypoints = np.zeros_like(self.waypoints)
        for i in range(self.dim):
            smoothed_waypoints[:, i] = np.convolve(
                self.waypoints[:, i],
                np.ones(window_size) / window_size,
                mode='same'
            )
        
        return Trajectory(smoothed_waypoints, self.timestamps)
    
    def reverse(self) -> 'Trajectory':
        """
        Get trajectory in reverse direction.
        
        Returns:
            Reversed Trajectory object
        """
        return Trajectory(self.waypoints[::-1], 
                         self.timestamps[::-1])
    
    def get_waypoints(self) -> np.ndarray:
        """Get waypoints array [n, dim]."""
        return self.waypoints.copy()
    
    def get_timestamps(self) -> np.ndarray:
        """Get timestamps array [n]."""
        return self.timestamps.copy()
    
    def __len__(self) -> int:
        """Get number of waypoints."""
        return len(self.waypoints)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Trajectory(waypoints={len(self)}, "
                f"dim={self.dim}, "
                f"duration={self.duration():.3f})")


class BezierTrajectory:
    """
    Trajectory parameterized by Bezier curve control points.
    
    Representation:
        x(t) = Σ_{i=0}^{d} C_i B_{i,d}(t),  t ∈ [0,1]
    
    where:
        - C_i: Control points (decision variables)
        - B_{i,d}(t) = C(d,i) t^i (1-t)^(d-i): Bernstein basis
        - d: Degree of curve
    
    Advantages:
        - Closed form derivatives
        - Convex hull property
        - Automatic endpoint continuity
        - Smooth trajectories
    
    References:
        - Hybrid-GCS-Theory.md Section 1.3
    
    Examples:
        >>> control_points = np.array([[0, 0], [1, 2], [2, 0]])
        >>> bezier = BezierTrajectory(control_points)
        >>> config_at_t = bezier.eval(0.5)
        >>> velocity = bezier.derivative(0.5)
        >>> traj = bezier.to_trajectory(n_samples=50)
    """
    
    def __init__(self, control_points: np.ndarray, 
                degree: Optional[int] = None):
        """
        Initialize Bezier trajectory.
        
        Args:
            control_points: Control points [n_points, dim]
            degree: Bezier degree (defaults to n_points - 1)
        """
        self.control_points = np.asarray(control_points, dtype=np.float64)
        
        if self.control_points.ndim != 2:
            raise ValueError(
                f"control_points must be 2D, got {self.control_points.ndim}D")
        
        self.n_points, self.dim = self.control_points.shape
        
        if self.n_points < 2:
            raise ValueError(f"Need at least 2 control points, got {self.n_points}")
        
        if degree is None:
            self.degree = self.n_points - 1
        else:
            self.degree = int(degree)
            if self.degree != self.n_points - 1:
                raise ValueError(
                    f"For {self.n_points} points, degree must be "
                    f"{self.n_points - 1}, got {self.degree}")
    
    def eval(self, t: float) -> np.ndarray:
        """
        Evaluate Bezier curve at parameter t.
        
        Args:
            t: Parameter in [0, 1]
        
        Returns:
            Point on curve [dim]
        """
        t = float(np.clip(t, 0.0, 1.0))
        result = np.zeros(self.dim, dtype=np.float64)
        
        for i in range(self.n_points):
            # Bernstein basis: B_{i,d}(t) = C(d,i) t^i (1-t)^(d-i)
            basis_coeff = float(self._binomial(self.degree, i))
            basis_val = basis_coeff * (t ** i) * ((1.0 - t) ** (self.degree - i))
            result += basis_val * self.control_points[i]
        
        return result
    
    def derivative(self, t: float, order: int = 1) -> np.ndarray:
        """
        Evaluate derivative of Bezier curve.
        
        For first derivative:
            dB/dt = d * Σ_i (P_{i+1} - P_i) B_{i,d-1}(t)
        
        Args:
            t: Parameter in [0, 1]
            order: Derivative order (1=velocity, 2=acceleration)
        
        Returns:
            Derivative [dim]
        """
        if order < 0:
            raise ValueError(f"order must be non-negative, got {order}")
        if order == 0:
            return self.eval(t)
        if order > self.degree:
            return np.zeros(self.dim, dtype=np.float64)
        
        # Compute first derivative and recurse for higher orders
        result = np.zeros(self.dim, dtype=np.float64)
        
        for i in range(self.n_points - 1):
            delta = self.control_points[i + 1] - self.control_points[i]
            basis_coeff = float(self._binomial(self.degree - 1, i))
            basis_val = basis_coeff * (t ** i) * ((1.0 - t) ** (self.degree - 1 - i))
            result += self.degree * basis_val * delta
        
        # Recurse for higher derivatives
        if order > 1:
            new_bezier = BezierTrajectory(
                np.diff(self.control_points, axis=0) * self.degree,
                degree=self.degree - 1
            )
            return new_bezier.derivative(t, order=order - 1)
        
        return result
    
    def to_trajectory(self, n_samples: int = 100) -> Trajectory:
        """
        Convert Bezier curve to waypoint trajectory.
        
        Args:
            n_samples: Number of samples for discretization
        
        Returns:
            Trajectory object
        """
        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")
        
        times = np.linspace(0, 1, n_samples)
        waypoints = np.array([self.eval(t) for t in times])
        
        return Trajectory(waypoints, times)
    
    @staticmethod
    def _binomial(n: int, k: int) -> int:
        """Compute binomial coefficient C(n,k)."""
        if k > n or k < 0:
            return 0
        if k == 0 or k == n:
            return 1
        # Use math.comb for efficiency (Python 3.8+)
        import math
        return math.comb(n, k)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"BezierTrajectory(degree={self.degree}, "
                f"dim={self.dim}, "
                f"n_points={self.n_points})")
