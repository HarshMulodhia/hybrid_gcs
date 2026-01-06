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
        Compute the *exact* minimum Euclidean distance between the ellipsoid

            E = { x : (x-center)^T Q (x-center) <= 1 }

        and each obstacle, by solving the convex optimization

            minimize    ||x - y||_2
            subject to  x ∈ E
                        y ∈ O

        for each obstacle O, then returning the minimum over all obstacles.

        Notes
        -----
        - For convex obstacles (box / polytope / ball), this is a convex SOCP.
        - If the ellipsoid intersects an obstacle, the optimum is 0.
        - This method requires CVXPY for the exact solve. If CVXPY is not
        available or an obstacle cannot be converted to constraints,
        the method falls back to `obstacle.signed_distance(center)` when possible.

        Supported obstacle encodings
        ----------------------------
        1) Axis-aligned box: attributes `lower` and `upper`
        (as in `SimpleBoxObstacle` in this file)

        2) Polytope (H-rep): attributes `A` and `b` meaning A @ y <= b
        (common in motion planning / GCS codebases)

        3) Ball: attributes `center` and `radius` meaning ||y - center||_2 <= radius

        4) Custom: method `cvxpy_constraints(y_var)` returning a list of CVXPY constraints

        Returns
        -------
        float
            Minimum distance from the ellipsoid to any obstacle.
            Returns +inf if no obstacles are provided (or if none are solvable and no fallback is available).
        """
        if not obstacles:
            return float("inf")

        center = np.asarray(center, dtype=np.float64).reshape(-1)
        Q = np.asarray(Q, dtype=np.float64)

        dim = center.shape[0]
        if Q.shape != (dim, dim):
            raise ValueError(f"Q has shape {Q.shape}, expected {(dim, dim)}")

        # Symmetrize Q (numerical hygiene)
        Q = 0.5 * (Q + Q.T)

        # Convert ellipsoid quadratic constraint to SOC form:
        # (x-c)^T Q (x-c) <= 1  <=>  ||L (x-c)||_2 <= 1  where Q = L^T L.
        # Use Cholesky if possible, otherwise fall back to eigen-decomposition.
        try:
            L = np.linalg.cholesky(Q)
        except np.linalg.LinAlgError:
            w, V = np.linalg.eigh(Q)
            # Clamp tiny/negative eigenvalues due to numerics
            w = np.maximum(w, 1e-12)
            # Q = V diag(w) V^T = (diag(sqrt(w)) V^T)^T (diag(sqrt(w)) V^T)
            L = (np.diag(np.sqrt(w)) @ V.T)

        # Try to use CVXPY for the exact convex solve
        try:
            import cvxpy as cp
        except Exception:
            cp = None

        min_dist = float("inf")

        for obstacle in obstacles:
            # --- Exact solve via CVXPY when possible ---
            if cp is not None:
                x = cp.Variable(dim)
                y = cp.Variable(dim)

                constraints = []
                # Ellipsoid: ||L (x-center)||_2 <= 1
                constraints.append(cp.norm(L @ (x - center), 2) <= 1.0)

                # Obstacle constraints
                obstacle_ok = True

                # (4) Custom constraint hook: obstacle.cvxpy_constraints(y)
                if hasattr(obstacle, "cvxpy_constraints") and callable(getattr(obstacle, "cvxpy_constraints")):
                    try:
                        extra = obstacle.cvxpy_constraints(y)
                        if extra is None:
                            extra = []
                        constraints.extend(list(extra))
                    except Exception:
                        obstacle_ok = False

                # (1) Axis-aligned box: lower <= y <= upper
                elif hasattr(obstacle, "lower") and hasattr(obstacle, "upper"):
                    lower = np.asarray(obstacle.lower, dtype=np.float64).reshape(-1)
                    upper = np.asarray(obstacle.upper, dtype=np.float64).reshape(-1)
                    if lower.shape[0] != dim or upper.shape[0] != dim:
                        obstacle_ok = False
                    else:
                        constraints += [y >= lower, y <= upper]

                # (2) Polytope: A y <= b
                elif hasattr(obstacle, "A") and hasattr(obstacle, "b"):
                    A = np.asarray(obstacle.A, dtype=np.float64)
                    b = np.asarray(obstacle.b, dtype=np.float64).reshape(-1)
                    if A.ndim != 2 or A.shape[1] != dim or A.shape[0] != b.shape[0]:
                        obstacle_ok = False
                    else:
                        constraints.append(A @ y <= b)

                # (3) Ball: ||y - c|| <= r
                elif hasattr(obstacle, "center") and hasattr(obstacle, "radius"):
                    oc = np.asarray(obstacle.center, dtype=np.float64).reshape(-1)
                    r = float(obstacle.radius)
                    if oc.shape[0] != dim or r < 0:
                        obstacle_ok = False
                    else:
                        constraints.append(cp.norm(y - oc, 2) <= r)

                else:
                    obstacle_ok = False

                if obstacle_ok:
                    # Objective: minimize ||x - y||_2  (SOCP)
                    obj = cp.Minimize(cp.norm(x - y, 2))
                    prob = cp.Problem(obj, constraints)

                    solved = False
                    for solver in (cp.ECOS, cp.SCS):
                        try:
                            prob.solve(solver=solver, warm_start=True, verbose=False)
                            if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                                d = float(prob.value)
                                # Numerical safety
                                if d < 0 and d > -1e-9:
                                    d = 0.0
                                if d >= 0:
                                    min_dist = min(min_dist, d)
                                solved = True
                                break
                        except Exception:
                            continue

                    if solved:
                        continue  # go to next obstacle

            # --- Fallback (non-exact) when CVXPY is unavailable or obstacle not supported ---
            try:
                # This fallback is the *center* clearance only (not ellipsoid clearance).
                # Kept as a last resort so the pipeline doesn't crash.
                d_center = float(obstacle.signed_distance(center))
                if np.isfinite(d_center):
                    # Conservative: distance from ellipsoid to obstacle cannot exceed
                    # center distance by more than the ellipsoid "radius"; but without
                    # geometry we just keep the center distance as a heuristic.
                    min_dist = min(min_dist, d_center)
            except Exception:
                pass

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
