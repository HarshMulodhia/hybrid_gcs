# Core API Reference

## ConfigSpace

```python
from hybrid_gcs.core import ConfigSpace
```

Defines state and action space bounds with validation, sampling, and common operations.

### Constructor

```python
ConfigSpace(
    dim: int,                              # Number of dimensions
    bounds_lower: np.ndarray,              # Lower bounds [dim]
    bounds_upper: np.ndarray,              # Upper bounds [dim]
    names: List[str],                      # Dimension names
    velocity_limits: Optional[np.ndarray], # Max velocities (optional)
    acceleration_limits: Optional[np.ndarray]  # Max accelerations (optional)
)
```

### Key Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `is_valid` | `(q) → bool` | Check if config is within bounds |
| `project` | `(q) → ndarray` | Clip config to valid bounds |
| `random_sample` | `() → ndarray` | Sample uniformly from space |
| `random_samples` | `(n) → ndarray` | Sample n configs |
| `distance` | `(q1, q2) → float` | Euclidean distance |
| `interpolate` | `(q1, q2, t) → ndarray` | Linear interpolation, t ∈ [0, 1] |
| `from_robot` | `(classmethod)` | Create from robot joint specs |

---

## Trajectory

```python
from hybrid_gcs.core import Trajectory, BezierTrajectory
```

### Trajectory (waypoint-based)

Smooth trajectory through waypoints via cubic spline interpolation.

| Method | Signature | Description |
|--------|-----------|-------------|
| `at_time` | `(t) → ndarray` | Configuration at time t |
| `velocity_at_time` | `(t) → ndarray` | First derivative at t |
| `acceleration_at_time` | `(t) → ndarray` | Second derivative at t |
| `length` | `() → float` | Arc length |
| `duration` | `() → float` | Time span |
| `resample` | `(n) → Trajectory` | Resample to n waypoints |
| `smooth` | `(factor) → Trajectory` | Moving-average smoothing |
| `reverse` | `() → Trajectory` | Reverse direction |

### BezierTrajectory

Trajectory parameterized by Bézier curve control points.

| Method | Signature | Description |
|--------|-----------|-------------|
| `eval` | `(t) → ndarray` | Evaluate at parameter t ∈ [0, 1] |
| `derivative` | `(t, order) → ndarray` | n-th derivative |
| `to_trajectory` | `(n_samples) → Trajectory` | Convert to waypoints |

---

## IRISDecomposer

```python
from hybrid_gcs.core.iris_decomposer import IRISDecomposer, Ellipsoid, SimpleBoxObstacle
```

Decomposes configuration space into convex regions using the IRIS algorithm.

### IRISDecomposer

| Method | Signature | Description |
|--------|-----------|-------------|
| `decompose` | `(seeds, obstacles, max_regions) → List[Ellipsoid]` | Decompose space |

### Ellipsoid

Convex set `E = {x : (x-c)^T Q (x-c) ≤ 1}`.

| Method | Signature | Description |
|--------|-----------|-------------|
| `contains` | `(point) → bool` | Point-in-ellipsoid test |
| `volume` | `() → float` | Ellipsoid volume |

### SimpleBoxObstacle

Axis-aligned box obstacle.

| Method | Signature | Description |
|--------|-----------|-------------|
| `contains` | `(point) → bool` | Point-in-box test |
| `signed_distance` | `(point) → float` | Signed distance (positive outside, negative inside) |

---

## MICPSolver / GCSGraph

```python
from hybrid_gcs.core.micp_solver import MICPSolver, GCSGraph
```

### GCSGraph

Graph data structure for the convex set graph.

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_vertex` | `(id, **kwargs)` | Add vertex with metadata |
| `add_edge` | `(src, dst, **kwargs)` | Add directed edge |
| `num_vertices` | `() → int` | Vertex count |
| `num_edges` | `() → int` | Edge count |

### MICPSolver

Solves shortest path in GCS via MICP.

| Method | Signature | Description |
|--------|-----------|-------------|
| `solve` | `(start, goal, **kwargs) → Optional[Trajectory]` | Plan trajectory |

Supported solvers: `scs` (free), `mosek`, `gurobi`.
