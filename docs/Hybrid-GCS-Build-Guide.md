# Hybrid-GCS Project: Implementation Guide & Professional Project Structure

**Version:** 2.0  
**Date:** January 3, 2026  
**Status:** Ready for Implementation  
**Standard:** PEP 8, Google Style Guide, Production-Grade

---

## Executive Summary

This guide provides a **step-by-step implementation pathway** for building the Hybrid-GCS system on top of the existing Motion-Planning-with-Graph-of-Convex-Sets repository. It includes:

- Professional project structure following industry best practices
- Detailed module specifications and interfaces
- Integration pathway from basic GCS to hybrid system
- Testing and validation framework
- Documentation standards

---

## Part 1: Project Structure

### 1.1 Directory Organization

```
hybrid_gcs/
│
├── README.md                          # Project overview, quick start
├── setup.py                           # Package installation
├── requirements.txt                   # Python dependencies
├── pyproject.toml                     # Modern Python packaging
├── CONTRIBUTING.md                    # Contributing guidelines
├── LICENSE                            # MIT License
├── .gitignore                         # Git configuration
│
├── hybrid_gcs/                        # Main package
│   ├── __init__.py                    # Package initialization
│   ├── __version__.py                 # Version control
│   │
│   ├── core/                          # GCS Core (extends basic_gcs)
│   │   ├── __init__.py
│   │   ├── config_space.py            # State/action space definitions
│   │   ├── convex_set.py              # Convex set abstractions
│   │   ├── iris_decomposer.py         # IRIS decomposition algorithm
│   │   ├── gcs_graph.py               # Graph data structures
│   │   ├── trajectory.py              # Trajectory representation
│   │   ├── bezier.py                  # Bezier curve utilities
│   │   ├── micp_solver.py             # MICP solver wrapper
│   │   ├── collision_checker.py       # Collision detection
│   │   └── kinematics.py              # Forward/inverse kinematics
│   │
│   ├── training/                      # Deep RL Training
│   │   ├── __init__.py
│   │   ├── policy_network.py          # Actor-Critic networks
│   │   ├── ppo_trainer.py             # PPO algorithm
│   │   ├── curriculum_scheduler.py    # Curriculum learning
│   │   ├── reward_shaper.py           # Reward engineering
│   │   ├── experience_buffer.py       # Replay memory
│   │   ├── losses.py                  # Loss functions
│   │   └── callbacks.py               # Training callbacks
│   │
│   ├── integration/                   # Hybrid Integration
│   │   ├── __init__.py
│   │   ├── feature_extractor.py       # State representation
│   │   ├── vision_encoder.py          # CNN for RGB-D
│   │   ├── hybrid_policy.py           # Action blending
│   │   ├── action_selector.py         # GCS/RL selection
│   │   ├── conflict_resolver.py       # Priority network
│   │   ├── safety_filter.py           # Real-time safety
│   │   └── corridor_planner.py        # Safe corridor generation
│   │
│   ├── environments/                  # Task Environments
│   │   ├── __init__.py
│   │   ├── base_env.py                # Abstract base class
│   │   ├── ycb_grasping_env.py        # YCB grasping task
│   │   ├── drone_navigation_env.py    # Single/multi-drone task
│   │   ├── manipulation_env.py        # Complex manipulation
│   │   ├── dynamics.py                # Physics models
│   │   └── sensors.py                 # Sensor simulation
│   │
│   ├── evaluation/                    # Performance Analysis
│   │   ├── __init__.py
│   │   ├── metrics.py                 # Metric definitions
│   │   ├── analyzer.py                # Trajectory analysis
│   │   ├── visualizer.py              # Plotting utilities
│   │   ├── benchmark.py               # Baseline comparisons
│   │   └── logger.py                  # Experiment tracking
│   │
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── config.py                  # Configuration management
│   │   ├── logging.py                 # Logging setup
│   │   ├── validators.py              # Type checking
│   │   ├── data_structures.py         # Common structures
│   │   └── profiler.py                # Performance profiling
│   │
│   └── cli/                           # Command-line Interface
│       ├── __init__.py
│       ├── train.py                   # Training CLI
│       ├── evaluate.py                # Evaluation CLI
│       ├── visualize.py               # Visualization CLI
│       └── deploy.py                  # Deployment CLI
│
├── tests/                             # Unit & Integration Tests
│   ├── __init__.py
│   ├── conftest.py                    # Pytest configuration
│   ├── test_core/
│   │   ├── test_config_space.py
│   │   ├── test_iris_decomposer.py
│   │   ├── test_micp_solver.py
│   │   └── test_trajectory.py
│   ├── test_training/
│   │   ├── test_policy_network.py
│   │   ├── test_ppo_trainer.py
│   │   └── test_reward_shaper.py
│   ├── test_integration/
│   │   ├── test_hybrid_policy.py
│   │   ├── test_action_blending.py
│   │   └── test_safety_filter.py
│   ├── test_environments/
│   │   ├── test_ycb_env.py
│   │   ├── test_drone_env.py
│   │   └── test_manipulation_env.py
│   └── test_evaluation/
│       ├── test_metrics.py
│       └── test_analyzer.py
│
├── examples/                          # Example Scripts
│   ├── 01_simple_navigation.py        # Basic GCS planning
│   ├── 02_rl_training.py              # Pure RL training
│   ├── 03_hybrid_grasping.py          # Hybrid grasping example
│   ├── 04_multi_agent_navigation.py   # Multi-drone example
│   ├── 05_manipulation_sequence.py    # Complex task example
│   └── configs/
│       ├── gcs_config.yaml
│       ├── rl_config.yaml
│       ├── hybrid_config.yaml
│       └── env_config.yaml
│
├── docs/                              # Documentation
│   ├── api/                           # API reference
│   │   ├── core.md
│   │   ├── training.md
│   │   ├── integration.md
│   │   └── environments.md
│   ├── tutorials/                     # Step-by-step guides
│   │   ├── getting_started.md
│   │   ├── custom_environment.md
│   │   ├── training_policy.md
│   │   └── deployment.md
│   ├── architecture.md                # System design
│   ├── algorithms.md                  # Algorithm descriptions
│   └── troubleshooting.md             # Common issues
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 01_gcs_planning_demo.ipynb
│   ├── 02_policy_training.ipynb
│   ├── 03_hybrid_analysis.ipynb
│   ├── 04_visualization.ipynb
│   └── 05_benchmark_results.ipynb
│
├── data/                              # Data & Assets
│   ├── objects/
│   │   ├── ycb/                       # YCB object models
│   │   └── custom/
│   ├── scenes/
│   │   ├── grasping_scenes.json
│   │   └── navigation_scenes.json
│   ├── pretrained/
│   │   ├── policy_weights.pth
│   │   └── critic_weights.pth
│   └── results/
│       ├── experiments/
│       └── benchmarks/
│
├── docker/                            # Containerization
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .dockerignore
│
└── scripts/                           # Utility Scripts
    ├── setup_environment.sh           # Environment setup
    ├── download_data.py               # Data downloader
    ├── run_tests.sh                   # Test runner
    └── generate_docs.sh               # Documentation builder
```

### 1.2 Python Package Configuration

**setup.py:**
```python
from setuptools import setup, find_packages

setup(
    name='hybrid-gcs',
    version='0.1.0',
    description='Hybrid planning combining GCS and Deep RL for robotics',
    author='Your Team',
    author_email='team@example.com',
    url='https://github.com/yourusername/hybrid-gcs',
    license='MIT',
    
    packages=find_packages(),
    python_requires='>=3.9',
    
    install_requires=[
        'numpy>=1.21.0',
        'scipy>=1.7.0',
        'torch>=1.10.0',
        'pybullet>=3.1.0',
        'pydantic>=1.8.0',
        'PyYAML>=5.4.0',
        'matplotlib>=3.4.0',
        'tensorboard>=2.8.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.7b0',
            'pylint>=2.9.0',
            'mypy>=0.910',
        ],
        'drake': [
            'drake>=1.2.0',  # Optional Drake for advanced GCS
        ],
        'gpu': [
            'torch-cuda>=1.10.0',
        ],
    },
    
    entry_points={
        'console_scripts': [
            'hybrid-gcs-train=hybrid_gcs.cli.train:main',
            'hybrid-gcs-eval=hybrid_gcs.cli.evaluate:main',
            'hybrid-gcs-vis=hybrid_gcs.cli.visualize:main',
        ],
    },
)
```

---

## Part 2: Core Module Architecture

### 2.1 Core Module: Configuration Space

**File:** `hybrid_gcs/core/config_space.py`

```python
"""
Configuration Space Management

Defines state and action spaces with bounds and constraints.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, List
import numpy as np


@dataclass
class ConfigSpace:
    """
    Represents a configuration space with bounds and properties.
    
    Attributes:
        dim: Dimension of configuration space
        bounds_lower: Lower bounds [dim]
        bounds_upper: Upper bounds [dim]
        names: Dimension names for debugging
        velocity_limits: Max velocities per dimension
        acceleration_limits: Max accelerations per dimension
    """
    
    dim: int
    bounds_lower: np.ndarray
    bounds_upper: np.ndarray
    names: List[str]
    velocity_limits: Optional[np.ndarray] = None
    acceleration_limits: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate configuration space."""
        assert self.bounds_lower.shape == (self.dim,)
        assert self.bounds_upper.shape == (self.dim,)
        assert len(self.names) == self.dim
        assert np.all(self.bounds_lower < self.bounds_upper)
    
    def is_valid(self, q: np.ndarray) -> bool:
        """Check if configuration is within bounds."""
        return (np.all(q >= self.bounds_lower) and 
                np.all(q <= self.bounds_upper))
    
    def project(self, q: np.ndarray) -> np.ndarray:
        """Project configuration to valid space."""
        return np.clip(q, self.bounds_lower, self.bounds_upper)
    
    def random_sample(self) -> np.ndarray:
        """Sample random valid configuration."""
        return np.random.uniform(self.bounds_lower, self.bounds_upper)
    
    def distance(self, q1: np.ndarray, q2: np.ndarray) -> float:
        """Euclidean distance between configurations."""
        return np.linalg.norm(q1 - q2)
    
    def interpolate(self, q1: np.ndarray, q2: np.ndarray, 
                   t: float) -> np.ndarray:
        """Linear interpolation between configurations."""
        return q1 + t * (q2 - q1)
    
    @classmethod
    def from_robot(cls, robot_model) -> 'ConfigSpace':
        """Create from robot URDF model."""
        dim = robot_model.num_dofs
        bounds_lower = robot_model.joint_limits[:, 0]
        bounds_upper = robot_model.joint_limits[:, 1]
        names = robot_model.joint_names
        velocity_limits = robot_model.velocity_limits
        
        return cls(
            dim=dim,
            bounds_lower=bounds_lower,
            bounds_upper=bounds_upper,
            names=names,
            velocity_limits=velocity_limits,
        )
```

### 2.2 Core Module: Trajectory Representation

**File:** `hybrid_gcs/core/trajectory.py`

```python
"""
Trajectory representations and utilities.

Supports multiple parameterizations: Bezier curves, waypoints, splines.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from scipy.interpolate import CubicSpline


@dataclass
class Trajectory:
    """
    Represents a trajectory as a sequence of waypoints.
    
    Can be converted to different representations:
    - Waypoints: List of configurations
    - Spline: Smooth curve through waypoints
    - Bezier: Bezier curve control points
    """
    
    waypoints: np.ndarray  # [N, dim]
    timestamps: Optional[np.ndarray] = None  # [N]
    spline: Optional[CubicSpline] = None
    
    def __post_init__(self):
        """Initialize trajectory."""
        if self.timestamps is None:
            self.timestamps = np.linspace(0, 1, len(self.waypoints))
        
        # Build cubic spline for smooth interpolation
        self.spline = CubicSpline(self.timestamps, self.waypoints)
    
    def at_time(self, t: float) -> np.ndarray:
        """Get configuration at time t."""
        return self.spline(t)
    
    def derivatives_at_time(self, t: float, n: int = 1) -> np.ndarray:
        """Get n-th derivative at time t."""
        return self.spline(t, n)
    
    def length(self) -> float:
        """Approximate trajectory length."""
        diffs = np.diff(self.waypoints, axis=0)
        distances = np.linalg.norm(diffs, axis=1)
        return np.sum(distances)
    
    def duration(self) -> float:
        """Total trajectory duration."""
        return self.timestamps[-1] - self.timestamps[0]
    
    def resample(self, n_waypoints: int) -> 'Trajectory':
        """Resample trajectory to n waypoints."""
        new_times = np.linspace(
            self.timestamps[0], self.timestamps[-1], n_waypoints)
        new_waypoints = self.spline(new_times)
        return Trajectory(new_waypoints, new_times)
    
    def smooth(self, smoothing_factor: float = 0.1) -> 'Trajectory':
        """Apply smoothing to trajectory."""
        # Simple moving average smoothing
        window_size = int(smoothing_factor * len(self.waypoints))
        if window_size < 2:
            return self
        
        smoothed = np.convolve(
            self.waypoints.T, 
            np.ones(window_size) / window_size, 
            mode='same'
        ).T
        
        return Trajectory(smoothed, self.timestamps)


class BezierTrajectory:
    """
    Trajectory parameterized by Bezier curve control points.
    
    x(t) = Σ_i C_i B_{i,d}(t)
    
    where B_{i,d} are Bernstein basis functions.
    """
    
    def __init__(self, control_points: np.ndarray, degree: Optional[int] = None):
        """
        Args:
            control_points: [n_points, dim]
            degree: Bezier degree (default: n_points - 1)
        """
        self.control_points = control_points
        self.n_points = len(control_points)
        self.dim = control_points.shape[1]
        self.degree = degree or (self.n_points - 1)
        
        assert self.degree == self.n_points - 1
    
    def eval(self, t: float) -> np.ndarray:
        """Evaluate Bezier curve at parameter t ∈ [0,1]."""
        result = np.zeros(self.dim)
        
        for i in range(self.n_points):
            # Bernstein basis: B_{i,d}(t) = C(d,i) t^i (1-t)^(d-i)
            basis = (np.math.comb(self.degree, i) * 
                    (t ** i) * ((1 - t) ** (self.degree - i)))
            result += basis * self.control_points[i]
        
        return result
    
    def derivative(self, t: float) -> np.ndarray:
        """Evaluate first derivative."""
        # dB/dt = d * Σ_i (P_{i+1} - P_i) B_{i,d-1}(t)
        result = np.zeros(self.dim)
        
        for i in range(self.n_points - 1):
            delta = self.control_points[i + 1] - self.control_points[i]
            basis = (np.math.comb(self.degree - 1, i) * 
                    (t ** i) * ((1 - t) ** (self.degree - 1 - i)))
            result += self.degree * basis * delta
        
        return result
    
    def to_trajectory(self, n_samples: int = 100) -> Trajectory:
        """Convert to waypoint trajectory."""
        times = np.linspace(0, 1, n_samples)
        waypoints = np.array([self.eval(t) for t in times])
        return Trajectory(waypoints, times)
```

### 2.3 Core Module: IRIS Decomposition

**File:** `hybrid_gcs/core/iris_decomposer.py`

**Key Functions:**

```python
"""
IRIS Decomposition: Iterative Regional Inflation by Semidefinite programming

Reference: Deits & Tedrake (2015)
"""

from typing import List, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


class IRISDecomposer:
    """
    Decomposes configuration space into convex regions using IRIS.
    """
    
    def __init__(self, 
                 config_space,
                 max_iterations: int = 20,
                 termination_threshold: float = 0.001):
        """
        Args:
            config_space: ConfigSpace object
            max_iterations: Max IRIS iterations per seed
            termination_threshold: Stop when growth rate < threshold
        """
        self.config_space = config_space
        self.max_iterations = max_iterations
        self.termination_threshold = termination_threshold
    
    def decompose(self, 
                  obstacles: List,
                  seed_points: List[np.ndarray],
                  max_regions: int = 50) -> 'GCSGraph':
        """
        Decompose space from multiple seed points.
        
        Args:
            obstacles: List of obstacle objects
            seed_points: Initial points for region growth
            max_regions: Maximum number of regions to create
        
        Returns:
            GCSGraph with convex regions as vertices
        """
        regions = []
        
        for seed in seed_points[:max_regions]:
            if self._in_collision(seed, obstacles):
                continue
            
            # Grow ellipsoid from seed
            region = self._grow_ellipsoid(seed, obstacles)
            if region is not None:
                regions.append(region)
        
        # Build adjacency graph
        graph = self._build_graph(regions)
        return graph
    
    def _grow_ellipsoid(self, center: np.ndarray, obstacles) -> Optional['Ellipsoid']:
        """Grow maximal ellipsoid from center."""
        dim = len(center)
        
        # Initialize small ellipsoid
        A = np.eye(dim)
        best_volume = np.linalg.det(A)
        
        for iteration in range(self.max_iterations):
            # Find minimum distance to obstacles
            min_dist = float('inf')
            
            for obstacle in obstacles:
                dist = self._distance_to_obstacle(center, A, obstacle)
                min_dist = min(min_dist, dist)
            
            if min_dist <= 0:
                break  # Hit obstacle
            
            # Scale ellipsoid
            scale = 0.95 * min_dist  # Conservative scaling
            A_new = np.diag([scale] * dim)
            
            volume_new = np.linalg.det(A_new)
            
            # Check convergence
            if (volume_new - best_volume) / best_volume < self.termination_threshold:
                break
            
            A = A_new
            best_volume = volume_new
        
        if best_volume > 0.001:
            return Ellipsoid(center, A)
        return None
    
    def _distance_to_obstacle(self, center: np.ndarray, 
                             A: np.ndarray, obstacle) -> float:
        """Minimum distance from ellipsoid center to obstacle."""
        # Implementation depends on obstacle type
        return obstacle.signed_distance(center)
    
    def _in_collision(self, q: np.ndarray, obstacles: List) -> bool:
        """Check if point is in collision."""
        for obstacle in obstacles:
            if obstacle.contains(q):
                return True
        return False
    
    def _build_graph(self, regions: List['Ellipsoid']) -> 'GCSGraph':
        """Build adjacency graph from regions."""
        graph = GCSGraph()
        
        # Add vertices
        for i, region in enumerate(regions):
            graph.add_vertex(i, region=region)
        
        # Add edges for overlapping/adjacent regions
        for i in range(len(regions)):
            for j in range(i + 1, len(regions)):
                if self._regions_adjacent(regions[i], regions[j]):
                    graph.add_edge(i, j)
        
        return graph


class Ellipsoid:
    """Represents an ellipsoid in configuration space."""
    
    def __init__(self, center: np.ndarray, shape_matrix: np.ndarray):
        """
        Args:
            center: Center of ellipsoid
            shape_matrix: Positive definite matrix defining shape
        """
        self.center = center
        self.shape_matrix = shape_matrix
        self.dim = len(center)
    
    def contains(self, point: np.ndarray) -> bool:
        """Check if point is inside ellipsoid."""
        diff = point - self.center
        quadratic = diff @ np.linalg.inv(self.shape_matrix) @ diff
        return quadratic <= 1.0
    
    def volume(self) -> float:
        """Volume of ellipsoid."""
        det = np.linalg.det(self.shape_matrix)
        unit_sphere_volume = (np.pi ** (self.dim / 2)) / np.math.gamma(self.dim / 2 + 1)
        return unit_sphere_volume * np.sqrt(det)
```

### 2.4 Core Module: MICP Solver Wrapper

**File:** `hybrid_gcs/core/micp_solver.py`

**Key Functions:**

```python
"""
Mixed-Integer Convex Program (MICP) Solver Wrapper

Interfaces with Mosek/Gurobi to solve GCS trajectory planning problems.
"""

from typing import Optional, Dict, Any
import numpy as np


class MICPSolver:
    """
    Solves shortest path in Graph of Convex Sets via MICP.
    """
    
    def __init__(self,
                 graph: 'GCSGraph',
                 solver_type: str = 'mosek',
                 time_limit: float = 30.0,
                 verbose: bool = False):
        """
        Args:
            graph: GCS graph with convex regions
            solver_type: 'mosek' | 'gurobi'
            time_limit: Maximum solver time (seconds)
            verbose: Print solver output
        """
        self.graph = graph
        self.solver_type = solver_type
        self.time_limit = time_limit
        self.verbose = verbose
        
        self._import_solver()
    
    def _import_solver(self):
        """Import solver library."""
        if self.solver_type == 'mosek':
            try:
                import mosek
                self.mosek = mosek
            except ImportError:
                raise ImportError("Install mosek: pip install mosek")
        elif self.solver_type == 'gurobi':
            try:
                import gurobipy
                self.gurobi = gurobipy
            except ImportError:
                raise ImportError("Install gurobi: pip install gurobipy")
    
    def solve(self,
              start: np.ndarray,
              goal: np.ndarray,
              **kwargs) -> Optional['Trajectory']:
        """
        Solve for collision-free trajectory from start to goal.
        
        Args:
            start: Start configuration
            goal: Goal configuration
            **kwargs: Additional solver options
        
        Returns:
            Trajectory object or None if infeasible
        """
        
        # Build MICP problem
        problem = self._build_problem(start, goal, **kwargs)
        
        # Solve
        if self.solver_type == 'mosek':
            solution = self._solve_mosek(problem)
        elif self.solver_type == 'gurobi':
            solution = self._solve_gurobi(problem)
        else:
            raise ValueError(f"Unknown solver: {self.solver_type}")
        
        if solution is None:
            return None
        
        # Extract trajectory
        trajectory = self._extract_trajectory(solution)
        return trajectory
    
    def _build_problem(self, start: np.ndarray, goal: np.ndarray,
                      **kwargs) -> Dict[str, Any]:
        """Build MICP problem formulation."""
        problem = {
            'start': start,
            'goal': goal,
            'graph': self.graph,
            'regions': len(self.graph.vertices),
            'options': kwargs,
        }
        return problem
    
    def _solve_mosek(self, problem: Dict) -> Optional[Dict]:
        """Solve using Mosek."""
        # Implementation with Mosek MICP solver
        # Returns solution dict with optimal trajectory or None
        pass
    
    def _solve_gurobi(self, problem: Dict) -> Optional[Dict]:
        """Solve using Gurobi."""
        # Implementation with Gurobi MICP solver
        pass
    
    def _extract_trajectory(self, solution: Dict) -> 'Trajectory':
        """Extract trajectory from solver solution."""
        waypoints = solution['trajectory']
        return Trajectory(waypoints)
```

---

## Part 3: Training Module Architecture

### 3.1 Training Module: Policy Networks

**File:** `hybrid_gcs/training/policy_network.py`

```python
"""
Policy networks for actor-critic learning.

Includes actor (policy) and critic (value) networks.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, Optional


class PolicyNetwork(nn.Module):
    """Actor-Critic network for continuous control."""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: Tuple[int, ...] = (256, 256),
                 activation: str = 'relu',
                 use_layer_norm: bool = True):
        """
        Args:
            state_dim: Input state dimension
            action_dim: Output action dimension
            hidden_dims: Hidden layer sizes
            activation: Activation function type
            use_layer_norm: Use layer normalization
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        if activation == 'relu':
            activation_fn = nn.ReLU
        elif activation == 'tanh':
            activation_fn = nn.Tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Feature extraction (shared between actor and critic)
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_fn())
            prev_dim = hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Actor head: Policy π(a|s)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_logstd = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head: Value V(s)
        self.critic = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass returning action distribution and value.
        
        Args:
            state: [batch_size, state_dim]
        
        Returns:
            dist: Normal distribution over actions
            value: State value estimate [batch_size, 1]
        """
        features = self.feature_net(state)
        
        # Actor: Gaussian policy
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_logstd)
        dist = Normal(mean, std)
        
        # Critic: Value estimate
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        dist, _ = self.forward(state)
        action = dist.rsample()  # Reparameterization trick
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        return action, log_prob
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """Get value without sampling action."""
        with torch.no_grad():
            _, value = self.forward(state)
        return value


class CNNEncoder(nn.Module):
    """CNN for encoding RGB-D images to feature vectors."""
    
    def __init__(self, 
                 input_channels: int = 4,  # RGB-D
                 output_dim: int = 256):
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size for 84×84 input
        self.fc_input_size = 64 * 7 * 7
        
        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.fc2 = nn.Linear(512, output_dim)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, channels, height, width]
        
        Returns:
            features: [batch, output_dim]
        """
        x = torch.relu(self.conv1(images))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
```

### 3.2 Training Module: PPO Trainer

**File:** `hybrid_gcs/training/ppo_trainer.py`

**Working:**

The PPO trainer implements the Proximal Policy Optimization algorithm with the following key steps:

1. **Trajectory Collection:** Sample trajectories from environment with current policy
2. **Advantage Computation:** Use GAE (Generalized Advantage Estimation) for smooth advantage estimates
3. **Policy Update:** Multiple epochs of gradient updates with clipped objectives
4. **Value Update:** Regress value network on actual returns
5. **Entropy Regularization:** Encourage exploration through entropy bonus

```python
"""
Proximal Policy Optimization (PPO) trainer.

Implements on-policy RL with trust region via clipping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import List, Dict, Tuple


class PPOTrainer:
    """PPO training loop."""
    
    def __init__(self,
                 policy: nn.Module,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_ratio: float = 0.2,
                 entropy_coeff: float = 0.01,
                 value_loss_coeff: float = 0.5,
                 max_grad_norm: float = 0.5,
                 epochs_per_batch: int = 4,
                 batch_size: int = 64,
                 device: str = 'cpu'):
        """
        Args:
            policy: PolicyNetwork module
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE parameter
            clip_ratio: PPO clipping parameter (ε)
            entropy_coeff: Entropy regularization coefficient
            value_loss_coeff: Value function loss weight
            max_grad_norm: Gradient clipping norm
            epochs_per_batch: Training epochs per batch
            batch_size: Minibatch size
            device: 'cpu' or 'cuda'
        """
        self.policy = policy.to(device)
        self.device = device
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.max_grad_norm = max_grad_norm
        self.epochs_per_batch = epochs_per_batch
        self.batch_size = batch_size
        
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    
    def compute_gae(self,
                   rewards: np.ndarray,
                   values: np.ndarray,
                   dones: np.ndarray,
                   next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards: [T] trajectory rewards
            values: [T] value estimates
            dones: [T] done flags
            next_value: Bootstrap value at trajectory end
        
        Returns:
            advantages: [T] advantage estimates
            returns: [T] return estimates
        """
        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = 0
        
        values = np.append(values, next_value)
        
        for t in reversed(range(len(rewards))):
            delta = (rewards[t] + 
                    self.gamma * values[t + 1] * (1 - dones[t]) -
                    values[t])
            gae = (delta + 
                  self.gamma * self.gae_lambda * (1 - dones[t]) * gae)
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = ((advantages - advantages.mean()) / 
                     (advantages.std() + 1e-8))
        
        return advantages, returns
    
    def update_policy(self,
                     trajectories: List[Dict]) -> Dict[str, float]:
        """
        Perform PPO update on batch of trajectories.
        
        Args:
            trajectories: List of trajectory dicts with keys:
                - states: [T, state_dim]
                - actions: [T, action_dim]
                - rewards: [T]
                - dones: [T]
                - values: [T]
                - log_probs: [T, 1]
        
        Returns:
            Dict of training losses
        """
        
        # Collect batch data
        states = np.concatenate([t['states'] for t in trajectories])
        actions = np.concatenate([t['actions'] for t in trajectories])
        rewards = np.concatenate([t['rewards'] for t in trajectories])
        dones = np.concatenate([t['dones'] for t in trajectories])
        values = np.concatenate([t['values'] for t in trajectories])
        old_log_probs = np.concatenate([t['log_probs'] for t in trajectories])
        
        # Compute advantages
        advantages, returns = self.compute_gae(rewards, values, dones)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states).to(self.device)
        actions_tensor = torch.FloatTensor(actions).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(
            states_tensor, actions_tensor, advantages_tensor,
            returns_tensor, old_log_probs_tensor
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        total_loss = 0.0
        n_updates = 0
        
        # Multiple epochs of updates
        for epoch in range(self.epochs_per_batch):
            for batch in dataloader:
                (batch_states, batch_actions, batch_advantages,
                 batch_returns, batch_old_log_probs) = batch
                
                # Forward pass
                dist, values_pred = self.policy(batch_states)
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                # Policy loss with clipping
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
                
                loss_policy = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()
                
                # Value function loss
                loss_value = ((values_pred - batch_returns) ** 2).mean()
                
                # Entropy regularization
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Total loss
                loss_total = (loss_policy + 
                             self.value_loss_coeff * loss_value -
                             self.entropy_coeff * entropy)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss_total.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                        self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss_total.item()
                n_updates += 1
        
        return {
            'total_loss': total_loss / n_updates,
            'policy_loss': loss_policy.item(),
            'value_loss': loss_value.item(),
            'entropy': entropy.item(),
        }
```

---

## Part 4: Environment Implementation

### 4.1 Base Environment Class

**File:** `hybrid_gcs/environments/base_env.py`

```python
"""
Abstract base class for Hybrid-GCS environments.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional
import gym
from gym import spaces
import numpy as np


class HybridGCSEnv(gym.Env, ABC):
    """
    Abstract base class for all Hybrid-GCS environments.
    
    Defines interface for environments to work with both GCS and RL.
    """
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.step_count = 0
        self.max_steps = config.get('max_steps', 500)
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return (obs, reward, done, info).
        
        Args:
            action: Action from policy or GCS planner
        
        Returns:
            obs: Observation
            reward: Scalar reward
            done: Episode termination flag
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_gcs_features(self) -> np.ndarray:
        """Get low-dimensional features for GCS planner."""
        pass
    
    @abstractmethod
    def get_rl_features(self) -> np.ndarray:
        """Get high-dimensional features for RL policy."""
        pass
    
    def render(self, mode: str = 'human'):
        """Render environment visualization."""
        pass
    
    def close(self):
        """Clean up resources."""
        pass
```

### 4.2 YCB Grasping Environment

**File:** `hybrid_gcs/environments/ycb_grasping_env.py`

```python
"""
YCB Object Grasping Environment

Single and dual-arm grasping with YCB objects.
"""

from typing import Dict, Tuple
import numpy as np
import gym
from gym import spaces

from .base_env import HybridGCSEnv


class YCBGraspingEnv(HybridGCSEnv):
    """YCB object grasping task."""
    
    def __init__(self, config: Dict):
        """
        Args:
            config: Configuration with:
                - robot_type: 'ur5' | 'kuka' | 'dual_arm'
                - gripper_type: 'parallel' | 'anthropomorphic'
                - num_objects: Number of objects in scene
                - difficulty: 'easy' | 'medium' | 'hard'
        """
        super().__init__(config)
        
        self.robot_type = config.get('robot_type', 'ur5')
        self.gripper_type = config.get('gripper_type', 'parallel')
        self.num_objects = config.get('num_objects', 5)
        self.difficulty = config.get('difficulty', 'easy')
        
        # Simulation setup
        self._setup_robot()
        self._setup_physics()
        
        # Action space: [arm_joints...] + [gripper]
        if self.robot_type == 'dual_arm':
            action_dim = 14 + 2  # 2×7 arm + 2 grippers
        else:
            action_dim = 7 + 1   # 7 arm + 1 gripper
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # Observation space
        obs_dim = 2048  # Will be set in reset
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.target_object = None
    
    def _setup_robot(self):
        """Initialize robot model."""
        # Load URDF model
        if self.robot_type == 'ur5':
            self.robot = self._load_ur5()
        elif self.robot_type == 'dual_arm':
            self.robot = self._load_dual_arm()
    
    def _setup_physics(self):
        """Initialize physics simulation."""
        import pybullet as p
        import pybullet_data
        
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        
        # Load plane
        p.loadURDF("plane.urdf")
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.step_count = 0
        
        # Reset robot to home position
        self.robot.reset_to_home()
        
        # Add objects
        self._add_objects()
        
        # Select target object
        self.target_object = self.objects[0]
        
        return self.get_rl_features()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute action."""
        
        # Apply action to robot
        self.robot.apply_action(action)
        
        # Step simulation
        self.physics_client.step()
        
        # Get observation
        obs = self.get_rl_features()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check termination
        done = self._check_done()
        
        # Additional info
        info = self._get_info()
        
        self.step_count += 1
        
        return obs, reward, done, info
    
    def get_gcs_features(self) -> np.ndarray:
        """Get low-dimensional features for GCS."""
        ee_pose = self.robot.get_end_effector_pose()
        obj_pos = self.target_object.get_position()
        
        features = np.concatenate([
            ee_pose,
            obj_pos,
        ])
        
        return features.astype(np.float32)
    
    def get_rl_features(self) -> np.ndarray:
        """Get high-dimensional features for RL."""
        # Joint states
        joint_pos = self.robot.get_joint_positions()
        joint_vel = self.robot.get_joint_velocities()
        
        # End-effector
        ee_pose = self.robot.get_end_effector_pose()
        
        # Object poses
        obj_poses = np.array([obj.get_pose().flatten() 
                             for obj in self.objects]).flatten()
        
        # Vision features (simulated RGB-D)
        vision_features = self._get_vision_features()
        
        features = np.concatenate([
            joint_pos, joint_vel, ee_pose,
            obj_poses,
            vision_features
        ])
        
        # Pad to fixed dimension
        target_dim = 2048
        if len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)))
        else:
            features = features[:target_dim]
        
        return features.astype(np.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward signal."""
        ee_pos = self.robot.get_end_effector_position()
        obj_pos = self.target_object.get_position()
        
        # Proximity reward
        dist = np.linalg.norm(ee_pos - obj_pos)
        reward = -dist
        
        # Contact bonus
        if self.robot.has_contact_with(self.target_object):
            reward += 5.0
        
        # Success bonus
        if self._is_object_lifted():
            reward += 50.0
        
        # Smooth action penalty
        reward -= 0.001 * np.sum(self.last_action ** 2)
        
        return float(reward)
    
    def _check_done(self) -> bool:
        """Check if episode should terminate."""
        # Success: object lifted
        if self._is_object_lifted():
            return True
        
        # Failure: max steps
        if self.step_count >= self.max_steps:
            return True
        
        # Failure: collision with non-target object
        for obj in self.objects[1:]:
            if self.robot.in_collision_with(obj):
                return True
        
        return False
    
    def _is_object_lifted(self, threshold: float = 0.1) -> bool:
        """Check if target object is lifted."""
        obj_pos = self.target_object.get_position()
        return obj_pos[2] > threshold
    
    def _get_info(self) -> Dict:
        """Return episode info."""
        return {
            'success': self._is_object_lifted(),
            'distance': np.linalg.norm(
                self.robot.get_end_effector_position() -
                self.target_object.get_position()
            ),
        }
    
    def _add_objects(self):
        """Add YCB objects to scene."""
        # Implementation based on difficulty
        pass
    
    def _get_vision_features(self) -> np.ndarray:
        """Get vision features from simulated camera."""
        # Implementation based on PyBullet camera
        pass
    
    def _load_ur5(self):
        """Load UR5 robot."""
        pass
    
    def _load_dual_arm(self):
        """Load dual-arm robot."""
        pass
```

---

## Part 5: Integration Module

### 5.1 Hybrid Policy

**File:** `hybrid_gcs/integration/hybrid_policy.py`

```python
"""
Hybrid Policy combining GCS and RL actions.
"""

from typing import Optional, Dict
import numpy as np
import torch


class HybridPolicy:
    """
    Combines GCS planner and RL policy actions.
    """
    
    def __init__(self,
                 gcs_solver,
                 rl_policy: torch.nn.Module,
                 blending_method: str = 'weighted',
                 blend_weight: float = 0.5,
                 device: str = 'cpu'):
        """
        Args:
            gcs_solver: GCS trajectory planner
            rl_policy: RL policy network
            blending_method: 'weighted' | 'hierarchical' | 'conflict'
            blend_weight: Initial blending weight
            device: 'cpu' or 'cuda'
        """
        self.gcs_solver = gcs_solver
        self.rl_policy = rl_policy
        self.blending_method = blending_method
        self.blend_weight = blend_weight
        self.device = device
        
        self.gcs_trajectory = None
        self.trajectory_idx = 0
    
    def get_action(self,
                  state: np.ndarray,
                  gcs_features: Optional[np.ndarray] = None,
                  replan: bool = False) -> np.ndarray:
        """
        Get hybrid action combining GCS and RL.
        
        Args:
            state: Full observation for RL
            gcs_features: Low-dim features for GCS
            replan: Force GCS replanning
        
        Returns:
            Hybrid action
        """
        
        # Get GCS action
        try:
            gcs_action = self._get_gcs_action(gcs_features, replan)
        except:
            gcs_action = None
        
        # Get RL action
        rl_action = self._get_rl_action(state)
        
        # Blend actions
        if gcs_action is None:
            return rl_action
        
        if self.blending_method == 'weighted':
            return self._blend_weighted(gcs_action, rl_action)
        elif self.blending_method == 'hierarchical':
            return self._blend_hierarchical(gcs_action, rl_action)
        else:
            return rl_action
    
    def _get_gcs_action(self, features: np.ndarray, 
                       replan: bool = False) -> np.ndarray:
        """Get action from GCS planner."""
        if self.gcs_trajectory is None or replan:
            self._plan_gcs_trajectory(features)
        
        if self.gcs_trajectory is None:
            return None
        
        desired_state = self.gcs_trajectory.at_time(
            self.trajectory_idx / self.gcs_trajectory.length())
        self.trajectory_idx += 1
        
        # Convert to action
        gcs_action = 0.1 * desired_state
        return gcs_action
    
    def _get_rl_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from RL policy."""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist, _ = self.rl_policy(state_tensor)
            action = dist.mean.cpu().numpy().flatten()
        
        return action
    
    def _plan_gcs_trajectory(self, features: np.ndarray):
        """Plan using GCS."""
        try:
            start = features[:3]
            goal = features[10:13]
            self.gcs_trajectory = self.gcs_solver.solve(start, goal)
            self.trajectory_idx = 0
        except Exception as e:
            print(f"GCS planning failed: {e}")
            self.gcs_trajectory = None
    
    def _blend_weighted(self, gcs_action: np.ndarray,
                       rl_action: np.ndarray) -> np.ndarray:
        """Linear blend between actions."""
        return ((1 - self.blend_weight) * gcs_action +
                self.blend_weight * rl_action)
    
    def _blend_hierarchical(self, gcs_action: np.ndarray,
                           rl_action: np.ndarray) -> np.ndarray:
        """Hierarchical selection."""
        if self.gcs_trajectory is not None:
            return gcs_action
        else:
            return rl_action
    
    def update_blend_weight(self, progress: float):
        """Update blend weight over training."""
        self.blend_weight = min(progress, 1.0)
```

### 5.2 Feature Extractor

**File:** `hybrid_gcs/integration/feature_extractor.py`

```python
"""
Feature extraction from observations.

Converts raw obs into GCS and RL feature representations.
"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn


class FeatureExtractor:
    """Extracts features for GCS and RL."""
    
    def __init__(self,
                 state_dim_gcs: int = 50,
                 state_dim_rl: int = 512,
                 use_vision: bool = True,
                 device: str = 'cpu'):
        """
        Args:
            state_dim_gcs: GCS feature dimension
            state_dim_rl: RL feature dimension
            use_vision: Use CNN vision encoder
            device: 'cpu' or 'cuda'
        """
        self.state_dim_gcs = state_dim_gcs
        self.state_dim_rl = state_dim_rl
        self.use_vision = use_vision
        self.device = device
        
        if use_vision:
            from .vision_encoder import CNNEncoder
            self.vision_encoder = CNNEncoder(output_dim=256).to(device)
    
    def extract(self, observation: Dict) -> Dict[str, np.ndarray]:
        """Extract features from observation."""
        
        # Kinematic features
        kinematic = np.concatenate([
            observation['joint_positions'].flatten(),
            observation['joint_velocities'].flatten(),
            observation['end_effector_pose'].flatten(),
        ])
        
        # Goal features
        goal = observation['goal'].flatten()
        
        # Object features
        objects = []
        for obj_pose in observation.get('object_poses', [])[:3]:
            objects.append(obj_pose.flatten())
        if not objects:
            objects = [np.zeros(7)]
        objects = np.concatenate(objects)
        
        # GCS features (low-dim)
        gcs_features = np.concatenate([kinematic, goal, objects[:20]])
        gcs_features = self._pad_or_truncate(gcs_features, self.state_dim_gcs)
        
        # RL features (high-dim with vision)
        rl_features = [kinematic, goal, objects]
        
        if self.use_vision:
            vision = self._get_vision_features(observation)
            rl_features.append(vision)
        
        rl_features = np.concatenate(rl_features)
        rl_features = self._pad_or_truncate(rl_features, self.state_dim_rl)
        
        return {
            'gcs': gcs_features.astype(np.float32),
            'rl': rl_features.astype(np.float32),
        }
    
    def _get_vision_features(self, observation: Dict) -> np.ndarray:
        """Extract vision features from RGB-D."""
        rgb = observation['rgb_image']
        depth = observation['depth_image']
        
        # Stack RGB-D
        rgbd = np.concatenate([rgb, depth[..., np.newaxis]], axis=-1)
        
        # Convert to tensor
        rgbd_tensor = torch.FloatTensor(rgbd).permute(2, 0, 1).unsqueeze(0)
        rgbd_tensor = rgbd_tensor.to(self.device)
        
        # Resize to 84×84
        rgbd_tensor = torch.nn.functional.interpolate(
            rgbd_tensor, size=(84, 84), mode='bilinear')
        
        # Extract features
        with torch.no_grad():
            features = self.vision_encoder(rgbd_tensor)
        
        return features.cpu().numpy().flatten()
    
    def _pad_or_truncate(self, features: np.ndarray,
                        target_dim: int) -> np.ndarray:
        """Pad or truncate to target dimension."""
        if len(features) >= target_dim:
            return features[:target_dim]
        else:
            padding = np.zeros(target_dim - len(features))
            return np.concatenate([features, padding])
```

---

## Part 6: Professional Standards

### 6.1 Code Quality Standards

**PEP 8 Compliance:**

```python
# ✅ GOOD: Clear naming, proper spacing
def compute_trajectory_length(waypoints: np.ndarray,
                             metric: str = 'euclidean') -> float:
    """Compute total trajectory length."""
    diffs = np.diff(waypoints, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.sum(distances)

# ❌ BAD: Poor naming, unclear intent
def f(w, m='e'):
    d = np.diff(w, axis=0)
    n = np.linalg.norm(d, axis=1)
    return np.sum(n)
```

**Documentation Standards:**

```python
def solve(self,
         start: np.ndarray,
         goal: np.ndarray,
         **kwargs) -> Optional[Trajectory]:
    """
    Solve shortest path in GCS from start to goal.
    
    This method uses Mixed-Integer Convex Programming to find
    the globally optimal trajectory through the convex region graph.
    
    Args:
        start: Start configuration [dim]
        goal: Goal configuration [dim]
        **kwargs: Additional solver options:
            - time_limit: Max solver time (seconds)
            - verbose: Print solver output
    
    Returns:
        Trajectory object if solution found, None otherwise
    
    Raises:
        ValueError: If start or goal not in configuration space
    
    Examples:
        >>> solver = MICPSolver(graph)
        >>> traj = solver.solve(start, goal, time_limit=30.0)
        >>> if traj is not None:
        ...     print(f"Path length: {traj.length()}")
    
    References:
        Marcucci, T., et al. (2023). "Motion Planning around
        Obstacles with Convex Optimization"
    """
    pass
```

### 6.2 Type Hints

**Always include type hints:**

```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

def train_epoch(self,
               trajectories: List[Dict[str, np.ndarray]],
               learning_rate: Optional[float] = None
               ) -> Tuple[float, Dict[str, float]]:
    """
    Train for one epoch.
    
    Args:
        trajectories: List of trajectory dicts
        learning_rate: Optional learning rate override
    
    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    pass
```

### 6.3 Testing Standards

**Example test file:**

```python
# tests/test_training/test_ppo_trainer.py

import pytest
import torch
import numpy as np
from hybrid_gcs.training.policy_network import PolicyNetwork
from hybrid_gcs.training.ppo_trainer import PPOTrainer


@pytest.fixture
def policy():
    """Create test policy network."""
    return PolicyNetwork(state_dim=10, action_dim=3)


@pytest.fixture
def trainer(policy):
    """Create test trainer."""
    return PPOTrainer(policy, learning_rate=1e-4)


def test_trainer_initialization(trainer):
    """Test trainer initializes correctly."""
    assert trainer.gamma == 0.99
    assert trainer.clip_ratio == 0.2


def test_gae_computation(trainer):
    """Test GAE calculation."""
    rewards = np.array([1.0, 1.0, 0.0])
    values = np.array([0.5, 0.3, 0.0])
    dones = np.array([0, 0, 1])
    
    advantages, returns = trainer.compute_gae(rewards, values, dones)
    
    assert advantages.shape == (3,)
    assert returns.shape == (3,)
    assert not np.any(np.isnan(advantages))
```

---

## Part 7: Quick Start Guide

### 7.1 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/hybrid-gcs
cd hybrid-gcs

# Create environment
python -m venv venv
source venv/bin/activate  # or: venv\Scripts\activate on Windows

# Install dependencies
pip install -e .
pip install -e ".[dev]"  # Include dev tools
```

### 7.2 First Example

```python
# examples/01_simple_navigation.py

from hybrid_gcs.core.config_space import ConfigSpace
from hybrid_gcs.core.iris_decomposer import IRISDecomposer
from hybrid_gcs.core.micp_solver import MICPSolver
import numpy as np

# 1. Define configuration space
config_space = ConfigSpace(
    dim=2,
    bounds_lower=np.array([0.0, 0.0]),
    bounds_upper=np.array([10.0, 10.0]),
    names=['x', 'y'],
)

# 2. Decompose space (with obstacles)
decomposer = IRISDecomposer(config_space)
obstacles = []  # Define obstacles
seeds = [np.array([1.0, 1.0]), np.array([9.0, 9.0])]
graph = decomposer.decompose(obstacles, seeds)

# 3. Solve for trajectory
solver = MICPSolver(graph, solver_type='mosek')
start = np.array([0.0, 0.0])
goal = np.array([10.0, 10.0])

trajectory = solver.solve(start, goal)
if trajectory is not None:
    print(f"Path found! Length: {trajectory.length()}")
```

---

**END OF IMPLEMENTATION GUIDE**

This guide provides a complete professional framework for implementing Hybrid-GCS. Follow the structure, adhere to standards, and build incrementally.

