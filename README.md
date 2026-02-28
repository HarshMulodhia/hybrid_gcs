# Hybrid-GCS

**Hybrid planning combining Graph of Convex Sets (GCS) with Deep Reinforcement Learning for autonomous robotics.**

[![CI](https://github.com/HarshMulodhia/hybrid_gcs/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshMulodhia/hybrid_gcs/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Hybrid-GCS integrates classical trajectory optimization via Graph of Convex Sets (GCS)
with modern deep reinforcement learning (PPO) to create a unified planning and control
framework for autonomous robotics.

| Aspect | GCS | Deep RL | Hybrid |
|-----------|---------------------|---------------------|--------------------------|
| Planning | Global optimization | Reactive learning | Both (complementary) |
| Safety | Guaranteed safe | No guarantees | Safety-filtered |
| Adaptation| Static plan | Dynamic learning | Real-time adaptation |
| Scaling | High dimensions (OK)| High-dim sensing (OK)| Both (excellent) |

### Key Modules

| Module | Description |
|--------|-------------|
| **core** | GCS algorithms — ConfigSpace, Trajectory, IRIS decomposition, MICP solver |
| **training** | Deep RL — PolicyNetwork (Actor-Critic), PPO trainer, reward shaping, curriculum learning |
| **integration** | Hybrid GCS+RL blending, safety filter, dual-pathway feature extraction |
| **environments** | Task environments — YCB grasping, drone navigation, complex manipulation |
| **visualization** | Foxglove Studio MCAP recording for 3D scene visualization |

## Three Application Domains

### 1. YCB Object Grasping (Single & Dual-Arm)

- **GCS**: Plans collision-free approach trajectory from EE position to pre-grasp location
- **RL**: Learns grasp quality feedback, contact force control, adaptive re-grasping
- **Hybrid**: Follows GCS trajectory when valid, switches to RL for last-meter adjustment
- **Success Metrics**: Success rate >85%, lift height >0.3m, robustness to ±5cm placement

### 2. Autonomous Drone Navigation (Single & Multi-Agent)

- **GCS**: Uses Space-Time GCS (ST-GCS) for time-optimal, collision-free trajectories
- **RL**: Learns reactive control, communication strategies, emergent behaviors
- **Hybrid**: GCS provides global path, RL handles dynamic replanning and interactions
- **Success Metrics**: Path efficiency 1.2× optimal, collision rate <5%, scalability to 50+ drones

### 3. Complex Manipulation (Reach, Pick, Push, Stack)

- **GCS**: Decomposes workspace into safe corridors for each manipulation primitive
- **RL**: Learns primitive execution (reach, grasp, push) and transitions
- **Hybrid**: GCS ensures collision-free primitive sequences, RL adapts within-primitives
- **Success Metrics**: Task completion >90%, multi-task generalization

## System Architecture

```
┌──────────────────┐
│ Perception       │ (RGB-D, kinematics, task)
└──────┬───────────┘
       │
       ▼
┌──────────────────────┐
│ Feature Extraction   │ Dual pathways:
├──────┬───────────────┤ - Low-D for GCS
│GCS   │RL Features    │ - High-D for RL
│Feats │(with vision)  │
└──────┴────┬──────────┘
            │
       ┌────┴────────────────┐
       ▼                     ▼
┌─────────────┐      ┌──────────────┐
│ GCS Module  │      │ RL Module    │
│ - Plan      │      │ - Policy net │
│ - Optimize  │      │ - Value net  │
│ - Trajectory│      │ - PPO trainer│
└──────┬──────┘      └────┬─────────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌──────────────────┐
        │ Integration      │ - Blend actions
        │ - Blending       │ - Resolve conflicts
        │ - Conflict Res   │ - Safety filter
        │ - Safety Filter  │
        └────────┬─────────┘
                 ▼
        ┌──────────────────┐
        │ Execution        │ Robot + Sim
        └──────────────────┘
```

## Installation

```bash
# Clone the repository
git clone https://github.com/HarshMulodhia/hybrid_gcs.git
cd hybrid_gcs

# Install in development mode
pip install -e ".[dev]"

# Install with all optional dependencies
pip install -e ".[full]"
```

### Optional Extras

| Extra | What it adds |
|-------|-------------|
| `dev` | pytest, black, pylint, mypy, sphinx |
| `solvers` | SCS (free convex solver) |
| `sim` | PyBullet physics simulation |
| `rl` | TensorBoard, matplotlib |
| `viz` | Foxglove MCAP recording (mcap, foxglove-schemas-protobuf) |
| `full` | All of the above |

## Quick Start

### 1. GCS Planning

```python
import numpy as np
from hybrid_gcs.core import (
    ConfigSpace, IRISDecomposer, SimpleBoxObstacle,
    GCSGraph, MICPSolver
)

# Define configuration space
space = ConfigSpace(
    dim=2,
    bounds_lower=np.array([0.0, 0.0]),
    bounds_upper=np.array([10.0, 10.0]),
    names=['x', 'y']
)

# Create obstacles and decompose
obstacles = [SimpleBoxObstacle(np.array([3, 3]), np.array([7, 7]))]
decomposer = IRISDecomposer(space, max_iterations=10)
regions = decomposer.decompose(
    seed_points=[np.array([1, 1]), np.array([9, 9])],
    obstacles=obstacles
)

# Build graph and solve
graph = GCSGraph()
for i, r in enumerate(regions):
    graph.add_vertex(i, region=r)
for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        graph.add_edge(i, j)
        graph.add_edge(j, i)

solver = MICPSolver(graph, space, solver_type='scs')
trajectory = solver.solve(np.array([0.5, 0.5]), np.array([9.5, 9.5]))
print(f"Path length: {trajectory.length():.2f}")
```

### 2. Hybrid Integration

```python
import numpy as np
from hybrid_gcs.integration import (
    WeightedBlender, SafetyFilter, SafetyFilterConfig,
    DualPathwayExtractor, FeatureExtractorConfig
)

# Action blending: combine GCS and RL outputs
blender = WeightedBlender(alpha=0.7)  # 70% GCS, 30% RL
gcs_action = np.array([1.0, 0.0, 0.5])
rl_action = np.array([0.8, 0.2, 0.3])
action = blender.blend(gcs_action, rl_action)

# Safety filter: enforce constraints
safety_config = SafetyFilterConfig(
    position_bounds_lower=np.array([-5.0, -5.0, 0.0]),
    position_bounds_upper=np.array([5.0, 5.0, 3.0]),
    max_velocity=2.0,
)
safety = SafetyFilter(safety_config)
safe_action = safety.filter_action(action, current_state=np.zeros(6))
```

### 3. Task Environments

```python
import numpy as np
from hybrid_gcs.environments import (
    GraspingEnv, GraspingConfig,
    DroneNavEnv, DroneNavConfig,
    ManipulationEnv, ManipulationConfig, ManipulationTask
)

# YCB Grasping
env = GraspingEnv(GraspingConfig(max_steps=200, seed=42))
obs = env.reset()
result = env.step(np.array([0.01, 0.0, -0.01, 0.0]))

# Drone Navigation (multi-agent)
drone_env = DroneNavEnv(DroneNavConfig(num_agents=3, seed=42))
obs = drone_env.reset()

# Complex Manipulation
manip_env = ManipulationEnv(ManipulationConfig(
    task=ManipulationTask.STACK, num_objects=2, seed=42
))
obs = manip_env.reset()
```

### 4. RL Policy Training

```python
import numpy as np
from hybrid_gcs.training import (
    PolicyNetwork, PolicyNetworkConfig,
    PPOTrainer, PPOConfig
)

# Create policy
policy = PolicyNetwork(PolicyNetworkConfig(state_dim=6, action_dim=2))

# Train with PPO
trainer = PPOTrainer(policy, PPOConfig(learning_rate=3e-4))
states = np.random.randn(64, 6).astype(np.float32)
actions = np.random.randn(64, 2).astype(np.float32)
rewards = np.ones(64, dtype=np.float32)
values = np.zeros(64, dtype=np.float32)
dones = np.zeros(64, dtype=bool)

stats = trainer.update(states, actions, rewards, values, dones, next_value=0.0)
print(f"Policy loss: {stats['policy_loss']:.4f}")
```

### 5. Foxglove Visualization

```python
from hybrid_gcs.visualization import FoxgloveRecorder, record_trajectory_scene

# Record a full scene to MCAP
with FoxgloveRecorder("scene.mcap") as recorder:
    recorder.add_robot_description("data/models/ur5e/ur5e.urdf")
    recorder.add_environment("data/models/environment/tabletop.urdf")
    recorder.add_obstacles(obstacles)
    recorder.add_convex_regions(regions)
    recorder.add_trajectory(trajectory)
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=hybrid_gcs --cov-report=term-missing

# Run specific module tests
python -m pytest tests/test_core/ -v
python -m pytest tests/test_training/ -v
python -m pytest tests/test_integration/ -v
python -m pytest tests/test_environments/ -v
```

## Project Structure

```
hybrid_gcs/
├── hybrid_gcs/                  # Main package
│   ├── core/                    # GCS algorithms
│   │   ├── config_space.py      # Configuration space
│   │   ├── trajectory.py        # Trajectory representations
│   │   ├── iris_decomposer.py   # IRIS decomposition
│   │   └── micp_solver.py       # MICP solver + GCSGraph
│   ├── training/                # Deep RL
│   │   ├── policy_network.py    # Actor-Critic networks + CNN encoder
│   │   ├── ppo_trainer.py       # PPO algorithm
│   │   ├── reward_shaper.py     # Reward composition (7 strategies)
│   │   ├── curriculum_scheduler.py  # Curriculum learning
│   │   └── experience_buffer.py # Replay memory
│   ├── integration/             # Hybrid GCS+RL
│   │   ├── blending.py          # Action blending (weighted, hierarchical, conflict)
│   │   ├── safety_filter.py     # Real-time constraint enforcement
│   │   └── feature_extractor.py # Dual pathway feature extraction
│   ├── environments/            # Task environments
│   │   ├── base_env.py          # Base environment interface
│   │   ├── grasping_env.py      # YCB object grasping
│   │   ├── drone_nav_env.py     # Drone navigation (single/multi)
│   │   └── manipulation_env.py  # Complex manipulation (reach/pick/push/stack)
│   └── visualization/           # Foxglove Studio
│       └── foxglove_recorder.py # MCAP scene recording
├── tests/                       # Unit & integration tests
├── examples/                    # Example scripts
├── docs/                        # Documentation
└── data/                        # Data & assets
```

## Mathematical Foundation

### GCS Problem

```
min Σ c_v(x_v) + Σ c_e(x_e)
s.t. Binary path selection
     x_v ∈ C_v if region v selected
     Continuity at boundaries
     Dynamics constraints
```

### RL Problem

```
max E[Σ γ^t r(s_t, a_t)]
s.t. Policy π(a|s)
     Value V(s)
     Trust region constraints (PPO)
```

### Hybrid Solution

```
Action = Blend(GCS_action, RL_action, conflict_resolver)
       + Safety_filter(action, constraints)
```

## Dependencies

- Python 3.9+
- NumPy, SciPy
- PyTorch (for RL and feature extraction)
- SCS/Mosek/Gurobi (MICP solver, optional)
- PyBullet (simulation, optional)

## References

- Marcucci et al. (2023) — *Motion Planning around Obstacles with Convex Optimization*
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Deits & Tedrake (2015) — *Computing Large Convex Regions of Obstacle-Free Space*

## License

MIT License — see [LICENSE](LICENSE) for details.
