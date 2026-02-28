# Hybrid-GCS

**Hybrid planning combining Graph of Convex Sets (GCS) with Deep Reinforcement Learning for autonomous robotics.**

[![CI](https://github.com/HarshMulodhia/hybrid_gcs/actions/workflows/ci.yml/badge.svg)](https://github.com/HarshMulodhia/hybrid_gcs/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Hybrid-GCS integrates classical trajectory optimization via Graph of Convex Sets (GCS) with modern deep reinforcement learning (PPO) to create a unified planning and control framework for robotics applications such as grasping, navigation, and manipulation.

### Key Components

| Module | Description |
|--------|-------------|
| **core** | GCS algorithms — ConfigSpace, Trajectory, IRIS decomposition, MICP solver |
| **training** | Deep RL — PolicyNetwork (Actor-Critic), PPO trainer, reward shaping, curriculum learning |
| **visualization** | Foxglove Studio MCAP recording for 3D scene visualization |
| **integration** | Hybrid GCS + RL blending, action selection, safety filtering *(planned)* |
| **environments** | Task environments — grasping, drone navigation, manipulation *(planned)* |
| **evaluation** | Metrics, trajectory analysis, benchmarks *(planned)* |

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

### 2. RL Policy Training

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

### 3. Foxglove Visualization

```python
from hybrid_gcs.visualization import FoxgloveRecorder, record_trajectory_scene

# Record a full scene to MCAP
with FoxgloveRecorder("scene.mcap") as recorder:
    recorder.add_robot_description("data/models/ur5e/ur5e.urdf")
    recorder.add_environment("data/models/environment/tabletop.urdf")
    recorder.add_obstacles(obstacles)
    recorder.add_convex_regions(regions)
    recorder.add_trajectory(trajectory)

# Or use the convenience function
record_trajectory_scene("scene.mcap", trajectory, obstacles=obstacles)
```

Open the `.mcap` file in [Foxglove Studio](https://foxglove.dev/studio), add a 3D panel, and optionally import the layout from `data/foxglove/hybrid_gcs_layout.json`.

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=hybrid_gcs --cov-report=term-missing

# Run specific module tests
python -m pytest tests/test_core/ -v
python -m pytest tests/test_training/ -v
```

## Examples

| Script | Description |
|--------|-------------|
| `examples/01_simple_navigation.py` | 2D GCS planning with IRIS decomposition |
| `examples/02_rl_training.py` | RL policy training loop with PPO |
| `examples/03_foxglove_visualization.py` | Foxglove MCAP scene recording |

```bash
python examples/01_simple_navigation.py
```

## Project Structure

```
hybrid_gcs/
├── hybrid_gcs/              # Main package
│   ├── core/                # GCS algorithms
│   │   ├── config_space.py  # Configuration space
│   │   ├── trajectory.py    # Trajectory representations
│   │   ├── iris_decomposer.py  # IRIS decomposition
│   │   └── micp_solver.py   # MICP solver + GCSGraph
│   ├── training/            # Deep RL
│   │   ├── policy_network.py     # Actor-Critic networks
│   │   ├── ppo_trainer.py        # PPO algorithm
│   │   ├── reward_shaper.py      # Reward composition
│   │   ├── curriculum_scheduler.py  # Curriculum learning
│   │   └── experience_buffer.py  # Replay memory
│   └── visualization/       # Foxglove Studio
│       └── foxglove_recorder.py  # MCAP scene recording
├── tests/                   # Unit & integration tests
├── examples/                # Example scripts
├── docs/                    # Documentation
└── data/                    # Data & assets
```

## References

- Marcucci et al. (2023) — *Motion Planning around Obstacles with Convex Optimization*
- Schulman et al. (2017) — *Proximal Policy Optimization Algorithms*
- Deits & Tedrake (2015) — *Computing Large Convex Regions of Obstacle-Free Space through Semidefinite Programming*

## License

MIT License — see [LICENSE](LICENSE) for details.
