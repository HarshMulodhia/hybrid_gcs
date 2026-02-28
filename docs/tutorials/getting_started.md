# Getting Started

## Prerequisites

- Python 3.9 or later
- pip package manager

## Installation

```bash
git clone https://github.com/HarshMulodhia/hybrid_gcs.git
cd hybrid_gcs
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "import hybrid_gcs; print(hybrid_gcs.__version__)"
# Expected output: 0.1.0
```

## Run the Tests

```bash
python -m pytest tests/ -v
```

## Your First GCS Plan

Create a file called `my_plan.py`:

```python
import numpy as np
from hybrid_gcs.core import (
    ConfigSpace, IRISDecomposer, SimpleBoxObstacle,
    GCSGraph, MICPSolver
)

# 1. Define a 2D workspace
space = ConfigSpace(
    dim=2,
    bounds_lower=np.array([0.0, 0.0]),
    bounds_upper=np.array([10.0, 10.0]),
    names=['x', 'y']
)

# 2. Add an obstacle
obstacles = [SimpleBoxObstacle(np.array([4, 4]), np.array([6, 6]))]

# 3. Decompose free space with IRIS
decomposer = IRISDecomposer(space, max_iterations=10)
regions = decomposer.decompose(
    seed_points=[np.array([2, 2]), np.array([8, 8])],
    obstacles=obstacles
)
print(f"Created {len(regions)} convex regions")

# 4. Build a graph and plan
graph = GCSGraph()
for i, r in enumerate(regions):
    graph.add_vertex(i, region=r)
for i in range(len(regions)):
    for j in range(i + 1, len(regions)):
        graph.add_edge(i, j)
        graph.add_edge(j, i)

solver = MICPSolver(graph, space, solver_type='scs')
traj = solver.solve(np.array([1.0, 1.0]), np.array([9.0, 9.0]))

if traj is not None:
    print(f"Trajectory found! Length: {traj.length():.2f}")
    # Sample along trajectory
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  t={t:.2f}: {traj.at_time(t)}")
```

Run it:

```bash
python my_plan.py
```

## Your First RL Training

```python
import numpy as np
from hybrid_gcs.training import (
    PolicyNetwork, PolicyNetworkConfig,
    PPOTrainer, PPOConfig
)

# Create a policy network
config = PolicyNetworkConfig(state_dim=4, action_dim=2, hidden_dim=64)
policy = PolicyNetwork(config)

# Create a PPO trainer
trainer = PPOTrainer(policy, PPOConfig(learning_rate=3e-4, epochs=4))

# Simulate a training step
T = 128
states = np.random.randn(T, 4).astype(np.float32)
actions = np.random.randn(T, 2).astype(np.float32)
rewards = np.ones(T, dtype=np.float32)
values = np.zeros(T, dtype=np.float32)
dones = np.zeros(T, dtype=bool)
dones[-1] = True

stats = trainer.update(states, actions, rewards, values, dones, next_value=0.0)
print(f"Update complete — policy_loss: {stats['policy_loss']:.4f}")
```

## Next Steps

- Explore the [Architecture](../architecture.md) for a system overview.
- Read the [Core API](../api/core.md) for detailed module docs.
- Look at the examples in `examples/` for complete working scripts.
