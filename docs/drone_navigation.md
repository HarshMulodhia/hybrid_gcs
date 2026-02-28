# Autonomous Drone Navigation

Single-agent and multi-agent drone navigation through an obstacle field.

## Overview

The drone navigation domain trains PPO policies to fly one or more
drones from random start positions to goal positions while avoiding
spherical obstacles and inter-agent collisions.  It uses the
`DroneNavEnv` environment with velocity-based kinematic simulation.

### Variants

| Variant | Flag | Description |
|---------|------|-------------|
| **Single-agent** | `--num-agents 1` (default) | One drone navigating to a single goal |
| **Multi-agent** | `--num-agents N` | *N* drones with independent goals and shared airspace |

## Environment Details

### World

- Bounds: 10 m × 10 m × 5 m
- Spherical obstacles of radius 0.5 m (configurable via `--num-obstacles`)
- Drone collision radius: 0.3 m

### Observation (per agent, 9 + num_obstacles)

| Index | Name | Description |
|-------|------|-------------|
| 0–2 | `position` | Drone XYZ position |
| 3–5 | `velocity` | Drone XYZ velocity |
| 6–8 | `goal` | Goal XYZ position |
| 9… | `obstacle_dists` | Euclidean distance to each obstacle |

For multi-agent mode, all agent observations are **concatenated** into a
single flat vector of length `num_agents × (9 + num_obstacles)`.

### Action (per agent, 3-dim)

| Index | Name | Description |
|-------|------|-------------|
| 0–2 | `acceleration` | 3-D acceleration command |

For multi-agent mode, all agent actions are concatenated
(`num_agents × 3`).

### Dynamics

```
velocity += acceleration * dt           (dt = 0.01 s)
velocity  = clamp(velocity, max_velocity=2.0)
position += velocity * dt
position  = clip(position, world_lo, world_hi)
```

### Reward Shaping

```
# Per-agent rewards (summed across agents)
reward -= dist_to_goal                   # dense approach signal
reward += 10.0     if dist < 0.3         # goal-reached bonus

# Penalties
reward -= 50.0     if obstacle collision  # obstacle radius + drone radius
reward -= 50.0     if inter-agent collision  # 2 × drone collision radius
```

The episode terminates when **all** agents reach their goals (success)
or **any** collision occurs (failure), or `max_steps` is reached.

## Training

```bash
# Single-agent
hybrid-gcs-train --env drone_nav --episodes 200 --seed 42

# Multi-agent (3 drones, 8 obstacles)
hybrid-gcs-train --env drone_nav --num-agents 3 --num-obstacles 8 --episodes 300
```

### PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3 × 10⁻⁴ | Standard PPO default |
| `gamma` | 0.995 | Long horizon encourages strategic obstacle avoidance |
| `gae_lambda` | 0.98 | Lower bias for multi-step credit assignment |
| `clip_ratio` | 0.2 | Conservative policy updates |
| `entropy_coef` | 0.005 | Modest exploration (collision avoidance needs precision) |
| `epochs` | 4 | Multiple passes per rollout |
| `batch_size` | 128 | Larger batches stabilize multi-agent gradients |
| `num_steps` | 2048 | Rollout length |

### Multi-Agent Training Notes

In multi-agent mode the policy network's input dimension scales with the
number of agents.  A single shared policy controls all drones (parameter
sharing).  For independent policies or communication-augmented training
see the `multi_agent` module (`AttentionComm`, `MultiAgentPolicy`,
`CentralizedCritic`).

## Evaluation

```bash
hybrid-gcs-eval --env drone_nav \
    --num-agents 3 --num-obstacles 8 \
    --checkpoint checkpoints/train/best.pth \
    --episodes 50
```

Key metrics:

- **success_rate** — fraction of episodes where all drones reach goals
  without collision
- **mean_reward** — average cumulative reward
- **mean_length** — average steps to termination

## Visualization

### Foxglove Studio

```bash
hybrid-gcs-vis --env drone_nav --num-agents 3 \
    --checkpoint checkpoints/train/best.pth \
    --backend foxglove \
    --output output/drone_nav.mcap
```

Open the `.mcap` file in [Foxglove Studio](https://studio.foxglove.dev).
Each drone appears as a colored sphere; goals are shown as
semi-transparent green spheres.

### PyBullet

```bash
hybrid-gcs-vis --env drone_nav --num-agents 3 \
    --checkpoint checkpoints/train/best.pth \
    --backend pybullet
```

A 3-D window shows a ground plane with drone markers (spheres) moving in
real time.  Use `--offscreen` for headless rendering.

## Python API

```python
from hybrid_gcs.environments import DroneNavEnv, DroneNavConfig
from hybrid_gcs.training import PolicyNetwork, PolicyNetworkConfig

config = DroneNavConfig(num_agents=2, num_obstacles=5, max_steps=500, seed=42)
env = DroneNavEnv(config)

pol_cfg = PolicyNetworkConfig(
    state_dim=env.observation_dim,
    action_dim=env.action_dim,
    hidden_dim=256,
)
policy = PolicyNetwork(pol_cfg)

obs = env.reset()
done = False
while not done:
    action, _ = policy.get_action(obs)
    result = env.step(action)
    obs, done = result.observation, result.done or result.truncated
```

## References

- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.*
- Lowe, R. et al. (2017). *Multi-Agent Actor-Critic for Mixed
  Cooperative-Competitive Environments.*
