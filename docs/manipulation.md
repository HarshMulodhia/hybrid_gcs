# Complex Manipulation Tasks

Multi-primitive manipulation: **Reach**, **Pick**, **Push**, and **Stack**.

## Overview

The manipulation domain provides four tasks of increasing complexity
sharing the same `ManipulationEnv` interface.  All tasks use simple
kinematic simulation with an end-effector and one or more objects on a
tabletop workspace.

### Task Variants

| Task | Flag | Objects | Goal |
|------|------|---------|------|
| **Reach** | `--task reach` | 1 | Move EE to a random target |
| **Pick** | `--task pick` | 1 | Grasp object and lift to target |
| **Push** | `--task push` | 1 | Push object to target on table |
| **Stack** | `--task stack` | 2 | Pick first object, place on second |

## Environment Details

### Workspace

- Bounds: x ∈ [0.3, 0.8], y ∈ [−0.3, 0.3], z ∈ [0.0, 0.6]
- Objects placed randomly on the table surface (z = 0)
- Targets sampled per task (see below)

### Observation (10 + 3 × num_objects)

| Index | Name | Description |
|-------|------|-------------|
| 0–2 | `ee_position` | End-effector XYZ |
| 3–5 | `ee_velocity` | End-effector velocity |
| 6 | `gripper_state` | 0 = open, 1 = closed |
| 7 … 7+3n−1 | `object_positions` | Flattened XYZ of each object |
| last 3 | `target_position` | Target XYZ |

### Action (4-dim)

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0–2 | `ee_velocity_cmd` | continuous | Velocity applied to EE |
| 3 | `gripper_cmd` | [0, 1] | Close gripper when > 0.5 |

### Dynamics

```
ee_position += ee_velocity * dt          (dt = 0.01 s)
ee_position  = clip(ee_position, ws_lo, ws_hi)
```

Grasping occurs when `gripper_cmd > 0.5` and the EE is within 0.05 m of
any object.  A grasped object follows the EE.  Opening the gripper
releases the object.

## Task-Specific Reward Shaping

### Reach

```
reward = -||ee - target||                # dense distance
reward += 10.0  if ||ee - target|| < 0.05   # success bonus
```

### Pick

```
reward = -||ee - obj|| - ||obj - target||   # approach + lift
reward += 1.0   if grasped                  # grasp bonus
reward += 10.0  if obj at target             # success bonus
```

### Push

```
reward = -0.5 * ||ee - obj|| - ||obj - target||  # approach + push
reward += 10.0  if ||obj - target|| < 0.05       # success bonus
```

### Stack

```
reward = -||ee - obj|| - ||obj - target||   # approach + place
reward += 1.0   if grasped                  # grasp bonus
reward += 10.0  if placed within 0.08 m     # stacking bonus
```

"Placed" means the object is within 0.08 m of the stack target **and**
the gripper has been released.

## Training

```bash
# Reach (simplest — good for verifying setup)
hybrid-gcs-train --env manipulation --task reach --episodes 100

# Pick
hybrid-gcs-train --env manipulation --task pick --episodes 200

# Push
hybrid-gcs-train --env manipulation --task push --episodes 200

# Stack (hardest — requires grasp + place + release)
hybrid-gcs-train --env manipulation --task stack --episodes 300
```

### PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3 × 10⁻⁴ | Standard PPO default |
| `gamma` | 0.99 | Multi-step tasks need far-sighted returns |
| `gae_lambda` | 0.95 | Balanced advantage estimation |
| `clip_ratio` | 0.2 | Safe policy update bound |
| `entropy_coef` | 0.01 | Encourages exploration of grasp strategies |
| `epochs` | 4 | Multiple gradient passes per rollout |
| `batch_size` | 64 | Mini-batch size |
| `num_steps` | 2048 | Rollout length before update |

### Curriculum Suggestion

For the **stack** task, consider a curriculum that starts with the
**reach** task, progresses to **pick**, and finally **stack**.  The
`CurriculumManager` in `hybrid_gcs.training` supports linear,
exponential, step, sigmoid, and performance-based schedules.

## Evaluation

```bash
hybrid-gcs-eval --env manipulation --task stack \
    --checkpoint checkpoints/train/best.pth --episodes 50
```

Key metrics:

| Metric | Description |
|--------|-------------|
| `success_rate` | Fraction of episodes achieving the task goal |
| `mean_reward` | Average cumulative reward |
| `mean_length` | Average episode steps (lower = more efficient) |

## Visualization

### Foxglove Studio

```bash
hybrid-gcs-vis --env manipulation --task pick \
    --checkpoint checkpoints/train/best.pth \
    --backend foxglove \
    --output output/manipulation_pick.mcap
```

Markers:

- **Blue sphere** — end-effector
- **Orange sphere** — object
- **Green sphere (transparent)** — target

### PyBullet

```bash
hybrid-gcs-vis --env manipulation --task pick \
    --checkpoint checkpoints/train/best.pth \
    --backend pybullet
```

Real-time 3-D rendering of the table, EE, and object markers.

## Python API

```python
from hybrid_gcs.environments import ManipulationEnv, ManipulationConfig, ManipulationTask
from hybrid_gcs.training import PolicyNetwork, PolicyNetworkConfig

config = ManipulationConfig(
    task=ManipulationTask.PICK,
    num_objects=1,
    max_steps=500,
    seed=42,
)
env = ManipulationEnv(config)

pol_cfg = PolicyNetworkConfig(
    state_dim=env.observation_dim,
    action_dim=env.action_dim,
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

- Plappert, M. et al. (2018). *Multi-Goal Reinforcement Learning.*
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.*
