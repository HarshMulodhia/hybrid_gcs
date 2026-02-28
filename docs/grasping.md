# YCB Object Grasping

Single-arm and dual-arm manipulation of objects on a tabletop workspace.

## Overview

The grasping domain trains a policy to reach for an object, close the
gripper, and lift the object to a target height.  It uses the
`GraspingEnv` environment with simple kinematic simulation.

### Variants

| Variant | Flag | Workspace (m) | Description |
|---------|------|---------------|-------------|
| **Single-arm** | *(default)* | x∈[0.3, 0.8], y∈[−0.3, 0.3], z∈[0.0, 0.6] | Standard UR5e-scale workspace |
| **Dual-arm** | `--dual-arm` | x∈[0.2, 0.9], y∈[−0.5, 0.5], z∈[0.0, 0.6] | Wider workspace for two-arm coordination |

## Environment Details

### Observation (11-dim)

| Index | Name | Description |
|-------|------|-------------|
| 0–2 | `ee_position` | End-effector XYZ position |
| 3–5 | `ee_velocity` | End-effector XYZ velocity |
| 6–8 | `object_position` | Object XYZ position |
| 9 | `object_grasped` | 1.0 if grasped, else 0.0 |
| 10 | `distance` | Euclidean EE-to-object distance |

### Action (4-dim)

| Index | Name | Range | Description |
|-------|------|-------|-------------|
| 0–2 | `ee_velocity_cmd` | continuous | Velocity command applied to EE |
| 3 | `gripper_cmd` | [0, 1] | Close gripper when > 0.5 |

### Dynamics

```
ee_position += ee_velocity * dt          (dt = 0.01 s)
ee_position  = clip(ee_position, ws_lo, ws_hi)
```

The gripper closes when `gripper_cmd > 0.5` **and** the EE is within
`grasp_threshold = 0.05 m` of the object.  Once grasped, the object
follows the EE.

### Reward Shaping

```
reward  = -distance                       # dense distance signal
reward += 1.0       if grasped            # grasp bonus
reward += 2.0 * z   if grasped            # lift incentive
reward += 10.0      if lifted to target   # success bonus
```

The episode terminates when the object reaches `lift_target = 0.3 m`
(success) or `max_steps` is reached (truncation).

## Training

```bash
# Single-arm (default)
hybrid-gcs-train --env grasping --episodes 200 --seed 42

# Dual-arm
hybrid-gcs-train --env grasping --dual-arm --episodes 200 --seed 42
```

### PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `learning_rate` | 3 × 10⁻⁴ | Standard PPO default |
| `gamma` | 0.99 | Long-horizon lifting requires far-sighted discounting |
| `gae_lambda` | 0.95 | Balanced bias-variance for advantage estimation |
| `clip_ratio` | 0.2 | Prevents large policy updates |
| `entropy_coef` | 0.01 | Encourages exploration during early training |
| `epochs` | 4 | Multiple passes per rollout |
| `batch_size` | 64 | Mini-batch size for gradient steps |
| `num_steps` | 2048 | Rollout length before PPO update |

## Evaluation

```bash
hybrid-gcs-eval --env grasping --checkpoint checkpoints/train/best.pth --episodes 50
```

Reported metrics:

- **mean_reward** — average cumulative reward per episode
- **success_rate** — fraction of episodes where the object was lifted
- **mean_length** — average episode length (lower is faster)

## Visualization

### Foxglove Studio

```bash
hybrid-gcs-vis --env grasping \
    --checkpoint checkpoints/train/best.pth \
    --backend foxglove \
    --output output/grasping.mcap
```

Open the generated `.mcap` file in [Foxglove Studio](https://studio.foxglove.dev).
Markers:

- **Blue sphere** — end-effector
- **Orange sphere** — object

### PyBullet

```bash
hybrid-gcs-vis --env grasping \
    --checkpoint checkpoints/train/best.pth \
    --backend pybullet
```

A GUI window shows the table, EE marker (sphere), and object (cube)
moving in real time.  Use `--offscreen` for headless rendering.

## Python API

```python
from hybrid_gcs.environments import GraspingEnv, GraspingConfig
from hybrid_gcs.training import PolicyNetwork, PolicyNetworkConfig

# Create environment
config = GraspingConfig(max_steps=500, seed=42)
env = GraspingEnv(config)

# Create policy
pol_cfg = PolicyNetworkConfig(state_dim=env.observation_dim, action_dim=env.action_dim)
policy = PolicyNetwork(pol_cfg)

# Run one episode
obs = env.reset()
done = False
while not done:
    action, value = policy.get_action(obs)
    result = env.step(action)
    obs, done = result.observation, result.done or result.truncated
```

## References

- Calli, B. et al. (2015). *YCB Object and Model Set.*
- Schulman, J. et al. (2017). *Proximal Policy Optimization Algorithms.*
