# Hybrid-GCS Documentation

Welcome to the Hybrid-GCS documentation.  This guide covers the full
system — from core GCS planning algorithms through Deep-RL training,
multi-agent coordination, safety filtering, and visualization.

## Project Overview

Hybrid-GCS is a production-grade robotics planning system that combines
**Graph of Convex Sets (GCS)** trajectory planning with **Deep
Reinforcement Learning (PPO)** for adaptive, collision-free control.

The system targets three application domains:

| Domain | Environment | Variants |
|--------|-------------|----------|
| [YCB Object Grasping](grasping.md) | `GraspingEnv` | Single-arm, Dual-arm |
| [Autonomous Drone Navigation](drone_navigation.md) | `DroneNavEnv` | Single-agent, Multi-agent |
| [Complex Manipulation](manipulation.md) | `ManipulationEnv` | Reach, Pick, Push, Stack |

## Quick Start

```bash
# Install (editable, with all extras)
pip install -e ".[full]"

# Train a grasping policy
hybrid-gcs-train --env grasping --episodes 100

# Evaluate the checkpoint
hybrid-gcs-eval --env grasping --checkpoint checkpoints/train/best.pth

# Visualize in Foxglove Studio
hybrid-gcs-vis --env grasping --checkpoint checkpoints/train/best.pth --backend foxglove
```

## Package Structure

```
hybrid_gcs/
├── core/               # GCS algorithms (ConfigSpace, IRIS, MICP, Trajectory)
├── training/           # Deep RL (PolicyNetwork, PPO, reward shaping, curriculum)
├── environments/       # Gym-like task environments
├── integration/        # Hybrid blending, safety filtering, feature extraction
├── multi_agent/        # Attention-based communication, Space-Time GCS
├── visualization/      # Foxglove MCAP recorder, PyBullet renderer
└── cli/                # Command-line scripts (train, evaluate, visualize)
```

## CLI Reference

### `hybrid-gcs-train`

Train a PPO policy for any supported domain.

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | *(required)* | `grasping`, `drone_nav`, or `manipulation` |
| `--task` | `reach` | Manipulation sub-task (`reach`, `pick`, `push`, `stack`) |
| `--episodes` | `100` | Number of training episodes |
| `--max-steps` | `500` | Max environment steps per episode |
| `--seed` | `42` | Random seed for reproducibility |
| `--lr` | domain default | Override learning rate |
| `--eval-interval` | `10` | Evaluate every N episodes |
| `--output-dir` | `checkpoints/train` | Checkpoint & log directory |
| `--dual-arm` | off | Enable dual-arm workspace (grasping only) |
| `--num-agents` | `1` | Number of drone agents (drone_nav only) |
| `--num-obstacles` | `5` | Number of obstacles (drone_nav only) |

### `hybrid-gcs-eval`

Evaluate a trained checkpoint with detailed metrics.

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | *(required)* | Domain name |
| `--checkpoint` | *(required)* | Path to `.pth` checkpoint |
| `--episodes` | `20` | Evaluation episodes |
| `--output` | — | Save metrics JSON to this path |
| `--record` | off | Record per-step episode data |

### `hybrid-gcs-vis`

Replay a trained policy in PyBullet or Foxglove Studio.

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | *(required)* | Domain name |
| `--checkpoint` | *(required)* | Path to `.pth` checkpoint |
| `--backend` | `foxglove` | `pybullet` or `foxglove` |
| `--output` | `output/visualization.mcap` | Output file path |
| `--offscreen` | off | PyBullet headless rendering |

## Visualization Backends

### Foxglove Studio

The `foxglove` backend writes an **MCAP** file containing per-step
`SceneUpdate` messages.  Open the file in
[Foxglove Studio](https://studio.foxglove.dev) to scrub through the
episode timeline with full 3-D sphere markers for the end-effector,
objects, drones, and goals.

### PyBullet

The `pybullet` backend launches a real-time 3-D GUI window (requires
`pybullet` to be installed).  A table, ground plane, and proxy markers
are loaded automatically.  When PyBullet is not available the backend
falls back to saving a JSON trajectory file.

## Further Reading

| Document | Contents |
|----------|----------|
| [Architecture](architecture.md) | System design, data flow, module dependencies |
| [Core API](api/core.md) | ConfigSpace, Trajectory, IRIS, MICP |
| [Training API](api/training.md) | PolicyNetwork, PPO, reward shaping |
| [Visualization API](api/visualization.md) | FoxgloveRecorder, PyBulletRenderer |
| [YCB Grasping](grasping.md) | Single & dual-arm grasping guide |
| [Drone Navigation](drone_navigation.md) | Single & multi-agent drone guide |
| [Complex Manipulation](manipulation.md) | Reach / Pick / Push / Stack guide |
| [Getting Started](tutorials/getting_started.md) | Installation & first run tutorial |
