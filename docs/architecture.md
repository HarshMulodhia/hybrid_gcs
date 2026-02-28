# Architecture

## System Overview

Hybrid-GCS combines two complementary planning paradigms:

1. **Graph of Convex Sets (GCS)** — Provides globally optimal, collision-free trajectory planning through convex decomposition and mixed-integer optimization.
2. **Deep Reinforcement Learning (PPO)** — Provides reactive, adaptive control that handles uncertainty and dynamic environments.

## Module Architecture

```
┌─────────────────────────────────────────────┐
│                Hybrid Policy                 │
│  (blends GCS planner + RL policy outputs)    │
├──────────────────┬──────────────────────────┤
│   GCS Planner    │      RL Policy           │
│                  │                          │
│  ConfigSpace     │  PolicyNetwork           │
│  IRIS Decomposer │  PPO Trainer             │
│  MICP Solver     │  Reward Shaper           │
│  Trajectory      │  Curriculum Scheduler    │
│                  │  Experience Buffer       │
├──────────────────┴──────────────────────────┤
│            Visualization (Foxglove)          │
│  MCAP recorder → Foxglove Studio 3D view    │
├─────────────────────────────────────────────┤
│              Safety Filter                   │
│  (collision checking, joint/torque limits)   │
├─────────────────────────────────────────────┤
│              Environments                    │
│  (YCB Grasping, Drone Nav, Manipulation)     │
└─────────────────────────────────────────────┘
```

## Data Flow

### GCS Planning Pipeline

1. **ConfigSpace** defines workspace bounds and joint limits.
2. **IRISDecomposer** partitions free space into convex regions (ellipsoids).
3. **GCSGraph** connects regions into a graph with edges between overlapping sets.
4. **MICPSolver** solves the shortest-path problem to produce a collision-free **Trajectory**.

### RL Training Pipeline

1. **PolicyNetwork** (Actor-Critic) maps observations to actions and value estimates.
2. **PPOTrainer** collects trajectories, computes GAE advantages, and updates the policy.
3. **RewardComposer** aggregates task-specific, safety, and efficiency rewards.
4. **CurriculumScheduler** progressively increases task difficulty during training.
5. **ExperienceBuffer** stores and samples transitions for training.

### Hybrid Integration (planned)

1. **FeatureExtractor** produces separate feature vectors for the GCS planner and RL policy.
2. **HybridPolicy** blends GCS and RL actions via weighted or hierarchical strategies.
3. **SafetyFilter** projects the blended action into a safe set before execution.

## File Dependencies

```
config_space.py
    └─► trajectory.py, iris_decomposer.py

iris_decomposer.py
    └─► collision checking (SimpleBoxObstacle)
    └─► Ellipsoid regions → GCSGraph

micp_solver.py
    └─► GCSGraph + ConfigSpace → Trajectory

policy_network.py
    └─► ppo_trainer.py (training loop)

reward_shaper.py
    └─► ppo_trainer.py (reward signal)

foxglove_recorder.py
    └─► Trajectory, SimpleBoxObstacle, Ellipsoid → MCAP file
    └─► URDF models (local primitives) → Foxglove Studio
```

## Performance Targets

| Operation | Target |
|-----------|--------|
| IRIS decomposition (2D) | < 1 s |
| MICP solve (20 regions) | 1–5 s |
| Policy forward pass (100 samples) | < 10 ms |
| PPO update (1 epoch) | < 100 ms |
| Safety filter (per action) | < 2 ms |
