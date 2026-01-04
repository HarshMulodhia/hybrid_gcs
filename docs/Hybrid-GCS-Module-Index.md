# Hybrid-GCS Module Index & Quick Reference

**Quick Navigation for Implementation**

---

## 📑 Core Module Reference

### **1. Configuration Space** (`hybrid_gcs/core/config_space.py`)

```python
class ConfigSpace:
    """Defines state/action bounds and properties"""
    
    # Key methods:
    - is_valid(q) → bool          # Check if config is valid
    - project(q) → np.ndarray     # Project to valid space
    - random_sample() → np.ndarray # Sample random config
    - distance(q1, q2) → float    # Compute distance
    - interpolate(q1, q2, t) → np.ndarray  # Linear interpolation
```

**Used by:** IRIS, MICP, trajectories  
**Test file:** `tests/test_core/test_config_space.py`  
**Example:** Every module needs ConfigSpace instance

---

### **2. Trajectory Representation** (`hybrid_gcs/core/trajectory.py`)

```python
class Trajectory:
    """Smooth trajectory through waypoints via splines"""
    
    # Key methods:
    - at_time(t) → np.ndarray              # Get config at time t
    - derivatives_at_time(t, n) → np.ndarray  # Get n-th derivative
    - length() → float                     # Total trajectory length
    - resample(n_waypoints) → Trajectory   # Resample to n points
    - smooth(factor) → Trajectory          # Apply smoothing

class BezierTrajectory:
    """Trajectory parameterized by Bezier control points"""
    
    # Key methods:
    - eval(t) → np.ndarray                 # Evaluate at parameter t
    - derivative(t) → np.ndarray           # Get first derivative
    - to_trajectory(n_samples) → Trajectory  # Convert to waypoint form
```

**Used by:** MICP solver output, RL planning, visualization  
**Test file:** `tests/test_core/test_trajectory.py`

---

### **3. IRIS Decomposition** (`hybrid_gcs/core/iris_decomposer.py`)

```python
class IRISDecomposer:
    """Decomposes space into convex regions via IRIS algorithm"""
    
    # Key methods:
    - decompose(obstacles, seeds, max_regions) → GCSGraph
    - _grow_ellipsoid(center, obstacles) → Ellipsoid
    - _build_graph(regions) → GCSGraph

class Ellipsoid:
    """Convex set represented as ellipsoid"""
    
    # Key methods:
    - contains(point) → bool
    - volume() → float
```

**Used by:** GCS graph construction  
**Test file:** `tests/test_core/test_iris_decomposer.py`  
**Reference:** Hybrid-GCS-Theory.md Section 1.3

---

### **4. MICP Solver** (`hybrid_gcs/core/micp_solver.py`)

```python
class MICPSolver:
    """Solves shortest path in GCS via Mixed-Integer Convex Programming"""
    
    # Key methods:
    - solve(start, goal, **kwargs) → Optional[Trajectory]
    - _build_problem(start, goal) → Dict
    - _solve_mosek(problem) → Optional[Dict]
    - _solve_gurobi(problem) → Optional[Dict]
    - _extract_trajectory(solution) → Trajectory
```

**Used by:** GCS planner, hybrid action selection  
**Test file:** `tests/test_core/test_micp_solver.py`  
**Solvers:** Mosek, Gurobi, SCS (free alternative)  
**Typical runtime:** 1-10 seconds for 20-50 regions

---

### **5. Collision Checker** (`hybrid_gcs/core/collision_checker.py`)

```python
class CollisionChecker:
    """Fast collision detection for workspace"""
    
    # Key methods:
    - in_collision(config) → bool
    - trajectory_collision(trajectory) → bool
    - distance_to_obstacle(config) → float
    - check_path_collision(q1, q2) → bool
```

**Used by:** IRIS, safety filter  
**Typical implementation:** FCL (Fast Collision Library)

---

### **6. Kinematics** (`hybrid_gcs/core/kinematics.py`)

```python
class RobotKinematics:
    """Forward and inverse kinematics for robot arm"""
    
    # Key methods:
    - forward(joint_config) → (position, orientation)
    - inverse(pose, q_init) → Optional[joint_config]
    - jacobian(joint_config) → np.ndarray
    - check_joint_limits(joint_config) → bool
```

**Used by:** YCB grasping, any arm control  
**Typical implementation:** PyKDL or Drake

---

## 🧠 Training Module Reference

### **7. Policy Network** (`hybrid_gcs/training/policy_network.py`)

```python
class PolicyNetwork(torch.nn.Module):
    """Actor-Critic network for continuous control"""
    
    # Key methods:
    - forward(state) → (dist, value)
    - get_action(state) → (action, log_prob)
    - get_value(state) → value

class CNNEncoder(torch.nn.Module):
    """CNN for encoding RGB-D images"""
    
    # Key methods:
    - forward(images) → features
```

**Inputs:** State [batch, state_dim]  
**Outputs:** Action distribution + value estimate  
**Architecture:** Shared trunk + separate actor/critic heads  
**Test file:** `tests/test_training/test_policy_network.py`

---

### **8. PPO Trainer** (`hybrid_gcs/training/ppo_trainer.py`)

```python
class PPOTrainer:
    """Proximal Policy Optimization training loop"""
    
    # Key methods:
    - compute_gae(rewards, values, dones) → (advantages, returns)
    - update_policy(trajectories) → Dict[loss metrics]
    
    # Key hyperparameters:
    - gamma: 0.99 (discount factor)
    - gae_lambda: 0.95 (GAE parameter)
    - clip_ratio: 0.2 (PPO clipping ε)
    - entropy_coeff: 0.01 (exploration)
```

**Training pipeline:**
1. Collect trajectories with current policy
2. Compute GAE advantages
3. Multiple epochs of gradient updates with clipping
4. Value network regression

**Test file:** `tests/test_training/test_ppo_trainer.py`

---

### **9. Reward Shaper** (`hybrid_gcs/training/reward_shaper.py`)

```python
class RewardShaper:
    """Shapes rewards for better learning"""
    
    # Key methods:
    - compute_task_reward(obs, action, next_obs) → float
    - compute_gcs_bonus(gcs_action, action) → float
    - compute_exploration_bonus(state) → float
```

**Components:**
- Task-specific reward (progress, goal)
- GCS alignment bonus (encourage following planner)
- Exploration bonus (encourage new states)
- Smoothness penalty

---

### **10. Curriculum Scheduler** (`hybrid_gcs/training/curriculum_scheduler.py`)

```python
class CurriculumScheduler:
    """Progressive difficulty scheduling"""
    
    # Key methods:
    - get_difficulty_level(iteration) → float
    - sample_task_config() → Dict
    - should_advance() → bool
```

**Strategies:**
- Increase number of objects over time
- Increase environment randomization
- Gradually reduce GCS reliance

---

### **11. Experience Buffer** (`hybrid_gcs/training/experience_buffer.py`)

```python
class ExperienceBuffer:
    """Stores and samples trajectories for training"""
    
    # Key methods:
    - append(trajectory)
    - sample_batch(batch_size) → Dict
    - get_minibatches(batch_size) → Iterator
```

**Storage:** [states, actions, rewards, dones, values, log_probs]

---

## 🔄 Integration Module Reference

### **12. Feature Extractor** (`hybrid_gcs/integration/feature_extractor.py`)

```python
class FeatureExtractor:
    """Extracts features for GCS and RL from observations"""
    
    # Key methods:
    - extract(observation) → Dict[gcs_features, rl_features]
    - _get_vision_features(observation) → np.ndarray
    - _pad_or_truncate(features, target_dim) → np.ndarray
```

**Output:**
- `gcs_features`: Low-dim (50-100) for planner
- `rl_features`: High-dim (512+) for policy

---

### **13. Hybrid Policy** (`hybrid_gcs/integration/hybrid_policy.py`)

```python
class HybridPolicy:
    """Combines GCS planner and RL policy"""
    
    # Key methods:
    - get_action(state, gcs_features, replan) → action
    - _get_gcs_action(features, replan) → action
    - _get_rl_action(state) → action
    - _blend_weighted(gcs_action, rl_action) → action
    - _blend_hierarchical(gcs_action, rl_action) → action
```

**Blending strategies:**
1. **Weighted:** `a = (1-w)*a_gcs + w*a_rl` (w increases over training)
2. **Hierarchical:** Use GCS if feasible, else RL
3. **Conflict resolution:** Priority network decides

---

### **14. Action Selector** (`hybrid_gcs/integration/action_selector.py`)

```python
class ActionSelector:
    """Selects between GCS and RL actions"""
    
    # Key methods:
    - select_action(state, gcs_action, rl_action) → action
    - should_use_gcs(state) → bool
    - compute_confidence(action) → float
```

---

### **15. Safety Filter** (`hybrid_gcs/integration/safety_filter.py`)

```python
class SafetyFilter:
    """Enforces real-time safety constraints"""
    
    # Key methods:
    - filter_action(action, state) → safe_action
    - project_to_safe_set(action) → safe_action
    - check_collision(action) → bool
    - enforce_joint_limits(action) → action
    - enforce_torque_limits(action) → action
```

**Implementation:** Fast QP solver (100Hz capable)

---

## 🎮 Environment Module Reference

### **16. Base Environment** (`hybrid_gcs/environments/base_env.py`)

```python
class HybridGCSEnv(gym.Env):
    """Abstract base for all Hybrid-GCS environments"""
    
    # Required methods:
    - reset() → obs
    - step(action) → (obs, reward, done, info)
    - get_gcs_features() → np.ndarray
    - get_rl_features() → np.ndarray
```

---

### **17. YCB Grasping** (`hybrid_gcs/environments/ycb_grasping_env.py`)

```python
class YCBGraspingEnv(HybridGCSEnv):
    """Single/dual-arm grasping with YCB objects"""
    
    # Config:
    - robot_type: 'ur5' | 'kuka' | 'dual_arm'
    - gripper_type: 'parallel' | 'anthropomorphic'
    - num_objects: 5-20
    - difficulty: 'easy' | 'medium' | 'hard'
    
    # Action: [arm_joints...] + [gripper]
    # Reward: proximity + contact + lift bonus
```

**Metrics:** Success rate, grasp quality, time-to-grasp

---

### **18. Drone Navigation** (`hybrid_gcs/environments/drone_navigation_env.py`)

```python
class DroneNavigationEnv(HybridGCSEnv):
    """Single/multi-agent drone navigation"""
    
    # Config:
    - num_drones: 1-4
    - environment_type: 'cluttered' | 'forest' | 'urban'
    - num_obstacles: 10-50
    - multi_agent: True | False
    
    # Action: [vx, vy, vz, ω_z] per drone
    # Reward: goal progress + collision avoidance
```

**Metrics:** Success rate, path length, safety

---

### **19. Manipulation** (`hybrid_gcs/environments/manipulation_env.py`)

```python
class ManipulationEnv(HybridGCSEnv):
    """Complex multi-step manipulation tasks"""
    
    # Config:
    - task: 'pick_place' | 'assembly' | 'stacking'
    - complexity: 'simple' | 'complex'
    
    # Action: End-effector velocity commands
    # Reward: Task progress + efficiency
```

**Metrics:** Success rate, task completion time, smoothness

---

## 📊 Evaluation Module Reference

### **20. Metrics** (`hybrid_gcs/evaluation/metrics.py`)

```python
class Metrics:
    """Computes performance metrics"""
    
    # Methods:
    - success_rate() → float
    - trajectory_length() → float
    - execution_time() → float
    - smoothness() → float
    - collision_count() → int
    - coverage() → float
```

---

### **21. Analyzer** (`hybrid_gcs/evaluation/analyzer.py`)

```python
class TrajectoryAnalyzer:
    """Analyzes trajectory quality"""
    
    # Methods:
    - compute_curvature(trajectory) → np.ndarray
    - compute_velocity_profile(trajectory) → np.ndarray
    - detect_anomalies(trajectory) → List
    - compare_trajectories(traj1, traj2) → Dict
```

---

## 📁 File Dependencies Map

```
config_space.py
    ↓ (used by)
trajectory.py, iris_decomposer.py, kinematics.py

iris_decomposer.py
    ↓ (uses)
collision_checker.py, config_space.py
    ↓ (creates)
GCS graph → micp_solver.py

micp_solver.py
    ↓ (outputs)
trajectory.py objects

feature_extractor.py
    ↓ (used by)
hybrid_policy.py, rl_policy.py

policy_network.py
    ↓ (used by)
ppo_trainer.py, hybrid_policy.py

ppo_trainer.py
    ↓ (updates)
policy_network.py using experience_buffer.py

hybrid_policy.py
    ↓ (uses)
micp_solver.py, policy_network.py, safety_filter.py

safety_filter.py
    ↓ (uses)
collision_checker.py, kinematics.py

environments
    ↓ (use)
All above modules
```

---

## 🚀 Implementation Order

**For minimal viable product (MVP):**

1. ✅ ConfigSpace
2. ✅ Trajectory
3. ✅ IRIS Decomposer
4. ✅ MICP Solver
5. ✅ PolicyNetwork + PPO Trainer
6. ✅ Feature Extractor
7. ✅ Hybrid Policy
8. ✅ YCB Grasping Environment
9. ✅ Metrics
10. ✅ CLI tools

**For full system:**

Add: Safety Filter, Curriculum, Conflict Resolver, Multi-environment support, Hardware interface

---

## 🔗 Cross-Module Interfaces

### **GCS → Trajectory**
```python
trajectory = micp_solver.solve(start, goal)
waypoint = trajectory.at_time(t)
deriv = trajectory.derivatives_at_time(t, n=1)
```

### **Feature Extractor → Hybrid Policy**
```python
features_dict = feature_extractor.extract(obs)
action = hybrid_policy.get_action(
    state=features_dict['rl'],
    gcs_features=features_dict['gcs']
)
```

### **PPO Trainer → PolicyNetwork**
```python
dist, value = policy_network(state_batch)
loss = ppo_trainer.compute_loss(dist, value, advantages)
```

### **Safety Filter → Collision Checker**
```python
safe_action = safety_filter.filter_action(action, state)
# Internally:
is_safe = collision_checker.in_collision(next_state)
```

---

## ⚡ Performance Targets

| Module | Operation | Target Time |
|--------|-----------|------------|
| IRIS | Decompose 2D space | <1 second |
| MICP | Solve 20 regions | 1-5 seconds |
| Feature Extract | Per observation | <1 ms |
| Policy Forward | 100 samples | <10 ms |
| PPO Update | 1 epoch | <100 ms |
| Safety Filter | Per action | <2 ms |
| **Total (realtime)** | Control cycle | <10 ms |

---

## 📞 Module Troubleshooting

### **IRIS decomposition slow?**
→ Reduce max_iterations, use fewer seed points, disable visualization

### **MICP solver infeasible?**
→ Add more regions, check collision detector, increase solver time limit

### **RL training not improving?**
→ Check reward shaping, increase curriculum difficulty, tune learning rate

### **High latency in closed-loop?**
→ Use hierarchical policy (GCS only when needed), reduce feature dimension, batch inference

### **Safety filter rejecting too many actions?**
→ Loosen constraints, use control barrier functions instead of hard constraints

---

**This quick reference lets you jump between modules efficiently during implementation!**

