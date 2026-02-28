# Training API Reference

## PolicyNetwork

```python
from hybrid_gcs.training import PolicyNetwork, PolicyNetworkConfig
```

### PolicyNetworkConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `state_dim` | int | — | Observation dimension |
| `action_dim` | int | — | Action dimension |
| `hidden_dim` | int | 256 | Hidden layer width |
| `num_hidden_layers` | int | 3 | Number of hidden layers |
| `activation` | str | `'relu'` | `'relu'` or `'tanh'` |
| `use_cnn` | bool | False | Enable CNN encoder for images |
| `log_std_init` | float | 0.0 | Initial log standard deviation |
| `device` | str | `'cpu'` | `'cpu'` or `'cuda'` |

### PolicyNetwork Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(state) → (Normal, value)` | Forward pass returning distribution + value |
| `get_action` | `(state, deterministic) → (action, value)` | NumPy interface for single action |
| `evaluate` | `(states, actions) → (log_probs, values, entropy)` | Evaluate batch for PPO |
| `get_value` | `(state) → value` | Value estimate only |

### PolicyNetworkWithLSTM

Same interface with additional LSTM hidden state for sequential observations.

---

## PPOTrainer

```python
from hybrid_gcs.training import PPOTrainer, PPOConfig
```

### PPOConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `learning_rate` | float | 3e-4 | Optimizer learning rate |
| `gamma` | float | 0.99 | Discount factor |
| `gae_lambda` | float | 0.95 | GAE lambda |
| `clip_ratio` | float | 0.2 | PPO clipping epsilon |
| `entropy_coef` | float | 0.01 | Entropy bonus |
| `value_coef` | float | 0.5 | Value loss weight |
| `epochs` | int | 4 | Epochs per update |
| `batch_size` | int | 64 | Mini-batch size |

### PPOTrainer Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `compute_gae` | `(rewards, values, next_value, dones) → (advantages, returns)` | GAE computation |
| `update` | `(states, actions, rewards, values, dones, next_value) → stats` | Full PPO update |
| `save_checkpoint` | `(path)` | Save model + optimizer state |
| `load_checkpoint` | `(path)` | Restore from checkpoint |

---

## RewardShaper

```python
from hybrid_gcs.training import (
    RewardComposer, DistanceReward, GoalReachReward,
    ActionPenalty, CollisionPenalty, SmoothnessReward,
    EfficiencyReward, RewardConfig, create_reward_composer
)
```

### Built-in Strategies

| Strategy | Key Args | Formula |
|----------|----------|---------|
| `DistanceReward` | `current_position, goal_position` | `-scale * ‖pos - goal‖` |
| `GoalReachReward` | `current_position, goal_position` | `bonus if ‖pos - goal‖ < threshold` |
| `ActionPenalty` | `action` | `-scale * ‖action‖²` |
| `CollisionPenalty` | `collision` (bool) | `-penalty if collision` |
| `SmoothnessReward` | `current_velocity, previous_velocity` | `-scale * ‖acceleration‖` |
| `EfficiencyReward` | `position_delta` | `-scale * ‖delta‖` |

### RewardComposer

Compose multiple strategies. Each strategy is called with its own keyword arguments.

```python
composer = RewardComposer()
composer.add_strategy(DistanceReward(weight=1.0, scale=0.1), "distance")
reward = composer.compute_reward(current_position=pos, goal_position=goal)
```

---

## CurriculumScheduler

```python
from hybrid_gcs.training import CurriculumManager, LinearCurriculum
```

| Schedule | Description |
|----------|-------------|
| `LinearCurriculum` | Linear ramp from initial to final difficulty |
| `ExponentialCurriculum` | Exponential growth |
| `StepCurriculum` | Discrete difficulty levels at milestones |
| `SigmoidCurriculum` | S-shaped progression |
| `PerformanceCurriculum` | Adapts based on agent success rate |

---

## ExperienceBuffer

```python
from hybrid_gcs.training import ExperienceBuffer, PrioritizedExperienceBuffer, TrajectoryBuffer
```

| Class | Description |
|-------|-------------|
| `ExperienceBuffer` | Fixed-capacity circular buffer for transitions |
| `PrioritizedExperienceBuffer` | Sampling weighted by TD-error priority |
| `TrajectoryBuffer` | Stores complete episodes for on-policy methods |
