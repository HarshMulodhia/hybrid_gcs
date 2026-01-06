"""
Reinforcement Learning Policy Training Example.

Demonstrates comprehensive RL training for robotic control tasks including:
- Environment setup and interaction
- Policy network training (PPO algorithm)
- Reward shaping and curriculum learning
- Evaluation and performance monitoring
- Model checkpointing and recovery

Reference: Hybrid-GCS-Build-Guide.md
"""

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json


@dataclass
class TrainingConfig:
    """Training configuration."""
    algorithm: str = 'ppo'
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 1.0
    
    total_timesteps: int = 1000000
    num_envs: int = 4
    steps_per_episode: int = 2048
    batch_size: int = 64
    num_epochs: int = 10
    
    eval_freq: int = 10000
    save_freq: int = 50000
    checkpoint_dir: str = './checkpoints/rl'
    log_dir: str = './logs/rl_training'


class PolicyNetwork(nn.Module):
    """Policy network for continuous control."""
    
    def __init__(self, state_dim: int, action_dim: int,
                hidden_sizes: List[int] = None):
        """
        Initialize policy network.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes
        """
        super().__init__()
        
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        
        # Build network
        layers = []
        input_size = state_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        # Policy head
        self.backbone = nn.Sequential(*layers)
        self.policy_mean = nn.Linear(input_size, action_dim)
        self.policy_log_std = nn.Parameter(
            torch.zeros(action_dim)
        )
        
        # Value head
        self.value_head = nn.Linear(input_size, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: State tensor
        
        Returns:
            Tuple of (action_mean, value)
        """
        features = self.backbone(state)
        action_mean = self.policy_mean(features)
        value = self.value_head(features)
        return action_mean, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from policy.
        
        Args:
            state: State array
            deterministic: Use deterministic policy
        
        Returns:
            Action array
        """
        state_tensor = torch.from_numpy(state).float()
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)
        
        with torch.no_grad():
            action_mean, _ = self(state_tensor)
        
        if deterministic:
            action = action_mean
        else:
            std = torch.exp(self.policy_log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action = dist.sample()
        
        return action.cpu().numpy()[0]
    
    def compute_loss(self, states: torch.Tensor, actions: torch.Tensor,
                    returns: torch.Tensor, advantages: torch.Tensor,
                    old_log_probs: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Compute PPO loss.
        
        Args:
            states: State batch
            actions: Action batch
            returns: Returns batch
            advantages: Advantages batch
            old_log_probs: Old log probabilities
        
        Returns:
            Loss and metrics
        """
        action_mean, values = self(states)
        
        # Policy loss
        std = torch.exp(self.policy_log_std)
        dist = torch.distributions.Normal(action_mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        ratio = torch.exp(log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_range,
                                   1 + self.clip_range)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(-1), returns)
        
        # Entropy bonus
        entropy = dist.entropy().sum(dim=-1).mean()
        
        # Total loss
        loss = policy_loss + self.value_loss_coeff * value_loss - self.entropy_coeff * entropy
        
        return loss, {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item()
        }


class RobotEnvironment:
    """Simulated robot environment."""
    
    def __init__(self, state_dim: int = 20, action_dim: int = 7):
        """
        Initialize environment.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.state = np.zeros(state_dim)
        self.goal = np.random.randn(state_dim) * 0.5
        self.step_count = 0
        self.max_steps = 100
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        self.state = np.random.randn(self.state_dim) * 0.1
        self.goal = np.random.randn(self.state_dim) * 0.5
        self.step_count = 0
        return self.state.copy()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.
        
        Args:
            action: Action to execute
        
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Clamp action
        action = np.clip(action, -1.0, 1.0)
        
        # Update state (simple dynamics)
        self.state = self.state + action * 0.1
        self.step_count += 1
        
        # Compute reward
        distance = np.linalg.norm(self.state - self.goal)
        reward = -distance  # Negative distance as reward
        
        # Bonus for reaching goal
        if distance < 0.1:
            reward += 10.0
        
        # Action penalty
        reward -= 0.01 * np.linalg.norm(action)
        
        # Episode done
        done = (self.step_count >= self.max_steps) or (distance < 0.1)
        
        info = {
            'distance_to_goal': float(distance),
            'success': distance < 0.1
        }
        
        return self.state.copy(), reward, done, info


class ExperienceBuffer:
    """Buffer for storing experience."""
    
    def __init__(self, buffer_size: int = 2048):
        """
        Initialize buffer.
        
        Args:
            buffer_size: Maximum buffer size
        """
        self.buffer_size = buffer_size
        self.clear()
    
    def clear(self) -> None:
        """Clear buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.log_probs = []
    
    def add(self, state: np.ndarray, action: np.ndarray,
           reward: float, value: float, done: bool,
           log_prob: float) -> None:
        """Add experience."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        self.log_probs.append(log_prob)
    
    def compute_returns_and_advantages(self, gamma: float = 0.99,
                                      gae_lambda: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages using GAE.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        
        Returns:
            Tuple of (returns, advantages)
        """
        returns = np.zeros(len(self.rewards))
        advantages = np.zeros(len(self.rewards))
        
        gae = 0
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * self.values[t + 1] * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages[t] = gae
            returns[t] = gae + self.values[t]
        
        return returns, advantages
    
    def get_batches(self, batch_size: int) -> List[Dict]:
        """
        Get minibatches for training.
        
        Args:
            batch_size: Batch size
        
        Returns:
            List of batches
        """
        returns, advantages = self.compute_returns_and_advantages()
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        indices = np.arange(len(self.states))
        np.random.shuffle(indices)
        
        batches = []
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i + batch_size]
            
            batches.append({
                'states': torch.from_numpy(np.array([self.states[j] for j in batch_indices])).float(),
                'actions': torch.from_numpy(np.array([self.actions[j] for j in batch_indices])).float(),
                'returns': torch.from_numpy(returns[batch_indices]).float(),
                'advantages': torch.from_numpy(advantages[batch_indices]).float(),
                'log_probs': torch.from_numpy(np.array([self.log_probs[j] for j in batch_indices])).float()
            })
        
        return batches


class RLTrainer:
    """RL training loop."""
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.env = RobotEnvironment()
        
        self.policy = PolicyNetwork(
            state_dim=self.env.state_dim,
            action_dim=self.env.action_dim
        )
        
        self.optimizer = Adam(self.policy.parameters(),
                            lr=config.learning_rate)
        
        # Create checkpoint directory
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        self.total_timesteps = 0
        self.episode_count = 0
        self.best_reward = -np.inf
        self.training_history = []
    
    def collect_experience(self, num_steps: int) -> ExperienceBuffer:
        """
        Collect experience from environment.
        
        Args:
            num_steps: Number of steps to collect
        
        Returns:
            Experience buffer
        """
        buffer = ExperienceBuffer(num_steps)
        
        state = self.env.reset()
        episode_reward = 0
        episode_info = {'distance': 0.0, 'success': False}
        
        for _ in range(num_steps):
            # Get action from policy
            state_tensor = torch.from_numpy(state).float()
            with torch.no_grad():
                action_mean, value = self.policy(state_tensor.unsqueeze(0))
            
            # Sample action
            std = torch.exp(self.policy.policy_log_std)
            dist = torch.distributions.Normal(action_mean, std)
            action_sampled = dist.sample()
            log_prob = dist.log_prob(action_sampled).sum().item()
            
            action = action_sampled.squeeze(0).numpy()
            
            # Step environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store experience
            buffer.add(
                state=state,
                action=action,
                reward=reward,
                value=value.item(),
                done=done,
                log_prob=log_prob
            )
            
            episode_reward += reward
            episode_info['distance'] = info['distance_to_goal']
            episode_info['success'] = info['success']
            
            self.total_timesteps += 1
            
            if done:
                self.episode_count += 1
                state = self.env.reset()
                
                if self.episode_count % 10 == 0:
                    print(f"Episode {self.episode_count}: reward={episode_reward:.3f}, "
                          f"distance={episode_info['distance']:.3f}")
                
                episode_reward = 0
            else:
                state = next_state
        
        return buffer
    
    def train_epoch(self, buffer: ExperienceBuffer) -> Dict:
        """
        Train for one epoch.
        
        Args:
            buffer: Experience buffer
        
        Returns:
            Training metrics
        """
        batches = buffer.get_batches(self.config.batch_size)
        
        losses = []
        for batch in batches:
            # Forward pass
            loss, metrics = self.policy.compute_loss(
                states=batch['states'],
                actions=batch['actions'],
                returns=batch['returns'],
                advantages=batch['advantages'],
                old_log_probs=batch['log_probs']
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()
            
            losses.append(loss.item())
        
        return {'loss': np.mean(losses)}
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """
        Evaluate policy.
        
        Args:
            num_episodes: Number of evaluation episodes
        
        Returns:
            Evaluation metrics
        """
        rewards = []
        successes = 0
        
        for _ in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action = self.policy.get_action(state, deterministic=True)
                state, reward, done, info = self.env.step(action)
                episode_reward += reward
                
                if info['success']:
                    successes += 1
            
            rewards.append(episode_reward)
        
        mean_reward = np.mean(rewards)
        success_rate = successes / num_episodes
        
        return {
            'mean_reward': mean_reward,
            'success_rate': success_rate,
            'std_reward': np.std(rewards)
        }
    
    def save_checkpoint(self, name: str = 'latest') -> None:
        """
        Save checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f'{name}.pth'
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_timesteps': self.total_timesteps,
            'episode_count': self.episode_count
        }, checkpoint_path)
        
        print(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, name: str = 'latest') -> None:
        """
        Load checkpoint.
        
        Args:
            name: Checkpoint name
        """
        checkpoint_path = Path(self.config.checkpoint_dir) / f'{name}.pth'
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            self.policy.load_state_dict(checkpoint['policy_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.total_timesteps = checkpoint['total_timesteps']
            self.episode_count = checkpoint['episode_count']
            print(f"Loaded checkpoint: {checkpoint_path}")
    
    def train(self) -> None:
        """Run training loop."""
        print("="*70)
        print("RL TRAINING")
        print("="*70)
        print(f"Algorithm: {self.config.algorithm}")
        print(f"Total timesteps: {self.config.total_timesteps}")
        print(f"Learning rate: {self.config.learning_rate}")
        print("="*70 + "\n")
        
        while self.total_timesteps < self.config.total_timesteps:
            # Collect experience
            print(f"\nCollecting experience... (total: {self.total_timesteps})")
            buffer = self.collect_experience(self.config.steps_per_episode)
            
            # Train epochs
            print(f"Training {self.config.num_epochs} epochs...")
            for epoch in range(self.config.num_epochs):
                metrics = self.train_epoch(buffer)
                if epoch % max(1, self.config.num_epochs // 5) == 0:
                    print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}")
            
            # Evaluate
            if self.total_timesteps % self.config.eval_freq == 0:
                eval_metrics = self.evaluate()
                print(f"\nEvaluation:")
                print(f"  Mean reward: {eval_metrics['mean_reward']:.3f}")
                print(f"  Success rate: {eval_metrics['success_rate']:.1%}")
                
                self.training_history.append({
                    'timesteps': self.total_timesteps,
                    'mean_reward': eval_metrics['mean_reward'],
                    'success_rate': eval_metrics['success_rate']
                })
                
                # Save best model
                if eval_metrics['mean_reward'] > self.best_reward:
                    self.best_reward = eval_metrics['mean_reward']
                    self.save_checkpoint('best')
            
            # Save checkpoint
            if self.total_timesteps % self.config.save_freq == 0:
                self.save_checkpoint('latest')
        
        print("\n" + "="*70)
        print("TRAINING COMPLETED")
        print("="*70)
        print(f"Total timesteps: {self.total_timesteps}")
        print(f"Episodes: {self.episode_count}")
        print(f"Best reward: {self.best_reward:.3f}")
        
        # Save training history
        history_path = Path(self.config.log_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")


def main():
    """Main execution."""
    # Create config
    config = TrainingConfig(
        total_timesteps=100000,
        eval_freq=5000,
        save_freq=25000
    )
    
    # Create trainer
    trainer = RLTrainer(config)
    
    # Train
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    trainer.load_checkpoint('best')
    final_eval = trainer.evaluate(num_episodes=20)
    print(f"Final mean reward: {final_eval['mean_reward']:.3f}")
    print(f"Final success rate: {final_eval['success_rate']:.1%}")


if __name__ == '__main__':
    main()
