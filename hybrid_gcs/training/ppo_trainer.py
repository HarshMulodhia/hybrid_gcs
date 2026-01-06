"""
Proximal Policy Optimization (PPO) Trainer for Hybrid-GCS.

Implements PPO algorithm with:
- Generalized Advantage Estimation (GAE)
- Clipped objective function
- Value function bootstrapping
- Entropy regularization

Paper: Schulman et al. (2017) - PPO Algorithms
"""

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .policy_network import PolicyNetwork, PolicyNetworkConfig


@dataclass
class PPOConfig:
    """Configuration for PPO Trainer."""
    
    # Learning
    learning_rate: float = 3e-4              # Optimizer learning rate
    weight_decay: float = 1e-5               # L2 regularization
    
    # PPO hyperparameters
    gamma: float = 0.99                      # Discount factor
    gae_lambda: float = 0.95                 # GAE lambda parameter
    clip_ratio: float = 0.2                  # Policy gradient clipping ratio
    entropy_coef: float = 0.01               # Entropy bonus coefficient
    value_coef: float = 0.5                  # Value loss coefficient
    max_grad_norm: float = 0.5               # Gradient clipping
    
    # Training loop
    epochs: int = 4                          # Number of epochs per update
    batch_size: int = 64                     # Mini-batch size
    num_steps: int = 2048                    # Trajectory length before update
    
    # Device
    device: str = 'cpu'


class PPOTrainer:
    """
    Proximal Policy Optimization Trainer.
    
    Trains a policy network using PPO algorithm with GAE for advantage estimation.
    """
    
    def __init__(self, policy: PolicyNetwork, config: PPOConfig):
        """
        Initialize PPO Trainer.
        
        Args:
            policy: PolicyNetwork instance
            config: PPOConfig instance
        """
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            policy.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Scheduler (optional)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=100
        )
        
        # Stats tracking
        self.total_updates = 0
        self.total_steps = 0
    
    def compute_gae(self, rewards: np.ndarray, values: np.ndarray,
                    next_value: float, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        GAE provides a bias-variance tradeoff between TD(0) and MC methods.
        
        Args:
            rewards: Reward trajectory [T]
            values: Value estimates [T]
            next_value: Value estimate for next state after trajectory
            dones: Done flags [T]
        
        Returns:
            (advantages, returns)
            - advantages: Advantage estimates [T]
            - returns: Target values [T]
        """
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        gae = 0.0
        
        for t in reversed(range(T)):
            # Get value for next state
            if t == T - 1:
                next_val = next_value
                done = dones[t]
            else:
                next_val = values[t + 1]
                done = dones[t]
            
            # Temporal difference error
            delta = rewards[t] + self.config.gamma * next_val * (1 - done) - values[t]
            
            # GAE accumulation
            gae = delta + self.config.gamma * self.config.gae_lambda * (1 - done) * gae
            
            advantages[t] = gae
            returns[t] = gae + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               values: np.ndarray, dones: np.ndarray, next_value: float) -> Dict[str, float]:
        """
        Perform PPO update.
        
        Args:
            states: State trajectory [T, state_dim]
            actions: Action trajectory [T, action_dim]
            rewards: Reward trajectory [T]
            values: Value estimates [T]
            dones: Done flags [T]
            next_value: Value estimate for next state
        
        Returns:
            Dictionary of loss statistics
        """
        # Convert to tensors
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, next_value, dones)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        # Get old policy probabilities
        with torch.no_grad():
            old_log_probs, _, _ = self.policy.evaluate(states_t, actions_t)
            old_log_probs = old_log_probs.detach()
        
        # Training epochs
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy_loss': 0.0,
            'total_loss': 0.0,
            'clip_frac': 0.0,
            'approx_kl': 0.0
        }
        
        num_batches = max(len(states) // self.config.batch_size, 1)
        
        for epoch in range(self.config.epochs):
            # Shuffle indices
            indices = np.random.permutation(len(states))
            
            epoch_stats = {k: 0.0 for k in stats.keys()}
            
            for batch_idx in range(num_batches):
                # Get batch
                batch_indices = indices[
                    batch_idx * self.config.batch_size:
                    (batch_idx + 1) * self.config.batch_size
                ]
                
                states_batch = states_t[batch_indices]
                actions_batch = actions_t[batch_indices]
                advantages_batch = advantages_t[batch_indices]
                returns_batch = returns_t[batch_indices]
                old_log_probs_batch = old_log_probs[batch_indices]
                
                # Forward pass
                log_probs, values, entropy = self.policy.evaluate(
                    states_batch, actions_batch
                )
                values = values.squeeze(-1) if values.dim() > 1 else values
                
                # Policy loss
                ratio = torch.exp(log_probs - old_log_probs_batch)
                surr1 = ratio * advantages_batch
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio,
                                   1 + self.config.clip_ratio) * advantages_batch
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.functional.mse_loss(values, returns_batch)
                
                # Entropy regularization
                entropy_loss = -entropy
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                        self.config.max_grad_norm)
                self.optimizer.step()
                
                # Stats
                with torch.no_grad():
                    clip_frac = (torch.abs(ratio - 1.0) >
                                self.config.clip_ratio).float().mean()
                    approx_kl = (old_log_probs_batch - log_probs).mean()
                
                epoch_stats['policy_loss'] += policy_loss.item()
                epoch_stats['value_loss'] += value_loss.item()
                epoch_stats['entropy_loss'] += entropy_loss.item()
                epoch_stats['total_loss'] += total_loss.item()
                epoch_stats['clip_frac'] += clip_frac.item()
                epoch_stats['approx_kl'] += approx_kl.item()
            
            # Average over batches
            for key in epoch_stats:
                epoch_stats[key] /= num_batches
                stats[key] += epoch_stats[key]
        
        # Average over epochs
        for key in stats:
            stats[key] /= self.config.epochs
        
        # Update scheduler
        self.scheduler.step()
        
        # Update counters
        self.total_updates += 1
        self.total_steps += len(states)
        
        return stats
    
    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
    
    def save_checkpoint(self, path: str):
        """Save trainer checkpoint."""
        checkpoint = {
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_updates': self.total_updates,
            'total_steps': self.total_steps,
            "ppo_config": asdict(self.config),
            "policy_config": asdict(self.policy.config),
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str):
        """Load trainer checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore config
        if "ppo_config" in checkpoint:
            self.config = PPOConfig(**checkpoint["ppo_config"])
            self.device = torch.device(self.config.device)

        # Rebuild policy to match checkpoint architecture
        if "policy_config" in checkpoint:
            pol_cfg = PolicyNetworkConfig(**checkpoint["policy_config"])
            self.policy = PolicyNetwork(pol_cfg).to(self.device)

        # Rebuild optimizer/scheduler bound to (possibly rebuilt) policy
        self.optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

        # Load states
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.total_updates = checkpoint["total_updates"]
        self.total_steps = checkpoint["total_steps"]
        