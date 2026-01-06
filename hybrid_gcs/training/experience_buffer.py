"""
Experience Buffer for Hybrid-GCS.

Stores and samples trajectories for RL training.
Supports efficient sampling and trajectory replay.
"""

from typing import Dict
import numpy as np
import torch
from collections import deque


class ExperienceBuffer:
    """
    Circular buffer for storing experience trajectories.
    
    Efficiently stores states, actions, rewards, and other trajectory data.
    Supports random and sequential sampling.
    """
    
    def __init__(self, capacity: int = 10000, state_dim: int = 6,
                 action_dim: int = 2, device: str = 'cpu'):
        """
        Initialize experience buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state observations
            action_dim: Dimension of actions
            device: Device for tensors (cpu/cuda)
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        
        # Pre-allocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, log_prob: float = 0.0,
            value: float = 0.0):
        """
        Add single transition to buffer.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
            log_prob: Log probability of action under policy
            value: Value estimate for state
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        self.log_probs[self.position] = log_prob
        self.values[self.position] = value
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_trajectory(self, trajectory: Dict[str, np.ndarray]):
        """
        Add complete trajectory to buffer.
        
        Args:
            trajectory: Dictionary with keys:
                - 'states': [T, state_dim]
                - 'actions': [T, action_dim]
                - 'rewards': [T]
                - 'dones': [T]
                - 'log_probs': [T] (optional)
                - 'values': [T] (optional)
        """
        T = len(trajectory['states'])
        
        for t in range(T):
            self.add(
                state=trajectory['states'][t],
                action=trajectory['actions'][t],
                reward=trajectory['rewards'][t],
                next_state=trajectory['states'][t + 1] if t + 1 < T
                          else np.zeros(self.state_dim),
                done=trajectory['dones'][t],
                log_prob=trajectory.get('log_probs', np.zeros(T))[t],
                value=trajectory.get('values', np.zeros(T))[t]
            )
    
    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Dictionary of batched tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device)
        }
    
    def sample_sequence(self, length: int, batch_size: int = 1) \
            -> Dict[str, torch.Tensor]:
        """
        Sample sequential trajectories.
        
        Args:
            length: Length of sequence
            batch_size: Number of sequences to sample
        
        Returns:
            Dictionary of batched sequences
        """
        sequences = []
        
        for _ in range(batch_size):
            # Random starting position
            start_idx = np.random.randint(0, max(1, self.size - length))
            
            # Get sequential indices
            indices = np.arange(start_idx, min(start_idx + length, self.size))
            sequences.append(indices)
        
        # Collect all indices
        all_indices = np.concatenate(sequences)
        
        return {
            'states': torch.FloatTensor(self.states[all_indices]).to(self.device)
                      .reshape(batch_size, length, self.state_dim),
            'actions': torch.FloatTensor(self.actions[all_indices]).to(self.device)
                       .reshape(batch_size, length, self.action_dim),
            'rewards': torch.FloatTensor(self.rewards[all_indices]).to(self.device)
                       .reshape(batch_size, length),
            'dones': torch.BoolTensor(self.dones[all_indices]).to(self.device)
                     .reshape(batch_size, length)
        }
    
    def get_all(self) -> Dict[str, torch.Tensor]:
        """
        Get all stored transitions.
        
        Returns:
            Dictionary of all stored data
        """
        return {
            'states': torch.FloatTensor(self.states[:self.size]).to(self.device),
            'actions': torch.FloatTensor(self.actions[:self.size]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[:self.size]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[:self.size]).to(self.device),
            'dones': torch.BoolTensor(self.dones[:self.size]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[:self.size]).to(self.device),
            'values': torch.FloatTensor(self.values[:self.size]).to(self.device)
        }
    
    def clear(self):
        """Clear buffer."""
        self.position = 0
        self.size = 0
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.size >= self.capacity
    
    def __len__(self) -> int:
        """Get buffer size."""
        return self.size
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ExperienceBuffer("
            f"size={self.size}/{self.capacity}, "
            f"state_dim={self.state_dim}, "
            f"action_dim={self.action_dim})"
        )


class PrioritizedExperienceBuffer(ExperienceBuffer):
    """
    Prioritized Experience Replay (PER).
    
    Samples transitions based on their TD-error (priority).
    More important transitions are sampled more frequently.
    """
    
    def __init__(self, capacity: int = 10000, state_dim: int = 6,
                 action_dim: int = 2, device: str = 'cpu',
                 alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized buffer.
        
        Args:
            capacity: Buffer capacity
            state_dim: State dimension
            action_dim: Action dimension
            device: Device for tensors
            alpha: Exponent determining how much prioritization is used (0=no priority)
            beta: Importance sampling exponent (0=no IS correction)
        """
        super().__init__(capacity, state_dim, action_dim, device)
        
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
    
    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool, log_prob: float = 0.0,
            value: float = 0.0, priority: float = None):
        """
        Add transition with priority.
        
        Args:
            state: State observation
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Done flag
            log_prob: Log probability
            value: Value estimate
            priority: Priority for sampling (default: max)
        """
        super().add(state, action, reward, next_state, done, log_prob, value)
        
        # Set priority
        if priority is None:
            priority = self.max_priority
        
        self.priorities[self.position - 1] = priority ** self.alpha
    
    def sample_batch(self, batch_size: int) -> Dict:
        """
        Sample batch using priorities.
        
        Args:
            batch_size: Number of samples
        
        Returns:
            Dictionary with batch data and importance weights
        """
        # Compute sampling probabilities
        probs = self.priorities[:self.size] / np.sum(self.priorities[:self.size])
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # Importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Get batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.BoolTensor(self.dones[indices]).to(self.device),
            'log_probs': torch.FloatTensor(self.log_probs[indices]).to(self.device),
            'values': torch.FloatTensor(self.values[indices]).to(self.device),
            'weights': torch.FloatTensor(weights).to(self.device),
            'indices': indices
        }
        
        return batch
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD-errors.
        
        Args:
            indices: Indices of transitions
            td_errors: Temporal difference errors
        """
        priorities = np.abs(td_errors) + 1e-6  # Add small epsilon for stability
        
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority ** self.alpha
            self.max_priority = max(self.max_priority, priority)


class TrajectoryBuffer:
    """
    Stores complete episodes/trajectories.
    
    Useful for on-policy algorithms like PPO.
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        Initialize trajectory buffer.
        
        Args:
            device: Device for tensors
        """
        self.device = torch.device(device)
        self.trajectories = deque()
        self.total_steps = 0
    
    def add_trajectory(self, trajectory: Dict[str, np.ndarray]):
        """
        Add trajectory to buffer.
        
        Args:
            trajectory: Dictionary with trajectory data
        """
        self.trajectories.append(trajectory)
        self.total_steps += len(trajectory['states'])
    
    def sample_trajectory(self) -> Dict[str, torch.Tensor]:
        """
        Sample random trajectory.
        
        Returns:
            Random trajectory as tensors
        """
        if not self.trajectories:
            raise ValueError("Buffer is empty")
        
        idx = np.random.randint(len(self.trajectories))
        traj = self.trajectories[idx]
        
        return {
            'states': torch.FloatTensor(traj['states']).to(self.device),
            'actions': torch.FloatTensor(traj['actions']).to(self.device),
            'rewards': torch.FloatTensor(traj['rewards']).to(self.device),
            'dones': torch.BoolTensor(traj['dones']).to(self.device),
            'log_probs': torch.FloatTensor(traj.get('log_probs',
                                                     np.zeros(len(traj['states']))))
                          .to(self.device),
            'values': torch.FloatTensor(traj.get('values',
                                                 np.zeros(len(traj['states']))))
                      .to(self.device)
        }
    
    def clear(self):
        """Clear buffer."""
        self.trajectories.clear()
        self.total_steps = 0
    
    def __len__(self) -> int:
        """Get number of stored trajectories."""
        return len(self.trajectories)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TrajectoryBuffer("
            f"trajectories={len(self.trajectories)}, "
            f"total_steps={self.total_steps})"
        )
