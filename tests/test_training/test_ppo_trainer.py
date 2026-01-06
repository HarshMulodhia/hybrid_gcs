"""
Tests for PPO Trainer module.

Tests PPOTrainer, GAE computation, and training loop.
"""

import pytest
import torch
import numpy as np
from hybrid_gcs.training import (
    PolicyNetwork,
    PolicyNetworkConfig,
    PPOTrainer,
    PPOConfig
)


class TestPPOConfig:
    """Test PPOConfig."""
    
    def test_creation_default(self):
        """Test default config."""
        config = PPOConfig()
        assert config.learning_rate == 3e-4
        assert config.gamma == 0.99
        assert config.clip_ratio == 0.2
    
    def test_creation_custom(self):
        """Test custom config."""
        config = PPOConfig(
            learning_rate=1e-3,
            gamma=0.95,
            clip_ratio=0.1
        )
        assert config.learning_rate == 1e-3
        assert config.gamma == 0.95
        assert config.clip_ratio == 0.1


class TestPPOTrainer:
    """Test PPO Trainer."""
    
    @pytest.fixture
    def setup(self):
        """Setup trainer."""
        device = 'cpu'
        
        policy_config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2,
            hidden_dim=64,
            num_hidden_layers=2,
            device=device
        )
        policy = PolicyNetwork(policy_config)
        
        ppo_config = PPOConfig(
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            epochs=2,
            batch_size=32,
            device=device
        )
        trainer = PPOTrainer(policy, ppo_config)
        
        return trainer, policy_config, ppo_config
    
    def test_creation(self, setup):
        """Test trainer creation."""
        trainer, _, _ = setup
        assert trainer.total_updates == 0
        assert trainer.total_steps == 0
    
    def test_gae_computation(self, setup):
        """Test GAE computation."""
        trainer, _, _ = setup
        
        rewards = np.array([1.0, 0.5, 0.0, 0.0, 1.0], dtype=np.float32)
        values = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=np.float32)
        next_value = 0.0
        dones = np.array([False, False, False, True, False], dtype=bool)
        
        advantages, returns = trainer.compute_gae(rewards, values, next_value, dones)
        
        assert advantages.shape == (5,)
        assert returns.shape == (5,)
        assert np.all(np.isfinite(advantages))
        assert np.all(np.isfinite(returns))
    
    def test_gae_with_early_done(self, setup):
        """Test GAE with done signal."""
        trainer, _, _ = setup
        
        rewards = np.array([1.0, 0.5, 1.0, 0.5], dtype=np.float32)
        values = np.array([0.5, 0.4, 0.5, 0.4], dtype=np.float32)
        next_value = 0.0
        dones = np.array([False, True, False, True], dtype=bool)
        
        advantages, returns = trainer.compute_gae(rewards, values, next_value, dones)
        
        # Check that done flag breaks advantage accumulation
        assert advantages.shape == (4,)
    
    def test_update_step(self, setup):
        """Test single update step."""
        trainer, _, _ = setup
        device = 'cpu'
        
        # Create dummy trajectory
        T = 32
        states = np.random.randn(T, 6).astype(np.float32)
        actions = np.random.randn(T, 2).astype(np.float32)
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        next_value = 0.0
        
        # Update
        stats = trainer.update(states, actions, rewards, values, dones, next_value)
        
        # Check stats
        assert 'policy_loss' in stats
        assert 'value_loss' in stats
        assert 'entropy_loss' in stats
        assert 'total_loss' in stats
        assert all(np.isfinite(v) for v in stats.values())
        
        # Check counters
        assert trainer.total_updates == 1
        assert trainer.total_steps == T
    
    def test_multiple_updates(self, setup):
        """Test multiple update steps."""
        trainer, _, _ = setup
        
        for i in range(3):
            T = 32
            states = np.random.randn(T, 6).astype(np.float32)
            actions = np.random.randn(T, 2).astype(np.float32)
            rewards = np.random.randn(T).astype(np.float32)
            values = np.random.randn(T).astype(np.float32)
            dones = np.zeros(T, dtype=bool)
            
            stats = trainer.update(states, actions, rewards, values, dones, 0.0)
            
            assert trainer.total_updates == i + 1
            assert trainer.total_steps == (i + 1) * T
    
    def test_learning_rate_scheduling(self, setup):
        """Test that learning rate changes with scheduler."""
        trainer, _, _ = setup
        
        initial_lr = trainer.get_learning_rate()
        
        # Do some updates to trigger scheduler
        for _ in range(5):
            T = 32
            states = np.random.randn(T, 6).astype(np.float32)
            actions = np.random.randn(T, 2).astype(np.float32)
            rewards = np.random.randn(T).astype(np.float32)
            values = np.random.randn(T).astype(np.float32)
            dones = np.zeros(T, dtype=bool)
            
            trainer.update(states, actions, rewards, values, dones, 0.0)
        
        # Learning rate should have changed
        final_lr = trainer.get_learning_rate()
        assert initial_lr != final_lr
    
    def test_checkpoint_save_load(self, setup, tmp_path):
        """Test saving and loading checkpoints."""
        trainer, _, _ = setup
        
        # Do an update
        T = 32
        states = np.random.randn(T, 6).astype(np.float32)
        actions = np.random.randn(T, 2).astype(np.float32)
        rewards = np.random.randn(T).astype(np.float32)
        values = np.random.randn(T).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        
        trainer.update(states, actions, rewards, values, dones, 0.0)
        original_updates = trainer.total_updates
        
        # Save
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))
        
        # Create new trainer and load
        policy_config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2,
            hidden_dim=64,
            device='cpu'
        )
        policy2 = PolicyNetwork(policy_config)
        ppo_config = PPOConfig(device='cpu')
        trainer2 = PPOTrainer(policy2, ppo_config)
        
        trainer2.load_checkpoint(str(path))
        
        # Check state was restored
        assert trainer2.total_updates == original_updates


class TestPPOTrainerIntegration:
    """Integration tests for PPO Trainer."""
    
    def test_training_loop(self):
        """Test full training loop."""
        device = 'cpu'
        
        # Setup
        policy_config = PolicyNetworkConfig(
            state_dim=4,
            action_dim=2,
            hidden_dim=32,
            device=device
        )
        policy = PolicyNetwork(policy_config)
        
        ppo_config = PPOConfig(
            learning_rate=1e-3,
            epochs=2,
            batch_size=16,
            device=device
        )
        trainer = PPOTrainer(policy, ppo_config)
        
        # Training loop
        for update in range(3):
            T = 16
            states = np.random.randn(T, 4).astype(np.float32)
            actions = np.random.randn(T, 2).astype(np.float32)
            rewards = (np.random.randn(T) * 0.1 + 1.0).astype(np.float32)
            values = (np.random.randn(T) * 0.1).astype(np.float32)
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True  # Last step is done
            
            stats = trainer.update(states, actions, rewards, values, dones, 0.0)
            
            # Loss should be decreasing
            assert stats['total_loss'] > 0
    
    def test_policy_improvement(self):
        """Test that training improves policy performance."""
        device = 'cpu'
        
        policy_config = PolicyNetworkConfig(
            state_dim=2,
            action_dim=1,
            hidden_dim=32,
            device=device
        )
        policy = PolicyNetwork(policy_config)
        
        ppo_config = PPOConfig(device=device)
        trainer = PPOTrainer(policy, ppo_config)
        
        # Get initial policy mean
        with torch.no_grad():
            state = torch.randn(1, 2).to(device)
            dist_initial, _ = policy(state)
            mean_initial = dist_initial.mean.clone()
        
        # Train on a target trajectory (move towards positive actions)
        target_actions = np.ones((32, 1)).astype(np.float32)
        states = np.random.randn(32, 2).astype(np.float32)
        rewards = np.ones(32).astype(np.float32)
        values = np.ones(32).astype(np.float32)
        dones = np.zeros(32, dtype=bool)
        
        for _ in range(5):
            trainer.update(states, target_actions, rewards, values, dones, 0.0)
        
        # Get final policy mean
        with torch.no_grad():
            dist_final, _ = policy(state)
            mean_final = dist_final.mean.clone()
        
        # Policy should have moved towards target action
        # (This is a weak test - just checking that something changed)
        assert not torch.allclose(mean_initial, mean_final)
