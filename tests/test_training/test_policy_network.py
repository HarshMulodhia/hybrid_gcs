"""
Tests for Policy Network module.

Tests PolicyNetwork, CNNEncoder, and PolicyNetworkWithLSTM.
"""

import pytest
import torch
import numpy as np
from hybrid_gcs.training import (
    PolicyNetwork,
    PolicyNetworkConfig,
    PolicyNetworkWithLSTM,
    CNNEncoder
)


class TestPolicyNetworkConfig:
    """Test PolicyNetworkConfig."""
    
    def test_creation_default(self):
        """Test default config creation."""
        config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2
        )
        assert config.state_dim == 6
        assert config.action_dim == 2
        assert config.hidden_dim == 256
        assert config.num_hidden_layers == 3
    
    def test_creation_custom(self):
        """Test custom config creation."""
        config = PolicyNetworkConfig(
            state_dim=10,
            action_dim=4,
            hidden_dim=128,
            num_hidden_layers=2
        )
        assert config.state_dim == 10
        assert config.action_dim == 4
        assert config.hidden_dim == 128
    
    def test_invalid_state_dim(self):
        """Test invalid state_dim."""
        with pytest.raises(AssertionError):
            PolicyNetworkConfig(state_dim=0, action_dim=2)
    
    def test_invalid_action_dim(self):
        """Test invalid action_dim."""
        with pytest.raises(AssertionError):
            PolicyNetworkConfig(state_dim=6, action_dim=0)


class TestCNNEncoder:
    """Test CNN Encoder."""
    
    def test_creation(self):
        """Test encoder creation."""
        encoder = CNNEncoder(
            input_channels=3,
            channels=[16, 32, 64],
            output_dim=256
        )
        assert encoder.input_channels == 3
        assert encoder.output_dim == 256
    
    def test_forward_pass(self):
        """Test forward pass."""
        encoder = CNNEncoder(3, [16, 32, 64], 256)
        
        # Input: [batch, channels, height, width]
        x = torch.randn(4, 3, 64, 64)
        output = encoder(x)
        
        assert output.shape == (4, 256)


class TestPolicyNetwork:
    """Test PolicyNetwork."""
    
    @pytest.fixture
    def policy(self):
        """Create policy network."""
        config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2,
            hidden_dim=64,
            num_hidden_layers=2,
            device='cpu'
        )
        return PolicyNetwork(config)
    
    def test_creation(self, policy):
        """Test policy creation."""
        assert policy.config.state_dim == 6
        assert policy.config.action_dim == 2
        assert policy.encoder is None
    
    def test_forward_pass(self, policy):
        """Test forward pass."""
        state = torch.randn(4, 6)
        dist, value = policy(state)
        
        assert dist.loc.shape == (4, 2)
        assert dist.scale.shape == (4, 2)
        assert value.shape == (4, 1)
    
    def test_get_action(self, policy):
        """Test get_action method."""
        state = np.random.randn(6).astype(np.float32)
        action, value = policy.get_action(state)
        
        assert action.shape == (2,)
        assert isinstance(value, (float, np.floating))
    
    def test_get_action_deterministic(self, policy):
        """Test deterministic action."""
        state = np.random.randn(6).astype(np.float32)
        action, value = policy.get_action(state, deterministic=True)
        
        assert action.shape == (2,)
    
    def test_evaluate(self, policy):
        """Test evaluate method."""
        states = torch.randn(4, 6)
        actions = torch.randn(4, 2)
        
        log_probs, values, entropy = policy.evaluate(states, actions)
        
        assert log_probs.shape == (4,)
        assert values.shape == (4,)
        assert entropy.dim() == 0
    
    def test_get_value(self, policy):
        """Test get_value method."""
        states = torch.randn(4, 6)
        values = policy.get_value(states)
        
        assert values.shape == (4,)
    
    def test_get_log_std(self, policy):
        """Test get_log_std method."""
        log_std = policy.get_log_std()
        assert log_std.shape == (2,)
    
    def test_set_log_std(self, policy):
        """Test set_log_std method."""
        new_log_std = torch.ones(2) * 0.5
        policy.set_log_std(new_log_std)
        
        updated_log_std = policy.get_log_std()
        assert torch.allclose(updated_log_std, new_log_std)
    
    def test_cnn_encoder_integration(self):
        """Test with CNN encoder."""
        config = PolicyNetworkConfig(
            state_dim=3,
            action_dim=2,
            hidden_dim=64,
            use_cnn=True,
            cnn_channels=[16, 32],
            device='cpu'
        )
        policy = PolicyNetwork(config)
        
        # Vision input: [batch, channels, height, width]
        state = torch.randn(2, 3, 64, 64)
        dist, value = policy(state)
        
        assert dist.loc.shape == (2, 2)
        assert value.shape == (2, 1)


class TestPolicyNetworkWithLSTM:
    """Test PolicyNetworkWithLSTM."""
    
    @pytest.fixture
    def policy_lstm(self):
        """Create LSTM policy."""
        config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2,
            hidden_dim=64,
            device='cpu'
        )
        return PolicyNetworkWithLSTM(config, lstm_hidden_dim=32)
    
    def test_creation(self, policy_lstm):
        """Test LSTM policy creation."""
        assert policy_lstm.lstm_hidden_dim == 32
    
    def test_forward_pass(self, policy_lstm):
        """Test forward pass with sequence."""
        # Input: [batch, sequence_length, state_dim]
        state = torch.randn(2, 5, 6)
        dist, value, hidden = policy_lstm(state)
        
        assert dist.loc.shape == (2, 2)
        assert value.shape == (2, 1)
        assert len(hidden) == 2  # (h, c)
    
    def test_hidden_state_consistency(self, policy_lstm):
        """Test that hidden states can be passed between calls."""
        state1 = torch.randn(2, 3, 6)
        state2 = torch.randn(2, 2, 6)
        
        dist1, value1, hidden = policy_lstm(state1)
        dist2, value2, _ = policy_lstm(state2, hidden)
        
        assert dist2.loc.shape == (2, 2)


class TestPolicyNetworkIntegration:
    """Integration tests for PolicyNetwork."""
    
    def test_gradient_flow(self):
        """Test that gradients flow properly."""
        config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2,
            device='cpu'
        )
        policy = PolicyNetwork(config)
        optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
        
        state = torch.randn(4, 6)
        actions = torch.randn(4, 2)
        target_return = torch.randn(4)
        
        # Forward pass
        log_probs, values, entropy = policy.evaluate(state, actions)
        
        # Loss
        loss = -log_probs.mean() + torch.nn.functional.mse_loss(values, target_return)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check that gradients exist
        for param in policy.parameters():
            assert param.grad is not None
    
    def test_save_and_load(self, tmp_path):
        """Test saving and loading policy."""
        config = PolicyNetworkConfig(
            state_dim=6,
            action_dim=2,
            device='cpu'
        )
        policy1 = PolicyNetwork(config)
        
        # Save
        path = tmp_path / "policy.pt"
        torch.save(policy1.state_dict(), path)
        
        # Load
        policy2 = PolicyNetwork(config)
        policy2.load_state_dict(torch.load(path))
        
        # Check weights are identical
        state = torch.randn(4, 6)
        with torch.no_grad():
            dist1, value1 = policy1(state)
            dist2, value2 = policy2(state)
            
            assert torch.allclose(value1, value2)
