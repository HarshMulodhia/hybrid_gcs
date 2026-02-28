"""
Unit tests for multi-agent communication module.

Tests AttentionComm, MultiAgentPolicy, and CentralizedCritic.
"""

import numpy as np
import pytest
import torch

from hybrid_gcs.multi_agent.communication import (
    AttentionComm,
    CentralizedCritic,
    MultiAgentPolicy,
)


class TestAttentionComm:
    """Test AttentionComm class."""

    def test_output_shape(self):
        """Test that output has correct shape."""
        comm = AttentionComm(feature_dim=16, comm_dim=8)
        features = torch.randn(3, 16)
        messages = comm(features)
        assert messages.shape == (3, 8)

    def test_single_agent(self):
        """Test with single agent (self-attention only)."""
        comm = AttentionComm(feature_dim=8, comm_dim=4)
        features = torch.randn(1, 8)
        messages = comm(features)
        assert messages.shape == (1, 4)

    def test_multi_head(self):
        """Test multi-head attention."""
        comm = AttentionComm(feature_dim=16, comm_dim=8, num_heads=2)
        features = torch.randn(4, 16)
        messages = comm(features)
        assert messages.shape == (4, 8)

    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        comm = AttentionComm(feature_dim=8, comm_dim=4)
        features = torch.randn(2, 8, requires_grad=True)
        messages = comm(features)
        loss = messages.sum()
        loss.backward()
        assert features.grad is not None

    def test_invalid_comm_dim(self):
        """Test that comm_dim not divisible by num_heads raises error."""
        with pytest.raises(AssertionError):
            AttentionComm(feature_dim=8, comm_dim=7, num_heads=2)


class TestMultiAgentPolicy:
    """Test MultiAgentPolicy class."""

    def test_forward_shape(self):
        """Test forward pass output shapes."""
        policy = MultiAgentPolicy(state_dim=6, action_dim=3, num_agents=2)
        states = torch.randn(2, 6)
        action_means, values = policy(states)
        assert action_means.shape == (2, 3)
        assert values.shape == (2, 1)

    def test_get_actions_deterministic(self):
        """Test deterministic action generation."""
        policy = MultiAgentPolicy(state_dim=4, action_dim=2, num_agents=3)
        states = torch.randn(3, 4)
        actions = policy.get_actions(states, deterministic=True)
        assert actions.shape == (3, 2)

    def test_get_actions_stochastic(self):
        """Test stochastic action sampling."""
        policy = MultiAgentPolicy(state_dim=4, action_dim=2, num_agents=2)
        states = torch.randn(2, 4)
        actions1 = policy.get_actions(states, deterministic=False)
        actions2 = policy.get_actions(states, deterministic=False)
        assert actions1.shape == (2, 2)
        # Stochastic actions should differ (with high probability)
        # Not a strict test since they could coincidentally match

    def test_gradient_flow(self):
        """Test that gradients flow through the full policy."""
        policy = MultiAgentPolicy(state_dim=4, action_dim=2, num_agents=2)
        states = torch.randn(2, 4, requires_grad=True)
        action_means, values = policy(states)
        loss = action_means.sum() + values.sum()
        loss.backward()
        assert states.grad is not None


class TestCentralizedCritic:
    """Test CentralizedCritic class."""

    def test_output_shape(self):
        """Test Q-value output shape."""
        critic = CentralizedCritic(state_dim=4, action_dim=2, num_agents=2)
        all_states = torch.randn(8, 8)  # batch=8, 2*4=8
        all_actions = torch.randn(8, 4)  # batch=8, 2*2=4
        q_values = critic(all_states, all_actions)
        assert q_values.shape == (8, 1)

    def test_single_agent(self):
        """Test with single agent."""
        critic = CentralizedCritic(state_dim=3, action_dim=2, num_agents=1)
        states = torch.randn(4, 3)
        actions = torch.randn(4, 2)
        q_values = critic(states, actions)
        assert q_values.shape == (4, 1)

    def test_gradient_flow(self):
        """Test gradient flow through critic."""
        critic = CentralizedCritic(state_dim=4, action_dim=2, num_agents=2)
        states = torch.randn(4, 8, requires_grad=True)
        actions = torch.randn(4, 4, requires_grad=True)
        q_values = critic(states, actions)
        loss = q_values.sum()
        loss.backward()
        assert states.grad is not None
        assert actions.grad is not None
