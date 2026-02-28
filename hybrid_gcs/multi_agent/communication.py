"""
Multi-Agent Communication for Hybrid-GCS (Section 4.1: Multi-Agent RL with Communication).

Implements Centralized Training with Decentralized Execution (CTDE):
- Training: Global critic Q(s_1,...,s_n, a_1,...,a_n)
- Execution: Each agent uses local policy pi_i(a_i | s_i, comm_i)

Attention-based communication mechanism:
- alpha_{ij} = softmax(Query_i . Key_j / sqrt(d))
- m_i = sum_j alpha_{ij} Value_j
- Local policy input: [s_i, m_i] -> pi_i(a_i | s_i, m_i)
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class AttentionComm(nn.Module):
    """
    Attention-based communication mechanism for multi-agent systems.

    Each agent computes Query/Key/Value projections from its features.
    Messages are aggregated via scaled dot-product attention, allowing
    agents to selectively attend to relevant information from peers.
    """

    def __init__(self, feature_dim: int, comm_dim: int, num_heads: int = 1):
        """
        Initialize attention-based communication.

        Args:
            feature_dim: Dimension of input agent features
            comm_dim: Dimension of output communication messages
            num_heads: Number of attention heads
        """
        super().__init__()

        self.feature_dim = feature_dim
        self.comm_dim = comm_dim
        self.num_heads = num_heads
        self.head_dim = comm_dim // num_heads

        assert comm_dim % num_heads == 0, "comm_dim must be divisible by num_heads"

        # Query/Key/Value projections
        self.query_proj = nn.Linear(feature_dim, comm_dim)
        self.key_proj = nn.Linear(feature_dim, comm_dim)
        self.value_proj = nn.Linear(feature_dim, comm_dim)

        # Output projection (combines heads)
        self.output_proj = nn.Linear(comm_dim, comm_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights with Xavier uniform."""
        for proj in [self.query_proj, self.key_proj, self.value_proj, self.output_proj]:
            nn.init.xavier_uniform_(proj.weight)
            nn.init.constant_(proj.bias, 0.0)

    def forward(self, agent_features: torch.Tensor) -> torch.Tensor:
        """
        Compute attention-based communication messages.

        Each agent attends to all agents (including itself) to produce
        an aggregated message: m_i = sum_j alpha_{ij} Value_j

        Args:
            agent_features: Agent feature tensor [num_agents, feature_dim]

        Returns:
            Communication messages [num_agents, comm_dim]
        """
        num_agents = agent_features.shape[0]

        # Compute Q, K, V projections
        queries = self.query_proj(agent_features)  # [num_agents, comm_dim]
        keys = self.key_proj(agent_features)  # [num_agents, comm_dim]
        values = self.value_proj(agent_features)  # [num_agents, comm_dim]

        # Reshape for multi-head attention: [num_heads, num_agents, head_dim]
        queries = queries.view(num_agents, self.num_heads, self.head_dim).transpose(
            0, 1
        )
        keys = keys.view(num_agents, self.num_heads, self.head_dim).transpose(0, 1)
        values = values.view(num_agents, self.num_heads, self.head_dim).transpose(0, 1)

        # Scaled dot-product attention: alpha_{ij} = softmax(Q_i . K_j / sqrt(d))
        scale = float(self.head_dim) ** 0.5
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale
        attn_weights = torch.softmax(
            attn_scores, dim=-1
        )  # [num_heads, num_agents, num_agents]

        # Aggregate messages: m_i = sum_j alpha_{ij} V_j
        messages = torch.matmul(
            attn_weights, values
        )  # [num_heads, num_agents, head_dim]

        # Concatenate heads and project
        messages = messages.transpose(0, 1).contiguous().view(num_agents, self.comm_dim)
        messages = self.output_proj(messages)  # [num_agents, comm_dim]

        return messages


class MultiAgentPolicy(nn.Module):
    """
    Decentralized policy with attention-based communication.

    Each agent uses a shared local policy network that takes as input the
    concatenation of its local state and a communication message from peers.
    Communication is computed via AttentionComm before policy evaluation.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        comm_dim: int = 32,
        hidden_dim: int = 128,
        num_agents: int = 2,
    ):
        """
        Initialize multi-agent policy.

        Args:
            state_dim: Dimension of each agent's local state
            action_dim: Dimension of each agent's action space
            comm_dim: Dimension of communication messages
            hidden_dim: Hidden layer dimension for policy network
            num_agents: Number of agents
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.comm_dim = comm_dim
        self.hidden_dim = hidden_dim
        self.num_agents = num_agents

        # State encoder produces features for communication
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # Attention-based communication
        self.comm = AttentionComm(
            feature_dim=hidden_dim,
            comm_dim=comm_dim,
        )

        # Shared local policy: [s_i, m_i] -> pi_i(a_i | s_i, m_i)
        policy_input_dim = hidden_dim + comm_dim
        self.policy_net = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor head (action mean)
        self.actor_mean = nn.Linear(hidden_dim, action_dim)

        # Learnable log standard deviation
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Value head (local value estimate)
        self.value_head = nn.Linear(hidden_dim, 1)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for module in [self.state_encoder, self.policy_net]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.constant_(self.value_head.bias, 0.0)

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute action means and value estimates for all agents.

        Args:
            states: Agent states [num_agents, state_dim]

        Returns:
            Tuple of (action_means, values):
                - action_means: [num_agents, action_dim]
                - values: [num_agents, 1]
        """
        # Encode states
        features = self.state_encoder(states)  # [num_agents, hidden_dim]

        # Compute communication messages
        messages = self.comm(features)  # [num_agents, comm_dim]

        # Concatenate features and messages: [s_i, m_i]
        policy_input = torch.cat([features, messages], dim=-1)

        # Shared policy network
        hidden = self.policy_net(policy_input)  # [num_agents, hidden_dim]

        # Action means and values
        action_means = self.actor_mean(hidden)  # [num_agents, action_dim]
        values = self.value_head(hidden)  # [num_agents, 1]

        return action_means, values

    def get_actions(
        self, states: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        """
        Get actions for all agents.

        Args:
            states: Agent states [num_agents, state_dim]
            deterministic: If True, return mean actions; otherwise sample

        Returns:
            Actions [num_agents, action_dim]
        """
        action_means, _ = self.forward(states)

        if deterministic:
            return action_means

        std = torch.exp(self.actor_log_std)
        dist = torch.distributions.Normal(action_means, std)
        return dist.rsample()


class CentralizedCritic(nn.Module):
    """
    Global critic for Centralized Training with Decentralized Execution (CTDE).

    Takes the concatenated states and actions of all agents to produce a
    single Q-value: Q(s_1,...,s_n, a_1,...,a_n).
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_agents: int = 2,
        hidden_dim: int = 256,
    ):
        """
        Initialize centralized critic.

        Args:
            state_dim: Dimension of each agent's state
            action_dim: Dimension of each agent's action
            num_agents: Number of agents
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.hidden_dim = hidden_dim

        input_dim = num_agents * (state_dim + action_dim)

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)

        # Final layer uses smaller gain for stable initial values
        final_layer = self.network[-1]
        nn.init.orthogonal_(final_layer.weight, gain=1.0)
        nn.init.constant_(final_layer.bias, 0.0)

    def forward(
        self, all_states: torch.Tensor, all_actions: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value from all agents' states and actions.

        Args:
            all_states: Concatenated states [batch, num_agents * state_dim]
            all_actions: Concatenated actions [batch, num_agents * action_dim]

        Returns:
            Q-value estimates [batch, 1]
        """
        x = torch.cat([all_states, all_actions], dim=-1)
        return self.network(x)
