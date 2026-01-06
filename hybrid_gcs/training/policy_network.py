"""
Policy Network Implementation for Hybrid-GCS.

Implements Actor-Critic architecture with:
- Shared backbone network
- Separate policy (actor) and value (critic) heads
- CNN encoder for vision-based observations
- Support for deterministic and stochastic actions
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
import numpy as np


@dataclass
class PolicyNetworkConfig:
    """Configuration for PolicyNetwork."""
    
    # Network architecture
    state_dim: int                          # Dimension of state observation
    action_dim: int                         # Dimension of action space
    hidden_dim: int = 256                   # Hidden layer dimension
    num_hidden_layers: int = 3              # Number of hidden layers
    activation: str = 'relu'                # Activation function
    
    # CNN encoder (for vision)
    use_cnn: bool = False                   # Use CNN encoder for images
    cnn_channels: list = None               # CNN channels: [16, 32, 64]
    cnn_kernel_size: int = 3                # CNN kernel size
    cnn_stride: int = 1                     # CNN stride
    
    # Output characteristics
    log_std_init: float = 0.0               # Initial log standard deviation
    action_scale: float = 1.0               # Action scaling factor
    min_log_std: float = -20.0              # Minimum log std (clipping)
    max_log_std: float = 2.0                # Maximum log std (clipping)
    
    # Device
    device: str = 'cpu'                     # Device (cpu/cuda)
    
    def __post_init__(self):
        """Validate configuration."""
        if self.cnn_channels is None:
            self.cnn_channels = [16, 32, 64]
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.action_dim > 0, "action_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"
        assert self.num_hidden_layers >= 1, "num_hidden_layers must be >= 1"


class CNNEncoder(nn.Module):
    """
    CNN Encoder for vision-based observations.
    
    Converts images [B, C, H, W] to feature vectors [B, feature_dim].
    """
    
    def __init__(self, input_channels: int, channels: list, kernel_size: int = 3,
                 stride: int = 1, output_dim: int = 256):
        """
        Initialize CNN encoder.
        
        Args:
            input_channels: Number of input channels
            channels: List of channel dimensions per layer
            kernel_size: Kernel size for convolutions
            stride: Stride for convolutions
            output_dim: Dimension of output features
        """
        super().__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        
        # Build convolutional layers
        layers = []
        in_channels = input_channels
        
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=kernel_size // 2))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
        
        # Compute flattened size (assuming 64x64 input)
        # After each MaxPool2d, spatial dims are halved
        # After 3 pools: 64 → 32 → 16 → 8
        self.conv_layers = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((8, 8)))
        flat_size = channels[-1] * 8 * 8
        
        # Output projection
        self.fc = nn.Linear(flat_size, output_dim)
        self.fc_activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Feature tensor [B, output_dim]
        """
        x = self.conv_layers(x)
        x = x.flatten(1)
        x = self.fc(x)
        x = self.fc_activation(x)
        return x


class PolicyNetwork(nn.Module):
    """
    Actor-Critic Policy Network.
    
    Jointly learns both policy (actor) and value function (critic).
    Supports both discrete and continuous action spaces.
    """
    
    def __init__(self, config: PolicyNetworkConfig):
        """
        Initialize PolicyNetwork.
        
        Args:
            config: PolicyNetworkConfig instance
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device(config.device)
        self.to(self.device)
        
        # CNN encoder (if using vision)
        if config.use_cnn:
            self.encoder = CNNEncoder(
                input_channels=3,  # Assume RGB
                channels=config.cnn_channels,
                kernel_size=config.cnn_kernel_size,
                stride=config.cnn_stride,
                output_dim=config.hidden_dim
            )
            backbone_input_dim = config.hidden_dim
        else:
            self.encoder = None
            backbone_input_dim = config.state_dim
        
        # Shared backbone network
        backbone_layers = []
        in_dim = backbone_input_dim
        
        for _ in range(config.num_hidden_layers):
            backbone_layers.append(nn.Linear(in_dim, config.hidden_dim))
            
            # Activation function
            if config.activation == 'relu':
                backbone_layers.append(nn.ReLU())
            elif config.activation == 'tanh':
                backbone_layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unknown activation: {config.activation}")
            
            in_dim = config.hidden_dim
        
        self.backbone = nn.Sequential(*backbone_layers)
        
        # Actor (policy) head - outputs mean and log_std for continuous actions
        self.actor_mean = nn.Linear(config.hidden_dim, config.action_dim)
        self.actor_log_std = nn.Parameter(
            torch.ones(config.action_dim) * config.log_std_init
        )
        
        # Critic (value) head
        self.critic = nn.Linear(config.hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for layer in self.backbone:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, state: torch.Tensor, deterministic: bool = False) \
            -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass through policy and value networks.
        
        Args:
            state: State observation [B, state_dim] or [B, C, H, W] for vision
            deterministic: If True, use mean action instead of sampling
        
        Returns:
            (action_distribution, value_estimate)
            - action_distribution: torch.distributions.Normal
            - value_estimate: [B, 1]
        """
        # Encode input if using CNN
        if self.encoder is not None:
            state = self.encoder(state)
        
        # Shared backbone
        features = self.backbone(state)
        
        # Policy head
        mean = self.actor_mean(features)
        log_std = self.actor_log_std.clamp(
            min=self.config.min_log_std,
            max=self.config.max_log_std
        )
        std = torch.exp(log_std)
        
        # Create distribution
        dist = Normal(mean, std)
        
        # Value head
        value = self.critic(features)
        
        return dist, value
    
    def get_action(self, state: np.ndarray, deterministic: bool = False) \
            -> Tuple[np.ndarray, float]:
        """
        Get action from state (numpy interface).
        
        Args:
            state: Numpy array state [state_dim]
            deterministic: Use mean action if True
        
        Returns:
            (action, value_estimate)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            dist, value = self.forward(state_tensor, deterministic=deterministic)
            
            if deterministic:
                action = dist.mean
            else:
                action = dist.rsample()
            
            action_np = action.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0, 0]
        
        return action_np, value_np
    
    def evaluate(self, states: torch.Tensor, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate policy on states and actions.
        
        Used during training to compute policy gradient and value loss.
        
        Args:
            states: State batch [B, state_dim]
            actions: Action batch [B, action_dim]
        
        Returns:
            (log_probs, values, entropy)
            - log_probs: [B] - log probability of actions
            - values: [B] - value estimates
            - entropy: scalar - distribution entropy
        """
        dist, value = self.forward(states)
        
        # Log probabilities
        log_probs = dist.log_prob(actions).sum(dim=-1)
        
        # Entropy
        entropy = dist.entropy().sum(dim=-1).mean()
        
        return log_probs, value.squeeze(-1), entropy
    
    def get_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimate for state.
        
        Args:
            state: State tensor [B, state_dim]
        
        Returns:
            Value estimate [B]
        """
        with torch.no_grad():
            _, value = self.forward(state)
        return value.squeeze(-1)
    
    def get_log_std(self) -> torch.Tensor:
        """Get current log standard deviation."""
        return self.actor_log_std.clone()
    
    def set_log_std(self, log_std: torch.Tensor):
        """Set log standard deviation."""
        self.actor_log_std.data = log_std.clamp(
            min=self.config.min_log_std,
            max=self.config.max_log_std
        )


class PolicyNetworkWithLSTM(nn.Module):
    """
    Policy Network with LSTM for temporal dependencies.
    
    Useful for non-Markovian environments or observations with history.
    """
    
    def __init__(self, config: PolicyNetworkConfig, lstm_hidden_dim: int = 128):
        """
        Initialize PolicyNetwork with LSTM.
        
        Args:
            config: PolicyNetworkConfig instance
            lstm_hidden_dim: LSTM hidden dimension
        """
        super().__init__()
        
        self.config = config
        self.lstm_hidden_dim = lstm_hidden_dim
        self.device = torch.device(config.device)
        
        # Input layer
        self.input_layer = nn.Linear(config.state_dim, config.hidden_dim)
        self.input_activation = nn.ReLU()
        
        # LSTM layer
        self.lstm = nn.LSTM(config.hidden_dim, lstm_hidden_dim, batch_first=True)
        
        # Policy head
        self.actor_mean = nn.Linear(lstm_hidden_dim, config.action_dim)
        self.actor_log_std = nn.Parameter(
            torch.ones(config.action_dim) * config.log_std_init
        )
        
        # Value head
        self.critic = nn.Linear(lstm_hidden_dim, 1)
        
        self.to(self.device)
    
    def forward(self, state: torch.Tensor, hidden_state: Optional[Tuple] = None) \
            -> Tuple[Normal, torch.Tensor, Tuple]:
        """
        Forward pass with LSTM.
        
        Args:
            state: State tensor [B, T, state_dim] (T=sequence length)
            hidden_state: LSTM hidden state (h, c) or None
        
        Returns:
            (distribution, value, hidden_state)
        """
        # Input embedding
        x = self.input_layer(state)
        x = self.input_activation(x)
        
        # LSTM forward
        x, hidden_state = self.lstm(x, hidden_state)
        
        # Use last timestep
        x = x[:, -1, :]
        
        # Policy head
        mean = self.actor_mean(x)
        log_std = self.actor_log_std.clamp(
            min=self.config.min_log_std,
            max=self.config.max_log_std
        )
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        
        # Value head
        value = self.critic(x)
        
        return dist, value, hidden_state
