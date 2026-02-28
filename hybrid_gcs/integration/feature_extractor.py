"""
Dual Pathway Feature Extractor for Hybrid-GCS.

Implements a dual-pathway neural network that extracts:
- Low-dimensional features for GCS planning (compact, structured)
- High-dimensional features for RL policy (rich, expressive)

This enables each component to operate on features best suited
to its requirements while sharing a common input observation.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass
class FeatureExtractorConfig:
    """Configuration for DualPathwayExtractor."""

    input_dim: int  # Dimension of input observations
    gcs_feature_dim: int = 16  # GCS feature dimension (compact)
    rl_feature_dim: int = 64  # RL feature dimension (expressive)
    hidden_dim: int = 128  # Hidden layer dimension

    def __post_init__(self):
        """Validate configuration."""
        assert self.input_dim > 0, "input_dim must be positive"
        assert self.gcs_feature_dim > 0, "gcs_feature_dim must be positive"
        assert self.rl_feature_dim > 0, "rl_feature_dim must be positive"
        assert self.hidden_dim > 0, "hidden_dim must be positive"


class DualPathwayExtractor(nn.Module):
    """
    Dual pathway feature extractor for GCS and RL.

    Produces two separate feature representations from a shared input:
    - GCS pathway: Linear -> ReLU -> Linear (compact features)
    - RL pathway: Linear -> ReLU -> Linear -> ReLU -> Linear (rich features)
    """

    def __init__(self, config: FeatureExtractorConfig):
        """
        Initialize dual pathway extractor.

        Args:
            config: FeatureExtractorConfig instance
        """
        super().__init__()

        self.config = config

        # GCS pathway: compact, low-dimensional features
        self.gcs_pathway = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.gcs_feature_dim),
        )

        # RL pathway: rich, high-dimensional features
        self.rl_pathway = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.rl_feature_dim),
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for pathway in [self.gcs_pathway, self.rl_pathway]:
            for layer in pathway:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                    nn.init.constant_(layer.bias, 0.0)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both pathways.

        Args:
            x: Input observation tensor [B, input_dim]

        Returns:
            Tuple of (gcs_features, rl_features):
                - gcs_features: [B, gcs_feature_dim]
                - rl_features: [B, rl_feature_dim]
        """
        gcs_features = self.gcs_pathway(x)
        rl_features = self.rl_pathway(x)
        return gcs_features, rl_features

    def extract_gcs_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract GCS features only.

        Args:
            x: Input observation tensor [B, input_dim]

        Returns:
            GCS feature tensor [B, gcs_feature_dim]
        """
        return self.gcs_pathway(x)

    def extract_rl_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract RL features only.

        Args:
            x: Input observation tensor [B, input_dim]

        Returns:
            RL feature tensor [B, rl_feature_dim]
        """
        return self.rl_pathway(x)
