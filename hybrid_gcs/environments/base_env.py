"""
Base Environment Interface for Hybrid-GCS.

Defines the abstract base class and common data structures for all task environments.
Follows a gym-like step/reset interface without requiring gym as a dependency.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class StepResult:
    """
    Result of a single environment step.

    Attributes:
        observation: Current observation array
        reward: Scalar reward for the step
        done: Whether the episode has ended (terminal state)
        truncated: Whether the episode was truncated (e.g., max steps)
        info: Additional diagnostic information
    """

    observation: np.ndarray
    reward: float
    done: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EnvConfig:
    """
    Base configuration for all environments.

    Attributes:
        max_steps: Maximum number of steps per episode
        dt: Simulation time step in seconds
        seed: Random seed for reproducibility (None for random)
        render: Whether to enable rendering
    """

    max_steps: int = 500
    dt: float = 0.01
    seed: Optional[int] = None
    render: bool = False


class BaseEnvironment(ABC):
    """
    Abstract base class for Hybrid-GCS task environments.

    Provides a gym-like interface (reset/step) without requiring gym as a dependency.
    All concrete environments must implement the abstract methods.

    Attributes:
        config: Environment configuration
        step_count: Current step within the episode
        rng: NumPy random number generator
    """

    def __init__(self, config: EnvConfig) -> None:
        """
        Initialize the base environment.

        Args:
            config: Environment configuration
        """
        self.config = config
        self.step_count: int = 0
        self.rng = np.random.default_rng(config.seed)

    @abstractmethod
    def reset(self) -> np.ndarray:
        """
        Reset the environment and return the initial observation.

        Returns:
            Initial observation array
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """
        Execute one step in the environment.

        Args:
            action: Action array to apply

        Returns:
            StepResult containing observation, reward, done, truncated, info
        """
        pass

    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """
        Return the current observation without advancing the environment.

        Returns:
            Current observation array
        """
        pass

    @property
    @abstractmethod
    def observation_dim(self) -> int:
        """Dimension of the observation space."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Dimension of the action space."""
        pass

    def close(self) -> None:
        """Clean up environment resources. No-op by default."""
        pass

    def seed(self, seed: int) -> None:
        """
        Set the random seed for reproducibility.

        Args:
            seed: Random seed value
        """
        self.rng = np.random.default_rng(seed)
