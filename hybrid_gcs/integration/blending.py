"""
Action Blending Methods for Hybrid-GCS.

Implements multiple strategies for blending GCS and RL actions:
- Weighted blending: Linear combination with configurable alpha
- Hierarchical blending: GCS-priority with safety-based fallback
- Conflict resolution: Projection-based conflict handling

References:
    - Marcucci et al. (2023): GCS trajectory planning
    - Schulman et al. (2017): PPO for RL action generation
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np


class BlendingMethod(ABC):
    """Base class for action blending methods."""

    @abstractmethod
    def blend(
        self, gcs_action: np.ndarray, rl_action: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Blend GCS and RL actions.

        Args:
            gcs_action: Action from GCS planner
            rl_action: Action from RL policy
            **kwargs: Additional arguments for blending

        Returns:
            Blended action
        """
        pass


class WeightedBlender(BlendingMethod):
    """
    Weighted linear blending of GCS and RL actions.

    Blends actions as: action = alpha * gcs_action + (1 - alpha) * rl_action
    Alpha can be dynamically updated during execution.
    """

    def __init__(self, alpha: float = 0.5):
        """
        Initialize weighted blender.

        Args:
            alpha: Blending weight for GCS action (0.0 to 1.0).
                   alpha=1.0 uses pure GCS, alpha=0.0 uses pure RL.
        """
        assert 0.0 <= alpha <= 1.0, "alpha must be in [0, 1]"
        self._alpha = alpha

    @property
    def alpha(self) -> float:
        """Get current blending weight."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float):
        """
        Set blending weight.

        Args:
            value: New alpha value in [0, 1]
        """
        assert 0.0 <= value <= 1.0, "alpha must be in [0, 1]"
        self._alpha = value

    def blend(
        self, gcs_action: np.ndarray, rl_action: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Blend actions via weighted linear combination.

        Args:
            gcs_action: Action from GCS planner
            rl_action: Action from RL policy
            **kwargs: Unused

        Returns:
            Blended action: alpha * gcs + (1 - alpha) * rl
        """
        return self._alpha * gcs_action + (1.0 - self._alpha) * rl_action


class HierarchicalBlender(BlendingMethod):
    """
    Hierarchical blending with GCS priority.

    Uses GCS action if it satisfies safety constraints. Falls back to RL
    action otherwise. In the transition zone, smoothly blends between the two.
    """

    def __init__(
        self,
        is_safe: Callable[[np.ndarray], bool],
        transition_zone: float = 0.1,
    ):
        """
        Initialize hierarchical blender.

        Args:
            is_safe: Callable that returns True if an action is safe
            transition_zone: Width of smooth blending zone around the
                safety boundary. When the GCS action is within this margin
                of being unsafe, a smooth blend is applied.
        """
        assert transition_zone >= 0.0, "transition_zone must be non-negative"
        self.is_safe = is_safe
        self.transition_zone = transition_zone

    def blend(
        self, gcs_action: np.ndarray, rl_action: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Blend actions hierarchically with safety priority.

        If the GCS action is safe, use it. If not, fall back to RL.
        In the transition zone, smoothly blend between the two using
        the magnitude difference as a proxy for safety margin.

        Args:
            gcs_action: Action from GCS planner
            rl_action: Action from RL policy
            **kwargs: Additional arguments (passed to is_safe if needed)

        Returns:
            Selected or blended action
        """
        if self.is_safe(gcs_action):
            if self.transition_zone > 0.0:
                diff = np.linalg.norm(gcs_action - rl_action)
                if diff < self.transition_zone:
                    # Smooth blend in transition zone
                    t = diff / self.transition_zone
                    return t * gcs_action + (1.0 - t) * rl_action
            return gcs_action.copy()
        return rl_action.copy()


class ConflictResolutionBlender(BlendingMethod):
    """
    Conflict resolution blending for GCS and RL actions.

    Detects conflicts between GCS and RL actions via angle and magnitude
    divergence. When a conflict is detected, projects the RL action onto
    the GCS-feasible direction. Otherwise, uses the RL action directly.
    """

    def __init__(
        self,
        angle_threshold: float = np.pi / 4,
        magnitude_threshold: float = 2.0,
    ):
        """
        Initialize conflict resolution blender.

        Args:
            angle_threshold: Maximum angle (radians) between actions
                before a conflict is detected. Default: pi/4 (45 degrees).
            magnitude_threshold: Maximum ratio of action magnitudes
                before a conflict is detected. Default: 2.0.
        """
        assert angle_threshold > 0.0, "angle_threshold must be positive"
        assert magnitude_threshold > 0.0, "magnitude_threshold must be positive"
        self.angle_threshold = angle_threshold
        self.magnitude_threshold = magnitude_threshold

    def _detect_conflict(
        self, gcs_action: np.ndarray, rl_action: np.ndarray
    ) -> bool:
        """
        Detect conflict between GCS and RL actions.

        A conflict is detected if either the angle between the actions
        exceeds the threshold or the magnitude ratio exceeds the threshold.

        Args:
            gcs_action: Action from GCS planner
            rl_action: Action from RL policy

        Returns:
            True if a conflict is detected
        """
        gcs_norm = np.linalg.norm(gcs_action)
        rl_norm = np.linalg.norm(rl_action)

        # Check magnitude divergence
        if gcs_norm > 0 and rl_norm > 0:
            ratio = max(gcs_norm, rl_norm) / min(gcs_norm, rl_norm)
            if ratio > self.magnitude_threshold:
                return True

            # Check angle divergence
            cos_angle = np.dot(gcs_action, rl_action) / (gcs_norm * rl_norm)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle)
            if angle > self.angle_threshold:
                return True

        return False

    def blend(
        self, gcs_action: np.ndarray, rl_action: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Blend actions with conflict resolution.

        When no conflict is detected, the RL action is used directly.
        When a conflict is detected, the RL action is projected onto
        the GCS-feasible direction.

        Args:
            gcs_action: Action from GCS planner
            rl_action: Action from RL policy
            **kwargs: Unused

        Returns:
            Resolved action
        """
        if not self._detect_conflict(gcs_action, rl_action):
            return rl_action.copy()

        # Project RL action onto GCS direction
        gcs_norm = np.linalg.norm(gcs_action)
        if gcs_norm < 1e-8:
            return rl_action.copy()

        gcs_direction = gcs_action / gcs_norm
        projection = np.dot(rl_action, gcs_direction) * gcs_direction

        # Use projected component (clamped to non-negative along GCS direction)
        proj_magnitude = np.dot(projection, gcs_direction)
        if proj_magnitude < 0:
            return gcs_action.copy()

        return projection
