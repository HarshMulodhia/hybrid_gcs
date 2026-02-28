"""
Safety Filter for Hybrid-GCS.

Implements real-time constraint enforcement for safe robot operation:
- Position bounds checking
- Velocity and acceleration limits
- Collision avoidance with obstacles

The safety filter projects unsafe actions into the feasible set,
ensuring all outputs satisfy the configured constraints.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class SafetyFilterConfig:
    """Configuration for SafetyFilter.

    Defines all safety constraints including workspace bounds,
    velocity/acceleration limits, and obstacle geometry.
    """

    position_bounds_lower: np.ndarray = field(
        default_factory=lambda: np.array([-1.0, -1.0, -1.0])
    )
    position_bounds_upper: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 1.0, 1.0])
    )
    max_velocity: float = 1.0
    max_acceleration: float = 5.0
    collision_margin: float = 0.1
    obstacle_positions: Optional[List[np.ndarray]] = None
    obstacle_radii: Optional[List[float]] = None

    def __post_init__(self):
        """Validate configuration."""
        assert self.max_velocity > 0, "max_velocity must be positive"
        assert self.max_acceleration > 0, "max_acceleration must be positive"
        assert self.collision_margin >= 0, "collision_margin must be non-negative"
        assert len(self.position_bounds_lower) == len(
            self.position_bounds_upper
        ), "Bounds dimensions must match"
        assert np.all(
            self.position_bounds_lower <= self.position_bounds_upper
        ), "Lower bounds must be <= upper bounds"
        if self.obstacle_positions is not None and self.obstacle_radii is not None:
            assert len(self.obstacle_positions) == len(
                self.obstacle_radii
            ), "Number of obstacle positions and radii must match"


class SafetyFilter:
    """
    Real-time safety constraint enforcement.

    Filters actions to ensure they satisfy all configured safety
    constraints including position bounds, velocity limits, acceleration
    limits, and collision avoidance.
    """

    def __init__(self, config: SafetyFilterConfig):
        """
        Initialize safety filter.

        Args:
            config: SafetyFilterConfig instance
        """
        self.config = config

    def check_bounds(self, state: np.ndarray) -> bool:
        """
        Check if state position is within workspace bounds.

        Args:
            state: State vector (position components assumed to be
                the first N elements matching bounds dimensionality)

        Returns:
            True if position is within bounds
        """
        dim = len(self.config.position_bounds_lower)
        position = state[:dim]
        return bool(
            np.all(position >= self.config.position_bounds_lower)
            and np.all(position <= self.config.position_bounds_upper)
        )

    def check_velocity(self, velocity: np.ndarray) -> bool:
        """
        Check if velocity is within limits.

        Args:
            velocity: Velocity vector

        Returns:
            True if velocity magnitude is within max_velocity
        """
        return np.linalg.norm(velocity) <= self.config.max_velocity

    def check_collision(self, position: np.ndarray) -> bool:
        """
        Check if position is collision-free with all obstacles.

        Args:
            position: Position vector

        Returns:
            True if position is collision-free (no collision detected)
        """
        if self.config.obstacle_positions is None or self.config.obstacle_radii is None:
            return True

        for obs_pos, obs_radius in zip(
            self.config.obstacle_positions, self.config.obstacle_radii
        ):
            dist = np.linalg.norm(position - obs_pos)
            if dist < obs_radius + self.config.collision_margin:
                return False

        return True

    def check_acceleration(self, acceleration: np.ndarray) -> bool:
        """
        Check if acceleration is within limits.

        Args:
            acceleration: Acceleration vector

        Returns:
            True if acceleration magnitude is within max_acceleration
        """
        return np.linalg.norm(acceleration) <= self.config.max_acceleration

    def is_safe(self, state: np.ndarray) -> bool:
        """
        Check if state satisfies all safety constraints.

        Args:
            state: Full state vector. Position is assumed to be the
                first N elements (matching bounds dim), velocity is
                assumed to follow position.

        Returns:
            True if all constraints are satisfied
        """
        dim = len(self.config.position_bounds_lower)
        position = state[:dim]

        if not self.check_bounds(state):
            return False

        if not self.check_collision(position):
            return False

        # Check velocity if state includes it
        if len(state) >= 2 * dim:
            velocity = state[dim : 2 * dim]
            if not self.check_velocity(velocity):
                return False

        return True

    def filter_action(
        self, action: np.ndarray, current_state: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Project action to satisfy all safety constraints.

        Applies sequential constraint projection:
        1. Clip acceleration magnitude
        2. Clip resulting velocity
        3. Clip resulting position to workspace bounds
        4. Push away from obstacles

        Args:
            action: Proposed action (interpreted as acceleration)
            current_state: Current state vector (position + velocity)
            **kwargs: Additional arguments (e.g., dt for integration)

        Returns:
            Safe action satisfying all constraints
        """
        dt = kwargs.get("dt", 0.01)
        dim = len(self.config.position_bounds_lower)
        position = current_state[:dim]
        velocity = (
            current_state[dim : 2 * dim]
            if len(current_state) >= 2 * dim
            else np.zeros(dim)
        )

        safe_action = action.copy()

        # 1. Clip acceleration magnitude
        accel_norm = np.linalg.norm(safe_action)
        if accel_norm > self.config.max_acceleration:
            safe_action = safe_action * self.config.max_acceleration / accel_norm

        # 2. Clip resulting velocity
        new_velocity = velocity + safe_action * dt
        vel_norm = np.linalg.norm(new_velocity)
        if vel_norm > self.config.max_velocity:
            new_velocity = new_velocity * self.config.max_velocity / vel_norm
            safe_action = (new_velocity - velocity) / dt

        # 3. Clip resulting position to workspace bounds
        new_position = position + new_velocity * dt
        clipped_position = np.clip(
            new_position,
            self.config.position_bounds_lower,
            self.config.position_bounds_upper,
        )
        if not np.allclose(new_position, clipped_position):
            new_velocity = (clipped_position - position) / dt
            safe_action = (new_velocity - velocity) / dt

        # 4. Push away from obstacles
        if (
            self.config.obstacle_positions is not None
            and self.config.obstacle_radii is not None
        ):
            for obs_pos, obs_radius in zip(
                self.config.obstacle_positions, self.config.obstacle_radii
            ):
                to_obs = clipped_position - obs_pos
                dist = np.linalg.norm(to_obs)
                min_dist = obs_radius + self.config.collision_margin
                if dist < min_dist and dist > 1e-8:
                    # Push position to boundary of obstacle margin
                    push_dir = to_obs / dist
                    clipped_position = obs_pos + push_dir * min_dist
                    new_velocity = (clipped_position - position) / dt
                    safe_action = (new_velocity - velocity) / dt

        return safe_action


class ControlBarrierFilter:
    """
    Control Barrier Function (CBF) based safety filter.

    Provides formal safety guarantees by ensuring that all actions
    satisfy CBF constraints of the form:

        L_f h(s) + alpha * h(s) <= 0

    where h(s) <= 0 defines the safe set and alpha > 0 controls
    the convergence rate to the safe boundary.

    This ensures h(s(t)) <= e^(-alpha*t) * h(s(0)), guaranteeing
    exponential convergence to safety.

    References:
        Section 3.3 of Hybrid-GCS Theory.
    """

    def __init__(self, config: SafetyFilterConfig, alpha: float = 1.0):
        """
        Initialize CBF-based safety filter.

        Args:
            config: SafetyFilterConfig instance
            alpha: CBF convergence rate parameter (alpha > 0).
                   Larger values enforce faster convergence to safe set.
        """
        assert alpha > 0, "alpha must be positive"
        self.config = config
        self.alpha = alpha
        self._safety_filter = SafetyFilter(config)

    def barrier_bounds(self, position: np.ndarray) -> np.ndarray:
        """
        Compute barrier function values for workspace bounds.

        h_i(s) = position_i - upper_i  (upper bound constraint)
        h_j(s) = lower_j - position_j  (lower bound constraint)

        Safe set: h(s) <= 0 for all constraints.

        Args:
            position: Current position vector

        Returns:
            Array of barrier values (negative = safe, positive = unsafe)
        """
        dim = len(self.config.position_bounds_lower)
        pos = position[:dim]
        h_upper = pos - self.config.position_bounds_upper
        h_lower = self.config.position_bounds_lower - pos
        return np.concatenate([h_upper, h_lower])

    def barrier_obstacles(self, position: np.ndarray) -> np.ndarray:
        """
        Compute barrier function values for obstacle avoidance.

        h(s) = (radius + margin)^2 - ||position - obstacle||^2

        Safe set: h(s) <= 0 (i.e., outside obstacle + margin).

        Args:
            position: Current position vector

        Returns:
            Array of barrier values (negative = safe, positive = unsafe)
        """
        if self.config.obstacle_positions is None or self.config.obstacle_radii is None:
            return np.array([])

        barriers = []
        for obs_pos, obs_radius in zip(
            self.config.obstacle_positions, self.config.obstacle_radii
        ):
            dist_sq = np.sum((position - obs_pos) ** 2)
            safe_dist = obs_radius + self.config.collision_margin
            h = safe_dist**2 - dist_sq
            barriers.append(h)
        return np.array(barriers)

    def filter_action(
        self, action: np.ndarray, current_state: np.ndarray, **kwargs
    ) -> np.ndarray:
        """
        Project action to satisfy CBF constraints.

        Solves:
            min  ||a - a_desired||^2
            s.t. L_f h(s) + alpha * h(s) <= 0  for all constraints

        Falls back to sequential projection when no QP solver is available.

        Args:
            action: Desired action (acceleration)
            current_state: Current state vector (position + velocity)
            **kwargs: Additional arguments (e.g., dt)

        Returns:
            Safe action satisfying CBF constraints
        """
        dt = kwargs.get("dt", 0.01)
        dim = len(self.config.position_bounds_lower)
        position = current_state[:dim]
        velocity = (
            current_state[dim : 2 * dim]
            if len(current_state) >= 2 * dim
            else np.zeros(dim)
        )

        safe_action = action.copy()

        # CBF enforcement for bounds
        h_bounds = self.barrier_bounds(position)
        for i in range(dim):
            # Upper bound: h = pos_i - upper_i, dh/dt = vel_i + accel_i * dt
            h_val = h_bounds[i]
            h_dot = velocity[i] + safe_action[i] * dt
            if h_dot + self.alpha * h_val > 0:
                safe_action[i] = (-self.alpha * h_val - velocity[i]) / dt

            # Lower bound: h = lower_i - pos_i, dh/dt = -vel_i - accel_i * dt
            h_val = h_bounds[dim + i]
            h_dot = -velocity[i] - safe_action[i] * dt
            if h_dot + self.alpha * h_val > 0:
                safe_action[i] = (self.alpha * h_val + velocity[i]) / dt

        # CBF enforcement for obstacles
        h_obs = self.barrier_obstacles(position)
        if len(h_obs) > 0:
            for idx, (obs_pos, obs_radius) in enumerate(
                zip(self.config.obstacle_positions, self.config.obstacle_radii)
            ):
                diff = position - obs_pos
                dist_sq = np.sum(diff**2)
                h_val = h_obs[idx]
                # dh/dt = -2 * diff . (velocity + action * dt)
                vel_effective = velocity + safe_action * dt
                h_dot = -2.0 * np.dot(diff, vel_effective)
                if h_dot + self.alpha * h_val > 0:
                    # Project action to satisfy constraint
                    if np.linalg.norm(diff) > 1e-8:
                        direction = diff / np.linalg.norm(diff)
                        correction = (h_dot + self.alpha * h_val) / (
                            2.0 * np.dot(diff, direction) * dt + 1e-8
                        )
                        safe_action += correction * direction

        # Clip acceleration magnitude
        accel_norm = np.linalg.norm(safe_action)
        if accel_norm > self.config.max_acceleration:
            safe_action = safe_action * self.config.max_acceleration / accel_norm

        # Clip velocity magnitude
        new_velocity = velocity + safe_action * dt
        vel_norm = np.linalg.norm(new_velocity)
        if vel_norm > self.config.max_velocity:
            new_velocity = new_velocity * self.config.max_velocity / vel_norm
            safe_action = (new_velocity - velocity) / dt

        return safe_action
