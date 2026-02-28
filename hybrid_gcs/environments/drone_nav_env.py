"""
Autonomous Drone Navigation Environment for Hybrid-GCS.

Simulates single or multi-agent drone navigation through an obstacle field.
Uses simple kinematic simulation (no PyBullet dependency).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_env import BaseEnvironment, EnvConfig, StepResult

# Default 3D world bounds: 10m x 10m x 5m
_DEFAULT_WORLD_LOWER = np.array([0.0, 0.0, 0.0], dtype=np.float64)
_DEFAULT_WORLD_UPPER = np.array([10.0, 10.0, 5.0], dtype=np.float64)


@dataclass
class DroneNavConfig(EnvConfig):
    """
    Configuration for the drone navigation environment.

    Attributes:
        world_bounds: (lower, upper) 3D bounds of the world
        num_agents: Number of drones (1 for single-agent)
        num_obstacles: Number of spherical obstacles
        obstacle_radius: Radius of each obstacle
        goal_threshold: Distance to goal considered as reached
        max_velocity: Maximum velocity magnitude per drone
        collision_radius: Collision radius for drones
    """

    world_bounds: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (_DEFAULT_WORLD_LOWER.copy(), _DEFAULT_WORLD_UPPER.copy())
    )
    num_agents: int = 1
    num_obstacles: int = 5
    obstacle_radius: float = 0.5
    goal_threshold: float = 0.3
    max_velocity: float = 2.0
    collision_radius: float = 0.3


class DroneNavEnv(BaseEnvironment):
    """
    Autonomous drone navigation environment.

    Simulates one or more drones navigating to goal positions while avoiding
    obstacles and each other. Uses simple velocity-based kinematic simulation.

    State per agent (9 + num_obstacles):
        [position(3), velocity(3), goal(3), obstacle_distances(num_obstacles)]

    For multi-agent: all agent states are concatenated.

    Action per agent (3-dim):
        [acceleration(3)]

    Dynamics: velocity += acceleration * dt, position += velocity * dt.
    For multi-agent: all agent actions are concatenated.

    Examples:
        >>> config = DroneNavConfig(num_agents=2, num_obstacles=3, seed=42)
        >>> env = DroneNavEnv(config)
        >>> obs = env.reset()
        >>> action = np.zeros(env.action_dim)
        >>> result = env.step(action)
    """

    def __init__(self, config: DroneNavConfig) -> None:
        """
        Initialize the drone navigation environment.

        Args:
            config: Drone navigation configuration
        """
        super().__init__(config)
        self.nav_config = config

        n = config.num_agents
        self._positions = np.zeros((n, 3), dtype=np.float64)
        self._velocities = np.zeros((n, 3), dtype=np.float64)
        self._goals = np.zeros((n, 3), dtype=np.float64)
        self._obstacles = np.zeros((config.num_obstacles, 3), dtype=np.float64)

    @property
    def _state_dim_per_agent(self) -> int:
        """State dimension for a single agent."""
        return 9 + self.nav_config.num_obstacles

    @property
    def observation_dim(self) -> int:
        """Total observation dimension across all agents."""
        return self.nav_config.num_agents * self._state_dim_per_agent

    @property
    def action_dim(self) -> int:
        """Total action dimension across all agents (3 per agent)."""
        return self.nav_config.num_agents * 3

    def reset(self) -> np.ndarray:
        """
        Reset environment: randomize positions, goals, and obstacles.

        Returns:
            Initial observation array
        """
        self.step_count = 0
        cfg = self.nav_config
        w_lo, w_hi = cfg.world_bounds

        # Random start positions for agents
        for i in range(cfg.num_agents):
            self._positions[i] = self.rng.uniform(w_lo, w_hi)
            self._goals[i] = self.rng.uniform(w_lo, w_hi)
        self._velocities = np.zeros_like(self._velocities)

        # Random obstacle positions (ensuring they are inside world bounds)
        for i in range(cfg.num_obstacles):
            self._obstacles[i] = self.rng.uniform(w_lo, w_hi)

        return self.get_observation()

    def step(self, action: np.ndarray) -> StepResult:
        """
        Execute one environment step for all agents.

        Args:
            action: Concatenated acceleration for all agents [num_agents * 3]

        Returns:
            StepResult with observation, reward, done, truncated, info
        """
        action = np.asarray(action, dtype=np.float64)
        self.step_count += 1
        cfg = self.nav_config
        dt = self.config.dt

        # Parse per-agent actions
        actions = action.reshape(cfg.num_agents, 3)

        # Update dynamics
        self._velocities += actions * dt
        # Clamp velocity magnitude
        for i in range(cfg.num_agents):
            speed = np.linalg.norm(self._velocities[i])
            if speed > cfg.max_velocity:
                self._velocities[i] *= cfg.max_velocity / speed

        self._positions += self._velocities * dt

        # Clamp positions to world bounds
        w_lo, w_hi = cfg.world_bounds
        self._positions = np.clip(self._positions, w_lo, w_hi)

        # Compute reward and check termination
        reward, done, info = self._evaluate()

        truncated = False
        if not done and self.step_count >= self.config.max_steps:
            truncated = True
            info["success"] = False

        return StepResult(
            observation=self.get_observation(),
            reward=reward,
            done=done,
            truncated=truncated,
            info=info,
        )

    def get_observation(self) -> np.ndarray:
        """
        Build observation by concatenating all agent states.

        Returns:
            Observation array [num_agents * (9 + num_obstacles)]
        """
        cfg = self.nav_config
        obs_parts = []
        for i in range(cfg.num_agents):
            obs_dists = np.array(
                [
                    np.linalg.norm(self._positions[i] - self._obstacles[j])
                    for j in range(cfg.num_obstacles)
                ],
                dtype=np.float64,
            )
            agent_obs = np.concatenate(
                [self._positions[i], self._velocities[i], self._goals[i], obs_dists]
            )
            obs_parts.append(agent_obs)
        return np.concatenate(obs_parts)

    def _evaluate(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward and check termination conditions.

        Returns:
            Tuple of (reward, done, info)
        """
        cfg = self.nav_config
        total_reward = 0.0
        all_reached = True
        collision = False
        info: Dict[str, Any] = {"agents_reached": []}

        for i in range(cfg.num_agents):
            dist_to_goal = np.linalg.norm(self._positions[i] - self._goals[i])
            reached = dist_to_goal < cfg.goal_threshold
            info["agents_reached"].append(reached)

            # Distance-based reward
            total_reward -= dist_to_goal

            # Goal bonus
            if reached:
                total_reward += 10.0
            else:
                all_reached = False

            # Obstacle collision penalty
            for j in range(cfg.num_obstacles):
                dist_to_obs = np.linalg.norm(self._positions[i] - self._obstacles[j])
                if dist_to_obs < cfg.obstacle_radius + cfg.collision_radius:
                    total_reward -= 50.0
                    collision = True

        # Inter-agent collision penalty
        for i in range(cfg.num_agents):
            for j in range(i + 1, cfg.num_agents):
                inter_dist = np.linalg.norm(self._positions[i] - self._positions[j])
                if inter_dist < 2.0 * cfg.collision_radius:
                    total_reward -= 50.0
                    collision = True

        done = all_reached or collision
        info["collision"] = collision
        info["success"] = all_reached and not collision

        return total_reward, done, info
