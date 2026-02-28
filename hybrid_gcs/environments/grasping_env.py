"""
YCB Object Grasping Environment for Hybrid-GCS.

Simulates single-arm reaching and grasping of objects on a table workspace.
Uses simple kinematic simulation (no PyBullet dependency).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_env import BaseEnvironment, EnvConfig, StepResult

# Default table workspace bounds: x=[0.3, 0.8], y=[-0.3, 0.3], z=[0.0, 0.6]
_DEFAULT_WS_LOWER = np.array([0.3, -0.3, 0.0], dtype=np.float64)
_DEFAULT_WS_UPPER = np.array([0.8, 0.3, 0.6], dtype=np.float64)


@dataclass
class GraspingConfig(EnvConfig):
    """
    Configuration for the YCB grasping environment.

    Attributes:
        workspace_bounds: (lower, upper) 3D bounds of the workspace
        num_dof: Number of degrees of freedom of the robot arm
        object_position: Fixed object position, or None for random placement
        grasp_threshold: Distance threshold to consider object graspable
        lift_target: Target height for successful lift
    """

    workspace_bounds: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (_DEFAULT_WS_LOWER.copy(), _DEFAULT_WS_UPPER.copy())
    )
    num_dof: int = 7
    object_position: Optional[np.ndarray] = None
    grasp_threshold: float = 0.05
    lift_target: float = 0.3


class GraspingEnv(BaseEnvironment):
    """
    YCB object grasping environment.

    Simulates an end-effector reaching for and grasping an object on a table.
    The environment uses simple kinematic simulation where the end-effector
    position is updated by delta commands.

    State (11-dim):
        [ee_position(3), ee_velocity(3), object_position(3),
         object_grasped(1), distance_to_object(1)]

    Action (4-dim):
        [ee_velocity_command(3), gripper_command(1)]

    The ee_velocity_command directly sets the end-effector velocity. Position
    is integrated as pos += velocity * dt. The gripper closes when
    gripper_command > 0.5. The episode ends when the object is lifted to
    the target height or max steps is reached.

    Examples:
        >>> config = GraspingConfig(max_steps=200, seed=42)
        >>> env = GraspingEnv(config)
        >>> obs = env.reset()
        >>> action = np.array([0.01, 0.0, 0.0, 0.0])
        >>> result = env.step(action)
    """

    def __init__(self, config: GraspingConfig) -> None:
        """
        Initialize the grasping environment.

        Args:
            config: Grasping environment configuration
        """
        super().__init__(config)
        self.grasping_config = config

        # State variables
        self._ee_pos = np.zeros(3, dtype=np.float64)
        self._ee_vel = np.zeros(3, dtype=np.float64)
        self._obj_pos = np.zeros(3, dtype=np.float64)
        self._grasped = False

    @property
    def observation_dim(self) -> int:
        """Dimension of observation: ee(3) + vel(3) + obj(3) + grasped(1) + dist(1)."""
        return 11

    @property
    def action_dim(self) -> int:
        """Dimension of action: delta_ee(3) + gripper(1)."""
        return 4

    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state.

        Returns:
            Initial observation (11-dim)
        """
        self.step_count = 0
        self._grasped = False
        self._ee_vel = np.zeros(3, dtype=np.float64)

        ws_lo, ws_hi = self.grasping_config.workspace_bounds

        # Initialize end-effector above the workspace center
        ws_center = (ws_lo + ws_hi) / 2.0
        self._ee_pos = np.array(
            [ws_center[0], ws_center[1], ws_hi[2]], dtype=np.float64
        )

        # Place object on the table surface (z = ws_lo[2])
        if self.grasping_config.object_position is not None:
            self._obj_pos = np.asarray(
                self.grasping_config.object_position, dtype=np.float64
            ).copy()
        else:
            self._obj_pos = np.array(
                [
                    self.rng.uniform(ws_lo[0], ws_hi[0]),
                    self.rng.uniform(ws_lo[1], ws_hi[1]),
                    ws_lo[2],
                ],
                dtype=np.float64,
            )

        return self.get_observation()

    def step(self, action: np.ndarray) -> StepResult:
        """
        Execute one environment step.

        Args:
            action: [ee_velocity_command(3), gripper_command(1)]

        Returns:
            StepResult with observation, reward, done, truncated, info
        """
        action = np.asarray(action, dtype=np.float64)
        self.step_count += 1
        dt = self.config.dt
        ws_lo, ws_hi = self.grasping_config.workspace_bounds

        # Update end-effector kinematics
        self._ee_vel = action[:3]
        gripper_cmd = action[3]

        self._ee_pos = self._ee_pos + self._ee_vel * dt
        self._ee_pos = np.clip(self._ee_pos, ws_lo, ws_hi)

        # Gripper logic
        dist = np.linalg.norm(self._ee_pos - self._obj_pos)
        if (
            gripper_cmd > 0.5
            and dist < self.grasping_config.grasp_threshold
            and not self._grasped
        ):
            self._grasped = True

        # If grasped, object follows end-effector
        if self._grasped:
            self._obj_pos = self._ee_pos.copy()

        # Compute reward
        reward = self._compute_reward(dist)

        # Termination conditions
        done = False
        truncated = False
        info: Dict[str, Any] = {
            "distance": dist,
            "grasped": self._grasped,
            "object_height": self._obj_pos[2],
        }

        if self._grasped and self._obj_pos[2] >= self.grasping_config.lift_target:
            done = True
            info["success"] = True
        elif self.step_count >= self.config.max_steps:
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
        Build the current observation vector.

        Returns:
            Observation array (11-dim)
        """
        dist = np.linalg.norm(self._ee_pos - self._obj_pos)
        return np.concatenate(
            [
                self._ee_pos,
                self._ee_vel,
                self._obj_pos,
                np.array([float(self._grasped)]),
                np.array([dist]),
            ]
        )

    def _compute_reward(self, distance: float) -> float:
        """
        Compute the step reward.

        Args:
            distance: Current distance between end-effector and object

        Returns:
            Scalar reward
        """
        # Distance reward: encourage moving closer to object
        reward = -distance

        # Grasp bonus
        if self._grasped:
            reward += 1.0
            # Lift reward: encourage lifting
            reward += self._obj_pos[2] * 2.0

        # Success bonus
        if (
            self._grasped
            and self._obj_pos[2] >= self.grasping_config.lift_target
        ):
            reward += 10.0

        return reward
