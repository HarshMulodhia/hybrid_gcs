"""
Complex Manipulation Environment for Hybrid-GCS.

Multi-primitive manipulation tasks: reach, pick, push, and stack.
Uses simple kinematic simulation (no PyBullet dependency).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_env import BaseEnvironment, EnvConfig, StepResult

# Default table workspace bounds
_DEFAULT_WS_LOWER = np.array([0.3, -0.3, 0.0], dtype=np.float64)
_DEFAULT_WS_UPPER = np.array([0.8, 0.3, 0.6], dtype=np.float64)


class ManipulationTask(Enum):
    """Enum of supported manipulation task types."""

    REACH = "reach"
    PICK = "pick"
    PUSH = "push"
    STACK = "stack"


@dataclass
class ManipulationConfig(EnvConfig):
    """
    Configuration for the manipulation environment.

    Attributes:
        task: Manipulation task type
        workspace_bounds: (lower, upper) 3D bounds of the workspace
        num_objects: Number of objects in the scene
        num_dof: Number of degrees of freedom of the robot arm
    """

    task: ManipulationTask = ManipulationTask.REACH
    workspace_bounds: Tuple[np.ndarray, np.ndarray] = field(
        default_factory=lambda: (_DEFAULT_WS_LOWER.copy(), _DEFAULT_WS_UPPER.copy())
    )
    num_objects: int = 1
    num_dof: int = 7


class ManipulationEnv(BaseEnvironment):
    """
    Complex manipulation environment with multiple task primitives.

    Supports four task types with increasing complexity:
        - REACH: Move end-effector to a target position
        - PICK: Reach, grasp, and lift an object
        - PUSH: Push an object to a target position
        - STACK: Pick an object and place it on top of another

    State (10 + 3 * num_objects):
        [ee_position(3), ee_velocity(3), gripper_state(1),
         object_positions(3 * num_objects), target_position(3)]

    Action (4-dim):
        [ee_velocity_command(3), gripper_command(1)]

    The ee_velocity_command directly sets the end-effector velocity.
    Position is integrated as pos += velocity * dt.

    Examples:
        >>> config = ManipulationConfig(
        ...     task=ManipulationTask.PICK, num_objects=1, seed=42
        ... )
        >>> env = ManipulationEnv(config)
        >>> obs = env.reset()
        >>> action = np.array([0.01, 0.0, -0.01, 1.0])
        >>> result = env.step(action)
    """

    _GRASP_THRESHOLD = 0.05
    _REACH_THRESHOLD = 0.05
    _STACK_THRESHOLD = 0.08
    _OBJECT_HEIGHT = 0.04

    def __init__(self, config: ManipulationConfig) -> None:
        """
        Initialize the manipulation environment.

        Args:
            config: Manipulation environment configuration
        """
        super().__init__(config)
        self.manip_config = config

        self._ee_pos = np.zeros(3, dtype=np.float64)
        self._ee_vel = np.zeros(3, dtype=np.float64)
        self._gripper_state = 0.0  # 0=open, 1=closed
        self._obj_positions = np.zeros(
            (config.num_objects, 3), dtype=np.float64
        )
        self._target_pos = np.zeros(3, dtype=np.float64)
        self._grasped_idx: Optional[int] = None

    @property
    def observation_dim(self) -> int:
        """Observation dimension: 10 + 3 * num_objects."""
        return 10 + 3 * self.manip_config.num_objects

    @property
    def action_dim(self) -> int:
        """Action dimension: delta_ee(3) + gripper(1)."""
        return 4

    def reset(self) -> np.ndarray:
        """
        Reset environment and randomize object/target positions.

        Returns:
            Initial observation array
        """
        self.step_count = 0
        self._grasped_idx = None
        self._gripper_state = 0.0
        self._ee_vel = np.zeros(3, dtype=np.float64)

        cfg = self.manip_config
        ws_lo, ws_hi = cfg.workspace_bounds
        ws_center = (ws_lo + ws_hi) / 2.0

        # End-effector starts above workspace center
        self._ee_pos = np.array(
            [ws_center[0], ws_center[1], ws_hi[2]], dtype=np.float64
        )

        # Place objects on the table surface
        for i in range(cfg.num_objects):
            self._obj_positions[i] = np.array(
                [
                    self.rng.uniform(ws_lo[0], ws_hi[0]),
                    self.rng.uniform(ws_lo[1], ws_hi[1]),
                    ws_lo[2],
                ],
                dtype=np.float64,
            )

        # Set target position based on task
        self._target_pos = self._sample_target(ws_lo, ws_hi)

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
        ws_lo, ws_hi = self.manip_config.workspace_bounds

        # Update end-effector
        self._ee_vel = action[:3]
        gripper_cmd = action[3]

        self._ee_pos = self._ee_pos + self._ee_vel * dt
        self._ee_pos = np.clip(self._ee_pos, ws_lo, ws_hi)

        # Gripper logic
        self._gripper_state = 1.0 if gripper_cmd > 0.5 else 0.0
        self._update_grasp()

        # Move grasped object with end-effector
        if self._grasped_idx is not None:
            self._obj_positions[self._grasped_idx] = self._ee_pos.copy()

        # Compute task-specific reward
        reward, done, info = self._evaluate_task()

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
        Build the current observation vector.

        Returns:
            Observation array [10 + 3 * num_objects]
        """
        return np.concatenate(
            [
                self._ee_pos,
                self._ee_vel,
                np.array([self._gripper_state]),
                self._obj_positions.flatten(),
                self._target_pos,
            ]
        )

    def _sample_target(
        self, ws_lo: np.ndarray, ws_hi: np.ndarray
    ) -> np.ndarray:
        """
        Sample a target position appropriate for the current task.

        Args:
            ws_lo: Lower workspace bounds
            ws_hi: Upper workspace bounds

        Returns:
            Target position (3-dim)
        """
        task = self.manip_config.task

        if task == ManipulationTask.REACH:
            # Random position in workspace
            return np.array(
                [
                    self.rng.uniform(ws_lo[0], ws_hi[0]),
                    self.rng.uniform(ws_lo[1], ws_hi[1]),
                    self.rng.uniform(ws_lo[2], ws_hi[2]),
                ],
                dtype=np.float64,
            )
        elif task == ManipulationTask.PICK:
            # Target is above the table
            return np.array(
                [
                    self.rng.uniform(ws_lo[0], ws_hi[0]),
                    self.rng.uniform(ws_lo[1], ws_hi[1]),
                    self.rng.uniform(0.2, ws_hi[2]),
                ],
                dtype=np.float64,
            )
        elif task == ManipulationTask.PUSH:
            # Target on the table surface
            return np.array(
                [
                    self.rng.uniform(ws_lo[0], ws_hi[0]),
                    self.rng.uniform(ws_lo[1], ws_hi[1]),
                    ws_lo[2],
                ],
                dtype=np.float64,
            )
        else:
            # STACK: target on top of second object (or random if only one)
            if self.manip_config.num_objects >= 2:
                base = self._obj_positions[1].copy()
                base[2] += self._OBJECT_HEIGHT
                return base
            return np.array(
                [
                    self.rng.uniform(ws_lo[0], ws_hi[0]),
                    self.rng.uniform(ws_lo[1], ws_hi[1]),
                    ws_lo[2] + self._OBJECT_HEIGHT,
                ],
                dtype=np.float64,
            )

    def _update_grasp(self) -> None:
        """Update grasp state based on gripper and proximity."""
        if self._gripper_state < 0.5:
            # Gripper open: release
            self._grasped_idx = None
            return

        if self._grasped_idx is not None:
            return  # Already holding something

        # Try to grasp the nearest object
        for i in range(self.manip_config.num_objects):
            dist = np.linalg.norm(self._ee_pos - self._obj_positions[i])
            if dist < self._GRASP_THRESHOLD:
                self._grasped_idx = i
                return

    def _evaluate_task(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Evaluate reward and termination based on current task.

        Returns:
            Tuple of (reward, done, info)
        """
        task = self.manip_config.task
        info: Dict[str, Any] = {"task": task.value}

        if task == ManipulationTask.REACH:
            return self._eval_reach(info)
        elif task == ManipulationTask.PICK:
            return self._eval_pick(info)
        elif task == ManipulationTask.PUSH:
            return self._eval_push(info)
        else:
            return self._eval_stack(info)

    def _eval_reach(
        self, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Evaluate REACH task: move ee to target."""
        dist = np.linalg.norm(self._ee_pos - self._target_pos)
        reward = -dist
        done = dist < self._REACH_THRESHOLD
        if done:
            reward += 10.0
        info["distance_to_target"] = dist
        info["success"] = done
        return reward, done, info

    def _eval_pick(
        self, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Evaluate PICK task: reach + grasp + lift to target."""
        obj_pos = self._obj_positions[0]
        dist_ee_obj = np.linalg.norm(self._ee_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - self._target_pos)

        reward = -dist_ee_obj - dist_obj_target
        if self._grasped_idx == 0:
            reward += 1.0
        done = self._grasped_idx == 0 and dist_obj_target < self._REACH_THRESHOLD
        if done:
            reward += 10.0

        info["dist_ee_obj"] = dist_ee_obj
        info["dist_obj_target"] = dist_obj_target
        info["grasped"] = self._grasped_idx == 0
        info["success"] = done
        return reward, done, info

    def _eval_push(
        self, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Evaluate PUSH task: push object to target on table."""
        obj_pos = self._obj_positions[0]
        dist_ee_obj = np.linalg.norm(self._ee_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - self._target_pos)

        # Push: reward for ee near object + object near target
        reward = -0.5 * dist_ee_obj - dist_obj_target
        done = dist_obj_target < self._REACH_THRESHOLD
        if done:
            reward += 10.0

        info["dist_ee_obj"] = dist_ee_obj
        info["dist_obj_target"] = dist_obj_target
        info["success"] = done
        return reward, done, info

    def _eval_stack(
        self, info: Dict[str, Any]
    ) -> Tuple[float, bool, Dict[str, Any]]:
        """Evaluate STACK task: pick first object and place on target."""
        obj_pos = self._obj_positions[0]
        dist_ee_obj = np.linalg.norm(self._ee_pos - obj_pos)
        dist_obj_target = np.linalg.norm(obj_pos - self._target_pos)

        reward = -dist_ee_obj - dist_obj_target
        if self._grasped_idx == 0:
            reward += 1.0

        # Done when object is near stack target and released
        placed = (
            dist_obj_target < self._STACK_THRESHOLD
            and self._grasped_idx is None
        )
        done = placed
        if done:
            reward += 10.0

        info["dist_ee_obj"] = dist_ee_obj
        info["dist_obj_target"] = dist_obj_target
        info["grasped"] = self._grasped_idx == 0
        info["placed"] = placed
        info["success"] = done
        return reward, done, info
