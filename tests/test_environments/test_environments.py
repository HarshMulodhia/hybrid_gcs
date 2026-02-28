"""
Unit tests for Environments module.

Tests base environment, grasping, drone navigation, and manipulation environments.
"""

import numpy as np
import pytest

from hybrid_gcs.environments import (
    BaseEnvironment,
    DroneNavConfig,
    DroneNavEnv,
    EnvConfig,
    GraspingConfig,
    GraspingEnv,
    ManipulationConfig,
    ManipulationEnv,
    ManipulationTask,
    StepResult,
)


class TestStepResult:
    """Test StepResult dataclass."""

    def test_creation(self):
        """Test creating a StepResult."""
        result = StepResult(
            observation=np.zeros(5),
            reward=1.0,
            done=False,
            truncated=False,
        )
        assert result.reward == 1.0
        assert not result.done
        assert result.info == {}


class TestGraspingEnv:
    """Test GraspingEnv class."""

    def test_creation(self):
        """Test environment creation."""
        env = GraspingEnv(GraspingConfig(seed=42))
        assert env.observation_dim == 11
        assert env.action_dim == 4

    def test_reset(self):
        """Test environment reset."""
        env = GraspingEnv(GraspingConfig(seed=42))
        obs = env.reset()
        assert obs.shape == (11,)
        assert env.step_count == 0

    def test_step(self):
        """Test a single environment step."""
        env = GraspingEnv(GraspingConfig(seed=42))
        env.reset()
        action = np.array([0.01, 0.0, -0.01, 0.0])
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert result.observation.shape == (11,)
        assert isinstance(result.reward, float)
        assert env.step_count == 1

    def test_observation_components(self):
        """Test that observation contains expected components."""
        env = GraspingEnv(GraspingConfig(seed=42))
        obs = env.reset()
        # ee_pos(3) + ee_vel(3) + obj_pos(3) + grasped(1) + dist(1) = 11
        assert len(obs) == 11
        # ee_vel should be zero after reset
        np.testing.assert_array_almost_equal(obs[3:6], [0.0, 0.0, 0.0])
        # grasped should be 0 after reset
        assert obs[9] == 0.0

    def test_fixed_object_position(self):
        """Test environment with fixed object position."""
        obj_pos = np.array([0.5, 0.0, 0.0])
        config = GraspingConfig(object_position=obj_pos, seed=42)
        env = GraspingEnv(config)
        obs = env.reset()
        np.testing.assert_array_almost_equal(obs[6:9], obj_pos)

    def test_truncation_at_max_steps(self):
        """Test episode truncation at max steps."""
        config = GraspingConfig(max_steps=5, seed=42)
        env = GraspingEnv(config)
        env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0])
        for _ in range(4):
            result = env.step(action)
            assert not result.truncated
        result = env.step(action)
        assert result.truncated

    def test_deterministic_reset(self):
        """Test that same seed produces same initial state."""
        env1 = GraspingEnv(GraspingConfig(seed=42))
        env2 = GraspingEnv(GraspingConfig(seed=42))
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_array_almost_equal(obs1, obs2)


class TestDroneNavEnv:
    """Test DroneNavEnv class."""

    def test_creation_single_agent(self):
        """Test single-agent environment creation."""
        env = DroneNavEnv(DroneNavConfig(num_agents=1, num_obstacles=3, seed=42))
        assert env.observation_dim == 12  # 9 + 3 obstacles
        assert env.action_dim == 3

    def test_creation_multi_agent(self):
        """Test multi-agent environment creation."""
        env = DroneNavEnv(DroneNavConfig(num_agents=3, num_obstacles=5, seed=42))
        assert env.observation_dim == 3 * (9 + 5)  # 3 agents * (9 + 5 obs)
        assert env.action_dim == 9  # 3 agents * 3

    def test_reset(self):
        """Test environment reset."""
        env = DroneNavEnv(DroneNavConfig(seed=42))
        obs = env.reset()
        assert obs.shape == (env.observation_dim,)
        assert env.step_count == 0

    def test_step(self):
        """Test a single step."""
        env = DroneNavEnv(DroneNavConfig(seed=42))
        env.reset()
        action = np.zeros(env.action_dim)
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert result.observation.shape == (env.observation_dim,)

    def test_multi_agent_step(self):
        """Test multi-agent step."""
        env = DroneNavEnv(DroneNavConfig(num_agents=2, seed=42))
        env.reset()
        action = np.zeros(env.action_dim)
        result = env.step(action)
        assert result.observation.shape == (env.observation_dim,)

    def test_truncation_at_max_steps(self):
        """Test episode truncation at max steps."""
        config = DroneNavConfig(max_steps=3, seed=42)
        env = DroneNavEnv(config)
        env.reset()
        action = np.zeros(env.action_dim)
        for _ in range(2):
            result = env.step(action)
        result = env.step(action)
        assert result.truncated or result.done

    def test_deterministic_reset(self):
        """Test that same seed produces same initial state."""
        env1 = DroneNavEnv(DroneNavConfig(seed=42))
        env2 = DroneNavEnv(DroneNavConfig(seed=42))
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_array_almost_equal(obs1, obs2)


class TestManipulationEnv:
    """Test ManipulationEnv class."""

    def test_creation_reach(self):
        """Test REACH task environment creation."""
        config = ManipulationConfig(task=ManipulationTask.REACH, seed=42)
        env = ManipulationEnv(config)
        assert env.observation_dim == 13  # 10 + 3*1
        assert env.action_dim == 4

    def test_creation_with_multiple_objects(self):
        """Test environment with multiple objects."""
        config = ManipulationConfig(task=ManipulationTask.STACK, num_objects=3, seed=42)
        env = ManipulationEnv(config)
        assert env.observation_dim == 19  # 10 + 3*3

    def test_reset(self):
        """Test environment reset."""
        config = ManipulationConfig(seed=42)
        env = ManipulationEnv(config)
        obs = env.reset()
        assert obs.shape == (env.observation_dim,)

    def test_step(self):
        """Test a single step."""
        config = ManipulationConfig(seed=42)
        env = ManipulationEnv(config)
        env.reset()
        action = np.array([0.01, 0.0, -0.01, 0.0])
        result = env.step(action)
        assert isinstance(result, StepResult)
        assert result.observation.shape == (env.observation_dim,)

    def test_reach_task(self):
        """Test REACH task basic execution."""
        config = ManipulationConfig(task=ManipulationTask.REACH, max_steps=10, seed=42)
        env = ManipulationEnv(config)
        obs = env.reset()
        for _ in range(10):
            action = np.array([0.0, 0.0, 0.0, 0.0])
            result = env.step(action)
        assert result.truncated or result.done

    def test_pick_task(self):
        """Test PICK task basic execution."""
        config = ManipulationConfig(task=ManipulationTask.PICK, max_steps=5, seed=42)
        env = ManipulationEnv(config)
        env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0])
        result = env.step(action)
        assert "grasped" in result.info

    def test_push_task(self):
        """Test PUSH task basic execution."""
        config = ManipulationConfig(task=ManipulationTask.PUSH, max_steps=5, seed=42)
        env = ManipulationEnv(config)
        env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0])
        result = env.step(action)
        assert "dist_obj_target" in result.info

    def test_stack_task(self):
        """Test STACK task basic execution."""
        config = ManipulationConfig(
            task=ManipulationTask.STACK, num_objects=2, max_steps=5, seed=42
        )
        env = ManipulationEnv(config)
        env.reset()
        action = np.array([0.0, 0.0, 0.0, 0.0])
        result = env.step(action)
        assert "placed" in result.info

    def test_deterministic_reset(self):
        """Test that same seed produces same initial state."""
        env1 = ManipulationEnv(ManipulationConfig(seed=42))
        env2 = ManipulationEnv(ManipulationConfig(seed=42))
        obs1 = env1.reset()
        obs2 = env2.reset()
        np.testing.assert_array_almost_equal(obs1, obs2)

    def test_all_task_types(self):
        """Test that all ManipulationTask values work."""
        for task in ManipulationTask:
            config = ManipulationConfig(task=task, num_objects=2, seed=42)
            env = ManipulationEnv(config)
            obs = env.reset()
            assert obs.shape == (env.observation_dim,)
            result = env.step(np.zeros(4))
            assert isinstance(result.reward, float)
