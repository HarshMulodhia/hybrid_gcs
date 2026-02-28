"""
Tests for the Hybrid-GCS CLI modules (train, evaluate, visualize).
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from hybrid_gcs.cli.train import (
    ENV_FACTORY,
    build_parser as train_parser,
    collect_rollout,
    evaluate_policy,
    train,
)
from hybrid_gcs.cli.evaluate import (
    build_parser as eval_parser,
    evaluate,
    load_policy,
    run_evaluation,
)
from hybrid_gcs.cli.visualize import (
    _record_episode,
    _sphere_entity,
    build_parser as vis_parser,
)
from hybrid_gcs.environments import (
    DroneNavConfig,
    DroneNavEnv,
    GraspingConfig,
    GraspingEnv,
    ManipulationConfig,
    ManipulationEnv,
    ManipulationTask,
)
from hybrid_gcs.training import PolicyNetwork, PolicyNetworkConfig, PPOConfig, PPOTrainer
from hybrid_gcs.visualization.pybullet_renderer import PyBulletRenderer


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory."""
    return tmp_path


@pytest.fixture
def grasping_env():
    return GraspingEnv(GraspingConfig(max_steps=50, seed=42))


@pytest.fixture
def drone_env():
    return DroneNavEnv(DroneNavConfig(max_steps=50, seed=42, num_agents=2, num_obstacles=3))


@pytest.fixture
def manip_env():
    return ManipulationEnv(
        ManipulationConfig(task=ManipulationTask.REACH, max_steps=50, seed=42)
    )


@pytest.fixture
def policy_and_checkpoint(grasping_env, tmp_dir):
    """Create a small policy, do one PPO update, save checkpoint."""
    pol_cfg = PolicyNetworkConfig(
        state_dim=grasping_env.observation_dim,
        action_dim=grasping_env.action_dim,
        hidden_dim=32,
        num_hidden_layers=2,
    )
    policy = PolicyNetwork(pol_cfg)
    ppo_cfg = PPOConfig(learning_rate=1e-3, epochs=1, batch_size=16, num_steps=64)
    trainer = PPOTrainer(policy, ppo_cfg)

    # Collect a short rollout to make the checkpoint realistic
    obs = grasping_env.reset()
    states, actions, rewards, values, dones = [], [], [], [], []
    for _ in range(64):
        a, v = policy.get_action(obs)
        r = grasping_env.step(a)
        states.append(obs)
        actions.append(a)
        rewards.append(r.reward)
        values.append(v)
        dones.append(float(r.done or r.truncated))
        obs = r.observation
        if r.done or r.truncated:
            obs = grasping_env.reset()

    _, nv = policy.get_action(obs)
    trainer.update(
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.float32),
        np.array(rewards, dtype=np.float32),
        np.array(values, dtype=np.float32),
        np.array(dones, dtype=np.float32),
        float(nv),
    )

    ckpt_path = str(tmp_dir / "test.pth")
    trainer.save_checkpoint(ckpt_path)
    return policy, ckpt_path


# ======================================================================
# Train CLI tests
# ======================================================================


class TestTrainCLI:
    """Tests for hybrid_gcs.cli.train."""

    def test_parser_defaults(self):
        parser = train_parser()
        args = parser.parse_args(["--env", "grasping"])
        assert args.env == "grasping"
        assert args.episodes == 100
        assert args.seed == 42

    def test_parser_all_envs(self):
        for env_name in ("grasping", "drone_nav", "manipulation"):
            args = train_parser().parse_args(["--env", env_name])
            assert args.env == env_name

    def test_parser_manipulation_task(self):
        args = train_parser().parse_args(["--env", "manipulation", "--task", "stack"])
        assert args.task == "stack"

    def test_parser_drone_agents(self):
        args = train_parser().parse_args(["--env", "drone_nav", "--num-agents", "3"])
        assert args.num_agents == 3

    def test_parser_dual_arm(self):
        args = train_parser().parse_args(["--env", "grasping", "--dual-arm"])
        assert args.dual_arm is True

    def test_env_factory_grasping(self):
        args = train_parser().parse_args(["--env", "grasping", "--max-steps", "50"])
        env = ENV_FACTORY["grasping"](args)
        assert env.observation_dim == 11
        assert env.action_dim == 4

    def test_env_factory_drone_nav(self):
        args = train_parser().parse_args(
            ["--env", "drone_nav", "--num-agents", "2", "--num-obstacles", "3", "--max-steps", "50"]
        )
        env = ENV_FACTORY["drone_nav"](args)
        assert env.observation_dim == 2 * (9 + 3)
        assert env.action_dim == 2 * 3

    def test_env_factory_manipulation(self):
        for task in ("reach", "pick", "push", "stack"):
            args = train_parser().parse_args(
                ["--env", "manipulation", "--task", task, "--max-steps", "50"]
            )
            env = ENV_FACTORY["manipulation"](args)
            assert env.action_dim == 4

    def test_collect_rollout(self, grasping_env):
        pol_cfg = PolicyNetworkConfig(
            state_dim=grasping_env.observation_dim,
            action_dim=grasping_env.action_dim,
            hidden_dim=32,
            num_hidden_layers=2,
        )
        policy = PolicyNetwork(pol_cfg)
        s, a, r, v, d, nv = collect_rollout(grasping_env, policy, num_steps=32)
        assert s.shape == (32, grasping_env.observation_dim)
        assert a.shape == (32, grasping_env.action_dim)
        assert len(r) == 32

    def test_evaluate_policy(self, grasping_env):
        pol_cfg = PolicyNetworkConfig(
            state_dim=grasping_env.observation_dim,
            action_dim=grasping_env.action_dim,
            hidden_dim=32,
            num_hidden_layers=2,
        )
        policy = PolicyNetwork(pol_cfg)
        metrics = evaluate_policy(grasping_env, policy, num_episodes=2)
        assert "mean_reward" in metrics
        assert "success_rate" in metrics
        assert 0.0 <= metrics["success_rate"] <= 1.0

    def test_train_short_grasping(self, tmp_dir):
        args = train_parser().parse_args(
            [
                "--env", "grasping",
                "--episodes", "2",
                "--max-steps", "50",
                "--eval-interval", "1",
                "--output-dir", str(tmp_dir),
                "--seed", "42",
            ]
        )
        path = train(args)
        assert Path(path).exists()
        assert (tmp_dir / "training_history.json").exists()
        assert (tmp_dir / "train_config.json").exists()

    def test_train_short_drone_nav(self, tmp_dir):
        args = train_parser().parse_args(
            [
                "--env", "drone_nav",
                "--num-agents", "2",
                "--num-obstacles", "3",
                "--episodes", "2",
                "--max-steps", "50",
                "--eval-interval", "1",
                "--output-dir", str(tmp_dir / "drone"),
                "--seed", "42",
            ]
        )
        path = train(args)
        assert Path(path).exists()

    def test_train_short_manipulation(self, tmp_dir):
        for task in ("reach", "pick", "push", "stack"):
            out = tmp_dir / f"manip_{task}"
            args = train_parser().parse_args(
                [
                    "--env", "manipulation",
                    "--task", task,
                    "--episodes", "2",
                    "--max-steps", "50",
                    "--eval-interval", "1",
                    "--output-dir", str(out),
                    "--seed", "42",
                ]
            )
            path = train(args)
            assert Path(path).exists()

    def test_train_dual_arm_grasping(self, tmp_dir):
        args = train_parser().parse_args(
            [
                "--env", "grasping",
                "--dual-arm",
                "--episodes", "2",
                "--max-steps", "50",
                "--eval-interval", "1",
                "--output-dir", str(tmp_dir / "dual"),
                "--seed", "42",
            ]
        )
        path = train(args)
        assert Path(path).exists()


# ======================================================================
# Evaluate CLI tests
# ======================================================================


class TestEvaluateCLI:
    """Tests for hybrid_gcs.cli.evaluate."""

    def test_parser_defaults(self):
        parser = eval_parser()
        args = parser.parse_args(["--env", "grasping", "--checkpoint", "x.pth"])
        assert args.episodes == 20
        assert args.record is False

    def test_load_policy(self, grasping_env, policy_and_checkpoint):
        _, ckpt_path = policy_and_checkpoint
        policy = load_policy(ckpt_path, grasping_env)
        assert isinstance(policy, PolicyNetwork)

    def test_run_evaluation(self, grasping_env, policy_and_checkpoint):
        policy, _ = policy_and_checkpoint
        metrics = run_evaluation(grasping_env, policy, num_episodes=3, record_episodes=True)
        assert "mean_reward" in metrics
        assert "episodes" in metrics
        assert len(metrics["episodes"]) == 3

    def test_evaluate_grasping(self, policy_and_checkpoint, tmp_dir):
        _, ckpt_path = policy_and_checkpoint
        out_json = str(tmp_dir / "metrics.json")
        args = eval_parser().parse_args(
            [
                "--env", "grasping",
                "--checkpoint", ckpt_path,
                "--episodes", "3",
                "--max-steps", "50",
                "--output", out_json,
                "--seed", "42",
            ]
        )
        metrics = evaluate(args)
        assert metrics["num_episodes"] == 3
        assert Path(out_json).exists()


# ======================================================================
# Visualize CLI tests
# ======================================================================


class TestVisualizeCLI:
    """Tests for hybrid_gcs.cli.visualize."""

    def test_parser_defaults(self):
        parser = vis_parser()
        args = parser.parse_args(["--env", "grasping", "--checkpoint", "x.pth"])
        assert args.backend == "foxglove"

    def test_record_episode(self, grasping_env):
        pol_cfg = PolicyNetworkConfig(
            state_dim=grasping_env.observation_dim,
            action_dim=grasping_env.action_dim,
            hidden_dim=32,
            num_hidden_layers=2,
        )
        policy = PolicyNetwork(pol_cfg)
        ep = _record_episode(grasping_env, policy)
        assert "frames" in ep
        assert len(ep["frames"]) > 0
        assert "observation" in ep["frames"][0]

    def test_sphere_entity(self):
        ent = _sphere_entity("test", np.array([1.0, 2.0, 3.0]), 0.1, (1, 0, 0, 1), 0, 0)
        assert ent.id == "test"
        assert len(ent.spheres) == 1

    def test_foxglove_visualization(self, policy_and_checkpoint, tmp_dir):
        _, ckpt_path = policy_and_checkpoint
        out_mcap = str(tmp_dir / "test.mcap")
        args = vis_parser().parse_args(
            [
                "--env", "grasping",
                "--checkpoint", ckpt_path,
                "--backend", "foxglove",
                "--output", out_mcap,
                "--max-steps", "50",
                "--seed", "42",
            ]
        )
        from hybrid_gcs.cli.visualize import visualize

        result = visualize(args)
        assert Path(result).exists()
        assert result.endswith(".mcap")


# ======================================================================
# PyBullet Renderer tests
# ======================================================================


class TestPyBulletRenderer:
    """Tests for hybrid_gcs.visualization.pybullet_renderer."""

    def test_renderer_creation(self):
        renderer = PyBulletRenderer(mode="direct")
        assert isinstance(renderer, PyBulletRenderer)

    def test_save_trajectory_json(self, tmp_dir):
        frames = [{"step": 0, "observation": [1, 2, 3]}]
        path = PyBulletRenderer.save_trajectory_json(frames, str(tmp_dir / "traj.json"))
        assert Path(path).exists()
        with open(path) as fh:
            data = json.load(fh)
        assert len(data) == 1

    def test_available_property(self):
        renderer = PyBulletRenderer()
        # available depends on whether pybullet is installed; just check the type
        assert isinstance(renderer.available, bool)

    def test_context_manager(self):
        renderer = PyBulletRenderer(mode="direct")
        if renderer.available:
            with renderer:
                assert renderer.physics_client >= 0
            assert renderer.physics_client == -1
