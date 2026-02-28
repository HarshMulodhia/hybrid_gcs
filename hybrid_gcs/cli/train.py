"""
Training Script for Hybrid-GCS.

Trains PPO policies for three application domains:
    - grasping: YCB object grasping (single-arm or dual-arm)
    - drone_nav: Autonomous drone navigation (single or multi-agent)
    - manipulation: Complex manipulation (reach, pick, push, stack)

Usage:
    hybrid-gcs-train --env grasping --seed 42
    hybrid-gcs-train --env drone_nav --num-agents 3 --episodes 500
    hybrid-gcs-train --env manipulation --task pick --episodes 300
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from hybrid_gcs.environments import (
    DroneNavConfig,
    DroneNavEnv,
    GraspingConfig,
    GraspingEnv,
    ManipulationConfig,
    ManipulationEnv,
    ManipulationTask,
)
from hybrid_gcs.training import (
    PolicyNetwork,
    PolicyNetworkConfig,
    PPOConfig,
    PPOTrainer,
)


# ---------------------------------------------------------------------------
# Domain-specific environment + reward configuration
# ---------------------------------------------------------------------------


def _create_grasping_env(args: argparse.Namespace) -> GraspingEnv:
    """Create a YCB grasping environment (single or dual-arm)."""
    cfg = GraspingConfig(
        max_steps=args.max_steps,
        dt=0.01,
        seed=args.seed,
        grasp_threshold=0.05,
        lift_target=0.3,
    )
    if args.dual_arm:
        # Dual-arm: widen workspace and allow symmetric placement
        ws_lo = np.array([0.2, -0.5, 0.0], dtype=np.float64)
        ws_hi = np.array([0.9, 0.5, 0.6], dtype=np.float64)
        cfg.workspace_bounds = (ws_lo, ws_hi)
    return GraspingEnv(cfg)


def _create_drone_nav_env(args: argparse.Namespace) -> DroneNavEnv:
    """Create a drone navigation environment (single or multi-agent)."""
    cfg = DroneNavConfig(
        max_steps=args.max_steps,
        dt=0.01,
        seed=args.seed,
        num_agents=args.num_agents,
        num_obstacles=args.num_obstacles,
        obstacle_radius=0.5,
        goal_threshold=0.3,
        max_velocity=2.0,
        collision_radius=0.3,
    )
    return DroneNavEnv(cfg)


def _create_manipulation_env(args: argparse.Namespace) -> ManipulationEnv:
    """Create a manipulation environment for the specified task."""
    task_map = {
        "reach": ManipulationTask.REACH,
        "pick": ManipulationTask.PICK,
        "push": ManipulationTask.PUSH,
        "stack": ManipulationTask.STACK,
    }
    task = task_map[args.task]
    num_objects = 2 if task == ManipulationTask.STACK else 1
    cfg = ManipulationConfig(
        task=task,
        max_steps=args.max_steps,
        dt=0.01,
        seed=args.seed,
        num_objects=num_objects,
    )
    return ManipulationEnv(cfg)


ENV_FACTORY = {
    "grasping": _create_grasping_env,
    "drone_nav": _create_drone_nav_env,
    "manipulation": _create_manipulation_env,
}

# ---------------------------------------------------------------------------
# PPO configurations tuned per domain
# ---------------------------------------------------------------------------

_PPO_DEFAULTS: Dict[str, Dict] = {
    "grasping": dict(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=4,
        batch_size=64,
        num_steps=2048,
    ),
    "drone_nav": dict(
        learning_rate=3e-4,
        gamma=0.995,
        gae_lambda=0.98,
        clip_ratio=0.2,
        entropy_coef=0.005,
        value_coef=0.5,
        epochs=4,
        batch_size=128,
        num_steps=2048,
    ),
    "manipulation": dict(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        epochs=4,
        batch_size=64,
        num_steps=2048,
    ),
}


def _policy_config(env_name: str, obs_dim: int, act_dim: int) -> PolicyNetworkConfig:
    """Return a PolicyNetworkConfig tuned for the given domain."""
    hidden = 256 if env_name == "drone_nav" else 128
    return PolicyNetworkConfig(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=hidden,
        num_hidden_layers=3,
        activation="relu",
        log_std_init=0.0,
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def collect_rollout(env, policy: PolicyNetwork, num_steps: int):
    """
    Collect a rollout of *num_steps* transitions.

    Returns numpy arrays (states, actions, rewards, values, dones) plus
    the value estimate for the state following the last transition.
    """
    states_list: List[np.ndarray] = []
    actions_list: List[np.ndarray] = []
    rewards_list: List[float] = []
    values_list: List[float] = []
    dones_list: List[float] = []

    obs = env.reset()

    for _ in range(num_steps):
        action, value = policy.get_action(obs, deterministic=False)
        result = env.step(action)

        states_list.append(obs)
        actions_list.append(action)
        rewards_list.append(result.reward)
        values_list.append(value)
        dones_list.append(float(result.done or result.truncated))

        obs = result.observation
        if result.done or result.truncated:
            obs = env.reset()

    # Bootstrap value for last state
    _, next_value = policy.get_action(obs, deterministic=False)

    return (
        np.array(states_list, dtype=np.float32),
        np.array(actions_list, dtype=np.float32),
        np.array(rewards_list, dtype=np.float32),
        np.array(values_list, dtype=np.float32),
        np.array(dones_list, dtype=np.float32),
        float(next_value),
    )


def evaluate_policy(env, policy: PolicyNetwork, num_episodes: int = 10) -> Dict:
    """Run deterministic evaluation episodes and return aggregate metrics."""
    ep_rewards: List[float] = []
    ep_successes: List[bool] = []
    ep_lengths: List[int] = []

    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False

        while not done:
            action, _ = policy.get_action(obs, deterministic=True)
            result = env.step(action)
            total_reward += result.reward
            steps += 1
            done = result.done or result.truncated
            if result.info.get("success", False):
                success = True
            obs = result.observation

        ep_rewards.append(total_reward)
        ep_successes.append(success)
        ep_lengths.append(steps)

    return {
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "success_rate": float(np.mean(ep_successes)),
        "mean_length": float(np.mean(ep_lengths)),
    }


def train(args: argparse.Namespace) -> str:
    """
    Main training entry point.

    Returns the path to the saved checkpoint.
    """
    # Reproducibility
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Create environment
    env = ENV_FACTORY[args.env](args)

    # Policy & PPO config
    pol_cfg = _policy_config(args.env, env.observation_dim, env.action_dim)
    policy = PolicyNetwork(pol_cfg)

    ppo_kwargs = dict(_PPO_DEFAULTS[args.env])
    if args.lr is not None:
        ppo_kwargs["learning_rate"] = args.lr
    ppo_cfg = PPOConfig(**ppo_kwargs)

    trainer = PPOTrainer(policy, ppo_cfg)

    # Output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history: List[Dict] = []
    best_reward = -np.inf

    print("=" * 70)
    print(f"Training: {args.env}")
    print(f"  episodes:   {args.episodes}")
    print(f"  max_steps:  {args.max_steps}")
    print(f"  seed:       {args.seed}")
    if args.env == "manipulation":
        print(f"  task:       {args.task}")
    if args.env == "drone_nav":
        print(f"  num_agents: {args.num_agents}")
    print("=" * 70)

    for ep in range(1, args.episodes + 1):
        # Collect rollout
        states, actions, rewards, values, dones, next_val = collect_rollout(
            env, policy, ppo_cfg.num_steps
        )

        # PPO update
        stats = trainer.update(states, actions, rewards, values, dones, next_val)

        # Periodic evaluation
        if ep % max(1, args.eval_interval) == 0:
            metrics = evaluate_policy(env, policy, num_episodes=5)
            metrics["episode"] = ep
            metrics.update(stats)
            history.append(metrics)

            mr = metrics["mean_reward"]
            sr = metrics["success_rate"]
            print(f"[Ep {ep:>4d}]  reward={mr:+.2f}  success={sr:.0%}  "
                  f"policy_loss={stats['policy_loss']:.4f}")

            # Save best
            if mr > best_reward:
                best_reward = mr
                best_path = str(out_dir / "best.pth")
                trainer.save_checkpoint(best_path)

    # Always save latest
    latest_path = str(out_dir / "latest.pth")
    trainer.save_checkpoint(latest_path)

    # Save training history
    history_path = out_dir / "training_history.json"
    with open(history_path, "w") as fh:
        json.dump(history, fh, indent=2)

    # Save run config for reproducibility
    config_path = out_dir / "train_config.json"
    with open(config_path, "w") as fh:
        json.dump(
            {
                "env": args.env,
                "task": getattr(args, "task", None),
                "seed": args.seed,
                "episodes": args.episodes,
                "max_steps": args.max_steps,
                "dual_arm": getattr(args, "dual_arm", False),
                "num_agents": getattr(args, "num_agents", 1),
                "policy_config": asdict(pol_cfg),
                "ppo_config": asdict(ppo_cfg),
            },
            fh,
            indent=2,
        )

    env.close()

    print("=" * 70)
    print(f"Training complete.  Best reward: {best_reward:+.2f}")
    print(f"  checkpoint: {latest_path}")
    print(f"  history:    {history_path}")
    print("=" * 70)

    return latest_path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="hybrid-gcs-train",
        description="Train Hybrid-GCS policies for robotics tasks.",
    )

    parser.add_argument(
        "--env",
        choices=["grasping", "drone_nav", "manipulation"],
        required=True,
        help="Environment / application domain.",
    )
    parser.add_argument(
        "--task",
        choices=["reach", "pick", "push", "stack"],
        default="reach",
        help="Manipulation task (only used when --env manipulation).",
    )
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument(
        "--eval-interval", type=int, default=10, help="Evaluate every N episodes."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/train",
        help="Directory for checkpoints and logs.",
    )

    # Grasping-specific
    parser.add_argument(
        "--dual-arm", action="store_true", help="Use dual-arm workspace (grasping only)."
    )

    # Drone-specific
    parser.add_argument(
        "--num-agents", type=int, default=1, help="Number of drone agents (drone_nav only)."
    )
    parser.add_argument(
        "--num-obstacles", type=int, default=5, help="Number of obstacles (drone_nav only)."
    )

    return parser


def main(argv: list = None) -> str:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":
    main()
