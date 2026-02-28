"""
Evaluation Script for Hybrid-GCS.

Loads trained PPO checkpoints and evaluates policies across all three
application domains with detailed metrics and optional episode recording.

Usage:
    hybrid-gcs-eval --env grasping --checkpoint checkpoints/train/best.pth
    hybrid-gcs-eval --env drone_nav --checkpoint checkpoints/train/best.pth --episodes 50
    hybrid-gcs-eval --env manipulation --task stack --checkpoint checkpoints/train/best.pth
"""

import argparse
import json
import sys
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
# Environment factories (mirrors train.py)
# ---------------------------------------------------------------------------


def _create_grasping_env(args: argparse.Namespace) -> GraspingEnv:
    cfg = GraspingConfig(max_steps=args.max_steps, dt=0.01, seed=args.seed)
    if getattr(args, "dual_arm", False):
        ws_lo = np.array([0.2, -0.5, 0.0], dtype=np.float64)
        ws_hi = np.array([0.9, 0.5, 0.6], dtype=np.float64)
        cfg.workspace_bounds = (ws_lo, ws_hi)
    return GraspingEnv(cfg)


def _create_drone_nav_env(args: argparse.Namespace) -> DroneNavEnv:
    cfg = DroneNavConfig(
        max_steps=args.max_steps,
        dt=0.01,
        seed=args.seed,
        num_agents=getattr(args, "num_agents", 1),
        num_obstacles=getattr(args, "num_obstacles", 5),
    )
    return DroneNavEnv(cfg)


def _create_manipulation_env(args: argparse.Namespace) -> ManipulationEnv:
    task_map = {
        "reach": ManipulationTask.REACH,
        "pick": ManipulationTask.PICK,
        "push": ManipulationTask.PUSH,
        "stack": ManipulationTask.STACK,
    }
    task = task_map[args.task]
    num_objects = 2 if task == ManipulationTask.STACK else 1
    cfg = ManipulationConfig(
        task=task, max_steps=args.max_steps, dt=0.01, seed=args.seed, num_objects=num_objects
    )
    return ManipulationEnv(cfg)


ENV_FACTORY = {
    "grasping": _create_grasping_env,
    "drone_nav": _create_drone_nav_env,
    "manipulation": _create_manipulation_env,
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_policy(checkpoint_path: str, env) -> PolicyNetwork:
    """
    Load a trained policy from a PPOTrainer checkpoint.

    The checkpoint contains ``policy_config`` so the correct architecture
    is rebuilt automatically.  Falls back to inferring the config from
    the environment when the key is missing.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "policy_config" in ckpt:
        pol_cfg = PolicyNetworkConfig(**ckpt["policy_config"])
    else:
        pol_cfg = PolicyNetworkConfig(
            state_dim=env.observation_dim,
            action_dim=env.action_dim,
            hidden_dim=128,
            num_hidden_layers=3,
        )

    policy = PolicyNetwork(pol_cfg)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def run_evaluation(
    env,
    policy: PolicyNetwork,
    num_episodes: int,
    record_episodes: bool = False,
) -> Dict:
    """
    Run *num_episodes* deterministic episodes.

    Returns aggregate metrics and optionally per-step episode recordings
    for later visualization.
    """
    ep_rewards: List[float] = []
    ep_successes: List[bool] = []
    ep_lengths: List[int] = []
    recorded: List[Dict] = []

    for ep_idx in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        success = False
        episode_data: List[Dict] = []

        while not done:
            action, value = policy.get_action(obs, deterministic=True)
            result = env.step(action)

            if record_episodes:
                episode_data.append(
                    {
                        "step": steps,
                        "observation": obs.tolist(),
                        "action": action.tolist(),
                        "reward": result.reward,
                        "done": result.done,
                        "info": {
                            k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in result.info.items()
                        },
                    }
                )

            total_reward += result.reward
            steps += 1
            done = result.done or result.truncated
            if result.info.get("success", False):
                success = True
            obs = result.observation

        ep_rewards.append(total_reward)
        ep_successes.append(success)
        ep_lengths.append(steps)

        if record_episodes:
            recorded.append({"episode": ep_idx, "steps": episode_data})

    metrics = {
        "num_episodes": num_episodes,
        "mean_reward": float(np.mean(ep_rewards)),
        "std_reward": float(np.std(ep_rewards)),
        "min_reward": float(np.min(ep_rewards)),
        "max_reward": float(np.max(ep_rewards)),
        "success_rate": float(np.mean(ep_successes)),
        "mean_length": float(np.mean(ep_lengths)),
    }

    if record_episodes:
        metrics["episodes"] = recorded

    return metrics


def evaluate(args: argparse.Namespace) -> Dict:
    """Main evaluation entry point."""
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env = ENV_FACTORY[args.env](args)
    policy = load_policy(args.checkpoint, env)

    print("=" * 70)
    print(f"Evaluating: {args.env}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  episodes:   {args.episodes}")
    if args.env == "manipulation":
        print(f"  task:       {args.task}")
    print("=" * 70)

    metrics = run_evaluation(
        env, policy, args.episodes, record_episodes=args.record
    )

    env.close()

    # Print summary
    print(f"\nResults ({args.episodes} episodes):")
    print(f"  Mean reward:   {metrics['mean_reward']:+.2f} ± {metrics['std_reward']:.2f}")
    print(f"  Min / Max:     {metrics['min_reward']:+.2f} / {metrics['max_reward']:+.2f}")
    print(f"  Success rate:  {metrics['success_rate']:.0%}")
    print(f"  Mean length:   {metrics['mean_length']:.1f}")

    # Optionally save results
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Strip episode recordings from the file if they exist (can be large)
        save_metrics = {k: v for k, v in metrics.items() if k != "episodes"}
        with open(out_path, "w") as fh:
            json.dump(save_metrics, fh, indent=2)
        print(f"\nMetrics saved to {out_path}")

    return metrics


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-gcs-eval",
        description="Evaluate trained Hybrid-GCS policies.",
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
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)."
    )
    parser.add_argument("--episodes", type=int, default=20, help="Number of evaluation episodes.")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output", type=str, default=None, help="Save metrics JSON to this path."
    )
    parser.add_argument(
        "--record", action="store_true", help="Record per-step episode data."
    )

    # Domain-specific
    parser.add_argument("--dual-arm", action="store_true", help="Dual-arm (grasping).")
    parser.add_argument("--num-agents", type=int, default=1, help="Agents (drone_nav).")
    parser.add_argument("--num-obstacles", type=int, default=5, help="Obstacles (drone_nav).")

    return parser


def main(argv: list = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    main()
