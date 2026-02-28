"""
Visualization Script for Hybrid-GCS.

Replays a trained policy in either PyBullet (3-D physics renderer) or
Foxglove Studio (MCAP recording) for all three application domains.

Usage:
    hybrid-gcs-vis --env grasping --checkpoint best.pth --backend pybullet
    hybrid-gcs-vis --env drone_nav --checkpoint best.pth --backend foxglove
    hybrid-gcs-vis --env manipulation --task stack --checkpoint best.pth --backend foxglove
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
from hybrid_gcs.training import PolicyNetwork, PolicyNetworkConfig

# Re-use the evaluation loader
from hybrid_gcs.cli.evaluate import ENV_FACTORY, load_policy


# ---------------------------------------------------------------------------
# Episode recording (backend-agnostic)
# ---------------------------------------------------------------------------


def _record_episode(env, policy: PolicyNetwork) -> Dict:
    """Run one deterministic episode and return per-step state data."""
    obs = env.reset()
    done = False
    frames: List[Dict] = []
    step = 0

    while not done:
        action, _ = policy.get_action(obs, deterministic=True)
        result = env.step(action)

        frames.append(
            {
                "step": step,
                "observation": obs.tolist(),
                "action": action.tolist(),
                "reward": result.reward,
                "info": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in result.info.items()
                },
            }
        )

        obs = result.observation
        step += 1
        done = result.done or result.truncated

    return {"frames": frames, "num_steps": step}


# ---------------------------------------------------------------------------
# PyBullet backend
# ---------------------------------------------------------------------------


def _visualize_pybullet(env, policy: PolicyNetwork, args: argparse.Namespace) -> str:
    """
    Render an episode using PyBullet's built-in GUI or off-screen renderer.

    When PyBullet is not installed the function falls back to writing a
    JSON trajectory file that can be loaded by external tools.
    """
    episode = _record_episode(env, policy)

    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        # Graceful fallback: save trajectory JSON
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(episode, fh, indent=2)
        print(f"PyBullet not installed. Trajectory saved to {out}")
        return str(out)

    mode = p.GUI if not args.offscreen else p.DIRECT
    physics_client = p.connect(mode)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Ground plane
    p.loadURDF("plane.urdf")

    # Determine scene objects based on domain
    markers: Dict[str, int] = {}
    if args.env == "grasping":
        table_id = p.loadURDF(
            "table/table.urdf", basePosition=[0.55, 0.0, 0.0], useFixedBase=True
        )
        markers["ee"] = p.loadURDF(
            "sphere2.urdf", globalScaling=0.03, basePosition=[0.55, 0.0, 0.6]
        )
        markers["object"] = p.loadURDF(
            "cube_small.urdf", basePosition=[0.55, 0.0, 0.02]
        )
    elif args.env == "drone_nav":
        for i in range(getattr(args, "num_agents", 1)):
            markers[f"drone_{i}"] = p.loadURDF(
                "sphere2.urdf", globalScaling=0.1, basePosition=[0, 0, 1]
            )
    else:  # manipulation
        p.loadURDF("table/table.urdf", basePosition=[0.55, 0.0, 0.0], useFixedBase=True)
        markers["ee"] = p.loadURDF(
            "sphere2.urdf", globalScaling=0.03, basePosition=[0.55, 0.0, 0.6]
        )
        markers["object"] = p.loadURDF(
            "cube_small.urdf", basePosition=[0.55, 0.0, 0.02]
        )

    # Replay frames
    for frame in episode["frames"]:
        obs = np.array(frame["observation"])
        if args.env == "grasping":
            ee_pos = obs[:3].tolist()
            obj_pos = obs[6:9].tolist()
            p.resetBasePositionAndOrientation(
                markers["ee"], ee_pos, [0, 0, 0, 1]
            )
            p.resetBasePositionAndOrientation(
                markers["object"], obj_pos, [0, 0, 0, 1]
            )
        elif args.env == "drone_nav":
            n_agents = getattr(args, "num_agents", 1)
            per_agent = len(obs) // n_agents
            for i in range(n_agents):
                pos = obs[i * per_agent : i * per_agent + 3].tolist()
                key = f"drone_{i}"
                if key in markers:
                    p.resetBasePositionAndOrientation(
                        markers[key], pos, [0, 0, 0, 1]
                    )
        else:  # manipulation
            ee_pos = obs[:3].tolist()
            obj_pos = obs[7:10].tolist()
            p.resetBasePositionAndOrientation(
                markers["ee"], ee_pos, [0, 0, 0, 1]
            )
            p.resetBasePositionAndOrientation(
                markers["object"], obj_pos, [0, 0, 0, 1]
            )

        p.stepSimulation()
        import time

        time.sleep(env.config.dt)

    p.disconnect()

    # Also save trajectory JSON
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(episode, fh, indent=2)
    print(f"PyBullet visualization complete.  Trajectory: {out}")
    return str(out)


# ---------------------------------------------------------------------------
# Foxglove backend
# ---------------------------------------------------------------------------


def _visualize_foxglove(env, policy: PolicyNetwork, args: argparse.Namespace) -> str:
    """
    Record an episode to an MCAP file for Foxglove Studio.

    Uses the existing FoxgloveRecorder to write 3-D scene updates that
    can be opened directly in Foxglove Studio.
    """
    from hybrid_gcs.visualization.foxglove_recorder import (
        FoxgloveRecorder,
        _make_color,
        _make_pose,
        _make_timestamp,
        _make_vector3,
    )

    episode = _record_episode(env, policy)

    out_path = Path(args.output)
    if out_path.suffix != ".mcap":
        out_path = out_path.with_suffix(".mcap")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with FoxgloveRecorder(str(out_path)) as recorder:
        recorder.add_frame_transform("world", "base_link")

        # Write per-step scene updates
        for frame in episode["frames"]:
            obs = np.array(frame["observation"])
            step = frame["step"]
            time_ns = step * 100_000_000  # 0.1 s per step

            sec = time_ns // 1_000_000_000
            nsec = time_ns % 1_000_000_000

            entities = []

            if args.env == "grasping":
                ee_pos = obs[:3]
                obj_pos = obs[6:9]
                entities.append(
                    _sphere_entity(
                        "ee", ee_pos, 0.04, (0.2, 0.6, 1.0, 0.9), sec, nsec
                    )
                )
                entities.append(
                    _sphere_entity(
                        "object", obj_pos, 0.04, (1.0, 0.6, 0.1, 0.9), sec, nsec
                    )
                )
            elif args.env == "drone_nav":
                n_agents = getattr(args, "num_agents", 1)
                per_agent = len(obs) // n_agents
                for i in range(n_agents):
                    pos = obs[i * per_agent : i * per_agent + 3]
                    goal = obs[i * per_agent + 6 : i * per_agent + 9]
                    hue = i / max(n_agents, 1)
                    color = (0.2 + 0.6 * hue, 0.8 - 0.5 * hue, 0.3, 0.9)
                    entities.append(
                        _sphere_entity(f"drone_{i}", pos, 0.15, color, sec, nsec)
                    )
                    entities.append(
                        _sphere_entity(
                            f"goal_{i}", goal, 0.1, (0.0, 1.0, 0.0, 0.4), sec, nsec
                        )
                    )
            else:  # manipulation
                ee_pos = obs[:3]
                obj_pos = obs[7:10]
                target_pos = obs[-3:]
                entities.append(
                    _sphere_entity(
                        "ee", ee_pos, 0.04, (0.2, 0.6, 1.0, 0.9), sec, nsec
                    )
                )
                entities.append(
                    _sphere_entity(
                        "object", obj_pos, 0.04, (1.0, 0.6, 0.1, 0.9), sec, nsec
                    )
                )
                entities.append(
                    _sphere_entity(
                        "target", target_pos, 0.04, (0.0, 1.0, 0.0, 0.5), sec, nsec
                    )
                )

            if entities:
                from hybrid_gcs.visualization.foxglove_recorder import _SCENE_UPDATE_SCHEMA

                channel_id = recorder._get_channel(
                    "/scene", "foxglove.SceneUpdate", _SCENE_UPDATE_SCHEMA
                )
                msg = {"deletions": [], "entities": entities}
                recorder._write_json(channel_id, msg, time_ns=time_ns)

    print(f"Foxglove MCAP written to {out_path}")
    print("Open in Foxglove Studio: https://studio.foxglove.dev")
    return str(out_path)


def _sphere_entity(
    entity_id: str,
    position: np.ndarray,
    size: float,
    color: tuple,
    sec: int,
    nsec: int,
) -> Dict:
    """Build a Foxglove SceneEntity with a single sphere."""
    from hybrid_gcs.visualization.foxglove_recorder import (
        _make_color,
        _make_pose,
        _make_timestamp,
        _make_vector3,
    )

    pos = [float(position[i]) if i < len(position) else 0.0 for i in range(3)]
    return {
        "timestamp": _make_timestamp(sec, nsec),
        "frame_id": "world",
        "id": entity_id,
        "lifetime": {"sec": 0, "nsec": 0},
        "frame_locked": True,
        "metadata": [],
        "arrows": [],
        "cubes": [],
        "spheres": [
            {
                "pose": _make_pose(pos[0], pos[1], pos[2]),
                "size": _make_vector3(size, size, size),
                "color": _make_color(*color),
            }
        ],
        "cylinders": [],
        "lines": [],
        "triangles": [],
        "texts": [],
        "models": [],
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

BACKENDS = {
    "pybullet": _visualize_pybullet,
    "foxglove": _visualize_foxglove,
}


def visualize(args: argparse.Namespace) -> str:
    """Main visualization entry point.  Returns output path."""
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    env = ENV_FACTORY[args.env](args)
    policy = load_policy(args.checkpoint, env)

    print("=" * 70)
    print(f"Visualizing: {args.env}  (backend={args.backend})")
    print(f"  checkpoint: {args.checkpoint}")
    print("=" * 70)

    result = BACKENDS[args.backend](env, policy, args)
    env.close()
    return result


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hybrid-gcs-vis",
        description="Visualize trained Hybrid-GCS policies in PyBullet or Foxglove Studio.",
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
        help="Manipulation task (only for --env manipulation).",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pth)."
    )
    parser.add_argument(
        "--backend",
        choices=["pybullet", "foxglove"],
        default="foxglove",
        help="Visualization backend.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/visualization.mcap",
        help="Output file path (.mcap for foxglove, .json for pybullet fallback).",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--offscreen",
        action="store_true",
        help="PyBullet off-screen rendering (no GUI window).",
    )

    # Domain-specific
    parser.add_argument("--dual-arm", action="store_true", help="Dual-arm (grasping).")
    parser.add_argument("--num-agents", type=int, default=1, help="Agents (drone_nav).")
    parser.add_argument("--num-obstacles", type=int, default=5, help="Obstacles (drone_nav).")

    return parser


def main(argv: list = None) -> str:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return visualize(args)


if __name__ == "__main__":
    main()
