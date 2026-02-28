"""
PyBullet Simulation Renderer for Hybrid-GCS.

Provides a lightweight wrapper that replays episode trajectories inside a
PyBullet physics simulation with 3-D rendering.  When PyBullet is not
installed the renderer writes a JSON trajectory file instead.

Usage:
    from hybrid_gcs.visualization.pybullet_renderer import PyBulletRenderer

    renderer = PyBulletRenderer(mode="gui")
    renderer.setup_scene("grasping")
    renderer.replay(episode_frames, dt=0.01)
    renderer.close()
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class PyBulletRenderer:
    """
    Replay recorded episodes inside a PyBullet physics scene.

    Attributes:
        mode: ``"gui"`` for on-screen rendering, ``"direct"`` for headless.
        physics_client: PyBullet client ID (``-1`` when not connected).
        markers: Mapping from marker name to PyBullet body ID.
    """

    def __init__(self, mode: str = "direct"):
        """
        Initialize the renderer.

        Args:
            mode: ``"gui"`` or ``"direct"`` (headless).
        """
        self.mode = mode
        self.physics_client: int = -1
        self.markers: Dict[str, int] = {}
        self._pybullet: Optional[Any] = None

        try:
            import pybullet as p
            import pybullet_data

            self._pybullet = p
            self._pybullet_data = pybullet_data
        except ImportError:
            self._pybullet = None

    @property
    def available(self) -> bool:
        """Return ``True`` if PyBullet is importable."""
        return self._pybullet is not None

    # ------------------------------------------------------------------
    # Scene setup
    # ------------------------------------------------------------------

    def connect(self) -> int:
        """Connect to the PyBullet server.  Returns the client ID."""
        if not self.available:
            return -1

        p = self._pybullet
        pb_mode = p.GUI if self.mode == "gui" else p.DIRECT
        self.physics_client = p.connect(pb_mode)
        p.setAdditionalSearchPath(self._pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        return self.physics_client

    def setup_scene(self, env_name: str, num_agents: int = 1) -> None:
        """
        Populate the scene with domain-appropriate objects.

        Args:
            env_name: One of ``"grasping"``, ``"drone_nav"``, ``"manipulation"``.
            num_agents: Number of drone agents (used only for ``drone_nav``).
        """
        if not self.available:
            return
        if self.physics_client < 0:
            self.connect()

        p = self._pybullet

        if env_name in ("grasping", "manipulation"):
            p.loadURDF("table/table.urdf", basePosition=[0.55, 0, 0], useFixedBase=True)
            self.markers["ee"] = p.loadURDF(
                "sphere2.urdf", globalScaling=0.03, basePosition=[0.55, 0, 0.6]
            )
            self.markers["object"] = p.loadURDF(
                "cube_small.urdf", basePosition=[0.55, 0, 0.02]
            )
        elif env_name == "drone_nav":
            for i in range(num_agents):
                self.markers[f"drone_{i}"] = p.loadURDF(
                    "sphere2.urdf", globalScaling=0.1, basePosition=[0, 0, 1]
                )

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def replay(
        self,
        frames: List[Dict],
        env_name: str,
        dt: float = 0.01,
        num_agents: int = 1,
    ) -> None:
        """
        Step through recorded frames, updating marker positions.

        Args:
            frames: List of per-step dicts with ``"observation"`` key.
            env_name: Domain name.
            dt: Time to sleep between frames.
            num_agents: Number of drone agents (``drone_nav`` only).
        """
        if not self.available:
            return

        import time

        p = self._pybullet

        for frame in frames:
            obs = np.array(frame["observation"])
            self._update_markers(p, obs, env_name, num_agents)
            p.stepSimulation()
            time.sleep(dt)

    def _update_markers(self, p, obs: np.ndarray, env_name: str, num_agents: int) -> None:
        """Move PyBullet markers to match the observation."""
        quat = [0, 0, 0, 1]

        if env_name == "grasping":
            if "ee" in self.markers:
                p.resetBasePositionAndOrientation(self.markers["ee"], obs[:3].tolist(), quat)
            if "object" in self.markers:
                p.resetBasePositionAndOrientation(self.markers["object"], obs[6:9].tolist(), quat)
        elif env_name == "drone_nav":
            per = len(obs) // num_agents
            for i in range(num_agents):
                key = f"drone_{i}"
                if key in self.markers:
                    pos = obs[i * per : i * per + 3].tolist()
                    p.resetBasePositionAndOrientation(self.markers[key], pos, quat)
        elif env_name == "manipulation":
            if "ee" in self.markers:
                p.resetBasePositionAndOrientation(self.markers["ee"], obs[:3].tolist(), quat)
            if "object" in self.markers:
                p.resetBasePositionAndOrientation(self.markers["object"], obs[7:10].tolist(), quat)

    # ------------------------------------------------------------------
    # Cleanup & fallback
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Disconnect from the PyBullet server."""
        if self.available and self.physics_client >= 0:
            self._pybullet.disconnect()
            self.physics_client = -1

    @staticmethod
    def save_trajectory_json(frames: List[Dict], path: str) -> str:
        """
        Fallback: persist frames as JSON when PyBullet is unavailable.

        Args:
            frames: Episode frames.
            path: Output file path.

        Returns:
            Resolved output path.
        """
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            json.dump(frames, fh, indent=2)
        return str(out.resolve())

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
