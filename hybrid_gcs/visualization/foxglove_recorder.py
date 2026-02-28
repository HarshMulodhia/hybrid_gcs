"""
Foxglove Studio MCAP Recorder for Hybrid-GCS.

Records robot scenes, trajectories, obstacles, and convex regions to
MCAP files for visualization in Foxglove Studio. Uses local URDF models
with primitive geometries for optimized rendering (no external URLs).

Usage:
    from hybrid_gcs.visualization import FoxgloveRecorder

    recorder = FoxgloveRecorder("output.mcap")
    recorder.add_robot_description("data/models/ur5e/ur5e.urdf")
    recorder.add_trajectory(trajectory, frame_id="world")
    recorder.add_obstacles(obstacles, frame_id="world")
    recorder.close()

Then open output.mcap in Foxglove Studio.
"""

import json
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np

from mcap.writer import Writer


# --- Foxglove JSON Schema Definitions ---

_SCENE_UPDATE_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "deletions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "object"},
                    "type": {"type": "integer"},
                    "id": {"type": "string"}
                }
            }
        },
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "timestamp": {"type": "object"},
                    "frame_id": {"type": "string"},
                    "id": {"type": "string"},
                    "lifetime": {"type": "object"},
                    "frame_locked": {"type": "boolean"},
                    "metadata": {"type": "array"},
                    "arrows": {"type": "array"},
                    "cubes": {"type": "array"},
                    "spheres": {"type": "array"},
                    "cylinders": {"type": "array"},
                    "lines": {"type": "array"},
                    "triangles": {"type": "array"},
                    "texts": {"type": "array"},
                    "models": {"type": "array"}
                }
            }
        }
    }
})

_FRAME_TRANSFORM_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "timestamp": {"type": "object"},
        "parent_frame_id": {"type": "string"},
        "child_frame_id": {"type": "string"},
        "translation": {"type": "object"},
        "rotation": {"type": "object"}
    }
})

_ROBOT_DESCRIPTION_SCHEMA = json.dumps({
    "type": "object",
    "properties": {
        "model_data": {"type": "string"},
        "model_encoding": {"type": "string"}
    }
})


def _make_timestamp(sec: int = 0, nsec: int = 0) -> Dict:
    """Create Foxglove timestamp."""
    return {"sec": sec, "nsec": nsec}


def _make_color(r: float, g: float, b: float, a: float = 1.0) -> Dict:
    """Create Foxglove color."""
    return {"r": r, "g": g, "b": b, "a": a}


def _make_pose(
    x: float = 0, y: float = 0, z: float = 0,
    qx: float = 0, qy: float = 0, qz: float = 0, qw: float = 1
) -> Dict:
    """Create Foxglove pose (position + orientation)."""
    return {
        "position": {"x": x, "y": y, "z": z},
        "orientation": {"x": qx, "y": qy, "z": qz, "w": qw}
    }


def _make_vector3(x: float, y: float, z: float) -> Dict:
    """Create Foxglove Vector3."""
    return {"x": x, "y": y, "z": z}


class FoxgloveRecorder:
    """
    Records Hybrid-GCS scenes to MCAP files for Foxglove Studio.

    Creates MCAP files containing robot descriptions, scene entities
    (trajectories, obstacles, regions), and frame transforms. The
    output can be opened directly in Foxglove Studio for interactive
    3D visualization.

    Uses local URDF models with primitive geometries (cylinder, box)
    for fast rendering without external mesh dependencies.

    Attributes:
        output_path: Path to output MCAP file
        writer: MCAP writer instance
        channels: Registered MCAP channels

    Example:
        >>> recorder = FoxgloveRecorder("scene.mcap")
        >>> recorder.add_robot_description("data/models/ur5e/ur5e.urdf")
        >>> recorder.add_trajectory(traj)
        >>> recorder.add_obstacles(obstacles)
        >>> recorder.close()
    """

    def __init__(self, output_path: str):
        """
        Initialize MCAP recorder.

        Args:
            output_path: Path to output .mcap file
        """
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self._file = open(self.output_path, "wb")
        self.writer = Writer(self._file)
        self.writer.start()

        self._channels: Dict[str, int] = {}
        self._schema_ids: Dict[str, int] = {}
        self._time_ns = 0
        self._closed = False

    def _register_schema(self, name: str, schema_data: str) -> int:
        """Register a JSON schema and return its ID."""
        if name in self._schema_ids:
            return self._schema_ids[name]

        schema_id = self.writer.register_schema(
            name=name,
            encoding="jsonschema",
            data=schema_data.encode("utf-8"),
        )
        self._schema_ids[name] = schema_id
        return schema_id

    def _get_channel(self, topic: str, schema_name: str, schema_data: str) -> int:
        """Get or create a channel."""
        if topic in self._channels:
            return self._channels[topic]

        schema_id = self._register_schema(schema_name, schema_data)
        channel_id = self.writer.register_channel(
            topic=topic,
            message_encoding="json",
            schema_id=schema_id,
        )
        self._channels[topic] = channel_id
        return channel_id

    def _advance_time(self, delta_ns: int = 100_000_000) -> int:
        """Advance internal clock and return current time in nanoseconds."""
        self._time_ns += delta_ns
        return self._time_ns

    def _write_json(self, channel_id: int, data: Dict, time_ns: Optional[int] = None):
        """Write a JSON message to the MCAP file."""
        if time_ns is None:
            time_ns = self._time_ns
        payload = json.dumps(data).encode("utf-8")
        self.writer.add_message(
            channel_id=channel_id,
            log_time=time_ns,
            data=payload,
            publish_time=time_ns,
        )

    def add_robot_description(self, urdf_path: str, topic: str = "/robot_description"):
        """
        Add robot URDF description for Foxglove.

        Reads a local URDF file and publishes it so Foxglove can
        render the robot model.

        Args:
            urdf_path: Path to URDF file (relative or absolute)
            topic: ROS topic name for the description

        Raises:
            FileNotFoundError: If URDF file does not exist
        """
        urdf_file = Path(urdf_path)
        if not urdf_file.is_absolute():
            # Try relative to package data dir
            pkg_root = Path(__file__).parent.parent.parent
            urdf_file = pkg_root / urdf_path
        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")

        urdf_content = urdf_file.read_text(encoding="utf-8")

        channel_id = self._get_channel(
            topic, "foxglove.SceneUpdate", _SCENE_UPDATE_SCHEMA
        )

        msg = {
            "deletions": [],
            "entities": [{
                "timestamp": _make_timestamp(0, 0),
                "frame_id": "world",
                "id": "robot_description",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": True,
                "metadata": [
                    {"key": "urdf", "value": urdf_content},
                    {"key": "model_encoding", "value": "urdf"},
                ],
                "arrows": [],
                "cubes": [],
                "spheres": [],
                "cylinders": [],
                "lines": [],
                "triangles": [],
                "texts": [
                    {
                        "pose": _make_pose(0, 0, 1.2),
                        "billboard": True,
                        "font_size": 14.0,
                        "scale_invariant": True,
                        "color": _make_color(1.0, 1.0, 1.0),
                        "text": "UR5e Robot",
                    }
                ],
                "models": [],
            }],
        }

        self._write_json(channel_id, msg, time_ns=0)

    def add_environment(self, urdf_path: str, topic: str = "/environment"):
        """
        Add environment URDF (table, room) for Foxglove.

        Args:
            urdf_path: Path to environment URDF file
            topic: ROS topic name
        """
        urdf_file = Path(urdf_path)
        if not urdf_file.is_absolute():
            pkg_root = Path(__file__).parent.parent.parent
            urdf_file = pkg_root / urdf_path
        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")

        urdf_content = urdf_file.read_text(encoding="utf-8")

        channel_id = self._get_channel(
            topic, "foxglove.SceneUpdate", _SCENE_UPDATE_SCHEMA
        )

        msg = {
            "deletions": [],
            "entities": [{
                "timestamp": _make_timestamp(0, 0),
                "frame_id": "world",
                "id": "environment",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": True,
                "metadata": [
                    {"key": "urdf", "value": urdf_content},
                    {"key": "model_encoding", "value": "urdf"},
                ],
                "arrows": [],
                "cubes": [],
                "spheres": [],
                "cylinders": [],
                "lines": [],
                "triangles": [],
                "texts": [],
                "models": [],
            }],
        }

        self._write_json(channel_id, msg, time_ns=0)

    def add_trajectory(
        self,
        trajectory,
        frame_id: str = "world",
        topic: str = "/trajectory",
        color: Tuple[float, float, float, float] = (0.0, 0.8, 0.2, 1.0),
        line_width: float = 0.01,
        n_samples: int = 50,
    ):
        """
        Add a trajectory visualization as a line strip.

        Args:
            trajectory: Trajectory object with at_time(t) method
            frame_id: Coordinate frame
            topic: ROS topic name
            color: RGBA color tuple
            line_width: Width of trajectory line
            n_samples: Number of samples along trajectory
        """
        channel_id = self._get_channel(
            topic, "foxglove.SceneUpdate", _SCENE_UPDATE_SCHEMA
        )

        # Sample trajectory points
        t_start = trajectory.timestamps[0]
        t_end = trajectory.timestamps[-1]
        times = np.linspace(t_start, t_end, n_samples)
        points = [trajectory.at_time(t) for t in times]

        # Build line strip points
        line_points = []
        for pt in points:
            if len(pt) == 2:
                line_points.append(_make_vector3(float(pt[0]), float(pt[1]), 0.01))
            else:
                line_points.append(_make_vector3(
                    float(pt[0]), float(pt[1]), float(pt[2])
                ))

        # Build line colors (gradient from green to blue)
        line_colors = []
        for i in range(len(line_points)):
            t = i / max(1, len(line_points) - 1)
            line_colors.append(_make_color(
                color[0] * (1 - t) + 0.2 * t,
                color[1] * (1 - t) + 0.3 * t,
                color[2] * (1 - t) + 1.0 * t,
                color[3],
            ))

        time_ns = self._advance_time()
        sec = time_ns // 1_000_000_000
        nsec = time_ns % 1_000_000_000

        msg = {
            "deletions": [],
            "entities": [{
                "timestamp": _make_timestamp(sec, nsec),
                "frame_id": frame_id,
                "id": "trajectory",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": True,
                "metadata": [],
                "arrows": [],
                "cubes": [],
                "spheres": [],
                "cylinders": [],
                "lines": [{
                    "type": 0,  # LINE_STRIP
                    "pose": _make_pose(),
                    "thickness": line_width,
                    "scale_invariant": False,
                    "points": line_points,
                    "color": _make_color(*color),
                    "colors": line_colors,
                    "indices": [],
                }],
                "triangles": [],
                "texts": [],
                "models": [],
            }],
        }

        # Add start/goal markers
        start_pt = points[0]
        goal_pt = points[-1]

        start_pos = (
            _make_vector3(float(start_pt[0]), float(start_pt[1]),
                          float(start_pt[2]) if len(start_pt) > 2 else 0.01)
        )
        goal_pos = (
            _make_vector3(float(goal_pt[0]), float(goal_pt[1]),
                          float(goal_pt[2]) if len(goal_pt) > 2 else 0.01)
        )

        msg["entities"].append({
            "timestamp": _make_timestamp(sec, nsec),
            "frame_id": frame_id,
            "id": "start_marker",
            "lifetime": {"sec": 0, "nsec": 0},
            "frame_locked": True,
            "metadata": [],
            "arrows": [],
            "cubes": [],
            "spheres": [{
                "pose": _make_pose(start_pos["x"], start_pos["y"], start_pos["z"]),
                "size": _make_vector3(0.08, 0.08, 0.08),
                "color": _make_color(0.0, 1.0, 0.0, 0.9),
            }],
            "cylinders": [],
            "lines": [],
            "triangles": [],
            "texts": [{
                "pose": _make_pose(start_pos["x"], start_pos["y"],
                                   start_pos["z"] + 0.15),
                "billboard": True,
                "font_size": 12.0,
                "scale_invariant": True,
                "color": _make_color(0.0, 1.0, 0.0),
                "text": "START",
            }],
            "models": [],
        })

        msg["entities"].append({
            "timestamp": _make_timestamp(sec, nsec),
            "frame_id": frame_id,
            "id": "goal_marker",
            "lifetime": {"sec": 0, "nsec": 0},
            "frame_locked": True,
            "metadata": [],
            "arrows": [],
            "cubes": [],
            "spheres": [{
                "pose": _make_pose(goal_pos["x"], goal_pos["y"], goal_pos["z"]),
                "size": _make_vector3(0.08, 0.08, 0.08),
                "color": _make_color(1.0, 0.0, 0.0, 0.9),
            }],
            "cylinders": [],
            "lines": [],
            "triangles": [],
            "texts": [{
                "pose": _make_pose(goal_pos["x"], goal_pos["y"],
                                   goal_pos["z"] + 0.15),
                "billboard": True,
                "font_size": 12.0,
                "scale_invariant": True,
                "color": _make_color(1.0, 0.0, 0.0),
                "text": "GOAL",
            }],
            "models": [],
        })

        self._write_json(channel_id, msg, time_ns=time_ns)

    def add_obstacles(
        self,
        obstacles: List,
        frame_id: str = "world",
        topic: str = "/obstacles",
        color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 0.6),
    ):
        """
        Add obstacle visualizations as cubes.

        Args:
            obstacles: List of SimpleBoxObstacle objects (with .lower, .upper)
            frame_id: Coordinate frame
            topic: ROS topic name
            color: RGBA color
        """
        channel_id = self._get_channel(
            topic, "foxglove.SceneUpdate", _SCENE_UPDATE_SCHEMA
        )

        cubes = []
        for i, obs in enumerate(obstacles):
            lower = np.asarray(obs.lower, dtype=np.float64)
            upper = np.asarray(obs.upper, dtype=np.float64)
            center = (lower + upper) / 2.0
            size = upper - lower

            # Ensure 3D
            if len(center) == 2:
                cx, cy = float(center[0]), float(center[1])
                sx, sy = float(size[0]), float(size[1])
                cz, sz = 0.25, 0.5
            else:
                cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
                sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

            cubes.append({
                "pose": _make_pose(cx, cy, cz),
                "size": _make_vector3(sx, sy, sz),
                "color": _make_color(*color),
            })

        time_ns = self._advance_time()
        sec = time_ns // 1_000_000_000
        nsec = time_ns % 1_000_000_000

        msg = {
            "deletions": [],
            "entities": [{
                "timestamp": _make_timestamp(sec, nsec),
                "frame_id": frame_id,
                "id": "obstacles",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": True,
                "metadata": [],
                "arrows": [],
                "cubes": cubes,
                "spheres": [],
                "cylinders": [],
                "lines": [],
                "triangles": [],
                "texts": [],
                "models": [],
            }],
        }

        self._write_json(channel_id, msg, time_ns=time_ns)

    def add_convex_regions(
        self,
        regions: List,
        frame_id: str = "world",
        topic: str = "/regions",
        base_color: Tuple[float, float, float, float] = (0.2, 0.4, 0.8, 0.15),
    ):
        """
        Add convex region visualizations as semi-transparent spheres.

        Represents IRIS ellipsoidal regions as spheres scaled by
        the ellipsoid axes for approximate visualization.

        Args:
            regions: List of Ellipsoid objects (with .center, .shape_matrix)
            frame_id: Coordinate frame
            topic: ROS topic name
            base_color: RGBA base color (varied per region)
        """
        channel_id = self._get_channel(
            topic, "foxglove.SceneUpdate", _SCENE_UPDATE_SCHEMA
        )

        spheres = []
        for i, region in enumerate(regions):
            center = np.asarray(region.center, dtype=np.float64)

            # Approximate ellipsoid as sphere with average radius
            Q = np.asarray(region.shape_matrix, dtype=np.float64)
            try:
                eigvals = np.linalg.eigvalsh(Q)
                radii = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))
            except np.linalg.LinAlgError:
                radii = np.ones(len(center))

            avg_radius = float(np.mean(radii))
            # Clamp for visualization
            avg_radius = min(avg_radius, 5.0)

            # Vary color per region
            hue_offset = (i * 0.618) % 1.0  # Golden ratio for spacing
            r = base_color[0] + 0.3 * np.sin(hue_offset * 6.28)
            g = base_color[1] + 0.3 * np.sin(hue_offset * 6.28 + 2.09)
            b = base_color[2] + 0.3 * np.sin(hue_offset * 6.28 + 4.19)

            if len(center) == 2:
                cx, cy, cz = float(center[0]), float(center[1]), 0.01
            else:
                cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

            diam = avg_radius * 2
            spheres.append({
                "pose": _make_pose(cx, cy, cz),
                "size": _make_vector3(diam, diam, diam),
                "color": _make_color(
                    float(np.clip(r, 0, 1)),
                    float(np.clip(g, 0, 1)),
                    float(np.clip(b, 0, 1)),
                    base_color[3],
                ),
            })

        time_ns = self._advance_time()
        sec = time_ns // 1_000_000_000
        nsec = time_ns % 1_000_000_000

        msg = {
            "deletions": [],
            "entities": [{
                "timestamp": _make_timestamp(sec, nsec),
                "frame_id": frame_id,
                "id": "convex_regions",
                "lifetime": {"sec": 0, "nsec": 0},
                "frame_locked": True,
                "metadata": [],
                "arrows": [],
                "cubes": [],
                "spheres": spheres,
                "cylinders": [],
                "lines": [],
                "triangles": [],
                "texts": [],
                "models": [],
            }],
        }

        self._write_json(channel_id, msg, time_ns=time_ns)

    def add_frame_transform(
        self,
        parent_frame: str,
        child_frame: str,
        translation: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 1),
        topic: str = "/tf",
    ):
        """
        Add a static frame transform.

        Args:
            parent_frame: Parent frame ID
            child_frame: Child frame ID
            translation: (x, y, z)
            rotation: (qx, qy, qz, qw)
            topic: Topic name
        """
        channel_id = self._get_channel(
            topic, "foxglove.FrameTransform", _FRAME_TRANSFORM_SCHEMA
        )

        msg = {
            "timestamp": _make_timestamp(0, 0),
            "parent_frame_id": parent_frame,
            "child_frame_id": child_frame,
            "translation": _make_vector3(*translation),
            "rotation": {"x": rotation[0], "y": rotation[1],
                         "z": rotation[2], "w": rotation[3]},
        }

        self._write_json(channel_id, msg, time_ns=0)

    def close(self):
        """Finalize and close the MCAP file."""
        if not self._closed:
            self.writer.finish()
            self._file.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


def record_trajectory_scene(
    output_path: str,
    trajectory,
    obstacles: Optional[List] = None,
    regions: Optional[List] = None,
    robot_urdf: Optional[str] = None,
    environment_urdf: Optional[str] = None,
) -> str:
    """
    Convenience function to record a complete planning scene.

    Creates an MCAP file with trajectory, obstacles, regions, and
    optionally robot/environment models.

    Args:
        output_path: Path to output .mcap file
        trajectory: Trajectory object
        obstacles: Optional list of SimpleBoxObstacle objects
        regions: Optional list of Ellipsoid regions
        robot_urdf: Optional path to robot URDF
        environment_urdf: Optional path to environment URDF

    Returns:
        Path to the created MCAP file

    Example:
        >>> from hybrid_gcs.visualization import record_trajectory_scene
        >>> path = record_trajectory_scene(
        ...     "scene.mcap",
        ...     trajectory,
        ...     obstacles=obstacles,
        ...     regions=regions,
        ... )
        >>> print(f"Open {path} in Foxglove Studio")
    """
    with FoxgloveRecorder(output_path) as recorder:
        # Add frame transform
        recorder.add_frame_transform("world", "base_link")

        # Add robot model
        if robot_urdf is not None:
            recorder.add_robot_description(robot_urdf)

        # Add environment
        if environment_urdf is not None:
            recorder.add_environment(environment_urdf)

        # Add obstacles
        if obstacles is not None and len(obstacles) > 0:
            recorder.add_obstacles(obstacles)

        # Add convex regions
        if regions is not None and len(regions) > 0:
            recorder.add_convex_regions(regions)

        # Add trajectory
        recorder.add_trajectory(trajectory)

    return str(Path(output_path).resolve())
