"""
Foxglove Studio MCAP Recorder for Hybrid-GCS.

Records robot scenes, trajectories, obstacles, and convex regions to
MCAP files for visualization in Foxglove Studio. Uses protobuf encoding
via ``foxglove-schemas-protobuf`` so Foxglove can natively parse every
message without schema mismatches.

Usage:
    from hybrid_gcs.visualization import FoxgloveRecorder

    recorder = FoxgloveRecorder("output.mcap")
    recorder.add_robot_description("data/models/ur5e/ur5e.urdf")
    recorder.add_trajectory(trajectory, frame_id="world")
    recorder.add_obstacles(obstacles, frame_id="world")
    recorder.close()

Then open output.mcap in Foxglove Studio.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from foxglove_schemas_protobuf.ArrowPrimitive_pb2 import ArrowPrimitive
from foxglove_schemas_protobuf.Color_pb2 import Color
from foxglove_schemas_protobuf.CubePrimitive_pb2 import CubePrimitive
from foxglove_schemas_protobuf.CylinderPrimitive_pb2 import CylinderPrimitive
from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
from foxglove_schemas_protobuf.KeyValuePair_pb2 import KeyValuePair
from foxglove_schemas_protobuf.LinePrimitive_pb2 import LinePrimitive
from foxglove_schemas_protobuf.Point3_pb2 import Point3
from foxglove_schemas_protobuf.Pose_pb2 import Pose
from foxglove_schemas_protobuf.Quaternion_pb2 import Quaternion
from foxglove_schemas_protobuf.SceneEntity_pb2 import SceneEntity
from foxglove_schemas_protobuf.SceneEntityDeletion_pb2 import SceneEntityDeletion
from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
from foxglove_schemas_protobuf.SpherePrimitive_pb2 import SpherePrimitive
from foxglove_schemas_protobuf.TextPrimitive_pb2 import TextPrimitive
from foxglove_schemas_protobuf.TriangleListPrimitive_pb2 import TriangleListPrimitive
from foxglove_schemas_protobuf.Vector3_pb2 import Vector3
from google.protobuf.duration_pb2 import Duration
from google.protobuf.timestamp_pb2 import Timestamp
from mcap_protobuf.writer import Writer as McapProtobufWriter

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _ts(sec: int = 0, nsec: int = 0) -> Timestamp:
    """Create a protobuf Timestamp."""
    t = Timestamp()
    t.seconds = sec
    t.nanos = nsec
    return t


def _dur(sec: int = 0, nsec: int = 0) -> Duration:
    """Create a protobuf Duration."""
    d = Duration()
    d.seconds = sec
    d.nanos = nsec
    return d


def _color(r: float, g: float, b: float, a: float = 1.0) -> Color:
    return Color(r=r, g=g, b=b, a=a)


def _vec3(x: float, y: float, z: float) -> Vector3:
    return Vector3(x=x, y=y, z=z)


def _point3(x: float, y: float, z: float) -> Point3:
    return Point3(x=x, y=y, z=z)


def _quat(x: float = 0, y: float = 0, z: float = 0, w: float = 1) -> Quaternion:
    return Quaternion(x=x, y=y, z=z, w=w)


def _pose(
    x: float = 0,
    y: float = 0,
    z: float = 0,
    qx: float = 0,
    qy: float = 0,
    qz: float = 0,
    qw: float = 1,
) -> Pose:
    return Pose(
        position=_vec3(x, y, z),
        orientation=_quat(qx, qy, qz, qw),
    )


# ---------------------------------------------------------------------------
# Recorder
# ---------------------------------------------------------------------------


class FoxgloveRecorder:
    """
    Records Hybrid-GCS scenes to MCAP files for Foxglove Studio.

    Uses **protobuf** encoding (``mcap-protobuf-support`` +
    ``foxglove-schemas-protobuf``) so every message is natively
    understood by Foxglove without custom JSON-schema workarounds.

    Attributes:
        output_path: Path to output MCAP file
        writer: mcap-protobuf Writer instance

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
        self.writer = McapProtobufWriter(self._file)

        self._time_ns: int = 0
        self._closed = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_time(self, delta_ns: int = 100_000_000) -> int:
        """Advance internal clock and return current time in nanoseconds."""
        self._time_ns += delta_ns
        return self._time_ns

    def _write(self, topic: str, msg, time_ns: Optional[int] = None):
        """Write a protobuf message to the MCAP file."""
        if time_ns is None:
            time_ns = self._time_ns
        self.writer.write_message(
            topic=topic,
            message=msg,
            log_time=time_ns,
            publish_time=time_ns,
        )

    @staticmethod
    def _ts_from_ns(time_ns: int) -> Timestamp:
        sec = time_ns // 1_000_000_000
        nsec = time_ns % 1_000_000_000
        return _ts(sec, nsec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_robot_description(self, urdf_path: str, topic: str = "/robot_description"):
        """
        Add robot URDF description for Foxglove.

        Reads a local URDF file and publishes it as metadata inside a
        ``SceneUpdate`` entity so Foxglove can display the model.

        Args:
            urdf_path: Path to URDF file (relative or absolute)
            topic:     MCAP topic name

        Raises:
            FileNotFoundError: If URDF file does not exist
        """
        urdf_file = Path(urdf_path)
        if not urdf_file.is_absolute():
            pkg_root = Path(__file__).parent.parent.parent
            urdf_file = pkg_root / urdf_path
        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")

        urdf_content = urdf_file.read_text(encoding="utf-8")

        entity = SceneEntity(
            timestamp=_ts(0, 0),
            frame_id="world",
            id="robot_description",
            lifetime=_dur(0, 0),
            frame_locked=True,
            metadata=[
                KeyValuePair(key="urdf", value=urdf_content),
                KeyValuePair(key="model_encoding", value="urdf"),
            ],
            texts=[
                TextPrimitive(
                    pose=_pose(0, 0, 1.2),
                    billboard=True,
                    font_size=14.0,
                    scale_invariant=True,
                    color=_color(1.0, 1.0, 1.0),
                    text="UR5e Robot",
                ),
            ],
        )
        msg = SceneUpdate(entities=[entity])
        self._write(topic, msg, time_ns=0)

    def add_environment(self, urdf_path: str, topic: str = "/environment"):
        """
        Add environment URDF (table, room) for Foxglove.

        Args:
            urdf_path: Path to environment URDF file
            topic:     MCAP topic name
        """
        urdf_file = Path(urdf_path)
        if not urdf_file.is_absolute():
            pkg_root = Path(__file__).parent.parent.parent
            urdf_file = pkg_root / urdf_path
        if not urdf_file.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_file}")

        urdf_content = urdf_file.read_text(encoding="utf-8")

        entity = SceneEntity(
            timestamp=_ts(0, 0),
            frame_id="world",
            id="environment",
            lifetime=_dur(0, 0),
            frame_locked=True,
            metadata=[
                KeyValuePair(key="urdf", value=urdf_content),
                KeyValuePair(key="model_encoding", value="urdf"),
            ],
        )
        msg = SceneUpdate(entities=[entity])
        self._write(topic, msg, time_ns=0)

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
        Add a trajectory visualization as a line strip with start/goal
        markers.

        Args:
            trajectory: Trajectory object with ``at_time(t)`` method
            frame_id:   Coordinate frame
            topic:      MCAP topic name
            color:      RGBA colour tuple
            line_width: Width of trajectory line
            n_samples:  Number of samples along trajectory
        """
        # Sample trajectory points
        t_start = trajectory.timestamps[0]
        t_end = trajectory.timestamps[-1]
        times = np.linspace(t_start, t_end, n_samples)
        points = [trajectory.at_time(t) for t in times]

        # Build line-strip points (promote 2-D to 3-D)
        line_points = []
        for pt in points:
            if len(pt) == 2:
                line_points.append(_point3(float(pt[0]), float(pt[1]), 0.01))
            else:
                line_points.append(_point3(float(pt[0]), float(pt[1]), float(pt[2])))

        # Gradient colours along the strip
        line_colors = []
        for i in range(len(line_points)):
            t = i / max(1, len(line_points) - 1)
            line_colors.append(
                _color(
                    color[0] * (1 - t) + 0.2 * t,
                    color[1] * (1 - t) + 0.3 * t,
                    color[2] * (1 - t) + 1.0 * t,
                    color[3],
                )
            )

        time_ns = self._advance_time()
        ts = self._ts_from_ns(time_ns)

        # --- trajectory line entity ---
        traj_entity = SceneEntity(
            timestamp=ts,
            frame_id=frame_id,
            id="trajectory",
            lifetime=_dur(0, 0),
            frame_locked=True,
            lines=[
                LinePrimitive(
                    type=0,  # LINE_STRIP
                    pose=_pose(),
                    thickness=line_width,
                    scale_invariant=False,
                    points=line_points,
                    color=_color(*color),
                    colors=line_colors,
                ),
            ],
        )

        # --- start marker ---
        sp = points[0]
        sx, sy = float(sp[0]), float(sp[1])
        sz = float(sp[2]) if len(sp) > 2 else 0.01
        start_entity = SceneEntity(
            timestamp=ts,
            frame_id=frame_id,
            id="start_marker",
            lifetime=_dur(0, 0),
            frame_locked=True,
            spheres=[
                SpherePrimitive(
                    pose=_pose(sx, sy, sz),
                    size=_vec3(0.08, 0.08, 0.08),
                    color=_color(0.0, 1.0, 0.0, 0.9),
                ),
            ],
            texts=[
                TextPrimitive(
                    pose=_pose(sx, sy, sz + 0.15),
                    billboard=True,
                    font_size=12.0,
                    scale_invariant=True,
                    color=_color(0.0, 1.0, 0.0),
                    text="START",
                ),
            ],
        )

        # --- goal marker ---
        gp = points[-1]
        gx, gy = float(gp[0]), float(gp[1])
        gz = float(gp[2]) if len(gp) > 2 else 0.01
        goal_entity = SceneEntity(
            timestamp=ts,
            frame_id=frame_id,
            id="goal_marker",
            lifetime=_dur(0, 0),
            frame_locked=True,
            spheres=[
                SpherePrimitive(
                    pose=_pose(gx, gy, gz),
                    size=_vec3(0.08, 0.08, 0.08),
                    color=_color(1.0, 0.0, 0.0, 0.9),
                ),
            ],
            texts=[
                TextPrimitive(
                    pose=_pose(gx, gy, gz + 0.15),
                    billboard=True,
                    font_size=12.0,
                    scale_invariant=True,
                    color=_color(1.0, 0.0, 0.0),
                    text="GOAL",
                ),
            ],
        )

        msg = SceneUpdate(entities=[traj_entity, start_entity, goal_entity])
        self._write(topic, msg, time_ns=time_ns)

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
            obstacles: List of ``SimpleBoxObstacle`` objects (``.lower``, ``.upper``)
            frame_id:  Coordinate frame
            topic:     MCAP topic name
            color:     RGBA colour
        """
        cubes = []
        for obs in obstacles:
            lower = np.asarray(obs.lower, dtype=np.float64)
            upper = np.asarray(obs.upper, dtype=np.float64)
            center = (lower + upper) / 2.0
            size = upper - lower

            if len(center) == 2:
                cx, cy = float(center[0]), float(center[1])
                sx, sy = float(size[0]), float(size[1])
                cz, sz = 0.25, 0.5
            else:
                cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
                sx, sy, sz = float(size[0]), float(size[1]), float(size[2])

            cubes.append(
                CubePrimitive(
                    pose=_pose(cx, cy, cz),
                    size=_vec3(sx, sy, sz),
                    color=_color(*color),
                )
            )

        time_ns = self._advance_time()
        ts = self._ts_from_ns(time_ns)

        entity = SceneEntity(
            timestamp=ts,
            frame_id=frame_id,
            id="obstacles",
            lifetime=_dur(0, 0),
            frame_locked=True,
            cubes=cubes,
        )
        msg = SceneUpdate(entities=[entity])
        self._write(topic, msg, time_ns=time_ns)

    def add_convex_regions(
        self,
        regions: List,
        frame_id: str = "world",
        topic: str = "/regions",
        base_color: Tuple[float, float, float, float] = (0.2, 0.4, 0.8, 0.15),
    ):
        """
        Add convex region visualizations as semi-transparent spheres.

        Represents IRIS ellipsoidal regions as spheres scaled by the
        average ellipsoid radius for approximate visualization.

        Args:
            regions:    List of ``Ellipsoid`` objects (``.center``, ``.shape_matrix``)
            frame_id:   Coordinate frame
            topic:      MCAP topic name
            base_color: RGBA base colour (varied per region)
        """
        spheres = []
        for i, region in enumerate(regions):
            center = np.asarray(region.center, dtype=np.float64)

            Q = np.asarray(region.shape_matrix, dtype=np.float64)
            try:
                eigvals = np.linalg.eigvalsh(Q)
                radii = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))
            except np.linalg.LinAlgError:
                radii = np.ones(len(center))

            avg_radius = min(float(np.mean(radii)), 5.0)

            hue_offset = (i * 0.618) % 1.0
            r = base_color[0] + 0.3 * np.sin(hue_offset * 6.28)
            g = base_color[1] + 0.3 * np.sin(hue_offset * 6.28 + 2.09)
            b = base_color[2] + 0.3 * np.sin(hue_offset * 6.28 + 4.19)

            if len(center) == 2:
                cx, cy, cz = float(center[0]), float(center[1]), 0.01
            else:
                cx, cy, cz = float(center[0]), float(center[1]), float(center[2])

            diam = avg_radius * 2
            spheres.append(
                SpherePrimitive(
                    pose=_pose(cx, cy, cz),
                    size=_vec3(diam, diam, diam),
                    color=_color(
                        float(np.clip(r, 0, 1)),
                        float(np.clip(g, 0, 1)),
                        float(np.clip(b, 0, 1)),
                        base_color[3],
                    ),
                )
            )

        time_ns = self._advance_time()
        ts = self._ts_from_ns(time_ns)

        entity = SceneEntity(
            timestamp=ts,
            frame_id=frame_id,
            id="convex_regions",
            lifetime=_dur(0, 0),
            frame_locked=True,
            spheres=spheres,
        )
        msg = SceneUpdate(entities=[entity])
        self._write(topic, msg, time_ns=time_ns)

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
            child_frame:  Child frame ID
            translation:  (x, y, z)
            rotation:     (qx, qy, qz, qw)
            topic:        MCAP topic name
        """
        msg = FrameTransform(
            timestamp=_ts(0, 0),
            parent_frame_id=parent_frame,
            child_frame_id=child_frame,
            translation=_vec3(*translation),
            rotation=_quat(*rotation),
        )
        self._write(topic, msg, time_ns=0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


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
        output_path:      Path to output .mcap file
        trajectory:       Trajectory object
        obstacles:        Optional list of SimpleBoxObstacle objects
        regions:          Optional list of Ellipsoid regions
        robot_urdf:       Optional path to robot URDF
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
        recorder.add_frame_transform("world", "base_link")

        if robot_urdf is not None:
            recorder.add_robot_description(robot_urdf)

        if environment_urdf is not None:
            recorder.add_environment(environment_urdf)

        if obstacles is not None and len(obstacles) > 0:
            recorder.add_obstacles(obstacles)

        if regions is not None and len(regions) > 0:
            recorder.add_convex_regions(regions)

        recorder.add_trajectory(trajectory)

    return str(Path(output_path).resolve())
