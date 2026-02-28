"""
Tests for Foxglove MCAP recorder.

Tests MCAP file creation, channel registration, and scene recording
for Foxglove Studio visualization.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from mcap.reader import make_reader

from hybrid_gcs.core import (
    ConfigSpace,
    GCSGraph,
    IRISDecomposer,
    MICPSolver,
    SimpleBoxObstacle,
    Trajectory,
)
from hybrid_gcs.visualization import FoxgloveRecorder, record_trajectory_scene

# Path to test models
_PKG_ROOT = Path(__file__).parent.parent.parent
_UR5E_URDF = _PKG_ROOT / "data" / "models" / "ur5e" / "ur5e.urdf"
_ENV_URDF = _PKG_ROOT / "data" / "models" / "environment" / "tabletop.urdf"


@pytest.fixture
def simple_trajectory():
    """Create a simple 2D trajectory."""
    waypoints = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [6.0, 2.0],
            [10.0, 10.0],
        ]
    )
    return Trajectory(waypoints)


@pytest.fixture
def simple_obstacles():
    """Create test obstacles."""
    return [
        SimpleBoxObstacle(lower=np.array([3.0, 3.0]), upper=np.array([7.0, 7.0])),
        SimpleBoxObstacle(lower=np.array([1.0, 8.0]), upper=np.array([4.0, 9.5])),
    ]


@pytest.fixture
def simple_regions():
    """Create test regions via IRIS decomposition."""
    config_space = ConfigSpace(
        dim=2,
        bounds_lower=np.array([0.0, 0.0]),
        bounds_upper=np.array([10.0, 10.0]),
        names=["x", "y"],
    )
    decomposer = IRISDecomposer(config_space, max_iterations=5)
    return decomposer.decompose([np.array([5.0, 5.0])])


class TestFoxgloveRecorder:
    """Test FoxgloveRecorder class."""

    def test_create_and_close(self, tmp_path):
        """Test basic recorder creation and closing."""
        output = str(tmp_path / "test.mcap")
        recorder = FoxgloveRecorder(output)
        recorder.close()

        assert Path(output).exists()
        assert Path(output).stat().st_size > 0

    def test_context_manager(self, tmp_path):
        """Test context manager usage."""
        output = str(tmp_path / "test.mcap")
        with FoxgloveRecorder(output) as recorder:
            pass

        assert Path(output).exists()

    def test_add_trajectory(self, tmp_path, simple_trajectory):
        """Test adding a trajectory."""
        output = str(tmp_path / "traj.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_trajectory(simple_trajectory)

        # Verify MCAP contents
        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/trajectory" in topics
            assert summary.statistics.message_count >= 1

    def test_add_obstacles(self, tmp_path, simple_obstacles):
        """Test adding obstacles."""
        output = str(tmp_path / "obs.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_obstacles(simple_obstacles)

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/obstacles" in topics

    def test_add_convex_regions(self, tmp_path, simple_regions):
        """Test adding convex regions."""
        output = str(tmp_path / "regions.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_convex_regions(simple_regions)

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/regions" in topics

    def test_add_robot_description(self, tmp_path):
        """Test adding robot URDF description."""
        if not _UR5E_URDF.exists():
            pytest.skip("UR5e URDF not found")

        output = str(tmp_path / "robot.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_robot_description(str(_UR5E_URDF))

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/robot_description" in topics

    def test_add_environment(self, tmp_path):
        """Test adding environment URDF."""
        if not _ENV_URDF.exists():
            pytest.skip("Environment URDF not found")

        output = str(tmp_path / "env.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_environment(str(_ENV_URDF))

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/environment" in topics

    def test_add_frame_transform(self, tmp_path):
        """Test adding frame transforms."""
        output = str(tmp_path / "tf.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_frame_transform("world", "base_link")

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/tf" in topics

    def test_missing_urdf_raises(self, tmp_path):
        """Test that missing URDF file raises FileNotFoundError."""
        output = str(tmp_path / "bad.mcap")
        with FoxgloveRecorder(output) as recorder:
            with pytest.raises(FileNotFoundError):
                recorder.add_robot_description("/nonexistent/robot.urdf")

    def test_full_scene(self, tmp_path, simple_trajectory, simple_obstacles, simple_regions):
        """Test recording a complete scene."""
        output = str(tmp_path / "full.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_frame_transform("world", "base_link")
            if _UR5E_URDF.exists():
                recorder.add_robot_description(str(_UR5E_URDF))
            if _ENV_URDF.exists():
                recorder.add_environment(str(_ENV_URDF))
            recorder.add_obstacles(simple_obstacles)
            recorder.add_convex_regions(simple_regions)
            recorder.add_trajectory(simple_trajectory)

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            # At minimum: tf, obstacles, regions, trajectory
            assert summary.statistics.message_count >= 4

    def test_trajectory_message_content(self, tmp_path, simple_trajectory):
        """Test that trajectory message has correct structure."""
        output = str(tmp_path / "msg.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_trajectory(simple_trajectory)

        with open(output, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=["/trajectory"]):
                data = json.loads(message.data)
                assert "entities" in data
                # Should have trajectory line + start marker + goal marker
                assert len(data["entities"]) == 3
                # First entity is the trajectory line
                traj_entity = data["entities"][0]
                assert traj_entity["id"] == "trajectory"
                assert len(traj_entity["lines"]) == 1
                assert len(traj_entity["lines"][0]["points"]) == 50

    def test_custom_topic_names(self, tmp_path, simple_trajectory):
        """Test recording with custom topic names."""
        output = str(tmp_path / "custom.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_trajectory(simple_trajectory, topic="/my_custom_trajectory")

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/my_custom_trajectory" in topics

    def test_3d_trajectory(self, tmp_path):
        """Test with a 3D trajectory."""
        waypoints = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 2.0],
            ]
        )
        traj = Trajectory(waypoints)

        output = str(tmp_path / "3d.mcap")
        with FoxgloveRecorder(output) as recorder:
            recorder.add_trajectory(traj)

        with open(output, "rb") as f:
            reader = make_reader(f)
            for schema, channel, message in reader.iter_messages(topics=["/trajectory"]):
                data = json.loads(message.data)
                points = data["entities"][0]["lines"][0]["points"]
                # Verify z-coordinates are preserved (not zeroed)
                has_nonzero_z = any(p["z"] > 0.01 for p in points)
                assert has_nonzero_z


class TestRecordTrajectoryScene:
    """Test record_trajectory_scene convenience function."""

    def test_basic(self, tmp_path, simple_trajectory):
        """Test basic convenience function."""
        output = str(tmp_path / "scene.mcap")
        path = record_trajectory_scene(output, simple_trajectory)

        assert Path(path).exists()

    def test_with_obstacles(self, tmp_path, simple_trajectory, simple_obstacles):
        """Test with obstacles."""
        output = str(tmp_path / "scene.mcap")
        record_trajectory_scene(output, simple_trajectory, obstacles=simple_obstacles)

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/obstacles" in topics
            assert "/trajectory" in topics

    def test_with_regions(self, tmp_path, simple_trajectory, simple_regions):
        """Test with regions."""
        output = str(tmp_path / "scene.mcap")
        record_trajectory_scene(output, simple_trajectory, regions=simple_regions)

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/regions" in topics

    def test_with_robot_urdf(self, tmp_path, simple_trajectory):
        """Test with robot URDF."""
        if not _UR5E_URDF.exists():
            pytest.skip("UR5e URDF not found")

        output = str(tmp_path / "scene.mcap")
        record_trajectory_scene(output, simple_trajectory, robot_urdf=str(_UR5E_URDF))

        with open(output, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()
            topics = [ch.topic for ch in summary.channels.values()]
            assert "/robot_description" in topics
