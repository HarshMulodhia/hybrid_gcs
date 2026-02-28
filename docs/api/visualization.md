# Visualization API Reference

## FoxgloveRecorder

```python
from hybrid_gcs.visualization import FoxgloveRecorder
```

Records Hybrid-GCS planning scenes to MCAP files for
[Foxglove Studio](https://foxglove.dev/studio) visualization.

Uses local URDF models with primitive geometries (box, cylinder) for
fast rendering without external mesh file dependencies.

### Constructor

```python
FoxgloveRecorder(output_path: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `output_path` | str | Path to output `.mcap` file |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `add_robot_description` | `(urdf_path, topic="/robot_description")` | Add robot URDF model |
| `add_environment` | `(urdf_path, topic="/environment")` | Add environment URDF (table, room) |
| `add_trajectory` | `(trajectory, frame_id, topic, color, n_samples)` | Record trajectory as line strip with start/goal markers |
| `add_obstacles` | `(obstacles, frame_id, topic, color)` | Record box obstacles |
| `add_convex_regions` | `(regions, frame_id, topic, base_color)` | Record IRIS ellipsoid regions as spheres |
| `add_frame_transform` | `(parent, child, translation, rotation)` | Add static TF frame |
| `close` | `()` | Finalize MCAP file |

### Context Manager

```python
with FoxgloveRecorder("output.mcap") as recorder:
    recorder.add_trajectory(trajectory)
    # File is automatically closed on exit
```

---

## record_trajectory_scene

```python
from hybrid_gcs.visualization import record_trajectory_scene
```

Convenience function to record a complete planning scene in one call.

```python
record_trajectory_scene(
    output_path: str,
    trajectory,
    obstacles: Optional[List] = None,
    regions: Optional[List] = None,
    robot_urdf: Optional[str] = None,
    environment_urdf: Optional[str] = None,
) -> str
```

Returns the absolute path to the created `.mcap` file.

### Example

```python
path = record_trajectory_scene(
    "scene.mcap",
    trajectory,
    obstacles=obstacles,
    regions=regions,
    robot_urdf="data/models/ur5e/ur5e.urdf",
    environment_urdf="data/models/environment/tabletop.urdf",
)
print(f"Open {path} in Foxglove Studio")
```

---

## MCAP Topics

| Topic | Schema | Content |
|-------|--------|---------|
| `/trajectory` | `foxglove.SceneUpdate` | Trajectory line strip, start/goal markers |
| `/obstacles` | `foxglove.SceneUpdate` | Box obstacle cubes |
| `/regions` | `foxglove.SceneUpdate` | Convex region spheres |
| `/robot_description` | `foxglove.SceneUpdate` | Robot URDF metadata |
| `/environment` | `foxglove.SceneUpdate` | Environment URDF metadata |
| `/tf` | `foxglove.FrameTransform` | Static frame transforms |

---

## Included URDF Models

| Model | Path | Description |
|-------|------|-------------|
| UR5e | `data/models/ur5e/ur5e.urdf` | 6-DOF robot arm (primitive geometries) |
| Tabletop | `data/models/environment/tabletop.urdf` | Table with legs + floor |

Models use only primitive shapes (cylinder, box) for optimized
Foxglove rendering — no external mesh files required.

---

## PyBulletRenderer

```python
from hybrid_gcs.visualization import PyBulletRenderer
```

Replays recorded episode trajectories in a PyBullet physics simulation
with 3-D rendering.  Falls back to JSON trajectory export when PyBullet
is not installed.

### Constructor

```python
PyBulletRenderer(mode: str = "direct")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mode` | str | `"direct"` | `"gui"` for on-screen, `"direct"` for headless |

### Methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `connect` | `() -> int` | Connect to PyBullet server; returns client ID |
| `setup_scene` | `(env_name, num_agents=1)` | Populate scene with domain objects |
| `replay` | `(frames, env_name, dt, num_agents)` | Step through recorded frames |
| `close` | `()` | Disconnect from PyBullet |
| `save_trajectory_json` | `(frames, path) -> str` | Static fallback: save JSON |

### Context Manager

```python
with PyBulletRenderer(mode="gui") as renderer:
    renderer.setup_scene("grasping")
    renderer.replay(frames, env_name="grasping", dt=0.01)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `available` | `bool` | `True` if PyBullet is importable |

---

## Foxglove Studio Layout

Import `data/foxglove/hybrid_gcs_layout.json` for an optimized panel
arrangement with a 3D view panel (75%) and raw message panel (25%).

### Setup Steps

1. Install [Foxglove Studio](https://foxglove.dev/studio) (desktop or web)
2. Open your `.mcap` file via **File → Open local file**
3. Add a **3D** panel from the panel menu
4. (Optional) Import the layout JSON for preconfigured views
