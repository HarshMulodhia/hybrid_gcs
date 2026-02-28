"""
Example 3: Foxglove Studio Visualization

Records GCS planning results (trajectory, obstacles, convex regions) to
an MCAP file for interactive 3D visualization in Foxglove Studio. Uses
local URDF models with primitive geometries for optimized rendering.

Usage:
    python examples/03_foxglove_visualization.py

    Then open the output .mcap file in Foxglove Studio:
    https://foxglove.dev/studio

    Import the layout from data/foxglove/hybrid_gcs_layout.json
    for an optimized panel arrangement.
"""

import numpy as np
import sys
from pathlib import Path

# Add hybrid_gcs to path
hybrid_gcs_path = Path(__file__).parent.parent
sys.path.insert(0, str(hybrid_gcs_path))

from hybrid_gcs.core import (
    ConfigSpace,
    Trajectory,
    BezierTrajectory,
    IRISDecomposer,
    SimpleBoxObstacle,
    MICPSolver,
    GCSGraph,
)
from hybrid_gcs.visualization import FoxgloveRecorder, record_trajectory_scene


def main():
    """Run Foxglove visualization example."""

    print("=" * 60)
    print("Hybrid-GCS Example 3: Foxglove Studio Visualization")
    print("=" * 60)

    # --- Step 1: GCS Planning (same as example 01) ---

    print("\n[Step 1] Setting up planning problem...")
    config_space = ConfigSpace(
        dim=2,
        bounds_lower=np.array([0.0, 0.0]),
        bounds_upper=np.array([10.0, 10.0]),
        names=["x", "y"],
    )

    obstacles = [
        SimpleBoxObstacle(lower=np.array([3.0, 3.0]), upper=np.array([7.0, 7.0])),
        SimpleBoxObstacle(lower=np.array([1.0, 8.0]), upper=np.array([4.0, 9.5])),
    ]
    print(f"  Created {len(obstacles)} obstacles")

    decomposer = IRISDecomposer(config_space, max_iterations=10, verbose=False)
    seed_points = [
        np.array([1.5, 1.5]),
        np.array([8.5, 1.5]),
        np.array([1.5, 8.5]),
        np.array([8.5, 8.5]),
        np.array([5.0, 0.5]),
    ]
    regions = decomposer.decompose(seed_points, obstacles, max_regions=10)
    print(f"  IRIS decomposition: {len(regions)} convex regions")

    graph = GCSGraph()
    for i, region in enumerate(regions):
        graph.add_vertex(i, region=region)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            graph.add_edge(i, j)
            graph.add_edge(j, i)

    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])
    solver = MICPSolver(graph, config_space, solver_type="scs", verbose=False)
    trajectory = solver.solve(start, goal)

    if trajectory is None:
        print("  ERROR: No trajectory found!")
        return 1

    print(f"  Trajectory: {len(trajectory)} waypoints, length={trajectory.length():.2f}")

    # --- Step 2: Record to MCAP (Foxglove format) ---

    print("\n[Step 2] Recording scene to MCAP...")

    # Resolve paths
    pkg_root = Path(__file__).parent.parent
    robot_urdf = str(pkg_root / "data" / "models" / "ur5e" / "ur5e.urdf")
    env_urdf = str(pkg_root / "data" / "models" / "environment" / "tabletop.urdf")
    output_dir = pkg_root / "data" / "foxglove"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(output_dir / "planning_scene.mcap")

    with FoxgloveRecorder(output_path) as recorder:
        # Add frame transforms
        recorder.add_frame_transform("world", "base_link")
        print("  ✓ Frame transforms")

        # Add robot and environment models
        recorder.add_robot_description(robot_urdf)
        print("  ✓ Robot model (UR5e)")

        recorder.add_environment(env_urdf)
        print("  ✓ Environment model (tabletop)")

        # Add obstacles
        recorder.add_obstacles(obstacles)
        print(f"  ✓ {len(obstacles)} obstacles")

        # Add convex regions
        recorder.add_convex_regions(regions)
        print(f"  ✓ {len(regions)} convex regions")

        # Add planned trajectory
        recorder.add_trajectory(trajectory)
        print(f"  ✓ Trajectory ({len(trajectory)} waypoints)")

    print(f"\n  Output: {output_path}")

    # --- Step 3: Also record using convenience function ---

    print("\n[Step 3] Recording with convenience function...")
    simple_output = str(output_dir / "trajectory_only.mcap")
    record_trajectory_scene(
        simple_output,
        trajectory,
        obstacles=obstacles,
        regions=regions,
    )
    print(f"  Output: {simple_output}")

    # --- Summary ---

    print("\n" + "=" * 60)
    print("Visualization files created!")
    print("=" * 60)
    print(f"\n  Full scene:      {output_path}")
    print(f"  Trajectory only: {simple_output}")
    print(f"  Layout config:   {output_dir / 'hybrid_gcs_layout.json'}")
    print(f"\nOpen in Foxglove Studio: https://foxglove.dev/studio")
    print("  1. File → Open local file → select .mcap file")
    print("  2. Add a 3D panel to view the scene")
    print("  3. (Optional) Import layout from hybrid_gcs_layout.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
