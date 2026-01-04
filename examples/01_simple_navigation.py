"""
Example 1: Simple 2D Navigation with GCS Planning

Demonstrates basic GCS planning functionality:
1. Create configuration space
2. Decompose space into convex regions using IRIS
3. Build GCS graph
4. Solve for collision-free trajectory
5. Visualize results

This is a simplified example for validation. For real applications,
integrate with proper visualization (matplotlib, rviz) and solvers.
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
    Ellipsoid,
    SimpleBoxObstacle,
    MICPSolver,
    GCSGraph
)


def main():
    """Run simple navigation example."""
    
    print("=" * 60)
    print("Hybrid-GCS Example 1: Simple 2D Navigation")
    print("=" * 60)
    
    # Step 1: Create configuration space
    print("\n[Step 1] Creating 2D configuration space...")
    config_space = ConfigSpace(
        dim=2,
        bounds_lower=np.array([0.0, 0.0]),
        bounds_upper=np.array([10.0, 10.0]),
        names=['x', 'y']
    )
    print(f"  ✓ {config_space}")
    print(f"  Bounds: [{config_space.bounds_lower[0]:.1f}, {config_space.bounds_upper[0]:.1f}] x "
          f"[{config_space.bounds_lower[1]:.1f}, {config_space.bounds_upper[1]:.1f}]")
    
    # Step 2: Test basic ConfigSpace operations
    print("\n[Step 2] Testing ConfigSpace operations...")
    
    # Random sampling
    samples = config_space.random_samples(5)
    print(f"  ✓ Random samples: {samples.shape[0]} samples of dim {samples.shape[1]}")
    
    # Distance and interpolation
    q1 = np.array([0.0, 0.0])
    q2 = np.array([10.0, 10.0])
    dist = config_space.distance(q1, q2)
    print(f"  ✓ Distance from {q1} to {q2}: {dist:.4f}")
    
    q_mid = config_space.interpolate(q1, q2, 0.5)
    print(f"  ✓ Midpoint interpolation: {q_mid}")
    
    # Step 3: Create simple obstacles
    print("\n[Step 3] Creating obstacles...")
    obstacles = [
        SimpleBoxObstacle(
            lower=np.array([3.0, 3.0]),
            upper=np.array([7.0, 7.0])
        ),
        SimpleBoxObstacle(
            lower=np.array([1.0, 8.0]),
            upper=np.array([4.0, 9.5])
        )
    ]
    print(f"  ✓ Created {len(obstacles)} obstacles")
    
    # Step 4: IRIS Decomposition
    print("\n[Step 4] Decomposing space with IRIS...")
    decomposer = IRISDecomposer(
        config_space,
        max_iterations=10,
        termination_threshold=0.001,
        verbose=False
    )
    
    # Seed points for region growth
    seed_points = [
        np.array([1.5, 1.5]),
        np.array([8.5, 1.5]),
        np.array([1.5, 8.5]),
        np.array([8.5, 8.5]),
        np.array([5.0, 0.5])
    ]
    
    regions = decomposer.decompose(seed_points, obstacles, max_regions=10)
    print(f"  ✓ Created {len(regions)} convex regions")
    for i, region in enumerate(regions):
        print(f"    Region {i}: volume={region.volume():.6f}")
    
    # Step 5: Build GCS Graph
    print("\n[Step 5] Building GCS graph...")
    graph = GCSGraph()
    
    # Add regions as vertices
    for i, region in enumerate(regions):
        graph.add_vertex(i, region=region)
    
    # Add edges between adjacent regions (simplified: connect all)
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            graph.add_edge(i, j)
            graph.add_edge(j, i)
    
    print(f"  ✓ Graph: {graph.num_vertices()} vertices, {graph.num_edges()} edges")
    
    # Step 6: Test trajectory representations
    print("\n[Step 6] Testing trajectory representations...")
    
    # Simple waypoint trajectory
    waypoints = np.array([
        [0.0, 0.0],
        [5.0, 5.0],
        [10.0, 10.0]
    ])
    traj = Trajectory(waypoints)
    print(f"  ✓ Waypoint trajectory: {len(traj)} waypoints, length={traj.length():.4f}")
    
    # Bezier curve
    control_points = np.array([
        [0.0, 0.0],
        [3.0, 8.0],
        [7.0, 3.0],
        [10.0, 10.0]
    ])
    bezier = BezierTrajectory(control_points)
    print(f"  ✓ Bezier trajectory: degree={bezier.degree}, dim={bezier.dim}")
    
    # Convert to waypoint form
    bezier_traj = bezier.to_trajectory(n_samples=20)
    print(f"  ✓ Converted to {len(bezier_traj)} waypoints")
    
    # Step 7: Trajectory operations
    print("\n[Step 7] Testing trajectory operations...")
    
    config_at_t = traj.at_time(0.5)
    print(f"  ✓ Configuration at t=0.5: {config_at_t}")
    
    velocity = traj.velocity_at_time(0.5)
    print(f"  ✓ Velocity at t=0.5: {velocity}")
    
    resampled = traj.resample(10)
    print(f"  ✓ Resampled trajectory: {len(resampled)} waypoints")
    
    # Step 8: GCS-based trajectory planning
    print("\n[Step 8] Planning trajectory with GCS...")
    
    start = np.array([0.5, 0.5])
    goal = np.array([9.5, 9.5])
    
    # Validate start and goal
    assert config_space.is_valid(start), "Start outside configuration space"
    assert config_space.is_valid(goal), "Goal outside configuration space"
    print(f"  ✓ Start: {start}, Goal: {goal}")
    
    solver = MICPSolver(graph, config_space, solver_type='scs', verbose=False)
    planned_traj = solver.solve(start, goal)
    
    if planned_traj is not None:
        print(f"  ✓ Trajectory found!")
        print(f"    Length: {planned_traj.length():.4f}")
        print(f"    Waypoints: {len(planned_traj)}")
        print(f"    Duration: {planned_traj.duration():.4f}")
    else:
        print(f"  ✗ No trajectory found")
    
    # Step 9: Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Configuration space: {config_space.dim}D")
    print(f"Obstacles: {len(obstacles)}")
    print(f"IRIS regions: {len(regions)}")
    print(f"GCS graph edges: {graph.num_edges()}")
    print(f"Planned trajectory length: {planned_traj.length():.4f}" if planned_traj else "No trajectory")
    print("\n✓ Phase 2 validation complete!")
    
    return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
