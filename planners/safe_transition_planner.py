"""Collision-aware safe Cartesian transition planner with box waypoint routing."""

from __future__ import annotations

import heapq
from dataclasses import replace
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from planners.base import Pose, MotionType


def construct_safety_envelope(model, config: PipelineConfig) -> trimesh.primitives.Box:
    """Compute and inflate oriented or axis-aligned bounding box safety envelope."""
    mesh = model.validation_mesh.mesh
    envelope_type = config.envelope_type.lower()
    
    # Effective clearance margin: safety clearance + tool envelope radius
    inflation = config.safety_clearance_mm + config.tool_envelope_radius_mm

    if envelope_type == "obb":
        # Calculate Oriented Bounding Box
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        obb_transform = np.linalg.inv(to_origin)
        inflated_extents = extents + 2.0 * inflation
        box = trimesh.primitives.Box(extents=inflated_extents, transform=obb_transform)
    else:
        # Calculate Axis-Aligned Bounding Box
        bounds = mesh.bounds
        aabb_min, aabb_max = bounds[0], bounds[1]
        aabb_center = 0.5 * (aabb_min + aabb_max)
        aabb_extents = aabb_max - aabb_min
        inflated_extents = aabb_extents + 2.0 * inflation
        T = np.eye(4)
        T[:3, 3] = aabb_center
        box = trimesh.primitives.Box(extents=inflated_extents, transform=T)
        
    return box


def calculate_exit_point(box: trimesh.primitives.Box, p_start: np.ndarray, dir_retract: np.ndarray) -> float:
    """Calculate the ray exit distance from p_start along dir_retract to the safety envelope boundary."""
    origins = np.array([p_start])
    directions = np.array([dir_retract])
    
    # Run raycast intersects against box mesh
    locations, _, _ = box.ray.intersects_location(origins, directions)
    
    if len(locations) > 0:
        distances = [float(np.linalg.norm(loc - p_start)) for loc in locations]
        pos_dists = [d for d in distances if d > 1e-2]
        if pos_dists:
            return min(pos_dists)
            
    # Fallback to box diagonal or default distance if no intersection
    return float(np.linalg.norm(box.extents)) * 0.5


def line_collision_with_mesh(mesh: trimesh.Trimesh, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Check if a line segment intersects a triangle mesh."""
    direction = p2 - p1
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return False
    dir_norm = direction / distance
    eps = 1.0
    if distance <= 2 * eps:
        mid = 0.5 * (p1 + p2)
        _, dists, _ = trimesh.proximity.closest_point(mesh, np.array([mid]))
        return bool(dists[0] < 1e-3)
    origin = p1 + eps * dir_norm
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([origin]),
        ray_directions=np.array([dir_norm]),
        multiple_hits=False
    )
    if len(locations) > 0:
        hit_dist = np.linalg.norm(locations[0] - origin)
        if hit_dist < (distance - 2 * eps):
            return True
    return False


def check_fixture_collision(fixture: trimesh.Trimesh | None, p1: np.ndarray, p2: np.ndarray) -> bool:
    """Verify if a line segment collides with the fixture mesh."""
    if fixture is None:
        return False
    direction = p2 - p1
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return False
    dir_norm = direction / distance
    locations, _, _ = fixture.ray.intersects_location(
        ray_origins=np.array([p1]),
        ray_directions=np.array([dir_norm]),
        multiple_hits=False
    )
    if len(locations) > 0:
        hit_dist = np.linalg.norm(locations[0] - p1)
        if hit_dist <= distance:
            return True
    return False


def is_point_inside_box(point: np.ndarray, box: trimesh.primitives.Box, margin: float = 1.0) -> bool:
    """Check if a point is inside the box with a margin (avoiding boundary floats)."""
    inv_t = np.linalg.inv(box.transform)
    p_local = trimesh.transform_points(np.array([point]), inv_t)[0]
    half = 0.5 * box.extents
    return bool(
        abs(p_local[0]) <= half[0] - margin and
        abs(p_local[1]) <= half[1] - margin and
        abs(p_local[2]) <= half[2] - margin
    )


def is_segment_safe(model, u: np.ndarray, v: np.ndarray, box: trimesh.primitives.Box, fixture_mesh: trimesh.Trimesh | None, config: PipelineConfig) -> bool:
    """Validate if a free-space path segment satisfies bounding box, mesh clearance, and fixture safety."""
    # 1. Bounding box penetration check
    ts_box = [0.2, 0.4, 0.5, 0.6, 0.8]
    for t in ts_box:
        pt = (1.0 - t) * u + t * v
        if is_point_inside_box(pt, box, margin=1.0):
            return False
            
    # 2. CAD collision check
    mesh = model.validation_mesh.mesh
    if line_collision_with_mesh(mesh, u, v):
        return False
        
    # 3. Fixture collision check
    if fixture_mesh is not None and check_fixture_collision(fixture_mesh, u, v):
        return False
        
    # 4. Scanner tool envelope and safety clearance validation
    dist = np.linalg.norm(v - u)
    if dist > 1e-6:
        interval = config.collision_sampling_interval_mm
        num_samples = max(2, int(np.ceil(dist / interval)))
        ts_clear = np.linspace(0.0, 1.0, num_samples)
        pts = np.array([(1.0 - t) * u + t * v for t in ts_clear])
        _, distances, _ = trimesh.proximity.closest_point(mesh, pts)
        req_clearance = config.safety_clearance_mm + config.tool_envelope_radius_mm
        if np.any(distances < (req_clearance - 1.0)):
            return False
            
    return True


def build_box_waypoints(box: trimesh.primitives.Box, config: PipelineConfig) -> list[np.ndarray]:
    """Generate OBB/AABB waypoints on the boundary and staging areas."""
    half = 0.5 * box.extents
    hx, hy, hz = half[0], half[1], half[2]
    T = box.transform
    
    # 1. Corners (8 vertices)
    local_corners = np.array([
        [-hx, -hy, -hz], [ hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz],
        [-hx, -hy,  hz], [ hx, -hy,  hz], [-hx,  hy,  hz], [ hx,  hy,  hz]
    ])
    corners = trimesh.transform_points(local_corners, T)
    
    # 2. Edge midpoints (12 points)
    local_edges = np.array([
        [0, -hy, -hz], [0,  hy, -hz], [0, -hy,  hz], [0,  hy,  hz],  # X edges
        [-hx, 0, -hz], [hx, 0, -hz], [-hx, 0,  hz], [hx, 0,  hz],  # Y edges
        [-hx, -hy, 0], [hx, -hy, 0], [-hx,  hy, 0], [hx,  hy, 0]   # Z edges
    ])
    edges = trimesh.transform_points(local_edges, T)
    
    # 3. Face centres (6 points)
    local_faces = np.array([
        [-hx, 0, 0], [hx, 0, 0],
        [0, -hy, 0], [0, hy, 0],
        [0, 0, -hz], [0, 0, hz]
    ])
    faces = trimesh.transform_points(local_faces, T)
    
    waypoints = list(corners) + list(edges) + list(faces)
    
    # 4. Top staging waypoints (global +Z offset)
    top_z_val = np.max([pt[2] for pt in corners])
    local_top_pts = np.array([
        [-hx, -hy,  hz], [ hx, -hy,  hz], [-hx,  hy,  hz], [ hx,  hy,  hz],
        [0, 0, hz]
    ])
    world_top = trimesh.transform_points(local_top_pts, T)
    for pt in world_top:
        stg = np.array([pt[0], pt[1], top_z_val + config.vertical_clearance_mm])
        waypoints.append(stg)
        
    # 5. Bottom staging waypoints (global -Z offset)
    if config.enable_bottom_sweep:
        bottom_z_val = np.min([pt[2] for pt in corners])
        local_bottom_pts = np.array([
            [-hx, -hy, -hz], [ hx, -hy, -hz], [-hx,  hy, -hz], [ hx,  hy, -hz],
            [0, 0, -hz]
        ])
        world_bottom = trimesh.transform_points(local_bottom_pts, T)
        for pt in world_bottom:
            stg = np.array([pt[0], pt[1], bottom_z_val - config.vertical_clearance_mm])
            waypoints.append(stg)
            
    # Filter waypoints to keep unique ones
    unique_wps: list[np.ndarray] = []
    for wp in waypoints:
        if not any(np.linalg.norm(wp - u) < 1e-2 for u in unique_wps):
            unique_wps.append(wp)
            
    return unique_wps


def plan_safe_transition(
    model,
    pose_start: Pose,
    pose_goal: Pose,
    box: trimesh.primitives.Box,
    fixture_mesh: trimesh.Trimesh | None,
    config: PipelineConfig,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """Find a collision-free route from pose_start to pose_goal.
    
    Returns:
        (strategy_name, list_of_safe_poses, list_of_rejected_direct_lines)
    """
    p_start = np.asarray(pose_start.position)
    q_start = np.asarray(pose_start.quaternion_xyzw)
    p_goal = np.asarray(pose_goal.position)
    q_goal = np.asarray(pose_goal.quaternion_xyzw)
    
    rejected_direct: list[tuple[np.ndarray, np.ndarray]] = []

    # Strategy A: Direct Free Transition
    if not config.enable_safe_traverse:
        return "direct", [(p_start, q_start), (p_goal, q_goal)], []
        
    # Limit orientation change for direct transition
    angle_diff = float(np.degrees((Rotation.from_quat(q_start).inv() * Rotation.from_quat(q_goal)).magnitude()))
    if angle_diff <= 60.0 and is_segment_safe(model, p_start, p_goal, box, fixture_mesh, config):
        return "direct", [(p_start, q_start), (p_goal, q_goal)], []
    else:
        rejected_direct.append((p_start, p_goal))

    # Geometrically calculate retract / approach points outside the inflated box
    # Retract direction: position - surface_point
    dir_retract = p_start - np.asarray(pose_start.surface_point)
    dir_retract_norm = np.linalg.norm(dir_retract)
    if dir_retract_norm > 1e-5:
        dir_retract = dir_retract / dir_retract_norm
    else:
        dir_retract = np.asarray(pose_start.surface_normal)
        dir_retract /= np.linalg.norm(dir_retract)
        
    d_retract = calculate_exit_point(box, p_start, dir_retract)
    p_retract = p_start + (d_retract + 20.0) * dir_retract

    # Approach direction
    dir_approach = p_goal - np.asarray(pose_goal.surface_point)
    dir_approach_norm = np.linalg.norm(dir_approach)
    if dir_approach_norm > 1e-5:
        dir_approach = dir_approach / dir_approach_norm
    else:
        dir_approach = np.asarray(pose_goal.surface_normal)
        dir_approach /= np.linalg.norm(dir_approach)
        
    d_approach = calculate_exit_point(box, p_goal, dir_approach)
    p_approach = p_goal + (d_approach + 20.0) * dir_approach

    # Strategy B: Retract-Direct-Approach
    seg1_safe = is_segment_safe(model, p_start, p_retract, box, fixture_mesh, config)
    seg2_safe = is_segment_safe(model, p_retract, p_approach, box, fixture_mesh, config)
    seg3_safe = is_segment_safe(model, p_approach, p_goal, box, fixture_mesh, config)
    
    if seg1_safe and seg2_safe and seg3_safe:
        path = [
            (p_start, q_start),
            (p_retract, q_start)
        ]
        if angle_diff > 1e-3:
            path.append((p_retract, q_goal))
        path.extend([
            (p_approach, q_goal),
            (p_goal, q_goal)
        ])
        return "retract_direct_approach", path, rejected_direct
    else:
        if seg1_safe and not seg2_safe:
            rejected_direct.append((p_retract, p_approach))

    # Strategy C: Bounding Box Route
    # Generate waypoints
    box_wps = build_box_waypoints(box, config)
    all_wps = [p_retract, p_approach] + box_wps
    
    n = len(all_wps)
    adj = {i: [] for i in range(n)}
    
    # Calculate clearances to steer Dijktra away from part
    clearances = []
    mesh = model.validation_mesh.mesh
    for wp in all_wps:
        _, dists, _ = trimesh.proximity.closest_point(mesh, np.array([wp]))
        clearances.append(float(dists[0]))

    # Build adjacency list
    for i in range(n):
        for j in range(i + 1, n):
            u, v = all_wps[i], all_wps[j]
            # Verify segment is safe
            if is_segment_safe(model, u, v, box, fixture_mesh, config):
                # Calculate cost: dist + corner penalty + vertical penalty + proximity penalty
                dist_val = np.linalg.norm(u - v)
                
                # Flat edge step penalty represents the corner penalty
                edge_cost = dist_val + config.corner_penalty
                
                # Vertical motion penalty
                edge_cost += config.vertical_motion_penalty * abs(u[2] - v[2])
                
                # Proximity penalty
                avg_clearance = 0.5 * (clearances[i] + clearances[j])
                if avg_clearance > 1e-3:
                    edge_cost += config.proximity_penalty / avg_clearance
                    
                adj[i].append((j, edge_cost))
                adj[j].append((i, edge_cost))

    # Run Dijkstra search from p_retract (index 0) to p_approach (index 1)
    queue = [(0.0, 0, [0])]
    visited = set()
    best_path = None
    best_cost = float("inf")
    
    while queue:
        cost, curr, path = heapq.heappop(queue)
        if curr == 1:
            if cost < best_cost:
                best_cost = cost
                best_path = path
            break
        if curr in visited:
            continue
        visited.add(curr)
        
        for neighbor, edge_cost in adj[curr]:
            if neighbor not in visited:
                heapq.heappush(queue, (cost + edge_cost, neighbor, path + [neighbor]))

    if best_path is not None and not config.prefer_top_clearance:
        strategy = "obb_route" if config.envelope_type == "obb" else "aabb_route"
        # Orientations: rotate to q_goal at p_retract, then traverse
        final_path = [(p_start, q_start), (p_retract, q_start)]
        if angle_diff > 1e-3:
            final_path.append((p_retract, q_goal))
        for node in best_path[1:-1]:
            final_path.append((all_wps[node], q_goal))
        final_path.extend([(p_approach, q_goal), (p_goal, q_goal)])
        return strategy, final_path, rejected_direct

    # Strategy D: Top Clearance Route
    # Max height of box
    top_z = np.max([pt[2] for pt in box.vertices])
    z_top_safe = top_z + config.vertical_clearance_mm
    
    w1_top = np.array([p_retract[0], p_retract[1], z_top_safe])
    w2_top = np.array([p_approach[0], p_approach[1], z_top_safe])
    
    t_seg1 = is_segment_safe(model, p_start, p_retract, box, fixture_mesh, config)
    t_seg2 = is_segment_safe(model, p_retract, w1_top, box, fixture_mesh, config)
    t_seg3 = is_segment_safe(model, w1_top, w2_top, box, fixture_mesh, config)
    t_seg4 = is_segment_safe(model, w2_top, p_approach, box, fixture_mesh, config)
    t_seg5 = is_segment_safe(model, p_approach, p_goal, box, fixture_mesh, config)
    
    if t_seg1 and t_seg2 and t_seg3 and t_seg4 and t_seg5:
        path = [
            (p_start, q_start),
            (p_retract, q_start)
        ]
        if angle_diff > 1e-3:
            path.append((p_retract, q_goal))
        path.extend([
            (w1_top, q_goal),
            (w2_top, q_goal),
            (p_approach, q_goal),
            (p_goal, q_goal)
        ])
        return "top_clearance", path, rejected_direct

    # Strategy E: Bottom Clearance Route
    if config.enable_bottom_sweep:
        bottom_z = np.min([pt[2] for pt in box.vertices])
        z_bottom_safe = bottom_z - config.vertical_clearance_mm
        
        w1_bot = np.array([p_retract[0], p_retract[1], z_bottom_safe])
        w2_bot = np.array([p_approach[0], p_approach[1], z_bottom_safe])
        
        b_seg1 = is_segment_safe(model, p_start, p_retract, box, fixture_mesh, config)
        b_seg2 = is_segment_safe(model, p_retract, w1_bot, box, fixture_mesh, config)
        b_seg3 = is_segment_safe(model, w1_bot, w2_bot, box, fixture_mesh, config)
        b_seg4 = is_segment_safe(model, w2_bot, p_approach, box, fixture_mesh, config)
        b_seg5 = is_segment_safe(model, p_approach, p_goal, box, fixture_mesh, config)
        
        if b_seg1 and b_seg2 and b_seg3 and b_seg4 and b_seg5:
            path = [
                (p_start, q_start),
                (p_retract, q_start)
            ]
            if angle_diff > 1e-3:
                path.append((p_retract, q_goal))
            path.extend([
                (w1_bot, q_goal),
                (w2_bot, q_goal),
                (p_approach, q_goal),
                (p_goal, q_goal)
            ])
            return "bottom_clearance", path, rejected_direct

    # Fallback/Failed transition
    return "failed", [(p_start, q_start), (p_goal, q_goal)], rejected_direct


def slerp_quaternions(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions with sign alignment."""
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    dot = np.dot(q1, q2)
    if dot < 0.0:
        q2 = -q2
        dot = -dot
    if dot > 0.9995:
        q = q1 + t * (q2 - q1)
        return q / np.linalg.norm(q)
    theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
    theta = theta_0 * t
    sin_theta = np.sin(theta)
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0
    q = (s0 * q1) + (s1 * q2)
    return q / np.linalg.norm(q)

