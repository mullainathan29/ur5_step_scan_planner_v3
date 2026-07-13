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


_COLLISION_CACHE: dict[tuple, tuple[bool, float]] = {}
_PROXIMITY_CACHE: dict[tuple, float] = {}

def clear_collision_caches() -> None:
    """Reset collision caches at the start of a mission planning session."""
    global _COLLISION_CACHE, _PROXIMITY_CACHE
    _COLLISION_CACHE.clear()
    _PROXIMITY_CACHE.clear()


def is_segment_safe(
    model,
    u: np.ndarray,
    v: np.ndarray,
    box: trimesh.primitives.Box,
    fixture_mesh: trimesh.Trimesh | None,
    config: PipelineConfig,
    return_clearance: bool = False
) -> bool | tuple[bool, float]:
    """Validate if a free-space path segment satisfies bounding box, mesh clearance, and fixture safety."""
    u_key = tuple(np.round(u, 3))
    v_key = tuple(np.round(v, 3))
    cache_key = tuple(sorted([u_key, v_key]))
    
    if cache_key in _COLLISION_CACHE:
        is_safe, min_clear = _COLLISION_CACHE[cache_key]
        if return_clearance:
            return is_safe, min_clear
        return is_safe

    # 1. Bounding box penetration check.  Free-space shortcuts must not pass
    # through the inflated envelope, but retract/approach legs are allowed to
    # start or end inside it while moving out to a safe waypoint.
    u_inside = is_point_inside_box(u, box, margin=1.0)
    v_inside = is_point_inside_box(v, box, margin=1.0)
    if not u_inside and not v_inside:
        ts_box = [0.2, 0.4, 0.5, 0.6, 0.8]
        for t in ts_box:
            pt = (1.0 - t) * u + t * v
            if is_point_inside_box(pt, box, margin=1.0):
                _COLLISION_CACHE[cache_key] = (False, 0.0)
                if return_clearance:
                    return False, 0.0
                return False
            
    # 2. CAD collision check
    mesh = model.validation_mesh.mesh
    if line_collision_with_mesh(mesh, u, v):
        _COLLISION_CACHE[cache_key] = (False, 0.0)
        if return_clearance:
            return False, 0.0
        return False
        
    # 3. Fixture collision check
    if fixture_mesh is not None and check_fixture_collision(fixture_mesh, u, v):
        _COLLISION_CACHE[cache_key] = (False, 0.0)
        if return_clearance:
            return False, 0.0
        return False
        
    # 4. Scanner tool envelope and safety clearance validation
    dist = np.linalg.norm(v - u)
    min_clear = float("inf")
    if dist > 1e-6:
        interval = config.collision_sampling_interval_mm
        num_samples = max(2, int(np.ceil(dist / interval)))
        ts_clear = np.linspace(0.0, 1.0, num_samples)
        pts = np.array([(1.0 - t) * u + t * v for t in ts_clear])
        
        # Check proximity cache for individual points to avoid redundant mesh queries
        distances = []
        uncached_pts = []
        uncached_idxs = []
        for idx, pt in enumerate(pts):
            pt_key = tuple(np.round(pt, 3))
            if pt_key in _PROXIMITY_CACHE:
                distances.append(_PROXIMITY_CACHE[pt_key])
            else:
                distances.append(-1.0)
                uncached_pts.append(pt)
                uncached_idxs.append(idx)
                
        if uncached_pts:
            _, query_dists, _ = trimesh.proximity.closest_point(mesh, np.array(uncached_pts))
            if len(query_dists) != len(uncached_pts):
                fill = float(query_dists[-1]) if len(query_dists) else float("inf")
                query_dists = np.full(len(uncached_pts), fill, dtype=float)
            for idx, q_d in zip(uncached_idxs, query_dists):
                pt_key = tuple(np.round(pts[idx], 3))
                d_val = float(q_d)
                _PROXIMITY_CACHE[pt_key] = d_val
                distances[idx] = d_val
                
        distances = np.array(distances)
        min_clear = float(np.min(distances))
        req_clearance = config.safety_clearance_mm + config.tool_envelope_radius_mm
        if np.any(distances < (req_clearance - 1.0)):
            _COLLISION_CACHE[cache_key] = (False, min_clear)
            if return_clearance:
                return False, min_clear
            return False
    else:
        # Single point clearance
        pt_key = tuple(np.round(u, 3))
        if pt_key in _PROXIMITY_CACHE:
            min_clear = _PROXIMITY_CACHE[pt_key]
        else:
            _, dists, _ = trimesh.proximity.closest_point(mesh, np.array([u]))
            min_clear = float(dists[0])
            _PROXIMITY_CACHE[pt_key] = min_clear
            
    _COLLISION_CACHE[cache_key] = (True, min_clear)
    if return_clearance:
        return True, min_clear
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


def catmull_rom_spline(pts: np.ndarray, num_samples: int, alpha: float = 0.5) -> np.ndarray:
    """Evaluate centripetal Catmull-Rom spline on a set of points."""
    N = len(pts)
    t = np.zeros(N)
    for i in range(1, N):
        dist = np.linalg.norm(pts[i] - pts[i-1])
        t[i] = t[i-1] + dist**alpha
        
    t_start = t[1]
    t_end = t[N-2]
    t_samples = np.linspace(t_start, t_end, num_samples)
    curve = np.zeros((num_samples, 3))
    
    for i, ts in enumerate(t_samples):
        idx = np.searchsorted(t, ts) - 1
        idx = np.clip(idx, 1, N - 3)
        
        p0 = pts[idx-1]
        p1 = pts[idx]
        p2 = pts[idx+1]
        p3 = pts[idx+2]
        
        t0, t1, t2, t3 = t[idx-1], t[idx], t[idx+1], t[idx+2]
        
        dt0 = t1 - t0 if t1 > t0 else 1.0
        dt1 = t2 - t1 if t2 > t1 else 1.0
        dt2 = t3 - t2 if t3 > t2 else 1.0
        
        A1 = ((t1 - ts) / dt0) * p0 + ((ts - t0) / dt0) * p1
        A2 = ((t2 - ts) / dt1) * p1 + ((ts - t1) / dt1) * p2
        A3 = ((t3 - ts) / dt2) * p2 + ((ts - t2) / dt2) * p3
        
        dt3 = t2 - t0 if t2 > t0 else 1.0
        dt4 = t3 - t1 if t3 > t1 else 1.0
        
        B1 = ((t2 - ts) / dt3) * A1 + ((ts - t0) / dt3) * A2
        B2 = ((t3 - ts) / dt4) * A2 + ((ts - t1) / dt4) * A3
        
        C = ((t2 - ts) / dt1) * B1 + ((ts - t1) / dt1) * B2
        curve[i] = C
        
    return curve


def cubic_bezier_curve(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, num_samples: int) -> np.ndarray:
    """Evaluate a cubic Bezier curve."""
    curve = np.zeros((num_samples, 3))
    for j in range(num_samples):
        t = j / (num_samples - 1)
        curve[j] = (1-t)**3 * p0 + 3*(1-t)**2 * t * p1 + 3*(1-t) * t**2 * p2 + t**3 * p3
    return curve


def circular_fillet_blend(p_prev: np.ndarray, p_curr: np.ndarray, p_next: np.ndarray, nominal_radius: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Compute circular fillet arc points at a corner."""
    v1 = p_prev - p_curr
    v2 = p_next - p_curr
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    if len1 < 1e-5 or len2 < 1e-5:
        return None
        
    v1 = v1 / len1
    v2 = v2 / len2
    
    dot_val = np.clip(np.dot(v1, v2), -1.0, 1.0)
    theta = np.arccos(dot_val)
    
    angle_deg = np.degrees(theta)
    if angle_deg > 175.0 or angle_deg < 5.0:
        return None
        
    max_d = 0.45 * min(len1, len2)
    tan_half = np.tan(theta / 2.0)
    if tan_half < 1e-5:
        return None
        
    max_R = max_d * tan_half
    R = min(nominal_radius, max_R)
    
    d = R / tan_half
    t1 = p_curr + d * v1
    t2 = p_curr + d * v2
    
    bisector = v1 + v2
    bisector_len = np.linalg.norm(bisector)
    if bisector_len < 1e-5:
        return None
    bisector /= bisector_len
    
    center_dist = R / np.sin(theta / 2.0)
    center = p_curr + center_dist * bisector
    
    normal = np.cross(v1, v2)
    normal_len = np.linalg.norm(normal)
    if normal_len < 1e-5:
        return None
    normal /= normal_len
    
    r1 = t1 - center
    r2 = t2 - center
    arc_angle = np.pi - theta
    
    arc_len = R * arc_angle
    num_pts = max(3, int(np.ceil(arc_len / 5.0)))
    
    arc_pts = []
    for j in range(num_pts):
        phi = (j / (num_pts - 1)) * arc_angle
        rot_r = r1 * np.cos(phi) + np.cross(normal, r1) * np.sin(phi)
        arc_pts.append(center + rot_r)
        
    return np.array(arc_pts), t1, t2, center


def slerp_trajectory_orientations(q_start: np.ndarray, q_goal: np.ndarray, arc_lengths: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Interpolate orientations using quaternion SLERP with active window."""
    N = len(arc_lengths)
    orientations = np.zeros((N, 4))
    if N == 0:
        return orientations
        
    total_len = arc_lengths[-1]
    if total_len < 1e-6:
        for i in range(N):
            orientations[i] = q_start
        return orientations
        
    if not config.orientation_smoothing_enabled:
        for i in range(N):
            orientations[i] = q_start if i < N - 1 else slerp_quaternions(q_start, q_goal, 1.0)
        return orientations

    W = min(max(config.orientation_interpolation_window_fraction, 1.0e-6), 1.0)
    f = (1.0 - W) / 2.0
    
    for i in range(N):
        u = arc_lengths[i] / total_len
        if u < f:
            t = 0.0
        elif u > 1.0 - f:
            t = 1.0
        else:
            t = (u - f) / W
            
        orientations[i] = slerp_quaternions(q_start, q_goal, t)
        
    return orientations


def adaptive_resample(positions: np.ndarray, orientations: np.ndarray, clearances: np.ndarray, config: PipelineConfig) -> list[int]:
    """Greedily select a subset of dense samples to satisfy step limits and clearance density."""
    N = len(positions)
    if N <= 2:
        return list(range(N))
        
    selected_indices = [0]
    curr = 0
    max_pos = min(config.max_position_step_mm, config.sample_spacing_mm) if config.sample_spacing_mm > 0.0 else config.max_position_step_mm
    max_ori = config.orientation_smoothing_maximum_step_deg
    
    while curr < N - 1:
        next_idx = curr + 1
        for candidate in range(curr + 2, N):
            p_dist = np.linalg.norm(positions[candidate] - positions[curr])
            if p_dist > max_pos:
                break
                
            q_curr = orientations[curr]
            q_cand = orientations[candidate]
            dot = np.abs(np.dot(q_curr, q_cand))
            angle = 2.0 * np.arccos(np.clip(dot, -1.0, 1.0))
            angle_deg = np.degrees(angle)
            if angle_deg > max_ori:
                break
                
            u = positions[curr]
            v = positions[candidate]
            seg_vec = v - u
            seg_len = np.linalg.norm(seg_vec)
            if seg_len > 1e-6:
                seg_dir = seg_vec / seg_len
                inter_pts = positions[curr+1:candidate]
                w = inter_pts - u
                proj = np.outer(np.dot(w, seg_dir), seg_dir)
                perp = w - proj
                perp_dists = np.linalg.norm(perp, axis=1)
                if np.any(perp_dists > 2.0):
                    break
            
            min_clear = np.min(clearances[curr:candidate+1])
            if min_clear < (config.safety_clearance_mm + config.tool_envelope_radius_mm + 20.0):
                if p_dist > 10.0:
                    break
                    
            next_idx = candidate
            
        selected_indices.append(next_idx)
        curr = next_idx
        
    return selected_indices


def compute_curvatures(positions: np.ndarray) -> np.ndarray:
    """Compute discrete Menger curvature at each point."""
    N = len(positions)
    curvatures = np.zeros(N)
    for i in range(1, N - 1):
        p_prev = positions[i-1]
        p_curr = positions[i]
        p_next = positions[i+1]
        
        e1 = p_curr - p_prev
        e2 = p_next - p_curr
        cross_val = np.cross(e1, e2)
        area2 = np.linalg.norm(cross_val)
        
        len1 = np.linalg.norm(e1)
        len2 = np.linalg.norm(e2)
        len3 = np.linalg.norm(p_next - p_prev)
        
        denom = len1 * len2 * len3
        if denom > 1e-6:
            curvatures[i] = 2.0 * area2 / denom
    return curvatures


def smooth_transition_path(
    model,
    path_poses: list[tuple[np.ndarray, np.ndarray]],
    box: trimesh.primitives.Box,
    fixture_mesh: trimesh.Trimesh | None,
    config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, str, str, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, float]]:
    """Smooths a transition path returned by plan_safe_transition using the hierarchy."""
    from time import perf_counter
    
    t_fit = 0.0
    t_val = 0.0
    t_ori = 0.0
    t_res = 0.0
    
    deduped: list[tuple[np.ndarray, np.ndarray]] = []
    for pos, quat in path_poses:
        pos_arr = np.asarray(pos, dtype=float)
        quat_arr = np.asarray(quat, dtype=float)
        if deduped and np.linalg.norm(pos_arr - deduped[-1][0]) < 1.0e-6:
            deduped[-1] = (deduped[-1][0], quat_arr)
        else:
            deduped.append((pos_arr, quat_arr))

    if len(deduped) == 1:
        deduped.append((deduped[0][0].copy(), deduped[0][1].copy()))

    positions = np.array([p[0] for p in deduped])
    q_start = deduped[0][1]
    q_goal = deduped[-1][1]
    
    N_wps = len(positions)
    
    original_polyline = [p for p in positions]
    rejected_candidates = []
    control_points = []
    
    T_start = positions[1] - positions[0]
    T_start_len = np.linalg.norm(T_start)
    if T_start_len > 1e-5:
        T_start /= T_start_len
    else:
        T_start = np.array([0.0, 0.0, 1.0])
        
    T_end = positions[-1] - positions[-2]
    T_end_len = np.linalg.norm(T_end)
    if T_end_len > 1e-5:
        T_end /= T_end_len
    else:
        T_end = np.array([0.0, 0.0, 1.0])
        
    mesh = model.validation_mesh.mesh
    req_clearance = config.safety_clearance_mm + config.tool_envelope_radius_mm

    def proximity_distances(points: np.ndarray) -> np.ndarray:
        _, distances, _ = trimesh.proximity.closest_point(mesh, points)
        if len(distances) != len(points):
            fill = float(distances[-1]) if len(distances) else float("inf")
            return np.full(len(points), fill, dtype=float)
        return np.asarray(distances, dtype=float)
    
    t_orig_clear_start = perf_counter()
    orig_clearances = []
    for pos in positions:
        pt_key = tuple(np.round(pos, 3))
        if pt_key in _PROXIMITY_CACHE:
            orig_clearances.append(_PROXIMITY_CACHE[pt_key])
        else:
            d_val = float(proximity_distances(np.array([pos]))[0])
            _PROXIMITY_CACHE[pt_key] = d_val
            orig_clearances.append(d_val)
    t_val += perf_counter() - t_orig_clear_start
            
    def validate_candidate_curve(curve_pts: np.ndarray) -> bool:
        nonlocal t_val
        t_start_val = perf_counter()
        if len(curve_pts) < 2:
            t_val += perf_counter() - t_start_val
            return False
        dists = proximity_distances(curve_pts)
        if np.any(dists <= 0.0):
            t_val += perf_counter() - t_start_val
            return False
            
        if fixture_mesh is not None:
            for k in range(len(curve_pts) - 1):
                if check_fixture_collision(fixture_mesh, curve_pts[k], curve_pts[k+1]):
                    t_val += perf_counter() - t_start_val
                    return False
                    
        for k in range(len(curve_pts) - 1):
            if line_collision_with_mesh(mesh, curve_pts[k], curve_pts[k+1]):
                t_val += perf_counter() - t_start_val
                return False
            if fixture_mesh is not None and check_fixture_collision(fixture_mesh, curve_pts[k], curve_pts[k+1]):
                t_val += perf_counter() - t_start_val
                return False

        for k, pt in enumerate(curve_pts):
            dists_to_orig = np.linalg.norm(positions - pt, axis=1)
            closest_idx = np.argmin(dists_to_orig)
            closest_orig_pt = positions[closest_idx]
            
            if is_point_inside_box(pt, box, margin=0.0):
                if not is_point_inside_box(closest_orig_pt, box, margin=0.0):
                    t_val += perf_counter() - t_start_val
                    return False
                    
            if dists[k] < (req_clearance - 1.0):
                t_val += perf_counter() - t_start_val
                return False
                
            min_dev = np.min(dists_to_orig)
            if min_dev > config.maximum_route_deviation_mm:
                t_val += perf_counter() - t_start_val
                return False
                
        t_val += perf_counter() - t_start_val
        return True

    def dense_polyline(points: np.ndarray) -> np.ndarray:
        dense = [points[0]]
        spacing = min(config.max_position_step_mm, config.sample_spacing_mm)
        spacing = spacing if spacing > 0.0 else config.max_position_step_mm
        for idx in range(len(points) - 1):
            p0 = points[idx]
            p1 = points[idx + 1]
            dist = float(np.linalg.norm(p1 - p0))
            count = max(1, int(np.ceil(dist / spacing)))
            for step in range(1, count + 1):
                dense.append((1.0 - step / count) * p0 + (step / count) * p1)
        return np.asarray(dense)

    def finish_candidate(
        cand_positions: np.ndarray,
        status: str,
        method: str,
        controls: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, str, str, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, float]]:
        nonlocal t_ori, t_res
        cand_positions = dense_polyline(cand_positions)
        t_ori_start = perf_counter()
        num_samples = len(cand_positions)
        arc_lengths = np.zeros(num_samples)
        for idx in range(1, num_samples):
            arc_lengths[idx] = arc_lengths[idx-1] + np.linalg.norm(cand_positions[idx] - cand_positions[idx-1])
        cand_orientations = slerp_trajectory_orientations(q_start, q_goal, arc_lengths, config)
        t_ori += perf_counter() - t_ori_start

        t_res_start = perf_counter()
        d_vals = proximity_distances(cand_positions)
        selected_idxs = adaptive_resample(cand_positions, cand_orientations, d_vals, config)
        t_res += perf_counter() - t_res_start

        return (
            cand_positions[selected_idxs],
            cand_orientations[selected_idxs],
            status,
            method,
            original_polyline,
            rejected_candidates,
            controls,
            {"fitting": t_fit, "validation": t_val, "orientation": t_ori, "resampling": t_res}
        )

    # 1. Catmull-Rom / B-Spline Spline Fitting
    if config.free_travel_smoothing_enabled and config.free_travel_smoothing_method in {"catmull_rom", "b_spline"} and N_wps >= 4:
        t_fit_start = perf_counter()
        P_start_virtual = positions[0] - T_start * np.linalg.norm(positions[1] - positions[0])
        P_end_virtual = positions[-1] + T_end * np.linalg.norm(positions[-1] - positions[-2])
        
        control_pts = np.vstack(([P_start_virtual], positions, [P_end_virtual]))
        tot_len = sum(np.linalg.norm(positions[k+1] - positions[k]) for k in range(N_wps - 1))
        spacing = max(config.sample_spacing_mm * 0.5, 1.0)
        num_samples = max(20, int(np.ceil(tot_len / spacing)) + 1)
        
        cand_positions = catmull_rom_spline(control_pts, num_samples, alpha=0.5)
        t_fit += perf_counter() - t_fit_start
        
        if validate_candidate_curve(cand_positions):
            dense_positions = dense_polyline(cand_positions)
            if not validate_candidate_curve(dense_positions):
                rejected_candidates.append(dense_positions)
            else:
                return finish_candidate(
                    dense_positions,
                    "smoothed",
                    "catmull_rom",
                    [pt for pt in positions],
                )
        else:
            rejected_candidates.append(cand_positions)

    # 2. Cubic Bézier Curve
    if config.free_travel_smoothing_enabled and N_wps >= 3:
        t_fit_start = perf_counter()
        L1 = np.linalg.norm(positions[1] - positions[0])
        L2 = np.linalg.norm(positions[-1] - positions[-2])
        C0 = positions[0]
        C1 = positions[0] + T_start * L1
        C2 = positions[-1] - T_end * L2
        C3 = positions[-1]
        
        tot_len = sum(np.linalg.norm(positions[k+1] - positions[k]) for k in range(N_wps - 1))
        spacing = max(config.sample_spacing_mm * 0.5, 1.0)
        num_samples = max(20, int(np.ceil(tot_len / spacing)) + 1)
        
        cand_positions = cubic_bezier_curve(C0, C1, C2, C3, num_samples)
        t_fit += perf_counter() - t_fit_start
            
        if validate_candidate_curve(cand_positions):
            dense_positions = dense_polyline(cand_positions)
            if not validate_candidate_curve(dense_positions):
                rejected_candidates.append(dense_positions)
            else:
                return finish_candidate(
                    dense_positions,
                    "smoothed",
                    "bezier",
                    [C0, C1, C2, C3],
                )
        else:
            rejected_candidates.append(cand_positions)

    # 3. Circular Fillet Corner Blending
    if config.free_travel_smoothing_enabled and N_wps >= 3:
        t_fit_start = perf_counter()
        filleted_wps = list(positions)
        blended_any = False
        partially_failed = False
        
        res_pts = [positions[0]]
        
        for i in range(1, N_wps - 1):
            p_prev = positions[i-1]
            p_curr = positions[i]
            p_next = positions[i+1]
            
            blend_res = None
            nominal_R = config.nominal_blend_radius_mm
            
            for retry in range(config.maximum_retries):
                res = circular_fillet_blend(p_prev, p_curr, p_next, nominal_R)
                if res is not None:
                    arc_pts, t1, t2, center = res
                    if validate_candidate_curve(arc_pts):
                        blend_res = res
                        break
                    else:
                        nominal_R = max(config.minimum_blend_radius_mm, nominal_R - (nominal_R - config.minimum_blend_radius_mm) / (config.maximum_retries - retry))
                        if nominal_R <= config.minimum_blend_radius_mm:
                            break
                            
            if blend_res is not None:
                arc_pts, t1, t2, center = blend_res
                res_pts.pop()
                res_pts.append(t1)
                res_pts.extend(arc_pts)
                res_pts.append(t2)
                blended_any = True
                control_points.append(p_curr)
            else:
                partially_failed = True
                res_pts.append(p_curr)
                
        res_pts.append(positions[-1])
        t_fit += perf_counter() - t_fit_start
        
        if blended_any:
            cand_positions = np.array(res_pts)
            num_samples = len(cand_positions)
            
            if validate_candidate_curve(cand_positions):
                status = "partially_smoothed" if partially_failed else "smoothed"
                dense_positions = dense_polyline(cand_positions)
                if validate_candidate_curve(dense_positions):
                    return finish_candidate(
                        dense_positions,
                        status,
                        "fillet",
                        control_points,
                    )
                rejected_candidates.append(dense_positions)

    # 4. Fallback D: Original Polyline
    t_fit_start = perf_counter()
    num_samples = len(positions)
    arc_lengths = np.zeros(num_samples)
    for k in range(1, num_samples):
        arc_lengths[k] = arc_lengths[k-1] + np.linalg.norm(positions[k] - positions[k-1])
    t_fit += perf_counter() - t_fit_start
        
    fallback_status = "original_safe_polyline" if validate_candidate_curve(positions) else "failed"
    return finish_candidate(dense_polyline(positions), fallback_status, "none", [])
