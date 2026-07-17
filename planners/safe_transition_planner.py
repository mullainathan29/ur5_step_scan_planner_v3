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
    
    margin = getattr(config, "forbidden_region_margin_mm", 15.0)
    tool_rad = getattr(config, "free_travel_tool_envelope_radius_mm", getattr(config, "tool_radius_mm", 35.0))
    inflation = tool_rad + margin

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

    # 1. Bounding box penetration check.
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
    
    margin = getattr(config, "forbidden_region_margin_mm", 15.0)
    tool_rad = getattr(config, "free_travel_tool_envelope_radius_mm", getattr(config, "tool_radius_mm", 35.0))
    req_clearance = tool_rad + margin
        
    spacing = getattr(config, "collision_sample_spacing_mm", None)
    if spacing is None:
        spacing = getattr(config, "collision_sampling_interval_mm", 5.0)

    if dist > 1e-6:
        num_samples = max(2, int(np.ceil(dist / spacing)))
        ts_clear = np.linspace(0.0, 1.0, num_samples)
        pts = np.array([(1.0 - t) * u + t * v for t in ts_clear])
        
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
        if np.any(distances < (req_clearance - 1.0)):
            _COLLISION_CACHE[cache_key] = (False, min_clear)
            if return_clearance:
                return False, min_clear
            return False
            
        if fixture_mesh is not None:
            # Check proximity to fixture
            f_uncached_pts = []
            f_uncached_idxs = []
            f_distances = [float("inf")] * len(pts)
            for idx, pt in enumerate(pts):
                f_uncached_pts.append(pt)
                f_uncached_idxs.append(idx)
            if f_uncached_pts:
                _, f_query_dists, _ = trimesh.proximity.closest_point(fixture_mesh, np.array(f_uncached_pts))
                if len(f_query_dists) != len(f_uncached_pts):
                    fill = float(f_query_dists[-1]) if len(f_query_dists) else float("inf")
                    f_query_dists = np.full(len(f_uncached_pts), fill, dtype=float)
                for idx, q_d in zip(f_uncached_idxs, f_query_dists):
                    f_distances[idx] = float(q_d)
            f_distances = np.array(f_distances)
            if np.any(f_distances < (req_clearance - 1.0)):
                _COLLISION_CACHE[cache_key] = (False, min_clear)
                if return_clearance:
                    return False, min_clear
                return False
    else:
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


def segment_intersects_aabb(u: np.ndarray, v: np.ndarray, half: np.ndarray) -> bool:
    """Check if line segment u-v intersects axis-aligned bounding box [-half, half]."""
    d = v - u
    t_min = 0.0
    t_max = 1.0
    for i in range(3):
        if abs(d[i]) < 1e-9:
            if abs(u[i]) > half[i]:
                return False
        else:
            t1 = (-half[i] - u[i]) / d[i]
            t2 = (half[i] - u[i]) / d[i]
            t_entry = min(t1, t2)
            t_exit = max(t1, t2)
            t_min = max(t_min, t_entry)
            t_max = min(t_max, t_exit)
            if t_min > t_max:
                return False
    return t_min <= t_max


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
    
    # 1. Geometrically calculate retract / approach points
    # Retract direction: position - surface_point
    dir_retract = p_start - np.asarray(pose_start.surface_point)
    dir_retract_norm = np.linalg.norm(dir_retract)
    if dir_retract_norm > 1e-5:
        dir_retract = dir_retract / dir_retract_norm
    else:
        dir_retract = np.asarray(pose_start.surface_normal)
        dir_retract /= np.linalg.norm(dir_retract)
        
    # Approach direction
    dir_approach = p_goal - np.asarray(pose_goal.surface_point)
    dir_approach_norm = np.linalg.norm(dir_approach)
    if dir_approach_norm > 1e-5:
        dir_approach = dir_approach / dir_approach_norm
    else:
        dir_approach = np.asarray(pose_goal.surface_normal)
        dir_approach /= np.linalg.norm(dir_approach)

    # Check if endpoints are inside the forbidden box
    # If they are inside, we retract/approach to exit it with a small margin of 5.0 mm
    start_inside = is_point_inside_box(p_start, box, margin=0.0)
    goal_inside = is_point_inside_box(p_goal, box, margin=0.0)

    if start_inside:
        d_retract = calculate_exit_point(box, p_start, dir_retract)
        p_retract = p_start + (d_retract + 5.0) * dir_retract
    else:
        p_retract = p_start.copy()

    if goal_inside:
        d_approach = calculate_exit_point(box, p_goal, dir_approach)
        p_approach = p_goal + (d_approach + 5.0) * dir_approach
    else:
        p_approach = p_goal.copy()

    # Strategy A: Direct Route first
    # Check if the segment from p_retract to p_approach is safe
    if is_segment_safe(model, p_retract, p_approach, box, fixture_mesh, config):
        path = [(p_start, q_start)]
        if start_inside:
            path.append((p_retract, q_start))
        if goal_inside:
            path.append((p_approach, q_goal))
        path.append((p_goal, q_goal))
        return "direct", path, []
    else:
        rejected_direct.append((p_retract, p_approach))

    # Strategy B: Shortest route around the box (detour search in local frame)
    inv_t = np.linalg.inv(box.transform)
    u_local = trimesh.transform_points(np.array([p_retract]), inv_t)[0]
    v_local = trimesh.transform_points(np.array([p_approach]), inv_t)[0]
    half = 0.5 * box.extents
    p_mid = 0.5 * (u_local + v_local)
    
    candidates = []
    
    def is_local_path_obb_safe(wps: list[np.ndarray]) -> bool:
        for idx in range(len(wps) - 1):
            if segment_intersects_aabb(wps[idx], wps[idx+1], half):
                return False
        return True
        
    def local_path_length(wps: list[np.ndarray]) -> float:
        return sum(np.linalg.norm(wps[idx+1] - wps[idx]) for idx in range(len(wps) - 1))

    # X, Y, Z face detour offset: 5.0 mm outside the box
    # 1. One-point detours (6 faces)
    for i in range(3):
        for s in [-1.0, 1.0]:
            p1 = p_mid.copy()
            p1[i] = s * (half[i] + 5.0)
            # Clip other coordinates to the segment's box bounds to keep it compact
            for k in range(3):
                if k != i:
                    p1[k] = np.clip(p1[k], min(u_local[k], v_local[k]), max(u_local[k], v_local[k]))
            wps = [u_local, p1, v_local]
            if is_local_path_obb_safe(wps):
                candidates.append((local_path_length(wps), "one_point", wps))
                
    # 2. Two-point detours (same face)
    for i in range(3):
        for s in [-1.0, 1.0]:
            w1 = u_local.copy()
            w1[i] = s * (half[i] + 5.0)
            w2 = v_local.copy()
            w2[i] = s * (half[i] + 5.0)
            wps = [u_local, w1, w2, v_local]
            if is_local_path_obb_safe(wps):
                candidates.append((local_path_length(wps), "two_point_same", wps))
                
    # 3. Two-point detours (different faces)
    for i1 in range(3):
        for s1 in [-1.0, 1.0]:
            for i2 in range(3):
                for s2 in [-1.0, 1.0]:
                    if i1 == i2 and s1 == s2:
                        continue
                    w1 = u_local.copy()
                    w1[i1] = s1 * (half[i1] + 5.0)
                    w2 = v_local.copy()
                    w2[i2] = s2 * (half[i2] + 5.0)
                    wps = [u_local, w1, w2, v_local]
                    if is_local_path_obb_safe(wps):
                        candidates.append((local_path_length(wps), "two_point_diff", wps))
                        
    # Sort by path length
    candidates.sort(key=lambda x: x[0])
    
    best_path = None
    best_strategy = "failed"
    
    for length, strategy_type, wps_local in candidates:
        wps_global = trimesh.transform_points(np.array(wps_local), box.transform)
        
        # Verify if the global segments are safe
        safe = True
        for idx in range(len(wps_global) - 1):
            if not is_segment_safe(model, wps_global[idx], wps_global[idx+1], box, fixture_mesh, config):
                safe = False
                break
        if safe:
            best_path = wps_global
            best_strategy = "obb_route" if strategy_type.startswith("two_point") else "direct"
            break
            
    if best_path is not None:
        path = [(p_start, q_start)]
        if start_inside:
            path.append((p_retract, q_start))
        for pt in best_path[1:-1]:
            path.append((pt, q_goal))
        if goal_inside:
            path.append((p_approach, q_goal))
        path.append((p_goal, q_goal))
        return best_strategy, path, rejected_direct

    # Fallback to direct path as fail-safe
    path = [(p_start, q_start)]
    if start_inside:
        path.append((p_retract, q_start))
    if goal_inside:
        path.append((p_approach, q_goal))
    path.append((p_goal, q_goal))
    return "failed", path, rejected_direct


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


def subdivide_polyline(positions: np.ndarray, max_segment_len: float) -> np.ndarray:
    subdivided = [positions[0]]
    for i in range(len(positions) - 1):
        p0 = positions[i]
        p1 = positions[i+1]
        dist = np.linalg.norm(p1 - p0)
        if dist > max_segment_len:
            count = int(np.ceil(dist / max_segment_len))
            for k in range(1, count):
                subdivided.append((1.0 - k / count) * p0 + (k / count) * p1)
        subdivided.append(p1)
    return np.array(subdivided)


def quintic_bezier_curve(
    p_start: np.ndarray,
    dir_start: np.ndarray,
    p_goal: np.ndarray,
    dir_end: np.ndarray,
    num_samples: int,
    w: float = 0.35
) -> np.ndarray:
    """Generate a quintic Bezier curve with zero second derivative at start and end."""
    L = np.linalg.norm(p_goal - p_start)
    T_start = dir_start * (L * w)
    T_end = dir_end * (L * w)
    
    P0 = p_start
    P1 = P0 + T_start / 5.0
    P2 = P0 + 2.0 * T_start / 5.0
    
    P5 = p_goal
    P4 = P5 - T_end / 5.0
    P3 = P5 - 2.0 * T_end / 5.0
    
    t = np.linspace(0.0, 1.0, num_samples)[:, np.newaxis]
    curve = (1-t)**5 * P0 + \
            5 * (1-t)**4 * t * P1 + \
            10 * (1-t)**3 * t**2 * P2 + \
            10 * (1-t)**2 * t**3 * P3 + \
            5 * (1-t) * t**4 * P4 + \
            t**5 * P5
    return curve


def fit_catmull_rom_spline(
    wps: np.ndarray,
    dir_start: np.ndarray,
    dir_end: np.ndarray,
    num_samples: int,
    w: float = 0.35
) -> np.ndarray:
    """Fit a centripetal Catmull-Rom spline with virtual end control points for tangent matching."""
    L_start = np.linalg.norm(wps[1] - wps[0])
    L_end = np.linalg.norm(wps[-1] - wps[-2])
    
    P_start_virtual = wps[0] - dir_start * (L_start * w)
    P_end_virtual = wps[-1] + dir_end * (L_end * w)
    
    control_pts = np.vstack(([P_start_virtual], wps, [P_end_virtual]))
    return catmull_rom_spline(control_pts, num_samples, alpha=0.5)


def smooth_transition_path(
    model,
    path_poses: list[tuple[np.ndarray, np.ndarray]],
    box: trimesh.primitives.Box,
    fixture_mesh: trimesh.Trimesh | None,
    config: PipelineConfig
) -> tuple[np.ndarray, np.ndarray, str, str, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, float], str]:
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
    
    dir_start = positions[1] - positions[0]
    dir_start_len = np.linalg.norm(dir_start)
    if dir_start_len > 1e-5:
        dir_start /= dir_start_len
    else:
        dir_start = np.array([0.0, 0.0, 1.0])
        
    dir_end = positions[-1] - positions[-2]
    dir_end_len = np.linalg.norm(dir_end)
    if dir_end_len > 1e-5:
        dir_end /= dir_end_len
    else:
        dir_end = np.array([0.0, 0.0, 1.0])
        
    mesh = model.validation_mesh.mesh
    last_fail_reason = "none"

    def proximity_distances(points: np.ndarray) -> np.ndarray:
        _, distances, _ = trimesh.proximity.closest_point(mesh, points)
        if len(distances) != len(points):
            fill = float(distances[-1]) if len(distances) else float("inf")
            return np.full(len(points), fill, dtype=float)
        return np.asarray(distances, dtype=float)

    def validate_candidate_curve(curve_pts: np.ndarray) -> bool:
        nonlocal t_val, last_fail_reason
        t_start_val = perf_counter()
        if len(curve_pts) < 2:
            last_fail_reason = "too few points"
            t_val += perf_counter() - t_start_val
            return False

        # 1. Check maximum curvature limit
        curvatures = compute_curvatures(curve_pts)
        if len(curvatures) > 0:
            max_curv = float(np.max(curvatures))
            if max_curv > config.maximum_curvature_1_per_mm:
                last_fail_reason = "curvature limit exceeded"
                t_val += perf_counter() - t_start_val
                return False

        # 2. Proximity check
        dists = proximity_distances(curve_pts)
        req_clearance = config.free_travel_tool_envelope_radius_mm + config.forbidden_region_margin_mm
        if np.any(dists < (req_clearance - 1.0)):
            last_fail_reason = "CAD proximity violation"
            t_val += perf_counter() - t_start_val
            return False
            
        # 3. Collision checks
        for k in range(len(curve_pts) - 1):
            if line_collision_with_mesh(mesh, curve_pts[k], curve_pts[k+1]):
                last_fail_reason = "CAD collision"
                t_val += perf_counter() - t_start_val
                return False
                
        # 4. Fixture checks
        if fixture_mesh is not None:
            _, f_dists, _ = trimesh.proximity.closest_point(fixture_mesh, curve_pts)
            if np.any(f_dists < (req_clearance - 1.0)):
                last_fail_reason = "fixture proximity violation"
                t_val += perf_counter() - t_start_val
                return False
            for k in range(len(curve_pts) - 1):
                if check_fixture_collision(fixture_mesh, curve_pts[k], curve_pts[k+1]):
                    last_fail_reason = "fixture collision"
                    t_val += perf_counter() - t_start_val
                    return False

        t_val += perf_counter() - t_start_val
        return True

    def finish_candidate(
        cand_positions: np.ndarray,
        status: str,
        method: str,
        controls: list[np.ndarray],
        reason_str: str = "none"
    ) -> tuple[np.ndarray, np.ndarray, str, str, list[np.ndarray], list[np.ndarray], list[np.ndarray], dict[str, float], str]:
        nonlocal t_ori, t_res
        
        # Dense spacing
        dense = [cand_positions[0]]
        spacing = config.sample_spacing_mm
        for idx in range(len(cand_positions) - 1):
            p0 = cand_positions[idx]
            p1 = cand_positions[idx + 1]
            dist = float(np.linalg.norm(p1 - p0))
            count = max(1, int(np.ceil(dist / spacing)))
            for step in range(1, count + 1):
                dense.append((1.0 - step / count) * p0 + (step / count) * p1)
        cand_positions = np.asarray(dense)
        
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
            {"fitting": t_fit, "validation": t_val, "orientation": t_ori, "resampling": t_res},
            reason_str
        )

    # Calculate total length
    tot_len = sum(np.linalg.norm(positions[k+1] - positions[k]) for k in range(N_wps - 1))
    spacing = max(config.sample_spacing_mm * 0.5, 1.0)
    num_samples = max(20, int(np.ceil(tot_len / spacing)) + 1)

    # 1. Quintic Bezier for N_wps == 2 (Direct transition)
    if config.free_travel_smoothing_enabled and N_wps == 2:
        t_fit_start = perf_counter()
        for w in [0.35, 0.25, 0.15, 0.05]:
            cand_positions = quintic_bezier_curve(positions[0], dir_start, positions[-1], dir_end, num_samples, w)
            if validate_candidate_curve(cand_positions):
                t_fit += perf_counter() - t_fit_start
                return finish_candidate(
                    cand_positions,
                    "smoothed",
                    "bezier",
                    [positions[0], positions[-1]],
                    "none"
                )
            else:
                rejected_candidates.append(cand_positions)
        t_fit += perf_counter() - t_fit_start

    # 2. Catmull-Rom Spline for N_wps >= 3 (Detour transition)
    if config.free_travel_smoothing_enabled and N_wps >= 3:
        t_fit_start = perf_counter()
        for w in [0.35, 0.25, 0.15, 0.05]:
            cand_positions = fit_catmull_rom_spline(positions, dir_start, dir_end, num_samples, w)
            if validate_candidate_curve(cand_positions):
                t_fit += perf_counter() - t_fit_start
                return finish_candidate(
                    cand_positions,
                    "smoothed",
                    "catmull_rom",
                    [pt for pt in positions],
                    "none"
                )
            else:
                rejected_candidates.append(cand_positions)
        t_fit += perf_counter() - t_fit_start

    # 3. Fallback: raw polyline
    t_fit_start = perf_counter()
    dense_positions = [positions[0]]
    for idx in range(len(positions) - 1):
        p0 = positions[idx]
        p1 = positions[idx + 1]
        dist = float(np.linalg.norm(p1 - p0))
        count = max(1, int(np.ceil(dist / spacing)))
        for step in range(1, count + 1):
            dense_positions.append((1.0 - step / count) * p0 + (step / count) * p1)
    dense_positions = np.asarray(dense_positions)
    t_fit += perf_counter() - t_fit_start
    
    fallback_status = "original_safe_polyline"
    return finish_candidate(
        dense_positions,
        fallback_status,
        "none",
        [pt for pt in positions],
        last_fail_reason
    )
