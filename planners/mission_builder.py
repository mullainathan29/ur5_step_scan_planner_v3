"""Single continuous Cartesian mission builder with safe transition routing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import trimesh

from app.config import PipelineConfig
from app.result import PipelineResult
from planners.base import MotionType, Pose
from planners.safe_transition_planner import (
    construct_safety_envelope, plan_safe_transition,
    clear_collision_caches, smooth_transition_path, compute_curvatures,
    slerp_quaternions,
)


def get_scan_segments(result: PipelineResult) -> list[tuple[Pose, ...]]:
    """Extract individual ordered scan segments (contiguous runs of scan_enabled poses)."""
    segments = []
    if result.sweeps:
        for sweep in result.sweeps:
            scan = tuple(p for p in sweep.poses if p.scan_enabled)
            if scan:
                scan = tuple(replace(
                    p,
                    source_sweep_id=sweep.sweep_id,
                    source_sweep_name=sweep.sweep_name,
                    source_feature_id=p.feature_id,
                    scan_status=-1,
                    scan_enabled=True
                ) for p in scan)
                segments.append(scan)
    else:
        current = []
        for p in result.poses:
            if p.scan_enabled:
                current.append(replace(
                    p,
                    source_sweep_id=p.sweep_id,
                    source_sweep_name=p.sweep_name,
                    source_feature_id=p.feature_id,
                    scan_status=-1,
                    scan_enabled=True
                ))
            else:
                if current:
                    segments.append(tuple(current))
                    current = []
        if current:
            segments.append(tuple(current))
    return segments


def build_continuous_mission(result: PipelineResult, config: PipelineConfig) -> tuple[Pose, ...]:
    """Connect ordered scan segments into a single flat continuous Cartesian mission using safe routing and smooth curve transitions."""
    from time import perf_counter

    t_start = perf_counter()
    clear_collision_caches()

    t_poly = 0.0
    t_fit = 0.0
    t_val = 0.0
    t_ori = 0.0
    t_res = 0.0

    scan_segments = get_scan_segments(result)
    if not scan_segments:
        return ()

    # Load fixture if enabled
    fixture_mesh = None
    if config.fixture_exclusion_enabled and config.fixture_mesh_path:
        if Path(config.fixture_mesh_path).exists():
            fixture_mesh = trimesh.load(config.fixture_mesh_path)

    mesh = result.model.validation_mesh.mesh
    bounds_min, bounds_max = mesh.bounds
    z_safe = float(bounds_max[2]) + config.stand_off_mm + config.bottom_clearance_mm

    # Compute safety envelope
    safety_box = construct_safety_envelope(result.model, config)
    result.summary.output_files["safety_box"] = safety_box

    # Home/End position center computation
    centre = 0.5 * (bounds_min + bounds_max)
    home_pos = np.array([centre[0], centre[1], z_safe])

    # Mission start pose
    first_scan_pose = scan_segments[0][0]
    q_home = np.asarray(first_scan_pose.quaternion_xyzw)
    home_pose = replace(
        first_scan_pose,
        segment_id=-1,
        pose_id=0,
        position=home_pos,
        surface_point=centre,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        quaternion_xyzw=tuple(float(v) for v in q_home),
        motion_type=MotionType.MISSION_START,
        scan_status=0,
        scan_enabled=False,
        transition_id=0,
        transition_strategy="direct",
        collision_checked=True,
        envelope_type=config.envelope_type
    )

    flat_poses: list[Pose] = []
    flat_poses.append(home_pose)

    rejected_direct_lines = []
    strategy_counts = {
        "direct": 0,
        "retract_direct_approach": 0,
        "obb_route": 0,
        "aabb_route": 0,
        "top_clearance": 0,
        "bottom_clearance": 0,
        "failed": 0
    }

    # Smoothing status counters
    smoothing_counts = {
        "smoothed": 0,
        "partially_smoothed": 0,
        "original_safe_polyline": 0,
        "failed": 0
    }

    # Store viewer visualization data
    original_safe_polylines = []
    smoothing_control_points = []
    rejected_smoothing_candidates = []

    total_free_travel_dist_before = 0.0
    total_free_travel_dist_after = 0.0
    max_free_travel_curvature = 0.0
    min_transition_clearance = float("inf")

    transition_diagnostics = []

    def _append_free_transition(
        pose_from: Pose,
        pose_to: Pose,
        transition_id: int,
        final_motion_type: MotionType,
    ) -> str:
        nonlocal t_poly, t_fit, t_val, t_ori, t_res
        nonlocal total_free_travel_dist_before, total_free_travel_dist_after
        nonlocal max_free_travel_curvature

        t0 = perf_counter()
        strategy, path, rejected = plan_safe_transition(
            result.model,
            pose_from,
            pose_to,
            safety_box,
            fixture_mesh,
            config
        )
        t_poly += perf_counter() - t0

        rejected_direct_lines.extend(rejected)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        (
            smooth_pos,
            smooth_ori,
            status,
            method,
            orig_poly,
            rej_cands,
            ctrl_pts,
            sub_timings,
            fallback_reason
        ) = smooth_transition_path(result.model, path, safety_box, fixture_mesh, config)
        t_fit += sub_timings["fitting"]
        t_val += sub_timings["validation"]
        t_ori += sub_timings["orientation"]
        t_res += sub_timings["resampling"]

        smoothing_counts[status] = smoothing_counts.get(status, 0) + 1
        original_safe_polylines.append(orig_poly)
        if len(ctrl_pts) > 0:
            smoothing_control_points.extend(ctrl_pts)
        rejected_smoothing_candidates.extend(rej_cands)

        d_before = sum(np.linalg.norm(orig_poly[k+1] - orig_poly[k]) for k in range(len(orig_poly) - 1))
        d_after = sum(np.linalg.norm(smooth_pos[k+1] - smooth_pos[k]) for k in range(len(smooth_pos) - 1))
        total_free_travel_dist_before += d_before
        total_free_travel_dist_after += d_after

        curvatures = compute_curvatures(smooth_pos)
        if len(curvatures) > 0:
            max_free_travel_curvature = max(max_free_travel_curvature, float(np.max(curvatures)))

        # Collect diagnostics
        _, dists, _ = trimesh.proximity.closest_point(result.model.validation_mesh.mesh, smooth_pos)
        min_clear = float(np.min(dists)) if len(dists) > 0 else 0.0
        max_curv = float(np.max(curvatures)) if len(curvatures) > 0 else 0.0
        
        transition_diagnostics.append({
            "transition_id": transition_id,
            "direct_accepted_or_blocked": "accepted" if strategy == "direct" else "blocked",
            "detour_point_count": len(orig_poly) - 2 if len(orig_poly) > 2 else 0,
            "curve_type": method,
            "path_length": float(d_after),
            "min_forbidden_dist": min_clear,
            "max_curvature": max_curv,
            "pose_count": len(smooth_pos),
            "fallback_reason": fallback_reason
        })

        M = len(smooth_pos)
        for k in range(1, M):
            pos = smooth_pos[k]
            q = smooth_ori[k]

            if k == M - 1:
                m_type = final_motion_type
            else:
                if k == 1:
                    m_type = MotionType.RETRACT
                elif k == M - 2:
                    m_type = MotionType.APPROACH
                else:
                    m_type = MotionType.TRANSITION
                    
            pose_k = replace(
                pose_to,
                position=pos,
                quaternion_xyzw=tuple(float(v) for v in q),
                motion_type=m_type,
                scan_status=0,
                scan_enabled=False,
                transition_id=transition_id,
                transition_strategy=strategy,
                smoothing_method=method,
                smoothing_status=status,
                local_curvature=float(curvatures[k]),
                envelope_type=config.envelope_type,
                collision_checked=True
            )
            flat_poses.append(pose_k)
        return strategy

    # Connect scan segments without changing any scan poses.
    num_segments = len(scan_segments)
    for i, curr_segment in enumerate(scan_segments):
        next_scan_start = curr_segment[0]
        normal_first = np.asarray(next_scan_start.surface_normal, dtype=float)
        normal_first /= np.linalg.norm(normal_first)
        lead_in_pose = replace(
            next_scan_start,
            position=np.asarray(next_scan_start.position) + 5.0 * normal_first,
            motion_type=MotionType.LEAD_IN,
            scan_status=0,
            scan_enabled=False,
            transition_id=i + 1,
            envelope_type=config.envelope_type,
            collision_checked=True,
        )

        pose_from = flat_poses[-1]
        if i == 0:
            pose_from = replace(
                pose_from,
                surface_point=next_scan_start.surface_point,
                surface_normal=next_scan_start.surface_normal,
            )

        strategy = _append_free_transition(
            pose_from,
            lead_in_pose,
            i + 1,
            MotionType.LEAD_IN,
        )

        flat_poses.extend(
            replace(p, mission_segment_id=i, scan_status=-1, scan_enabled=True)
            for p in curr_segment
        )

        last_scan = curr_segment[-1]
        normal_last = np.asarray(last_scan.surface_normal, dtype=float)
        normal_last /= np.linalg.norm(normal_last)
        lead_out_pose = replace(
            last_scan,
            position=np.asarray(last_scan.position) + 5.0 * normal_last,
            motion_type=MotionType.LEAD_OUT,
            scan_status=0,
            scan_enabled=False,
            transition_id=i + 1,
            transition_strategy=strategy,
            smoothing_method="none",
            smoothing_status="original_safe_polyline",
            local_curvature=0.0,
            envelope_type=config.envelope_type,
            collision_checked=True,
        )
        flat_poses.append(lead_out_pose)

    # Mission end transition to home-like end retract pose.
    last_pose = flat_poses[-1]
    end_pose_target = replace(
        last_pose,
        segment_id=-1,
        position=home_pos,
        surface_point=centre,
        surface_normal=np.array([0.0, 0.0, 1.0]),
        quaternion_xyzw=last_pose.quaternion_xyzw,
        motion_type=MotionType.MISSION_END,
        scan_status=0,
        scan_enabled=False,
        transition_id=num_segments + 1,
        transition_strategy="direct",
        collision_checked=True,
        envelope_type=config.envelope_type,
    )
    _append_free_transition(
        last_pose,
        end_pose_target,
        num_segments + 1,
        MotionType.MISSION_END,
    )

    # 3. Final numbering, orientation sign alignment, and clearance calculation
    final_poses = []

    # Vectorised clearance calculation
    all_positions = np.array([p.position for p in flat_poses])
    _, all_dists, _ = trimesh.proximity.closest_point(mesh, all_positions)
    if len(all_dists) != len(flat_poses):
        all_dists = np.asarray([
            trimesh.proximity.closest_point(mesh, np.array([pose.position]))[1][0]
            for pose in flat_poses
        ], dtype=float)

    for idx, pose in enumerate(flat_poses):
        q = np.asarray(pose.quaternion_xyzw)
        if idx > 0:
            q_prev = np.asarray(final_poses[-1].quaternion_xyzw)
            if np.dot(q, q_prev) < 0.0:
                q = -q
        q = q / np.linalg.norm(q)
        
        clearance_val = float(all_dists[idx])
        if pose.scan_status == 0:
            min_transition_clearance = min(min_transition_clearance, clearance_val)

        final_poses.append(replace(
            pose,
            global_pose_id=idx,
            pose_id=idx,
            quaternion_xyzw=tuple(float(v) for v in q),
            clearance_mm=clearance_val
        ))

    t_assembly = perf_counter() - t_start - (t_poly + t_fit + t_val + t_ori + t_res)

    # Print diagnostics report
    print("\n" + "="*90)
    print("FREE-TRAVEL TRANSITION DIAGNOSTICS REPORT")
    print("="*90)
    print(f"{'ID':<4} | {'Direct/Blocked':<14} | {'Detours':<8} | {'Curve Type':<11} | {'Length (mm)':<12} | {'Min Dist (mm)':<13} | {'Max Curv':<9} | {'Poses':<5} | {'Fallback Reason'}")
    print("-"*120)
    for td in transition_diagnostics:
        print(f"{td['transition_id']:<4} | {td['direct_accepted_or_blocked']:<14} | {td['detour_point_count']:<8} | {td['curve_type']:<11} | {td['path_length']:7.1f}      | {td['min_forbidden_dist']:8.1f}      | {td['max_curvature']:8.4f}  | {td['pose_count']:<5} | {td['fallback_reason']}")
    print("="*120)
    print("SUMMARY STATISTICS:")
    direct_smooth = sum(1 for td in transition_diagnostics if td['direct_accepted_or_blocked'] == "accepted" and td['curve_type'] != "none")
    one_point_detours = sum(1 for td in transition_diagnostics if td['detour_point_count'] == 1 and td['curve_type'] != "none")
    two_point_detours = sum(1 for td in transition_diagnostics if td['detour_point_count'] == 2 and td['curve_type'] != "none")
    fallback_polylines = sum(1 for td in transition_diagnostics if td['curve_type'] == "none")
    failed_transitions = strategy_counts.get("failed", 0)
    print(f"- Direct smooth transitions: {direct_smooth}")
    print(f"- One-point detours:         {one_point_detours}")
    print(f"- Two-point detours:         {two_point_detours}")
    print(f"- Fallback polylines:        {fallback_polylines}")
    print(f"- Failed transitions:        {failed_transitions}")
    print(f"- Total free-travel distance: {total_free_travel_dist_after:.1f} mm")
    print(f"- Total runtime:             {perf_counter() - t_start:.3f} s")
    print("="*90 + "\n")

    # Cache outputs for viewer
    result.summary.output_files.update({
        "rejected_direct_lines": rejected_direct_lines,
        "original_safe_polylines": original_safe_polylines,
        "smoothing_control_points": smoothing_control_points,
        "rejected_smoothing_candidates": rejected_smoothing_candidates,
        "num_scan_blocks": len(scan_segments),
        "num_free_travel_blocks": len(scan_segments) + 1,
        "direct_transitions_accepted": strategy_counts.get("direct", 0),
        "direct_transitions_rejected": len(rejected_direct_lines),
        "obb_routes_generated": strategy_counts.get("obb_route", 0) + strategy_counts.get("aabb_route", 0),
        "top_clearance_routes_generated": strategy_counts.get("top_clearance", 0),
        "bottom_clearance_routes_generated": strategy_counts.get("bottom_clearance", 0),
        "failed_transitions": strategy_counts.get("failed", 0),
        
        # New mission smoothing stats
        "smoothed_transition_count": smoothing_counts.get("smoothed", 0),
        "partially_smoothed_transition_count": smoothing_counts.get("partially_smoothed", 0),
        "unsmoothed_safe_transition_count": smoothing_counts.get("original_safe_polyline", 0),
        "total_free_travel_distance_before": total_free_travel_dist_before,
        "total_free_travel_distance_after": total_free_travel_dist_after,
        "max_free_travel_curvature": max_free_travel_curvature,
        "minimum_transition_clearance": min_transition_clearance,
        
        # Timing breakdown payload
        "mission_timings": {
            "safe_polyline_generation": t_poly,
            "curve_fitting": t_fit,
            "curve_collision_validation": t_val,
            "orientation_interpolation": t_ori,
            "adaptive_resampling": t_res,
            "final_mission_assembly": t_assembly
        }
    })

    return tuple(final_poses)
