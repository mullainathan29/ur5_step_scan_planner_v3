"""Single continuous Cartesian mission builder with safe transition routing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from app.result import PipelineResult, PipelineSummary
from planners.base import MotionType, Pose
from planners.safe_transition_planner import (
    construct_safety_envelope, plan_safe_transition, slerp_quaternions
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
    """Connect ordered scan segments into a single flat continuous Cartesian mission using safe routing."""
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
    
    # 1. Mission start pose
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

    # Connect scan segments
    num_segments = len(scan_segments)
    for i in range(num_segments):
        curr_segment = scan_segments[i]
        prev_pose = flat_poses[-1]
        next_scan_start = curr_segment[0]
        
        # We need to plan transition from prev_pose to next_scan_start
        # Convert prev_pose to a Pose with correct surface properties if it is start pose
        if i == 0:
            pose_from = replace(
                prev_pose,
                surface_point=next_scan_start.surface_point,
                surface_normal=next_scan_start.surface_normal
            )
        else:
            pose_from = scan_segments[i-1][-1]
            
        strategy, path, rejected = plan_safe_transition(
            result.model,
            pose_from,
            next_scan_start,
            safety_box,
            fixture_mesh,
            config
        )
        
        rejected_direct_lines.extend(rejected)
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # Build path poses
        transition_poses: list[Pose] = []
        path_len = len(path)
        
        if strategy == "direct":
            # For a direct transition, path has 2 points: [start, goal]
            # We explicitly add the APPROACH pose at pos_approach
            normal_first = np.asarray(next_scan_start.surface_normal)
            normal_first /= np.linalg.norm(normal_first)
            pos_approach = np.asarray(next_scan_start.position) + config.bottom_clearance_mm * normal_first
            
            approach_pose = replace(
                next_scan_start,
                position=pos_approach,
                motion_type=MotionType.APPROACH,
                scan_status=0,
                scan_enabled=False,
                transition_id=i + 1,
                transition_strategy=strategy,
                envelope_type=config.envelope_type,
                collision_checked=True
            )
            transition_poses.append(approach_pose)
        else:
            for k in range(1, path_len - 1):
                pos, q = path[k]
                
                # Map semantic motion types
                if k == 1:
                    m_type = MotionType.RETRACT
                elif k == 2 and np.linalg.norm(pos - path[1][0]) < 1e-3:
                    # Rotate waypoint at retract point
                    m_type = MotionType.REPOSITION
                elif k == path_len - 2:
                    m_type = MotionType.APPROACH
                else:
                    m_type = MotionType.TRANSITION
                    
                pose_k = replace(
                    next_scan_start,
                    position=pos,
                    quaternion_xyzw=tuple(float(v) for v in q),
                    motion_type=m_type,
                    scan_status=0,
                    scan_enabled=False,
                    transition_id=i + 1,
                    transition_strategy=strategy,
                    envelope_type=config.envelope_type,
                    collision_checked=True
                )
                transition_poses.append(pose_k)
            
        # Add LEAD_IN pose right before first scan pose
        pos_lead_in = np.asarray(next_scan_start.position) + 5.0 * np.asarray(next_scan_start.surface_normal)
        lead_in_pose = replace(
            next_scan_start,
            position=pos_lead_in,
            motion_type=MotionType.LEAD_IN,
            scan_status=0,
            scan_enabled=False,
            transition_id=i + 1,
            transition_strategy=strategy,
            envelope_type=config.envelope_type,
            collision_checked=True
        )
        
        flat_poses.extend(transition_poses)
        flat_poses.append(lead_in_pose)
        
        # Add actual scan poses
        flat_poses.extend(replace(p, mission_segment_id=i, scan_status=-1, scan_enabled=True) for p in curr_segment)
        
        # Add lead-out pose right after scanning segment
        last_scan = curr_segment[-1]
        normal_last = np.asarray(last_scan.surface_normal)
        normal_last /= np.linalg.norm(normal_last)
        pos_lead_out = np.asarray(last_scan.position) + 5.0 * normal_last
        lead_out_pose = replace(
            last_scan,
            position=pos_lead_out,
            motion_type=MotionType.LEAD_OUT,
            scan_status=0,
            scan_enabled=False,
            transition_id=i + 1,
            transition_strategy=strategy,
            envelope_type=config.envelope_type,
            collision_checked=True
        )
        flat_poses.append(lead_out_pose)

    # 2. Mission end transition to home-like end retract pose
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
        envelope_type=config.envelope_type
    )

    strategy, path, rejected = plan_safe_transition(
        result.model,
        last_pose,
        end_pose_target,
        safety_box,
        fixture_mesh,
        config
    )
    rejected_direct_lines.extend(rejected)
    strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

    end_transition_poses = []
    path_len = len(path)
    for k in range(1, path_len - 1):
        pos, q = path[k]
        m_type = MotionType.RETRACT if k == 1 else MotionType.TRANSITION
        pose_k = replace(
            end_pose_target,
            position=pos,
            quaternion_xyzw=tuple(float(v) for v in q),
            motion_type=m_type,
            scan_status=0,
            scan_enabled=False,
            transition_id=num_segments + 1,
            transition_strategy=strategy,
            envelope_type=config.envelope_type,
            collision_checked=True
        )
        end_transition_poses.append(pose_k)
        
    flat_poses.extend(end_transition_poses)
    
    end_pose = replace(
        end_pose_target,
        position=path[-1][0],
        quaternion_xyzw=tuple(float(v) for v in path[-1][1])
    )
    flat_poses.append(end_pose)

    # Cache rejected lines in result
    result.summary.output_files["rejected_direct_lines"] = rejected_direct_lines

    # 3. Subdivision Pass: limit orientation step (5 deg) and position step (20 mm)
    subdivided = []
    if flat_poses:
        subdivided.append(flat_poses[0])
        for i in range(1, len(flat_poses)):
            p1 = subdivided[-1]
            p2 = flat_poses[i]
            if p1.scan_status == 0 or p2.scan_status == 0:
                pos1 = np.asarray(p1.position)
                pos2 = np.asarray(p2.position)
                q1 = np.asarray(p1.quaternion_xyzw)
                q2 = np.asarray(p2.quaternion_xyzw)
                
                dist = float(np.linalg.norm(pos2 - pos1))
                angle = float(np.degrees((Rotation.from_quat(q1).inv() * Rotation.from_quat(q2)).magnitude()))
                
                n_dist = int(np.ceil(dist / config.max_position_step_mm)) if config.max_position_step_mm > 0 else 1
                n_angle = int(np.ceil(angle / config.max_orientation_step_deg)) if config.max_orientation_step_deg > 0 else 1
                n = max(n_dist, n_angle)
                
                if n > 1:
                    for j in range(1, n):
                        t = j / n
                        pos_t = (1 - t) * pos1 + t * pos2
                        q_t = slerp_quaternions(q1, q2, t)
                        
                        sp1 = np.asarray(p1.surface_point)
                        sp2 = np.asarray(p2.surface_point)
                        sp_t = (1 - t) * sp1 + t * sp2
                        
                        sn1 = np.asarray(p1.surface_normal)
                        sn2 = np.asarray(p2.surface_normal)
                        sn_t = (1 - t) * sn1 + t * sn2
                        sn_t_norm = np.linalg.norm(sn_t)
                        if sn_t_norm > 1e-6:
                            sn_t /= sn_t_norm
                            
                        if p2.motion_type in {MotionType.LEAD_IN, MotionType.APPROACH}:
                            m_type = MotionType.APPROACH
                        elif p2.motion_type in {MotionType.RETRACT, MotionType.REPOSITION}:
                            m_type = MotionType.RETRACT
                        else:
                            m_type = MotionType.TRANSITION

                        interp_pose = replace(
                            p2,
                            position=pos_t,
                            quaternion_xyzw=tuple(float(v) for v in q_t),
                            surface_point=sp_t,
                            surface_normal=sn_t,
                            motion_type=m_type,
                            scan_status=0,
                            scan_enabled=False,
                            transition_id=p2.transition_id,
                            transition_strategy=p2.transition_strategy,
                            envelope_type=p2.envelope_type,
                            collision_checked=True
                        )
                        subdivided.append(interp_pose)
            subdivided.append(p2)
        flat_poses = subdivided

    # 4. Final numbering, orientation sign alignment, and clearance calculation
    final_poses = []
    for idx, pose in enumerate(flat_poses):
        q = np.asarray(pose.quaternion_xyzw)
        if idx > 0:
            q_prev = np.asarray(final_poses[-1].quaternion_xyzw)
            if np.dot(q, q_prev) < 0.0:
                q = -q
        q = q / np.linalg.norm(q)
        
        # Calculate clearance to CAD mesh
        _, dists, _ = trimesh.proximity.closest_point(mesh, np.array([pose.position]))
        clearance_val = float(dists[0])
        
        final_poses.append(replace(
            pose,
            global_pose_id=idx,
            pose_id=idx,
            quaternion_xyzw=tuple(float(v) for v in q),
            clearance_mm=clearance_val
        ))

    # Add transition statistics to result output metadata
    result.summary.output_files.update({
        "num_scan_blocks": len(scan_segments),
        "num_free_travel_blocks": len(scan_segments) + 1,
        "direct_transitions_accepted": strategy_counts.get("direct", 0),
        "direct_transitions_rejected": len(rejected_direct_lines),
        "obb_routes_generated": strategy_counts.get("obb_route", 0) + strategy_counts.get("aabb_route", 0),
        "top_clearance_routes_generated": strategy_counts.get("top_clearance", 0),
        "bottom_clearance_routes_generated": strategy_counts.get("bottom_clearance", 0),
        "failed_transitions": strategy_counts.get("failed", 0)
    })

    return tuple(final_poses)
