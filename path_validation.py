"""Post-planning continuity validation and rejection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from planners.base import PlannerSettings, ScanMotionSegment, MotionType, Pose


@dataclass(frozen=True)
class SegmentValidation:
    segment_id: int
    valid: bool
    reasons: tuple[str, ...]


def validate_scan_segment(
    motion: ScanMotionSegment,
    settings: PlannerSettings,
    *,
    expected_region_id: int,
) -> SegmentValidation:
    poses = motion.scan_poses
    reasons: list[str] = []
    if len(poses) < 2:
        reasons.append("zero-length segment")
    if any(pose.region_id != expected_region_id for pose in poses):
        reasons.append("crosses unrelated regions")
    scan_runs: list[list] = []
    current_run: list = []
    for pose in motion.poses:
        if pose.scan_enabled:
            current_run.append(pose)
        elif current_run:
            scan_runs.append(current_run)
            current_run = []
    if current_run:
        scan_runs.append(current_run)
    distances = [
        float(np.linalg.norm(second.position - first.position))
        for run in scan_runs
        for first, second in zip(run, run[1:])
    ]
    if any(distance > 1.5 * settings.travel_spacing_mm + 1.0e-7 for distance in distances):
        reasons.append("trim gap or excessive scan-point spacing")
    if poses and not all(
        300.0 <= np.linalg.norm(pose.position - pose.surface_point) <= 320.0
        for pose in poses
    ):
        reasons.append("stand-off outside 300-320 mm")
    orientation_changes = [
        float(
            np.degrees(
                (
                    Rotation.from_quat(first.quaternion_xyzw).inv()
                    * Rotation.from_quat(second.quaternion_xyzw)
                ).magnitude()
            )
        )
        for run in scan_runs
        for first, second in zip(run, run[1:])
    ]
    if any(change > 5.0 + 1.0e-7 for change in orientation_changes):
        reasons.append("orientation change exceeds 5 degrees")
    points = [tuple(np.round(pose.position, decimals=9)) for pose in poses]
    if len(points) != len(set(points)):
        reasons.append("duplicate scan poses")
    if distances and sum(distances) <= 1.0e-9:
        reasons.append("zero-length segment")
    return SegmentValidation(
        segment_id=motion.segment_id,
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )


def validate_continuous_mission(
    poses: tuple[Pose, ...],
    settings: PlannerSettings,
) -> dict[str, Any]:
    """Verify continuity, contiguity, and step-limits for the complete mission."""
    reasons: list[str] = []
    
    # 1. global_pose_id is continuous (0 to N-1)
    for idx, p in enumerate(poses):
        if p.global_pose_id != idx:
            reasons.append("global_pose_id is not continuous or incorrect starting index")
            break
            
    # 2. scan_status is always -1 or 0
    if not all(p.scan_status in {-1, 0} for p in poses):
        reasons.append("scan_status has invalid values (not -1 or 0)")
        
    # 3. scan_status and scan_enabled agree
    for p in poses:
        if (p.scan_status == -1 and not p.scan_enabled) or (p.scan_status == 0 and p.scan_enabled):
            reasons.append(f"scan_status and scan_enabled contradict at global_pose_id {p.global_pose_id}")
            break
            
    # 4. no scanning pose exists in a free-space transition
    transition_types = {MotionType.TRANSITION, MotionType.CLEARANCE, MotionType.REPOSITION, MotionType.APPROACH, MotionType.RETRACT, MotionType.LEAD_IN, MotionType.LEAD_OUT}
    for p in poses:
        if p.scan_status == -1 and p.motion_type in transition_types:
            reasons.append(f"Scanning pose exists in transition motion type at global_pose_id {p.global_pose_id}")
            break
            
    # 5. position and orientation steps remain within limits
    max_pos_step = 20.0
    max_ori_step = 5.0
    
    largest_pos_jump = 0.0
    largest_ori_jump = 0.0
    
    for k in range(len(poses) - 1):
        p1, p2 = poses[k], poses[k+1]
        dist = float(np.linalg.norm(np.asarray(p2.position) - np.asarray(p1.position)))
        angle = float(np.degrees((Rotation.from_quat(p1.quaternion_xyzw).inv() * Rotation.from_quat(p2.quaternion_xyzw)).magnitude()))
        
        largest_pos_jump = max(largest_pos_jump, dist)
        largest_ori_jump = max(largest_ori_jump, angle)
        
        if dist < 1e-3 and angle < 1e-3:
            # Zero length duplicate poses are not allowed
            reasons.append(f"Zero-length duplicate pose at global_pose_id {p2.global_pose_id}")
            
        if dist > max_pos_step + 1e-3:
            reasons.append(f"Position step {dist:.2f} mm exceeds limit of {max_pos_step} mm between {p1.global_pose_id} and {p2.global_pose_id}")
        if angle > max_ori_step + 1e-3:
            reasons.append(f"Orientation step {angle:.2f} deg exceeds limit of {max_ori_step} deg between {p1.global_pose_id} and {p2.global_pose_id}")
            
    # Count blocks and distances
    scan_blocks = []
    free_travel_blocks = []
    
    current_status = None
    current_block = []
    
    for p in poses:
        if p.scan_status != current_status:
            if current_block:
                if current_status == -1:
                    scan_blocks.append(current_block)
                else:
                    free_travel_blocks.append(current_block)
            current_status = p.scan_status
            current_block = [p]
        else:
            current_block.append(p)
    if current_block:
        if current_status == -1:
            scan_blocks.append(current_block)
        else:
            free_travel_blocks.append(current_block)
            
    total_scan_dist = 0.0
    total_free_travel_dist = 0.0
    for k in range(len(poses) - 1):
        d = float(np.linalg.norm(np.asarray(poses[k+1].position) - np.asarray(poses[k].position)))
        if poses[k].scan_status == -1 and poses[k+1].scan_status == -1:
            total_scan_dist += d
        else:
            total_free_travel_dist += d

    num_full_retracts = sum(1 for p in poses if p.motion_type == MotionType.RETRACT)
    total_transitions = max(0, len(scan_blocks) - 1)
    num_direct_transitions = max(0, total_transitions - num_full_retracts)
    
    # 6. every next scan begins after an approach/lead-in
    for block in scan_blocks:
        start_idx = block[0].global_pose_id
        if start_idx > 1:
            p_prev1 = poses[start_idx - 1]
            p_prev2 = poses[start_idx - 2]
            if p_prev1.motion_type != MotionType.LEAD_IN or p_prev2.motion_type != MotionType.APPROACH:
                reasons.append(f"Scan block starting at global_pose_id {start_idx} is not preceded by APPROACH and LEAD_IN")

    # 7. quaternion norm and sign continuity
    for idx, p in enumerate(poses):
        q = np.asarray(p.quaternion_xyzw)
        norm = np.linalg.norm(q)
        if abs(norm - 1.0) > 1e-3:
            reasons.append(f"Quaternion norm {norm:.4f} is not 1.0 at global_pose_id {p.global_pose_id}")
            break
        if idx > 0:
            q_prev = np.asarray(poses[idx - 1].quaternion_xyzw)
            if np.dot(q, q_prev) < -1e-5:
                reasons.append(f"Quaternion sign discontinuity at global_pose_id {p.global_pose_id}")
                break

    # 8. safety clearance, failed, and collision_checked validations
    if hasattr(settings, "free_travel_tool_envelope_radius_mm") or hasattr(settings, "forbidden_region_margin_mm") or hasattr(settings, "effective_minimum_clearance_mm"):
        forbidden_margin = getattr(settings, "forbidden_region_margin_mm", 15.0)
        tool_rad = getattr(settings, "free_travel_tool_envelope_radius_mm", getattr(settings, "tool_radius_mm", 35.0))
        min_req_clearance = getattr(settings, "effective_minimum_clearance_mm", tool_rad + forbidden_margin)
    else:
        # Legacy fallback
        min_req_clearance = getattr(settings, "safety_clearance_mm", 80.0) + getattr(settings, "tool_envelope_radius_mm", 60.0)
        
    max_curvature_limit = getattr(settings, "maximum_curvature_1_per_mm", 0.025)

    from collections import defaultdict
    transition_poses = defaultdict(list)
    for p in poses:
        if p.scan_status == 0 and p.transition_id >= 0:
            transition_poses[p.transition_id].append(p)
            
    for tid, tposes in sorted(transition_poses.items()):
        # Check failed strategy
        if any(p.transition_strategy == "failed" for p in tposes):
            reasons.append(f"Transition {tid} enters forbidden region.")
            continue
            
        # Check collision checked
        if any(not p.collision_checked for p in tposes):
            reasons.append(f"Transition {tid} has poses that have not been collision checked.")
            continue
            
        # Check CAD / Fixture collision or intersection
        if any(p.clearance_mm <= 0.0 for p in tposes):
            reasons.append(f"Transition {tid} intersects CAD mesh.")
            continue
            
        if any("fixture" in getattr(p, "warning_flags", "").lower() for p in tposes):
            reasons.append(f"Transition {tid} collides with fixture.")
            continue

        # Check clearance
        below_clearance_poses = [p for p in tposes if p.clearance_mm < min_req_clearance - 1.0e-3]
        if below_clearance_poses:
            min_clear = min(p.clearance_mm for p in below_clearance_poses)
            reasons.append(
                f"Transition {tid}:\n"
                f"- {len(below_clearance_poses)} poses below required clearance;\n"
                f"- minimum clearance: {min_clear:.1f} mm;\n"
                f"- required clearance: {min_req_clearance:.1f} mm."
            )
            continue


    # Also validate transition poses that have transition_id < 0 (if any exist)
    for p in poses:
        if p.scan_status == 0 and p.transition_id < 0:
            if p.transition_strategy == "failed":
                reasons.append(f"Transition strategy failed at global_pose_id {p.global_pose_id}")
            elif not p.collision_checked:
                reasons.append(f"Pose has not been collision checked at global_pose_id {p.global_pose_id}")
            elif p.clearance_mm <= 0.0:
                reasons.append(f"Pose at global_pose_id {p.global_pose_id} intersects CAD mesh.")
            elif p.clearance_mm < min_req_clearance - 1.0e-3:
                reasons.append(f"Clearance {p.clearance_mm:.2f} mm is below required effective tool clearance {min_req_clearance:.2f} mm at global_pose_id {p.global_pose_id}")

    # Calculate clearances
    free_travel_poses = [p for p in poses if p.scan_status == 0]
    min_clearance = float(min(p.clearance_mm for p in poses)) if poses else 0.0
    min_clearance_free_travel = float(min(p.clearance_mm for p in free_travel_poses)) if free_travel_poses else 0.0
    avg_clearance = float(np.mean([p.clearance_mm for p in free_travel_poses])) if free_travel_poses else 0.0

    # Calculate additional safe routing distance compared with unsafe direct routing
    direct_routing_distance_sum = 0.0
    if len(scan_blocks) > 0:
        direct_routing_distance_sum += float(np.linalg.norm(np.asarray(poses[0].position) - np.asarray(scan_blocks[0][0].position)))
        for idx in range(len(scan_blocks) - 1):
            p_from = scan_blocks[idx][-1]
            p_to = scan_blocks[idx+1][0]
            direct_routing_distance_sum += float(np.linalg.norm(np.asarray(p_from.position) - np.asarray(p_to.position)))
        direct_routing_distance_sum += float(np.linalg.norm(np.asarray(poses[-1].position) - np.asarray(scan_blocks[-1][-1].position)))
    additional_safe_dist = max(0.0, total_free_travel_dist - direct_routing_distance_sum)

    report = {
        "valid": len(reasons) == 0,
        "reasons": tuple(reasons),
        "num_scan_blocks": len(scan_blocks),
        "num_free_travel_blocks": len(free_travel_blocks),
        "num_full_retracts": num_full_retracts,
        "num_direct_transitions": num_direct_transitions,
        "total_scan_distance": total_scan_dist,
        "total_free_travel_distance": total_free_travel_dist,
        "largest_position_jump": largest_pos_jump,
        "largest_orientation_jump": largest_ori_jump,
        "invalid_transition_count": 0 if len(reasons) == 0 else 1,
        "minimum_mission_clearance": min_clearance,
        "average_transition_clearance": avg_clearance,
        "additional_safe_routing_distance": additional_safe_dist,
        "raw_distance_to_cad": min_clearance_free_travel,
        "effective_required_clearance": min_req_clearance,
        "remaining_clearance_margin": max(0.0, min_clearance_free_travel - min_req_clearance) if free_travel_poses else 0.0,
    }
    return report
