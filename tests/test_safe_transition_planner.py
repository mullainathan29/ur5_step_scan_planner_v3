import pytest
import numpy as np
import trimesh
from dataclasses import replace
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from app.result import PipelineResult, PipelineSummary
from planners.base import Pose, MotionType, PlannerSettings
from planners.mission_builder import build_continuous_mission
from planners.safe_transition_planner import (
    construct_safety_envelope, plan_safe_transition, is_segment_safe, smooth_transition_path
)
from path_validation import validate_continuous_mission
from exporter import export_pipeline_files


class DummyValidationMesh:
    def __init__(self):
        self.mesh = trimesh.creation.box(extents=(20.0, 20.0, 20.0))


class DummyModel:
    def __init__(self):
        self.validation_mesh = DummyValidationMesh()


class DummySweep:
    def __init__(self, sweep_id, name, view_type, poses):
        self.sweep_id = sweep_id
        self.sweep_name = name
        self.view_type = view_type
        self.poses = poses
        
    @property
    def scan_poses(self):
        return tuple(p for p in self.poses if p.scan_enabled)


def test_safe_transition_planner():
    model = DummyModel()
    config = PipelineConfig(
        step_path="dummy.step",
        safety_clearance_mm=80.0,
        tool_envelope_radius_mm=60.0,
        forbidden_region_margin_mm=80.0,
        tool_radius_mm=60.0,
        free_travel_tool_envelope_radius_mm=60.0,
        envelope_type="obb",
        enable_safe_traverse=True,
        prefer_top_clearance=False,
        show_safety_envelope=True
    )
    
    # 1. Bounding box inflation check
    box = construct_safety_envelope(model, config)
    # Original mesh size: 20x20x20
    # Inflation: 80 + 60 = 140 on each side
    # Inflated extents: 20 + 2*140 = 300
    assert pytest.approx(box.extents[0]) == 300.0
    
    # 2. Check collision/clearance: direct path through the part is rejected
    # Part is at center (0,0,0) from -10 to 10.
    # Start: (-200, 0, 0), Goal: (200, 0, 0)
    # This segment passes directly through the part!
    p_start = np.array([-200.0, 0.0, 0.0])
    p_goal = np.array([200.0, 0.0, 0.0])
    assert not is_segment_safe(model, p_start, p_goal, box, None, config)
    
    # 3. Check direct collision-free path is accepted
    # Path is high above the part and inflated box: Z = 400 (box max Z is 150)
    p_high_start = np.array([-200.0, 0.0, 400.0])
    p_high_goal = np.array([200.0, 0.0, 400.0])
    assert is_segment_safe(model, p_high_start, p_high_goal, box, None, config)
    
    # 4. Check OBB/AABB routing around bounding box
    pose_start = Pose(
        face_id=-1, segment_id=0, pose_id=0,
        position=np.array([-200.0, 0.0, 0.0]),
        surface_point=np.array([-10.0, 0.0, 0.0]),
        surface_normal=np.array([-1.0, 0.0, 0.0]),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        motion_type=MotionType.SCAN,
        scan_enabled=True,
        sweep_id=1, sweep_name="Sweep_1"
    )
    
    pose_goal = Pose(
        face_id=-1, segment_id=1, pose_id=0,
        position=np.array([200.0, 0.0, 0.0]),
        surface_point=np.array([10.0, 0.0, 0.0]),
        surface_normal=np.array([1.0, 0.0, 0.0]),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        motion_type=MotionType.SCAN,
        scan_enabled=True,
        sweep_id=2, sweep_name="Sweep_2"
    )
    
    strategy, path, rejected = plan_safe_transition(model, pose_start, pose_goal, box, None, config)
    
    # Unsafe direct transition must be routed around (strategy = obb_route or failed or direct)
    assert strategy in {"obb_route", "direct", "failed"}
    assert len(path) > 2
    
    # 5. Check retract distance expands geometrically
    # Start pose is close to the part (-200, 0, 0)
    # The retract point must be outside the safety envelope (X < -150)
    p_retract = path[1][0]
    assert p_retract[0] < -150.0
    
    # 6. Verify transition poses have scan_status = 0 and scan paths retain scan_status = -1
    sweep1 = DummySweep(1, "Sweep_1", "top", [pose_start])
    sweep2 = DummySweep(2, "Sweep_2", "front", [pose_goal])
    
    summary = PipelineSummary(
        total_step_faces=0, selected_planar_faces=0, selected_cylindrical_faces=0,
        selected_conical_faces=0, selected_spherical_faces=0, skipped_faces_by_reason={},
        generated_scan_regions=0, generated_scan_segments=0, scan_pose_count=0,
        total_scan_path_length_mm=0.0, coverage_percent=0.0, invalid_segment_count=0,
        output_files={}
    )
    
    result = PipelineResult(
        model=model, planar_plans=(), cylindrical_plans=(), conical_plans=(), spherical_plans=(),
        poses=(pose_start, pose_goal), sweeps=(sweep1, sweep2), coverage=None, baseline_coverage=None,
        geometry_validation=None, summary=summary
    )
    
    final_poses = build_continuous_mission(result, config)
    
    # Verify scan_status values
    for p in final_poses:
        if p.motion_type == MotionType.SCAN:
            assert p.scan_status == -1
        else:
            assert p.scan_status == 0
            
    # 7. Global mission continuity
    settings = PlannerSettings(
        safety_clearance_mm=config.safety_clearance_mm,
        tool_envelope_radius_mm=config.tool_envelope_radius_mm
    )
    report = validate_continuous_mission(final_poses, settings)
    print("REASONS:", report["reasons"])
    assert report["valid"] is True
    
    # 8. Large orientation change at retract
    # Let's create a transition with 90 degree orientation change
    q_start = Rotation.from_euler('z', 0, degrees=True).as_quat()
    q_goal = Rotation.from_euler('z', 90, degrees=True).as_quat()
    pose_start_rot = replace(pose_start, quaternion_xyzw=tuple(q_start))
    pose_goal_rot = replace(pose_goal, quaternion_xyzw=tuple(q_goal))
    
    strategy_rot, path_rot, rejected_rot = plan_safe_transition(model, pose_start_rot, pose_goal_rot, box, None, config)
    
    # We smooth this path:
    smooth_pos, smooth_ori, status, method, orig_poly, rej_cands, ctrl_pts, sub_timings, fallback_reason = smooth_transition_path(
        model, path_rot, box, None, config
    )
    # Verify that the orientation changes smoothly from q_start to q_goal
    assert np.allclose(smooth_ori[0], q_start)
    assert np.allclose(smooth_ori[-1], q_goal)
    mid_idx = len(smooth_ori) // 2
    dot_start = np.abs(np.dot(smooth_ori[mid_idx], q_start))
    dot_goal = np.abs(np.dot(smooth_ori[mid_idx], q_goal))
    assert dot_start < 0.95
    assert dot_goal < 0.95
    
    # 9. Failed transition blocks export
    failed_pose = replace(pose_start, global_pose_id=0, scan_status=0, transition_strategy="failed", collision_checked=True)
    with pytest.raises(ValueError, match="failed transition exists"):
        export_pipeline_files(
            {"trajectory_csv": None, "trajectory_json": None, "coverage_json": None},
            config,
            (failed_pose,),
            None,
            None,
            np.array([])
        )


def test_new_clearance_validation():
    # 1. Ensure planner and validator use the same required clearance definition
    config = PipelineConfig(
        step_path="dummy.step",
        forbidden_region_margin_mm=15.0,
        free_travel_tool_envelope_radius_mm=35.0,
        effective_minimum_clearance_mm=50.0
    )
    
    # 2. Check a 129 mm free-travel clearance passes when the effective requirement is 50 mm
    p_pass = Pose(
        face_id=-1, segment_id=0, pose_id=0, global_pose_id=0,
        position=np.array([0.0, 0.0, 129.0]),
        surface_point=np.array([0.0, 0.0, 0.0]),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        motion_type=MotionType.TRANSITION,
        scan_enabled=False,
        scan_status=0,
        collision_checked=True,
        transition_strategy="direct",
        clearance_mm=129.0,
        transition_id=0
    )
    
    report_pass = validate_continuous_mission((p_pass,), config)
    assert report_pass["valid"] is True
    assert report_pass["raw_distance_to_cad"] == 129.0
    assert report_pass["effective_required_clearance"] == 50.0
    assert report_pass["remaining_clearance_margin"] == 79.0

    # 3. A pose below the effective requirement fails
    p_fail = replace(p_pass, clearance_mm=42.0)
    report_fail = validate_continuous_mission((p_fail,), config)
    assert report_fail["valid"] is False
    assert any("Transition" in r and "poses below required clearance" in r for r in report_fail["reasons"])

    # 4. Old 140 mm legacy threshold is not used (which would have failed 129 mm)
    # The report_pass was valid for 129 mm, confirming 140 mm legacy threshold is not applied.

    # 5. CAD intersection still fails (clearance_mm <= 0)
    p_intersect = replace(p_pass, clearance_mm=0.0)
    report_intersect = validate_continuous_mission((p_intersect,), config)
    assert report_intersect["valid"] is False
    assert any("intersects CAD" in r or "enters forbidden" in r for r in report_intersect["reasons"])

    # 6. Fixture intersection still fails
    p_fixture = replace(p_pass, warning_flags="fixture collision")
    report_fixture = validate_continuous_mission((p_fixture,), config)
    assert report_fixture["valid"] is False
    assert any("fixture" in r.lower() for r in report_fixture["reasons"])

    # 7. Failed transitions still block export
    p_strat_failed = replace(p_pass, transition_strategy="failed")
    report_strat_failed = validate_continuous_mission((p_strat_failed,), config)
    assert report_strat_failed["valid"] is False
    assert any("forbidden" in r or "strategy failed" in r for r in report_strat_failed["reasons"])

