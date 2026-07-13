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
    construct_safety_envelope, plan_safe_transition, is_segment_safe
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
    
    # Unsafe direct transition must be routed around (strategy = obb_route or top_clearance)
    assert strategy in {"obb_route", "top_clearance"}
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
    
    # The orientation rotation happens at path_rot[2] (which is the rotated retract point)
    # The retract point itself path_rot[1] is outside the safety envelope.
    assert np.linalg.norm(path_rot[1][0]) > 140.0
    assert np.allclose(path_rot[1][1], q_start)
    assert np.allclose(path_rot[2][1], q_goal)
    assert np.linalg.norm(path_rot[2][0] - path_rot[1][0]) < 1e-3  # Same position, orientation changed
    
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
