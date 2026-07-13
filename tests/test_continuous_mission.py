import pytest
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from planners.base import Pose, MotionType, PlannerSettings
from planners.mission_builder import build_continuous_mission, slerp_quaternions
from path_validation import validate_continuous_mission
from app.config import PipelineConfig
from app.result import PipelineResult, PipelineSummary


def test_slerp_quaternions():
    q1 = np.array([0.0, 0.0, 0.0, 1.0])
    q2 = Rotation.from_euler('z', 120, degrees=True).as_quat()
    
    q_mid = slerp_quaternions(q1, q2, 0.5)
    assert np.linalg.norm(q_mid) == pytest.approx(1.0)
    
    q_mid_neg = slerp_quaternions(q1, -q2, 0.5)
    assert np.linalg.norm(q_mid_neg) == pytest.approx(1.0)
    assert abs(np.dot(q_mid, q_mid_neg)) == pytest.approx(1.0)


class DummyValidationMesh:
    def __init__(self):
        self.mesh = trimesh.creation.box(extents=(100.0, 100.0, 20.0))


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


def test_build_continuous_mission():
    model = DummyModel()
    config = PipelineConfig(
        step_path="dummy.step",
        stand_off_mm=120.0,
        bottom_clearance_mm=25.0,
        safety_clearance_mm=10.0,
        tool_envelope_radius_mm=5.0
    )
    
    pose_templates = [
        Pose(
            face_id=-1,
            segment_id=0,
            pose_id=0,
            position=(0.0, 0.0, 10.0),
            surface_point=(0.0, 0.0, 0.0),
            surface_normal=(0.0, 0.0, 1.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            motion_type=MotionType.SCAN,
            scan_enabled=True,
            sweep_id=1,
            sweep_name="Sweep_1"
        ),
        Pose(
            face_id=-1,
            segment_id=0,
            pose_id=1,
            position=(10.0, 0.0, 10.0),
            surface_point=(10.0, 0.0, 0.0),
            surface_normal=(0.0, 0.0, 1.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            motion_type=MotionType.SCAN,
            scan_enabled=True,
            sweep_id=1,
            sweep_name="Sweep_1"
        )
    ]
    
    sweep1 = DummySweep(1, "Sweep_1", "top", pose_templates)
    
    pose_templates_2 = [
        Pose(
            face_id=-1,
            segment_id=1,
            pose_id=0,
            position=(20.0, 20.0, 10.0),
            surface_point=(20.0, 20.0, 0.0),
            surface_normal=(0.0, 0.0, 1.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            motion_type=MotionType.SCAN,
            scan_enabled=True,
            sweep_id=2,
            sweep_name="Sweep_2"
        ),
        Pose(
            face_id=-1,
            segment_id=1,
            pose_id=1,
            position=(30.0, 20.0, 10.0),
            surface_point=(30.0, 20.0, 0.0),
            surface_normal=(0.0, 0.0, 1.0),
            quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
            motion_type=MotionType.SCAN,
            scan_enabled=True,
            sweep_id=2,
            sweep_name="Sweep_2"
        )
    ]
    
    sweep2 = DummySweep(2, "Sweep_2", "front", pose_templates_2)
    
    import unittest.mock as mock
    with mock.patch("trimesh.proximity.closest_point") as mock_closest, \
         mock.patch("trimesh.load") as mock_load:
        
        mock_closest.return_value = (None, np.array([50.0, 50.0, 50.0, 50.0, 50.0]), None)
        
        model.validation_mesh.mesh.ray = mock.MagicMock()
        model.validation_mesh.mesh.ray.intersects_location.return_value = ([], [], [])
        
        summary = PipelineSummary(
            total_step_faces=0,
            selected_planar_faces=0,
            selected_cylindrical_faces=0,
            selected_conical_faces=0,
            selected_spherical_faces=0,
            skipped_faces_by_reason={},
            generated_scan_regions=0,
            generated_scan_segments=0,
            scan_pose_count=0,
            total_scan_path_length_mm=0.0,
            coverage_percent=0.0,
            invalid_segment_count=0,
            output_files={}
        )
        
        result = PipelineResult(
            model=model,
            planar_plans=(),
            cylindrical_plans=(),
            conical_plans=(),
            spherical_plans=(),
            poses=tuple(pose_templates + pose_templates_2),
            sweeps=(sweep1, sweep2),
            coverage=None,
            baseline_coverage=None,
            geometry_validation=None,
            summary=summary
        )
        
        final_poses = build_continuous_mission(result, config)
        
        assert len(final_poses) > 0
        
        # 1. global_pose_id continuity check
        for idx, p in enumerate(final_poses):
            assert p.global_pose_id == idx
            
        # 2. scan_status / scan_enabled check
        for p in final_poses:
            if p.scan_status == -1:
                assert p.scan_enabled is True
            elif p.scan_status == 0:
                assert p.scan_enabled is False
            else:
                assert False, "scan_status must be -1 or 0"
                
        # 3. quaternion sign check
        for idx in range(1, len(final_poses)):
            dot = np.dot(final_poses[idx-1].quaternion_xyzw, final_poses[idx].quaternion_xyzw)
            assert dot >= -1e-5
            
        # 4. validation check
        settings = PlannerSettings(
            safety_clearance_mm=10.0,
            tool_envelope_radius_mm=5.0
        )
        report = validate_continuous_mission(final_poses, settings)
        
        print("REASONS:", report["reasons"])
        assert report["valid"] is True
        assert report["num_scan_blocks"] == 2
        assert report["total_scan_distance"] > 0.0
        assert report["total_free_travel_distance"] > 0.0
