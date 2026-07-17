from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from PySide6 import QtWidgets

from gui.main_window import MainWindow
from gui.viewer_widget import ViewerWidget, ActorRecord, DEV_LAYERS
from app.result import PipelineSummary, PipelineResult
from app.config import PipelineConfig

def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

class DummyFace:
    def __init__(self, face_id):
        self.face_id = face_id
        self.center = SimpleNamespace(x=0.0, y=0.0, z=0.0)

class DummyValidationMesh:
    def __init__(self):
        self.mesh = trimesh.creation.box(extents=[10, 10, 10])

class DummyModel:
    def __init__(self):
        self.faces = [DummyFace(1)]
        self.validation_mesh = DummyValidationMesh()
        self.source_path = Path("dummy.step")

def make_dummy_result(poses=None, sweeps=None):
    summary = PipelineSummary(
        total_step_faces=1,
        selected_planar_faces=1,
        selected_cylindrical_faces=0,
        selected_conical_faces=0,
        selected_spherical_faces=0,
        skipped_faces_by_reason={},
        generated_scan_regions=1,
        generated_scan_segments=1,
        scan_pose_count=len(poses) if poses else 0,
        total_scan_path_length_mm=10.0,
        coverage_percent=95.0,
        invalid_segment_count=0,
        output_files={}
    )
    
    return PipelineResult(
        model=DummyModel(),
        planar_plans=(),
        cylindrical_plans=(),
        conical_plans=(),
        spherical_plans=(),
        poses=tuple(poses) if poses else (),
        coverage=None,
        baseline_coverage=None,
        geometry_validation=None,
        summary=summary,
        sweeps=tuple(sweeps) if sweeps else ()
    )

def test_operator_mode_contains_only_mission_actors():
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    pose1 = SimpleNamespace(
        position=[0.0, 0.0, 0.0],
        surface_point=[0.0, 0.0, 0.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=0,
        segment_id=0,
        pose_id=0
    )
    pose2 = SimpleNamespace(
        position=[1.0, 1.0, 1.0],
        surface_point=[1.0, 1.0, 1.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=1,
        segment_id=0,
        pose_id=1
    )
    result = make_dummy_result(poses=[pose1, pose2])
    
    # Initialize in Operator Mode
    viewer.set_developer_mode(False)
    viewer.show_result(result, config)
    
    # Verify that all developer/planning artefacts are invisible
    dev_records = [r for r in viewer.controller.records if r.layer in DEV_LAYERS]
    assert len(dev_records) > 0
    for record in dev_records:
        assert not record.actor.GetVisibility()

    # Verify that CAD, Coverage, Scan, Free travel and start/end markers are visible
    operator_layers = {"cad", "coverage", "mission_scan", "mission_free_travel", "mission_start_end"}
    op_records = [r for r in viewer.controller.records if r.layer in operator_layers]
    assert len(op_records) > 0
    for record in op_records:
        # Note: mission_free_travel might be empty if there are no free travel segments, but others should be visible
        if record.layer != "mission_free_travel":
            assert record.actor.GetVisibility()

def test_developer_mode_shows_planning_artefacts():
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    pose1 = SimpleNamespace(
        position=[0.0, 0.0, 0.0],
        surface_point=[0.0, 0.0, 0.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=0,
        segment_id=0,
        pose_id=0
    )
    pose2 = SimpleNamespace(
        position=[1.0, 1.0, 1.0],
        surface_point=[1.0, 1.0, 1.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=1,
        segment_id=0,
        pose_id=1
    )
    dummy_sweep = SimpleNamespace(
        sweep_id=1,
        view_type="plane",
        scan_poses=[pose1, pose2],
        poses=[pose1, pose2],
        colour="blue"
    )
    result = make_dummy_result(poses=[pose1, pose2], sweeps=[dummy_sweep])
    
    # Enable Developer Mode
    viewer.set_developer_mode(True)
    viewer.show_result(result, config)
    
    # Assert that sweeps nominal path (layer "nominal") is visible
    nominal_records = [r for r in viewer.controller.records if r.layer == "nominal"]
    assert len(nominal_records) > 0
    for r in nominal_records:
        assert r.actor.GetVisibility()

def test_scan_free_travel_splitting():
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    # Create poses alternating between Scan (-1) and Free Travel (0)
    poses = [
        SimpleNamespace(position=[0,0,0], surface_point=[0,0,0], quaternion_xyzw=[0,0,0,1], scan_enabled=True, scan_status=-1, global_pose_id=0, segment_id=0, pose_id=0),
        SimpleNamespace(position=[1,0,0], surface_point=[1,0,0], quaternion_xyzw=[0,0,0,1], scan_enabled=True, scan_status=-1, global_pose_id=1, segment_id=0, pose_id=1),
        SimpleNamespace(position=[2,0,0], surface_point=[2,0,0], quaternion_xyzw=[0,0,0,1], scan_enabled=False, scan_status=0, global_pose_id=2, segment_id=0, pose_id=2),
        SimpleNamespace(position=[3,0,0], surface_point=[3,0,0], quaternion_xyzw=[0,0,0,1], scan_enabled=False, scan_status=0, global_pose_id=3, segment_id=0, pose_id=3),
        SimpleNamespace(position=[4,0,0], surface_point=[4,0,0], quaternion_xyzw=[0,0,0,1], scan_enabled=True, scan_status=-1, global_pose_id=4, segment_id=0, pose_id=4),
        SimpleNamespace(position=[5,0,0], surface_point=[5,0,0], quaternion_xyzw=[0,0,0,1], scan_enabled=True, scan_status=-1, global_pose_id=5, segment_id=0, pose_id=5),
    ]
    result = make_dummy_result(poses=poses)
    viewer.show_result(result, config)
    
    # We should have exactly 2 scan segments and 1 free travel segment
    # Check that they are rendered correctly
    scan_actors = [r for r in viewer.controller.records if r.layer == "mission_scan"]
    free_actors = [r for r in viewer.controller.records if r.layer == "mission_free_travel"]
    
    assert len(scan_actors) == 2
    assert len(free_actors) == 1

def test_no_duplicate_mission_rendering():
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    pose1 = SimpleNamespace(
        position=[0.0, 0.0, 0.0],
        surface_point=[0.0, 0.0, 0.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=0,
        segment_id=0,
        pose_id=0
    )
    pose2 = SimpleNamespace(
        position=[1.0, 1.0, 1.0],
        surface_point=[1.0, 1.0, 1.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=1,
        segment_id=0,
        pose_id=1
    )
    dummy_sweep = SimpleNamespace(
        sweep_id=1,
        view_type="plane",
        scan_poses=[pose1, pose2],
        poses=[pose1, pose2],
        colour="blue"
    )
    result = make_dummy_result(poses=[pose1, pose2], sweeps=[dummy_sweep])
    viewer.show_result(result, config)
    
    # Verify that nominal sweep actors are in _recovery_path_actors, NOT in _scan_path_actors
    # thereby ensuring they are not controlled by the Operator "Scan path" checkbox.
    # Check that _scan_path_actors contains only the mission_scan actors.
    for actor in viewer._scan_path_actors:
        record = next(r for r in viewer.controller.records if r.actor is actor)
        assert record.layer == "mission_scan"

def test_actor_count_stable_after_validation():
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    pose1 = SimpleNamespace(
        position=[0.0, 0.0, 0.0],
        surface_point=[0.0, 0.0, 0.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=0,
        segment_id=0,
        pose_id=0
    )
    pose2 = SimpleNamespace(
        position=[1.0, 1.0, 1.0],
        surface_point=[1.0, 1.0, 1.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=-1,
        global_pose_id=1,
        segment_id=0,
        pose_id=1
    )
    result = make_dummy_result(poses=[pose1, pose2])
    viewer.show_result(result, config)
    
    initial_actor_count = len(viewer.controller.records)
    
    # Run validation updates
    viewer.update_validation(result, config)
    count_after_first = len(viewer.controller.records)
    
    viewer.update_validation(result, config)
    count_after_second = len(viewer.controller.records)
    
    # The actor counts should be stable and not growing
    assert count_after_first == count_after_second
