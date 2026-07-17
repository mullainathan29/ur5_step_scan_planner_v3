from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import trimesh
from PySide6 import QtWidgets

from gui.main_window import MainWindow
from gui.viewer_widget import ViewerWidget
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

def test_show_result_rejects_incomplete_result():
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    # 1. Show a valid result to populate some actors
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
    dummy_sweep = SimpleNamespace(
        sweep_id=1,
        view_type="plane",
        scan_poses=[pose1, pose1],
        poses=[pose1, pose1],
        colour="blue"
    )
    result = make_dummy_result(poses=[pose1], sweeps=[dummy_sweep])
    viewer.show_result(result, config)
    
    initial_cad_count = len(viewer._cad_actors)
    assert initial_cad_count > 0
    
    # 2. Call show_result with an incomplete result (poses=empty)
    incomplete_result = make_dummy_result(poses=[], sweeps=[])
    viewer.show_result(incomplete_result, config)
    
    # Verify that the viewer did NOT clear the existing CAD actors (early return)
    assert len(viewer._cad_actors) == initial_cad_count

def test_validation_update_does_not_clear_trajectories(monkeypatch):
    _application()
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: 0)
    window = MainWindow()
    config = PipelineConfig(Path("dummy.step"))
    window.config = config
    
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
        scan_poses=[pose1, pose1],
        poses=[pose1, pose1],
        colour="blue"
    )
    initial_result = make_dummy_result(poses=[pose1, pose2], sweeps=[dummy_sweep])
    
    window.result = initial_result
    window.current_result = initial_result
    window.viewer.show_result(initial_result, config)
    
    assert len(window.viewer._scan_path_actors) > 0
    initial_scan_actors = list(window.viewer._scan_path_actors)
    
    # Create validation-only result payload
    validation_summary = PipelineSummary(
        total_step_faces=1,
        selected_planar_faces=1,
        selected_cylindrical_faces=0,
        selected_conical_faces=0,
        selected_spherical_faces=0,
        skipped_faces_by_reason={},
        generated_scan_regions=1,
        generated_scan_segments=1,
        scan_pose_count=1,
        total_scan_path_length_mm=10.0,
        coverage_percent=98.0,
        invalid_segment_count=0,
        output_files={"mission_validation_report": {
            "valid": True,
            "reasons": [],
            "num_scan_blocks": 1,
            "num_free_travel_blocks": 1,
            "num_full_retracts": 0,
            "num_direct_transitions": 0,
            "total_scan_distance": 10.0,
            "total_free_travel_distance": 10.0,
            "largest_position_jump": 0.0,
            "largest_orientation_jump": 0.0,
            "minimum_mission_clearance": 10.0
        }}
    )
    
    class DummyGeometryValidation:
        def __init__(self):
            self.pose_metrics = ()
            
    validation_payload = SimpleNamespace(
        value=SimpleNamespace(
            geometry_validation=DummyGeometryValidation(),
            summary=validation_summary
        ),
        timings=SimpleNamespace(seconds={})
    )
    
    # Trigger validation completed slot
    window._validation_updated(validation_payload)
    
    # Check that scan actors were preserved
    assert len(window.viewer._scan_path_actors) == len(initial_scan_actors)
    assert window.viewer._scan_path_actors == initial_scan_actors
    
    # Check that summary properties were merged
    assert window.current_result.summary.coverage_percent == 98.0
    assert "mission_validation_report" in window.current_result.summary.output_files

def test_safety_envelope_opacity(monkeypatch):
    _application()
    viewer = ViewerWidget()
    config = PipelineConfig(Path("dummy.step"))
    
    class DummySafetyBox:
        def __init__(self):
            self.vertices = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]])
            self.faces = np.array([[0,1,2]])
            
    added_meshes = []
    def fake_add_mesh(mesh, *args, **kwargs):
        added_meshes.append(kwargs)
        class DummyActor:
            def SetVisibility(self, val):
                pass
        return DummyActor()
    
    monkeypatch.setattr(viewer.interactor, "add_mesh", fake_add_mesh)
    
    pose1 = SimpleNamespace(
        position=[0.0, 0.0, 0.0],
        surface_point=[0.0, 0.0, 0.0],
        quaternion_xyzw=[0, 0, 0, 1],
        scan_enabled=True,
        scan_status=0,
        global_pose_id=0,
        segment_id=0,
        pose_id=0,
        clearance_mm=10.0
    )
    summary = PipelineSummary(
        total_step_faces=1,
        selected_planar_faces=1,
        selected_cylindrical_faces=0,
        selected_conical_faces=0,
        selected_spherical_faces=0,
        skipped_faces_by_reason={},
        generated_scan_regions=1,
        generated_scan_segments=1,
        scan_pose_count=1,
        total_scan_path_length_mm=10.0,
        coverage_percent=95.0,
        invalid_segment_count=0,
        output_files={"safety_box": DummySafetyBox()}
    )
    result = PipelineResult(
        model=DummyModel(),
        planar_plans=(),
        cylindrical_plans=(),
        conical_plans=(),
        spherical_plans=(),
        poses=(pose1,),
        coverage=None,
        baseline_coverage=None,
        geometry_validation=None,
        summary=summary
    )
    
    viewer.show_result(result, config)
    
    safety_box_calls = [c for c in added_meshes if c.get("color") == "cyan" and c.get("style") == "surface"]
    assert len(safety_box_calls) == 1
    assert safety_box_calls[0]["opacity"] == 0.03
