from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from app.pipeline import run_pipeline
from cad_model import load_step_model
from planners.view_based import ViewBasedSweepPlanner, ViewSweepSettings


def _box_model(x: float, y: float, z: float):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "box.step"
    cq.exporters.export(cq.Workplane("XY").box(x, y, z).val(), str(path), exportType="STEP")
    return temporary, load_step_model(path), path


def test_one_top_pass_when_width_is_below_laser_width() -> None:
    temporary, model, _ = _box_model(300.0, 100.0, 40.0)
    try:
        mission = ViewBasedSweepPlanner(ViewSweepSettings(
            laser_width_mm=200.0, top_sweep=True, front_sweep=False,
            right_sweep=False, back_sweep=False, left_sweep=False,
            isometric_recovery=False,
        )).plan(model)
        assert [sweep.sweep_name for sweep in mission.sweeps] == ["top"]
        assert len(mission.sweeps[0].scan_poses) < 100
    finally:
        temporary.cleanup()


def test_multiple_top_passes_use_minimum_overlap_count() -> None:
    temporary, model, _ = _box_model(400.0, 300.0, 40.0)
    try:
        mission = ViewBasedSweepPlanner(ViewSweepSettings(
            laser_width_mm=200.0, overlap_fraction=0.20,
            top_sweep=True, front_sweep=False, right_sweep=False,
            back_sweep=False, left_sweep=False, isometric_recovery=False,
        )).plan(model)
        assert [sweep.sweep_name for sweep in mission.sweeps] == ["top_1", "top_2"]
    finally:
        temporary.cleanup()


def test_principal_order_sides_axes_travel_and_stand_off() -> None:
    temporary, model, _ = _box_model(80.0, 60.0, 40.0)
    try:
        settings = ViewSweepSettings(laser_width_mm=500.0, isometric_recovery=False)
        mission = ViewBasedSweepPlanner(settings).plan(model)
        assert [s.view_type for s in mission.sweeps] == [
            "top", "front", "right", "back", "left"
        ]
        assert all(len([s for s in mission.sweeps if s.view_type == view]) == 1
                   for view in ("front", "right", "back", "left"))
        for sweep in mission.sweeps:
            for pose in sweep.scan_poses:
                rotation = Rotation.from_quat(pose.quaternion_xyzw)
                laser = np.asarray(pose.laser_line_direction)
                travel = np.asarray(pose.travel_direction)
                assert np.allclose(rotation.apply([1, 0, 0]), laser)
                assert np.allclose(rotation.apply([0, 1, 0]), travel)
                assert np.isclose(np.linalg.norm(pose.position - pose.surface_point), 310.0)
            displacement = sweep.scan_poses[-1].position - sweep.scan_poses[0].position
            assert np.dot(displacement, np.asarray(sweep.scan_poses[0].travel_direction)) > 0
    finally:
        temporary.cleanup()


def test_isometric_recovery_is_not_generated_without_uncovered_area(monkeypatch) -> None:
    temporary, model, _ = _box_model(20.0, 20.0, 20.0)
    try:
        from coverage import validate_coverage as real_validate
        complete = real_validate(model, ())
        complete.triangle_status.flags.writeable = True
        complete.triangle_status[complete.eligible_triangle_mask] = 1
        complete.triangle_status.flags.writeable = False
        monkeypatch.setattr("planners.view_based.validate_coverage", lambda *args, **kwargs: complete)
        mission = ViewBasedSweepPlanner().plan(model)
        assert all(sweep.view_type != "isometric" for sweep in mission.sweeps)
    finally:
        temporary.cleanup()


def test_view_based_export_schema() -> None:
    temporary, _model, path = _box_model(20.0, 15.0, 10.0)
    try:
        output = Path(temporary.name) / "output"
        result = run_pipeline(PipelineConfig(
            step_path=path, planning_mode="simple_view_based", display=False,
            output_dir=output, isometric_recovery=False, laser_ray_count=5,
        ))
        payload = json.loads(result.summary.output_files["trajectory_json"].read_text())
        required = {
            "sweep_id", "sweep_name", "view_type", "pose_id", "motion_type",
            "scan_enabled", "x", "y", "z", "qx", "qy", "qz", "qw",
            "rx", "ry", "rz", "roll_deg", "pitch_deg", "yaw_deg",
            "laser_line_x", "laser_line_y", "laser_line_z", "travel_x",
            "travel_y", "travel_z", "predicted_stand_off_mm",
            "predicted_visible_laser_width_mm",
        }
        assert required <= payload["trajectory"][0].keys()
        assert result.sweeps[0].view_type == "top"
    finally:
        temporary.cleanup()
