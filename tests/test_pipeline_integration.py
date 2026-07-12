from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cadquery as cq

from app.config import PipelineConfig
from app.pipeline import run_pipeline
from planners.base import CylinderStrategy


def _export_step(shape: cq.Shape, directory: Path, name: str) -> Path:
    path = directory / name
    cq.exporters.export(shape, str(path), exportType="STEP")
    return path


def _assert_exports(result) -> None:
    assert set(result.summary.output_files) == {
        "trajectory_csv",
        "trajectory_json",
        "coverage_json",
        "validation_report_json",
    }
    for path in result.summary.output_files.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    trajectory = json.loads(
        result.summary.output_files["trajectory_json"].read_text()
    )
    coverage = json.loads(
        result.summary.output_files["coverage_json"].read_text()
    )
    report = json.loads(
        result.summary.output_files["validation_report_json"].read_text()
    )
    assert len(trajectory["trajectory"]) == len(result.poses)
    assert "total_exterior_surface" in coverage["surface_report"]
    assert "safety_notice" in report


def test_box_complete_pipeline() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        step = _export_step(
            cq.Workplane("XY").box(20.0, 15.0, 10.0).val(),
            root,
            "box.step",
        )
        result = run_pipeline(
            PipelineConfig(
                step_path=step,
                surface_types=("plane",),
                lane_spacing_mm=5.0,
                travel_spacing_mm=5.0,
                minimum_face_area_mm2=0.0,
                display=False,
                geometry_validation=True,
                output_dir=root / "output",
                laser_width_mm=10.0,
                laser_ray_count=5,
            )
        )

        assert result.summary.total_step_faces == 6
        assert result.summary.selected_planar_faces == 6
        assert result.summary.selected_cylindrical_faces == 0
        assert result.summary.generated_scan_segments > 0
        assert result.summary.scan_pose_count > 0
        assert result.summary.total_scan_path_length_mm > 0.0
        assert result.geometry_validation is not None
        assert result.geometry_validation.minimum_stand_off_mm == 310.0
        report = json.loads(
            result.summary.output_files[
                "validation_report_json"
            ].read_text()
        )
        assert "geometry_validation" in report
        _assert_exports(result)


def test_cylinder_complete_pipeline() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        step = _export_step(
            cq.Solid.makeCylinder(5.0, 12.0),
            root,
            "cylinder.step",
        )
        result = run_pipeline(
            PipelineConfig(
                step_path=step,
                surface_types=("cylinder",),
                lane_spacing_mm=5.0,
                travel_spacing_mm=2.0,
                minimum_face_area_mm2=0.0,
                cylinder_strategy=CylinderStrategy.LONGITUDINAL,
                display=False,
                output_dir=root / "output",
                laser_width_mm=8.0,
                laser_ray_count=5,
            )
        )

        assert result.summary.total_step_faces == 3
        assert result.summary.selected_planar_faces == 0
        assert result.summary.selected_cylindrical_faces == 1
        assert len(result.cylindrical_plans) == 1
        assert result.summary.generated_scan_segments > 0
        _assert_exports(result)


def test_box_with_hole_excludes_internal_cylinder_by_default() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        block = cq.Workplane("XY").box(20.0, 20.0, 10.0).val()
        bore = cq.Solid.makeCylinder(
            3.0, 12.0, pnt=cq.Vector(0.0, 0.0, -6.0)
        )
        step = _export_step(block.cut(bore), root, "box_hole.step")

        excluded = run_pipeline(
            PipelineConfig(
                step_path=step,
                surface_types=("plane", "cylinder"),
                lane_spacing_mm=5.0,
                travel_spacing_mm=2.0,
                minimum_face_area_mm2=0.0,
                display=False,
                output_dir=root / "excluded",
                laser_width_mm=8.0,
                laser_ray_count=5,
            )
        )
        included = run_pipeline(
            PipelineConfig(
                step_path=step,
                surface_types=("cylinder",),
                lane_spacing_mm=5.0,
                travel_spacing_mm=2.0,
                minimum_face_area_mm2=0.0,
                include_internal_surfaces=True,
                display=False,
                output_dir=root / "included",
                laser_width_mm=8.0,
                laser_ray_count=5,
            )
        )

        assert excluded.summary.selected_cylindrical_faces == 0
        assert "internal surface excluded" in excluded.summary.skipped_faces_by_reason
        assert included.summary.selected_cylindrical_faces == 1
        assert (
            included.coverage.report[
                "total_exterior_surface"
            ].eligible_area_mm2
            > excluded.coverage.report[
                "total_exterior_surface"
            ].eligible_area_mm2
        )
        _assert_exports(excluded)
        _assert_exports(included)
