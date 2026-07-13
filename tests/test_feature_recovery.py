from __future__ import annotations

import json
import tempfile
from dataclasses import replace
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from app.config import PipelineConfig
from app.pipeline import run_pipeline
from cad_model import load_step_model
from coverage import CoverageSettings, validate_coverage
from features import analyse_hole_coverage, detect_hole_features
from mission_optimizer import optimise_scan_mission
from planners.view_based import ViewBasedSweepPlanner, ViewSweepSettings
from recovery import (
    RecoveryCandidate, candidate_accessible, generate_bottom_sweep,
    estimate_candidate_complexity, generate_hole_recovery_candidates,
    generate_lower_isometric,
)


def _model(kind: str):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / f"{kind}.step"
    block = cq.Workplane("XY").box(40, 40, 20).val()
    if kind == "through":
        shape = block.cut(cq.Solid.makeCylinder(5, 30, pnt=cq.Vector(0, 0, -15)))
    elif kind == "blind":
        shape = block.cut(cq.Solid.makeCylinder(5, 10, pnt=cq.Vector(0, 0, 0)))
    elif kind == "boss":
        shape = block.fuse(cq.Solid.makeCylinder(5, 10, pnt=cq.Vector(0, 0, 10)))
    else:
        shape = block
    cq.exporters.export(shape, str(path), exportType="STEP")
    return temporary, load_step_model(path), path


def _top_only(model):
    return ViewBasedSweepPlanner(ViewSweepSettings(
        top_sweep=True, front_sweep=False, right_sweep=False,
        back_sweep=False, left_sweep=False, isometric_recovery=False,
        travel_spacing_mm=5,
    )).plan(model)


def test_through_hole_detection_and_boss_rejection() -> None:
    temporary, model, _ = _model("through")
    boss_temporary, boss, _ = _model("boss")
    try:
        holes = detect_hole_features(model)
        assert len(holes) == 1
        assert holes[0].through_hole
        assert np.isclose(holes[0].radius_mm, 5.0)
        assert len(holes[0].accessible_opening_directions) == 2
        assert detect_hole_features(boss) == ()
    finally:
        temporary.cleanup(); boss_temporary.cleanup()


def test_blind_hole_detection_and_bottom_face() -> None:
    temporary, model, _ = _model("blind")
    try:
        hole = detect_hole_features(model)[0]
        assert not hole.through_hole
        assert hole.bottom_face_id is not None
        assert np.isclose(hole.depth_mm, 10.0)
    finally:
        temporary.cleanup()


def test_stepped_counterbore_is_one_physical_hole() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "counterbore.step"
        shape = (
            cq.Workplane("XY").box(30, 30, 20)
            .faces(">Z").workplane().cboreHole(6, 12, 5).val()
        )
        cq.exporters.export(shape, str(path), exportType="STEP")
        holes = detect_hole_features(load_step_model(path))
        assert len(holes) == 1
        assert len(holes[0].sections) == 2
        assert [round(section.diameter_mm) for section in holes[0].sections] == [12, 6]
        assert holes[0].sections[0].end_depth_mm == holes[0].sections[1].start_depth_mm


def test_hole_chamfer_is_grouped_with_bore_wall() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "chamfer.step"
        shape = (
            cq.Workplane("XY").box(30, 30, 20)
            .faces(">Z").workplane().hole(6)
            .faces(">Z").edges("%CIRCLE").chamfer(1).val()
        )
        cq.exporters.export(shape, str(path), exportType="STEP")
        holes = detect_hole_features(load_step_model(path))
        assert len(holes) == 1
        assert {section.section_type for section in holes[0].sections} == {
            "cylindrical", "conical"
        }


def test_candidate_complexity_and_adaptive_reduction() -> None:
    temporary, model, _ = _model("through")
    try:
        hole = detect_hole_features(model)[0]
        report = analyse_hole_coverage(
            model, hole,
            validate_coverage(model, (), CoverageSettings(include_internal_surfaces=True)),
        )
        estimate = estimate_candidate_complexity(
            (hole,) * 3, tilt_count=3, azimuth_count=4,
            orbital_mode="enabled", blind_bottom_enabled=False,
            bottom_sweep_enabled=True, lower_isometric_enabled=False,
            maximum_views_per_hole=2,
        )
        assert estimate.cross_hole_candidates == 36
        assert estimate.orbital_candidates == 3
        assert estimate.estimated_candidates == 40
        candidates = generate_hole_recovery_candidates(
            hole, report, ViewSweepSettings(), maximum_views=8,
            tilt_angles_deg=(20, 35, 50), azimuth_count=4,
            orbital_mode="adaptive",
        )
        assert len([row for row in candidates
                    if row.strategy == "tilted_cross_hole"]) == 1
        complete = replace(report, complete=True)
        assert generate_hole_recovery_candidates(
            hole, complete, ViewSweepSettings()
        ) == ()
    finally:
        temporary.cleanup()


def test_top_sweep_leaves_hole_incomplete_and_tilted_views_add_hits() -> None:
    temporary, model, _ = _model("through")
    try:
        hole = detect_hole_features(model)[0]
        mission = _top_only(model)
        settings = CoverageSettings(laser_width_mm=200, laser_ray_count=21,
                                    include_internal_surfaces=True)
        before_coverage = validate_coverage(model, mission.poses, settings)
        before = analyse_hole_coverage(model, hole, before_coverage)
        assert not before.complete
        candidates = generate_hole_recovery_candidates(
            hole, before, ViewSweepSettings(travel_spacing_mm=2),
            tilt_angles_deg=(25, 45), azimuth_count=4, maximum_views=6,
        )
        assert {candidate.strategy for candidate in candidates} >= {
            "tilted_cross_hole", "orbital_hole"
        }
        individual_hits = []
        for candidate in candidates[:4]:
            result = validate_coverage(model, candidate.sweep.poses, settings)
            wall = np.isin(model.validation_mesh.triangle_face_ids,
                           hole.cylindrical_face_ids)
            individual_hits.append(set(np.flatnonzero(
                wall & (result.triangle_status == 1)
            )))
        assert len(set().union(*individual_hits)) >= max(map(len, individual_hits))
    finally:
        temporary.cleanup()


def test_orbital_orientation_continuity_and_rim_occlusion() -> None:
    temporary, model, _ = _model("through")
    try:
        hole = detect_hole_features(model)[0]
        before = analyse_hole_coverage(
            model, hole, validate_coverage(model, (), CoverageSettings(include_internal_surfaces=True))
        )
        orbit = next(candidate for candidate in generate_hole_recovery_candidates(
            hole, before, ViewSweepSettings(), maximum_views=3,
            tilt_angles_deg=(35,), azimuth_count=4,
        ) if candidate.strategy == "orbital_hole")
        quaternions = np.asarray([pose.quaternion_xyzw for pose in orbit.sweep.scan_poses])
        assert np.all(np.sum(quaternions[:-1] * quaternions[1:], axis=1) >= 0)
        result = validate_coverage(
            model, orbit.sweep.poses,
            CoverageSettings(include_internal_surfaces=True, laser_ray_count=21),
        )
        # First-hit validation rejects at least some rays blocked by the rim.
        assert result.valid_hit_count < result.ray_count
    finally:
        temporary.cleanup()


def test_blind_bottom_candidates_and_underside_generation() -> None:
    temporary, model, _ = _model("blind")
    try:
        hole = detect_hole_features(model)[0]
        before = analyse_hole_coverage(
            model, hole, validate_coverage(model, (), CoverageSettings(include_internal_surfaces=True))
        )
        candidates = generate_hole_recovery_candidates(
            hole, before, ViewSweepSettings(), maximum_views=6,
            tilt_angles_deg=(35,), azimuth_count=2,
        )
        assert any(candidate.strategy == "blind_bottom" for candidate in candidates)
        bottom = generate_bottom_sweep(model, ViewSweepSettings(), 100)
        lower = generate_lower_isometric(model, ViewSweepSettings(), 101)
        assert bottom.sweep.view_type == "bottom"
        assert lower.sweep.view_type == "lower_isometric"
        assert all(p.position[2] < p.surface_point[2] for p in bottom.sweep.scan_poses)
    finally:
        temporary.cleanup()


def test_fixture_exclusion_rejects_bottom_candidate() -> None:
    temporary, model, _ = _model("box")
    try:
        candidate = generate_bottom_sweep(model, ViewSweepSettings(), 1)
        fixture = trimesh.creation.box(extents=(1000, 1000, 500))
        fixture.apply_translation((0, 0, -250))
        path = Path(temporary.name) / "fixture.stl"
        fixture.export(path)
        assert not candidate_accessible(candidate, model, path)
    finally:
        temporary.cleanup()


def test_optimizer_rejects_duplicate_and_orders_mission() -> None:
    temporary, model, _ = _model("box")
    try:
        mission = ViewBasedSweepPlanner(ViewSweepSettings(isometric_recovery=False)).plan(model)
        bottom = generate_bottom_sweep(model, ViewSweepSettings(), 50)
        duplicate = RecoveryCandidate(51, "bottom_duplicate", "direct_bottom",
                                      bottom.sweep)
        result = optimise_scan_mission(
            model, tuple(mission.sweeps), (bottom, duplicate), (),
            CoverageSettings(laser_ray_count=5), minimum_gain_percent=0,
            maximum_optional_sweeps=2,
        )
        selected = [row for row in result.candidate_evaluations if row.selected]
        assert len(selected) <= 1
        names = [sweep.view_type for sweep in result.sweeps]
        assert names[0] == "top"
        if "bottom" in names:
            assert names[-1] == "bottom"
    finally:
        temporary.cleanup()


def test_feature_export_reports_and_pose_schema() -> None:
    temporary, _model_value, path = _model("through")
    try:
        result = run_pipeline(PipelineConfig(
            step_path=path, planning_mode="simple_view_based", display=False,
            output_dir=Path(temporary.name) / "out", laser_ray_count=5,
            travel_spacing_mm=10, maximum_views_per_hole=2,
            hole_candidate_tilt_angles_deg=(35,), hole_candidate_azimuth_count=2,
            maximum_optional_sweeps=1, enable_bottom_sweep=False,
            enable_lower_isometric_recovery=False,
        ))
        for key in ("feature_report_json", "hole_coverage_report_json",
                    "optimisation_report_json"):
            assert result.summary.output_files[key].is_file()
        trajectory = json.loads(result.summary.output_files["trajectory_json"].read_text())
        required = {
            "feature_id", "feature_type", "target_hole_id",
            "recovery_strategy", "candidate_score", "coverage_gain",
            "hole_coverage_before", "hole_coverage_after",
            "bottom_coverage_gain", "mandatory_or_optional",
        }
        assert required <= trajectory["trajectory"][0].keys()
    finally:
        temporary.cleanup()
