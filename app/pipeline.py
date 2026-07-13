"""Orchestration for the complete current B-Rep planning pipeline."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import cadquery as cq

from app.config import PipelineConfig
from app.result import PipelineResult, PipelineSummary
from cad_model import load_step_model
from coverage import (
    CoverageSettings,
    detect_exterior_face_ids,
    validate_coverage,
)
from exporter import export_pipeline_files
from face_analysis import SurfaceType, extract_face_records
from geometry_validation import validate_pose_geometry
from features import analyse_hole_coverage, detect_hole_features
from mission_optimizer import optimisation_payload, optimise_scan_mission
from planners.base import PlannerSettings, Pose
from planners.cylinder import plan_cylindrical_face
from planners.cone import plan_conical_face
from planners.sphere import plan_spherical_face
from planners.view_based import ViewBasedSweepPlanner, ViewSweepSettings
from planners.plane import plan_planar_face
from path_validation import validate_scan_segment
from regions import build_cylindrical_regions, build_planar_regions
from viewer import show_pipeline
from region_motion import assemble_region_motion
from recovery import (
    RecoveryCandidate, candidate_accessible, generate_bottom_sweep,
    generate_focused_conical_orbits, generate_hole_recovery_candidates,
    generate_lower_diagonal_sweeps,
    generate_lower_isometric,
)


def _output_paths(config: PipelineConfig) -> dict[str, Path]:
    output_dir = config.output_dir
    if output_dir is None:
        output_dir = config.step_path.resolve().parent / (
            f"{config.step_path.stem}_output"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "trajectory_csv": output_dir / "trajectory.csv",
        "trajectory_json": output_dir / "trajectory.json",
        "coverage_json": output_dir / "coverage.json",
        "validation_report_json": output_dir / "validation_report.json",
    }


def _scan_path_length(poses: tuple[Pose, ...]) -> float:
    by_segment: dict[int, list[Pose]] = defaultdict(list)
    for pose in poses:
        if pose.scan_enabled:
            by_segment[pose.segment_id].append(pose)
    length = 0.0
    for segment in by_segment.values():
        for first, second in zip(segment, segment[1:]):
            length += float(np.linalg.norm(second.position - first.position))
    return length


def _all_motion_poses(plan) -> tuple[Pose, ...]:
    return tuple(
        pose
        for motion in plan.motion_segments
        for pose in motion.poses
    )


def _annotate_and_filter_plan(
    plan,
    region_id: int,
    member_face_ids: tuple[int, ...],
    settings: PlannerSettings,
    source_faces=None,
):
    kept_motions = []
    validations = []
    region_motion_metrics = []
    for motion in plan.motion_segments:
        wrapped_sources = (
            [(face_id, cq.Face(face)) for face_id, face in source_faces]
            if source_faces else []
        )
        annotated = []
        for pose in motion.poses:
            face_id = pose.face_id
            if wrapped_sources:
                vertex = cq.Vertex.makeVertex(*pose.surface_point)
                face_id = min(
                    wrapped_sources,
                    key=lambda item: item[1].distance(vertex),
                )[0]
            annotated.append(
                replace(
                    pose,
                    face_id=face_id,
                    region_id=region_id,
                    member_face_ids=member_face_ids,
                )
            )
        annotated_poses = tuple(annotated)
        annotated_motion = replace(motion, poses=annotated_poses)
        validation = validate_scan_segment(
            annotated_motion, settings, expected_region_id=region_id
        )
        validations.append(validation)
        if validation.valid:
            kept_motions.append(annotated_motion)
    if hasattr(plan, "raster_segments"):
        plan = replace(
            plan,
            raster_segments=tuple(
                motion.raster_segment for motion in kept_motions
            ),
            motion_segments=tuple(kept_motions),
        )
    else:
        plan = replace(
            plan,
            path_segments=tuple(
                motion.raster_segment for motion in kept_motions
            ),
            motion_segments=tuple(kept_motions),
        )
    return plan, tuple(validations)


def _display_pipeline(result: PipelineResult, config: PipelineConfig) -> None:
    show_pipeline(result, config)


def _run_view_based_pipeline(config: PipelineConfig, model) -> PipelineResult:
    coverage_settings = CoverageSettings(
        laser_width_mm=config.laser_width_mm,
        laser_ray_count=config.laser_ray_count,
        min_range_mm=config.min_sensor_range_mm,
        max_range_mm=config.max_sensor_range_mm,
        max_incidence_angle_deg=config.max_incidence_angle_deg,
        include_internal_surfaces=config.include_internal_surfaces,
    )
    planner = ViewBasedSweepPlanner(ViewSweepSettings(
        stand_off_mm=config.stand_off_mm,
        travel_spacing_mm=config.travel_spacing_mm,
        laser_width_mm=config.laser_width_mm,
        overlap_fraction=config.sweep_overlap_fraction,
        top_sweep=config.top_sweep,
        front_sweep=config.front_sweep,
        right_sweep=config.right_sweep,
        back_sweep=config.back_sweep,
        left_sweep=config.left_sweep,
        isometric_recovery=config.isometric_recovery,
        isometric_elevation_deg=config.isometric_elevation_deg,
        isometric_azimuth_step_deg=config.isometric_azimuth_step_deg,
        maximum_isometric_sweeps=config.maximum_isometric_sweeps,
        minimum_recovery_gain_percent=config.minimum_recovery_gain_percent,
    ))
    mission = planner.plan(model, coverage_settings)
    holes = detect_hole_features(model) if config.enable_hole_recovery else ()
    internal_settings = replace(coverage_settings, include_internal_surfaces=True)
    exterior = detect_exterior_face_ids(model)
    initial_optical = validate_coverage(
        model, mission.poses, internal_settings, exterior_face_ids=exterior
    )
    initial_hole_reports = tuple(
        analyse_hole_coverage(model, hole, initial_optical,
                              config.hole_target_coverage_percent)
        for hole in holes
    )
    holes = tuple(
        replace(hole, current_optical_coverage_percent=report.cylindrical_wall_coverage_percent)
        for hole, report in zip(holes, initial_hole_reports)
    )
    mandatory = tuple(sweep for sweep in mission.sweeps
                      if sweep.view_type != "isometric")
    optionals: list[RecoveryCandidate] = [
        RecoveryCandidate(sweep.sweep_id, sweep.sweep_name,
                          "upper_isometric", sweep)
        for sweep in mission.sweeps if sweep.view_type == "isometric"
    ]
    next_id = max((sweep.sweep_id for sweep in mission.sweeps), default=-1) + 1
    if config.enable_hole_recovery:
        for hole, report in zip(holes, initial_hole_reports):
            focused = generate_focused_conical_orbits(
                hole, report, planner.settings, model=model,
                candidate_id_start=next_id,
                cone_angles_deg=config.focused_orbit_cone_angles_deg,
                focus_depth_fraction=config.focused_orbit_focus_depth_fraction,
                minimum_arc_deg=config.focused_orbit_minimum_arc_deg,
                maximum_arc_deg=config.focused_orbit_maximum_arc_deg,
                angular_spacing_deg=config.focused_orbit_angular_spacing_deg,
                maximum_orbits=config.maximum_focused_orbits_per_hole,
                laser_ray_count=config.laser_ray_count,
            ) if config.enable_focused_hole_orbit else ()
            remaining_views = max(0, config.maximum_views_per_hole - len(focused))
            legacy = generate_hole_recovery_candidates(
                hole, report, planner.settings,
                candidate_id_start=next_id + len(focused),
                tilt_angles_deg=config.hole_candidate_tilt_angles_deg,
                azimuth_count=config.hole_candidate_azimuth_count,
                maximum_views=remaining_views,
                enable_orbital=config.enable_orbital_hole_sweep,
                enable_blind_bottom=config.enable_blind_hole_bottom_recovery,
                orbital_mode=("disabled" if config.enable_focused_hole_orbit
                              else config.orbital_hole_mode),
            ) if remaining_views else ()
            generated = (*focused, *legacy)
            optionals.extend(generated)
            next_id += len(generated)
    fixture_path = (
        config.fixture_mesh_path if config.fixture_exclusion_enabled else None
    )
    if config.enable_lower_isometric_recovery:
        for lower_index in range(config.maximum_lower_recovery_sweeps):
            optionals.append(generate_lower_isometric(
                model, planner.settings, next_id,
                min(80.0, config.lower_isometric_elevation_deg + 5.0 * lower_index),
            )); next_id += 1
        optionals.extend(generate_lower_diagonal_sweeps(
            model, planner.settings, next_id,
            config.lower_isometric_elevation_deg,
        )); next_id += 2
    if config.enable_bottom_sweep:
        optionals.append(generate_bottom_sweep(model, planner.settings, next_id))
    optionals = [candidate for candidate in optionals if candidate_accessible(
        candidate, model, fixture_path, config.bottom_clearance_mm
    )]
    optimisation = optimise_scan_mission(
        model, mandatory, tuple(optionals), holes, coverage_settings,
        target_total_coverage_percent=config.total_target_coverage_percent,
        target_hole_coverage_percent=config.hole_target_coverage_percent,
        minimum_gain_percent=config.minimum_candidate_gain_percent,
        maximum_optional_sweeps=config.maximum_optional_sweeps,
        floating_part_assumption=not config.fixture_exclusion_enabled,
    )
    sweeps = optimisation.sweeps
    poses = tuple(pose for sweep in sweeps for pose in sweep.poses)
    coverage = optimisation.coverage
    output_files = _output_paths(config)
    output_dir = next(iter(output_files.values())).parent
    output_files.update({
        "feature_report_json": output_dir / "feature_report.json",
        "hole_coverage_report_json": output_dir / "hole_coverage_report.json",
        "optimisation_report_json": output_dir / "optimisation_report.json",
    })
    geometry_validation = validate_pose_geometry(poses) if config.geometry_validation else None
    summary = PipelineSummary(
        total_step_faces=len(model.faces), selected_planar_faces=0,
        selected_cylindrical_faces=0, selected_conical_faces=0,
        selected_spherical_faces=0,
        skipped_faces_by_reason={"view-based mission does not select individual faces":
                                 tuple(face.face_id for face in model.faces)},
        generated_scan_regions=len(sweeps),
        generated_scan_segments=len(sweeps),
        scan_pose_count=sum(p.scan_enabled for p in poses),
        total_scan_path_length_mm=_scan_path_length(poses),
        coverage_percent=coverage.report["total_exterior_surface"].coverage_percent,
        invalid_segment_count=len(coverage.invalid_segment_ids),
        output_files=output_files,
    )
    validation_payload = {
        "summary": {**asdict(summary), "output_files": {
            key: str(value) for key, value in output_files.items()}},
        "planning_mode": "simple_view_based",
        "sweep_order": [sweep.sweep_name for sweep in sweeps],
        "coverage_contribution_area_mm2": mission.coverage_contribution_area_mm2,
        "usable_laser_line_fraction": mission.usable_laser_line_fraction,
        "surface_coverage": {name: asdict(row) for name, row in coverage.report.items()},
        "safety_notice": (
            "Cartesian optical planning only; no UR5 IK, joint-limit, full "
            "collision, MoveIt2, or ROS 2 validation has been performed."
        ),
    }
    validation_payload["feature_report"] = {
        "floating_part_assumption": not config.fixture_exclusion_enabled,
        "holes": [
            {**asdict(hole),
             "centre": hole.centre.tolist(), "axis": hole.axis.tolist(),
             "entry_opening_centre": hole.entry_opening_centre.tolist(),
             "entry_opening_normal": hole.entry_opening_normal.tolist()}
            for hole in holes
        ],
    }
    validation_payload["hole_coverage_report"] = {
        "target_percent": config.hole_target_coverage_percent,
        "before": [asdict(row) for row in optimisation.hole_reports_before],
        "after": [asdict(row) for row in optimisation.hole_reports_after],
    }
    validation_payload["optimisation_report"] = optimisation_payload(optimisation)
    export_pipeline_files(output_files, config, poses, coverage, validation_payload,
                          model.validation_mesh.triangle_face_ids)
    result = PipelineResult(
        model=model, planar_plans=(), cylindrical_plans=(), conical_plans=(),
        spherical_plans=(), poses=poses, coverage=coverage,
        baseline_coverage=mission.principal_coverage or coverage,
        geometry_validation=geometry_validation, summary=summary,
        sweeps=sweeps, hole_features=holes,
        hole_coverage_reports=optimisation.hole_reports_after,
        optimisation_report=optimisation,
    )
    if config.display:
        _display_pipeline(result, config)
    return result


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Build regions, reject discontinuous segments, validate, and export."""
    config.validate()
    model = load_step_model(config.step_path)
    if config.planning_mode == "simple_view_based":
        return _run_view_based_pipeline(config, model)
    exterior_face_ids = detect_exterior_face_ids(model)
    externally_eligible = (
        set(range(len(model.faces)))
        if config.include_internal_surfaces
        else set(exterior_face_ids)
    )
    requested = set(config.surface_types)
    skipped: dict[str, list[int]] = defaultdict(list)
    selected_planar: set[int] = set()
    selected_cylindrical: set[int] = set()
    selected_conical: set[int] = set()
    selected_spherical: set[int] = set()
    for record in model.faces:
        name = (
            "plane" if record.geometry_type is SurfaceType.PLANE else
            "cylinder" if record.geometry_type is SurfaceType.CYLINDER else
            "cone" if record.geometry_type is SurfaceType.CONE else None
            if record.geometry_type is not SurfaceType.SPHERE else "sphere"
        )
        if name is None:
            skipped[f"unsupported surface type: {record.geometry_type.value}"].append(record.face_id)
        elif name not in requested:
            skipped["surface type not requested"].append(record.face_id)
        elif record.area_mm2 < config.minimum_face_area_mm2:
            skipped["below minimum face area"].append(record.face_id)
        elif record.face_id not in externally_eligible:
            skipped["internal surface excluded"].append(record.face_id)
        elif name == "plane":
            selected_planar.add(record.face_id)
        elif name == "cylinder":
            selected_cylindrical.add(record.face_id)
        elif name == "cone":
            selected_conical.add(record.face_id)
        else:
            selected_spherical.add(record.face_id)

    planner_settings = PlannerSettings(
        lane_spacing_mm=config.lane_spacing_mm,
        travel_spacing_mm=config.travel_spacing_mm,
        stand_off_mm=config.stand_off_mm,
    )
    coverage_settings = CoverageSettings(
        laser_width_mm=config.laser_width_mm,
        laser_ray_count=config.laser_ray_count,
        min_range_mm=config.min_sensor_range_mm,
        max_range_mm=config.max_sensor_range_mm,
        max_incidence_angle_deg=config.max_incidence_angle_deg,
        include_internal_surfaces=config.include_internal_surfaces,
    )

    # Baseline per-face plans are retained only for before/after metrics.
    baseline_planar = [
        plan_planar_face(model.topods_faces[i], model.faces[i], planner_settings)
        for i in sorted(selected_planar)
    ]
    baseline_cylindrical = [
        plan_cylindrical_face(
            model.topods_faces[i], model.faces[i], planner_settings,
            strategy=config.cylinder_strategy,
        )
        for i in sorted(selected_cylindrical)
    ]
    baseline_conical = [
        plan_conical_face(model.topods_faces[i], model.faces[i], planner_settings)
        for i in sorted(selected_conical)
    ]
    baseline_spherical = [
        plan_spherical_face(model.topods_faces[i], model.faces[i], planner_settings)
        for i in sorted(selected_spherical)
    ]
    baseline_planar_poses = tuple(
        pose for plan in baseline_planar for pose in _all_motion_poses(plan)
    )
    baseline_cylindrical_poses = tuple(
        pose for plan in baseline_cylindrical for pose in _all_motion_poses(plan)
    )
    baseline_conical_poses = tuple(
        pose for plan in baseline_conical for pose in _all_motion_poses(plan)
    )
    baseline_spherical_poses = tuple(
        pose for plan in baseline_spherical for pose in _all_motion_poses(plan)
    )

    planar_regions = build_planar_regions(
        model.topods_faces, model.faces, selected_planar,
        normal_tolerance_deg=config.planar_normal_tolerance_deg,
        plane_distance_tolerance_mm=config.plane_distance_tolerance_mm,
    )
    cylindrical_regions = build_cylindrical_regions(
        model.topods_faces, model.faces, selected_cylindrical,
        axis_angle_tolerance_deg=config.cylinder_axis_angle_tolerance_deg,
        axis_distance_tolerance_mm=config.cylinder_axis_distance_tolerance_mm,
        radius_tolerance_mm=config.cylinder_radius_tolerance_mm,
        region_id_start=len(planar_regions),
    )
    planar_plans = []
    cylindrical_plans = []
    conical_plans = []
    spherical_plans = []
    validations = []
    region_motion_metrics = []
    next_segment_id = 0
    for region in planar_regions:
        for planning_face in region.planning_faces:
            record = replace(
                extract_face_records((planning_face,))[0],
                face_id=region.member_face_ids[0],
            )
            plan = plan_planar_face(
                planning_face, record, planner_settings,
                segment_id_start=next_segment_id,
            )
            plan, rows = _annotate_and_filter_plan(
                plan,
                region.region_id,
                region.member_face_ids,
                planner_settings,
                source_faces=[
                    (i, model.topods_faces[i])
                    for i in region.member_face_ids
                ],
            )
            validations.extend(rows)
            combined, metrics = assemble_region_motion(
                plan.motion_segments,
                planner_settings,
                segment_id_start=next_segment_id,
            )
            region_motion_metrics.append(metrics)
            plan = replace(plan, motion_segments=combined)
            planar_plans.append(plan)
            next_segment_id += len(plan.motion_segments)
    for region in cylindrical_regions:
        for planning_face in region.planning_faces:
            record = replace(
                extract_face_records((planning_face,))[0],
                face_id=region.member_face_ids[0],
            )
            if record.geometry_type is not SurfaceType.CYLINDER:
                continue
            plan = plan_cylindrical_face(
                planning_face, record, planner_settings,
                strategy=config.cylinder_strategy,
                segment_id_start=next_segment_id,
            )
            plan, rows = _annotate_and_filter_plan(
                plan,
                region.region_id,
                region.member_face_ids,
                planner_settings,
                source_faces=[
                    (i, model.topods_faces[i])
                    for i in region.member_face_ids
                ],
            )
            validations.extend(rows)
            combined, metrics = assemble_region_motion(
                plan.motion_segments,
                planner_settings,
                segment_id_start=next_segment_id,
            )
            region_motion_metrics.append(metrics)
            plan = replace(plan, motion_segments=combined)
            cylindrical_plans.append(plan)
            next_segment_id += len(plan.motion_segments)
    for face_id in sorted(selected_conical):
        plan = plan_conical_face(
            model.topods_faces[face_id],
            model.faces[face_id],
            planner_settings,
            segment_id_start=next_segment_id,
        )
        region_id = len(planar_regions) + len(cylindrical_regions) + len(conical_plans)
        plan, rows = _annotate_and_filter_plan(
            plan, region_id, (face_id,), planner_settings,
            source_faces=[(face_id, model.topods_faces[face_id])],
        )
        validations.extend(rows)
        combined, metrics = assemble_region_motion(
            plan.motion_segments, planner_settings,
            segment_id_start=next_segment_id,
        )
        region_motion_metrics.append(metrics)
        plan = replace(plan, motion_segments=combined)
        conical_plans.append(plan)
        next_segment_id += len(plan.motion_segments)
    for face_id in sorted(selected_spherical):
        plan = plan_spherical_face(
            model.topods_faces[face_id], model.faces[face_id],
            planner_settings, segment_id_start=next_segment_id,
        )
        region_id = len(planar_regions)+len(cylindrical_regions)+len(conical_plans)+len(spherical_plans)
        plan, rows = _annotate_and_filter_plan(
            plan, region_id, (face_id,), planner_settings,
            source_faces=[(face_id, model.topods_faces[face_id])],
        )
        validations.extend(rows)
        combined, metrics = assemble_region_motion(
            plan.motion_segments, planner_settings, segment_id_start=next_segment_id
        )
        region_motion_metrics.append(metrics)
        plan=replace(plan,motion_segments=combined)
        spherical_plans.append(plan)
        next_segment_id += len(plan.motion_segments)

    poses = tuple(
        pose for plan in (*planar_plans, *cylindrical_plans, *conical_plans, *spherical_plans)
        for pose in _all_motion_poses(plan)
    )
    merged_planar_poses = tuple(
        pose for plan in planar_plans for pose in _all_motion_poses(plan)
    )
    merged_cylindrical_poses = tuple(
        pose for plan in cylindrical_plans for pose in _all_motion_poses(plan)
    )
    validate = lambda candidate_poses: validate_coverage(
        model, candidate_poses, coverage_settings,
        exterior_face_ids=exterior_face_ids,
    )
    baseline_planar_coverage = validate(baseline_planar_poses)
    baseline_cylindrical_coverage = validate(baseline_cylindrical_poses)
    merged_planar_coverage = validate(merged_planar_poses)
    merged_cylindrical_coverage = validate(merged_cylindrical_poses)
    coverage = validate(poses)
    baseline_coverage = validate(
        (*baseline_planar_poses, *baseline_cylindrical_poses, *baseline_conical_poses, *baseline_spherical_poses)
    )
    geometry_validation = (
        validate_pose_geometry(poses) if config.geometry_validation else None
    )
    output_files = _output_paths(config)
    summary = PipelineSummary(
        total_step_faces=len(model.faces),
        selected_planar_faces=len(selected_planar),
        selected_cylindrical_faces=len(selected_cylindrical),
        selected_conical_faces=len(selected_conical),
        selected_spherical_faces=len(selected_spherical),
        skipped_faces_by_reason={
            reason: tuple(face_ids) for reason, face_ids in sorted(skipped.items())
        },
        generated_scan_regions=len(planar_regions)+len(cylindrical_regions)+len(conical_plans)+len(spherical_plans),
        generated_scan_segments=next_segment_id,
        scan_pose_count=sum(pose.scan_enabled for pose in poses),
        total_scan_path_length_mm=_scan_path_length(poses),
        coverage_percent=coverage.report[
            "total_exterior_surface"
        ].coverage_percent,
        invalid_segment_count=sum(not row.valid for row in validations),
        output_files=output_files,
    )
    validation_payload = {
        "summary": {
            **asdict(summary),
            "output_files": {
                name: str(path) for name, path in output_files.items()
            },
        },
        "coverage_counters": {
            "scan_pose_count": coverage.scan_pose_count,
            "ray_count": coverage.ray_count,
            "first_hit_count": coverage.first_hit_count,
            "valid_hit_count": coverage.valid_hit_count,
            "invalid_range_hit_count": coverage.invalid_range_hit_count,
            "invalid_incidence_hit_count": coverage.invalid_incidence_hit_count,
            "valid_segment_ids": sorted(coverage.valid_segment_ids),
            "invalid_segment_ids": sorted(coverage.invalid_segment_ids),
        },
        "surface_coverage": {
            name: asdict(row) for name, row in coverage.report.items()
        },
        "region_merging": {
            "planar_coverage_before_percent": baseline_planar_coverage.report["planar_faces"].coverage_percent,
            "planar_coverage_after_percent": merged_planar_coverage.report["planar_faces"].coverage_percent,
            "cylindrical_coverage_before_percent": baseline_cylindrical_coverage.report["cylindrical_faces"].coverage_percent,
            "cylindrical_coverage_after_percent": merged_cylindrical_coverage.report["cylindrical_faces"].coverage_percent,
            "scan_path_length_mm": _scan_path_length(poses),
            "scan_segment_count": next_segment_id,
            "scan_pose_count": sum(p.scan_enabled for p in poses),
            "estimated_reduction_in_approach_retract_motions": 2 * max(
                0,
                sum(len(p.motion_segments) for p in (*baseline_planar, *baseline_cylindrical))
                - next_segment_id,
            ),
            "planar_regions": [
                {"region_id": r.region_id, "member_face_ids": list(r.member_face_ids), "total_area_mm2": r.total_area_mm2}
                for r in planar_regions
            ],
            "cylindrical_regions": [
                {"region_id": r.region_id, "member_face_ids": list(r.member_face_ids), "total_area_mm2": r.total_area_mm2}
                for r in cylindrical_regions
            ],
        },
        "continuous_region_paths": {
            "old_segment_count": sum(m.old_segment_count for m in region_motion_metrics),
            "new_region_level_segment_count": sum(m.new_segment_count for m in region_motion_metrics),
            "old_approach_retract_count": sum(m.old_approach_retract_count for m in region_motion_metrics),
            "new_approach_retract_count": sum(m.new_approach_retract_count for m in region_motion_metrics),
            "lane_transition_count": sum(m.lane_transition_count for m in region_motion_metrics),
            "transition_distance_mm": sum(m.transition_distance_mm for m in region_motion_metrics),
            "estimated_cycle_time_reduction_percent": (
                100.0 * (
                    1.0 - sum(m.new_approach_retract_count for m in region_motion_metrics)
                    / max(1, sum(m.old_approach_retract_count for m in region_motion_metrics))
                )
            ),
        },
        "segment_rejections": [
            {"segment_id": row.segment_id, "reasons": list(row.reasons)}
            for row in validations if not row.valid
        ],
        "safety_notice": (
            "Cartesian planning only; no UR5 IK, collision checking, ROS 2, "
            "or mission optimisation has been performed."
        ),
    }
    if geometry_validation is not None:
        validation_payload["geometry_validation"] = {
            "pose_stand_offs": [
                {
                    "face_id": metric.face_id,
                    "segment_id": metric.segment_id,
                    "pose_id": metric.pose_id,
                    "stand_off_mm": metric.stand_off_mm,
                    "connector_status": metric.connector_status.value,
                }
                for metric in geometry_validation.pose_metrics
            ],
            "statistics": {
                "minimum_stand_off_mm": (
                    geometry_validation.minimum_stand_off_mm
                ),
                "maximum_stand_off_mm": (
                    geometry_validation.maximum_stand_off_mm
                ),
                "mean_stand_off_mm": geometry_validation.mean_stand_off_mm,
                "stand_off_standard_deviation_mm": (
                    geometry_validation.stand_off_standard_deviation_mm
                ),
                "maximum_orientation_change_deg": (
                    geometry_validation.maximum_orientation_change_deg
                ),
                "maximum_position_change_mm": (
                    geometry_validation.maximum_position_change_mm
                ),
            },
            "segments": [
                asdict(row)
                for row in geometry_validation.segment_statistics
            ],
        }
    export_pipeline_files(
        output_files,
        config,
        poses,
        coverage,
        validation_payload,
        model.validation_mesh.triangle_face_ids,
    )
    result = PipelineResult(
        model=model,
        planar_plans=tuple(planar_plans),
        cylindrical_plans=tuple(cylindrical_plans),
        conical_plans=tuple(conical_plans),
        spherical_plans=tuple(spherical_plans),
        poses=poses,
        coverage=coverage,
        baseline_coverage=baseline_coverage,
        geometry_validation=geometry_validation,
        summary=summary,
    )
    if config.display:
        _display_pipeline(result, config)
    return result
