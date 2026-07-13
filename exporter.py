"""CSV and JSON exports for the unified B-Rep pipeline."""

from __future__ import annotations

import csv
import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from coverage import CoverageResult
from planners.base import Pose


def _pose_row(pose: Pose) -> dict[str, Any]:
    rotation = Rotation.from_quat(pose.quaternion_xyzw)
    # RPY is export-only. SciPy selects a valid zero-third-angle convention at
    # gimbal lock; quaternions remain the authoritative orientation.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="Gimbal lock detected", category=UserWarning
        )
        rpy = rotation.as_euler("xyz", degrees=True)
    rotvec = rotation.as_rotvec()
    laser_direction = (
        np.asarray(pose.laser_line_direction, dtype=float)
        if pose.laser_line_direction is not None else rotation.apply([1.0, 0.0, 0.0])
    )
    travel_direction = (
        np.asarray(pose.travel_direction, dtype=float)
        if pose.travel_direction is not None else rotation.apply([0.0, 1.0, 0.0])
    )
    predicted_stand_off = (
        pose.predicted_stand_off_mm if pose.predicted_stand_off_mm is not None
        else float(np.linalg.norm(pose.position - pose.surface_point))
    )
    return {
        "sweep_id": pose.sweep_id,
        "sweep_name": pose.sweep_name,
        "view_type": pose.view_type,
        "face_id": pose.face_id,
        "region_id": pose.region_id,
        "member_face_ids": list(pose.member_face_ids),
        "segment_id": pose.segment_id,
        "pose_id": pose.pose_id,
        "motion_type": pose.motion_type.value,
        "scan_enabled": pose.scan_enabled,
        "surface_x": float(pose.surface_point[0]),
        "surface_y": float(pose.surface_point[1]),
        "surface_z": float(pose.surface_point[2]),
        "normal_x": float(pose.surface_normal[0]),
        "normal_y": float(pose.surface_normal[1]),
        "normal_z": float(pose.surface_normal[2]),
        "x": float(pose.position[0]),
        "y": float(pose.position[1]),
        "z": float(pose.position[2]),
        "qx": float(pose.quaternion_xyzw[0]),
        "qy": float(pose.quaternion_xyzw[1]),
        "qz": float(pose.quaternion_xyzw[2]),
        "qw": float(pose.quaternion_xyzw[3]),
        "rx": float(rotvec[0]),
        "ry": float(rotvec[1]),
        "rz": float(rotvec[2]),
        "roll_deg": float(rpy[0]),
        "pitch_deg": float(rpy[1]),
        "yaw_deg": float(rpy[2]),
        "laser_line_x": float(laser_direction[0]),
        "laser_line_y": float(laser_direction[1]),
        "laser_line_z": float(laser_direction[2]),
        "travel_x": float(travel_direction[0]),
        "travel_y": float(travel_direction[1]),
        "travel_z": float(travel_direction[2]),
        "predicted_stand_off_mm": float(predicted_stand_off),
        "predicted_visible_laser_width_mm": (
            None if pose.predicted_visible_laser_width_mm is None
            else float(pose.predicted_visible_laser_width_mm)
        ),
        "laser_line_direction": [float(value) for value in laser_direction],
        "travel_direction": [float(value) for value in travel_direction],
        "feature_id": pose.feature_id,
        "feature_type": pose.feature_type,
        "target_hole_id": pose.target_hole_id,
        "recovery_strategy": pose.recovery_strategy,
        "candidate_score": pose.candidate_score,
        "coverage_gain": pose.coverage_gain,
        "hole_coverage_before": pose.hole_coverage_before,
        "hole_coverage_after": pose.hole_coverage_after,
        "bottom_coverage_gain": pose.bottom_coverage_gain,
        "mandatory_or_optional": pose.mandatory_or_optional,
    }


def _config_payload(config: PipelineConfig) -> dict[str, Any]:
    return {
        "source_step": str(config.step_path),
        "planning_mode": config.planning_mode,
        "surface_types": list(config.surface_types),
        "lane_spacing_mm": config.lane_spacing_mm,
        "travel_spacing_mm": config.travel_spacing_mm,
        "stand_off_mm": config.stand_off_mm,
        "minimum_face_area_mm2": config.minimum_face_area_mm2,
        "include_internal_surfaces": config.include_internal_surfaces,
        "cylinder_strategy": config.cylinder_strategy.value,
        "geometry_validation": config.geometry_validation,
        "laser_width_mm": config.laser_width_mm,
        "laser_ray_count": config.laser_ray_count,
        "min_sensor_range_mm": config.min_sensor_range_mm,
        "max_sensor_range_mm": config.max_sensor_range_mm,
        "max_incidence_angle_deg": config.max_incidence_angle_deg,
        "sweep_overlap_fraction": config.sweep_overlap_fraction,
        "isometric_elevation_deg": config.isometric_elevation_deg,
        "isometric_azimuth_step_deg": config.isometric_azimuth_step_deg,
        "maximum_isometric_sweeps": config.maximum_isometric_sweeps,
        "minimum_recovery_gain_percent": config.minimum_recovery_gain_percent,
        "enable_hole_recovery": config.enable_hole_recovery,
        "hole_target_coverage_percent": config.hole_target_coverage_percent,
        "enable_bottom_sweep": config.enable_bottom_sweep,
        "enable_lower_isometric_recovery": config.enable_lower_isometric_recovery,
        "total_target_coverage_percent": config.total_target_coverage_percent,
    }


def export_pipeline_files(
    output_files: dict[str, Path],
    config: PipelineConfig,
    poses: tuple[Pose, ...],
    coverage: CoverageResult,
    validation_payload: dict[str, Any],
    triangle_face_ids: np.ndarray,
) -> None:
    rows = [_pose_row(pose) for pose in poses]
    csv_path = output_files["trajectory_csv"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    trajectory_payload = {
        "coordinate_frame": "CAD_STEP",
        "position_unit": "mm",
        "quaternion_order": "xyzw",
        "euler_sequence": "xyz",
        "configuration": _config_payload(config),
        "trajectory": rows,
    }
    output_files["trajectory_json"].write_text(
        json.dumps(trajectory_payload, indent=2), encoding="utf-8"
    )

    coverage_payload = {
        "configuration": _config_payload(config),
        "surface_report": {
            name: asdict(row) for name, row in coverage.report.items()
        },
        "exterior_face_ids": sorted(coverage.exterior_face_ids),
        "internal_face_ids": sorted(coverage.internal_face_ids),
        "triangle_face_ids": triangle_face_ids.tolist(),
        "triangle_status": coverage.triangle_status.tolist(),
        "status_codes": {
            "0": "uncovered",
            "1": "valid_covered",
            "2": "visible_invalid_range_or_incidence",
        },
    }
    output_files["coverage_json"].write_text(
        json.dumps(coverage_payload, indent=2), encoding="utf-8"
    )
    output_files["validation_report_json"].write_text(
        json.dumps(validation_payload, indent=2), encoding="utf-8"
    )
    for key, payload_key in (
        ("feature_report_json", "feature_report"),
        ("hole_coverage_report_json", "hole_coverage_report"),
        ("optimisation_report_json", "optimisation_report"),
    ):
        if key in output_files:
            output_files[key].write_text(
                json.dumps(validation_payload.get(payload_key, {}), indent=2),
                encoding="utf-8",
            )


def export_staged_trajectory(
    path: str | Path,
    config: PipelineConfig,
    poses: tuple[Pose, ...],
) -> Path:
    """Export the current GUI-stage trajectory without running validation."""

    destination = Path(path)
    rows = [_pose_row(pose) for pose in poses]
    if destination.suffix.lower() == ".csv":
        with destination.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
            if rows:
                writer.writeheader()
                writer.writerows(rows)
    elif destination.suffix.lower() == ".json":
        destination.write_text(json.dumps({
            "coordinate_frame": "CAD_STEP",
            "position_unit": "mm",
            "quaternion_order": "xyzw",
            "euler_sequence": "xyz",
            "configuration": _config_payload(config),
            "trajectory": rows,
            "staged_gui_export": True,
        }, indent=2), encoding="utf-8")
    else:
        raise ValueError("Staged trajectory export requires .csv or .json.")
    return destination
