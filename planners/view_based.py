"""Minimal profile-scanner missions based on principal projected views."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil

import numpy as np
from scipy.spatial.transform import Rotation

from cad_model import CadModel
from coverage import (
    CoverageResult, CoverageSettings, TriangleStatus, detect_exterior_face_ids,
    validate_coverage,
)
from planners.base import MotionType, Pose


@dataclass(frozen=True)
class ViewSweepSettings:
    stand_off_mm: float = 310.0
    travel_spacing_mm: float = 5.0
    laser_width_mm: float = 200.0
    overlap_fraction: float = 0.20
    approach_clearance_mm: float = 120.0
    lead_distance_mm: float = 20.0
    top_sweep: bool = True
    front_sweep: bool = True
    right_sweep: bool = True
    back_sweep: bool = True
    left_sweep: bool = True
    isometric_recovery: bool = True
    isometric_elevation_deg: float = 35.0
    isometric_azimuth_step_deg: float = 10.0
    maximum_isometric_sweeps: int = 1
    minimum_recovery_gain_percent: float = 0.5

    def validate(self) -> None:
        if self.stand_off_mm <= 0 or self.travel_spacing_mm <= 0:
            raise ValueError("Stand-off and travel spacing must be positive.")
        if self.laser_width_mm <= 0:
            raise ValueError("Laser-line width must be positive.")
        if not 0 <= self.overlap_fraction < 1:
            raise ValueError("Overlap must be in the range [0, 1).")
        if not 0 < self.isometric_elevation_deg < 90:
            raise ValueError("Isometric elevation must be between 0 and 90 degrees.")
        if self.isometric_azimuth_step_deg <= 0:
            raise ValueError("Isometric azimuth step must be positive.")
        if self.maximum_isometric_sweeps < 0:
            raise ValueError("Maximum isometric sweeps must not be negative.")
        if self.minimum_recovery_gain_percent < 0:
            raise ValueError("Minimum recovery gain must not be negative.")


@dataclass(frozen=True)
class ViewSweep:
    sweep_id: int
    sweep_name: str
    view_type: str
    pass_index: int
    poses: tuple[Pose, ...]
    colour: str

    @property
    def scan_poses(self) -> tuple[Pose, ...]:
        return tuple(pose for pose in self.poses if pose.scan_enabled)


@dataclass(frozen=True)
class ViewBasedMission:
    sweeps: tuple[ViewSweep, ...]
    principal_coverage: CoverageResult | None
    final_coverage: CoverageResult | None
    coverage_contribution_area_mm2: dict[str, float]
    usable_laser_line_fraction: dict[str, float]

    @property
    def poses(self) -> tuple[Pose, ...]:
        return tuple(pose for sweep in self.sweeps for pose in sweep.poses)


VIEW_COLOURS = {
    "top": "deepskyblue", "front": "limegreen", "right": "orange",
    "back": "magenta", "left": "gold", "isometric": "violet",
}


def minimum_pass_count(projected_width_mm: float, laser_width_mm: float,
                       overlap_fraction: float = 0.20) -> int:
    if projected_width_mm <= laser_width_mm:
        return 1
    pitch = laser_width_mm * (1.0 - overlap_fraction)
    return 1 + int(ceil((projected_width_mm - laser_width_mm) / pitch))


def _quaternion(x_axis: np.ndarray, y_axis: np.ndarray,
                z_axis: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.column_stack((x_axis, y_axis, z_axis))
    return tuple(float(value) for value in Rotation.from_matrix(matrix).as_quat())


def _sample_values(start: float, end: float, spacing: float) -> np.ndarray:
    count = max(2, int(ceil(abs(end - start) / spacing)) + 1)
    return np.linspace(start, end, count)


def _lane_centres(minimum: float, maximum: float, count: int,
                  laser_width: float) -> np.ndarray:
    if count == 1:
        return np.asarray([(minimum + maximum) * 0.5])
    inset = min(laser_width * 0.5, (maximum - minimum) * 0.5)
    return np.linspace(minimum + inset, maximum - inset, count)


def _bounds(model: CadModel) -> tuple[np.ndarray, np.ndarray]:
    box = model.shape.BoundingBox()
    return (
        np.asarray([box.xmin, box.ymin, box.zmin], dtype=float),
        np.asarray([box.xmax, box.ymax, box.zmax], dtype=float),
    )


def _principal_sweeps(
    model: CadModel, settings: ViewSweepSettings, cancellation_check=None
) -> list[ViewSweep]:
    cancellation_check = cancellation_check or (lambda: None)
    minimum, maximum = _bounds(model)
    centre = 0.5 * (minimum + maximum)
    definitions = (
        ("top", np.array([0., 0., -1.]), 2, settings.top_sweep),
        ("front", np.array([0., 1., 0.]), 1, settings.front_sweep),
        ("right", np.array([-1., 0., 0.]), 0, settings.right_sweep),
        ("back", np.array([0., -1., 0.]), 1, settings.back_sweep),
        ("left", np.array([1., 0., 0.]), 0, settings.left_sweep),
    )
    sweeps: list[ViewSweep] = []
    sweep_id = 0
    for view, measurement, normal_axis, enabled in definitions:
        cancellation_check()
        if not enabled:
            continue
        tangent_axes = [axis for axis in range(3) if axis != normal_axis]
        dimensions = maximum - minimum
        travel_axis_index = max(tangent_axes, key=lambda axis: dimensions[axis])
        laser_axis_index = next(axis for axis in tangent_axes if axis != travel_axis_index)
        laser = np.zeros(3)
        laser[laser_axis_index] = 1.0
        travel = np.cross(measurement, laser)
        if travel[travel_axis_index] < 0:
            laser *= -1.0
            travel = np.cross(measurement, laser)
        pass_count = minimum_pass_count(
            dimensions[laser_axis_index], settings.laser_width_mm,
            settings.overlap_fraction,
        )
        centres = _lane_centres(
            minimum[laser_axis_index], maximum[laser_axis_index], pass_count,
            settings.laser_width_mm,
        )
        envelope = maximum[normal_axis] if measurement[normal_axis] < 0 else minimum[normal_axis]
        for pass_index, lane in enumerate(centres):
            cancellation_check()
            start, end = minimum[travel_axis_index], maximum[travel_axis_index]
            travel_sign = float(travel[travel_axis_index])
            if travel_sign < 0:
                start, end = end, start
            values = _sample_values(start, end, settings.travel_spacing_mm)
            base = centre.copy()
            base[normal_axis] = envelope
            base[laser_axis_index] = lane
            quaternion = _quaternion(laser, travel, measurement)
            visible_width = min(settings.laser_width_mm, dimensions[laser_axis_index])
            name = view if pass_count == 1 else f"{view}_{pass_index + 1}"

            def make_pose(value, motion, scan, pose_id, outward_extra=0.0):
                surface = base.copy()
                surface[travel_axis_index] = value
                position = surface - (settings.stand_off_mm + outward_extra) * measurement
                return Pose(
                    face_id=-1, segment_id=sweep_id, pose_id=pose_id,
                    motion_type=motion, scan_enabled=scan,
                    surface_point=surface, surface_normal=-measurement,
                    position=position, quaternion_xyzw=quaternion,
                    region_id=sweep_id, sweep_id=sweep_id, sweep_name=name,
                    view_type=view, laser_line_direction=tuple(laser),
                    travel_direction=tuple(travel),
                    predicted_stand_off_mm=settings.stand_off_mm,
                    predicted_visible_laser_width_mm=visible_width,
                )

            lead_start = start - travel_sign * settings.lead_distance_mm
            lead_end = end + travel_sign * settings.lead_distance_mm
            poses = [
                make_pose(lead_start, MotionType.APPROACH, False, 0,
                          settings.approach_clearance_mm),
                make_pose(lead_start, MotionType.LEAD_IN, False, 1),
            ]
            poses.extend(make_pose(value, MotionType.SCAN, True, index + 2)
                         for index, value in enumerate(values))
            poses.extend((
                make_pose(lead_end, MotionType.LEAD_OUT, False, len(poses)),
                make_pose(lead_end, MotionType.RETRACT, False, len(poses) + 1,
                          settings.approach_clearance_mm),
            ))
            sweeps.append(ViewSweep(sweep_id, name, view, pass_index,
                                    tuple(poses), VIEW_COLOURS[view]))
            sweep_id += 1
    return sweeps


def _isometric_candidate(model: CadModel, settings: ViewSweepSettings,
                         sweep_id: int, recovery_index: int) -> ViewSweep:
    minimum, maximum = _bounds(model)
    centre = 0.5 * (minimum + maximum)
    radius = 0.5 * float(np.linalg.norm(maximum - minimum)) + settings.stand_off_mm
    elevation = np.deg2rad(min(80.0, settings.isometric_elevation_deg + 5.0 * recovery_index))
    azimuths = np.deg2rad(np.arange(0.0, 360.0 + settings.isometric_azimuth_step_deg,
                                    settings.isometric_azimuth_step_deg))
    poses: list[Pose] = []
    previous_quaternion: np.ndarray | None = None
    for index, azimuth in enumerate(azimuths):
        radial = np.array([np.cos(elevation) * np.cos(azimuth),
                           np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
        position = centre + radius * radial
        measurement = -radial
        laser = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
        laser /= np.linalg.norm(laser)
        y_axis = np.cross(measurement, laser)
        y_axis /= np.linalg.norm(y_axis)
        surface = position + settings.stand_off_mm * measurement
        quaternion = np.asarray(_quaternion(laser, y_axis, measurement))
        if previous_quaternion is not None and np.dot(previous_quaternion, quaternion) < 0:
            quaternion *= -1.0
        previous_quaternion = quaternion
        poses.append(Pose(
            face_id=-1, segment_id=sweep_id, pose_id=index + 2,
            motion_type=MotionType.SCAN, scan_enabled=True,
            surface_point=surface, surface_normal=-measurement,
            position=position, quaternion_xyzw=tuple(quaternion),
            region_id=sweep_id, sweep_id=sweep_id,
            sweep_name=f"isometric_{recovery_index + 1}", view_type="isometric",
            laser_line_direction=tuple(laser), travel_direction=tuple(laser),
            predicted_stand_off_mm=settings.stand_off_mm,
            predicted_visible_laser_width_mm=settings.laser_width_mm,
        ))
    if poses:
        first, last = poses[0], poses[-1]
        poses.insert(0, replace(
            first, pose_id=1, motion_type=MotionType.LEAD_IN, scan_enabled=False
        ))
        poses.insert(0, replace(
            first, pose_id=0, motion_type=MotionType.APPROACH, scan_enabled=False,
            position=first.position + settings.approach_clearance_mm * first.surface_normal,
        ))
        poses.append(replace(
            last, pose_id=len(poses), motion_type=MotionType.LEAD_OUT,
            scan_enabled=False,
        ))
        poses.append(replace(
            last, pose_id=len(poses), motion_type=MotionType.RETRACT,
            scan_enabled=False,
            position=last.position + settings.approach_clearance_mm * last.surface_normal,
        ))
    return ViewSweep(sweep_id, f"isometric_{recovery_index + 1}", "isometric",
                     recovery_index, tuple(poses), VIEW_COLOURS["isometric"])


class ViewBasedSweepPlanner:
    def __init__(self, settings: ViewSweepSettings | None = None) -> None:
        self.settings = settings or ViewSweepSettings()
        self.settings.validate()

    def plan(self, model: CadModel,
             coverage_settings: CoverageSettings | None = None) -> ViewBasedMission:
        coverage_settings = coverage_settings or CoverageSettings(
            laser_width_mm=self.settings.laser_width_mm)
        exterior_face_ids = detect_exterior_face_ids(model)
        validate = lambda poses: validate_coverage(
            model, poses, coverage_settings, exterior_face_ids=exterior_face_ids
        )
        sweeps = _principal_sweeps(model, self.settings)
        principal_poses = tuple(p for sweep in sweeps for p in sweep.poses)
        principal = validate(principal_poses)
        current = principal
        if self.settings.isometric_recovery and np.any(
            current.eligible_triangle_mask
            & (current.triangle_status != TriangleStatus.VALID_COVERED)
        ):
            for recovery_index in range(self.settings.maximum_isometric_sweeps):
                candidate = _isometric_candidate(model, self.settings, len(sweeps), recovery_index)
                candidate_poses = tuple(p for sweep in (*sweeps, candidate) for p in sweep.poses)
                updated = validate(candidate_poses)
                old_percent = current.report["total_exterior_surface"].coverage_percent
                new_percent = updated.report["total_exterior_surface"].coverage_percent
                if new_percent - old_percent < self.settings.minimum_recovery_gain_percent:
                    break
                sweeps.append(candidate)
                current = updated
        contributions: dict[str, float] = {}
        usable_fractions: dict[str, float] = {}
        previous_mask = np.zeros(len(model.validation_mesh.mesh.faces), dtype=bool)
        areas = np.asarray(model.validation_mesh.mesh.area_faces)
        accumulated: list[Pose] = []
        for sweep in sweeps:
            accumulated.extend(sweep.poses)
            result = validate(accumulated)
            valid = result.triangle_status == TriangleStatus.VALID_COVERED
            contributions[sweep.view_type] = contributions.get(sweep.view_type, 0.0) + float(
                areas[valid & ~previous_mask].sum())
            independent = validate(sweep.poses)
            usable_fractions[sweep.sweep_name] = (
                independent.valid_hit_count / independent.ray_count
                if independent.ray_count else 0.0
            )
            previous_mask |= valid
        return ViewBasedMission(
            tuple(sweeps), principal, current, contributions, usable_fractions
        )


def generate_preview_sweeps(
    model: CadModel,
    settings: ViewSweepSettings,
    *,
    cancellation_check=None,
) -> tuple[ViewSweep, ...]:
    """Generate unchanged sweep mathematics without optical simulation."""

    settings.validate()
    cancellation_check = cancellation_check or (lambda: None)
    cancellation_check()
    sweeps = _principal_sweeps(model, settings, cancellation_check)
    cancellation_check()
    if settings.isometric_recovery and settings.maximum_isometric_sweeps > 0:
        # Preview generation is deliberately optical-free. It displays the
        # configured upper recovery sweep; Run coverage decides its usefulness.
        sweeps.append(_isometric_candidate(model, settings, len(sweeps), 0))
    cancellation_check()
    return tuple(sweeps)
