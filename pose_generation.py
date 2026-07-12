"""Scanner pose construction from exact B-Rep surface samples."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from planners.base import (
    MotionType,
    PlaneFrame,
    PlannerSettings,
    Pose,
    RasterSegment,
    ScanMotionSegment,
)


def _quaternion_for_axes(
    laser_axis: np.ndarray,
    travel_axis: np.ndarray,
    outward_normal: np.ndarray,
) -> tuple[float, float, float, float]:
    """Return scanner orientation with +X laser, +Y travel, +Z surfaceward."""

    scanner_z = -outward_normal
    matrix = np.column_stack((laser_axis, travel_axis, scanner_z))
    quaternion = Rotation.from_matrix(matrix).as_quat()
    return tuple(float(value) for value in quaternion)


def create_planar_motion_segment(
    raster_segment: RasterSegment,
    frame: PlaneFrame,
    settings: PlannerSettings,
    segment_id: int,
) -> ScanMotionSegment:
    """Offset one connected raster section and add transition poses."""

    samples = raster_segment.samples
    if len(samples) < 2:
        raise ValueError("A raster segment requires at least two surface samples.")

    outward = frame.outward_normal
    first = samples[0]
    last = samples[-1]
    travel = last.point - first.point
    travel /= np.linalg.norm(travel)
    scanner_z = -outward
    laser = np.cross(travel, scanner_z)
    laser /= np.linalg.norm(laser)
    quaternion = _quaternion_for_axes(laser, travel, outward)

    scan_positions = [
        sample.point + settings.stand_off_mm * outward for sample in samples
    ]
    lead_in = scan_positions[0] - settings.lead_distance_mm * travel
    lead_out = scan_positions[-1] + settings.lead_distance_mm * travel
    approach = lead_in + settings.approach_clearance_mm * outward
    retract = lead_out + settings.approach_clearance_mm * outward

    poses: list[Pose] = []

    def add_pose(
        motion_type: MotionType,
        scan_enabled: bool,
        surface_point: np.ndarray,
        normal: np.ndarray,
        position: np.ndarray,
    ) -> None:
        poses.append(
            Pose(
                face_id=raster_segment.face_id,
                segment_id=segment_id,
                pose_id=len(poses),
                motion_type=motion_type,
                scan_enabled=scan_enabled,
                surface_point=surface_point.copy(),
                surface_normal=normal.copy(),
                position=position.copy(),
                quaternion_xyzw=quaternion,
            )
        )

    add_pose(
        MotionType.APPROACH,
        False,
        first.point,
        first.normal,
        approach,
    )
    add_pose(
        MotionType.LEAD_IN,
        False,
        first.point,
        first.normal,
        lead_in,
    )
    for sample, position in zip(samples, scan_positions):
        add_pose(
            MotionType.SCAN,
            True,
            sample.point,
            sample.normal,
            position,
        )
    add_pose(
        MotionType.LEAD_OUT,
        False,
        last.point,
        last.normal,
        lead_out,
    )
    add_pose(
        MotionType.RETRACT,
        False,
        last.point,
        last.normal,
        retract,
    )
    return ScanMotionSegment(
        segment_id=segment_id,
        raster_segment=raster_segment,
        poses=tuple(poses),
    )


def _continuous_quaternions(
    samples: tuple,
) -> tuple[tuple[float, float, float, float], ...]:
    """Build curved scanner frames and enforce quaternion hemisphere continuity."""

    quaternions: list[np.ndarray] = []
    for index, sample in enumerate(samples):
        if index == 0:
            tangent = samples[1].point - sample.point
        elif index == len(samples) - 1:
            tangent = sample.point - samples[index - 1].point
        else:
            tangent = samples[index + 1].point - samples[index - 1].point
        normal = sample.normal / np.linalg.norm(sample.normal)
        tangent = tangent - np.dot(tangent, normal) * normal
        tangent /= np.linalg.norm(tangent)
        scanner_z = -normal
        laser = np.cross(tangent, scanner_z)
        laser /= np.linalg.norm(laser)
        quaternion = np.asarray(
            _quaternion_for_axes(laser, tangent, normal), dtype=np.float64
        )
        if quaternions and np.dot(quaternions[-1], quaternion) < 0.0:
            quaternion = -quaternion
        quaternions.append(quaternion)
    return tuple(tuple(float(value) for value in q) for q in quaternions)


def create_curved_motion_segment(
    path_segment: RasterSegment,
    settings: PlannerSettings,
    segment_id: int,
) -> ScanMotionSegment:
    """Create continuously oriented poses for one curved path section."""

    samples = path_segment.samples
    if len(samples) < 2:
        raise ValueError("A curved path segment requires at least two samples.")

    quaternions = _continuous_quaternions(samples)
    first_travel = samples[1].point - samples[0].point
    first_travel /= np.linalg.norm(first_travel)
    last_travel = samples[-1].point - samples[-2].point
    last_travel /= np.linalg.norm(last_travel)
    scan_positions = [
        sample.point + settings.stand_off_mm * sample.normal for sample in samples
    ]
    lead_in = scan_positions[0] - settings.lead_distance_mm * first_travel
    lead_out = scan_positions[-1] + settings.lead_distance_mm * last_travel
    approach = lead_in + settings.approach_clearance_mm * samples[0].normal
    retract = lead_out + settings.approach_clearance_mm * samples[-1].normal

    poses: list[Pose] = []

    def add(
        motion_type: MotionType,
        scan_enabled: bool,
        sample_index: int,
        position: np.ndarray,
        quaternion: tuple[float, float, float, float],
    ) -> None:
        sample = samples[sample_index]
        poses.append(
            Pose(
                face_id=path_segment.face_id,
                segment_id=segment_id,
                pose_id=len(poses),
                motion_type=motion_type,
                scan_enabled=scan_enabled,
                surface_point=sample.point.copy(),
                surface_normal=sample.normal.copy(),
                position=position.copy(),
                quaternion_xyzw=quaternion,
            )
        )

    add(MotionType.APPROACH, False, 0, approach, quaternions[0])
    add(MotionType.LEAD_IN, False, 0, lead_in, quaternions[0])
    for index, (position, quaternion) in enumerate(
        zip(scan_positions, quaternions)
    ):
        add(MotionType.SCAN, True, index, position, quaternion)
    add(MotionType.LEAD_OUT, False, -1, lead_out, quaternions[-1])
    add(MotionType.RETRACT, False, -1, retract, quaternions[-1])
    return ScanMotionSegment(
        segment_id=segment_id,
        raster_segment=path_segment,
        poses=tuple(poses),
    )
