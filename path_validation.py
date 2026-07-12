"""Post-planning continuity validation and rejection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from planners.base import PlannerSettings, ScanMotionSegment


@dataclass(frozen=True)
class SegmentValidation:
    segment_id: int
    valid: bool
    reasons: tuple[str, ...]


def validate_scan_segment(
    motion: ScanMotionSegment,
    settings: PlannerSettings,
    *,
    expected_region_id: int,
) -> SegmentValidation:
    poses = motion.scan_poses
    reasons: list[str] = []
    if len(poses) < 2:
        reasons.append("zero-length segment")
    if any(pose.region_id != expected_region_id for pose in poses):
        reasons.append("crosses unrelated regions")
    scan_runs: list[list] = []
    current_run: list = []
    for pose in motion.poses:
        if pose.scan_enabled:
            current_run.append(pose)
        elif current_run:
            scan_runs.append(current_run)
            current_run = []
    if current_run:
        scan_runs.append(current_run)
    distances = [
        float(np.linalg.norm(second.position - first.position))
        for run in scan_runs
        for first, second in zip(run, run[1:])
    ]
    if any(distance > 1.5 * settings.travel_spacing_mm + 1.0e-7 for distance in distances):
        reasons.append("trim gap or excessive scan-point spacing")
    if poses and not all(
        300.0 <= np.linalg.norm(pose.position - pose.surface_point) <= 320.0
        for pose in poses
    ):
        reasons.append("stand-off outside 300-320 mm")
    orientation_changes = [
        float(
            np.degrees(
                (
                    Rotation.from_quat(first.quaternion_xyzw).inv()
                    * Rotation.from_quat(second.quaternion_xyzw)
                ).magnitude()
            )
        )
        for run in scan_runs
        for first, second in zip(run, run[1:])
    ]
    if any(change > 5.0 + 1.0e-7 for change in orientation_changes):
        reasons.append("orientation change exceeds 5 degrees")
    points = [tuple(np.round(pose.position, decimals=9)) for pose in poses]
    if len(points) != len(set(points)):
        reasons.append("duplicate scan poses")
    if distances and sum(distances) <= 1.0e-9:
        reasons.append("zero-length segment")
    return SegmentValidation(
        segment_id=motion.segment_id,
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
    )
