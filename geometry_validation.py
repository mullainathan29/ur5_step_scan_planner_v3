"""Read-only geometry validation for generated scanner poses."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from planners.base import Pose


class ConnectorStatus(str, Enum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class PoseGeometryMetric:
    face_id: int
    segment_id: int
    pose_id: int
    stand_off_mm: float
    connector_status: ConnectorStatus
    surface_point: np.ndarray
    scanner_point: np.ndarray


@dataclass(frozen=True)
class SegmentGeometryStats:
    segment_id: int
    pose_count: int
    maximum_orientation_change_deg: float
    maximum_position_change_mm: float


@dataclass(frozen=True)
class GeometryValidationResult:
    pose_metrics: tuple[PoseGeometryMetric, ...]
    segment_statistics: tuple[SegmentGeometryStats, ...]
    minimum_stand_off_mm: float
    maximum_stand_off_mm: float
    mean_stand_off_mm: float
    stand_off_standard_deviation_mm: float
    maximum_orientation_change_deg: float
    maximum_position_change_mm: float


def connector_status(distance_mm: float) -> ConnectorStatus:
    if 300.0 <= distance_mm <= 320.0:
        return ConnectorStatus.GREEN
    if 280.0 <= distance_mm < 300.0 or 320.0 < distance_mm <= 340.0:
        return ConnectorStatus.YELLOW
    return ConnectorStatus.RED


def _orientation_change_deg(first: Pose, second: Pose) -> float:
    first_rotation = Rotation.from_quat(first.quaternion_xyzw)
    second_rotation = Rotation.from_quat(second.quaternion_xyzw)
    return float(
        np.degrees((first_rotation.inv() * second_rotation).magnitude())
    )


def validate_pose_geometry(poses: Iterable[Pose]) -> GeometryValidationResult:
    """Measure existing scan poses without changing their geometry."""

    scan_poses = [pose for pose in poses if pose.scan_enabled]
    metrics = tuple(
        PoseGeometryMetric(
            face_id=pose.face_id,
            segment_id=pose.segment_id,
            pose_id=pose.pose_id,
            stand_off_mm=float(
                np.linalg.norm(pose.position - pose.surface_point)
            ),
            connector_status=connector_status(
                float(np.linalg.norm(pose.position - pose.surface_point))
            ),
            surface_point=pose.surface_point.copy(),
            scanner_point=pose.position.copy(),
        )
        for pose in scan_poses
    )

    by_segment: dict[int, list[Pose]] = defaultdict(list)
    for pose in scan_poses:
        by_segment[pose.segment_id].append(pose)

    segment_statistics: list[SegmentGeometryStats] = []
    for segment_id, segment in sorted(by_segment.items()):
        orientation_changes = [
            _orientation_change_deg(first, second)
            for first, second in zip(segment, segment[1:])
        ]
        position_changes = [
            float(np.linalg.norm(second.position - first.position))
            for first, second in zip(segment, segment[1:])
        ]
        segment_statistics.append(
            SegmentGeometryStats(
                segment_id=segment_id,
                pose_count=len(segment),
                maximum_orientation_change_deg=max(
                    orientation_changes, default=0.0
                ),
                maximum_position_change_mm=max(position_changes, default=0.0),
            )
        )

    distances = np.asarray(
        [metric.stand_off_mm for metric in metrics], dtype=np.float64
    )
    return GeometryValidationResult(
        pose_metrics=metrics,
        segment_statistics=tuple(segment_statistics),
        minimum_stand_off_mm=float(distances.min()) if len(distances) else 0.0,
        maximum_stand_off_mm=float(distances.max()) if len(distances) else 0.0,
        mean_stand_off_mm=float(distances.mean()) if len(distances) else 0.0,
        stand_off_standard_deviation_mm=(
            float(distances.std()) if len(distances) else 0.0
        ),
        maximum_orientation_change_deg=max(
            (
                row.maximum_orientation_change_deg
                for row in segment_statistics
            ),
            default=0.0,
        ),
        maximum_position_change_mm=max(
            (row.maximum_position_change_mm for row in segment_statistics),
            default=0.0,
        ),
    )
