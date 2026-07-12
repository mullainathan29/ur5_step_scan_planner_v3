from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from geometry_validation import (
    ConnectorStatus,
    connector_status,
    validate_pose_geometry,
)
from planners.base import MotionType, Pose


def _pose(
    pose_id: int,
    surface: tuple[float, float, float],
    scanner: tuple[float, float, float],
    angle_deg: float,
) -> Pose:
    quaternion = Rotation.from_euler("z", angle_deg, degrees=True).as_quat()
    return Pose(
        face_id=4,
        segment_id=2,
        pose_id=pose_id,
        motion_type=MotionType.SCAN,
        scan_enabled=True,
        surface_point=np.asarray(surface, dtype=float),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        position=np.asarray(scanner, dtype=float),
        quaternion_xyzw=tuple(float(value) for value in quaternion),
    )


def test_connector_colour_limits() -> None:
    assert connector_status(310.0) is ConnectorStatus.GREEN
    assert connector_status(300.0) is ConnectorStatus.GREEN
    assert connector_status(320.0) is ConnectorStatus.GREEN
    assert connector_status(299.0) is ConnectorStatus.YELLOW
    assert connector_status(321.0) is ConnectorStatus.YELLOW
    assert connector_status(280.0) is ConnectorStatus.YELLOW
    assert connector_status(340.0) is ConnectorStatus.YELLOW
    assert connector_status(279.9) is ConnectorStatus.RED
    assert connector_status(340.1) is ConnectorStatus.RED


def test_geometry_statistics_and_consecutive_changes() -> None:
    poses = (
        _pose(0, (0, 0, 0), (0, 0, 300), 0.0),
        _pose(1, (0, 5, 0), (0, 5, 310), 10.0),
        _pose(2, (0, 11, 0), (0, 11, 320), 25.0),
    )

    result = validate_pose_geometry(poses)

    assert result.minimum_stand_off_mm == 300.0
    assert result.maximum_stand_off_mm == 320.0
    assert result.mean_stand_off_mm == 310.0
    assert np.isclose(
        result.stand_off_standard_deviation_mm,
        np.std([300.0, 310.0, 320.0]),
    )
    assert np.isclose(result.maximum_orientation_change_deg, 15.0)
    assert np.isclose(result.maximum_position_change_mm, np.sqrt(136.0))
    assert len(result.segment_statistics) == 1
    assert all(
        metric.connector_status is ConnectorStatus.GREEN
        for metric in result.pose_metrics
    )


def test_non_scan_poses_do_not_generate_connector_metrics() -> None:
    scan = _pose(0, (0, 0, 0), (0, 0, 310), 0.0)
    transition = Pose(
        **{
            **scan.__dict__,
            "pose_id": 1,
            "motion_type": MotionType.APPROACH,
            "scan_enabled": False,
            "position": np.array([0.0, 0.0, 430.0]),
        }
    )
    result = validate_pose_geometry((transition, scan))
    assert len(result.pose_metrics) == 1
    assert result.pose_metrics[0].pose_id == scan.pose_id
