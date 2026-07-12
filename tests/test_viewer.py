from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from app.config import PipelineConfig
from planners.base import MotionType, Pose
from viewer import select_viewer_geometry


def _pose(index: int, scan: bool) -> Pose:
    return Pose(
        face_id=0,
        segment_id=0,
        pose_id=index,
        motion_type=MotionType.SCAN if scan else MotionType.APPROACH,
        scan_enabled=scan,
        surface_point=np.zeros(3),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        position=np.array([0.0, 0.0, 310.0]),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
    )


def test_viewer_defaults_are_uncluttered() -> None:
    config = PipelineConfig(step_path=Path("model.step"))
    assert config.connector_stride == 10
    assert config.frame_stride == 20
    assert config.show_connectors is False
    assert config.show_face_labels is False


def test_viewer_strides_only_scan_poses() -> None:
    poses = tuple(
        _pose(index, scan=index % 3 != 0) for index in range(31)
    )
    result = SimpleNamespace(poses=poses)
    config = replace(
        PipelineConfig(step_path=Path("model.step")),
        connector_stride=4,
        frame_stride=5,
        laser_footprint_stride=6,
    )
    geometry = select_viewer_geometry(result, config)
    scan_indices = tuple(i for i, pose in enumerate(poses) if pose.scan_enabled)
    assert geometry.connector_pose_indices == scan_indices[::4]
    assert geometry.frame_pose_indices == scan_indices[::5]
    assert geometry.laser_footprint_pose_indices == scan_indices[::6]
    assert all(poses[i].scan_enabled for i in geometry.connector_pose_indices)

