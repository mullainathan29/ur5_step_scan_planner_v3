from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import trimesh
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from app.result import PipelineResult, PipelineSummary
from exporter import _pose_row
from planners.base import MotionType, Pose
from planners.mission_builder import build_continuous_mission
from planners.safe_transition_planner import (
    adaptive_resample,
    compute_curvatures,
    smooth_transition_path,
    slerp_trajectory_orientations,
)


class DummyValidationMesh:
    def __init__(self) -> None:
        self.mesh = trimesh.creation.box(extents=(20.0, 20.0, 20.0))


class DummyModel:
    def __init__(self) -> None:
        self.validation_mesh = DummyValidationMesh()


class DummySweep:
    def __init__(self, sweep_id: int, poses: tuple[Pose, ...]) -> None:
        self.sweep_id = sweep_id
        self.sweep_name = f"sweep_{sweep_id}"
        self.view_type = "test"
        self.poses = poses

    @property
    def scan_poses(self) -> tuple[Pose, ...]:
        return tuple(pose for pose in self.poses if pose.scan_enabled)


def _config(**overrides) -> PipelineConfig:
    values = {
        "step_path": "dummy.step",
        "stand_off_mm": 40.0,
        "bottom_clearance_mm": 10.0,
        "safety_clearance_mm": 1.0,
        "tool_envelope_radius_mm": 1.0,
        "sample_spacing_mm": 10.0,
        "max_position_step_mm": 20.0,
        "orientation_smoothing_maximum_step_deg": 5.0,
        "maximum_route_deviation_mm": 200.0,
    }
    values.update(overrides)
    return PipelineConfig(**values)


def _pose(x: float, y: float, pose_id: int, sweep_id: int = 1) -> Pose:
    return Pose(
        face_id=1,
        segment_id=sweep_id,
        pose_id=pose_id,
        motion_type=MotionType.SCAN,
        scan_enabled=True,
        surface_point=np.array([x, y, 0.0]),
        surface_normal=np.array([0.0, 0.0, 1.0]),
        position=np.array([x, y, 40.0]),
        quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        sweep_id=sweep_id,
        sweep_name=f"sweep_{sweep_id}",
    )


def _result(*sweeps: DummySweep) -> PipelineResult:
    poses = tuple(pose for sweep in sweeps for pose in sweep.poses)
    return PipelineResult(
        model=DummyModel(),
        planar_plans=(),
        cylindrical_plans=(),
        conical_plans=(),
        spherical_plans=(),
        poses=poses,
        coverage=None,
        baseline_coverage=None,
        geometry_validation=None,
        summary=PipelineSummary(
            total_step_faces=0,
            selected_planar_faces=0,
            selected_cylindrical_faces=0,
            selected_conical_faces=0,
            selected_spherical_faces=0,
            skipped_faces_by_reason={},
            generated_scan_regions=0,
            generated_scan_segments=0,
            scan_pose_count=0,
            total_scan_path_length_mm=0.0,
            coverage_percent=0.0,
            invalid_segment_count=0,
            output_files={},
        ),
        sweeps=sweeps,
    )


def _patch_clearance(monkeypatch: pytest.MonkeyPatch, distance: float = 1000.0) -> None:
    def closest_point(_mesh, points):
        points = np.asarray(points, dtype=float)
        return points, np.full(len(points), distance, dtype=float), None

    monkeypatch.setattr("trimesh.proximity.closest_point", closest_point)
    monkeypatch.setattr("planners.safe_transition_planner.line_collision_with_mesh", lambda *args: False)


def _far_box() -> trimesh.primitives.Box:
    transform = np.eye(4)
    transform[:3, 3] = [500.0, 500.0, 500.0]
    return trimesh.primitives.Box(extents=(10.0, 10.0, 10.0), transform=transform)


def test_scan_paths_unchanged_and_smoothing_only_free_travel(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    config = _config()
    scan_a = (_pose(0.0, 0.0, 0, 1), _pose(10.0, 0.0, 1, 1))
    scan_b = (_pose(40.0, 0.0, 0, 2), _pose(50.0, 0.0, 1, 2))
    result = _result(DummySweep(1, scan_a), DummySweep(2, scan_b))
    original_scan_positions = [tuple(p.position) for p in (*scan_a, *scan_b)]

    mission = build_continuous_mission(result, config)
    mission_scan_positions = [tuple(p.position) for p in mission if p.scan_status == -1]

    assert mission_scan_positions == original_scan_positions
    assert all(p.scan_status == 0 and not p.scan_enabled for p in mission if p.smoothing_status)
    assert all(not p.smoothing_status for p in mission if p.scan_status == -1)


def test_tangent_continuity_for_catmull_rom(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    path = [
        (np.array([0.0, 0.0, 50.0]), q),
        (np.array([20.0, 0.0, 50.0]), q),
        (np.array([40.0, 20.0, 50.0]), q),
        (np.array([40.0, 40.0, 50.0]), q),
    ]

    positions, *_ = smooth_transition_path(DummyModel(), path, _far_box(), None, _config())

    start_tangent = positions[1] - positions[0]
    end_tangent = positions[-1] - positions[-2]
    assert start_tangent[0] > 0.0
    assert end_tangent[1] > 0.0


def test_cad_collision_rejects_smooth_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    original_segments = {
        ((0.0, 0.0, 50.0), (20.0, 0.0, 50.0)),
        ((20.0, 0.0, 50.0), (20.0, 20.0, 50.0)),
    }

    def collision(_mesh, p0, p1):
        key = (tuple(np.round(p0, 6)), tuple(np.round(p1, 6)))
        return key not in original_segments

    monkeypatch.setattr("planners.safe_transition_planner.line_collision_with_mesh", collision)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    path = [
        (np.array([0.0, 0.0, 50.0]), q),
        (np.array([20.0, 0.0, 50.0]), q),
        (np.array([20.0, 20.0, 50.0]), q),
    ]

    _, _, status, method, *_ = smooth_transition_path(DummyModel(), path, _far_box(), None, _config())

    assert status == "original_safe_polyline"
    assert method == "none"


def test_safety_envelope_rejects_shortcut(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    path = [
        (np.array([-20.0, -20.0, 0.0]), q),
        (np.array([-20.0, 20.0, 0.0]), q),
        (np.array([20.0, 20.0, 0.0]), q),
    ]
    box = trimesh.primitives.Box(extents=(20.0, 20.0, 20.0))

    monkeypatch.setattr(
        "planners.safe_transition_planner.cubic_bezier_curve",
        lambda *args: np.array([[-20.0, -20.0, 0.0], [0.0, 0.0, 0.0], [20.0, 20.0, 0.0]]),
    )
    monkeypatch.setattr("planners.safe_transition_planner.circular_fillet_blend", lambda *args: None)

    _, _, status, method, *_ = smooth_transition_path(DummyModel(), path, box, None, _config())

    assert status == "original_safe_polyline"
    assert method == "none"


def test_original_safe_polyline_fallback_is_resampled(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    path = [(np.array([0.0, 0.0, 50.0]), q), (np.array([45.0, 0.0, 50.0]), q)]

    positions, _, status, *_ = smooth_transition_path(DummyModel(), path, _far_box(), None, _config(free_travel_smoothing_enabled=False))

    assert status == "original_safe_polyline"
    assert max(np.linalg.norm(positions[i + 1] - positions[i]) for i in range(len(positions) - 1)) <= 10.0 + 1e-6


def test_slerp_continuity_and_max_orientation_step_enforcement() -> None:
    config = _config(orientation_smoothing_maximum_step_deg=5.0)
    q0 = np.array([0.0, 0.0, 0.0, 1.0])
    q1 = Rotation.from_euler("z", 45.0, degrees=True).as_quat()
    arc = np.linspace(0.0, 100.0, 101)

    orientations = slerp_trajectory_orientations(q0, q1, arc, config)
    indices = adaptive_resample(
        np.column_stack((arc, np.zeros_like(arc), np.zeros_like(arc))),
        orientations,
        np.full_like(arc, 1000.0),
        config,
    )

    selected = orientations[indices]
    steps = [
        np.degrees((Rotation.from_quat(selected[i]).inv() * Rotation.from_quat(selected[i + 1])).magnitude())
        for i in range(len(selected) - 1)
    ]
    assert all(np.dot(orientations[i], orientations[i + 1]) >= 0.0 for i in range(len(orientations) - 1))
    assert max(steps) <= 5.0 + 1e-6


def test_adaptive_sampling_keeps_more_points_near_obstacles() -> None:
    config = _config(sample_spacing_mm=20.0, max_position_step_mm=20.0)
    positions = np.column_stack((np.linspace(0.0, 100.0, 101), np.zeros(101), np.zeros(101)))
    orientations = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (101, 1))
    high_clearance = np.full(101, 1000.0)
    low_clearance = np.full(101, config.safety_clearance_mm + config.tool_envelope_radius_mm + 5.0)

    sparse = adaptive_resample(positions, orientations, high_clearance, config)
    dense = adaptive_resample(positions, orientations, low_clearance, config)

    assert len(dense) > len(sparse)


def test_export_metadata_fields_present() -> None:
    pose = replace(
        _pose(0.0, 0.0, 0),
        scan_enabled=False,
        scan_status=0,
        transition_id=7,
        transition_strategy="top_clearance",
        smoothing_method="catmull_rom",
        smoothing_status="smoothed",
        local_curvature=0.01,
        collision_checked=True,
    )

    row = _pose_row(pose)

    assert row["transition_id"] == 7
    assert row["smoothing_method"] == "catmull_rom"
    assert row["smoothing_status"] == "smoothed"
    assert row["local_curvature"] == pytest.approx(0.01)
    assert row["collision_checked"] is True


def test_viewer_export_path_equivalence_and_one_flat_mission(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    result = _result(
        DummySweep(1, (_pose(0.0, 0.0, 0, 1), _pose(10.0, 0.0, 1, 1))),
        DummySweep(2, (_pose(40.0, 0.0, 0, 2), _pose(50.0, 0.0, 1, 2))),
    )

    mission = build_continuous_mission(result, _config())
    exported_points = [tuple(_pose_row(pose)[axis] for axis in ("x", "y", "z")) for pose in mission]
    viewer_points = [tuple(np.asarray(pose.position, dtype=float)) for pose in mission]

    assert exported_points == viewer_points
    assert [pose.global_pose_id for pose in mission] == list(range(len(mission)))
    assert result.summary.output_files["mission_timings"].keys() >= {
        "safe_polyline_generation",
        "curve_fitting",
        "curve_collision_validation",
        "orientation_interpolation",
        "adaptive_resampling",
        "final_mission_assembly",
    }


def test_local_curvature_metadata_on_smoothed_free_travel(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_clearance(monkeypatch)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    path = [
        (np.array([0.0, 0.0, 50.0]), q),
        (np.array([20.0, 0.0, 50.0]), q),
        (np.array([40.0, 20.0, 50.0]), q),
        (np.array([40.0, 40.0, 50.0]), q),
    ]

    positions, *_ = smooth_transition_path(DummyModel(), path, _far_box(), None, _config())
    curvatures = compute_curvatures(positions)

    assert len(curvatures) == len(positions)
    assert np.max(curvatures) > 0.0
