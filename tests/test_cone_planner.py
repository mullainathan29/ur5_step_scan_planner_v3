import math
import tempfile
from pathlib import Path

import cadquery as cq
import numpy as np

from cad_model import load_step_model
from face_analysis import SurfaceType
from planners.base import ConeStrategy, PlannerSettings
from planners.cone import plan_conical_face


def _load_cone(shape):
    temporary = tempfile.TemporaryDirectory()
    path = Path(temporary.name) / "cone.step"
    cq.exporters.export(shape, str(path), exportType="STEP")
    model = load_step_model(path)
    face_id = next(r.face_id for r in model.faces if r.geometry_type is SurfaceType.CONE)
    return temporary, model, model.topods_faces[face_id], model.faces[face_id]


def _assert_plan(model, plan, settings):
    assert plan.path_segments
    for motion in plan.motion_segments:
        quaternions = np.asarray([p.quaternion_xyzw for p in motion.scan_poses])
        if len(quaternions) > 1:
            assert np.all(np.sum(quaternions[:-1] * quaternions[1:], axis=1) > 0)
        for first, second in zip(motion.scan_poses, motion.scan_poses[1:]):
            assert np.linalg.norm(second.surface_point - first.surface_point) <= 1.5 * settings.travel_spacing_mm
        for pose in motion.scan_poses:
            assert np.isclose(np.linalg.norm(pose.position - pose.surface_point), 310.0)


def test_full_external_cone_and_stand_off() -> None:
    temporary, model, face, record = _load_cone(cq.Solid.makeCone(6, 0, 12))
    settings = PlannerSettings(lane_spacing_mm=4, travel_spacing_mm=2)
    plan = plan_conical_face(face, record, settings)
    assert np.isclose(plan.geometry.semi_angle_rad, math.atan2(6, 12))
    _assert_plan(model, plan, settings)
    temporary.cleanup()


def test_truncated_and_half_cones() -> None:
    for shape, expected_span in (
        (cq.Solid.makeCone(6, 3, 12), 2 * math.pi),
        (cq.Solid.makeCone(6, 3, 12, angleDegrees=180), math.pi),
    ):
        temporary, model, face, record = _load_cone(shape)
        settings = PlannerSettings(lane_spacing_mm=4, travel_spacing_mm=2)
        plan = plan_conical_face(face, record, settings)
        assert np.isclose(
            plan.geometry.angular_bounds[1] - plan.geometry.angular_bounds[0],
            expected_span,
        )
        _assert_plan(model, plan, settings)
        temporary.cleanup()


def test_internal_tapered_hole_has_reversed_orientation() -> None:
    block = cq.Workplane("XY").box(20, 20, 12).val()
    cutter = cq.Solid.makeCone(4, 2, 14, pnt=cq.Vector(0, 0, -7))
    temporary, model, face, record = _load_cone(block.cut(cutter))
    assert record.reversed_orientation
    plan = plan_conical_face(face, record, PlannerSettings(lane_spacing_mm=3, travel_spacing_mm=2))
    assert plan.path_segments
    temporary.cleanup()


def test_cone_circumferential_and_helical_strategies() -> None:
    temporary, model, face, record = _load_cone(cq.Solid.makeCone(6, 3, 12, angleDegrees=180))
    settings = PlannerSettings(lane_spacing_mm=4, travel_spacing_mm=2)
    circumferential = plan_conical_face(
        face, record, settings, strategy=ConeStrategy.CIRCUMFERENTIAL
    )
    helical = plan_conical_face(
        face, record, settings, strategy=ConeStrategy.HELICAL
    )
    assert circumferential.path_segments
    assert helical.path_segments
    _assert_plan(model, circumferential, settings)
    temporary.cleanup()
