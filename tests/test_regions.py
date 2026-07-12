from __future__ import annotations

from dataclasses import replace

import cadquery as cq
import numpy as np

from face_analysis import extract_face_records
from planners.base import PlannerSettings
from planners.cylinder import plan_cylindrical_face
from planners.plane import plan_planar_face
from regions import build_cylindrical_regions, build_planar_regions


def _adjacent_records(faces):
    records = list(extract_face_records(tuple(face.wrapped for face in faces)))
    records[0] = replace(records[0], adjacent_face_ids=(1,))
    records[1] = replace(records[1], adjacent_face_ids=(0,))
    return tuple(records)


def test_adjacent_coplanar_rectangles_merge_and_plan_as_one_region() -> None:
    faces = (
        cq.Face.makePlane(10.0, 10.0, cq.Vector(0, 0, 0)),
        cq.Face.makePlane(10.0, 10.0, cq.Vector(10, 0, 0)),
    )
    records = _adjacent_records(faces)
    regions = build_planar_regions(
        tuple(face.wrapped for face in faces), records, {0, 1}
    )

    assert len(regions) == 1
    region = regions[0]
    assert region.member_face_ids == (0, 1)
    assert np.isclose(region.total_area_mm2, 200.0)
    assert len(region.planning_faces) == 1

    merged_record = replace(
        extract_face_records(region.planning_faces)[0], face_id=0
    )
    settings = PlannerSettings(
        lane_spacing_mm=2.0, travel_spacing_mm=2.0, stand_off_mm=310.0
    )
    plan = plan_planar_face(region.planning_faces[0], merged_record, settings)
    for segment in plan.raster_segments:
        for first, second in zip(segment.samples, segment.samples[1:]):
            assert (
                np.linalg.norm(second.point - first.point)
                <= settings.travel_spacing_mm + 1.0e-8
            )
    for motion in plan.motion_segments:
        for pose in motion.scan_poses:
            assert np.isclose(
                np.linalg.norm(pose.position - pose.surface_point), 310.0
            )


def test_coplanar_faces_separated_by_gap_do_not_merge() -> None:
    faces = (
        cq.Face.makePlane(10.0, 10.0, cq.Vector(0, 0, 0)),
        cq.Face.makePlane(10.0, 10.0, cq.Vector(20, 0, 0)),
    )
    records = extract_face_records(tuple(face.wrapped for face in faces))
    regions = build_planar_regions(
        tuple(face.wrapped for face in faces), records, {0, 1}
    )
    assert len(regions) == 2


def test_planar_region_preserves_hole() -> None:
    outer = cq.Wire.makePolygon(
        [
            cq.Vector(-10, -10, 0),
            cq.Vector(10, -10, 0),
            cq.Vector(10, 10, 0),
            cq.Vector(-10, 10, 0),
        ],
        close=True,
    )
    inner = cq.Wire.makeCircle(
        3.0, cq.Vector(0, 0, 0), cq.Vector(0, 0, 1)
    )
    face = cq.Face.makeFromWires(outer, [inner])
    records = extract_face_records((face.wrapped,))
    region = build_planar_regions((face.wrapped,), records, {0})[0]
    assert len(region.merged_inner_holes) == 1


def _cylinder_face(radius: float, height: float, z: float):
    solid = cq.Solid.makeCylinder(
        radius, height, pnt=cq.Vector(0, 0, z)
    )
    return next(face for face in solid.Faces() if face.geomType() == "CYLINDER")


def test_adjacent_same_axis_same_radius_cylinders_merge() -> None:
    faces = (_cylinder_face(5, 5, 0), _cylinder_face(5, 5, 5))
    records = _adjacent_records(faces)
    regions = build_cylindrical_regions(
        tuple(face.wrapped for face in faces), records, {0, 1}
    )
    assert len(regions) == 1
    assert regions[0].member_face_ids == (0, 1)
    assert np.isclose(regions[0].combined_axial_limits[1] - regions[0].combined_axial_limits[0], 10.0)


def test_cylinders_with_different_radii_do_not_merge() -> None:
    faces = (_cylinder_face(5, 5, 0), _cylinder_face(6, 5, 5))
    records = _adjacent_records(faces)
    regions = build_cylindrical_regions(
        tuple(face.wrapped for face in faces), records, {0, 1}
    )
    assert len(regions) == 2


def test_cylinders_with_angular_trim_gap_remain_separate() -> None:
    first_solid = cq.Solid.makeCylinder(5, 5, angleDegrees=60)
    second_solid = cq.Solid.makeCylinder(5, 5, angleDegrees=60).rotate(
        cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 180
    )
    faces = (
        next(f for f in first_solid.Faces() if f.geomType() == "CYLINDER"),
        next(f for f in second_solid.Faces() if f.geomType() == "CYLINDER"),
    )
    records = extract_face_records(tuple(face.wrapped for face in faces))
    regions = build_cylindrical_regions(
        tuple(face.wrapped for face in faces), records, {0, 1}
    )
    assert len(regions) == 2
