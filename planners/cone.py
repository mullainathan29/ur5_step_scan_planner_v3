"""Exact UV planning for trimmed conical STEP faces."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.TopAbs import TopAbs_State
from OCP.TopoDS import TopoDS_Face
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec

from face_analysis import FaceRecord, SurfaceType
from planners.base import (
    ConeGeometry,
    ConeStrategy,
    ConicalPlan,
    PlannerSettings,
    PlanningError,
    RasterSegment,
    SurfaceSample,
)
from pose_generation import create_curved_motion_segment


def _array(value) -> np.ndarray:
    return np.array([value.X(), value.Y(), value.Z()], dtype=np.float64)


def exact_cone_geometry(face: TopoDS_Face) -> ConeGeometry:
    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_SurfaceType.GeomAbs_Cone:
        raise PlanningError("The supplied face is not an exact cone.")
    cone = adaptor.Cone()
    axes = cone.Position()
    return ConeGeometry(
        apex=_array(cone.Apex()),
        axis_direction=_array(axes.Direction()),
        x_direction=_array(axes.XDirection()),
        y_direction=_array(axes.YDirection()),
        reference_radius_mm=float(cone.RefRadius()),
        semi_angle_rad=float(cone.SemiAngle()),
        angular_bounds=(float(adaptor.FirstUParameter()), float(adaptor.LastUParameter())),
        axial_bounds=(float(adaptor.FirstVParameter()), float(adaptor.LastVParameter())),
    )


def _inside(face: TopoDS_Face, u: float, v: float, tol: float) -> bool:
    state = BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), tol).State()
    return state in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON)


def _point_normal(adaptor, record, u, v):
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    adaptor.D1(float(u), float(v), point, du, dv)
    normal = np.cross(_array(du), _array(dv))
    magnitude = np.linalg.norm(normal)
    if magnitude <= 1.0e-10:
        raise PlanningError("Cone derivative is singular at the apex.")
    normal /= magnitude
    if record.reversed_orientation:
        normal = -normal
    return _array(point), normal


def _split_valid(face, values, tol):
    runs, current = [], []
    for uv in values:
        if _inside(face, *uv, tol):
            current.append(uv)
        elif current:
            if len(current) >= 2:
                runs.append(current)
            current = []
    if len(current) >= 2:
        runs.append(current)
    return runs


def _segment(adaptor, record, lane, section, values):
    samples = []
    for index, (u, v) in enumerate(values):
        point, normal = _point_normal(adaptor, record, u, v)
        samples.append(SurfaceSample(
            face_id=record.face_id, lane_index=lane, section_index=section,
            sample_index=index, uv=(float(u), float(v)),
            point=point, normal=normal,
        ))
    return RasterSegment(
        face_id=record.face_id, lane_index=lane, section_index=section,
        laser_coordinate=float(values[0][0]), samples=tuple(samples),
    )


def _longitudinal(face, record, geometry, settings):
    adaptor = BRepAdaptor_Surface(face, True)
    u0, u1 = geometry.angular_bounds
    v0, v1 = geometry.axial_bounds
    max_radius = max(
        abs(geometry.reference_radius_mm + v * math.sin(geometry.semi_angle_rad))
        for v in (v0, v1)
    )
    lane_count = max(1, math.ceil(max_radius * (u1 - u0) / settings.lane_spacing_mm))
    angles = u0 + (np.arange(lane_count) + 0.5) * (u1 - u0) / lane_count
    intervals = max(1, math.ceil(abs(v1 - v0) / settings.travel_spacing_mm))
    output = []
    for lane, u in enumerate(angles):
        v_values = np.linspace(v0, v1, intervals + 1)
        if lane % 2:
            v_values = v_values[::-1]
        values = [
            (float(u), float(v)) for v in v_values
            if abs(geometry.reference_radius_mm + v * math.sin(geometry.semi_angle_rad)) > 1.0e-6
        ]
        for section, run in enumerate(_split_valid(face, values, settings.geometric_tolerance_mm)):
            output.append(_segment(adaptor, record, lane, section, run))
    return output


def _circumferential(face, record, geometry, settings):
    adaptor = BRepAdaptor_Surface(face, True)
    u0, u1 = geometry.angular_bounds
    v0, v1 = geometry.axial_bounds
    lane_count = max(1, math.ceil(abs(v1 - v0) / settings.lane_spacing_mm))
    axial = np.linspace(v0, v1, lane_count + 2)[1:-1]
    output = []
    for lane, v in enumerate(axial):
        radius = abs(geometry.reference_radius_mm + v * math.sin(geometry.semi_angle_rad))
        if radius <= 1.0e-6:
            continue
        intervals = max(1, math.ceil(radius * (u1 - u0) / settings.travel_spacing_mm))
        angles = np.linspace(u0, u1, intervals + 1)
        if lane % 2:
            angles = angles[::-1]
        values = [(float(u), float(v)) for u in angles]
        for section, run in enumerate(_split_valid(face, values, settings.geometric_tolerance_mm)):
            output.append(_segment(adaptor, record, lane, section, run))
    return output


def _helical(face, record, geometry, settings):
    adaptor = BRepAdaptor_Surface(face, True)
    u0, u1 = geometry.angular_bounds
    v0, v1 = geometry.axial_bounds
    turns = max(1, math.ceil(abs(v1 - v0) / settings.lane_spacing_mm))
    count = max(2, math.ceil(
        (abs(v1 - v0) + turns * abs(u1 - u0) * geometry.reference_radius_mm)
        / settings.travel_spacing_mm
    ))
    theta = np.linspace(0.0, turns * (u1 - u0), count + 1)
    values = [
        (float(u0 + (t % (u1 - u0))), float(v0 + (v1 - v0) * i / count))
        for i, t in enumerate(theta)
    ]
    return [
        _segment(adaptor, record, 0, section, run)
        for section, run in enumerate(_split_valid(face, values, settings.geometric_tolerance_mm))
    ]


def plan_conical_face(
    face: TopoDS_Face,
    record: FaceRecord,
    settings: PlannerSettings | None = None,
    *,
    strategy: ConeStrategy = ConeStrategy.LONGITUDINAL,
    segment_id_start: int = 0,
) -> ConicalPlan:
    settings = settings or PlannerSettings()
    settings.validate()
    strategy = ConeStrategy(strategy)
    if record.geometry_type is not SurfaceType.CONE:
        raise PlanningError(f"Face {record.face_id} is not classified as CONE.")
    geometry = exact_cone_geometry(face)
    builder = {
        ConeStrategy.LONGITUDINAL: _longitudinal,
        ConeStrategy.CIRCUMFERENTIAL: _circumferential,
        ConeStrategy.HELICAL: _helical,
    }[strategy]
    paths = builder(face, record, geometry, settings)
    motions = tuple(
        create_curved_motion_segment(path, settings, segment_id_start + i)
        for i, path in enumerate(paths)
    )
    return ConicalPlan(
        face_id=record.face_id, strategy=strategy, geometry=geometry,
        path_segments=tuple(paths), motion_segments=motions,
    )


def plan_conical_faces(
    faces: Iterable[TopoDS_Face], records: Iterable[FaceRecord],
    settings: PlannerSettings | None = None,
    *, strategy: ConeStrategy = ConeStrategy.LONGITUDINAL,
) -> tuple[ConicalPlan, ...]:
    return tuple(
        plan_conical_face(face, record, settings, strategy=strategy)
        for face, record in zip(faces, records)
        if record.geometry_type is SurfaceType.CONE
    )
