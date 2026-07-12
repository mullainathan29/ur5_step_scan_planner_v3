"""Exact trimmed spherical STEP-face planning."""

from __future__ import annotations

import math
import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.TopAbs import TopAbs_State
from OCP.TopoDS import TopoDS_Face
from OCP.gp import gp_Pnt, gp_Pnt2d, gp_Vec

from face_analysis import FaceRecord, SurfaceType
from planners.base import (
    PlannerSettings, PlanningError, RasterSegment, SphereGeometry,
    SphereStrategy, SphericalPlan, SurfaceSample,
)
from pose_generation import create_curved_motion_segment


def _array(value):
    return np.array([value.X(), value.Y(), value.Z()], dtype=float)


def exact_sphere_geometry(face: TopoDS_Face) -> SphereGeometry:
    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_SurfaceType.GeomAbs_Sphere:
        raise PlanningError("The supplied face is not an exact sphere.")
    sphere = adaptor.Sphere()
    return SphereGeometry(
        center=_array(sphere.Location()),
        radius_mm=float(sphere.Radius()),
        angular_bounds=(float(adaptor.FirstUParameter()), float(adaptor.LastUParameter())),
        latitude_bounds=(float(adaptor.FirstVParameter()), float(adaptor.LastVParameter())),
    )


def _inside(face, u, v, tol):
    return BRepClass_FaceClassifier(face, gp_Pnt2d(u, v), tol).State() in (
        TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON
    )


def _evaluate(adaptor, record, u, v):
    point, du, dv = gp_Pnt(), gp_Vec(), gp_Vec()
    adaptor.D1(u, v, point, du, dv)
    normal = np.cross(_array(du), _array(dv))
    if np.linalg.norm(normal) < 1e-9:
        raise PlanningError("Sphere derivative is singular near a pole.")
    normal /= np.linalg.norm(normal)
    if record.reversed_orientation:
        normal = -normal
    return _array(point), normal


def _runs(face, values, tol):
    output, current = [], []
    for uv in values:
        if _inside(face, *uv, tol):
            current.append(uv)
        elif current:
            if len(current) > 1: output.append(current)
            current = []
    if len(current) > 1: output.append(current)
    return output


def _make(adaptor, record, lane, section, values):
    samples = []
    for i, uv in enumerate(values):
        point, normal = _evaluate(adaptor, record, *uv)
        samples.append(SurfaceSample(
            record.face_id, lane, section, i, tuple(uv), point, normal
        ))
    return RasterSegment(record.face_id, lane, section, values[0][0], tuple(samples))


def plan_spherical_face(
    face: TopoDS_Face, record: FaceRecord,
    settings: PlannerSettings | None = None, *,
    strategy: SphereStrategy = SphereStrategy.AUTOMATIC,
    segment_id_start: int = 0,
) -> SphericalPlan:
    settings = settings or PlannerSettings()
    settings.validate()
    if record.geometry_type is not SurfaceType.SPHERE:
        raise PlanningError("Face is not classified as SPHERE.")
    geometry = exact_sphere_geometry(face)
    u0, u1 = geometry.angular_bounds
    v0, v1 = geometry.latitude_bounds
    pole_margin = max(1e-5, settings.travel_spacing_mm / geometry.radius_mm * 0.05)
    v0, v1 = max(v0, -math.pi / 2 + pole_margin), min(v1, math.pi / 2 - pole_margin)
    chosen = SphereStrategy(strategy)
    if chosen is SphereStrategy.AUTOMATIC:
        chosen = SphereStrategy.LATITUDE if (u1-u0) >= (v1-v0) else SphereStrategy.LONGITUDE
    adaptor = BRepAdaptor_Surface(face, True)
    paths = []
    if chosen is SphereStrategy.LATITUDE:
        lanes = max(1, math.ceil(geometry.radius_mm * (v1-v0) / settings.lane_spacing_mm))
        latitudes = np.linspace(v0, v1, lanes + 2)[1:-1]
        for lane, v in enumerate(latitudes):
            radius = geometry.radius_mm * abs(math.cos(v))
            count = max(1, math.ceil(radius * (u1-u0) / settings.travel_spacing_mm))
            us = np.linspace(u0, u1, count+1)
            if lane % 2: us = us[::-1]
            for section, run in enumerate(_runs(face, [(float(u),float(v)) for u in us], settings.geometric_tolerance_mm)):
                paths.append(_make(adaptor, record, lane, section, run))
    else:
        lanes = max(1, math.ceil(geometry.radius_mm * (u1-u0) / settings.lane_spacing_mm))
        longitudes = u0 + (np.arange(lanes)+0.5)*(u1-u0)/lanes
        count = max(1, math.ceil(geometry.radius_mm*(v1-v0)/settings.travel_spacing_mm))
        for lane, u in enumerate(longitudes):
            vs=np.linspace(v0,v1,count+1)
            if lane%2: vs=vs[::-1]
            for section, run in enumerate(_runs(face, [(float(u),float(v)) for v in vs], settings.geometric_tolerance_mm)):
                paths.append(_make(adaptor, record, lane, section, run))
    motions=tuple(create_curved_motion_segment(p,settings,segment_id_start+i) for i,p in enumerate(paths))
    return SphericalPlan(record.face_id,chosen,geometry,tuple(paths),motions)
