"""Exact raster planning for trimmed planar STEP faces."""

from __future__ import annotations

import math
from collections.abc import Iterable

import cadquery as cq
import numpy as np
from OCP.BRepAdaptor import (
    BRepAdaptor_Curve,
    BRepAdaptor_Curve2d,
    BRepAdaptor_Surface,
)
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.TopAbs import TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face
from OCP.gp import gp_Pnt

from face_analysis import FaceRecord, SurfaceType
from planners.base import (
    BoundaryCurve2D,
    BoundaryWire2D,
    PlaneFrame,
    PlanarRasterPlan,
    PlannerSettings,
    PlanningError,
    RasterSegment,
    SurfaceSample,
)
from pose_generation import create_planar_motion_segment


def _array_from_direction(direction: object) -> np.ndarray:
    return np.array(
        [direction.X(), direction.Y(), direction.Z()], dtype=np.float64
    )


def _array_from_point(point: object) -> np.ndarray:
    return np.array([point.X(), point.Y(), point.Z()], dtype=np.float64)


def exact_plane_frame(
    face: TopoDS_Face,
    record: FaceRecord,
) -> PlaneFrame:
    """Obtain the exact OCC plane and construct a right-handed scanner frame."""

    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_SurfaceType.GeomAbs_Plane:
        raise PlanningError(f"Face {record.face_id} is not an exact plane.")

    plane = adaptor.Plane()
    axes = plane.Position()
    origin = _array_from_point(axes.Location())
    laser_axis = _array_from_direction(axes.XDirection())
    normal = np.array(
        [
            record.representative_normal.x,
            record.representative_normal.y,
            record.representative_normal.z,
        ],
        dtype=np.float64,
    )
    normal /= np.linalg.norm(normal)

    # local +Z points toward the surface. Enforce X cross Y = Z.
    scanner_z = -normal
    travel_axis = np.cross(scanner_z, laser_axis)
    travel_axis /= np.linalg.norm(travel_axis)
    laser_axis = np.cross(travel_axis, scanner_z)
    laser_axis /= np.linalg.norm(laser_axis)
    return PlaneFrame(
        origin=origin,
        laser_axis=laser_axis,
        travel_axis=travel_axis,
        outward_normal=normal,
    )


def map_boundary_wires_to_plane(
    face: TopoDS_Face,
    record: FaceRecord,
) -> tuple[BoundaryWire2D, ...]:
    """Map each boundary edge to its exact native 2D OCC p-curve."""

    mapped: list[BoundaryWire2D] = []
    for wire_index, topods_wire in enumerate(record.boundary_wires):
        wire = cq.Wire(topods_wire)
        curves: list[BoundaryCurve2D] = []
        for edge in wire.Edges():
            curve = BRepAdaptor_Curve2d(edge.wrapped, face)
            curves.append(
                BoundaryCurve2D(
                    edge=edge.wrapped,
                    curve=curve,
                    first_parameter=float(curve.FirstParameter()),
                    last_parameter=float(curve.LastParameter()),
                )
            )
        mapped.append(
            BoundaryWire2D(is_outer=(wire_index == 0), curves=tuple(curves))
        )
    return tuple(mapped)


def _point_on_plane(frame: PlaneFrame, u: float, v: float) -> gp_Pnt:
    point = frame.origin + u * frame.laser_axis + v * frame.travel_axis
    return gp_Pnt(float(point[0]), float(point[1]), float(point[2]))


def _plane_coordinates(frame: PlaneFrame, point: np.ndarray) -> tuple[float, float]:
    relative = point - frame.origin
    return (
        float(np.dot(relative, frame.laser_axis)),
        float(np.dot(relative, frame.travel_axis)),
    )


def _native_uv_coordinates(
    face: TopoDS_Face,
    point: np.ndarray,
) -> tuple[float, float]:
    """Project a 3D point into the OCC plane's native U/V coordinates."""

    plane_axes = BRepAdaptor_Surface(face, True).Plane().Position()
    origin = _array_from_point(plane_axes.Location())
    native_u_axis = _array_from_direction(plane_axes.XDirection())
    native_v_axis = _array_from_direction(plane_axes.YDirection())
    relative = point - origin
    return (
        float(np.dot(relative, native_u_axis)),
        float(np.dot(relative, native_v_axis)),
    )


def _raster_coordinates(
    minimum: float,
    maximum: float,
    spacing: float,
    tolerance: float,
) -> np.ndarray:
    span = maximum - minimum
    if span <= tolerance:
        return np.empty(0, dtype=np.float64)
    if span <= spacing:
        return np.array([(minimum + maximum) * 0.5], dtype=np.float64)
    count = max(1, int(math.floor(span / spacing)))
    covered = (count - 1) * spacing
    start = 0.5 * (minimum + maximum - covered)
    return start + spacing * np.arange(count, dtype=np.float64)


def _clipped_edges(
    face: TopoDS_Face,
    frame: PlaneFrame,
    u: float,
    v_min: float,
    v_max: float,
    tolerance: float,
) -> list[tuple[float, float, BRepAdaptor_Curve]]:
    margin = max(v_max - v_min, 1.0) + 10.0 * tolerance
    line_edge = BRepBuilderAPI_MakeEdge(
        _point_on_plane(frame, u, v_min - margin),
        _point_on_plane(frame, u, v_max + margin),
    ).Edge()
    common = BRepAlgoAPI_Common(line_edge, face)
    common.Build()
    if not common.IsDone():
        raise PlanningError("OpenCascade failed to clip a raster line to the face.")

    sections: list[tuple[float, float, BRepAdaptor_Curve]] = []
    explorer = TopExp_Explorer(common.Shape(), TopAbs_EDGE)
    while explorer.More():
        edge = TopoDS.Edge_s(explorer.Current())
        curve = BRepAdaptor_Curve(edge)
        first = float(curve.FirstParameter())
        last = float(curve.LastParameter())
        first_point = _array_from_point(curve.Value(first))
        last_point = _array_from_point(curve.Value(last))
        _, first_v = _plane_coordinates(frame, first_point)
        _, last_v = _plane_coordinates(frame, last_point)
        if abs(last_v - first_v) > tolerance:
            sections.append((min(first_v, last_v), max(first_v, last_v), curve))
        explorer.Next()
    sections.sort(key=lambda section: section[0])
    return sections


def _sample_section(
    face: TopoDS_Face,
    face_id: int,
    lane_index: int,
    section_index: int,
    u: float,
    v_start: float,
    v_end: float,
    frame: PlaneFrame,
    settings: PlannerSettings,
    reverse: bool,
) -> RasterSegment:
    length = v_end - v_start
    interval_count = max(1, int(math.ceil(length / settings.travel_spacing_mm)))
    coordinates = np.linspace(v_start, v_end, interval_count + 1)
    if reverse:
        coordinates = coordinates[::-1]
    samples: list[SurfaceSample] = []
    for sample_index, v in enumerate(coordinates):
        point = frame.origin + u * frame.laser_axis + v * frame.travel_axis
        native_uv = _native_uv_coordinates(face, point)
        samples.append(
            SurfaceSample(
                face_id=face_id,
                lane_index=lane_index,
                section_index=section_index,
                sample_index=sample_index,
                uv=native_uv,
                point=point.astype(np.float64),
                normal=frame.outward_normal.copy(),
            )
        )
    return RasterSegment(
        face_id=face_id,
        lane_index=lane_index,
        section_index=section_index,
        laser_coordinate=float(u),
        samples=tuple(samples),
    )


def plan_planar_face(
    face: TopoDS_Face,
    record: FaceRecord,
    settings: PlannerSettings | None = None,
    *,
    segment_id_start: int = 0,
) -> PlanarRasterPlan:
    """Generate exact clipped rasters and scanner motions for one planar face."""

    settings = settings or PlannerSettings()
    settings.validate()
    if record.geometry_type is not SurfaceType.PLANE:
        raise PlanningError(f"Face {record.face_id} is not classified as PLANE.")

    frame = exact_plane_frame(face, record)
    boundary_wires = map_boundary_wires_to_plane(face, record)

    # Plane native U follows OCC X. frame travel can be the negative native V
    # direction for a forward scanner frame, so project all four UV corners.
    u_values: list[float] = []
    v_values: list[float] = []
    adaptor = BRepAdaptor_Surface(face, True)
    bounds = record.uv_bounds
    for native_u in (bounds.u_min, bounds.u_max):
        for native_v in (bounds.v_min, bounds.v_max):
            point = _array_from_point(adaptor.Value(native_u, native_v))
            u, v = _plane_coordinates(frame, point)
            u_values.append(u)
            v_values.append(v)
    u_min, u_max = min(u_values), max(u_values)
    v_min, v_max = min(v_values), max(v_values)

    rasters: list[RasterSegment] = []
    lane_coordinates = _raster_coordinates(
        u_min,
        u_max,
        settings.lane_spacing_mm,
        settings.geometric_tolerance_mm,
    )
    for lane_index, u in enumerate(lane_coordinates):
        sections = _clipped_edges(
            face,
            frame,
            float(u),
            v_min,
            v_max,
            settings.geometric_tolerance_mm,
        )
        reverse = bool(lane_index % 2)
        ordered = list(reversed(sections)) if reverse else sections
        for section_index, (section_min, section_max, _curve) in enumerate(ordered):
            rasters.append(
                _sample_section(
                    face,
                    record.face_id,
                    lane_index,
                    section_index,
                    float(u),
                    section_min,
                    section_max,
                    frame,
                    settings,
                    reverse,
                )
            )

    motions = tuple(
        create_planar_motion_segment(
            raster,
            frame,
            settings,
            segment_id_start + index,
        )
        for index, raster in enumerate(rasters)
    )
    return PlanarRasterPlan(
        face_id=record.face_id,
        frame=frame,
        boundary_wires_2d=boundary_wires,
        raster_segments=tuple(rasters),
        motion_segments=motions,
    )


def plan_planar_faces(
    faces: Iterable[TopoDS_Face],
    records: Iterable[FaceRecord],
    settings: PlannerSettings | None = None,
) -> tuple[PlanarRasterPlan, ...]:
    """Plan every planar face while keeping each face and section independent."""

    settings = settings or PlannerSettings()
    plans: list[PlanarRasterPlan] = []
    next_segment_id = 0
    for face, record in zip(faces, records):
        if record.geometry_type is not SurfaceType.PLANE:
            continue
        plan = plan_planar_face(
            face,
            record,
            settings,
            segment_id_start=next_segment_id,
        )
        plans.append(plan)
        next_segment_id += len(plan.motion_segments)
    return tuple(plans)
