"""Exact longitudinal and helical planning for trimmed cylindrical STEP faces."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
from OCP.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeEdge
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.GeomAbs import GeomAbs_SurfaceType
from OCP.TopAbs import TopAbs_EDGE, TopAbs_State
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS, TopoDS_Face
from OCP.gp import gp_Pnt, gp_Pnt2d

from face_analysis import FaceRecord, SurfaceType
from planners.base import (
    CylinderGeometry,
    CylinderStrategy,
    CylindricalPlan,
    PlannerSettings,
    PlanningError,
    RasterSegment,
    SurfaceSample,
)
from pose_generation import create_curved_motion_segment


def _point(point: object) -> np.ndarray:
    return np.array([point.X(), point.Y(), point.Z()], dtype=np.float64)


def _direction(direction: object) -> np.ndarray:
    return np.array(
        [direction.X(), direction.Y(), direction.Z()], dtype=np.float64
    )


def exact_cylinder_geometry(face: TopoDS_Face) -> CylinderGeometry:
    """Extract exact axis, radius, and native trimmed parameter bounds."""

    adaptor = BRepAdaptor_Surface(face, True)
    if adaptor.GetType() != GeomAbs_SurfaceType.GeomAbs_Cylinder:
        raise PlanningError("The supplied face is not an exact cylinder.")
    cylinder = adaptor.Cylinder()
    axes = cylinder.Position()
    return CylinderGeometry(
        axis_origin=_point(axes.Location()),
        axis_direction=_direction(axes.Direction()),
        x_direction=_direction(axes.XDirection()),
        y_direction=_direction(axes.YDirection()),
        radius_mm=float(cylinder.Radius()),
        angular_bounds=(
            float(adaptor.FirstUParameter()),
            float(adaptor.LastUParameter()),
        ),
        axial_bounds=(
            float(adaptor.FirstVParameter()),
            float(adaptor.LastVParameter()),
        ),
        periodic=bool(adaptor.IsUPeriodic()),
    )


def _exact_point_normal(
    adaptor: BRepAdaptor_Surface,
    record: FaceRecord,
    u: float,
    v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate an exact surface point and orientation-aware OCC normal."""

    from OCP.gp import gp_Pnt, gp_Vec

    point = gp_Pnt()
    du = gp_Vec()
    dv = gp_Vec()
    adaptor.D1(float(u), float(v), point, du, dv)
    du_array = _direction(du)
    dv_array = _direction(dv)
    normal = np.cross(du_array, dv_array)
    normal /= np.linalg.norm(normal)
    if record.reversed_orientation:
        normal = -normal
    return _point(point), normal


def _lane_angles(geometry: CylinderGeometry, spacing_mm: float) -> np.ndarray:
    u_min, u_max = geometry.angular_bounds
    arc_length = geometry.radius_mm * (u_max - u_min)
    count = max(1, int(math.ceil(arc_length / spacing_mm)))
    angular_step = (u_max - u_min) / count
    return u_min + (np.arange(count, dtype=np.float64) + 0.5) * angular_step


def _axis_line_point(
    geometry: CylinderGeometry,
    angle: float,
    axial: float,
) -> np.ndarray:
    radial = (
        math.cos(angle) * geometry.x_direction
        + math.sin(angle) * geometry.y_direction
    )
    return (
        geometry.axis_origin
        + geometry.radius_mm * radial
        + axial * geometry.axis_direction
    )


def _clipped_axial_sections(
    face: TopoDS_Face,
    geometry: CylinderGeometry,
    angle: float,
    tolerance: float,
) -> list[tuple[float, float]]:
    v_min, v_max = geometry.axial_bounds
    margin = max(v_max - v_min, 1.0) + 10.0 * tolerance
    start = _axis_line_point(geometry, angle, v_min - margin)
    end = _axis_line_point(geometry, angle, v_max + margin)
    line = BRepBuilderAPI_MakeEdge(
        gp_Pnt(*[float(value) for value in start]),
        gp_Pnt(*[float(value) for value in end]),
    ).Edge()
    common = BRepAlgoAPI_Common(line, face)
    common.Build()
    if not common.IsDone():
        raise PlanningError("OpenCascade failed to trim a cylindrical lane.")

    sections: list[tuple[float, float]] = []
    explorer = TopExp_Explorer(common.Shape(), TopAbs_EDGE)
    while explorer.More():
        curve = BRepAdaptor_Curve(TopoDS.Edge_s(explorer.Current()))
        first = _point(curve.Value(curve.FirstParameter()))
        last = _point(curve.Value(curve.LastParameter()))
        first_v = float(np.dot(first - geometry.axis_origin, geometry.axis_direction))
        last_v = float(np.dot(last - geometry.axis_origin, geometry.axis_direction))
        if abs(last_v - first_v) > tolerance:
            sections.append((min(first_v, last_v), max(first_v, last_v)))
        explorer.Next()
    sections.sort()
    if not sections:
        sections = _classified_axial_sections(
            face, angle, v_min, v_max, tolerance
        )
    return sections


def _classified_axial_sections(
    face: TopoDS_Face,
    angle: float,
    v_min: float,
    v_max: float,
    tolerance: float,
) -> list[tuple[float, float]]:
    """Fallback UV trim clipping for unstable coincident 3D booleans."""

    probe_values = np.linspace(v_min, v_max, 257)
    valid = [_is_inside(face, angle, float(v), tolerance) for v in probe_values]

    def boundary(a: float, b: float, a_valid: bool) -> float:
        for _ in range(60):
            middle = 0.5 * (a + b)
            if _is_inside(face, angle, middle, tolerance) == a_valid:
                a = middle
            else:
                b = middle
            if abs(b - a) <= tolerance:
                break
        return a if a_valid else b

    sections: list[tuple[float, float]] = []
    start: float | None = v_min if valid[0] else None
    for index in range(1, len(probe_values)):
        if valid[index] == valid[index - 1]:
            continue
        transition = boundary(
            float(probe_values[index - 1]),
            float(probe_values[index]),
            valid[index - 1],
        )
        if valid[index - 1] and start is not None:
            sections.append((start, transition))
            start = None
        elif valid[index]:
            start = transition
    if valid[-1] and start is not None:
        sections.append((start, v_max))
    return [
        section
        for section in sections
        if section[1] - section[0] > tolerance
    ]


def _sample_uv_section(
    adaptor: BRepAdaptor_Surface,
    record: FaceRecord,
    lane_index: int,
    section_index: int,
    uv_values: list[tuple[float, float]],
) -> RasterSegment:
    samples = tuple(
        SurfaceSample(
            face_id=record.face_id,
            lane_index=lane_index,
            section_index=section_index,
            sample_index=index,
            uv=(float(u), float(v)),
            point=point,
            normal=normal,
        )
        for index, ((u, v), (point, normal)) in enumerate(
            (
                (uv, _exact_point_normal(adaptor, record, *uv))
                for uv in uv_values
            )
        )
    )
    return RasterSegment(
        face_id=record.face_id,
        lane_index=lane_index,
        section_index=section_index,
        laser_coordinate=float(uv_values[0][0]),
        samples=samples,
    )


def _longitudinal_paths(
    face: TopoDS_Face,
    record: FaceRecord,
    geometry: CylinderGeometry,
    settings: PlannerSettings,
) -> list[RasterSegment]:
    adaptor = BRepAdaptor_Surface(face, True)
    paths: list[RasterSegment] = []
    for lane_index, angle in enumerate(
        _lane_angles(geometry, settings.lane_spacing_mm)
    ):
        sections = _clipped_axial_sections(
            face, geometry, float(angle), settings.geometric_tolerance_mm
        )
        reverse = bool(lane_index % 2)
        ordered = list(reversed(sections)) if reverse else sections
        for section_index, (v_min, v_max) in enumerate(ordered):
            intervals = max(
                1, int(math.ceil((v_max - v_min) / settings.travel_spacing_mm))
            )
            axial_values = np.linspace(v_min, v_max, intervals + 1)
            if reverse:
                axial_values = axial_values[::-1]
            paths.append(
                _sample_uv_section(
                    adaptor,
                    record,
                    lane_index,
                    section_index,
                    [(float(angle), float(v)) for v in axial_values],
                )
            )
    return paths


def _is_inside(face: TopoDS_Face, u: float, v: float, tolerance: float) -> bool:
    state = BRepClass_FaceClassifier(
        face, gp_Pnt2d(float(u), float(v)), tolerance
    ).State()
    return state in (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON)


def _helical_paths(
    face: TopoDS_Face,
    record: FaceRecord,
    geometry: CylinderGeometry,
    settings: PlannerSettings,
) -> list[RasterSegment]:
    """Sample exact cylinder UV points and split at trims or angular resets."""

    adaptor = BRepAdaptor_Surface(face, True)
    u_min, u_max = geometry.angular_bounds
    v_min, v_max = geometry.axial_bounds
    angular_span = u_max - u_min
    axial_span = v_max - v_min
    turns = max(1, int(math.ceil(axial_span / settings.lane_spacing_mm)))
    total_theta = angular_span * turns
    approximate_length = math.hypot(
        geometry.radius_mm * total_theta, axial_span
    )
    intervals = max(
        1, int(math.ceil(approximate_length / settings.travel_spacing_mm))
    )
    theta_values = np.linspace(0.0, total_theta, intervals + 1)

    paths: list[RasterSegment] = []
    current: list[tuple[float, float]] = []
    section_index = 0
    previous_point: np.ndarray | None = None
    max_step = settings.travel_spacing_mm * 1.5 + settings.geometric_tolerance_mm

    def finish() -> None:
        nonlocal current, section_index
        if len(current) >= 2:
            paths.append(
                _sample_uv_section(
                    adaptor, record, 0, section_index, current
                )
            )
            section_index += 1
        current = []

    for theta in theta_values:
        fraction = float(theta / total_theta) if total_theta > 0.0 else 0.0
        unwrapped_u = u_min + float(theta)
        if math.isclose(theta, total_theta):
            native_u = u_max
        else:
            native_u = u_min + ((unwrapped_u - u_min) % angular_span)
        v = v_min + fraction * axial_span
        if not _is_inside(
            face, native_u, v, settings.geometric_tolerance_mm
        ):
            finish()
            previous_point = None
            continue
        point, _ = _exact_point_normal(adaptor, record, native_u, v)
        if previous_point is not None and np.linalg.norm(point - previous_point) > max_step:
            finish()
        current.append((native_u, v))
        previous_point = point
    finish()
    return paths


def plan_cylindrical_face(
    face: TopoDS_Face,
    record: FaceRecord,
    settings: PlannerSettings | None = None,
    *,
    strategy: CylinderStrategy = CylinderStrategy.LONGITUDINAL,
    segment_id_start: int = 0,
) -> CylindricalPlan:
    """Plan one exact cylindrical face using the selected strategy."""

    settings = settings or PlannerSettings()
    settings.validate()
    strategy = CylinderStrategy(strategy)
    if record.geometry_type is not SurfaceType.CYLINDER:
        raise PlanningError(f"Face {record.face_id} is not classified as CYLINDER.")
    geometry = exact_cylinder_geometry(face)
    if strategy is CylinderStrategy.LONGITUDINAL:
        paths = _longitudinal_paths(face, record, geometry, settings)
    else:
        paths = _helical_paths(face, record, geometry, settings)
    motions = tuple(
        create_curved_motion_segment(
            path, settings, segment_id_start + index
        )
        for index, path in enumerate(paths)
    )
    return CylindricalPlan(
        face_id=record.face_id,
        strategy=strategy,
        geometry=geometry,
        path_segments=tuple(paths),
        motion_segments=motions,
    )


def plan_cylindrical_faces(
    faces: Iterable[TopoDS_Face],
    records: Iterable[FaceRecord],
    settings: PlannerSettings | None = None,
    *,
    strategy: CylinderStrategy = CylinderStrategy.LONGITUDINAL,
) -> tuple[CylindricalPlan, ...]:
    """Plan all cylindrical faces without connecting unrelated faces."""

    settings = settings or PlannerSettings()
    plans: list[CylindricalPlan] = []
    next_segment_id = 0
    for face, record in zip(faces, records):
        if record.geometry_type is not SurfaceType.CYLINDER:
            continue
        plan = plan_cylindrical_face(
            face,
            record,
            settings,
            strategy=strategy,
            segment_id_start=next_segment_id,
        )
        plans.append(plan)
        next_segment_id += len(plan.motion_segments)
    return tuple(plans)
