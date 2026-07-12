"""Exact topology-driven planar and cylindrical scan-region construction."""

from __future__ import annotations

from dataclasses import dataclass

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Face

from face_analysis import FaceRecord, SurfaceType, extract_face_records
from planners.base import BoundaryWire2D, CylinderGeometry, PlaneFrame
from planners.cylinder import exact_cylinder_geometry
from planners.plane import exact_plane_frame, map_boundary_wires_to_plane


@dataclass(frozen=True)
class PlanarScanRegion:
    region_id: int
    member_face_ids: tuple[int, ...]
    total_area_mm2: float
    common_plane_frame: PlaneFrame
    merged_outer_boundaries: tuple[BoundaryWire2D, ...]
    merged_inner_holes: tuple[BoundaryWire2D, ...]
    bounding_size_mm: tuple[float, float]
    representative_normal: np.ndarray
    planning_faces: tuple[TopoDS_Face, ...]


@dataclass(frozen=True)
class CylindricalScanRegion:
    region_id: int
    member_face_ids: tuple[int, ...]
    axis_origin: np.ndarray
    axis_direction: np.ndarray
    radius_mm: float
    combined_axial_limits: tuple[float, float]
    combined_angular_intervals: tuple[tuple[float, float], ...]
    inward_facing: bool
    total_area_mm2: float
    planning_faces: tuple[TopoDS_Face, ...]


def _components(ids: list[int], compatible) -> list[list[int]]:
    remaining = set(ids)
    groups: list[list[int]] = []
    while remaining:
        root = remaining.pop()
        group = [root]
        queue = [root]
        while queue:
            current = queue.pop()
            for candidate in list(remaining):
                if compatible(current, candidate):
                    remaining.remove(candidate)
                    group.append(candidate)
                    queue.append(candidate)
        groups.append(sorted(group))
    return sorted(groups, key=lambda group: group[0])


def _clean_fused_faces(
    member_ids: list[int], faces: tuple[TopoDS_Face, ...]
) -> tuple[TopoDS_Face, ...]:
    shape: cq.Shape = cq.Face(faces[member_ids[0]])
    for face_id in member_ids[1:]:
        shape = shape.fuse(cq.Face(faces[face_id]))
    cleaned = shape.clean()
    return tuple(face.wrapped for face in cleaned.Faces())


def build_planar_regions(
    faces: tuple[TopoDS_Face, ...],
    records: tuple[FaceRecord, ...],
    eligible_face_ids: set[int],
    *,
    normal_tolerance_deg: float = 2.0,
    plane_distance_tolerance_mm: float = 0.1,
    region_id_start: int = 0,
) -> tuple[PlanarScanRegion, ...]:
    ids = [
        record.face_id
        for record in records
        if record.face_id in eligible_face_ids
        and record.geometry_type is SurfaceType.PLANE
    ]
    normals = {
        face_id: np.array(
            [
                records[face_id].representative_normal.x,
                records[face_id].representative_normal.y,
                records[face_id].representative_normal.z,
            ]
        )
        for face_id in ids
    }
    offsets = {
        face_id: float(
            np.dot(
                normals[face_id],
                np.array(
                    [
                        records[face_id].center.x,
                        records[face_id].center.y,
                        records[face_id].center.z,
                    ]
                ),
            )
        )
        for face_id in ids
    }

    def compatible(first: int, second: int) -> bool:
        if second not in records[first].adjacent_face_ids:
            return False
        dot = float(np.clip(np.dot(normals[first], normals[second]), -1.0, 1.0))
        angle = float(np.degrees(np.arccos(dot)))
        return (
            angle < normal_tolerance_deg
            and abs(offsets[first] - offsets[second])
            < plane_distance_tolerance_mm
        )

    regions: list[PlanarScanRegion] = []
    for index, members in enumerate(_components(ids, compatible)):
        planning_faces = _clean_fused_faces(members, faces)
        frame = exact_plane_frame(faces[members[0]], records[members[0]])
        outer: list[BoundaryWire2D] = []
        holes: list[BoundaryWire2D] = []
        u_values: list[float] = []
        v_values: list[float] = []
        for planning_face in planning_faces:
            wrapped = cq.Face(planning_face)
            planning_record = extract_face_records((planning_face,))[0]
            mapped_wires = map_boundary_wires_to_plane(
                planning_face, planning_record
            )
            outer.extend(wire for wire in mapped_wires if wire.is_outer)
            holes.extend(wire for wire in mapped_wires if not wire.is_outer)
            for vertex in wrapped.Vertices():
                point = np.array([vertex.X, vertex.Y, vertex.Z])
                delta = point - frame.origin
                u_values.append(float(np.dot(delta, frame.laser_axis)))
                v_values.append(float(np.dot(delta, frame.travel_axis)))
        regions.append(
            PlanarScanRegion(
                region_id=region_id_start + index,
                member_face_ids=tuple(members),
                total_area_mm2=sum(records[i].area_mm2 for i in members),
                common_plane_frame=frame,
                merged_outer_boundaries=tuple(outer),
                merged_inner_holes=tuple(holes),
                bounding_size_mm=(
                    max(u_values) - min(u_values),
                    max(v_values) - min(v_values),
                ),
                representative_normal=normals[members[0]].copy(),
                planning_faces=planning_faces,
            )
        )
    return tuple(regions)


def _ranges_touch(first: tuple[float, float], second: tuple[float, float], tol: float) -> bool:
    return first[0] <= second[1] + tol and second[0] <= first[1] + tol


def _axial_interval_in_frame(
    geometry: CylinderGeometry, frame: CylinderGeometry
) -> tuple[float, float]:
    offset = float(
        np.dot(
            geometry.axis_origin - frame.axis_origin,
            frame.axis_direction,
        )
    )
    sign = 1.0 if np.dot(
        geometry.axis_direction, frame.axis_direction
    ) >= 0.0 else -1.0
    values = (
        offset + sign * geometry.axial_bounds[0],
        offset + sign * geometry.axial_bounds[1],
    )
    return min(values), max(values)


def build_cylindrical_regions(
    faces: tuple[TopoDS_Face, ...],
    records: tuple[FaceRecord, ...],
    eligible_face_ids: set[int],
    *,
    axis_angle_tolerance_deg: float = 2.0,
    axis_distance_tolerance_mm: float = 0.1,
    radius_tolerance_mm: float = 0.1,
    region_id_start: int = 0,
) -> tuple[CylindricalScanRegion, ...]:
    ids = [
        record.face_id
        for record in records
        if record.face_id in eligible_face_ids
        and record.geometry_type is SurfaceType.CYLINDER
    ]
    geometry = {face_id: exact_cylinder_geometry(faces[face_id]) for face_id in ids}

    def compatible(first: int, second: int) -> bool:
        if second not in records[first].adjacent_face_ids:
            return False
        a, b = geometry[first], geometry[second]
        dot = abs(float(np.clip(np.dot(a.axis_direction, b.axis_direction), -1, 1)))
        angle = float(np.degrees(np.arccos(dot)))
        axis_distance = float(
            np.linalg.norm(
                np.cross(b.axis_origin - a.axis_origin, a.axis_direction)
            )
        )
        b_axial_in_a = _axial_interval_in_frame(b, a)
        ranges_contiguous = _ranges_touch(a.axial_bounds, b_axial_in_a, 0.1) or _ranges_touch(
            a.angular_bounds, b.angular_bounds, 1.0e-6
        )
        return (
            angle < axis_angle_tolerance_deg
            and axis_distance < axis_distance_tolerance_mm
            and abs(a.radius_mm - b.radius_mm) < radius_tolerance_mm
            and records[first].reversed_orientation
            == records[second].reversed_orientation
            and ranges_contiguous
        )

    regions: list[CylindricalScanRegion] = []
    for index, members in enumerate(_components(ids, compatible)):
        planning_faces = _clean_fused_faces(members, faces)
        base = geometry[members[0]]
        axial_intervals = [
            _axial_interval_in_frame(geometry[i], base) for i in members
        ]
        regions.append(
            CylindricalScanRegion(
                region_id=region_id_start + index,
                member_face_ids=tuple(members),
                axis_origin=base.axis_origin.copy(),
                axis_direction=base.axis_direction.copy(),
                radius_mm=base.radius_mm,
                combined_axial_limits=(
                    min(interval[0] for interval in axial_intervals),
                    max(interval[1] for interval in axial_intervals),
                ),
                combined_angular_intervals=tuple(
                    geometry[i].angular_bounds for i in members
                ),
                inward_facing=records[members[0]].reversed_orientation,
                total_area_mm2=sum(records[i].area_mm2 for i in members),
                planning_faces=planning_faces,
            )
        )
    return tuple(regions)
