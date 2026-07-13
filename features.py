"""STEP B-Rep internal-hole detection and optical feature reporting."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, pi

import numpy as np

from cad_model import CadModel
from coverage import CoverageResult, TriangleStatus, detect_exterior_face_ids
from face_analysis import SurfaceType
from planners.cylinder import exact_cylinder_geometry


@dataclass(frozen=True)
class HoleSection:
    diameter_mm: float
    start_depth_mm: float
    end_depth_mm: float
    face_ids: tuple[int, ...]
    section_type: str


@dataclass(frozen=True)
class HoleFeature:
    feature_id: int
    cylindrical_face_ids: tuple[int, ...]
    conical_face_ids: tuple[int, ...]
    centre: np.ndarray
    axis: np.ndarray
    radius_mm: float
    depth_mm: float
    entry_opening_centre: np.ndarray
    entry_opening_normal: np.ndarray
    through_hole: bool
    accessible_opening_directions: tuple[tuple[float, float, float], ...]
    host_face_id: int | None
    bottom_face_id: int | None
    current_optical_coverage_percent: float = 0.0
    sections: tuple[HoleSection, ...] = ()


@dataclass(frozen=True)
class HoleCoverageReport:
    feature_id: int
    rim_coverage_percent: float
    cylindrical_wall_coverage_percent: float
    bottom_face_coverage_percent: float
    maximum_visible_depth_mm: float
    uncovered_angular_sectors_deg: tuple[tuple[float, float], ...]
    uncovered_axial_depth_ranges_mm: tuple[tuple[float, float], ...]
    complete: bool


def _canonical_axis(axis: np.ndarray) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    dominant = int(np.argmax(np.abs(axis)))
    return -axis if axis[dominant] < 0 else axis


def _same_hole(first, second, axis_tol=1.0e-4, radius_tol=1.0e-3) -> bool:
    a0, r0, origin0 = first
    a1, r1, origin1 = second
    parallel = abs(float(np.dot(a0, a1))) >= 1.0 - axis_tol
    offset = origin1 - origin0
    axis_distance = np.linalg.norm(offset - np.dot(offset, a0) * a0)
    return parallel and abs(r0 - r1) <= radius_tol and axis_distance <= radius_tol


def detect_hole_features(
    model: CadModel,
    *,
    lightweight: bool = False,
    cancellation_check=None,
) -> tuple[HoleFeature, ...]:
    """Group reversed, non-exterior cylindrical faces into physical holes."""

    cancellation_check = cancellation_check or (lambda: None)
    exterior = frozenset() if lightweight else detect_exterior_face_ids(model)
    candidates = []
    for record, face in zip(model.faces, model.topods_faces):
        cancellation_check()
        if record.geometry_type not in {SurfaceType.CYLINDER, SurfaceType.CONE}:
            continue
        # Reversed analytic cylinders are cavity walls. Requiring either
        # reversed orientation or optical inaccessibility rejects bosses.
        if not record.reversed_orientation and (lightweight or record.face_id in exterior):
            continue
        if record.geometry_type is SurfaceType.CYLINDER:
            geometry = exact_cylinder_geometry(face)
            axis = _canonical_axis(geometry.axis_direction)
            endpoints = tuple(
                geometry.axis_origin + value * geometry.axis_direction
                for value in geometry.axial_bounds
            )
            radii = (geometry.radius_mm, geometry.radius_mm)
            origin = geometry.axis_origin
            kind = "cylindrical"
        else:
            from planners.cone import exact_cone_geometry
            from OCP.BRepAdaptor import BRepAdaptor_Surface
            geometry = exact_cone_geometry(face)
            axis = _canonical_axis(geometry.axis_direction)
            radii = tuple(abs(
                geometry.reference_radius_mm
                + value * np.sin(geometry.semi_angle_rad)
            ) for value in geometry.axial_bounds)
            adaptor = BRepAdaptor_Surface(face, True)
            u = geometry.angular_bounds[0]
            radial = np.cos(u) * geometry.x_direction + np.sin(u) * geometry.y_direction
            surface_points = [adaptor.Value(float(u), float(value))
                              for value in geometry.axial_bounds]
            endpoints = tuple(
                np.asarray([point.X(), point.Y(), point.Z()]) - radius * radial
                for point, radius in zip(surface_points, radii)
            )
            origin = endpoints[0]
            kind = "conical"
        candidates.append({
            "face_id": record.face_id, "record": record, "axis": axis,
            "origin": origin, "endpoints": endpoints, "radii": radii,
            "kind": kind,
        })

    def coaxial(first, second) -> bool:
        if abs(float(np.dot(first["axis"], second["axis"]))) < 1.0 - 1.0e-4:
            return False
        offset = second["origin"] - first["origin"]
        return np.linalg.norm(offset - np.dot(offset, first["axis"]) * first["axis"]) <= 1.0e-3

    def consecutive(first, second) -> bool:
        distances = [np.linalg.norm(a - b) for a in first["endpoints"]
                     for b in second["endpoints"]]
        return min(distances) <= max(0.1, 0.02 * max((*first["radii"], *second["radii"])))

    def connected(first, second) -> bool:
        first_id, second_id = first["face_id"], second["face_id"]
        first_adj = set(first["record"].adjacent_face_ids)
        second_adj = set(second["record"].adjacent_face_ids)
        adjacency = second_id in first_adj or first_id in second_adj
        shared_opening_face = bool(first_adj & second_adj)
        return coaxial(first, second) and consecutive(first, second) and (
            adjacency or shared_opening_face
        )

    # Connected components prevent unrelated coaxial holes from merging while
    # allowing counterbores separated by a shared annular STEP face.
    unvisited = set(range(len(candidates)))
    groups: list[list[dict]] = []
    while unvisited:
        seed = unvisited.pop()
        component = [seed]
        stack = [seed]
        while stack:
            current = stack.pop()
            neighbours = [index for index in tuple(unvisited)
                          if connected(candidates[current], candidates[index])]
            for index in neighbours:
                unvisited.remove(index); stack.append(index); component.append(index)
        groups.append([candidates[index] for index in component])

    output = []
    for feature_id, group in enumerate(groups):
        # A conical section is treated as a chamfer/countersink belonging to a
        # bore. Standalone reversed cones are ambiguous pockets/chamfers and
        # must not inflate the physical-hole count without a cylindrical wall.
        if not any(item["kind"] == "cylindrical" for item in group):
            continue
        axis = group[0]["axis"]
        origin = group[0]["origin"]
        axial_values = []
        cylindrical_ids = []
        adjacent = set()
        for item in group:
            face_id, record = item["face_id"], item["record"]
            if item["kind"] == "cylindrical":
                cylindrical_ids.append(face_id)
            adjacent.update(record.adjacent_face_ids)
            for point in item["endpoints"]:
                axial_values.append(float(np.dot(point - origin, axis)))
        low, high = min(axial_values), max(axial_values)
        low_point, high_point = origin + low * axis, origin + high * axis
        planar_adjacent = [
            face_id for face_id in adjacent
            if model.faces[face_id].geometry_type is SurfaceType.PLANE
        ]
        conical_ids = tuple(sorted(
            item["face_id"] for item in group if item["kind"] == "conical"
        ))
        openings = []
        for point, direction in ((low_point, -axis), (high_point, axis)):
            host = min(
                planar_adjacent,
                key=lambda face_id: np.linalg.norm(
                    np.asarray([model.faces[face_id].center.x,
                                model.faces[face_id].center.y,
                                model.faces[face_id].center.z]) - point
                ),
                default=None,
            )
            if host is not None:
                normal = model.faces[host].representative_normal
                normal_array = np.asarray([normal.x, normal.y, normal.z])
                if float(np.dot(normal_array, direction)) > 0.5:
                    openings.append((point, normal_array / np.linalg.norm(normal_array), host))
        through = len(openings) >= 2
        if openings:
            entry_point, entry_normal, host_face = max(
                openings, key=lambda row: row[0][2]
            )
        else:
            # Conservative fallback: the endpoint nearest the model exterior.
            entry_point, entry_normal, host_face = high_point, axis, None
        bottom_candidates = [face_id for face_id in planar_adjacent if face_id != host_face]
        bottom_face = bottom_candidates[0] if not through and bottom_candidates else None
        section_rows = []
        for item in group:
            depths = sorted(float(np.dot(entry_point - point, entry_normal))
                            for point in item["endpoints"])
            section_rows.append(HoleSection(
                diameter_mm=2.0 * float(max(item["radii"])),
                start_depth_mm=max(0.0, depths[0]),
                end_depth_mm=max(0.0, depths[1]),
                face_ids=(item["face_id"],),
                section_type=item["kind"],
            ))
        section_rows.sort(key=lambda row: (row.start_depth_mm, row.end_depth_mm))
        merged_sections: list[HoleSection] = []
        for section in section_rows:
            match_index = next((index for index, existing in enumerate(merged_sections)
                                if existing.section_type == section.section_type
                                and abs(existing.diameter_mm - section.diameter_mm) <= 1.0e-3
                                and abs(existing.start_depth_mm - section.start_depth_mm) <= 1.0e-3
                                and abs(existing.end_depth_mm - section.end_depth_mm) <= 1.0e-3), None)
            if match_index is None:
                merged_sections.append(section)
            else:
                existing = merged_sections[match_index]
                merged_sections[match_index] = HoleSection(
                    existing.diameter_mm, existing.start_depth_mm,
                    existing.end_depth_mm,
                    tuple(sorted((*existing.face_ids, *section.face_ids))),
                    existing.section_type,
                )
        output.append(HoleFeature(
            feature_id=len(output),
            cylindrical_face_ids=tuple(sorted(cylindrical_ids)),
            conical_face_ids=conical_ids,
            centre=0.5 * (low_point + high_point), axis=axis,
            radius_mm=float(max(max(item["radii"]) for item in group)),
            depth_mm=float(high - low), entry_opening_centre=entry_point,
            entry_opening_normal=entry_normal, through_hole=through,
            accessible_opening_directions=tuple(
                tuple(float(v) for v in row[1]) for row in openings
            ) or (tuple(float(v) for v in entry_normal),),
            host_face_id=host_face, bottom_face_id=bottom_face,
            sections=tuple(merged_sections),
        ))
    return tuple(output)


def _area_percent(face_ids, model, coverage) -> float:
    mapping = model.validation_mesh.triangle_face_ids
    mask = np.isin(mapping, list(face_ids))
    areas = np.asarray(model.validation_mesh.mesh.area_faces)
    total = float(areas[mask].sum())
    valid = mask & (coverage.triangle_status == TriangleStatus.VALID_COVERED)
    return 100.0 * float(areas[valid].sum()) / total if total else 0.0


def analyse_hole_coverage(
    model: CadModel,
    feature: HoleFeature,
    coverage: CoverageResult,
    target_percent: float = 90.0,
    angular_bins: int = 12,
    depth_bins: int = 8,
) -> HoleCoverageReport:
    mapping = model.validation_mesh.triangle_face_ids
    centres = np.asarray(model.validation_mesh.mesh.triangles_center)
    wall_mask = np.isin(mapping, feature.cylindrical_face_ids)
    valid = coverage.triangle_status == TriangleStatus.VALID_COVERED
    axis = feature.axis
    relative = centres - feature.entry_opening_centre
    axial = -relative @ feature.entry_opening_normal
    radial = relative - np.outer(relative @ axis, axis)
    reference = np.array([1., 0., 0.])
    if abs(np.dot(reference, axis)) > 0.9:
        reference = np.array([0., 1., 0.])
    x_axis = reference - np.dot(reference, axis) * axis
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(axis, x_axis)
    angles = np.mod(np.arctan2(radial @ y_axis, radial @ x_axis), 2 * pi)

    angular_edges = np.linspace(0, 2 * pi, angular_bins + 1)
    missing_angles = []
    for index in range(angular_bins):
        bin_mask = wall_mask & (angles >= angular_edges[index]) & (angles < angular_edges[index + 1])
        if np.any(bin_mask) and not np.any(bin_mask & valid):
            missing_angles.append((float(np.degrees(angular_edges[index])),
                                   float(np.degrees(angular_edges[index + 1]))))
    depth_edges = np.linspace(0, feature.depth_mm, depth_bins + 1)
    missing_depth = []
    for index in range(depth_bins):
        bin_mask = wall_mask & (axial >= depth_edges[index]) & (axial < depth_edges[index + 1])
        if np.any(bin_mask) and not np.any(bin_mask & valid):
            missing_depth.append((float(depth_edges[index]), float(depth_edges[index + 1])))
    visible_depth = float(np.max(axial[wall_mask & valid], initial=0.0))
    wall = _area_percent(feature.cylindrical_face_ids, model, coverage)
    rim = _area_percent(feature.conical_face_ids, model, coverage)
    bottom = _area_percent((feature.bottom_face_id,), model, coverage) if feature.bottom_face_id is not None else 100.0
    complete = wall >= target_percent and not missing_angles and visible_depth >= 0.9 * feature.depth_mm
    return HoleCoverageReport(
        feature.feature_id, rim, wall, bottom, visible_depth,
        tuple(missing_angles), tuple(missing_depth), complete,
    )
