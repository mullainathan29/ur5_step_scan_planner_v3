"""Tessellated-mesh coverage validation for exact B-Rep scan poses."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation

from cad_model import CadModel
from face_analysis import SurfaceType
from planners.base import Pose


class CoverageError(RuntimeError):
    """Raised when coverage validation cannot be completed."""


class TriangleStatus(IntEnum):
    UNCOVERED = 0
    VALID_COVERED = 1
    VISIBLE_INVALID = 2


@dataclass(frozen=True)
class CoverageSettings:
    """Physical laser-line and acceptance settings."""

    laser_width_mm: float = 200.0
    laser_ray_count: int = 21
    min_range_mm: float = 200.0
    max_range_mm: float = 420.0
    max_incidence_angle_deg: float = 60.0
    include_internal_surfaces: bool = False
    exterior_probe_count_per_face: int = 3

    def validate(self) -> None:
        if self.laser_width_mm <= 0.0:
            raise ValueError("laser_width_mm must be greater than zero.")
        if self.laser_ray_count < 3:
            raise ValueError("laser_ray_count must be at least 3.")
        if self.min_range_mm < 0.0:
            raise ValueError("min_range_mm must not be negative.")
        if self.max_range_mm <= self.min_range_mm:
            raise ValueError("max_range_mm must be greater than min_range_mm.")
        if not 0.0 <= self.max_incidence_angle_deg <= 90.0:
            raise ValueError(
                "max_incidence_angle_deg must be between 0 and 90 degrees."
            )
        if self.exterior_probe_count_per_face < 1:
            raise ValueError("exterior_probe_count_per_face must be at least 1.")


@dataclass(frozen=True)
class CoverageBreakdown:
    """Area-based coverage for one surface group."""

    eligible_area_mm2: float
    covered_area_mm2: float
    visible_invalid_area_mm2: float
    uncovered_area_mm2: float
    coverage_percent: float


@dataclass(frozen=True)
class CoverageResult:
    """Triangle masks and grouped area report from validation."""

    triangle_status: np.ndarray
    eligible_triangle_mask: np.ndarray
    exterior_face_ids: frozenset[int]
    internal_face_ids: frozenset[int]
    report: dict[str, CoverageBreakdown]
    scan_pose_count: int
    ray_count: int
    first_hit_count: int
    valid_hit_count: int
    invalid_range_hit_count: int
    invalid_incidence_hit_count: int
    valid_segment_ids: frozenset[int]
    invalid_segment_ids: frozenset[int]


def _first_hits(
    mesh,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )
    return (
        np.asarray(locations, dtype=np.float64),
        np.asarray(ray_indices, dtype=np.int64),
        np.asarray(triangle_indices, dtype=np.int64),
    )


def detect_exterior_face_ids(
    model: CadModel,
    *,
    probes_per_face: int = 3,
    cancellation_check=None,
) -> frozenset[int]:
    """Identify faces directly reachable from outside along exact OCC normals.

    A bore wall's outward normal points into its enclosed void. A ray started
    beyond the model along that normal is therefore blocked by an exterior face
    before reaching the bore wall, so the wall is classified as internal.
    """

    mesh = model.validation_mesh.mesh
    mapping = model.validation_mesh.triangle_face_ids
    triangle_areas = np.asarray(mesh.area_faces, dtype=np.float64)
    triangle_centers = np.asarray(mesh.triangles_center, dtype=np.float64)
    clearance = float(np.linalg.norm(mesh.extents)) * 2.0 + 1.0
    exterior: set[int] = set()
    cancellation_check = cancellation_check or (lambda: None)

    for record, topods_face in zip(model.faces, model.topods_faces):
        cancellation_check()
        triangle_ids = np.flatnonzero(mapping == record.face_id)
        if not len(triangle_ids):
            continue
        order = triangle_ids[np.argsort(triangle_areas[triangle_ids])[::-1]]
        selected = order[: min(probes_per_face, len(order))]
        face = cq.Face(topods_face)

        origins: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        for triangle_id in selected:
            center = triangle_centers[int(triangle_id)]
            try:
                normal_vector = face.normalAt(cq.Vector(*center))
                normal = np.array(
                    [normal_vector.x, normal_vector.y, normal_vector.z],
                    dtype=np.float64,
                )
            except Exception:
                normal = np.array(
                    [
                        record.representative_normal.x,
                        record.representative_normal.y,
                        record.representative_normal.z,
                    ],
                    dtype=np.float64,
                )
            normal /= np.linalg.norm(normal)
            origins.append(center + clearance * normal)
            directions.append(-normal)

        first = mesh.ray.intersects_first(
            np.asarray(origins), np.asarray(directions)
        )
        for triangle_id in np.asarray(first, dtype=np.int64):
            if triangle_id >= 0 and int(mapping[triangle_id]) == record.face_id:
                exterior.add(record.face_id)
                break
    return frozenset(exterior)


def _surface_group(surface_type: SurfaceType) -> str:
    if surface_type is SurfaceType.PLANE:
        return "planar_faces"
    if surface_type is SurfaceType.CYLINDER:
        return "cylindrical_faces"
    if surface_type is SurfaceType.CONE:
        return "conical_faces"
    return "freeform_faces"


def _breakdown(
    areas: np.ndarray,
    eligible: np.ndarray,
    valid: np.ndarray,
    invalid: np.ndarray,
) -> CoverageBreakdown:
    eligible_area = float(areas[eligible].sum())
    covered_area = float(areas[eligible & valid].sum())
    invalid_area = float(areas[eligible & invalid & ~valid].sum())
    uncovered_area = max(0.0, eligible_area - covered_area - invalid_area)
    coverage_percent = (
        100.0 * covered_area / eligible_area if eligible_area > 0.0 else 0.0
    )
    return CoverageBreakdown(
        eligible_area_mm2=eligible_area,
        covered_area_mm2=covered_area,
        visible_invalid_area_mm2=invalid_area,
        uncovered_area_mm2=uncovered_area,
        coverage_percent=coverage_percent,
    )


def validate_coverage(
    model: CadModel,
    poses: Iterable[Pose],
    settings: CoverageSettings | None = None,
    *,
    exterior_face_ids: frozenset[int] | None = None,
    cancellation_check=None,
    progress_callback=None,
) -> CoverageResult:
    """Cast pose-oriented laser rays and accumulate triangle-area coverage."""

    settings = settings or CoverageSettings()
    cancellation_check = cancellation_check or (lambda: None)
    progress_callback = progress_callback or (lambda *_: None)
    settings.validate()
    mesh = model.validation_mesh.mesh
    mapping = model.validation_mesh.triangle_face_ids
    triangle_count = len(mesh.faces)
    if len(mapping) != triangle_count:
        raise CoverageError("Triangle-to-STEP-face mapping is inconsistent.")

    if exterior_face_ids is None:
        exterior_face_ids = detect_exterior_face_ids(
            model, probes_per_face=settings.exterior_probe_count_per_face,
            cancellation_check=cancellation_check,
        )
    all_face_ids = frozenset(record.face_id for record in model.faces)
    internal_face_ids = all_face_ids - exterior_face_ids
    eligible_face_ids = (
        all_face_ids if settings.include_internal_surfaces else exterior_face_ids
    )
    eligible_mask = np.isin(mapping, list(eligible_face_ids))
    valid_mask = np.zeros(triangle_count, dtype=bool)
    invalid_mask = np.zeros(triangle_count, dtype=bool)

    scan_poses = [pose for pose in poses if pose.scan_enabled]
    offsets = np.linspace(
        -0.5 * settings.laser_width_mm,
        0.5 * settings.laser_width_mm,
        settings.laser_ray_count,
    )
    incidence_cosine = float(
        np.cos(np.deg2rad(settings.max_incidence_angle_deg))
    )

    total_rays = 0
    first_hit_count = 0
    valid_hit_count = 0
    invalid_range_count = 0
    invalid_incidence_count = 0
    segment_ids = {pose.segment_id for pose in scan_poses}
    valid_segment_ids: set[int] = set()

    for pose_index, pose in enumerate(scan_poses):
        cancellation_check()
        if pose_index % 25 == 0:
            progress_callback(pose_index, len(scan_poses))
        rotation = Rotation.from_quat(pose.quaternion_xyzw)
        laser_axis = rotation.apply([1.0, 0.0, 0.0])
        ray_direction = rotation.apply([0.0, 0.0, 1.0])
        origins = (
            pose.position[None, :]
            + offsets[:, None] * laser_axis[None, :]
        )
        directions = np.tile(ray_direction, (len(origins), 1))
        locations, ray_indices, triangle_indices = _first_hits(
            mesh, origins, directions
        )
        total_rays += len(origins)
        first_hit_count += len(triangle_indices)

        for location, ray_index, triangle_id in zip(
            locations, ray_indices, triangle_indices
        ):
            triangle_id = int(triangle_id)
            if not eligible_mask[triangle_id]:
                continue
            distance = float(
                np.linalg.norm(location - origins[int(ray_index)])
            )
            range_valid = settings.min_range_mm <= distance <= settings.max_range_mm
            triangle_normal = np.asarray(
                mesh.face_normals[triangle_id], dtype=np.float64
            )
            incidence = abs(float(np.dot(triangle_normal, -ray_direction)))
            incidence_valid = incidence >= incidence_cosine

            if range_valid and incidence_valid:
                valid_mask[triangle_id] = True
                valid_hit_count += 1
                valid_segment_ids.add(pose.segment_id)
            else:
                invalid_mask[triangle_id] = True
                if not range_valid:
                    invalid_range_count += 1
                if not incidence_valid:
                    invalid_incidence_count += 1

    status = np.full(
        triangle_count, TriangleStatus.UNCOVERED, dtype=np.int8
    )
    status[eligible_mask & invalid_mask] = TriangleStatus.VISIBLE_INVALID
    status[eligible_mask & valid_mask] = TriangleStatus.VALID_COVERED

    areas = np.asarray(mesh.area_faces, dtype=np.float64)
    reports: dict[str, CoverageBreakdown] = {}
    group_names = (
        "planar_faces",
        "cylindrical_faces",
        "conical_faces",
        "freeform_faces",
    )
    face_groups = {
        record.face_id: _surface_group(record.geometry_type)
        for record in model.faces
    }
    for group_name in group_names:
        group_face_ids = [
            face_id
            for face_id, name in face_groups.items()
            if name == group_name and face_id in eligible_face_ids
        ]
        group_mask = eligible_mask & np.isin(mapping, group_face_ids)
        reports[group_name] = _breakdown(
            areas, group_mask, valid_mask, invalid_mask
        )
    reports["total_exterior_surface"] = _breakdown(
        areas, eligible_mask, valid_mask, invalid_mask
    )

    status.setflags(write=False)
    eligible_mask.setflags(write=False)
    return CoverageResult(
        triangle_status=status,
        eligible_triangle_mask=eligible_mask,
        exterior_face_ids=exterior_face_ids,
        internal_face_ids=frozenset(internal_face_ids),
        report=reports,
        scan_pose_count=len(scan_poses),
        ray_count=total_rays,
        first_hit_count=first_hit_count,
        valid_hit_count=valid_hit_count,
        invalid_range_hit_count=invalid_range_count,
        invalid_incidence_hit_count=invalid_incidence_count,
        valid_segment_ids=frozenset(valid_segment_ids),
        invalid_segment_ids=frozenset(segment_ids - valid_segment_ids),
    )


def coverage_rgb(result: CoverageResult) -> np.ndarray:
    """Return per-triangle green/red/yellow display colors."""

    colors = np.tile(np.array([220, 40, 40], dtype=np.uint8), (len(result.triangle_status), 1))
    colors[result.triangle_status == TriangleStatus.VALID_COVERED] = [30, 180, 60]
    colors[result.triangle_status == TriangleStatus.VISIBLE_INVALID] = [245, 200, 35]
    return colors
