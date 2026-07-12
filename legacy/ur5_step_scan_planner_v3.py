#!/usr/bin/env python3
"""
UR5 + Micro-Epsilon scanCONTROL LLT30x0-200
STEP-Based View Scan Planner — Version 2
==========================================

Purpose
-------
Generate an ordered, robot-oriented scan trajectory from a STEP model.

Scan order:
    1. Top
    2. Front
    3. Right
    4. Back
    5. Left
    6. Top-front-right isometric
    7. Top-back-right isometric
    8. Top-back-left isometric
    9. Top-front-left isometric

The algorithm deliberately avoids connecting random surface samples.
Instead, each view is planned as one or more continuous scan lanes:

    approach -> lead-in -> scan poses -> lead-out -> retract

At every scan pose:
    scanner_position = surface_point + stand_off * outward_normal
    scanner_local_+Z = -outward_normal

The laser-line direction is kept tangent to the surface, and the robot travel
axis is constructed orthogonally to it.

Important
---------
This tool generates Cartesian poses in the CAD/STEP coordinate frame. It does
NOT yet:
    - transform the trajectory into the UR5 base frame,
    - solve inverse kinematics,
    - check UR5 self-collision or workcell collision,
    - validate cable routing,
    - execute the robot.

Those stages must be added before physical execution.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import cadquery as cq
import numpy as np
import pyvista as pv
import trimesh
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


EPS = 1e-10


@dataclass
class ViewDefinition:
    name: str
    outward: np.ndarray
    preferred_travel: np.ndarray
    category: str


@dataclass
class Pose:
    segment_id: int
    pose_id: int
    view_name: str
    motion_type: str
    scan_enabled: bool

    surface_x: float
    surface_y: float
    surface_z: float

    normal_x: float
    normal_y: float
    normal_z: float

    x: float
    y: float
    z: float

    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    # Preferred robot-orientation representations for interpolation.
    qx: float
    qy: float
    qz: float
    qw: float

    # Axis-angle / rotation-vector representation in radians.
    rx: float
    ry: float
    rz: float


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < EPS:
        return np.zeros(3, dtype=float)
    return v / n


def vector_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = normalize(a)
    b = normalize(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def cadquery_shape_from_step(path: str) -> cq.Shape:
    result = cq.importers.importStep(path)
    shape = result.val()
    if shape is None:
        raise ValueError("CadQuery did not return a valid shape.")
    return shape


def tessellate_cadquery_shape(
    shape: cq.Shape,
    linear_tolerance: float,
    angular_tolerance: float,
) -> tuple[trimesh.Trimesh, list[dict]]:
    """
    Tessellate STEP faces individually.

    STEP remains the authoritative model:
      - face identity is preserved,
      - face geometry type is recorded,
      - face area is recorded.

    The triangle mesh is used only for:
      - display,
      - visibility,
      - ray intersection,
      - retrieving local surface normals.
    """
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    triangle_to_step_face: list[int] = []
    face_metadata: list[dict] = []
    vertex_offset = 0

    faces = shape.Faces()
    if not faces:
        raise ValueError("The STEP model contains no faces.")

    for face_index, face in enumerate(faces):
        vertices, triangles = face.tessellate(
            tolerance=float(linear_tolerance),
            angularTolerance=float(angular_tolerance),
        )

        if not vertices or not triangles:
            continue

        verts_np = np.array([[v.x, v.y, v.z] for v in vertices], dtype=float)
        tris_np = np.asarray(triangles, dtype=np.int64)

        all_vertices.append(verts_np)
        all_faces.append(tris_np + vertex_offset)
        triangle_to_step_face.extend([face_index] * len(tris_np))
        vertex_offset += len(verts_np)

        try:
            geom_type = str(face.geomType())
        except Exception:
            geom_type = "UNKNOWN"

        try:
            area = float(face.Area())
        except Exception:
            area = float("nan")

        face_metadata.append(
            {
                "face_index": face_index,
                "geometry_type": geom_type,
                "area_mm2": area,
            }
        )

    if not all_faces:
        raise ValueError("STEP tessellation produced no triangles.")

    vertices = np.vstack(all_vertices)
    triangles = np.vstack(all_faces)

    mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=False)
    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_unreferenced_vertices()

    try:
        trimesh.repair.fix_normals(mesh, multibody=True)
    except Exception:
        pass

    # The merge/update operations can alter triangle indexing. Rebuild a safe
    # triangle-face map only when lengths still agree; otherwise mark unknown.
    if len(triangle_to_step_face) != len(mesh.faces):
        triangle_to_step_face = [-1] * len(mesh.faces)

    mesh.metadata["triangle_to_step_face"] = np.asarray(
        triangle_to_step_face, dtype=np.int64
    )
    return mesh, face_metadata


def load_model(
    path: str,
    linear_tolerance: float,
    angular_tolerance: float,
) -> tuple[trimesh.Trimesh, list[dict], str]:
    suffix = Path(path).suffix.lower()

    if suffix in {".step", ".stp"}:
        shape = cadquery_shape_from_step(path)
        mesh, metadata = tessellate_cadquery_shape(
            shape,
            linear_tolerance=linear_tolerance,
            angular_tolerance=angular_tolerance,
        )
        return mesh, metadata, "STEP"

    if suffix == ".stl":
        loaded = trimesh.load(path, force="mesh", process=False)
        if isinstance(loaded, trimesh.Scene):
            loaded = trimesh.util.concatenate(tuple(loaded.geometry.values()))
        if not isinstance(loaded, trimesh.Trimesh):
            raise TypeError("The STL could not be interpreted as a triangle mesh.")

        mesh = loaded.copy()
        mesh.merge_vertices()
        mesh.update_faces(mesh.unique_faces())
        mesh.update_faces(mesh.nondegenerate_faces())
        mesh.remove_unreferenced_vertices()
        try:
            trimesh.repair.fix_normals(mesh, multibody=True)
        except Exception:
            pass
        return mesh, [], "STL"

    raise ValueError("Supported inputs are STEP (.step/.stp) and STL (.stl).")


def trimesh_to_pyvista(mesh: trimesh.Trimesh) -> pv.PolyData:
    faces = np.hstack(
        [
            np.full((len(mesh.faces), 1), 3, dtype=np.int64),
            np.asarray(mesh.faces, dtype=np.int64),
        ]
    ).ravel()
    return pv.PolyData(np.asarray(mesh.vertices, dtype=float), faces)


def default_views(include_isometric: bool = True) -> list[ViewDefinition]:
    z = np.array([0.0, 0.0, 1.0])

    views = [
        ViewDefinition(
            "top",
            normalize([0.0, 0.0, 1.0]),
            normalize([1.0, 0.0, 0.0]),
            "principal",
        ),
        ViewDefinition(
            "front",
            normalize([0.0, -1.0, 0.0]),
            normalize([1.0, 0.0, 0.0]),
            "principal",
        ),
        ViewDefinition(
            "right",
            normalize([1.0, 0.0, 0.0]),
            normalize([0.0, 1.0, 0.0]),
            "principal",
        ),
        ViewDefinition(
            "back",
            normalize([0.0, 1.0, 0.0]),
            normalize([-1.0, 0.0, 0.0]),
            "principal",
        ),
        ViewDefinition(
            "left",
            normalize([-1.0, 0.0, 0.0]),
            normalize([0.0, -1.0, 0.0]),
            "principal",
        ),
    ]

    if include_isometric:
        elevation = math.radians(35.2643897)
        horizontal = math.cos(elevation)
        vertical = math.sin(elevation)

        iso_specs = [
            ("iso_top_front_right", [horizontal, -horizontal, vertical]),
            ("iso_top_back_right", [horizontal, horizontal, vertical]),
            ("iso_top_back_left", [-horizontal, horizontal, vertical]),
            ("iso_top_front_left", [-horizontal, -horizontal, vertical]),
        ]

        for name, direction in iso_specs:
            outward = normalize(direction)
            travel = np.cross(z, outward)
            if np.linalg.norm(travel) < EPS:
                travel = np.array([1.0, 0.0, 0.0])
            views.append(
                ViewDefinition(
                    name,
                    outward,
                    normalize(travel),
                    "isometric",
                )
            )

    return views


def build_view_frame(
    view: ViewDefinition,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        outward: from part toward scanner
        travel: nominal robot movement direction
        laser: nominal laser-line direction

    The laser axis is orthogonal to both outward and travel.
    """
    outward = normalize(view.outward)

    travel = view.preferred_travel - np.dot(view.preferred_travel, outward) * outward
    if np.linalg.norm(travel) < EPS:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(reference, outward)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        travel = reference - np.dot(reference, outward) * outward
    travel = normalize(travel)

    laser = normalize(np.cross(outward, travel))
    travel = normalize(np.cross(laser, outward))
    return outward, travel, laser


def view_projection_extents(
    mesh: trimesh.Trimesh,
    outward: np.ndarray,
    travel: np.ndarray,
    laser: np.ndarray,
) -> dict:
    vertices = np.asarray(mesh.vertices, dtype=float)
    return {
        "outward_min": float((vertices @ outward).min()),
        "outward_max": float((vertices @ outward).max()),
        "travel_min": float((vertices @ travel).min()),
        "travel_max": float((vertices @ travel).max()),
        "laser_min": float((vertices @ laser).min()),
        "laser_max": float((vertices @ laser).max()),
    }


def ray_first_hits(
    mesh: trimesh.Trimesh,
    origins: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return first intersection for every ray where a hit exists.

    Returns:
        locations
        ray_indices
        triangle_indices
    """
    locations, ray_indices, triangle_indices = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=False,
    )

    return (
        np.asarray(locations, dtype=float),
        np.asarray(ray_indices, dtype=np.int64),
        np.asarray(triangle_indices, dtype=np.int64),
    )



def fixed_view_rotation(
    outward: np.ndarray,
    laser_axis: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float, float], tuple[float, float, float, float], tuple[float, float, float]]:
    """
    Fixed scanner frame for one view.

    local +Z points toward the part, local +X follows the laser line,
    and local +Y follows the nominal travel direction.
    """
    z_axis = -normalize(outward)
    x_axis = laser_axis - np.dot(laser_axis, z_axis) * z_axis
    if np.linalg.norm(x_axis) < EPS:
        reference = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(reference, z_axis)) > 0.9:
            reference = np.array([0.0, 1.0, 0.0])
        x_axis = reference - np.dot(reference, z_axis) * z_axis
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    matrix = np.column_stack((x_axis, y_axis, z_axis))
    rot = Rotation.from_matrix(matrix)
    rpy = tuple(float(v) for v in rot.as_euler("xyz", degrees=True))
    quat = tuple(float(v) for v in rot.as_quat())  # x, y, z, w
    rotvec = tuple(float(v) for v in rot.as_rotvec())
    return matrix, rpy, quat, rotvec


def split_fixed_view_samples(
    samples: list[dict],
    max_scanner_step_mm: float,
    max_normal_change_deg: float,
) -> list[list[dict]]:
    """
    Split whenever the scanner-space path or the surface geometry changes too
    abruptly. This prevents a scan lane from jumping from a top face to a side
    face while remaining in the same robot segment.
    """
    if not samples:
        return []

    segments: list[list[dict]] = [[samples[0]]]
    for sample in samples[1:]:
        previous = segments[-1][-1]
        scanner_step = float(
            np.linalg.norm(sample["scanner_point"] - previous["scanner_point"])
        )
        normal_change = vector_angle_deg(sample["normal"], previous["normal"])

        if (
            scanner_step > max_scanner_step_mm
            or normal_change > max_normal_change_deg
        ):
            segments.append([sample])
        else:
            segments[-1].append(sample)

    return segments


def make_pose(
    segment_id: int,
    pose_id: int,
    view_name: str,
    motion_type: str,
    scan_enabled: bool,
    surface: np.ndarray,
    normal: np.ndarray,
    scanner: np.ndarray,
    rpy: tuple[float, float, float],
    quat: tuple[float, float, float, float],
    rotvec: tuple[float, float, float],
) -> Pose:
    return Pose(
        segment_id=segment_id,
        pose_id=pose_id,
        view_name=view_name,
        motion_type=motion_type,
        scan_enabled=scan_enabled,
        surface_x=float(surface[0]),
        surface_y=float(surface[1]),
        surface_z=float(surface[2]),
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
        normal_z=float(normal[2]),
        x=float(scanner[0]),
        y=float(scanner[1]),
        z=float(scanner[2]),
        roll_deg=float(rpy[0]),
        pitch_deg=float(rpy[1]),
        yaw_deg=float(rpy[2]),
        qx=float(quat[0]),
        qy=float(quat[1]),
        qz=float(quat[2]),
        qw=float(quat[3]),
        rx=float(rotvec[0]),
        ry=float(rotvec[1]),
        rz=float(rotvec[2]),
    )


def create_fixed_view_motion_segment(
    segment_id: int,
    view_name: str,
    scan_samples: list[dict],
    outward: np.ndarray,
    laser_axis: np.ndarray,
    approach_clearance_mm: float,
    lead_distance_mm: float,
) -> list[Pose]:
    if len(scan_samples) < 2:
        return []

    _, rpy, quat, rotvec = fixed_view_rotation(outward, laser_axis)

    scanner_points = np.asarray(
        [sample["scanner_point"] for sample in scan_samples], dtype=float
    )
    start_tangent = normalize(scanner_points[1] - scanner_points[0])
    end_tangent = normalize(scanner_points[-1] - scanner_points[-2])

    if np.linalg.norm(start_tangent) < EPS:
        start_tangent = normalize(np.cross(laser_axis, outward))
    if np.linalg.norm(end_tangent) < EPS:
        end_tangent = start_tangent

    start = scan_samples[0]
    end = scan_samples[-1]

    lead_in = start["scanner_point"] - lead_distance_mm * start_tangent
    lead_out = end["scanner_point"] + lead_distance_mm * end_tangent

    # Retract in the selected view direction, not in the local triangle normal.
    approach = lead_in + approach_clearance_mm * outward
    retract = lead_out + approach_clearance_mm * outward

    poses: list[Pose] = []
    pose_id = 0

    poses.append(
        make_pose(
            segment_id, pose_id, view_name, "approach", False,
            start["surface_point"], start["normal"], approach, rpy, quat, rotvec
        )
    )
    pose_id += 1
    poses.append(
        make_pose(
            segment_id, pose_id, view_name, "lead_in", False,
            start["surface_point"], start["normal"], lead_in, rpy, quat, rotvec
        )
    )
    pose_id += 1

    for sample in scan_samples:
        poses.append(
            make_pose(
                segment_id, pose_id, view_name, "scan", True,
                sample["surface_point"], sample["normal"],
                sample["scanner_point"], rpy, quat, rotvec
            )
        )
        pose_id += 1

    poses.append(
        make_pose(
            segment_id, pose_id, view_name, "lead_out", False,
            end["surface_point"], end["normal"], lead_out, rpy, quat, rotvec
        )
    )
    pose_id += 1
    poses.append(
        make_pose(
            segment_id, pose_id, view_name, "retract", False,
            end["surface_point"], end["normal"], retract, rpy, quat, rotvec
        )
    )
    return poses


def simulate_laser_line_coverage(
    mesh: trimesh.Trimesh,
    scanner_point: np.ndarray,
    outward: np.ndarray,
    laser_axis: np.ndarray,
    laser_width_mm: float,
    laser_ray_count: int,
    min_range_mm: float,
    max_range_mm: float,
    max_incidence_angle_deg: float,
) -> tuple[set[int], int]:
    """
    Approximate the LLT laser line by a fan of parallel rays distributed along
    scanner local +X. This is a conservative geometric coverage model at the
    selected nominal working distance.

    It is not an optical calibration model, but it verifies:
      - first-surface visibility,
      - sensor range,
      - local incidence angle,
      - coverage across the usable laser line.
    """
    ray_count = max(3, int(laser_ray_count))
    offsets = np.linspace(-laser_width_mm * 0.5, laser_width_mm * 0.5, ray_count)
    origins = scanner_point[None, :] + offsets[:, None] * laser_axis[None, :]
    directions = np.tile(-normalize(outward), (ray_count, 1))

    locations, ray_indices, triangle_indices = ray_first_hits(
        mesh, origins, directions
    )

    valid_faces: set[int] = set()
    valid_rays = 0
    incidence_cos = math.cos(math.radians(max_incidence_angle_deg))

    for i, ray_idx in enumerate(ray_indices):
        hit = locations[i]
        triangle_idx = int(triangle_indices[i])
        distance = float(np.linalg.norm(hit - origins[int(ray_idx)]))
        normal = normalize(mesh.face_normals[triangle_idx])

        if not (min_range_mm <= distance <= max_range_mm):
            continue
        if float(np.dot(normal, outward)) < incidence_cos:
            continue

        valid_faces.add(triangle_idx)
        valid_rays += 1

    return valid_faces, valid_rays


def plan_view_fixed(
    mesh: trimesh.Trimesh,
    view: ViewDefinition,
    stand_off_mm: float,
    laser_width_mm: float,
    overlap_fraction: float,
    travel_sample_spacing_mm: float,
    max_incidence_angle_deg: float,
    approach_clearance_mm: float,
    lead_distance_mm: float,
    minimum_segment_points: int,
    max_scanner_step_mm: float,
    max_normal_change_deg: float,
    laser_ray_count: int,
    min_sensor_range_mm: float,
    max_sensor_range_mm: float,
    minimum_valid_laser_fraction: float,
    segment_id_start: int,
) -> tuple[list[Pose], int, set[int], dict]:
    outward, travel, laser = build_view_frame(view)
    ext = view_projection_extents(mesh, outward, travel, laser)

    lane_spacing = max(
        laser_width_mm * (1.0 - overlap_fraction),
        travel_sample_spacing_mm,
    )

    travel_values = np.arange(
        ext["travel_min"],
        ext["travel_max"] + travel_sample_spacing_mm,
        travel_sample_spacing_mm,
    )

    laser_span = ext["laser_max"] - ext["laser_min"]
    if laser_span <= laser_width_mm:
        lane_values = np.array(
            [(ext["laser_min"] + ext["laser_max"]) * 0.5], dtype=float
        )
    else:
        first = ext["laser_min"] + laser_width_mm * 0.5
        last = ext["laser_max"] - laser_width_mm * 0.5
        lane_values = np.arange(first, last + lane_spacing * 0.5, lane_spacing)
        if len(lane_values) == 0:
            lane_values = np.array([(first + last) * 0.5], dtype=float)

    ray_clearance = max(mesh.extents) + stand_off_mm + approach_clearance_mm + 100.0
    origin_outward_coord = ext["outward_max"] + ray_clearance
    incidence_cos = math.cos(math.radians(max_incidence_angle_deg))

    output: list[Pose] = []
    covered_faces: set[int] = set()
    segment_id = segment_id_start
    accepted_centres = 0
    rejected_centres = 0

    for lane_index, lane_value in enumerate(lane_values):
        ordered_travel = travel_values if lane_index % 2 == 0 else travel_values[::-1]

        origins = (
            np.outer(np.full(len(ordered_travel), origin_outward_coord), outward)
            + np.outer(ordered_travel, travel)
            + np.outer(np.full(len(ordered_travel), lane_value), laser)
        )
        directions = np.tile(-outward, (len(origins), 1))
        locations, ray_indices, triangle_indices = ray_first_hits(
            mesh, origins, directions
        )
        hit_map = {
            int(ray_idx): (locations[i], int(triangle_indices[i]))
            for i, ray_idx in enumerate(ray_indices)
        }

        samples: list[dict] = []
        for ray_idx in range(len(origins)):
            if ray_idx not in hit_map:
                rejected_centres += 1
                continue

            surface_point, triangle_index = hit_map[ray_idx]
            normal = normalize(mesh.face_normals[triangle_index])

            if float(np.dot(normal, outward)) < incidence_cos:
                rejected_centres += 1
                continue

            # Fixed-view scanner pose. This is the key correction: scanner
            # orientation and stand-off direction remain associated with the
            # selected top/side/isometric view.
            scanner_point = surface_point + stand_off_mm * outward

            line_faces, valid_rays = simulate_laser_line_coverage(
                mesh=mesh,
                scanner_point=scanner_point,
                outward=outward,
                laser_axis=laser,
                laser_width_mm=laser_width_mm,
                laser_ray_count=laser_ray_count,
                min_range_mm=min_sensor_range_mm,
                max_range_mm=max_sensor_range_mm,
                max_incidence_angle_deg=max_incidence_angle_deg,
            )
            valid_fraction = valid_rays / max(1, laser_ray_count)

            if valid_fraction < minimum_valid_laser_fraction:
                rejected_centres += 1
                continue

            samples.append(
                {
                    "surface_point": surface_point,
                    "normal": normal,
                    "scanner_point": scanner_point,
                    "triangle_index": triangle_index,
                    "covered_faces": line_faces,
                }
            )
            accepted_centres += 1

        split_segments = split_fixed_view_samples(
            samples,
            max_scanner_step_mm=max_scanner_step_mm,
            max_normal_change_deg=max_normal_change_deg,
        )

        for segment in split_segments:
            if len(segment) < minimum_segment_points:
                continue

            segment_poses = create_fixed_view_motion_segment(
                segment_id=segment_id,
                view_name=view.name,
                scan_samples=segment,
                outward=outward,
                laser_axis=laser,
                approach_clearance_mm=approach_clearance_mm,
                lead_distance_mm=lead_distance_mm,
            )
            if not segment_poses:
                continue

            output.extend(segment_poses)
            for sample in segment:
                covered_faces.update(sample["covered_faces"])
            segment_id += 1

    stats = {
        "view_name": view.name,
        "accepted_scan_centres": accepted_centres,
        "rejected_scan_centres": rejected_centres,
        "segments": segment_id - segment_id_start,
        "covered_triangles": len(covered_faces),
    }
    return output, segment_id, covered_faces, stats


def validate_generated_segments(
    poses: list[Pose],
    max_scanner_step_mm: float,
    max_orientation_step_deg: float,
) -> list[str]:
    warnings: list[str] = []
    grouped: dict[int, list[Pose]] = {}
    for pose in poses:
        if pose.motion_type == "scan":
            grouped.setdefault(pose.segment_id, []).append(pose)

    for segment_id, segment in grouped.items():
        for a, b in zip(segment, segment[1:]):
            pa = np.array([a.x, a.y, a.z], dtype=float)
            pb = np.array([b.x, b.y, b.z], dtype=float)
            position_step = float(np.linalg.norm(pb - pa))

            qa = Rotation.from_quat([a.qx, a.qy, a.qz, a.qw])
            qb = Rotation.from_quat([b.qx, b.qy, b.qz, b.qw])
            orientation_step = math.degrees((qa.inv() * qb).magnitude())

            if position_step > max_scanner_step_mm + 1e-6:
                warnings.append(
                    f"Segment {segment_id}: scanner step {position_step:.2f} mm"
                )
            if orientation_step > max_orientation_step_deg + 1e-6:
                warnings.append(
                    f"Segment {segment_id}: orientation step "
                    f"{orientation_step:.2f} deg"
                )
    return warnings


def generate_scan_plan(
    mesh: trimesh.Trimesh,
    stand_off_mm: float,
    laser_width_mm: float,
    overlap_percent: float,
    travel_sample_spacing_mm: float,
    max_incidence_angle_deg: float,
    approach_clearance_mm: float,
    lead_distance_mm: float,
    minimum_segment_points: int,
    include_isometric: bool,
    max_scanner_step_mm: float,
    max_normal_change_deg: float,
    max_orientation_step_deg: float,
    laser_ray_count: int,
    min_sensor_range_mm: float,
    max_sensor_range_mm: float,
    minimum_valid_laser_fraction: float,
) -> tuple[list[Pose], np.ndarray, dict]:
    poses: list[Pose] = []
    covered_faces: set[int] = set()
    view_stats: list[dict] = []
    segment_id = 0

    for view in default_views(include_isometric=include_isometric):
        view_poses, segment_id, view_covered, stats = plan_view_fixed(
            mesh=mesh,
            view=view,
            stand_off_mm=stand_off_mm,
            laser_width_mm=laser_width_mm,
            overlap_fraction=overlap_percent / 100.0,
            travel_sample_spacing_mm=travel_sample_spacing_mm,
            max_incidence_angle_deg=max_incidence_angle_deg,
            approach_clearance_mm=approach_clearance_mm,
            lead_distance_mm=lead_distance_mm,
            minimum_segment_points=minimum_segment_points,
            max_scanner_step_mm=max_scanner_step_mm,
            max_normal_change_deg=max_normal_change_deg,
            laser_ray_count=laser_ray_count,
            min_sensor_range_mm=min_sensor_range_mm,
            max_sensor_range_mm=max_sensor_range_mm,
            minimum_valid_laser_fraction=minimum_valid_laser_fraction,
            segment_id_start=segment_id,
        )
        poses.extend(view_poses)
        covered_faces.update(view_covered)
        view_stats.append(stats)

    coverage_mask = np.zeros(len(mesh.faces), dtype=bool)
    if covered_faces:
        indices = np.fromiter(covered_faces, dtype=np.int64)
        indices = indices[(indices >= 0) & (indices < len(mesh.faces))]
        coverage_mask[indices] = True

    warnings = validate_generated_segments(
        poses,
        max_scanner_step_mm=max_scanner_step_mm,
        max_orientation_step_deg=max_orientation_step_deg,
    )

    total_area = float(mesh.area)
    covered_area = float(mesh.area_faces[coverage_mask].sum()) if coverage_mask.any() else 0.0
    coverage_percent = 100.0 * covered_area / total_area if total_area > EPS else 0.0

    stats = {
        "coverage_percent_by_triangle_area": coverage_percent,
        "covered_triangle_count": int(coverage_mask.sum()),
        "triangle_count": int(len(mesh.faces)),
        "segment_count": len({p.segment_id for p in poses}),
        "scan_pose_count": sum(p.motion_type == "scan" for p in poses),
        "validation_warnings": warnings,
        "views": view_stats,
    }
    return poses, coverage_mask, stats




class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()

        self.setWindowTitle(
            "UR5 STEP Scan Planner V3 — LLT30x0-200"
        )
        self.resize(1550, 920)

        self.mesh: Optional[trimesh.Trimesh] = None
        self.model_path: Optional[str] = None
        self.model_type: Optional[str] = None
        self.face_metadata: list[dict] = []
        self.poses: list[Pose] = []
        self.coverage_mask: Optional[np.ndarray] = None
        self.plan_stats: dict = {}

        self.plotter = QtInteractor(self)
        self._build_ui()

    def _spin(
        self,
        minimum: float,
        maximum: float,
        value: float,
        suffix: str,
        decimals: int = 2,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setDecimals(decimals)
        return widget

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QHBoxLayout(root)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        controls = QWidget()
        controls.setMinimumWidth(350)
        controls.setMaximumWidth(450)
        controls_layout = QVBoxLayout(controls)

        file_group = QGroupBox("CAD model")
        file_layout = QVBoxLayout(file_group)
        self.file_label = QLabel("No model loaded")
        self.file_label.setWordWrap(True)
        load_button = QPushButton("Upload STEP / STL")
        load_button.clicked.connect(self.load_model_dialog)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(load_button)

        mesh_group = QGroupBox("Model information")
        mesh_layout = QFormLayout(mesh_group)
        self.type_label = QLabel("-")
        self.vertices_label = QLabel("-")
        self.triangles_label = QLabel("-")
        self.step_faces_label = QLabel("-")
        self.size_label = QLabel("-")
        mesh_layout.addRow("Input type:", self.type_label)
        mesh_layout.addRow("Display vertices:", self.vertices_label)
        mesh_layout.addRow("Display triangles:", self.triangles_label)
        mesh_layout.addRow("STEP faces:", self.step_faces_label)
        mesh_layout.addRow("Size XYZ:", self.size_label)

        tess_group = QGroupBox("STEP tessellation")
        tess_layout = QFormLayout(tess_group)
        self.linear_tolerance_spin = self._spin(0.01, 10.0, 0.25, " mm", 3)
        self.angular_tolerance_spin = self._spin(0.01, 1.0, 0.1, " rad", 3)
        tess_layout.addRow("Linear tolerance:", self.linear_tolerance_spin)
        tess_layout.addRow("Angular tolerance:", self.angular_tolerance_spin)

        scanner_group = QGroupBox("LLT30x0-200 scanner")
        scanner_layout = QFormLayout(scanner_group)
        self.standoff_spin = self._spin(200.0, 420.0, 310.0, " mm")
        self.laser_width_spin = self._spin(1.0, 300.0, 200.0, " mm")
        self.overlap_spin = self._spin(0.0, 90.0, 20.0, " %", 1)
        self.travel_spacing_spin = self._spin(0.25, 50.0, 5.0, " mm")
        self.min_range_spin = self._spin(100.0, 500.0, 200.0, " mm")
        self.max_range_spin = self._spin(100.0, 600.0, 420.0, " mm")
        self.laser_ray_count_spin = QSpinBox()
        self.laser_ray_count_spin.setRange(3, 101)
        self.laser_ray_count_spin.setSingleStep(2)
        self.laser_ray_count_spin.setValue(21)

        scanner_layout.addRow("Stand-off distance:", self.standoff_spin)
        scanner_layout.addRow("Laser-line width:", self.laser_width_spin)
        scanner_layout.addRow("Lane overlap:", self.overlap_spin)
        scanner_layout.addRow("Travel pose spacing:", self.travel_spacing_spin)
        scanner_layout.addRow("Minimum sensor range:", self.min_range_spin)
        scanner_layout.addRow("Maximum sensor range:", self.max_range_spin)
        scanner_layout.addRow("Laser coverage rays:", self.laser_ray_count_spin)

        planning_group = QGroupBox("Planning")
        planning_layout = QFormLayout(planning_group)

        self.normal_alignment_spin = self._spin(5.0, 89.0, 60.0, "°", 1)
        self.approach_clearance_spin = self._spin(10.0, 500.0, 120.0, " mm")
        self.lead_distance_spin = self._spin(0.0, 200.0, 20.0, " mm")
        self.max_scanner_step_spin = self._spin(1.0, 100.0, 12.0, " mm")
        self.max_normal_change_spin = self._spin(1.0, 90.0, 15.0, "°", 1)
        self.max_orientation_step_spin = self._spin(0.1, 45.0, 5.0, "°", 1)
        self.minimum_valid_laser_fraction_spin = self._spin(
            0.0, 1.0, 0.20, "", 2
        )

        self.minimum_segment_points_spin = QSpinBox()
        self.minimum_segment_points_spin.setRange(2, 100)
        self.minimum_segment_points_spin.setValue(5)

        self.include_isometric_check = QCheckBox()
        self.include_isometric_check.setChecked(True)

        planning_layout.addRow(
            "Maximum incidence angle:",
            self.normal_alignment_spin,
        )
        planning_layout.addRow(
            "Approach/retract clearance:",
            self.approach_clearance_spin,
        )
        planning_layout.addRow("Lead-in/out distance:", self.lead_distance_spin)
        planning_layout.addRow(
            "Maximum scanner step:",
            self.max_scanner_step_spin,
        )
        planning_layout.addRow(
            "Maximum normal change:",
            self.max_normal_change_spin,
        )
        planning_layout.addRow(
            "Maximum orientation step:",
            self.max_orientation_step_spin,
        )
        planning_layout.addRow(
            "Minimum valid laser fraction:",
            self.minimum_valid_laser_fraction_spin,
        )
        planning_layout.addRow(
            "Minimum scan points:",
            self.minimum_segment_points_spin,
        )
        planning_layout.addRow("Include isometric views:", self.include_isometric_check)

        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.generate_button = QPushButton("Generate Scan Plan")
        self.generate_button.setEnabled(False)
        self.generate_button.clicked.connect(self.generate_plan)

        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.setEnabled(False)
        self.export_csv_button.clicked.connect(self.export_csv)

        self.export_json_button = QPushButton("Export JSON")
        self.export_json_button.setEnabled(False)
        self.export_json_button.clicked.connect(self.export_json)

        clear_button = QPushButton("Clear Trajectory")
        clear_button.clicked.connect(self.clear_trajectory)

        actions_layout.addWidget(self.generate_button)
        actions_layout.addWidget(self.export_csv_button)
        actions_layout.addWidget(self.export_json_button)
        actions_layout.addWidget(clear_button)

        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)

        note = QLabel(
            "Frame: STEP/CAD coordinates\n"
            "Scanner +Z: toward surface\n"
            "Scanner +X: laser line\n"
            "Scanner +Y: travel direction\n"
            "Fixed orientation within each scan segment\n"
            "Green mesh: covered; red mesh: not covered"
        )
        note.setWordWrap(True)

        controls_layout.addWidget(file_group)
        controls_layout.addWidget(mesh_group)
        controls_layout.addWidget(tess_group)
        controls_layout.addWidget(scanner_group)
        controls_layout.addWidget(planning_group)
        controls_layout.addWidget(actions_group)
        controls_layout.addWidget(note)
        controls_layout.addStretch()
        controls_layout.addWidget(self.status_label)

        splitter.addWidget(controls)
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(root)

        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.show_grid()

    def load_model_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select STEP or STL model",
            "",
            "CAD files (*.step *.stp *.stl);;STEP files (*.step *.stp);;"
            "STL files (*.stl);;All files (*.*)",
        )
        if not path:
            return

        try:
            self.status_label.setText("Loading and tessellating CAD model...")
            QApplication.processEvents()

            mesh, face_metadata, model_type = load_model(
                path,
                linear_tolerance=self.linear_tolerance_spin.value(),
                angular_tolerance=self.angular_tolerance_spin.value(),
            )

            self.mesh = mesh
            self.model_path = path
            self.model_type = model_type
            self.face_metadata = face_metadata
            self.poses = []
            self.coverage_mask = None
            self.plan_stats = {}

            self.file_label.setText(Path(path).name)
            self.type_label.setText(model_type)
            self.vertices_label.setText(f"{len(mesh.vertices):,}")
            self.triangles_label.setText(f"{len(mesh.faces):,}")
            self.step_faces_label.setText(
                f"{len(face_metadata):,}" if model_type == "STEP" else "N/A"
            )

            size = mesh.extents
            self.size_label.setText(
                f"{size[0]:.2f} × {size[1]:.2f} × {size[2]:.2f} mm"
            )

            self._show_mesh()
            self.generate_button.setEnabled(True)
            self.export_csv_button.setEnabled(False)
            self.export_json_button.setEnabled(False)

            self.status_label.setText(
                f"{model_type} loaded. Ready to generate scan plan."
            )

        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "CAD loading error", str(exc))
            self.status_label.setText("Model loading failed.")

    def _show_mesh(self) -> None:
        if self.mesh is None:
            return

        self.plotter.clear()
        self.plotter.add_mesh(
            trimesh_to_pyvista(self.mesh),
            color="lightgray",
            show_edges=True,
            opacity=1.0,
        )
        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

    def generate_plan(self) -> None:
        if self.mesh is None:
            return

        try:
            if self.min_range_spin.value() >= self.max_range_spin.value():
                raise ValueError(
                    "Minimum sensor range must be smaller than maximum sensor range."
                )

            self.generate_button.setEnabled(False)
            self.status_label.setText(
                "Generating fixed-view paths and simulating laser coverage..."
            )
            QApplication.processEvents()

            (
                self.poses,
                self.coverage_mask,
                self.plan_stats,
            ) = generate_scan_plan(
                mesh=self.mesh,
                stand_off_mm=self.standoff_spin.value(),
                laser_width_mm=self.laser_width_spin.value(),
                overlap_percent=self.overlap_spin.value(),
                travel_sample_spacing_mm=self.travel_spacing_spin.value(),
                max_incidence_angle_deg=self.normal_alignment_spin.value(),
                approach_clearance_mm=self.approach_clearance_spin.value(),
                lead_distance_mm=self.lead_distance_spin.value(),
                minimum_segment_points=self.minimum_segment_points_spin.value(),
                include_isometric=self.include_isometric_check.isChecked(),
                max_scanner_step_mm=self.max_scanner_step_spin.value(),
                max_normal_change_deg=self.max_normal_change_spin.value(),
                max_orientation_step_deg=self.max_orientation_step_spin.value(),
                laser_ray_count=self.laser_ray_count_spin.value(),
                min_sensor_range_mm=self.min_range_spin.value(),
                max_sensor_range_mm=self.max_range_spin.value(),
                minimum_valid_laser_fraction=(
                    self.minimum_valid_laser_fraction_spin.value()
                ),
            )

            if not self.poses:
                raise RuntimeError(
                    "No valid scan trajectory was generated. Reduce the minimum "
                    "valid laser fraction, increase the incidence-angle limit, "
                    "or reduce the minimum segment points."
                )

            self._show_plan()

            warnings = self.plan_stats.get("validation_warnings", [])
            coverage = self.plan_stats.get(
                "coverage_percent_by_triangle_area", 0.0
            )
            scan_count = self.plan_stats.get("scan_pose_count", 0)
            segment_count = self.plan_stats.get("segment_count", 0)
            view_count = len({p.view_name for p in self.poses})

            warning_text = (
                f" Validation warnings: {len(warnings)}."
                if warnings
                else " Segment continuity validation passed."
            )
            self.status_label.setText(
                f"Generated {scan_count:,} scan poses, "
                f"{segment_count:,} segments, {view_count} views. "
                f"Estimated covered triangle area: {coverage:.1f}%."
                + warning_text
            )
            self.export_csv_button.setEnabled(True)
            self.export_json_button.setEnabled(True)

        except Exception as exc:
            traceback.print_exc()
            QMessageBox.critical(self, "Planning error", str(exc))
            self.status_label.setText("Scan-plan generation failed.")
        finally:
            self.generate_button.setEnabled(True)

    def _show_plan(self) -> None:
        assert self.mesh is not None

        self.plotter.clear()
        poly = trimesh_to_pyvista(self.mesh)

        if (
            self.coverage_mask is not None
            and len(self.coverage_mask) == len(self.mesh.faces)
        ):
            coverage_values = self.coverage_mask.astype(np.int8)
            poly.cell_data["coverage"] = coverage_values
            self.plotter.add_mesh(
                poly,
                scalars="coverage",
                categories=True,
                clim=[0, 1],
                cmap=["red", "green"],
                show_edges=False,
                opacity=0.60,
                scalar_bar_args={"title": "Coverage 0=missed 1=covered"},
            )
        else:
            self.plotter.add_mesh(
                poly,
                color="lightgray",
                show_edges=False,
                opacity=0.55,
            )

        view_names = list(dict.fromkeys(p.view_name for p in self.poses))
        palette = [
            "blue",
            "orange",
            "purple",
            "cyan",
            "magenta",
            "brown",
            "black",
            "navy",
            "darkgreen",
        ]
        view_colors = {
            name: palette[i % len(palette)]
            for i, name in enumerate(view_names)
        }

        segment_ids = sorted({p.segment_id for p in self.poses})
        for segment_id in segment_ids:
            segment = [p for p in self.poses if p.segment_id == segment_id]
            scan_points = np.array(
                [[p.x, p.y, p.z] for p in segment if p.motion_type == "scan"],
                dtype=float,
            )
            transition_points = np.array(
                [[p.x, p.y, p.z] for p in segment],
                dtype=float,
            )
            color = view_colors[segment[0].view_name]

            if len(scan_points) >= 2:
                self.plotter.add_mesh(
                    pv.lines_from_points(scan_points, close=False),
                    color=color,
                    line_width=5,
                )
            if len(transition_points) >= 2:
                self.plotter.add_mesh(
                    pv.lines_from_points(transition_points, close=False),
                    color="gray",
                    line_width=1,
                    opacity=0.45,
                )

        scan_poses = [p for p in self.poses if p.motion_type == "scan"]
        if scan_poses:
            stride = max(1, len(scan_poses) // 250)
            selected = scan_poses[::stride]
            origins = np.array([[p.x, p.y, p.z] for p in selected])
            rotations = [
                Rotation.from_quat([p.qx, p.qy, p.qz, p.qw])
                for p in selected
            ]
            directions = np.array(
                [rotation.apply([0.0, 0.0, 1.0]) for rotation in rotations]
            )
            arrow_length = max(self.mesh.extents) * 0.03
            self.plotter.add_arrows(
                origins,
                directions * arrow_length,
                color="darkgreen",
            )

        self.plotter.add_axes()
        self.plotter.show_grid()
        self.plotter.reset_camera()

    def clear_trajectory(self) -> None:
        self.poses = []
        self.coverage_mask = None
        self.plan_stats = {}
        self.export_csv_button.setEnabled(False)
        self.export_json_button.setEnabled(False)
        self._show_mesh()
        self.status_label.setText("Trajectory cleared.")

    def export_csv(self) -> None:
        if not self.poses:
            return

        stem = Path(self.model_path).stem if self.model_path else "scan_plan"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export scan plan CSV",
            f"{stem}_scan_plan.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return

        fieldnames = list(asdict(self.poses[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for pose in self.poses:
                writer.writerow(asdict(pose))

        self.status_label.setText(f"CSV exported: {path}")

    def export_json(self) -> None:
        if not self.poses:
            return

        stem = Path(self.model_path).stem if self.model_path else "scan_plan"
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export scan plan JSON",
            f"{stem}_scan_plan.json",
            "JSON files (*.json)",
        )
        if not path:
            return

        payload = {
            "source_model": self.model_path,
            "source_type": self.model_type,
            "coordinate_frame": "CAD_STEP",
            "position_unit": "mm",
            "orientation_unit": "degree",
            "euler_sequence": "xyz",
            "scanner_model": "Micro-Epsilon scanCONTROL LLT30x0-200",
            "scanner_axes": {
                "+Z": "toward_surface",
                "+X": "laser_line",
                "+Y": "travel_direction",
            },
            "parameters": {
                "stand_off_mm": self.standoff_spin.value(),
                "laser_width_mm": self.laser_width_spin.value(),
                "overlap_percent": self.overlap_spin.value(),
                "travel_pose_spacing_mm": self.travel_spacing_spin.value(),
                "maximum_incidence_angle_deg":
                    self.normal_alignment_spin.value(),
                "approach_clearance_mm":
                    self.approach_clearance_spin.value(),
                "lead_distance_mm": self.lead_distance_spin.value(),
                "maximum_scanner_step_mm":
                    self.max_scanner_step_spin.value(),
                "maximum_normal_change_deg":
                    self.max_normal_change_spin.value(),
                "maximum_orientation_step_deg":
                    self.max_orientation_step_spin.value(),
                "minimum_sensor_range_mm": self.min_range_spin.value(),
                "maximum_sensor_range_mm": self.max_range_spin.value(),
                "laser_coverage_ray_count":
                    self.laser_ray_count_spin.value(),
                "minimum_valid_laser_fraction":
                    self.minimum_valid_laser_fraction_spin.value(),
                "include_isometric":
                    self.include_isometric_check.isChecked(),
            },
            "plan_statistics": self.plan_stats,
            "step_faces": self.face_metadata,
            "trajectory": [asdict(p) for p in self.poses],
        }

        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

        self.status_label.setText(f"JSON exported: {path}")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("UR5 STEP Scan Planner")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
