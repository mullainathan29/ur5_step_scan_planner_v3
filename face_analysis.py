"""Typed STEP B-Rep face extraction and topology analysis."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import cadquery as cq
from OCP.TopAbs import TopAbs_Orientation
from OCP.TopoDS import TopoDS_Face, TopoDS_Wire


class FaceAnalysisError(RuntimeError):
    """Raised when a STEP face cannot be analyzed reliably."""


class SurfaceType(str, Enum):
    """Surface types supported by the phase-one B-Rep model."""

    PLANE = "PLANE"
    CYLINDER = "CYLINDER"
    CONE = "CONE"
    SPHERE = "SPHERE"
    TORUS = "TORUS"
    BSPLINE = "BSPLINE"
    OTHER = "OTHER"


SUPPORTED_SURFACE_TYPES = frozenset(
    {
        SurfaceType.PLANE,
        SurfaceType.CYLINDER,
        SurfaceType.CONE,
        SurfaceType.SPHERE,
        SurfaceType.TORUS,
        SurfaceType.BSPLINE,
    }
)


@dataclass(frozen=True)
class Point3D:
    """A serializable Cartesian vector or point in millimetres."""

    x: float
    y: float
    z: float


@dataclass(frozen=True)
class UVBounds:
    """Native OpenCascade parameter bounds for a trimmed face."""

    u_min: float
    u_max: float
    v_min: float
    v_max: float


@dataclass(frozen=True)
class FaceRecord:
    """Planning metadata for one authoritative STEP B-Rep face.

    boundary_wires contains native TopoDS_Wire objects. The first item is the
    outer wire; any remaining items are inner wires.
    """

    face_id: int
    geometry_type: SurfaceType
    area_mm2: float
    uv_bounds: UVBounds
    center: Point3D
    representative_normal: Point3D
    reversed_orientation: bool
    boundary_wires: tuple[TopoDS_Wire, ...]
    adjacent_face_ids: tuple[int, ...]


def classify_geometry_type(face: cq.Face) -> SurfaceType:
    """Map CadQuery/OpenCascade's surface label to the supported taxonomy."""

    label = str(face.geomType()).upper()
    aliases = {
        "PLANE": SurfaceType.PLANE,
        "CYLINDER": SurfaceType.CYLINDER,
        "CONE": SurfaceType.CONE,
        "SPHERE": SurfaceType.SPHERE,
        "TORUS": SurfaceType.TORUS,
        "BSPLINE": SurfaceType.BSPLINE,
        "B-SPLINE": SurfaceType.BSPLINE,
    }
    return aliases.get(label, SurfaceType.OTHER)


def _point3d(vector: cq.Vector) -> Point3D:
    return Point3D(float(vector.x), float(vector.y), float(vector.z))


def _ordered_wires(face: cq.Face) -> tuple[TopoDS_Wire, ...]:
    outer = face.outerWire()
    inner = face.innerWires()
    return (outer.wrapped, *(wire.wrapped for wire in inner))


def _edge_face_adjacency(faces: list[cq.Face]) -> list[set[int]]:
    """Return face adjacency based on shared topological edges."""

    edge_owners: dict[int, list[tuple[object, int]]] = {}
    for face_id, face in enumerate(faces):
        for edge in face.Edges():
            wrapped = edge.wrapped
            edge_owners.setdefault(hash(wrapped), []).append((wrapped, face_id))

    adjacency = [set() for _ in faces]
    for candidates in edge_owners.values():
        for i, (edge_a, face_a) in enumerate(candidates):
            for edge_b, face_b in candidates[i + 1 :]:
                if face_a != face_b and edge_a.IsSame(edge_b):
                    adjacency[face_a].add(face_b)
                    adjacency[face_b].add(face_a)
    return adjacency


def extract_face_records(
    topods_faces: Iterable[TopoDS_Face],
) -> tuple[FaceRecord, ...]:
    """Analyze native faces without tessellating nominal planning geometry."""

    wrapped_faces = [cq.Face(face) for face in topods_faces]
    if not wrapped_faces:
        raise FaceAnalysisError("The STEP model contains no TopoDS faces.")

    adjacency = _edge_face_adjacency(wrapped_faces)
    records: list[FaceRecord] = []

    for face_id, face in enumerate(wrapped_faces):
        try:
            u_min, u_max, v_min, v_max = face.uvBounds()
            representative_u = 0.5 * (u_min + u_max)
            representative_v = 0.5 * (v_min + v_max)
            center = face.positionAt(representative_u, representative_v)
            normal = face.normalAt()
            records.append(
                FaceRecord(
                    face_id=face_id,
                    geometry_type=classify_geometry_type(face),
                    area_mm2=float(face.Area()),
                    uv_bounds=UVBounds(
                        float(u_min), float(u_max), float(v_min), float(v_max)
                    ),
                    center=_point3d(center),
                    representative_normal=_point3d(normal),
                    reversed_orientation=(
                        face.wrapped.Orientation()
                        == TopAbs_Orientation.TopAbs_REVERSED
                    ),
                    boundary_wires=_ordered_wires(face),
                    adjacent_face_ids=tuple(sorted(adjacency[face_id])),
                )
            )
        except Exception as exc:
            raise FaceAnalysisError(
                f"Failed to analyze STEP face {face_id}: {exc}"
            ) from exc

    return tuple(records)
