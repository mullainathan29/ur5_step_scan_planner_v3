"""Authoritative STEP B-Rep loading with a separate validation mesh."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from OCP.TopoDS import TopoDS_Face

from face_analysis import FaceAnalysisError, FaceRecord, extract_face_records


class CadModelError(RuntimeError):
    """Base exception for CAD model loading failures."""


class UnsupportedCadFormatError(CadModelError):
    """Raised when nominal B-Rep planning is requested for a non-STEP file."""


class StepImportError(CadModelError):
    """Raised when OpenCascade cannot import a usable STEP shape."""


@dataclass(frozen=True)
class ValidationMesh:
    """Tessellation reserved for rendering and later validation.

    triangle_face_ids[i] identifies the authoritative STEP face that produced
    mesh.faces[i]. The mesh is deliberately not merged or reprocessed, because
    those operations can invalidate this mapping.
    """

    mesh: trimesh.Trimesh
    triangle_face_ids: np.ndarray


@dataclass(frozen=True)
class CadModel:
    """A loaded STEP model retaining both B-Rep and validation representations."""

    source_path: Path
    shape: cq.Shape
    topods_faces: tuple[TopoDS_Face, ...]
    faces: tuple[FaceRecord, ...]
    validation_mesh: ValidationMesh


@dataclass(frozen=True)
class CadBRep:
    """Cached authoritative geometry before any display tessellation."""

    source_path: Path
    shape: cq.Shape
    topods_faces: tuple[TopoDS_Face, ...]
    faces: tuple[FaceRecord, ...]


def _validate_step_path(path: str | Path) -> Path:
    source = Path(path).expanduser()
    if source.suffix.lower() not in {".step", ".stp"}:
        raise UnsupportedCadFormatError(
            f"Expected a .step or .stp file, received '{source.suffix or '<none>'}'."
        )
    if not source.is_file():
        raise StepImportError(f"STEP file does not exist: {source}")
    return source.resolve()


def tessellate_for_validation(
    faces: tuple[TopoDS_Face, ...],
    linear_tolerance: float = 0.25,
    angular_tolerance: float = 0.1,
    cancellation_check=None,
) -> ValidationMesh:
    """Create a non-authoritative mesh while preserving triangle face IDs."""

    if linear_tolerance <= 0.0:
        raise ValueError("linear_tolerance must be greater than zero.")
    if angular_tolerance <= 0.0:
        raise ValueError("angular_tolerance must be greater than zero.")

    cancellation_check = cancellation_check or (lambda: None)
    vertices: list[np.ndarray] = []
    triangles: list[np.ndarray] = []
    triangle_face_ids: list[int] = []
    vertex_offset = 0

    for face_id, topods_face in enumerate(faces):
        cancellation_check()
        face = cq.Face(topods_face)
        face_vertices, face_triangles = face.tessellate(
            tolerance=float(linear_tolerance),
            angularTolerance=float(angular_tolerance),
        )
        if not face_vertices or not face_triangles:
            continue

        vertex_array = np.asarray(
            [(vertex.x, vertex.y, vertex.z) for vertex in face_vertices],
            dtype=np.float64,
        )
        triangle_array = np.asarray(face_triangles, dtype=np.int64)
        triangle_points = vertex_array[triangle_array]
        triangle_array = triangle_array[
            trimesh.triangles.nondegenerate(triangle_points)
        ]
        if not len(triangle_array):
            continue
        vertices.append(vertex_array)
        triangles.append(triangle_array + vertex_offset)
        triangle_face_ids.extend([face_id] * len(triangle_array))
        vertex_offset += len(vertex_array)

    if not triangles:
        raise StepImportError("STEP validation tessellation produced no triangles.")

    mesh = trimesh.Trimesh(
        vertices=np.vstack(vertices),
        faces=np.vstack(triangles),
        process=False,
    )
    mapping = np.asarray(triangle_face_ids, dtype=np.int64)
    if len(mapping) != len(mesh.faces):
        raise StepImportError(
            "Validation triangle-to-STEP-face mapping is inconsistent."
        )
    mapping.setflags(write=False)
    mesh.metadata["triangle_face_ids"] = mapping
    return ValidationMesh(mesh=mesh, triangle_face_ids=mapping)


def load_step_model(
    path: str | Path,
    *,
    linear_tolerance: float = 0.25,
    angular_tolerance: float = 0.1,
) -> CadModel:
    """Load STEP B-Rep faces and independently build a validation mesh."""

    brep = load_step_brep(path)
    validation_mesh = tessellate_for_validation(
        brep.topods_faces,
        linear_tolerance=linear_tolerance,
        angular_tolerance=angular_tolerance,
    )
    return cad_model_from_brep(brep, validation_mesh)


def load_step_brep(
    path: str | Path,
    *,
    stage_callback=None,
    cancellation_check=None,
) -> CadBRep:
    """Import and analyse STEP without creating a triangle mesh."""

    from time import perf_counter

    source = _validate_step_path(path)
    cancellation_check = cancellation_check or (lambda: None)
    started = perf_counter()
    try:
        imported = cq.importers.importStep(str(source))
        shape = imported.val()
    except Exception as exc:
        raise StepImportError(f"Failed to import STEP file '{source}': {exc}") from exc

    if shape is None:
        raise StepImportError(f"STEP import returned no shape for '{source}'.")
    if stage_callback:
        stage_callback("STEP load", perf_counter() - started)
    cancellation_check()

    started = perf_counter()
    cadquery_faces = shape.Faces()
    topods_faces = tuple(face.wrapped for face in cadquery_faces)
    if not topods_faces:
        raise StepImportError(f"STEP model contains no faces: {source}")

    try:
        records = extract_face_records(topods_faces)
    except FaceAnalysisError as exc:
        raise StepImportError(f"Failed to analyze STEP model '{source}': {exc}") from exc
    if stage_callback:
        stage_callback("face extraction", perf_counter() - started)
    cancellation_check()
    return CadBRep(
        source_path=source,
        shape=shape,
        topods_faces=topods_faces,
        faces=records,
    )


def cad_model_from_brep(brep: CadBRep, validation_mesh: ValidationMesh) -> CadModel:
    return CadModel(
        source_path=brep.source_path,
        shape=brep.shape,
        topods_faces=brep.topods_faces,
        faces=brep.faces,
        validation_mesh=validation_mesh,
    )
