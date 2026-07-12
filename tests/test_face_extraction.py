from __future__ import annotations

import math
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.TopoDS import TopoDS_Face, TopoDS_Wire

from cad_model import UnsupportedCadFormatError, load_step_model
from diagnostics.inspect_step import main as inspect_step_main
from face_analysis import SUPPORTED_SURFACE_TYPES, SurfaceType


class FaceExtractionTests(unittest.TestCase):
    def _write_step(self, shape: cq.Shape, suffix: str = ".step") -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / f"model{suffix}"
        cq.exporters.export(shape, str(path), exportType="STEP")
        return path

    def test_box_extracts_six_authoritative_planar_faces(self) -> None:
        path = self._write_step(cq.Workplane("XY").box(10, 20, 30).val())

        model = load_step_model(path)

        self.assertEqual(len(model.topods_faces), 6)
        self.assertEqual(len(model.faces), 6)
        self.assertTrue(all(isinstance(face, TopoDS_Face) for face in model.topods_faces))
        self.assertEqual(
            {record.geometry_type for record in model.faces}, {SurfaceType.PLANE}
        )
        self.assertAlmostEqual(sum(record.area_mm2 for record in model.faces), 2200.0)
        for record in model.faces:
            self.assertEqual(record.face_id, model.faces.index(record))
            self.assertEqual(len(record.adjacent_face_ids), 4)
            self.assertEqual(len(record.boundary_wires), 1)
            self.assertIsInstance(record.boundary_wires[0], TopoDS_Wire)
            normal = record.representative_normal
            self.assertAlmostEqual(
                math.sqrt(normal.x**2 + normal.y**2 + normal.z**2), 1.0
            )
            bounds = record.uv_bounds
            self.assertLess(bounds.u_min, bounds.u_max)
            self.assertLess(bounds.v_min, bounds.v_max)

    def test_cylinder_preserves_curved_face_and_adjacency(self) -> None:
        path = self._write_step(cq.Solid.makeCylinder(5.0, 12.0), ".stp")

        model = load_step_model(path)

        counts = {
            kind: sum(record.geometry_type is kind for record in model.faces)
            for kind in SurfaceType
        }
        self.assertEqual(counts[SurfaceType.CYLINDER], 1)
        self.assertEqual(counts[SurfaceType.PLANE], 2)
        cylindrical = next(
            record
            for record in model.faces
            if record.geometry_type is SurfaceType.CYLINDER
        )
        self.assertAlmostEqual(cylindrical.area_mm2, 2.0 * math.pi * 5.0 * 12.0)
        self.assertEqual(len(cylindrical.adjacent_face_ids), 2)

    def test_validation_mesh_mapping_is_complete_and_separate(self) -> None:
        path = self._write_step(cq.Workplane("XY").box(4, 5, 6).val())

        model = load_step_model(path)
        validation = model.validation_mesh

        self.assertEqual(len(validation.triangle_face_ids), len(validation.mesh.faces))
        self.assertTrue(np.all(validation.triangle_face_ids >= 0))
        self.assertTrue(np.all(validation.triangle_face_ids < len(model.faces)))
        self.assertFalse(validation.triangle_face_ids.flags.writeable)
        self.assertIsNot(model.shape, validation.mesh)

    def test_supported_surface_taxonomy_includes_required_types(self) -> None:
        self.assertEqual(
            SUPPORTED_SURFACE_TYPES,
            {
                SurfaceType.PLANE,
                SurfaceType.CYLINDER,
                SurfaceType.CONE,
                SurfaceType.SPHERE,
                SurfaceType.TORUS,
                SurfaceType.BSPLINE,
            },
        )

    def test_required_curved_surface_types_survive_step_round_trip(self) -> None:
        spline_points = [
            [
                cq.Vector(90.0 + i * 2.0, j * 2.0, 0.25 * i * j)
                for j in range(3)
            ]
            for i in range(3)
        ]
        shapes = [
            cq.Solid.makeCylinder(4.0, 8.0),
            cq.Solid.makeCone(2.0, 4.0, 8.0).translate(cq.Vector(20, 0, 0)),
            cq.Solid.makeSphere(5.0).translate(cq.Vector(40, 0, 0)),
            cq.Solid.makeTorus(7.0, 2.0).translate(cq.Vector(65, 0, 0)),
            cq.Face.makeSplineApprox(spline_points),
        ]
        path = self._write_step(cq.Compound.makeCompound(shapes))

        model = load_step_model(path)

        extracted = {record.geometry_type for record in model.faces}
        self.assertTrue(SUPPORTED_SURFACE_TYPES.issubset(extracted))

    def test_rejects_non_step_nominal_input(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".stl") as file:
            with self.assertRaises(UnsupportedCadFormatError):
                load_step_model(file.name)

    def test_diagnostic_prints_grouped_summary(self) -> None:
        path = self._write_step(cq.Solid.makeCylinder(3.0, 8.0))

        output = StringIO()
        with redirect_stdout(output):
            exit_code = inspect_step_main([str(path)])

        self.assertEqual(exit_code, 0)
        summary = output.getvalue()
        self.assertIn("Total B-Rep faces: 3", summary)
        self.assertIn("PLANE", summary)
        self.assertIn("CYLINDER", summary)
        self.assertIn("Validation mesh:", summary)


if __name__ == "__main__":
    unittest.main()
