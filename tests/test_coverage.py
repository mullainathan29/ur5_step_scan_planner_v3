from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation

from cad_model import load_step_model
from coverage import (
    CoverageSettings,
    TriangleStatus,
    detect_exterior_face_ids,
    validate_coverage,
)
from face_analysis import SurfaceType
from planners.base import PlannerSettings
from planners.plane import plan_planar_face


class CoverageValidationTests(unittest.TestCase):
    def _load_shape(self, shape: cq.Shape):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "coverage.step"
        cq.exporters.export(shape, str(path), exportType="STEP")
        return load_step_model(path, linear_tolerance=0.5)

    def _plan_single_plane(self, size: float = 40.0):
        model = self._load_shape(
            cq.Face.makeFromWires(
                cq.Wire.makePolygon(
                    [
                        cq.Vector(-size / 2, -size / 2, 0),
                        cq.Vector(size / 2, -size / 2, 0),
                        cq.Vector(size / 2, size / 2, 0),
                        cq.Vector(-size / 2, size / 2, 0),
                    ],
                    close=True,
                )
            )
        )
        plan = plan_planar_face(
            model.topods_faces[0],
            model.faces[0],
            PlannerSettings(
                lane_spacing_mm=10.0,
                travel_spacing_mm=10.0,
                stand_off_mm=310.0,
            ),
        )
        poses = tuple(
            pose
            for motion in plan.motion_segments
            for pose in motion.scan_poses
        )
        return model, poses

    def test_valid_laser_rays_accumulate_planar_triangle_area(self) -> None:
        model, poses = self._plan_single_plane()

        result = validate_coverage(
            model,
            poses,
            CoverageSettings(laser_width_mm=10.0, laser_ray_count=5),
        )

        planar = result.report["planar_faces"]
        self.assertGreater(result.first_hit_count, 0)
        self.assertGreater(result.valid_hit_count, 0)
        self.assertGreater(planar.covered_area_mm2, 0.0)
        self.assertTrue(
            np.any(result.triangle_status == TriangleStatus.VALID_COVERED)
        )
        self.assertEqual(result.report["cylindrical_faces"].covered_area_mm2, 0.0)

    def test_visible_out_of_range_hits_are_yellow(self) -> None:
        model, poses = self._plan_single_plane()
        near_poses = tuple(
            replace(
                pose,
                position=pose.surface_point + 100.0 * pose.surface_normal,
            )
            for pose in poses
        )

        result = validate_coverage(
            model,
            near_poses,
            CoverageSettings(laser_width_mm=10.0, laser_ray_count=5),
        )

        self.assertEqual(result.valid_hit_count, 0)
        self.assertGreater(result.invalid_range_hit_count, 0)
        self.assertTrue(
            np.any(result.triangle_status == TriangleStatus.VISIBLE_INVALID)
        )

    def test_visible_excess_incidence_hits_are_yellow(self) -> None:
        model, poses = self._plan_single_plane(size=1000.0)
        tilt = Rotation.from_euler("x", 20.0, degrees=True)
        tilted = tuple(
            replace(
                pose,
                quaternion_xyzw=tuple(
                    (
                        Rotation.from_quat(pose.quaternion_xyzw) * tilt
                    ).as_quat()
                ),
            )
            for pose in poses
        )

        result = validate_coverage(
            model,
            tilted,
            CoverageSettings(
                laser_width_mm=10.0,
                laser_ray_count=3,
                max_incidence_angle_deg=10.0,
            ),
        )

        self.assertEqual(result.valid_hit_count, 0)
        self.assertGreater(result.invalid_incidence_hit_count, 0)
        self.assertTrue(
            np.any(result.triangle_status == TriangleStatus.VISIBLE_INVALID)
        )

    def test_internal_bore_wall_is_excluded_unless_enabled(self) -> None:
        block = cq.Workplane("XY").box(20.0, 20.0, 10.0).val()
        bore = cq.Solid.makeCylinder(
            3.0,
            12.0,
            pnt=cq.Vector(0, 0, -6),
        )
        model = self._load_shape(block.cut(bore))
        bore_face_id = next(
            record.face_id
            for record in model.faces
            if record.geometry_type is SurfaceType.CYLINDER
        )

        exterior = detect_exterior_face_ids(model)
        excluded = validate_coverage(model, ())
        included = validate_coverage(
            model,
            (),
            CoverageSettings(include_internal_surfaces=True),
        )

        self.assertNotIn(bore_face_id, exterior)
        self.assertIn(bore_face_id, excluded.internal_face_ids)
        self.assertGreater(
            included.report["cylindrical_faces"].eligible_area_mm2,
            excluded.report["cylindrical_faces"].eligible_area_mm2,
        )
        self.assertGreater(
            included.report["total_exterior_surface"].eligible_area_mm2,
            excluded.report["total_exterior_surface"].eligible_area_mm2,
        )


if __name__ == "__main__":
    unittest.main()
