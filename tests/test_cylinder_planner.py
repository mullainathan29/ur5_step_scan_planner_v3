from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.BRepClass import BRepClass_FaceClassifier
from OCP.TopAbs import TopAbs_State
from OCP.gp import gp_Pnt2d
from scipy.spatial.transform import Rotation

from cad_model import load_step_model
from face_analysis import SurfaceType
from planners.base import CylinderStrategy, MotionType, PlannerSettings
from planners.cylinder import plan_cylindrical_face


class CylindricalPlannerTests(unittest.TestCase):
    def _round_trip_cylinder_face(self, shape: cq.Shape):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "cylinder.step"
        cq.exporters.export(shape, str(path), exportType="STEP")
        model = load_step_model(path)
        face_id = next(
            record.face_id
            for record in model.faces
            if record.geometry_type is SurfaceType.CYLINDER
        )
        return model, model.topods_faces[face_id], model.faces[face_id]

    def _assert_trimmed_and_pose_valid(self, model, plan, settings) -> None:
        face = model.topods_faces[plan.face_id]
        for path, motion in zip(plan.path_segments, plan.motion_segments):
            self.assertGreaterEqual(len(path.samples), 2)
            self.assertEqual(motion.approach_pose.motion_type, MotionType.APPROACH)
            self.assertEqual(motion.retract_pose.motion_type, MotionType.RETRACT)
            self.assertFalse(motion.approach_pose.scan_enabled)
            self.assertFalse(motion.retract_pose.scan_enabled)
            self.assertEqual(len(motion.scan_poses), len(path.samples))
            self.assertEqual(len(motion.transition_poses), 4)

            quaternions = np.asarray(
                [pose.quaternion_xyzw for pose in motion.scan_poses]
            )
            if len(quaternions) > 1:
                self.assertTrue(
                    np.all(np.sum(quaternions[:-1] * quaternions[1:], axis=1) > 0)
                )

            for sample, pose in zip(path.samples, motion.scan_poses):
                state = BRepClass_FaceClassifier(
                    face,
                    gp_Pnt2d(*sample.uv),
                    settings.geometric_tolerance_mm,
                ).State()
                self.assertIn(
                    state, (TopAbs_State.TopAbs_IN, TopAbs_State.TopAbs_ON)
                )
                self.assertAlmostEqual(np.linalg.norm(sample.normal), 1.0)
                offset = pose.position - pose.surface_point
                self.assertAlmostEqual(np.linalg.norm(offset), settings.stand_off_mm)
                self.assertGreater(np.dot(offset, sample.normal), 0.0)
                rotation = Rotation.from_quat(pose.quaternion_xyzw)
                self.assertTrue(
                    np.allclose(rotation.apply([0, 0, 1]), -sample.normal)
                )

    def test_full_cylinder_longitudinal_and_helical(self) -> None:
        radius = 5.0
        height = 10.0
        model, face, record = self._round_trip_cylinder_face(
            cq.Solid.makeCylinder(radius, height)
        )
        settings = PlannerSettings(
            lane_spacing_mm=5.0,
            travel_spacing_mm=1.0,
            stand_off_mm=310.0,
        )

        longitudinal = plan_cylindrical_face(face, record, settings)

        self.assertEqual(
            longitudinal.strategy, CylinderStrategy.LONGITUDINAL
        )
        self.assertAlmostEqual(longitudinal.geometry.radius_mm, radius)
        self.assertAlmostEqual(
            longitudinal.geometry.angular_bounds[1]
            - longitudinal.geometry.angular_bounds[0],
            2.0 * math.pi,
        )
        self.assertEqual(len(longitudinal.path_segments), 7)
        self._assert_trimmed_and_pose_valid(model, longitudinal, settings)

        helical = plan_cylindrical_face(
            face,
            record,
            settings,
            strategy=CylinderStrategy.HELICAL,
        )
        self.assertEqual(helical.strategy, CylinderStrategy.HELICAL)
        self.assertGreaterEqual(len(helical.path_segments), 1)
        self._assert_trimmed_and_pose_valid(model, helical, settings)

    def test_half_cylinder_respects_angular_and_axial_trims(self) -> None:
        model, face, record = self._round_trip_cylinder_face(
            cq.Solid.makeCylinder(6.0, 12.0, angleDegrees=180.0)
        )
        settings = PlannerSettings(
            lane_spacing_mm=4.0,
            travel_spacing_mm=2.0,
        )

        plan = plan_cylindrical_face(face, record, settings)

        self.assertAlmostEqual(
            plan.geometry.angular_bounds[1] - plan.geometry.angular_bounds[0],
            math.pi,
        )
        expected_lanes = math.ceil(math.pi * 6.0 / settings.lane_spacing_mm)
        self.assertEqual(len(plan.path_segments), expected_lanes)
        for path in plan.path_segments:
            self.assertTrue(
                all(
                    plan.geometry.angular_bounds[0]
                    <= sample.uv[0]
                    <= plan.geometry.angular_bounds[1]
                    for sample in path.samples
                )
            )
        self._assert_trimmed_and_pose_valid(model, plan, settings)

        helical = plan_cylindrical_face(
            face,
            record,
            settings,
            strategy=CylinderStrategy.HELICAL,
        )
        self.assertGreaterEqual(len(helical.path_segments), 1)
        self._assert_trimmed_and_pose_valid(model, helical, settings)

    def test_trimmed_cylindrical_hole_wall_uses_inward_radial_normal(self) -> None:
        block = cq.Workplane("XY").box(20.0, 20.0, 10.0).val()
        bore = cq.Solid.makeCylinder(
            3.0,
            12.0,
            pnt=cq.Vector(0, 0, -6),
        )
        model, face, record = self._round_trip_cylinder_face(block.cut(bore))
        settings = PlannerSettings(
            lane_spacing_mm=3.0,
            travel_spacing_mm=1.5,
        )

        plan = plan_cylindrical_face(face, record, settings)

        self.assertTrue(record.reversed_orientation)
        self.assertGreater(len(plan.path_segments), 0)
        for path in plan.path_segments:
            for sample in path.samples:
                axis_point = (
                    plan.geometry.axis_origin
                    + sample.uv[1] * plan.geometry.axis_direction
                )
                radial = sample.point - axis_point
                radial /= np.linalg.norm(radial)
                self.assertLess(np.dot(sample.normal, radial), -0.999999)
        self._assert_trimmed_and_pose_valid(model, plan, settings)


if __name__ == "__main__":
    unittest.main()
