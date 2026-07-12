from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import cadquery as cq
import numpy as np
from scipy.spatial.transform import Rotation

from cad_model import load_step_model
from planners.base import MotionType, PlannerSettings
from planners.plane import plan_planar_face


def _rectangle_wire(width: float, height: float) -> cq.Wire:
    half_width = width * 0.5
    half_height = height * 0.5
    return cq.Wire.makePolygon(
        [
            cq.Vector(-half_width, -half_height, 0),
            cq.Vector(half_width, -half_height, 0),
            cq.Vector(half_width, half_height, 0),
            cq.Vector(-half_width, half_height, 0),
        ],
        close=True,
    )


class PlanarRasterTests(unittest.TestCase):
    def _round_trip_face(self, face: cq.Face):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "face.step"
        cq.exporters.export(face, str(path), exportType="STEP")
        model = load_step_model(path)
        self.assertEqual(len(model.faces), 1)
        return model, model.topods_faces[0], model.faces[0]

    def test_rectangle_raster_sampling_and_scanner_pose(self) -> None:
        face = cq.Face.makeFromWires(_rectangle_wire(20.0, 10.0))
        model, topods_face, record = self._round_trip_face(face)
        settings = PlannerSettings(
            lane_spacing_mm=5.0,
            travel_spacing_mm=3.0,
            stand_off_mm=310.0,
            approach_clearance_mm=25.0,
            lead_distance_mm=2.0,
        )

        plan = plan_planar_face(topods_face, record, settings)

        self.assertEqual(len(plan.boundary_wires_2d), 1)
        self.assertEqual(len(plan.raster_segments), 4)
        self.assertEqual(len(plan.motion_segments), 4)
        for raster, motion in zip(plan.raster_segments, plan.motion_segments):
            self.assertEqual(len(raster.samples), 5)
            self.assertEqual(len(motion.poses), len(raster.samples) + 4)
            self.assertEqual(
                [pose.motion_type for pose in motion.poses[:2]],
                [MotionType.APPROACH, MotionType.LEAD_IN],
            )
            self.assertEqual(
                [pose.motion_type for pose in motion.poses[-2:]],
                [MotionType.LEAD_OUT, MotionType.RETRACT],
            )
            for first, second in zip(raster.samples, raster.samples[1:]):
                self.assertLessEqual(
                    np.linalg.norm(second.point - first.point),
                    settings.travel_spacing_mm + 1.0e-9,
                )
            for pose in motion.poses:
                if pose.scan_enabled:
                    offset = pose.position - pose.surface_point
                    self.assertAlmostEqual(np.linalg.norm(offset), 310.0)
                    self.assertGreater(np.dot(offset, pose.surface_normal), 0.0)

            scan_pose = next(p for p in motion.poses if p.scan_enabled)
            rotation = Rotation.from_quat(scan_pose.quaternion_xyzw)
            actual_travel = raster.samples[-1].point - raster.samples[0].point
            actual_travel /= np.linalg.norm(actual_travel)
            self.assertTrue(
                np.isclose(
                    abs(np.dot(rotation.apply([1, 0, 0]), plan.frame.laser_axis)),
                    1.0,
                )
            )
            self.assertTrue(
                np.allclose(rotation.apply([0, 1, 0]), actual_travel)
            )
            self.assertTrue(
                np.allclose(
                    rotation.apply([0, 0, 1]), -plan.frame.outward_normal
                )
            )
        self.assertIsNot(model.validation_mesh.mesh, model.shape)

    def test_circular_face_is_clipped_to_exact_circle(self) -> None:
        radius = 5.0
        face = cq.Face.makeFromWires(
            cq.Wire.makeCircle(
                radius,
                cq.Vector(0, 0, 0),
                cq.Vector(0, 0, 1),
            )
        )
        _, topods_face, record = self._round_trip_face(face)

        plan = plan_planar_face(
            topods_face,
            record,
            PlannerSettings(lane_spacing_mm=2.0, travel_spacing_mm=1.0),
        )

        self.assertGreaterEqual(len(plan.raster_segments), 5)
        for raster in plan.raster_segments:
            first = raster.samples[0].point
            last = raster.samples[-1].point
            for endpoint in (first, last):
                relative = endpoint - plan.frame.origin
                radial_distance = math.hypot(
                    np.dot(relative, plan.frame.laser_axis),
                    np.dot(relative, plan.frame.travel_axis),
                )
                self.assertAlmostEqual(radial_distance, radius, places=6)

    def test_rectangular_hole_creates_disconnected_scan_sections(self) -> None:
        outer = _rectangle_wire(20.0, 12.0)
        inner = _rectangle_wire(6.0, 4.0)
        face = cq.Face.makeFromWires(outer, [inner])
        _, topods_face, record = self._round_trip_face(face)

        plan = plan_planar_face(
            topods_face,
            record,
            PlannerSettings(lane_spacing_mm=2.0, travel_spacing_mm=1.0),
        )

        self.assertEqual(len(plan.boundary_wires_2d), 2)
        sections_per_lane: dict[int, int] = {}
        for raster in plan.raster_segments:
            sections_per_lane[raster.lane_index] = (
                sections_per_lane.get(raster.lane_index, 0) + 1
            )
            for sample in raster.samples:
                relative = sample.point - plan.frame.origin
                u = float(np.dot(relative, plan.frame.laser_axis))
                v = float(np.dot(relative, plan.frame.travel_axis))
                self.assertFalse(abs(u) < 3.0 - 1.0e-7 and abs(v) < 2.0 - 1.0e-7)

        self.assertTrue(any(count == 2 for count in sections_per_lane.values()))
        self.assertEqual(len(plan.motion_segments), len(plan.raster_segments))
        self.assertEqual(
            len({motion.segment_id for motion in plan.motion_segments}),
            len(plan.raster_segments),
        )
        for motion in plan.motion_segments:
            scan_poses = [pose for pose in motion.poses if pose.scan_enabled]
            self.assertEqual(len(scan_poses), len(motion.raster_segment.samples))


if __name__ == "__main__":
    unittest.main()
