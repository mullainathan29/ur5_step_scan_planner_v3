from dataclasses import replace

import cadquery as cq

from face_analysis import extract_face_records
from planners.base import MotionType, PlannerSettings
from planners.plane import plan_planar_face
from region_motion import assemble_region_motion


def test_region_lanes_use_one_approach_retract_and_non_scanning_transitions() -> None:
    face = cq.Face.makePlane(20.0, 20.0)
    record = extract_face_records((face.wrapped,))[0]
    settings = PlannerSettings(
        lane_spacing_mm=5.0,
        travel_spacing_mm=5.0,
        stand_off_mm=310.0,
    )
    plan = plan_planar_face(face.wrapped, record, settings)
    annotated = tuple(
        replace(
            motion,
            poses=tuple(
                replace(pose, region_id=7, member_face_ids=(0,))
                for pose in motion.poses
            ),
        )
        for motion in plan.motion_segments
    )

    combined, metrics = assemble_region_motion(
        annotated, settings, segment_id_start=0
    )

    assert metrics.old_segment_count > 1
    assert metrics.new_segment_count == 1
    assert len(combined) == 1
    motion = combined[0]
    assert sum(p.motion_type is MotionType.APPROACH for p in motion.poses) == 1
    assert sum(p.motion_type is MotionType.RETRACT for p in motion.poses) == 1
    transitions = [
        pose for pose in motion.poses
        if pose.motion_type is MotionType.LANE_TRANSITION
    ]
    assert transitions
    assert all(not pose.scan_enabled for pose in transitions)
    assert metrics.new_approach_retract_count == 2
    assert metrics.old_approach_retract_count == 2 * metrics.old_segment_count
