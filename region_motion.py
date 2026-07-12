"""Continuous region-level motion assembly from already-clipped scan lanes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from planners.base import MotionType, PlannerSettings, Pose, ScanMotionSegment


@dataclass(frozen=True)
class RegionMotionMetrics:
    old_segment_count: int
    new_segment_count: int
    old_approach_retract_count: int
    new_approach_retract_count: int
    lane_transition_count: int
    transition_distance_mm: float


def _reverse_scan_poses(poses: tuple[Pose, ...]) -> tuple[Pose, ...]:
    reversed_poses = []
    local_half_turn = Rotation.from_euler("z", 180.0, degrees=True)
    for pose in reversed(poses):
        rotation = Rotation.from_quat(pose.quaternion_xyzw) * local_half_turn
        reversed_poses.append(
            replace(
                pose,
                quaternion_xyzw=tuple(float(v) for v in rotation.as_quat()),
            )
        )
    return tuple(reversed_poses)


def _orientation_difference(first: Pose, second: Pose) -> float:
    a = Rotation.from_quat(first.quaternion_xyzw)
    b = Rotation.from_quat(second.quaternion_xyzw)
    return float(np.degrees((a.inv() * b).magnitude()))


def _nearest_order(motions: tuple[ScanMotionSegment, ...]):
    if not motions:
        return []
    remaining = list(motions)
    ordered = [(remaining.pop(0), False)]
    while remaining:
        current_scan = (
            _reverse_scan_poses(ordered[-1][0].scan_poses)
            if ordered[-1][1]
            else ordered[-1][0].scan_poses
        )
        current_end = current_scan[-1].position
        choices = []
        for index, motion in enumerate(remaining):
            scan = motion.scan_poses
            reversed_scan = _reverse_scan_poses(scan)
            choices.append(
                (
                    _orientation_difference(current_scan[-1], scan[0]) > 5.0,
                    float(np.linalg.norm(scan[0].position - current_end)),
                    index,
                    False,
                )
            )
            choices.append(
                (
                    _orientation_difference(current_scan[-1], reversed_scan[0]) > 5.0,
                    float(np.linalg.norm(reversed_scan[0].position - current_end)),
                    index,
                    True,
                )
            )
        _, _, index, reverse = min(choices, key=lambda row: (row[0], row[1]))
        ordered.append((remaining.pop(index), reverse))
    return ordered


def assemble_region_motion(
    motions: tuple[ScanMotionSegment, ...],
    settings: PlannerSettings,
    *,
    segment_id_start: int,
) -> tuple[tuple[ScanMotionSegment, ...], RegionMotionMetrics]:
    """Consolidate lanes while retaining full retracts across unsafe gaps."""

    ordered = _nearest_order(motions)
    groups: list[list[tuple[ScanMotionSegment, bool]]] = []
    transition_distance = 0.0
    transition_count = 0
    for item in ordered:
        if not groups:
            groups.append([item])
            continue
        previous_motion, previous_reverse = groups[-1][-1]
        previous_scan = (
            _reverse_scan_poses(previous_motion.scan_poses)
            if previous_reverse else previous_motion.scan_poses
        )
        next_scan = (
            _reverse_scan_poses(item[0].scan_poses) if item[1] else item[0].scan_poses
        )
        distance = float(
            np.linalg.norm(next_scan[0].position - previous_scan[-1].position)
        )
        orientation = _orientation_difference(previous_scan[-1], next_scan[0])
        if orientation <= 5.0:
            groups[-1].append(item)
            transition_distance += distance
            transition_count += 1
        else:
            groups.append([item])

    combined: list[ScanMotionSegment] = []
    for group_index, group in enumerate(groups):
        segment_id = segment_id_start + group_index
        poses: list[Pose] = []
        first_motion, first_reverse = group[0]
        first_scan = (
            _reverse_scan_poses(first_motion.scan_poses)
            if first_reverse else first_motion.scan_poses
        )
        # Region-level approach and lead-in use the selected first lane.
        poses.extend(first_motion.poses[:2])
        poses.extend(first_scan)
        for motion, reverse in group[1:]:
            scan = _reverse_scan_poses(motion.scan_poses) if reverse else motion.scan_poses
            for endpoint in (poses[-1], scan[0]):
                poses.append(
                    replace(
                        endpoint,
                        motion_type=MotionType.LANE_TRANSITION,
                        scan_enabled=False,
                    )
                )
            poses.extend(scan)
        last_motion, last_reverse = group[-1]
        end_poses = last_motion.poses[-2:]
        if last_reverse:
            # End transitions inherit the selected final scan frame and point.
            final = poses[-1]
            end_poses = tuple(
                replace(
                    pose,
                    surface_point=final.surface_point.copy(),
                    surface_normal=final.surface_normal.copy(),
                    quaternion_xyzw=final.quaternion_xyzw,
                )
                for pose in end_poses
            )
        poses.extend(end_poses)
        poses = [
            replace(pose, segment_id=segment_id, pose_id=index)
            for index, pose in enumerate(poses)
        ]
        combined.append(
            ScanMotionSegment(
                segment_id=segment_id,
                raster_segment=first_motion.raster_segment,
                poses=tuple(poses),
            )
        )
    return tuple(combined), RegionMotionMetrics(
        old_segment_count=len(motions),
        new_segment_count=len(combined),
        old_approach_retract_count=2 * len(motions),
        new_approach_retract_count=2 * len(combined),
        lane_transition_count=transition_count,
        transition_distance_mm=transition_distance,
    )
