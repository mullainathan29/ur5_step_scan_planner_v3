"""Feature and underside scan candidates without robot-body insertion."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

from cad_model import CadModel
from features import HoleCoverageReport, HoleFeature
from planners.base import MotionType, Pose
from planners.view_based import ViewSweep, ViewSweepSettings, _bounds, _quaternion


@dataclass(frozen=True)
class RecoveryCandidate:
    candidate_id: int
    name: str
    strategy: str
    sweep: ViewSweep
    target_hole_id: int = -1
    mandatory: bool = False
    candidate_score: float = 0.0
    coverage_gain_percent: float = 0.0
    hole_coverage_before: float = 0.0
    hole_coverage_after: float = 0.0
    bottom_coverage_gain: float = 0.0


@dataclass(frozen=True)
class CandidateComplexityEstimate:
    physical_holes: int
    incomplete_holes: int
    cross_hole_candidates: int
    orbital_candidates: int
    bottom_candidates: int
    estimated_candidates: int
    pruned_candidate_limit: int


def estimate_candidate_complexity(
    features,
    reports=None,
    *,
    tilt_count: int,
    azimuth_count: int,
    orbital_mode: str = "disabled",
    blind_bottom_enabled: bool = True,
    bottom_sweep_enabled: bool = True,
    lower_isometric_enabled: bool = True,
    maximum_views_per_hole: int = 2,
) -> CandidateComplexityEstimate:
    features = tuple(features)
    reports_by_id = {row.feature_id: row for row in (reports or ())}
    eligible = [feature for feature in features
                if feature.feature_id not in reports_by_id
                or not reports_by_id[feature.feature_id].complete]
    cross = len(eligible) * max(1, tilt_count) * max(1, azimuth_count)
    orbital = len(eligible) if orbital_mode in {"adaptive", "enabled"} else 0
    blind = sum(not feature.through_hole for feature in eligible)
    bottom = (2 * blind if blind_bottom_enabled else 0)
    bottom += int(bottom_sweep_enabled)
    bottom += 3 * int(lower_isometric_enabled)
    estimated = cross + orbital + bottom
    pruned = len(eligible) * maximum_views_per_hole
    pruned += int(bottom_sweep_enabled) + 3 * int(lower_isometric_enabled)
    return CandidateComplexityEstimate(
        len(features), len(eligible), cross, orbital, bottom,
        estimated, pruned,
    )


def _normalise(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    return array / np.linalg.norm(array)


def _continuous_quaternion(axes, previous=None):
    quaternion = np.asarray(_quaternion(*axes))
    if previous is not None and np.dot(previous, quaternion) < 0:
        quaternion *= -1
    return tuple(quaternion)


def _motion_poses(
    points: list[np.ndarray], targets: list[np.ndarray], laser_axes: list[np.ndarray],
    sweep_id: int, name: str, view_type: str, strategy: str,
    stand_off: float, visible_width: float, feature_id: int = -1,
    approach_clearance: float = 120.0,
) -> tuple[Pose, ...]:
    scans = []
    previous_q = None
    for index, (position, target, laser) in enumerate(zip(points, targets, laser_axes)):
        measurement = _normalise(target - position)
        laser = _normalise(laser - np.dot(laser, measurement) * measurement)
        travel_axis = _normalise(np.cross(measurement, laser))
        if len(points) > 1:
            neighbour = points[index + 1] if index + 1 < len(points) else points[index]
            previous = points[index] if index + 1 < len(points) else points[index - 1]
            physical_travel = _normalise(neighbour - previous)
        else:
            physical_travel = travel_axis
        quaternion = _continuous_quaternion((laser, travel_axis, measurement), previous_q)
        previous_q = np.asarray(quaternion)
        scans.append(Pose(
            face_id=-1, segment_id=sweep_id, pose_id=index + 2,
            motion_type=MotionType.SCAN, scan_enabled=True,
            surface_point=target, surface_normal=-measurement,
            position=position, quaternion_xyzw=quaternion,
            region_id=sweep_id, sweep_id=sweep_id, sweep_name=name,
            view_type=view_type, laser_line_direction=tuple(laser),
            travel_direction=tuple(physical_travel),
            predicted_stand_off_mm=float(np.linalg.norm(position - target)),
            predicted_visible_laser_width_mm=visible_width,
            feature_id=feature_id, feature_type="hole" if feature_id >= 0 else "underside",
            target_hole_id=feature_id, recovery_strategy=strategy,
            mandatory_or_optional="optional",
        ))
    if not scans:
        return ()
    first, last = scans[0], scans[-1]
    return (
        replace(first, pose_id=0, motion_type=MotionType.APPROACH,
                scan_enabled=False,
                position=first.position + approach_clearance * first.surface_normal),
        replace(first, pose_id=1, motion_type=MotionType.LEAD_IN, scan_enabled=False),
        *scans,
        replace(last, pose_id=len(scans) + 2, motion_type=MotionType.LEAD_OUT,
                scan_enabled=False),
        replace(last, pose_id=len(scans) + 3, motion_type=MotionType.RETRACT,
                scan_enabled=False,
                position=last.position + approach_clearance * last.surface_normal),
    )


def _focused_arc_ranges(report, minimum_arc_deg, maximum_arc_deg, maximum_orbits):
    sectors = sorted(report.uncovered_angular_sectors_deg)
    if not sectors:
        return ()
    merged = []
    for start, end in sectors:
        if merged and start <= merged[-1][1] + 1.0e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([float(start), float(end)])
    if len(merged) > 1 and merged[0][0] == 0.0 and merged[-1][1] == 360.0:
        merged = [[merged[-1][0] - 360.0, merged[0][1]], *merged[1:-1]]
    arcs = []
    for start, end in merged:
        centre = 0.5 * (start + end)
        span = min(max(end - start, minimum_arc_deg), maximum_arc_deg)
        arcs.append((centre - span / 2.0, centre + span / 2.0))
    return tuple(sorted(arcs, key=lambda row: row[1] - row[0], reverse=True)
                 [:max(0, maximum_orbits)])


def _focused_orbit_candidate(feature, settings, sweep_id, cone_angle_deg, arc,
                              focus_depth_fraction, angular_spacing_deg):
    centre = np.asarray(feature.entry_opening_centre, dtype=float)
    axis = _normalise(-np.asarray(feature.entry_opening_normal, dtype=float))
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = _normalise(reference - np.dot(reference, axis) * axis)
    e2 = _normalise(np.cross(axis, e1))
    focus_depth = min(max(1.0e-6, focus_depth_fraction * feature.depth_mm),
                      2.0 * feature.radius_mm,
                      max(1.0e-6, feature.depth_mm - 1.0e-6))
    focus = centre + focus_depth * axis
    preferred_distance = float(np.clip(settings.stand_off_mm, 290.0, 330.0))
    cone = np.deg2rad(cone_angle_deg)
    start_deg, end_deg = arc
    count = max(2, int(np.ceil(abs(end_deg - start_deg) / angular_spacing_deg)) + 1)
    angles = np.deg2rad(np.linspace(start_deg, end_deg, count))
    positions, lasers = [], []
    for theta in angles:
        radial = np.cos(theta) * e1 + np.sin(theta) * e2
        tangent = _normalise(-np.sin(theta) * e1 + np.cos(theta) * e2)
        measurement = _normalise(np.cos(cone) * axis - np.sin(cone) * radial)
        positions.append(focus - preferred_distance * measurement)
        lasers.append(_normalise(np.cross(tangent, measurement)))
    name = (f"hole_{feature.feature_id}_focused_{int(round(cone_angle_deg))}_"
            f"{int(round(start_deg))}_{int(round(end_deg))}")
    poses = _motion_poses(
        positions, [focus] * len(positions), lasers, sweep_id, name,
        "hole_recovery", "focused_conical_orbit", preferred_distance,
        2.0 * feature.radius_mm, feature.feature_id,
    )
    metadata = dict(
        focus_point=tuple(float(value) for value in focus),
        focus_depth_mm=float(focus_depth),
        orbit_radius_mm=float(preferred_distance * np.sin(cone)),
        cone_angle_deg=float(cone_angle_deg), orbit_start_deg=float(start_deg),
        orbit_end_deg=float(end_deg), hole_axis=tuple(float(value) for value in axis),
    )
    poses = tuple(replace(pose, **metadata) for pose in poses)
    return RecoveryCandidate(
        sweep_id, name, "focused_conical_orbit",
        ViewSweep(sweep_id, name, "hole_recovery", 0, poses, "turquoise"),
        feature.feature_id,
    )


def _score_focused_orbit(candidate, feature, model, laser_ray_count,
                         laser_width_mm, minimum_arc_deg):
    # Cover the full physical laser line and add local aperture-resolution
    # rays. Uniform preview rays alone can miss a small bore between samples.
    full_line_offsets = np.linspace(
        -0.5 * laser_width_mm, 0.5 * laser_width_mm,
        max(3, laser_ray_count),
    )
    opening_offsets = np.linspace(-feature.radius_mm, feature.radius_mm, 9)
    offsets = np.unique(np.concatenate((full_line_offsets, opening_offsets)))
    origins, directions = [], []
    for pose in candidate.sweep.scan_poses:
        rotation = Rotation.from_quat(pose.quaternion_xyzw)
        laser = rotation.apply([1.0, 0.0, 0.0])
        direction = rotation.apply([0.0, 0.0, 1.0])
        origins.extend(pose.position + offset * laser for offset in offsets)
        directions.extend([direction] * len(offsets))
    origins = np.asarray(origins)
    directions = np.asarray(directions)
    locations, ray_ids, triangle_ids = model.validation_mesh.mesh.ray.intersects_location(
        origins, directions, multiple_hits=False,
    )
    ray_ids = np.asarray(ray_ids, dtype=int)
    triangle_ids = np.asarray(triangle_ids, dtype=int)
    if not len(triangle_ids):
        return None
    distances = np.linalg.norm(np.asarray(locations) - origins[ray_ids], axis=1)
    normals = np.asarray(model.validation_mesh.mesh.face_normals)[triangle_ids]
    incidence = np.abs(np.einsum("ij,ij->i", normals, directions[ray_ids]))
    mapping = model.validation_mesh.triangle_face_ids
    areas = np.asarray(model.validation_mesh.mesh.area_faces)
    useful_hits = (np.isin(mapping[triangle_ids], feature.cylindrical_face_ids)
                   & (distances >= 290.0) & (distances <= 330.0)
                   & (incidence >= np.cos(np.deg2rad(60.0))))
    pose_indices = ray_ids // len(offsets)
    useful_distances = []
    pose_valid = []
    for pose_index in range(len(candidate.sweep.scan_poses)):
        values = distances[useful_hits & (pose_indices == pose_index)]
        pose_valid.append(bool(len(values)))
        useful_distances.append(float(np.median(values)) if len(values) else np.nan)
    runs, run_start = [], None
    for index, valid in enumerate((*pose_valid, False)):
        if valid and run_start is None:
            run_start = index
        elif not valid and run_start is not None:
            runs.append((run_start, index)); run_start = None
    if not runs:
        return None
    start_index, stop_index = max(runs, key=lambda row: row[1] - row[0])
    original_scans = candidate.sweep.scan_poses
    total_span = float(original_scans[0].orbit_end_deg - original_scans[0].orbit_start_deg)
    step = total_span / max(len(original_scans) - 1, 1)
    retained_span = step * max(stop_index - start_index - 1, 0)
    if retained_span + 1.0e-6 < minimum_arc_deg:
        return None
    retained_rays = ((pose_indices >= start_index) & (pose_indices < stop_index))
    useful_hits &= retained_rays
    covered_ids = np.unique(triangle_ids[useful_hits])
    if not len(covered_ids):
        return None
    wall_area = float(areas[covered_ids].sum())
    centres = np.asarray(model.validation_mesh.mesh.triangles_center)[covered_ids]
    relative = centres - feature.entry_opening_centre
    depths = relative @ (-feature.entry_opening_normal)
    depth_gain = max(0.0, float(np.ptp(depths))) if len(depths) > 1 else 0.0
    axis = _normalise(feature.axis)
    reference = np.array([1.0, 0.0, 0.0])
    if abs(float(np.dot(reference, axis))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = _normalise(reference - np.dot(reference, axis) * axis)
    e2 = np.cross(axis, e1)
    radial = relative - np.outer(relative @ axis, axis)
    angles = np.unwrap(np.arctan2(radial @ e2, radial @ e1))
    angular_gain = float(np.degrees(np.ptp(angles))) if len(angles) > 1 else 0.0
    scans = candidate.sweep.scan_poses[start_index:stop_index]
    corrected_scans = []
    trimmed_start = float(scans[0].orbit_start_deg + start_index * step)
    trimmed_end = float(scans[0].orbit_start_deg + (stop_index - 1) * step)
    for pose, hit_distance in zip(scans, useful_distances[start_index:stop_index]):
        z_axis = Rotation.from_quat(pose.quaternion_xyzw).apply([0.0, 0.0, 1.0])
        correction = 310.0 - hit_distance
        corrected_scans.append(replace(
            pose, position=pose.position - correction * z_axis,
            predicted_stand_off_mm=310.0, orbit_start_deg=trimmed_start,
            orbit_end_deg=trimmed_end,
        ))
    first, last = corrected_scans[0], corrected_scans[-1]
    approach_clearance = 120.0
    corrected_poses = (
        replace(candidate.sweep.poses[0], position=first.position
                + approach_clearance * first.surface_normal,
                orbit_start_deg=trimmed_start, orbit_end_deg=trimmed_end),
        replace(candidate.sweep.poses[1], position=first.position,
                orbit_start_deg=trimmed_start, orbit_end_deg=trimmed_end),
        *corrected_scans,
        replace(candidate.sweep.poses[-2], position=last.position,
                orbit_start_deg=trimmed_start, orbit_end_deg=trimmed_end),
        replace(candidate.sweep.poses[-1], position=last.position
                + approach_clearance * last.surface_normal,
                orbit_start_deg=trimmed_start, orbit_end_deg=trimmed_end),
    )
    candidate = replace(candidate, sweep=replace(candidate.sweep,
                                                 poses=corrected_poses))
    scans = candidate.sweep.scan_poses
    path_length = sum(float(np.linalg.norm(b.position - a.position))
                      for a, b in zip(scans, scans[1:]))
    unrelated_hits = int(np.count_nonzero(
        ~np.isin(mapping[triangle_ids], feature.cylindrical_face_ids)
    ))
    score = (wall_area + 0.2 * angular_gain + depth_gain
             - 0.01 * path_length - 0.02 * unrelated_hits)
    sweep = replace(candidate.sweep, poses=tuple(replace(
        pose, predicted_hole_wall_gain=wall_area,
        predicted_depth_gain=depth_gain,
        predicted_angular_sector_gain=angular_gain,
    ) for pose in candidate.sweep.poses))
    return score, replace(candidate, sweep=sweep, candidate_score=score,
                          coverage_gain_percent=wall_area)


def generate_focused_conical_orbits(
    feature: HoleFeature, report: HoleCoverageReport, settings: ViewSweepSettings,
    *, model: CadModel | None = None, candidate_id_start: int = 0,
    cone_angles_deg: tuple[float, ...] = (20.0, 35.0, 50.0),
    focus_depth_fraction: float = 0.6, minimum_arc_deg: float = 45.0,
    maximum_arc_deg: float = 240.0, angular_spacing_deg: float = 5.0,
    maximum_orbits: int = 1, laser_ray_count: int = 21,
) -> tuple[RecoveryCandidate, ...]:
    """Return the best valid cone angle for each bounded missing-wall arc."""
    if report.complete or not feature.cylindrical_face_ids:
        return ()
    arcs = _focused_arc_ranges(report, minimum_arc_deg, maximum_arc_deg,
                               maximum_orbits)
    selected = []
    for index, arc in enumerate(arcs):
        candidates = [_focused_orbit_candidate(
            feature, settings, candidate_id_start + index, angle, arc,
            focus_depth_fraction, angular_spacing_deg,
        ) for angle in cone_angles_deg]
        if model is None:
            if candidates:
                selected.append(candidates[0])
            continue
        scored = [value for candidate in candidates
                  if (value := _score_focused_orbit(
                      candidate, feature, model, laser_ray_count,
                      settings.laser_width_mm, minimum_arc_deg)) is not None]
        if scored:
            selected.append(max(scored, key=lambda value: value[0])[1])
    return tuple(selected)


def generate_hole_recovery_candidates(
    feature: HoleFeature,
    report: HoleCoverageReport,
    settings: ViewSweepSettings,
    *,
    candidate_id_start: int = 0,
    tilt_angles_deg: tuple[float, ...] = (20.0, 35.0, 50.0),
    azimuth_count: int = 4,
    maximum_views: int = 8,
    enable_orbital: bool = True,
    enable_blind_bottom: bool = True,
    orbital_mode: str | None = None,
    shallow_primary_only: bool = True,
) -> tuple[RecoveryCandidate, ...]:
    if report.complete:
        return ()
    orbital_mode = orbital_mode or ("enabled" if enable_orbital else "disabled")
    enable_orbital = orbital_mode in {"adaptive", "enabled"}
    normal = _normalise(feature.entry_opening_normal)
    reference = np.array([1., 0., 0.])
    if abs(np.dot(reference, normal)) > 0.9:
        reference = np.array([0., 1., 0.])
    u = _normalise(reference - np.dot(reference, normal) * normal)
    v = np.cross(normal, u)
    candidates = []
    candidate_id = candidate_id_start
    reserved = (2 if enable_orbital else 0) + (
        2 if enable_blind_bottom and not feature.through_hole else 0
    )
    tilted_limit = max(1, maximum_views - reserved)
    shallow = feature.depth_mm <= 4.0 * feature.radius_mm
    selected_tilts = tilt_angles_deg[:1] if shallow and shallow_primary_only else tilt_angles_deg
    if shallow and shallow_primary_only:
        azimuths = np.asarray([0.0])
    elif shallow:
        azimuths = np.asarray([0.0, 0.5 * np.pi])[:max(1, azimuth_count)]
    elif report.uncovered_angular_sectors_deg:
        centres = [0.5 * (start + end) for start, end
                   in report.uncovered_angular_sectors_deg]
        azimuths = np.deg2rad(np.asarray(centres[:max(1, azimuth_count)]))
    else:
        azimuths = np.linspace(0, 2 * np.pi, max(1, azimuth_count), endpoint=False)
    if shallow and shallow_primary_only:
        tilted_limit = 1
    for tilt_deg in selected_tilts:
        for azimuth in azimuths:
            if len(candidates) >= tilted_limit:
                break
            radial = np.cos(azimuth) * u + np.sin(azimuth) * v
            measurement = _normalise(-normal + np.tan(np.deg2rad(tilt_deg)) * radial)
            target_depth = min(feature.depth_mm * 0.65, max(feature.radius_mm, 1.0))
            target = feature.entry_opening_centre - target_depth * normal
            centre = target - settings.stand_off_mm * measurement
            travel = _normalise(np.cross(measurement, radial))
            values = np.linspace(-feature.radius_mm, feature.radius_mm,
                                 max(3, int(ceil(2 * feature.radius_mm / settings.travel_spacing_mm)) + 1))
            points = [centre + value * travel for value in values]
            targets = [target + value * travel for value in values]
            lasers = [radial] * len(points)
            name = f"hole_{feature.feature_id}_tilt_{int(tilt_deg)}_az_{int(np.degrees(azimuth))}"
            poses = _motion_poses(points, targets, lasers, candidate_id, name,
                                  "hole_recovery", "tilted_cross_hole",
                                  settings.stand_off_mm, 2 * feature.radius_mm,
                                  feature.feature_id)
            sweep = ViewSweep(candidate_id, name, "hole_recovery", len(candidates),
                              poses, "red")
            candidates.append(RecoveryCandidate(candidate_id, name,
                                                "tilted_cross_hole", sweep,
                                                feature.feature_id))
            candidate_id += 1
    if enable_orbital:
        angles = np.linspace(0, 2 * np.pi, max(13, azimuth_count * 4 + 1))
        orbit_radius = 1.5 * feature.radius_mm
        target = feature.entry_opening_centre - min(feature.depth_mm * 0.5,
                                                    feature.radius_mm) * normal
        points = [feature.entry_opening_centre + settings.stand_off_mm * normal
                  + orbit_radius * (np.cos(a) * u + np.sin(a) * v) for a in angles]
        targets = [target] * len(points)
        lasers = [-np.sin(a) * u + np.cos(a) * v for a in angles]
        name = f"hole_{feature.feature_id}_orbit"
        poses = _motion_poses(points, targets, lasers, candidate_id, name,
                              "hole_recovery", "orbital_hole",
                              settings.stand_off_mm, 2 * feature.radius_mm,
                              feature.feature_id)
        orbital_strategy = (
            "adaptive_orbital_hole" if orbital_mode == "adaptive" else "orbital_hole"
        )
        candidates.append(RecoveryCandidate(candidate_id, name, orbital_strategy,
                                            ViewSweep(candidate_id, name,
                                                      "hole_recovery", 0, poses,
                                                      "crimson"), feature.feature_id))
        candidate_id += 1
        if report.uncovered_angular_sectors_deg:
            start_deg, end_deg = report.uncovered_angular_sectors_deg[0]
            arc_angles = np.deg2rad(np.linspace(start_deg - 10.0, end_deg + 10.0, 9))
            arc_points = [feature.entry_opening_centre + settings.stand_off_mm * normal
                          + orbit_radius * (np.cos(a) * u + np.sin(a) * v)
                          for a in arc_angles]
            arc_lasers = [-np.sin(a) * u + np.cos(a) * v for a in arc_angles]
            name = f"hole_{feature.feature_id}_arc_{int(start_deg)}_{int(end_deg)}"
            poses = _motion_poses(
                arc_points, [target] * len(arc_points), arc_lasers,
                candidate_id, name, "hole_recovery", "arc_recovery",
                settings.stand_off_mm, 2 * feature.radius_mm,
                feature.feature_id,
            )
            candidates.append(RecoveryCandidate(
                candidate_id, name, "arc_recovery",
                ViewSweep(candidate_id, name, "hole_recovery", 0, poses,
                          "orangered"), feature.feature_id,
            ))
            candidate_id += 1
    if enable_blind_bottom and not feature.through_hole and feature.bottom_face_id is not None:
        target = feature.entry_opening_centre - feature.depth_mm * normal
        for sign in (-1.0, 1.0):
            measurement = _normalise(-normal + sign * 0.35 * u)
            centre = target - settings.stand_off_mm * measurement
            points = [centre + value * v for value in np.linspace(-feature.radius_mm,
                                                                   feature.radius_mm, 5)]
            targets = [target + value * v for value in np.linspace(-feature.radius_mm,
                                                                    feature.radius_mm, 5)]
            name = f"hole_{feature.feature_id}_bottom_{'a' if sign < 0 else 'b'}"
            poses = _motion_poses(points, targets, [u] * len(points), candidate_id,
                                  name, "hole_recovery", "blind_bottom",
                                  settings.stand_off_mm, 2 * feature.radius_mm,
                                  feature.feature_id)
            candidates.append(RecoveryCandidate(candidate_id, name, "blind_bottom",
                                                ViewSweep(candidate_id, name,
                                                          "hole_recovery", 0,
                                                          poses, "darkred"),
                                                feature.feature_id))
            candidate_id += 1
    return tuple(candidates[:maximum_views])


def generate_bottom_sweep(model: CadModel, settings: ViewSweepSettings,
                          sweep_id: int) -> RecoveryCandidate:
    minimum, maximum = _bounds(model)
    dimensions = maximum - minimum
    travel_index = 0 if dimensions[0] >= dimensions[1] else 1
    laser_index = 1 - travel_index
    measurement = np.array([0., 0., 1.])
    laser = np.zeros(3); laser[laser_index] = 1
    travel = np.cross(measurement, laser)
    if travel[travel_index] < 0:
        laser *= -1; travel = np.cross(measurement, laser)
    count = max(2, int(ceil(dimensions[travel_index] / settings.travel_spacing_mm)) + 1)
    values = np.linspace(minimum[travel_index], maximum[travel_index], count)
    surface = 0.5 * (minimum + maximum); surface[2] = minimum[2]
    points, targets = [], []
    for value in values:
        target = surface.copy(); target[travel_index] = value
        points.append(target - settings.stand_off_mm * measurement)
        targets.append(target)
    poses = _motion_poses(points, targets, [laser] * len(points), sweep_id,
                          "bottom", "bottom", "direct_bottom",
                          settings.stand_off_mm,
                          min(settings.laser_width_mm, dimensions[laser_index]))
    return RecoveryCandidate(sweep_id, "bottom", "direct_bottom",
                             ViewSweep(sweep_id, "bottom", "bottom", 0,
                                       poses, "cyan"))


def generate_lower_isometric(model: CadModel, settings: ViewSweepSettings,
                             sweep_id: int, elevation_deg: float = 35.0) -> RecoveryCandidate:
    minimum, maximum = _bounds(model)
    centre = 0.5 * (minimum + maximum)
    radius = 0.5 * np.linalg.norm(maximum - minimum) + settings.stand_off_mm
    elevation = -abs(np.deg2rad(elevation_deg))
    angles = np.linspace(0, 2 * np.pi, 37)
    points, targets, lasers = [], [], []
    for angle in angles:
        radial = np.array([np.cos(elevation) * np.cos(angle),
                           np.cos(elevation) * np.sin(angle), np.sin(elevation)])
        points.append(centre + radius * radial)
        targets.append(centre)
        lasers.append(np.array([-np.sin(angle), np.cos(angle), 0.]))
    poses = _motion_poses(points, targets, lasers, sweep_id, "lower_isometric",
                          "lower_isometric", "lower_isometric_orbit",
                          settings.stand_off_mm, settings.laser_width_mm)
    return RecoveryCandidate(sweep_id, "lower_isometric", "lower_isometric_orbit",
                             ViewSweep(sweep_id, "lower_isometric",
                                       "lower_isometric", 0, poses, "purple"))


def generate_lower_diagonal_sweeps(
    model: CadModel, settings: ViewSweepSettings, sweep_id_start: int,
    elevation_deg: float = 35.0,
) -> tuple[RecoveryCandidate, RecoveryCandidate]:
    minimum, maximum = _bounds(model)
    centre = 0.5 * (minimum + maximum)
    diagonal = 0.5 * float(np.linalg.norm(maximum - minimum)) + settings.stand_off_mm
    output = []
    for index, azimuth in enumerate((45.0, 225.0)):
        azimuth_rad = np.deg2rad(azimuth)
        elevation = -abs(np.deg2rad(elevation_deg))
        outward = np.array([np.cos(elevation) * np.cos(azimuth_rad),
                            np.cos(elevation) * np.sin(azimuth_rad), np.sin(elevation)])
        midpoint = centre + diagonal * outward
        travel = np.array([-np.sin(azimuth_rad), np.cos(azimuth_rad), 0.0])
        values = np.linspace(-0.5 * np.linalg.norm(maximum[:2] - minimum[:2]),
                             0.5 * np.linalg.norm(maximum[:2] - minimum[:2]), 21)
        points = [midpoint + value * travel for value in values]
        targets = [centre + value * travel for value in values]
        name = f"lower_diagonal_{index + 1}"
        poses = _motion_poses(
            points, targets, [travel] * len(points), sweep_id_start + index,
            name, "lower_isometric", "opposed_lower_diagonal",
            settings.stand_off_mm, settings.laser_width_mm,
        )
        output.append(RecoveryCandidate(
            sweep_id_start + index, name, "opposed_lower_diagonal",
            ViewSweep(sweep_id_start + index, name, "lower_isometric", index,
                      poses, "mediumpurple"),
        ))
    return tuple(output)


def candidate_accessible(candidate: RecoveryCandidate, model: CadModel,
                         fixture_path: Path | None = None,
                         bottom_clearance_mm: float = 0.0) -> bool:
    minimum, maximum = _bounds(model)
    clearance_min, clearance_max = minimum.copy(), maximum.copy()
    clearance_min -= bottom_clearance_mm
    clearance_max += bottom_clearance_mm
    for pose in candidate.sweep.scan_poses:
        if np.all(pose.position > clearance_min) and np.all(pose.position < clearance_max):
            return False
    if fixture_path is not None:
        fixture = trimesh.load_mesh(fixture_path, process=False)
        bounds = np.asarray(fixture.bounds)
        for pose in candidate.sweep.poses:
            if np.all(pose.position >= bounds[0]) and np.all(pose.position <= bounds[1]):
                return False
        # Reject both scanner path crossings and fixture-blocked measurement
        # rays. Intersections beyond the finite target/segment are ignored.
        segments = [
            (first.position, second.position)
            for first, second in zip(candidate.sweep.poses, candidate.sweep.poses[1:])
        ]
        segments.extend(
            (pose.position, pose.surface_point) for pose in candidate.sweep.scan_poses
        )
        for start, end in segments:
            vector = np.asarray(end) - np.asarray(start)
            length = float(np.linalg.norm(vector))
            if length <= 1.0e-9:
                continue
            locations, _, _ = fixture.ray.intersects_location(
                np.asarray([start]), np.asarray([vector / length]),
                multiple_hits=False,
            )
            if len(locations) and np.linalg.norm(locations[0] - start) <= length:
                return False
    return True
