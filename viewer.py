"""Layered PyVista viewer for the unified B-Rep pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from app.result import PipelineResult
from coverage import coverage_rgb


@dataclass(frozen=True)
class ViewerGeometry:
    connector_pose_indices: tuple[int, ...]
    frame_pose_indices: tuple[int, ...]
    laser_footprint_pose_indices: tuple[int, ...]


def select_viewer_geometry(result: PipelineResult, config: PipelineConfig) -> ViewerGeometry:
    scan_indices = tuple(
        index for index, pose in enumerate(result.poses) if pose.scan_enabled
    )
    return ViewerGeometry(
        connector_pose_indices=scan_indices[:: config.connector_stride],
        frame_pose_indices=scan_indices[:: config.frame_stride],
        laser_footprint_pose_indices=scan_indices[
            :: config.laser_footprint_stride
        ],
    )


def _line_polydata(lines: list[tuple[np.ndarray, np.ndarray]]) -> pv.PolyData:
    points: list[np.ndarray] = []
    cells: list[list[int]] = []
    for first, second in lines:
        start = len(points)
        points.extend((first, second))
        cells.append([2, start, start + 1])
    poly = pv.PolyData()
    if cells:
        poly.points = np.asarray(points)
        poly.lines = np.asarray(cells, dtype=np.int64).ravel()
    return poly


def show_pipeline(result: PipelineResult, config: PipelineConfig) -> None:
    model = result.model
    mesh = model.validation_mesh.mesh
    cells = np.column_stack(
        (np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces)
    ).ravel()
    coverage_poly = pv.PolyData(np.asarray(mesh.vertices), cells)
    coverage_poly.cell_data["coverage_rgb"] = coverage_rgb(result.coverage)
    geometry = select_viewer_geometry(result, config)

    plotter = pv.Plotter()
    actors: dict[str, list] = {name: [] for name in (
        "cad", "coverage", "nominal", "scanner", "transition",
        "connectors", "frames", "laser", "labels"
    )}
    actors["cad"].append(
        plotter.add_mesh(coverage_poly, color="lightgray", opacity=0.35)
    )
    actors["coverage"].append(
        plotter.add_mesh(
            coverage_poly,
            scalars="coverage_rgb",
            rgb=True,
            opacity=0.78,
        )
    )

    actor_meta: list[tuple[object, int, set[int], str]] = []
    for sweep in result.sweeps:
        scan = sweep.scan_poses
        if not scan:
            continue
        nominal = np.asarray([pose.surface_point for pose in scan])
        scanner = np.asarray([pose.position for pose in scan])
        transition = np.asarray([pose.position for pose in sweep.poses])
        for layer, points, width in (
            ("nominal", nominal, 4), ("scanner", scanner, 2),
            ("transition", transition, 1),
        ):
            if len(points) >= 2:
                actor = plotter.add_mesh(
                    pv.lines_from_points(points), color=sweep.colour,
                    line_width=width,
                )
                if layer == "transition":
                    actor.SetVisibility(False)
                actors[layer].append(actor)
    for plan in (*result.planar_plans, *result.cylindrical_plans, *result.conical_plans, *result.spherical_plans):
        for motion in plan.motion_segments:
            scan = motion.scan_poses
            if not scan:
                continue
            region_id = scan[0].region_id
            face_ids = set(scan[0].member_face_ids)
            surface_type = (
                "plane" if hasattr(plan, "raster_segments") else
                "cone" if plan.__class__.__name__ == "ConicalPlan" else
                "sphere" if plan.__class__.__name__ == "SphericalPlan" else
                "cylinder"
            )
            nominal = np.asarray(
                [sample.point for sample in motion.raster_segment.samples]
            )
            scanner = np.asarray([pose.position for pose in scan])
            if len(nominal) >= 2:
                actor = plotter.add_mesh(
                    pv.lines_from_points(nominal), color="blue", line_width=3
                )
                actors["nominal"].append(actor)
                actor_meta.append((actor, region_id, face_ids, surface_type))
            if len(scanner) >= 2:
                actor = plotter.add_mesh(
                    pv.lines_from_points(scanner), color="orange", line_width=2
                )
                actors["scanner"].append(actor)
                actor_meta.append((actor, region_id, face_ids, surface_type))
            transition = np.asarray([pose.position for pose in motion.poses])
            if len(transition) >= 2:
                actor = plotter.add_mesh(
                    pv.lines_from_points(transition),
                    color="gray",
                    line_width=1,
                )
                actor.SetVisibility(False)
                actors["transition"].append(actor)
                actor_meta.append((actor, region_id, face_ids, surface_type))

    connector_lines = []
    connector_colors = []
    color_map = {
        "green": [30, 190, 60],
        "yellow": [245, 200, 35],
        "red": [220, 35, 35],
    }
    metric_by_key = {}
    if result.geometry_validation:
        metric_by_key = {
            (m.segment_id, m.pose_id): m
            for m in result.geometry_validation.pose_metrics
        }
    for index in geometry.connector_pose_indices:
        pose = result.poses[index]
        connector_lines.append((pose.surface_point, pose.position))
        metric = metric_by_key.get((pose.segment_id, pose.pose_id))
        connector_colors.append(
            color_map[metric.connector_status.value if metric else "green"]
        )
    connector_poly = _line_polydata(connector_lines)
    if connector_lines:
        connector_poly.cell_data["rgb"] = np.asarray(
            connector_colors, dtype=np.uint8
        )
        actor = plotter.add_mesh(
            connector_poly, scalars="rgb", rgb=True, line_width=1
        )
        actor.SetVisibility(config.show_connectors)
        actors["connectors"].append(actor)

    frame_length = max(float(np.max(mesh.extents)) * 0.015, 1.0)
    frame_poses = [result.poses[i] for i in geometry.frame_pose_indices]
    if frame_poses:
        origins = np.asarray([pose.position for pose in frame_poses])
        rotations = [Rotation.from_quat(p.quaternion_xyzw) for p in frame_poses]
        for axis, color in (([1, 0, 0], "red"), ([0, 1, 0], "green"), ([0, 0, 1], "blue")):
            vectors = np.asarray([rotation.apply(axis) for rotation in rotations])
            actor = plotter.add_arrows(origins, vectors * frame_length, color=color)
            actor.SetVisibility(False)
            actors["frames"].append(actor)

    footprint_lines = []
    for index in geometry.laser_footprint_pose_indices:
        pose = result.poses[index]
        laser = Rotation.from_quat(pose.quaternion_xyzw).apply([1, 0, 0])
        half = 0.5 * config.laser_width_mm * laser
        footprint_lines.append((pose.surface_point - half, pose.surface_point + half))
    if footprint_lines:
        actor = plotter.add_mesh(
            _line_polydata(footprint_lines), color="cyan", line_width=1
        )
        actor.SetVisibility(False)
        actors["laser"].append(actor)

    label_points = np.asarray([
        [record.center.x, record.center.y, record.center.z]
        for record in model.faces
    ])
    label_actor = plotter.add_point_labels(
        label_points,
        [str(record.face_id) for record in model.faces],
        font_size=8,
        point_size=2,
        shape_opacity=0.25,
        always_visible=False,
    )
    label_actor.SetVisibility(config.show_face_labels)
    actors["labels"].append(label_actor)

    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title="Unified STEP B-Rep Scan Planner")
