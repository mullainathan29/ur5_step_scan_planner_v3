"""Qt-embedded PyVista renderer with no in-scene UI widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv
from trimesh import proximity
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from coverage import coverage_rgb


LAYER_DEFAULTS = {
    "cad": True, "coverage": True, "nominal": True, "scanner": True,
    "transition": False, "connectors": False, "frames": False,
    "laser": False, "labels": False, "recovery": True, "focused": False,
    "features": True,
    "mission_complete": True,
    "mission_scan": True,
    "mission_free_travel": True,
    "mission_start_end": True,
    "mission_transition_waypoints": True,
    "mission_original_safe_polyline": False,
    "mission_smoothing_control_points": False,
    "mission_rejected_smoothing": False,
    "safety_envelope": True,
    "rejected_direct_transitions": False,
    "min_clearance_markers": False,
}

DEV_LAYERS = {
    "nominal", "scanner", "transition", "connectors", "frames",
    "laser", "labels", "recovery", "focused", "features",
    "mission_complete", "mission_transition_waypoints",
    "mission_original_safe_polyline", "mission_smoothing_control_points",
    "mission_rejected_smoothing", "safety_envelope",
    "rejected_direct_transitions", "min_clearance_markers"
}


@dataclass
class ActorRecord:
    actor: Any
    layer: str
    region_id: int | None = None
    face_ids: frozenset[int] = frozenset()
    surface_type: str | None = None
    hole_id: int = -1


class LayerController:
    """Pure visibility/filter state, independently testable from VTK."""

    def __init__(self) -> None:
        self.records: list[ActorRecord] = []
        self.layer_visibility = dict(LAYER_DEFAULTS)
        self.region_id: int | None = None
        self.face_id: int | None = None
        self.surface_type: str | None = None
        self.only_selected_region = False
        self.selected_hole_id = -1
        self.developer_mode = True

    def add(self, record: ActorRecord) -> None:
        self.records.append(record)

    def refresh(self) -> None:
        complete_visible = self.layer_visibility.get("mission_complete", True)
        for record in self.records:
            if record.layer in DEV_LAYERS and not self.developer_mode:
                record.actor.SetVisibility(False)
                continue

            visible = self.layer_visibility.get(record.layer, True)
            if record.layer.startswith("mission_"):
                if self.developer_mode:
                    visible = visible and complete_visible
            if record.region_id is not None:
                if self.only_selected_region and self.region_id is not None:
                    visible = visible and record.region_id == self.region_id
                if self.face_id is not None:
                    visible = visible and self.face_id in record.face_ids
                if self.surface_type is not None:
                    visible = visible and record.surface_type == self.surface_type
            if (record.hole_id >= 0 and self.selected_hole_id >= 0
                    and record.layer in {"focused", "recovery"}):
                visible = visible and record.hole_id == self.selected_hole_id
            record.actor.SetVisibility(bool(visible))

    def set_layer(self, layer: str, visible: bool) -> None:
        self.layer_visibility[layer] = visible
        self.refresh()

    def set_filter(self, region_id=None, face_id=None, surface_type=None,
                   only_selected_region=False) -> None:
        self.region_id = region_id
        self.face_id = face_id
        self.surface_type = surface_type
        self.only_selected_region = only_selected_region
        self.refresh()


def _mesh_polydata(model, coverage=None) -> pv.PolyData:
    mesh = model.validation_mesh.mesh
    cells = np.column_stack((np.full(len(mesh.faces), 3, dtype=np.int64), mesh.faces)).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices), cells)
    if coverage is not None:
        poly.cell_data["coverage_rgb"] = coverage_rgb(coverage)
    return poly


def _line_polydata(lines) -> pv.PolyData:
    points, cells = [], []
    for first, second in lines:
        start = len(points)
        points.extend((first, second))
        cells.append([2, start, start + 1])
    poly = pv.PolyData()
    if cells:
        poly.points = np.asarray(points)
        poly.lines = np.asarray(cells, dtype=np.int64).ravel()
    return poly


def _pyvista_bounds(mesh) -> tuple[float, float, float, float, float, float]:
    bounds = np.asarray(mesh.bounds, dtype=float)
    return (
        float(bounds[0, 0]), float(bounds[1, 0]),
        float(bounds[0, 1]), float(bounds[1, 1]),
        float(bounds[0, 2]), float(bounds[1, 2]),
    )


class ViewerWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.interactor = QtInteractor(self)
        layout.addWidget(self.interactor)
        self.developer_mode = False
        self.controller = LayerController()
        self.controller.developer_mode = self.developer_mode
        self._cad_bounds = None
        self._hole_actors = {}
        self.interactor.add_axes()
        self.interactor.show_grid()

        # Persistent actor references
        self._cad_actors = []
        self._coverage_actors = []
        self._scan_path_actors = []
        self._free_travel_actors = []
        self._recovery_path_actors = []
        self._safety_envelope_actors = []
        self._detected_holes_actors = []
        self._connector_actors = []
        self._frame_actors = []
        self._laser_actors = []
        self._label_actors = []
        self._rejected_line_actors = []
        self._min_clearance_actors = []

    def clear(self) -> None:
        self.interactor.clear()
        self.controller = LayerController()
        self.controller.developer_mode = self.developer_mode
        self._cad_bounds = None
        self._hole_actors = {}
        self._cad_actors = []
        self._coverage_actors = []
        self._scan_path_actors = []
        self._free_travel_actors = []
        self._recovery_path_actors = []
        self._safety_envelope_actors = []
        self._detected_holes_actors = []
        self._connector_actors = []
        self._frame_actors = []
        self._laser_actors = []
        self._label_actors = []
        self._rejected_line_actors = []
        self._min_clearance_actors = []

    def _remove_actors(self, actor_list) -> None:
        for actor in actor_list:
            try:
                self.interactor.remove_actor(actor)
            except Exception:
                pass
        # Remove from controller records
        set_to_remove = set(actor_list)
        self.controller.records = [r for r in self.controller.records if r.actor not in set_to_remove]
        actor_list.clear()

    def show_model(self, model) -> None:
        self.clear()
        poly = _mesh_polydata(model)
        actor = self.interactor.add_mesh(poly, color="lightgray", opacity=0.8)
        self.controller.add(ActorRecord(actor, "cad"))
        self._cad_actors.append(actor)
        self._cad_bounds = _pyvista_bounds(model.validation_mesh.mesh)
        self.fit_cad()

    def show_result(self, result, config: PipelineConfig) -> None:
        # 9. Ensure show_result() rejects incomplete/validation-only results before clearing existing actors.
        if result is None or (not result.poses and not result.sweeps and not result.planar_plans and not result.cylindrical_plans and not result.conical_plans and not result.spherical_plans):
            print("[Viewer] show_result rejected incomplete/validation-only result")
            return

        self.clear()
        model = result.model
        poly = _mesh_polydata(model, result.coverage)
        self._cad_bounds = _pyvista_bounds(model.validation_mesh.mesh)
        
        # 1. CAD
        cad_actor = self.interactor.add_mesh(poly, color="lightgray", opacity=0.35)
        self.controller.add(ActorRecord(cad_actor, "cad"))
        self._cad_actors.append(cad_actor)

        # 2. Coverage
        if result.coverage is not None:
            cov_actor = self.interactor.add_mesh(poly, scalars="coverage_rgb", rgb=True, opacity=0.78)
            self.controller.add(ActorRecord(cov_actor, "coverage"))
            self._coverage_actors.append(cov_actor)

        # 3 & 4. Free travel & Scan from continuous mission (from result.poses)
        is_continuous = len(result.poses) > 0 and all(p.global_pose_id >= 0 for p in result.poses)
        if is_continuous:
            scan_segments = []
            free_travel_segments = []
            
            current_segment = []
            current_is_scan = None
            
            for pose in result.poses:
                is_scan = (pose.scan_status == -1)
                if current_is_scan is None:
                    current_is_scan = is_scan
                    current_segment = [pose]
                elif is_scan == current_is_scan:
                    current_segment.append(pose)
                else:
                    if current_is_scan:
                        scan_segments.append([p.position for p in current_segment])
                        current_segment = [current_segment[-1], pose]
                    else:
                        current_segment.append(pose)
                        free_travel_segments.append([p.position for p in current_segment])
                        current_segment = [pose]
                    current_is_scan = is_scan
            
            if current_segment:
                if current_is_scan:
                    scan_segments.append([p.position for p in current_segment])
                else:
                    free_travel_segments.append([p.position for p in current_segment])

            # 3. Free travel
            for seg in free_travel_segments:
                if len(seg) >= 2:
                    actor = self.interactor.add_mesh(
                        pv.lines_from_points(np.asarray(seg)),
                        color="cyan",
                        line_width=3,
                        opacity=0.8,
                        render_lines_as_tubes=True,
                    )
                    self.controller.add(ActorRecord(actor, "mission_free_travel"))
                    self._free_travel_actors.append(actor)

            # 4. Scan
            for seg in scan_segments:
                if len(seg) >= 2:
                    actor = self.interactor.add_mesh(
                        pv.lines_from_points(np.asarray(seg)),
                        color="darkblue",
                        line_width=5,
                        opacity=1.0,
                        render_lines_as_tubes=True,
                    )
                    self.controller.add(ActorRecord(actor, "mission_scan"))
                    self._scan_path_actors.append(actor)

            # 5. Start/end markers
            start_pos = np.asarray(result.poses[0].position)
            end_pos = np.asarray(result.poses[-1].position)
            
            start_sphere = pv.Sphere(radius=4.0, center=start_pos)
            end_sphere = pv.Sphere(radius=4.0, center=end_pos)
            
            actor_start = self.interactor.add_mesh(start_sphere, color="green", opacity=1.0)
            actor_end = self.interactor.add_mesh(end_sphere, color="red", opacity=1.0)
            
            self.controller.add(ActorRecord(actor_start, "mission_start_end"))
            self.controller.add(ActorRecord(actor_end, "mission_start_end"))
            self._free_travel_actors.append(actor_start)
            self._free_travel_actors.append(actor_end)

        # 6. Developer Overlays

        # Detected Holes
        for feature in result.hole_features:
            marker = pv.Sphere(
                radius=max(feature.radius_mm * 1.05, 0.5),
                center=feature.entry_opening_centre,
                theta_resolution=24, phi_resolution=12,
            )
            actor = self.interactor.add_mesh(
                marker, color="yellow", opacity=0.18, style="wireframe",
                line_width=2,
            )
            self._hole_actors[feature.feature_id] = actor
            self.controller.add(ActorRecord(actor, "features"))
            self._detected_holes_actors.append(actor)

        # Sweeps
        for sweep in result.sweeps:
            scan = sweep.scan_poses
            if not scan:
                continue
            metadata = dict(region_id=sweep.sweep_id, face_ids=frozenset(),
                            surface_type=sweep.view_type)
            nominal = np.asarray([pose.surface_point for pose in scan])
            scanner = np.asarray([pose.position for pose in scan])
            transition = np.asarray([pose.position for pose in sweep.poses])
            path_layer = "recovery" if sweep.view_type in {
                "focused_conical_orbit", "hole_recovery", "lower_isometric", "bottom"
            } else "scanner"
            for layer, points, width in (
                ("nominal", nominal, 4), ("scanner", scanner, 2),
                ("transition", transition, 1),
            ):
                if layer == "scanner":
                    layer = path_layer
                if len(points) >= 2:
                    actor = self.interactor.add_mesh(
                        pv.lines_from_points(points), color=sweep.colour,
                        line_width=width,
                    )
                    self.controller.add(ActorRecord(actor, layer, **metadata))
                    # Note: for developer planning artefacts we store them in _recovery_path_actors or similar to keep _scan_path_actors strictly for the continuous mission
                    if layer == "nominal":
                        self._recovery_path_actors.append(actor)
                    elif layer in ("scanner", "recovery"):
                        self._recovery_path_actors.append(actor)
                    elif layer == "transition":
                        self._recovery_path_actors.append(actor)

        # Focused conical orbits
        feature_by_id = {feature.feature_id: feature for feature in result.hole_features}
        palette = ("turquoise", "coral", "violet", "gold", "lime", "deepskyblue")
        focused = [candidate for candidate in result.recovery_candidates
                   if candidate.strategy == "focused_conical_orbit"]
        for candidate in focused:
            scan = candidate.sweep.scan_poses
            feature = feature_by_id.get(candidate.target_hole_id)
            if not scan or feature is None:
                continue
            color = palette[feature.feature_id % len(palette)]
            points = np.asarray([pose.position for pose in scan])
            actor = self.interactor.add_mesh(
                pv.lines_from_points(points), color=color, line_width=4,
            )
            self.controller.add(ActorRecord(actor, "focused", hole_id=feature.feature_id))
            self._recovery_path_actors.append(actor)

            axis = -np.asarray(feature.entry_opening_normal, dtype=float)
            axis /= np.linalg.norm(axis)
            axis_length = max(feature.depth_mm, 2.0 * feature.radius_mm)
            axis_actor = self.interactor.add_arrows(
                np.asarray([feature.entry_opening_centre]),
                np.asarray([axis * axis_length]), color=color,
            )
            self.controller.add(ActorRecord(axis_actor, "focused", hole_id=feature.feature_id))
            self._recovery_path_actors.append(axis_actor)

            reference = np.array([1.0, 0.0, 0.0])
            if abs(float(np.dot(reference, axis))) > 0.9:
                reference = np.array([0.0, 1.0, 0.0])
            e1 = reference - np.dot(reference, axis) * axis
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(axis, e1)
            circle_angles = np.linspace(0.0, 2.0 * np.pi, 49)
            circle = np.asarray([
                feature.entry_opening_centre + feature.radius_mm
                * (np.cos(angle) * e1 + np.sin(angle) * e2)
                for angle in circle_angles
            ])
            circle_actor = self.interactor.add_mesh(
                pv.lines_from_points(circle), color=color, line_width=2,
            )
            self.controller.add(ActorRecord(circle_actor, "focused", hole_id=feature.feature_id))
            self._recovery_path_actors.append(circle_actor)

            focus = np.asarray(scan[0].focus_point)
            focus_actor = self.interactor.add_mesh(
                pv.Sphere(radius=max(feature.radius_mm * 0.12, 0.25), center=focus),
                color=color,
            )
            self.controller.add(ActorRecord(focus_actor, "focused", hole_id=feature.feature_id))
            self._recovery_path_actors.append(focus_actor)

            connectors = [(pose.position, focus)
                          for pose in scan[::max(1, config.laser_footprint_stride)]]
            if connectors:
                connector_actor = self.interactor.add_mesh(
                    _line_polydata(connectors), color=color, line_width=1,
                )
                self.controller.add(ActorRecord(connector_actor, "focused", hole_id=feature.feature_id))
                self._recovery_path_actors.append(connector_actor)

        # Plan motion segments
        plans = (*result.planar_plans, *result.cylindrical_plans,
                 *result.conical_plans, *result.spherical_plans)
        for plan in plans:
            surface = (
                "plane" if hasattr(plan, "raster_segments") else
                "cone" if plan.__class__.__name__ == "ConicalPlan" else
                "sphere" if plan.__class__.__name__ == "SphericalPlan" else "cylinder"
            )
            for motion in plan.motion_segments:
                scan = motion.scan_poses
                if not scan:
                    continue
                metadata = dict(region_id=scan[0].region_id,
                                face_ids=frozenset(scan[0].member_face_ids),
                                surface_type=surface)
                nominal = np.asarray([sample.point for sample in motion.raster_segment.samples])
                scanner = np.asarray([pose.position for pose in scan])
                transition = np.asarray([pose.position for pose in motion.poses])
                for layer, points, color, width in (
                    ("nominal", nominal, "blue", 3),
                    ("scanner", scanner, "orange", 2),
                    ("transition", transition, "gray", 1),
                ):
                    if len(points) >= 2:
                        actor = self.interactor.add_mesh(
                            pv.lines_from_points(points), color=color, line_width=width
                        )
                        self.controller.add(ActorRecord(actor, layer, **metadata))
                        self._recovery_path_actors.append(actor)

        if is_continuous:
            for polyline in result.summary.output_files.get("original_safe_polylines", []):
                points = np.asarray(polyline)
                if len(points) >= 2:
                    actor = self.interactor.add_mesh(
                        pv.lines_from_points(points),
                        color="moccasin",
                        line_width=1,
                        opacity=0.45,
                    )
                    self.controller.add(ActorRecord(actor, "mission_original_safe_polyline"))
                    self._recovery_path_actors.append(actor)

            for candidate in result.summary.output_files.get("rejected_smoothing_candidates", []):
                points = np.asarray(candidate)
                if len(points) >= 2:
                    actor = self.interactor.add_mesh(
                        pv.lines_from_points(points),
                        color="crimson",
                        line_width=1,
                        opacity=0.35,
                    )
                    self.controller.add(ActorRecord(actor, "mission_rejected_smoothing"))
                    self._recovery_path_actors.append(actor)

            control_points = result.summary.output_files.get("smoothing_control_points", [])
            if control_points:
                actor = self.interactor.add_mesh(
                    pv.PolyData(np.asarray(control_points)),
                    color="white",
                    point_size=5,
                    render_points_as_spheres=True,
                    opacity=0.7,
                )
                self.controller.add(ActorRecord(actor, "mission_smoothing_control_points"))
                self._recovery_path_actors.append(actor)
                    
            transition_pts = [p.position for p in result.poses if p.scan_status == 0]
            if transition_pts:
                waypoints_poly = pv.PolyData(np.asarray(transition_pts))
                actor_wps = self.interactor.add_mesh(
                    waypoints_poly,
                    color="yellow",
                    point_size=6,
                    render_points_as_spheres=True,
                    opacity=0.8,
                )
                self.controller.add(ActorRecord(actor_wps, "mission_transition_waypoints"))
                self._recovery_path_actors.append(actor_wps)

        # Call update_validation to render all validation-specific actors
        self.update_validation(result, config)

        self.controller.refresh()
        self.fit_trajectory()

    def update_validation(self, result, config: PipelineConfig) -> None:
        self._remove_actors(self._connector_actors)
        self._remove_actors(self._frame_actors)
        self._remove_actors(self._laser_actors)
        self._remove_actors(self._label_actors)
        self._remove_actors(self._rejected_line_actors)
        self._remove_actors(self._min_clearance_actors)
        self._remove_actors(self._safety_envelope_actors)

        model = result.model
        scan_poses = [pose for pose in result.poses if pose.scan_enabled]
        
        # 2. Render Connector lines
        metrics = {(m.segment_id, m.pose_id): m for m in
                   (result.geometry_validation.pose_metrics if result.geometry_validation else ())}
        colors = {"green": [30, 190, 60], "yellow": [245, 200, 35], "red": [220, 35, 35]}
        connector_lines = [(p.surface_point, p.position) for p in scan_poses[::config.connector_stride]]
        if connector_lines:
            connector_poly = _line_polydata(connector_lines)
            connector_poly.cell_data["rgb"] = np.asarray([
                colors[metrics[(p.segment_id, p.pose_id)].connector_status.value]
                if (p.segment_id, p.pose_id) in metrics else colors["green"]
                for p in scan_poses[::config.connector_stride]
            ], dtype=np.uint8)
            actor = self.interactor.add_mesh(
                connector_poly, scalars="rgb", rgb=True, line_width=1
            )
            self.controller.add(ActorRecord(actor, "connectors"))
            self._connector_actors.append(actor)

        # 3. Render Frames
        selected = scan_poses[::config.frame_stride]
        if selected:
            origins = np.asarray([p.position for p in selected])
            rotations = [Rotation.from_quat(p.quaternion_xyzw) for p in selected]
            length = max(float(np.max(model.validation_mesh.mesh.extents)) * 0.015, 1.0)
            for axis, color in (([1, 0, 0], "red"), ([0, 1, 0], "green"), ([0, 0, 1], "blue")):
                vectors = np.asarray([rotation.apply(axis) for rotation in rotations])
                actor = self.interactor.add_arrows(origins, vectors * length, color=color)
                self.controller.add(ActorRecord(actor, "frames"))
                self._frame_actors.append(actor)
                
        # 4. Render Laser footprints and Face labels
        if config.planning_quality != "fast_preview":
            footprints = []
            for pose in scan_poses[::config.laser_footprint_stride]:
                axis = Rotation.from_quat(pose.quaternion_xyzw).apply([1, 0, 0])
                half = 0.5 * config.laser_width_mm * axis
                footprints.append((pose.surface_point - half, pose.surface_point + half))
            if footprints:
                actor = self.interactor.add_mesh(
                    _line_polydata(footprints), color="cyan", line_width=1
                )
                self.controller.add(ActorRecord(actor, "laser"))
                self._laser_actors.append(actor)
                
            points = np.asarray([[f.center.x, f.center.y, f.center.z] for f in model.faces])
            label_actor = self.interactor.add_point_labels(
                points, [str(f.face_id) for f in model.faces], font_size=8,
                point_size=2, shape_opacity=0.25, always_visible=False)
            self.controller.add(ActorRecord(label_actor, "labels"))
            self._label_actors.append(label_actor)

        is_continuous = len(result.poses) > 0 and all(p.global_pose_id >= 0 for p in result.poses)
        if is_continuous:
            # 5. Render Safety Envelope Box
            if result.summary.output_files and "safety_box" in result.summary.output_files:
                safety_box = result.summary.output_files["safety_box"]
                box_vertices = safety_box.vertices
                box_faces = safety_box.faces
                pv_faces = np.column_stack([np.full(len(box_faces), 3), box_faces]).ravel()
                box_poly = pv.PolyData(box_vertices, pv_faces)
                
                box_actor = self.interactor.add_mesh(
                    box_poly,
                    color="cyan",
                    opacity=0.03, # Reduced opacity to 0.03 to avoid visual occlusion
                    style="surface",
                    show_edges=True,
                    edge_color="cyan",
                    line_width=1.5
                )
                self.controller.add(ActorRecord(box_actor, "safety_envelope"))
                self._safety_envelope_actors.append(box_actor)

            # 6. Render Rejected direct lines
            if result.summary.output_files and "rejected_direct_lines" in result.summary.output_files:
                rejected_lines = result.summary.output_files["rejected_direct_lines"]
                if rejected_lines:
                    rejected_poly = _line_polydata(rejected_lines)
                    rej_actor = self.interactor.add_mesh(
                        rejected_poly,
                        color="red",
                        line_width=2,
                        opacity=0.8
                    )
                    self.controller.add(ActorRecord(rej_actor, "rejected_direct_transitions"))
                    self._rejected_line_actors.append(rej_actor)

            # 7. Render Minimum clearance markers
            free_poses = [p for p in result.poses if p.scan_status == 0]
            if free_poses:
                min_pose = min(free_poses, key=lambda p: getattr(p, "clearance_mm", 999.0))
                p_pos = np.asarray(min_pose.position)
                closest_pts, _, _ = proximity.closest_point(model.validation_mesh.mesh, np.array([p_pos]))
                c_pt = closest_pts[0]
                
                clearance_line = _line_polydata([(p_pos, c_pt)])
                line_actor = self.interactor.add_mesh(
                    clearance_line,
                    color="magenta",
                    line_width=3,
                    opacity=1.0
                )
                self.controller.add(ActorRecord(line_actor, "min_clearance_markers"))
                self._min_clearance_actors.append(line_actor)
                
                sphere_actor = self.interactor.add_mesh(
                    pv.Sphere(radius=2.0, center=c_pt),
                    color="magenta",
                    opacity=1.0
                )
                self.controller.add(ActorRecord(sphere_actor, "min_clearance_markers"))
                self._min_clearance_actors.append(sphere_actor)
                
                clearance_val = getattr(min_pose, "clearance_mm", 0.0)
                label_actor = self.interactor.add_point_labels(
                    np.array([0.5 * (p_pos + c_pt)]),
                    [f"Min Clearance: {clearance_val:.1f} mm"],
                    font_size=10,
                    text_color="magenta",
                    always_visible=True
                )
                self.controller.add(ActorRecord(label_actor, "min_clearance_markers"))
                self._min_clearance_actors.append(label_actor)

    def set_layer_visibility(self, layer: str, visible: bool) -> None:
        self.controller.set_layer(layer, visible)
        self.interactor.render()

    def set_selection(self, region_id=None, face_id=None, surface_type=None,
                      only_selected_region=False) -> None:
        self.controller.set_filter(region_id, face_id, surface_type,
                                   only_selected_region)
        self.interactor.render()

    def fit_cad(self) -> None:
        if self._cad_bounds is not None:
            self.interactor.reset_camera(bounds=self._cad_bounds)
            self.interactor.render()

    def fit_trajectory(self) -> None:
        self.interactor.reset_camera()
        self.interactor.render()

    def reset_camera(self) -> None:
        self.interactor.view_isometric()
        self.fit_trajectory()

    def select_hole(self, feature_id: int) -> None:
        self.controller.selected_hole_id = feature_id
        for hole_id, actor in self._hole_actors.items():
            actor.GetProperty().SetColor(
                (1.0, 0.1, 0.1) if hole_id == feature_id else (1.0, 1.0, 0.0)
            )
            actor.GetProperty().SetLineWidth(5 if hole_id == feature_id else 2)
        self.controller.refresh()
        self.interactor.render()

    def set_developer_mode(self, enabled: bool) -> None:
        self.developer_mode = enabled
        self.controller.developer_mode = enabled
        self.controller.refresh()
        self.interactor.render()
