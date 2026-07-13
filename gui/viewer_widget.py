"""Qt-embedded PyVista renderer with no in-scene UI widgets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtWidgets
from scipy.spatial.transform import Rotation

from app.config import PipelineConfig
from coverage import coverage_rgb


LAYER_DEFAULTS = {
    "cad": True, "coverage": True, "nominal": True, "scanner": True,
    "transition": False, "connectors": False, "frames": False,
    "laser": False, "labels": False,
    "features": True,
}


@dataclass
class ActorRecord:
    actor: Any
    layer: str
    region_id: int | None = None
    face_ids: frozenset[int] = frozenset()
    surface_type: str | None = None


class LayerController:
    """Pure visibility/filter state, independently testable from VTK."""

    def __init__(self) -> None:
        self.records: list[ActorRecord] = []
        self.layer_visibility = dict(LAYER_DEFAULTS)
        self.region_id: int | None = None
        self.face_id: int | None = None
        self.surface_type: str | None = None
        self.only_selected_region = False

    def add(self, record: ActorRecord) -> None:
        self.records.append(record)

    def refresh(self) -> None:
        for record in self.records:
            visible = self.layer_visibility.get(record.layer, True)
            if record.region_id is not None:
                if self.only_selected_region and self.region_id is not None:
                    visible = visible and record.region_id == self.region_id
                if self.face_id is not None:
                    visible = visible and self.face_id in record.face_ids
                if self.surface_type is not None:
                    visible = visible and record.surface_type == self.surface_type
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
        self.controller = LayerController()
        self._cad_bounds = None
        self._hole_actors = {}
        self.interactor.add_axes()
        self.interactor.show_grid()

    def clear(self) -> None:
        self.interactor.clear()
        self.controller = LayerController()
        self._cad_bounds = None
        self._hole_actors = {}

    def show_model(self, model) -> None:
        self.clear()
        poly = _mesh_polydata(model)
        actor = self.interactor.add_mesh(poly, color="lightgray", opacity=0.8)
        self.controller.add(ActorRecord(actor, "cad"))
        self._cad_bounds = _pyvista_bounds(model.validation_mesh.mesh)
        self.fit_cad()

    def show_result(self, result, config: PipelineConfig) -> None:
        self.clear()
        model = result.model
        poly = _mesh_polydata(model, result.coverage)
        self._cad_bounds = _pyvista_bounds(model.validation_mesh.mesh)
        self.controller.add(ActorRecord(
            self.interactor.add_mesh(poly, color="lightgray", opacity=0.35), "cad"
        ))
        if result.coverage is not None:
            self.controller.add(ActorRecord(
                self.interactor.add_mesh(poly, scalars="coverage_rgb", rgb=True, opacity=0.78),
                "coverage",
            ))
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
        for sweep in result.sweeps:
            scan = sweep.scan_poses
            if not scan:
                continue
            metadata = dict(region_id=sweep.sweep_id, face_ids=frozenset(),
                            surface_type=sweep.view_type)
            nominal = np.asarray([pose.surface_point for pose in scan])
            scanner = np.asarray([pose.position for pose in scan])
            transition = np.asarray([pose.position for pose in sweep.poses])
            for layer, points, width in (
                ("nominal", nominal, 4), ("scanner", scanner, 2),
                ("transition", transition, 1),
            ):
                if len(points) >= 2:
                    actor = self.interactor.add_mesh(
                        pv.lines_from_points(points), color=sweep.colour,
                        line_width=width,
                    )
                    self.controller.add(ActorRecord(actor, layer, **metadata))
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

        scan_poses = [pose for pose in result.poses if pose.scan_enabled]
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
            self.controller.add(ActorRecord(self.interactor.add_mesh(
                connector_poly, scalars="rgb", rgb=True, line_width=1), "connectors"))

        selected = scan_poses[::config.frame_stride]
        if selected:
            origins = np.asarray([p.position for p in selected])
            rotations = [Rotation.from_quat(p.quaternion_xyzw) for p in selected]
            length = max(float(np.max(model.validation_mesh.mesh.extents)) * 0.015, 1.0)
            for axis, color in (([1, 0, 0], "red"), ([0, 1, 0], "green"), ([0, 0, 1], "blue")):
                vectors = np.asarray([rotation.apply(axis) for rotation in rotations])
                self.controller.add(ActorRecord(
                    self.interactor.add_arrows(origins, vectors * length, color=color), "frames"))
        if config.planning_quality != "fast_preview":
            footprints = []
            for pose in scan_poses[::config.laser_footprint_stride]:
                axis = Rotation.from_quat(pose.quaternion_xyzw).apply([1, 0, 0])
                half = 0.5 * config.laser_width_mm * axis
                footprints.append((pose.surface_point - half, pose.surface_point + half))
            if footprints:
                self.controller.add(ActorRecord(self.interactor.add_mesh(
                    _line_polydata(footprints), color="cyan", line_width=1), "laser"))
            points = np.asarray([[f.center.x, f.center.y, f.center.z] for f in model.faces])
            label_actor = self.interactor.add_point_labels(
                points, [str(f.face_id) for f in model.faces], font_size=8,
                point_size=2, shape_opacity=0.25, always_visible=False)
            self.controller.add(ActorRecord(label_actor, "labels"))
        self.controller.refresh()
        self.fit_cad()

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
        for hole_id, actor in self._hole_actors.items():
            actor.GetProperty().SetColor(
                (1.0, 0.1, 0.1) if hole_id == feature_id else (1.0, 1.0, 0.0)
            )
            actor.GetProperty().SetLineWidth(5 if hole_id == feature_id else 2)
        self.interactor.render()
