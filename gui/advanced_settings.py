"""Advanced configuration dialog for technical planning settings."""

from __future__ import annotations

from pathlib import Path

import yaml
from PySide6 import QtWidgets

from app.configuration import PARAMETERS, advanced_summary


ADVANCED_TABS = {
    "Scanner model": (
        ("Planning mode", "planning_mode"),
        ("Surface taxonomy", "surface_types"),
        ("Cylinder strategy", "cylinder_strategy"),
        ("Minimum face area", "minimum_face_area_mm2"),
        ("Laser-line width", "laser_line_width_mm"),
    ),
    "Optical validation": (
        ("Standard laser rays", "standard_laser_ray_count"),
        ("Final laser rays", "final_laser_ray_count"),
        ("Maximum incidence angle", "max_incidence_angle_deg"),
    ),
    "Principal sweeps": (
        ("Lane spacing", "lane_spacing_mm"),
        ("Sweep overlap", "sweep_overlap_fraction"),
        ("Isometric azimuth spacing", "isometric_azimuth_step_deg"),
    ),
    "Hole recovery": (
        ("Hole target coverage", "hole_target_coverage_percent"),
        ("Maximum recovery views", "maximum_views_per_hole"),
        ("Candidate cone angles", "focused_orbit_cone_angles_deg"),
        ("Candidate azimuth count", "hole_candidate_azimuth_count"),
        ("Minimum candidate gain", "minimum_candidate_gain_percent"),
        ("Orbit arc limits", "focused_orbit_minimum_arc_deg"),
        ("Focused orbit maximum arc", "focused_orbit_maximum_arc_deg"),
        ("Focused hole orbit", "enable_focused_hole_orbit"),
    ),
    "Bottom recovery": (
        ("Blind bottom recovery", "enable_blind_hole_bottom_recovery"),
        ("Lower isometric recovery", "enable_lower_isometric_recovery"),
        ("Bottom clearance", "bottom_clearance_mm"),
        ("Maximum lower recovery sweeps", "maximum_lower_recovery_sweeps"),
    ),
    "Mission optimisation": (
        ("Maximum optional sweeps", "maximum_optional_sweeps"),
        ("Total target coverage", "total_target_coverage_percent"),
    ),
    "Safe traverse": (
        ("OBB/AABB choice", "envelope_type"),
        ("Tool-envelope radius", "tool_envelope_radius_mm"),
        ("Vertical clearance", "vertical_clearance_mm"),
        ("Transition graph weights", "corner_penalty"),
        ("Vertical motion penalty", "vertical_motion_penalty"),
        ("Proximity penalty", "proximity_penalty"),
        ("Collision sample spacing", "collision_sampling_interval_mm"),
        ("Enable safe traverse", "enable_safe_traverse"),
        ("Safety clearance", "safety_clearance_mm"),
    ),
    "Free-travel smoothing": (
        ("Spline method", "free_travel_smoothing_method"),
        ("Blend radius", "nominal_blend_radius_mm"),
        ("Minimum blend radius", "minimum_blend_radius_mm"),
        ("Smoothing retries", "maximum_retries"),
        ("Smoothing sample spacing", "sample_spacing_mm"),
        ("Orientation interpolation limits", "orientation_smoothing_maximum_step_deg"),
        ("Interpolation window", "orientation_interpolation_window_fraction"),
    ),
    "Performance": (
        ("Worker count", "worker_count"),
        ("Mesh tolerance", "mesh_tolerance_mm"),
        ("Graph waypoint density", "graph_waypoint_density"),
    ),
    "Viewer": (
        ("Connector stride", "connector_stride"),
        ("Frame stride", "frame_stride"),
        ("Laser footprint stride", "laser_footprint_stride"),
        ("Debug display settings", "debug_display"),
    ),
}


class AdvancedSettingsDialog(QtWidgets.QDialog):
    """Modal editor/inspector for values intentionally hidden from operators."""

    def __init__(self, controls, parent=None) -> None:
        super().__init__(parent)
        self.controls = controls
        self.setWindowTitle("Advanced Settings")
        self.resize(720, 560)
        layout = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        self.field_widgets: dict[str, QtWidgets.QWidget] = {}
        self._build_tabs()
        self.summary = QtWidgets.QPlainTextEdit()
        self.summary.setReadOnly(True)
        self.summary.setPlainText(advanced_summary(controls.planning_config(Path("model.step"))))
        layout.addWidget(self.summary)

        buttons = QtWidgets.QDialogButtonBox()
        self.apply_button = buttons.addButton("Apply", QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole)
        self.reset_button = buttons.addButton("Reset to profile", QtWidgets.QDialogButtonBox.ButtonRole.ResetRole)
        self.save_button = buttons.addButton("Save user overrides", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.cancel_button = buttons.addButton("Cancel", QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(buttons)

        self.apply_button.clicked.connect(self.apply)
        self.reset_button.clicked.connect(self.reset_to_profile)
        self.save_button.clicked.connect(self.save_user_overrides)
        self.cancel_button.clicked.connect(self.reject)

    def _build_tabs(self) -> None:
        for tab_name, rows in ADVANCED_TABS.items():
            page = QtWidgets.QWidget()
            form = QtWidgets.QFormLayout(page)
            for label, field in rows:
                widget = self._widget_for(field)
                widget.setObjectName(field)
                metadata = PARAMETERS.get(field)
                widget.setToolTip(metadata.tooltip if metadata else "Advanced configuration value loaded from YAML.")
                form.addRow(label, widget)
                self.field_widgets[field] = widget
            self.tabs.addTab(page, tab_name)

    def _widget_for(self, field: str) -> QtWidgets.QWidget:
        c = self.controls
        mapping = {
            "planning_mode": c.planning_mode,
            "surface_types": c.surface_types_editor,
            "cylinder_strategy": c.cylinder_strategy,
            "minimum_face_area_mm2": c.minimum_face_area,
            "laser_line_width_mm": c.laser_width,
            "standard_laser_ray_count": c.standard_laser_rays,
            "final_laser_ray_count": c.final_laser_rays,
            "max_incidence_angle_deg": c.max_incidence_angle,
            "lane_spacing_mm": c.lane_spacing,
            "sweep_overlap_fraction": c.overlap,
            "isometric_azimuth_step_deg": c.isometric_azimuth_step,
            "hole_target_coverage_percent": c.hole_target_coverage,
            "maximum_views_per_hole": c.maximum_views_per_hole,
            "focused_orbit_cone_angles_deg": c.candidate_tilt_angles,
            "hole_candidate_azimuth_count": c.candidate_azimuth_count,
            "minimum_candidate_gain_percent": c.minimum_candidate_gain,
            "focused_orbit_minimum_arc_deg": c.focused_orbit_minimum_arc,
            "focused_orbit_maximum_arc_deg": c.focused_orbit_maximum_arc,
            "enable_focused_hole_orbit": c.enable_focused_hole_orbit,
            "enable_blind_hole_bottom_recovery": c.enable_blind_bottom,
            "enable_lower_isometric_recovery": c.enable_lower_isometric,
            "bottom_clearance_mm": c.bottom_clearance,
            "maximum_lower_recovery_sweeps": c.maximum_lower_sweeps,
            "maximum_optional_sweeps": c.maximum_optional_sweeps,
            "total_target_coverage_percent": c.total_target_coverage,
            "envelope_type": c.envelope_type_widget,
            "tool_envelope_radius_mm": c.tool_envelope_radius,
            "vertical_clearance_mm": c.vertical_clearance,
            "corner_penalty": c.corner_penalty,
            "vertical_motion_penalty": c.vertical_motion_penalty,
            "proximity_penalty": c.proximity_penalty,
            "collision_sampling_interval_mm": c.collision_sampling_interval,
            "enable_safe_traverse": c.enable_safe_traverse,
            "safety_clearance_mm": c.safety_clearance,
            "free_travel_smoothing_method": c.smoothing_method_widget,
            "nominal_blend_radius_mm": c.nominal_blend_radius,
            "minimum_blend_radius_mm": c.minimum_blend_radius,
            "maximum_retries": c.maximum_retries,
            "sample_spacing_mm": c.smoothing_sample_spacing,
            "orientation_smoothing_maximum_step_deg": c.orientation_smoothing_max_step,
            "orientation_interpolation_window_fraction": c.orientation_interpolation_window,
            "worker_count": c.worker_count,
            "mesh_tolerance_mm": c.mesh_tolerance,
            "graph_waypoint_density": c.graph_waypoint_density,
            "connector_stride": c.connector_stride,
            "frame_stride": c.frame_stride,
            "laser_footprint_stride": c.laser_stride,
            "debug_display": c.debug_display,
        }
        return mapping[field]

    def apply(self) -> None:
        self.controls.planning_changed.emit()
        self.summary.setPlainText(advanced_summary(self.controls.planning_config(Path("model.step"))))
        self.accept()

    def reset_to_profile(self) -> None:
        self.controls.reset_to_profile(confirm=False)
        self.summary.setPlainText(advanced_summary(self.controls.planning_config(Path("model.step"))))

    def save_user_overrides(self) -> None:
        message = QtWidgets.QMessageBox(self)
        message.setIcon(QtWidgets.QMessageBox.Icon.Question)
        message.setWindowTitle("Save user overrides")
        message.setText("Save current advanced settings to config/user_overrides.yaml?")
        message.setStandardButtons(
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
        )
        if message.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        path = Path(__file__).resolve().parents[1] / "config" / "user_overrides.yaml"
        path.write_text(
            "# User overrides are applied after the selected quality profile.\n"
            "# Written only after explicit confirmation from the Advanced Settings dialog.\n"
            + yaml.safe_dump(self.controls.advanced_overrides(), sort_keys=False),
            encoding="utf-8",
        )
