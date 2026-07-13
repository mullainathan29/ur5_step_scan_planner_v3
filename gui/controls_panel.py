"""Operator-level PySide controls backed by YAML configuration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from app.configuration import pipeline_config_from_yaml, tooltip
from recovery import estimate_candidate_complexity


LAYER_LABELS = {
    "cad": "CAD model",
    "coverage": "Coverage",
    "mission_scan": "Scan path",
    "mission_free_travel": "Free-travel path",
    "recovery": "Recovery paths",
    "safety_envelope": "Safety envelope",
    "features": "Detected holes",
}

DEBUG_LAYER_LABELS = {
    "scanner": "Scanner paths",
    "focused": "Focused hole orbits",
    "mission_complete": "Complete mission",
    "mission_start_end": "Mission start/end",
    "mission_transition_waypoints": "Transition waypoints",
    "mission_original_safe_polyline": "Original safe polyline",
    "mission_smoothing_control_points": "Smoothing control points",
    "mission_rejected_smoothing": "Rejected smoothing candidates",
    "rejected_direct_transitions": "Rejected direct transitions",
    "min_clearance_markers": "Minimum-clearance markers",
    "connectors": "Connector lines",
    "frames": "Scanner frames",
    "laser": "Laser footprints",
    "labels": "Face IDs",
}


class ControlsPanel(QtWidgets.QScrollArea):
    planning_changed = QtCore.Signal()
    selection_changed = QtCore.Signal()
    layer_changed = QtCore.Signal(str, bool)
    stride_changed = QtCore.Signal()
    hole_selection_changed = QtCore.Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._estimated_features = ()
        self._estimated_reports = ()
        self._loading_profile = False
        self._explicit_gui_changes = False

        self.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)

        self._build_model_group(layout)
        self._build_scanner_group(layout)
        self._build_coverage_group(layout)
        self._build_mission_group(layout)
        self._build_status_group(layout)
        self._build_viewer_group(layout)
        self._create_internal_advanced_controls()
        layout.addStretch()

        self.set_export_enabled(False)
        self._apply_tooltips()
        self._connect_signals()
        self._load_profile_values(visible=True)

    def _build_model_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Model")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.upload_button = QtWidgets.QPushButton("Upload STEP")
        self.filename_label = QtWidgets.QLabel("No model loaded")
        self.filename_label.setWordWrap(True)
        self.face_statistics_label = QtWidgets.QLabel("Faces: -")
        self.face_statistics_label.hide()
        self.dimensions_label = QtWidgets.QLabel("Dimensions: -")
        self.planning_quality = QtWidgets.QComboBox()
        for label, value in (
            ("Fast preview", "fast_preview"),
            ("Standard", "standard"),
            ("Final validation", "final_validation"),
        ):
            self.planning_quality.addItem(label, value)
        group_layout.addWidget(self.upload_button)
        row = QtWidgets.QFormLayout()
        row.addRow("Planning quality", self.planning_quality)
        group_layout.addLayout(row)
        group_layout.addWidget(self.filename_label)
        group_layout.addWidget(self.dimensions_label)
        layout.addWidget(group)

    def _build_scanner_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Scanner")
        form = QtWidgets.QFormLayout(group)
        self.stand_off = self._double_spin(310.0, 1.0, 10000.0)
        self.travel_spacing = self._double_spin(5.0, 0.01, 10000.0)
        self.laser_width = self._double_spin(200.0, 0.01, 10000.0)
        form.addRow("Stand-off distance (mm)", self.stand_off)
        form.addRow("Travel spacing (mm)", self.travel_spacing)
        form.addRow("Laser-line width (mm)", self.laser_width)
        layout.addWidget(group)

    def _build_coverage_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Scan Coverage")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.top_sweep = QtWidgets.QCheckBox("Top scan")
        self.side_sweeps = QtWidgets.QCheckBox("Side scans")
        self.isometric_recovery = QtWidgets.QCheckBox("Upper isometric recovery")
        self.bottom_coverage = QtWidgets.QCheckBox("Bottom/underside scan")
        self.enable_hole_recovery = QtWidgets.QCheckBox("Hole recovery")
        self.enable_focused_hole_orbit = QtWidgets.QCheckBox("Focused hole orbit")
        for widget in (
            self.top_sweep,
            self.side_sweeps,
            self.isometric_recovery,
            self.bottom_coverage,
            self.enable_hole_recovery,
            self.enable_focused_hole_orbit,
        ):
            widget.setChecked(True)
            group_layout.addWidget(widget)
        layout.addWidget(group)

        self.hole_tree = QtWidgets.QTreeWidget()
        self.hole_tree.setHeaderLabels(["Hole", "Diameter", "Depth", "Before", "Recovery", "Strategy", "Final"])
        self.hole_tree.hide()
        self.hole_target_coverage = self._double_spin(85.0, 0.0, 100.0)
        self.hole_target_coverage.hide()
        self.candidate_estimate_label = QtWidgets.QLabel("Estimated candidates: 0")
        self.candidate_estimate_label.hide()

    def _build_mission_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Mission")
        group_layout = QtWidgets.QVBoxLayout(group)
        form = QtWidgets.QFormLayout()
        self.total_target_coverage = self._double_spin(90.0, 0.0, 100.0)
        self.enable_safe_traverse = QtWidgets.QCheckBox("Enable safe traverse")
        self.enable_safe_traverse.setChecked(True)
        self.safety_clearance = self._double_spin(80.0, 20.0, 300.0)
        self.smooth_free_travel = QtWidgets.QCheckBox("Smooth free travel")
        self.smooth_free_travel.setChecked(True)
        form.addRow("Total target coverage (%)", self.total_target_coverage)
        form.addRow("Enable safe traverse", self.enable_safe_traverse)
        form.addRow("Safety clearance (mm)", self.safety_clearance)
        form.addRow("Smooth free travel", self.smooth_free_travel)
        group_layout.addLayout(form)

        self.generate_button = QtWidgets.QPushButton("Generate mission")
        self.validate_button = QtWidgets.QPushButton("Validate mission")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_json_button = QtWidgets.QPushButton("Export JSON")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        for widget in (
            self.generate_button,
            self.validate_button,
            self.export_csv_button,
            self.export_json_button,
            self.cancel_button,
        ):
            group_layout.addWidget(widget)
        layout.addWidget(group)

        self.coverage_button = QtWidgets.QPushButton("Run coverage")
        self.recovery_button = QtWidgets.QPushButton("Generate recovery")
        self.optimise_button = QtWidgets.QPushButton("Optimise mission")
        self.build_mission_button = QtWidgets.QPushButton("Build continuous mission")
        self.export_mission_button = QtWidgets.QPushButton("Export mission")
        for widget in (
            self.coverage_button,
            self.recovery_button,
            self.optimise_button,
            self.build_mission_button,
            self.export_mission_button,
        ):
            widget.hide()

    def _build_status_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Status And Results")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.stats_model_loaded = QtWidgets.QLabel("Model loaded: no")
        self.stats_detected_holes = QtWidgets.QLabel("Detected holes: -")
        self.stats_total_poses = QtWidgets.QLabel("Total poses: -")
        self.stats_scan_poses = QtWidgets.QLabel("Scan poses: -")
        self.stats_free_poses = QtWidgets.QLabel("Free-travel poses: -")
        self.stats_scan_dist = QtWidgets.QLabel("Scan distance: -")
        self.stats_free_dist = QtWidgets.QLabel("Free-travel distance: -")
        self.stats_total_dist = QtWidgets.QLabel("Total path length: -")
        self.stats_est_time = QtWidgets.QLabel("Estimated mission time: -")
        self.stats_coverage = QtWidgets.QLabel("Coverage: -")
        self.stats_clearance = QtWidgets.QLabel("Minimum clearance: -")
        self.stats_validation = QtWidgets.QLabel("Validation status: not run")
        for widget in (
            self.stats_model_loaded,
            self.stats_detected_holes,
            self.stats_total_poses,
            self.stats_scan_poses,
            self.stats_free_poses,
            self.stats_scan_dist,
            self.stats_free_dist,
            self.stats_total_dist,
            self.stats_est_time,
            self.stats_coverage,
            self.stats_clearance,
            self.stats_validation,
        ):
            group_layout.addWidget(widget)
        layout.addWidget(group)

    def _build_viewer_group(self, layout: QtWidgets.QVBoxLayout) -> None:
        group = QtWidgets.QGroupBox("Viewer")
        group_layout = QtWidgets.QVBoxLayout(group)
        self.layer_checks: dict[str, QtWidgets.QCheckBox] = {}
        for key, label in LAYER_LABELS.items():
            check = QtWidgets.QCheckBox(label)
            check.setChecked(key in {"cad", "coverage", "mission_scan", "mission_free_travel"})
            check.toggled.connect(lambda value, layer=key: self.layer_changed.emit(layer, value))
            self.layer_checks[key] = check
            group_layout.addWidget(check)
        for key, label in DEBUG_LAYER_LABELS.items():
            check = QtWidgets.QCheckBox(label)
            check.setChecked(False)
            check.hide()
            check.toggled.connect(lambda value, layer=key: self.layer_changed.emit(layer, value))
            self.layer_checks[key] = check
        self.fit_cad_button = QtWidgets.QPushButton("Fit CAD")
        self.fit_trajectory_button = QtWidgets.QPushButton("Fit mission")
        self.reset_camera_button = QtWidgets.QPushButton("Reset camera")
        self.advanced_button = QtWidgets.QPushButton("Advanced settings...")
        for widget in (
            self.fit_cad_button,
            self.fit_trajectory_button,
            self.reset_camera_button,
            self.advanced_button,
        ):
            group_layout.addWidget(widget)
        layout.addWidget(group)

    def _create_internal_advanced_controls(self) -> None:
        self.planning_mode = QtWidgets.QComboBox()
        self.planning_mode.addItem("Simple view-based sweeps", "simple_view_based")
        self.planning_mode.addItem("Advanced face-based planning", "advanced_face_based")
        self.standard_laser_rays = self._spin(21)
        self.final_laser_rays = self._spin(101)
        self.max_incidence_angle = self._double_spin(60.0, 1.0, 89.0)
        self.lane_spacing = self._double_spin(20.0, 0.01, 10000.0)
        self.overlap = self._double_spin(20.0, 0.0, 99.0)
        self.minimum_face_area = self._double_spin(25.0, 0.0, 1e12)
        self.include_internal = QtWidgets.QCheckBox()
        self.cylinder_strategy = QtWidgets.QComboBox()
        self.cylinder_strategy.addItems(["longitudinal", "helical"])
        self.surface_types_editor = QtWidgets.QLineEdit("plane, cylinder")
        self.surface_checks = {}
        for name in ("plane", "cylinder", "cone", "sphere", "torus"):
            check = QtWidgets.QCheckBox()
            check.setChecked(name in ("plane", "cylinder"))
            self.surface_checks[name] = check
        self.view_checks = {}
        for name in ("top", "front", "right", "back", "left"):
            check = QtWidgets.QCheckBox()
            check.setChecked(True)
            self.view_checks[name] = check
        self.isometric_elevation = self._double_spin(35.0, 1.0, 89.0)
        self.isometric_azimuth_step = self._double_spin(10.0, 1.0, 90.0)
        self.maximum_recovery_sweeps = self._spin(1)
        self.maximum_views_per_hole = self._spin(2)
        self.candidate_tilt_angles = QtWidgets.QLineEdit("25, 40")
        self.candidate_azimuth_count = self._spin(2)
        self.focused_orbit_minimum_arc = self._double_spin(45.0, 1.0, 360.0)
        self.focused_orbit_maximum_arc = self._double_spin(240.0, 1.0, 360.0)
        self.enable_orbital_hole = QtWidgets.QCheckBox()
        self.enable_blind_bottom = QtWidgets.QCheckBox()
        self.enable_blind_bottom.setChecked(True)
        self.enable_bottom_sweep = QtWidgets.QCheckBox()
        self.enable_bottom_sweep.setChecked(True)
        self.enable_lower_isometric = QtWidgets.QCheckBox()
        self.enable_lower_isometric.setChecked(True)
        self.bottom_clearance = self._double_spin(25.0, 0.0, 10000.0)
        self.lower_isometric_elevation = self._double_spin(35.0, 1.0, 89.0)
        self.maximum_lower_sweeps = self._spin(1)
        self.fixture_exclusion = QtWidgets.QCheckBox()
        self.fixture_path = QtWidgets.QLineEdit()
        self.fixture_button = QtWidgets.QPushButton()
        self.minimum_candidate_gain = self._double_spin(1.0, 0.0, 100.0)
        self.maximum_optional_sweeps = self._spin(3)
        self.connector_stride = self._spin(10)
        self.frame_stride = self._spin(20)
        self.laser_stride = self._spin(20)
        self.debug_display = QtWidgets.QCheckBox()
        self.region_combo = QtWidgets.QComboBox()
        self.face_combo = QtWidgets.QComboBox()
        self.surface_combo = QtWidgets.QComboBox()
        self.surface_combo.addItems(["All", "Plane", "Cylinder", "Cone", "Sphere"])
        self.only_selected_region = QtWidgets.QCheckBox()

        self.prefer_top_clearance = QtWidgets.QCheckBox()
        self.show_safety_envelope = QtWidgets.QCheckBox()
        self.show_safety_envelope.setChecked(True)
        self.tool_envelope_radius = self._double_spin(60.0, 0.0, 1000.0)
        self.envelope_type_widget = QtWidgets.QLineEdit("obb")
        self.vertical_clearance = self._double_spin(100.0, 0.0, 10000.0)
        self.orientation_weight = self._double_spin(1.0, 0.0, 100.0)
        self.vertical_motion_penalty = self._double_spin(50.0, 0.0, 1000.0)
        self.corner_penalty = self._double_spin(20.0, 0.0, 1000.0)
        self.proximity_penalty = self._double_spin(10.0, 0.0, 1000.0)
        self.graph_waypoint_density = self._double_spin(2.0, 0.1, 10.0)
        self.collision_sampling_interval = self._double_spin(5.0, 0.1, 100.0)
        self.max_position_step = self._double_spin(20.0, 0.1, 1000.0)
        self.max_orientation_step = self._double_spin(5.0, 0.1, 180.0)
        self.smoothing_method_widget = QtWidgets.QLineEdit("catmull_rom")
        self.nominal_blend_radius = self._double_spin(60.0, 0.1, 10000.0)
        self.minimum_blend_radius = self._double_spin(10.0, 0.1, 10000.0)
        self.smoothing_sample_spacing = self._double_spin(10.0, 0.1, 1000.0)
        self.maximum_retries = self._spin(5)
        self.orientation_smoothing_max_step = self._double_spin(5.0, 0.1, 180.0)
        self.orientation_interpolation_window = self._double_spin(0.8, 0.01, 1.0)
        self.worker_count = self._spin(1)
        self.mesh_tolerance = self._double_spin(0.25, 0.01, 100.0)

    def _apply_tooltips(self) -> None:
        mapping = {
            self.planning_quality: "planning_quality",
            self.stand_off: "stand_off_mm",
            self.travel_spacing: "travel_spacing_mm",
            self.laser_width: "laser_width_mm",
            self.top_sweep: "top_sweep",
            self.side_sweeps: "side_sweeps",
            self.isometric_recovery: "isometric_recovery",
            self.bottom_coverage: "enable_bottom_sweep",
            self.enable_hole_recovery: "enable_hole_recovery",
            self.enable_focused_hole_orbit: "enable_focused_hole_orbit",
            self.total_target_coverage: "total_target_coverage_percent",
            self.enable_safe_traverse: "enable_safe_traverse",
            self.safety_clearance: "safety_clearance_mm",
            self.smooth_free_travel: "free_travel_smoothing_enabled",
        }
        for widget, field in mapping.items():
            widget.setToolTip(tooltip(field))
        self.generate_button.setToolTip(
            "Runs scan generation, coverage, recovery, mission optimisation, safe traverse, smoothing, and viewer update."
        )
        action_tip = "Runs this operator action using the current validated configuration."
        for widget in (
            self.upload_button,
            self.validate_button,
            self.export_csv_button,
            self.export_json_button,
            self.cancel_button,
            self.fit_cad_button,
            self.fit_trajectory_button,
            self.reset_camera_button,
            self.advanced_button,
        ):
            widget.setToolTip(action_tip)
        for check in self.layer_checks.values():
            check.setToolTip("Shows or hides this viewer layer without replanning.")
        self.hole_tree.setToolTip("Physical holes; expand a row to inspect stepped bore sections.")
        for label in self.widget().findChildren(QtWidgets.QLabel):
            if not label.toolTip():
                label.setToolTip("Mission status reported from the latest generated or validated result.")

    def _connect_signals(self) -> None:
        self.planning_quality.currentIndexChanged.connect(self._quality_changed)
        self.planning_mode.currentIndexChanged.connect(self.planning_changed)
        for widget in (
            self.stand_off,
            self.travel_spacing,
            self.laser_width,
            self.hole_target_coverage,
            self.total_target_coverage,
            self.safety_clearance,
        ):
            widget.valueChanged.connect(self.planning_changed)
        for widget in (
            self.top_sweep,
            self.side_sweeps,
            self.isometric_recovery,
            self.bottom_coverage,
            self.enable_hole_recovery,
            self.enable_focused_hole_orbit,
            self.enable_safe_traverse,
            self.smooth_free_travel,
        ):
            widget.toggled.connect(self.planning_changed)
        for widget in (
            self.stand_off,
            self.travel_spacing,
            self.laser_width,
            self.total_target_coverage,
            self.safety_clearance,
            self.top_sweep,
            self.side_sweeps,
            self.isometric_recovery,
            self.bottom_coverage,
            self.enable_hole_recovery,
            self.enable_focused_hole_orbit,
            self.enable_safe_traverse,
            self.smooth_free_travel,
        ):
            signal = widget.valueChanged if hasattr(widget, "valueChanged") else widget.toggled
            signal.connect(self._mark_explicit_gui_change)
        self.hole_tree.itemSelectionChanged.connect(self._emit_hole_selection)
        self.advanced_button.clicked.connect(self.show_advanced_settings)

    @staticmethod
    def _double_spin(value, minimum, maximum):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(2)
        widget.setValue(value)
        return widget

    @staticmethod
    def _spin(value):
        widget = QtWidgets.QSpinBox()
        widget.setRange(1, 100000)
        widget.setValue(value)
        return widget

    def _gui_overrides(self):
        return {
            "planning": {
                "quality": self.planning_quality.currentData(),
                "mode": self.planning_mode.currentData(),
                "lane_spacing_mm": self.lane_spacing.value(),
                "minimum_face_area_mm2": self.minimum_face_area.value(),
                "cylinder_strategy": self.cylinder_strategy.currentText(),
                "surface_types": self.surface_types(),
            },
            "scanner": {
                "nominal_standoff_mm": self.stand_off.value(),
                "travel_spacing_mm": self.travel_spacing.value(),
                "laser_line_width_mm": self.laser_width.value(),
            },
            "validation": {
                "standard_laser_ray_count": self.standard_laser_rays.value(),
                "final_laser_ray_count": self.final_laser_rays.value(),
                "maximum_incidence_angle_deg": self.max_incidence_angle.value(),
            },
            "principal_coverage": {
                "top_sweep": self.top_sweep.isChecked(),
                "side_sweeps": self.side_sweeps.isChecked(),
                "upper_isometric_recovery": self.isometric_recovery.isChecked(),
                "bottom_coverage": self.bottom_coverage.isChecked(),
                "sweep_overlap_fraction": self.overlap.value() / 100.0,
                "isometric_azimuth_step_deg": self.isometric_azimuth_step.value(),
            },
            "hole_recovery": {
                "enabled": self.enable_hole_recovery.isChecked(),
                "target_coverage_percent": self.hole_target_coverage.value(),
                "maximum_views_per_hole": self.maximum_views_per_hole.value(),
                "candidate_azimuth_count": self.candidate_azimuth_count.value(),
                "minimum_candidate_gain_percent": self.minimum_candidate_gain.value(),
                "blind_bottom_enabled": self.enable_blind_bottom.isChecked(),
                "lower_isometric_enabled": self.enable_lower_isometric.isChecked(),
                "maximum_lower_recovery_sweeps": self.maximum_lower_sweeps.value(),
                "focused_conical_orbit": {
                    "enabled": self.enable_focused_hole_orbit.isChecked(),
                    "candidate_cone_angles_deg": self.focused_orbit_cone_angles_deg(),
                    "minimum_arc_deg": self.focused_orbit_minimum_arc.value(),
                    "maximum_arc_deg": self.focused_orbit_maximum_arc.value(),
                },
            },
            "mission": {
                "total_target_coverage_percent": self.total_target_coverage.value(),
                "maximum_optional_sweeps": self.maximum_optional_sweeps.value(),
            },
            "viewer": {
                "connector_stride": self.connector_stride.value(),
                "frame_stride": self.frame_stride.value(),
                "laser_footprint_stride": self.laser_stride.value(),
            },
            "transition_planning": {
                "enable_safe_traverse": self.enable_safe_traverse.isChecked(),
                "safety_clearance_mm": self.safety_clearance.value(),
                "vertical_clearance_mm": self.vertical_clearance.value(),
                "prefer_top_clearance": self.prefer_top_clearance.isChecked(),
                "show_safety_envelope": self.show_safety_envelope.isChecked(),
                "tool_envelope_radius_mm": self.tool_envelope_radius.value(),
                "envelope_type": self.envelope_type_widget.text(),
                "orientation_weight": self.orientation_weight.value(),
                "vertical_motion_penalty": self.vertical_motion_penalty.value(),
                "corner_penalty": self.corner_penalty.value(),
                "proximity_penalty": self.proximity_penalty.value(),
                "graph_waypoint_density": self.graph_waypoint_density.value(),
                "collision_sampling_interval_mm": self.collision_sampling_interval.value(),
            },
            "free_travel_smoothing": {
                "enabled": self.smooth_free_travel.isChecked(),
                "method": self.smoothing_method_widget.text(),
                "nominal_blend_radius_mm": self.nominal_blend_radius.value(),
                "minimum_blend_radius_mm": self.minimum_blend_radius.value(),
                "sample_spacing_mm": self.smoothing_sample_spacing.value(),
                "maximum_retries": self.maximum_retries.value(),
            },
            "orientation_smoothing": {
                "maximum_step_deg": self.orientation_smoothing_max_step.value(),
                "interpolation_window_fraction": self.orientation_interpolation_window.value(),
            },
        }

    def planning_config(self, step_path, output_dir=None):
        config = pipeline_config_from_yaml(
            step_path,
            self.planning_quality.currentData(),
            gui_overrides=self._gui_overrides(),
            output_dir=output_dir,
        )
        return replace(
            config,
            include_internal_surfaces=self.include_internal.isChecked(),
            bottom_clearance_mm=self.bottom_clearance.value(),
            fixture_exclusion_enabled=self.fixture_exclusion.isChecked(),
            fixture_mesh_path=(
                Path(self.fixture_path.text()).expanduser()
                if self.fixture_path.text().strip()
                else None
            ),
        )

    def _quality_changed(self) -> None:
        if self._loading_profile:
            return
        if self._explicit_gui_changes:
            message = QtWidgets.QMessageBox(self)
            message.setIcon(QtWidgets.QMessageBox.Icon.Question)
            message.setWindowTitle("Reset visible settings?")
            message.setText("Apply the selected quality profile to visible settings?")
            message.setInformativeText("Choose No to keep explicit GUI edits and only update advanced defaults.")
            message.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if message.exec() == QtWidgets.QMessageBox.StandardButton.No:
                self._load_profile_values(visible=False)
                return
        self._load_profile_values(visible=True)
        self._explicit_gui_changes = False

    def _load_profile_values(self, *, visible: bool) -> None:
        self._loading_profile = True
        config = pipeline_config_from_yaml(Path("model.step"), self.planning_quality.currentData())
        self.planning_mode.setCurrentIndex(self.planning_mode.findData(config.planning_mode))
        self.standard_laser_rays.setValue(config.standard_laser_ray_count)
        self.final_laser_rays.setValue(config.final_laser_ray_count)
        self.max_incidence_angle.setValue(config.max_incidence_angle_deg)
        self.lane_spacing.setValue(config.lane_spacing_mm)
        self.minimum_face_area.setValue(config.minimum_face_area_mm2)
        self.surface_types_editor.setText(", ".join(config.surface_types))
        self.cylinder_strategy.setCurrentText(config.cylinder_strategy.value)
        self.overlap.setValue(100.0 * config.sweep_overlap_fraction)
        self.isometric_azimuth_step.setValue(config.isometric_azimuth_step_deg)
        self.maximum_views_per_hole.setValue(config.maximum_views_per_hole)
        self.candidate_tilt_angles.setText(", ".join(f"{value:g}" for value in config.focused_orbit_cone_angles_deg))
        self.candidate_azimuth_count.setValue(config.hole_candidate_azimuth_count)
        self.minimum_candidate_gain.setValue(config.minimum_candidate_gain_percent)
        self.maximum_optional_sweeps.setValue(config.maximum_optional_sweeps)
        self.focused_orbit_minimum_arc.setValue(config.focused_orbit_minimum_arc_deg)
        self.focused_orbit_maximum_arc.setValue(config.focused_orbit_maximum_arc_deg)
        self.enable_blind_bottom.setChecked(config.enable_blind_hole_bottom_recovery)
        self.enable_lower_isometric.setChecked(config.enable_lower_isometric_recovery)
        self.bottom_clearance.setValue(config.bottom_clearance_mm)
        self.maximum_lower_sweeps.setValue(config.maximum_lower_recovery_sweeps)
        self.prefer_top_clearance.setChecked(config.prefer_top_clearance)
        self.show_safety_envelope.setChecked(config.show_safety_envelope)
        self.tool_envelope_radius.setValue(config.tool_envelope_radius_mm)
        self.envelope_type_widget.setText(config.envelope_type)
        self.vertical_clearance.setValue(config.vertical_clearance_mm)
        self.orientation_weight.setValue(config.orientation_weight)
        self.vertical_motion_penalty.setValue(config.vertical_motion_penalty)
        self.corner_penalty.setValue(config.corner_penalty)
        self.proximity_penalty.setValue(config.proximity_penalty)
        self.graph_waypoint_density.setValue(config.graph_waypoint_density)
        self.collision_sampling_interval.setValue(config.collision_sampling_interval_mm)
        self.max_position_step.setValue(config.max_position_step_mm)
        self.max_orientation_step.setValue(config.max_orientation_step_deg)
        self.smoothing_method_widget.setText(config.free_travel_smoothing_method)
        self.nominal_blend_radius.setValue(config.nominal_blend_radius_mm)
        self.minimum_blend_radius.setValue(config.minimum_blend_radius_mm)
        self.smoothing_sample_spacing.setValue(config.sample_spacing_mm)
        self.maximum_retries.setValue(config.maximum_retries)
        self.orientation_smoothing_max_step.setValue(config.orientation_smoothing_maximum_step_deg)
        self.orientation_interpolation_window.setValue(config.orientation_interpolation_window_fraction)
        self.connector_stride.setValue(config.connector_stride)
        self.frame_stride.setValue(config.frame_stride)
        self.laser_stride.setValue(config.laser_footprint_stride)
        if visible:
            self.hole_target_coverage.setValue(config.hole_target_coverage_percent)
            self.total_target_coverage.setValue(config.total_target_coverage_percent)
            self.enable_focused_hole_orbit.setChecked(config.enable_focused_hole_orbit)
            self.safety_clearance.setValue(config.safety_clearance_mm)
            self.enable_safe_traverse.setChecked(config.enable_safe_traverse)
            self.smooth_free_travel.setChecked(config.free_travel_smoothing_enabled)
        self.planning_changed.emit()
        self._refresh_candidate_estimate()
        self._loading_profile = False

    def apply_fast_recovery_defaults(self) -> None:
        self.planning_quality.setCurrentIndex(self.planning_quality.findData("fast_preview"))
        self._load_profile_values(visible=True)

    def update_candidate_estimate(self, features, reports=()):
        try:
            estimate = estimate_candidate_complexity(
                features,
                reports,
                tilt_count=len(self.focused_orbit_cone_angles_deg()),
                azimuth_count=1,
                orbital_mode="adaptive" if self.enable_focused_hole_orbit.isChecked() else "disabled",
                blind_bottom_enabled=self.enable_blind_bottom.isChecked(),
                bottom_sweep_enabled=self.bottom_coverage.isChecked(),
                lower_isometric_enabled=self.bottom_coverage.isChecked(),
                maximum_views_per_hole=self.maximum_views_per_hole.value(),
            )
            self.candidate_estimate_label.setText(
                f"Estimated candidates: {estimate.estimated_candidates} (after pruning <= {estimate.pruned_candidate_limit})"
            )
            return estimate
        except Exception:
            self.candidate_estimate_label.setText("Estimated candidates: -")
            return None

    def focused_orbit_cone_angles_deg(self):
        return tuple(float(value.strip()) for value in self.candidate_tilt_angles.text().split(",") if value.strip())

    def surface_types(self) -> tuple[str, ...]:
        return tuple(value.strip() for value in self.surface_types_editor.text().split(",") if value.strip())

    def _refresh_candidate_estimate(self, *_args):
        self.update_candidate_estimate(self._estimated_features, self._estimated_reports)

    def set_selection_values(self, regions, faces):
        for combo, values in ((self.region_combo, regions), (self.face_combo, faces)):
            combo.clear()
            combo.addItem("All", None)
            for value in values:
                combo.addItem(str(value), int(value))

    def set_hole_features(self, features, before_reports, after_reports, evaluations):
        self._estimated_features = tuple(features)
        self._estimated_reports = tuple(after_reports or before_reports)
        self.hole_tree.clear()
        before = {row.feature_id: row for row in before_reports}
        after = {row.feature_id: row for row in after_reports}
        selected = {row.candidate_id: row for row in evaluations if row.selected}
        for feature in features:
            prior = before.get(feature.feature_id)
            final = after.get(feature.feature_id)
            strategies = sorted({row.strategy for row in selected.values() if f"hole_{feature.feature_id}_" in row.name})
            item = QtWidgets.QTreeWidgetItem([
                str(feature.feature_id),
                f"{2 * feature.radius_mm:.2f}",
                f"{feature.depth_mm:.2f}",
                f"{prior.cylindrical_wall_coverage_percent:.1f}%" if prior else "-",
                "Yes" if final and not final.complete else "No",
                ", ".join(strategies) or "None",
                f"{final.cylindrical_wall_coverage_percent:.1f}%" if final else "-",
            ])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, feature.feature_id)
            self.hole_tree.addTopLevelItem(item)
        self.stats_detected_holes.setText(f"Detected holes: {len(features)}")
        self.update_candidate_estimate(features, after_reports or before_reports)

    def _emit_hole_selection(self):
        items = self.hole_tree.selectedItems()
        self.hole_selection_changed.emit(
            int(items[0].data(0, QtCore.Qt.ItemDataRole.UserRole))
            if items and items[0].parent() is None
            else -1
        )

    def show_advanced_settings(self):
        from gui.advanced_settings import AdvancedSettingsDialog

        AdvancedSettingsDialog(self, self).exec()

    def reset_to_profile(self, *, confirm: bool = True) -> None:
        if confirm:
            message = QtWidgets.QMessageBox(self)
            message.setWindowTitle("Reset to profile")
            message.setText("Reset GUI and advanced values to the selected quality profile?")
            message.setStandardButtons(
                QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No
            )
            if message.exec() != QtWidgets.QMessageBox.StandardButton.Yes:
                return
        self._explicit_gui_changes = False
        self._load_profile_values(visible=True)

    def advanced_overrides(self) -> dict:
        return self._gui_overrides()

    def _mark_explicit_gui_change(self, *_args) -> None:
        if not self._loading_profile:
            self._explicit_gui_changes = True

    def update_mission_stats(
        self,
        total_poses: int,
        scan_poses: int,
        free_poses: int,
        scan_dist: float,
        free_dist: float,
        total_dist: float,
        est_time: float,
        coverage_percent: float | None = None,
        min_clearance: float | None = None,
        detected_holes: int | None = None,
        report: dict | None = None,
    ) -> None:
        self.stats_model_loaded.setText("Model loaded: yes")
        if detected_holes is not None:
            self.stats_detected_holes.setText(f"Detected holes: {detected_holes}")
        self.stats_total_poses.setText(f"Total poses: {total_poses}")
        self.stats_scan_poses.setText(f"Scan poses: {scan_poses}")
        self.stats_free_poses.setText(f"Free-travel poses: {free_poses}")
        self.stats_scan_dist.setText(f"Scan distance: {scan_dist:.1f} mm")
        self.stats_free_dist.setText(f"Free-travel distance: {free_dist:.1f} mm")
        self.stats_total_dist.setText(f"Total path length: {total_dist:.1f} mm")
        self.stats_est_time.setText(f"Estimated mission time: {est_time:.1f} s")
        if coverage_percent is not None:
            self.stats_coverage.setText(f"Coverage: {coverage_percent:.1f}%")
        if min_clearance is not None:
            self.stats_clearance.setText(f"Minimum clearance: {min_clearance:.1f} mm")
        if report is not None:
            self.stats_validation.setText("Validation status: valid" if report.get("valid") else "Validation status: failed")

    def reset_mission_stats(self) -> None:
        self.stats_model_loaded.setText("Model loaded: yes" if self.filename_label.text() != "No model loaded" else "Model loaded: no")
        self.stats_detected_holes.setText("Detected holes: -")
        self.stats_total_poses.setText("Total poses: -")
        self.stats_scan_poses.setText("Scan poses: -")
        self.stats_free_poses.setText("Free-travel poses: -")
        self.stats_scan_dist.setText("Scan distance: -")
        self.stats_free_dist.setText("Free-travel distance: -")
        self.stats_total_dist.setText("Total path length: -")
        self.stats_est_time.setText("Estimated mission time: -")
        self.stats_coverage.setText("Coverage: -")
        self.stats_clearance.setText("Minimum clearance: -")
        self.stats_validation.setText("Validation status: not run")

    def set_export_enabled(self, enabled: bool) -> None:
        self.export_csv_button.setEnabled(enabled)
        self.export_json_button.setEnabled(enabled)
