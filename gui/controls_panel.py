"""Compact routine-operation controls backed by documented YAML settings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6 import QtCore, QtWidgets

from app.configuration import (
    advanced_summary, pipeline_config_from_yaml, tooltip,
)
from recovery import estimate_candidate_complexity


LAYER_LABELS = {
    "cad": "CAD model",
    "coverage": "Coverage",
    "scanner": "Scanner paths",
    "recovery": "Recovery paths",
    "focused": "Focused hole orbits",
    "features": "Detected holes",
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
        self.setWidgetResizable(True)
        body = QtWidgets.QWidget()
        self.setWidget(body)
        layout = QtWidgets.QVBoxLayout(body)

        model = QtWidgets.QGroupBox("Model")
        model_layout = QtWidgets.QVBoxLayout(model)
        self.upload_button = QtWidgets.QPushButton("Upload STEP")
        self.filename_label = QtWidgets.QLabel("No model loaded")
        self.filename_label.setWordWrap(True)
        self.face_statistics_label = QtWidgets.QLabel("Faces: —")
        self.dimensions_label = QtWidgets.QLabel("Dimensions: —")
        self.planning_quality = QtWidgets.QComboBox()
        for label, value in (("Fast preview", "fast_preview"),
                             ("Standard", "standard"),
                             ("Final validation", "final_validation")):
            self.planning_quality.addItem(label, value)
        self.planning_mode = QtWidgets.QComboBox()
        self.planning_mode.addItem("Simple view-based sweeps", "simple_view_based")
        self.planning_mode.addItem("Advanced face-based planning", "advanced_face_based")
        model_layout.addWidget(self.upload_button)
        for label, widget in (("Planning quality", self.planning_quality),
                              ("Planning mode", self.planning_mode)):
            row = QtWidgets.QFormLayout(); row.addRow(label, widget); model_layout.addLayout(row)
        for widget in (self.filename_label, self.face_statistics_label, self.dimensions_label):
            model_layout.addWidget(widget)
        layout.addWidget(model)

        scanner = QtWidgets.QGroupBox("Scanner")
        scanner_form = QtWidgets.QFormLayout(scanner)
        self.stand_off = self._double_spin(310.0, 1.0, 10000.0)
        self.travel_spacing = self._double_spin(5.0, 0.01, 10000.0)
        self.laser_width = self._double_spin(200.0, 0.01, 10000.0)
        scanner_form.addRow("Stand-off (mm)", self.stand_off)
        scanner_form.addRow("Travel spacing (mm)", self.travel_spacing)
        scanner_form.addRow("Laser-line width (mm)", self.laser_width)
        layout.addWidget(scanner)

        principal = QtWidgets.QGroupBox("Principal coverage")
        principal_layout = QtWidgets.QVBoxLayout(principal)
        self.top_sweep = QtWidgets.QCheckBox("Top sweep")
        self.side_sweeps = QtWidgets.QCheckBox("Side sweeps")
        self.isometric_recovery = QtWidgets.QCheckBox("Upper isometric recovery")
        self.bottom_coverage = QtWidgets.QCheckBox("Bottom/underside coverage")
        for widget in (self.top_sweep, self.side_sweeps,
                       self.isometric_recovery, self.bottom_coverage):
            widget.setChecked(True); principal_layout.addWidget(widget)
        layout.addWidget(principal)

        recovery = QtWidgets.QGroupBox("Feature recovery")
        recovery_form = QtWidgets.QFormLayout(recovery)
        self.enable_hole_recovery = QtWidgets.QCheckBox(); self.enable_hole_recovery.setChecked(True)
        self.hole_target_coverage = self._double_spin(85.0, 0.0, 100.0)
        self.enable_focused_hole_orbit = QtWidgets.QCheckBox(); self.enable_focused_hole_orbit.setChecked(True)
        recovery_form.addRow("Enable hole recovery", self.enable_hole_recovery)
        recovery_form.addRow("Hole target coverage (%)", self.hole_target_coverage)
        recovery_form.addRow("Enable focused hole orbit", self.enable_focused_hole_orbit)
        self.candidate_estimate_label = QtWidgets.QLabel("Estimated candidates: 0")
        recovery_form.addRow("Complexity", self.candidate_estimate_label)
        self.hole_tree = QtWidgets.QTreeWidget()
        self.hole_tree.setHeaderLabels(["Hole", "Diameter", "Depth", "Before", "Recovery", "Strategy", "Final"])
        recovery_form.addRow("Detected holes", self.hole_tree)
        layout.addWidget(recovery)

        mission = QtWidgets.QGroupBox("Mission")
        mission_layout = QtWidgets.QVBoxLayout(mission)
        total_row = QtWidgets.QFormLayout()
        self.total_target_coverage = self._double_spin(90.0, 0.0, 100.0)
        total_row.addRow("Total target coverage (%)", self.total_target_coverage)
        mission_layout.addLayout(total_row)
        self.generate_button = QtWidgets.QPushButton("Generate paths")
        self.coverage_button = QtWidgets.QPushButton("Run coverage")
        self.recovery_button = QtWidgets.QPushButton("Generate recovery")
        self.optimise_button = QtWidgets.QPushButton("Optimise mission")
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_json_button = QtWidgets.QPushButton("Export JSON")
        self.cancel_button = QtWidgets.QPushButton("Cancel"); self.cancel_button.setEnabled(False)
        for widget in (self.generate_button, self.coverage_button, self.recovery_button,
                       self.optimise_button, self.validate_button,
                       self.export_csv_button, self.export_json_button, self.cancel_button):
            mission_layout.addWidget(widget)
        layout.addWidget(mission)

        layers = QtWidgets.QGroupBox("Viewer")
        layers_layout = QtWidgets.QVBoxLayout(layers)
        self.layer_checks = {}
        for key, label in LAYER_LABELS.items():
            check = QtWidgets.QCheckBox(label); check.setChecked(key != "focused")
            check.toggled.connect(lambda value, layer=key: self.layer_changed.emit(layer, value))
            self.layer_checks[key] = check; layers_layout.addWidget(check)
        self.fit_cad_button = QtWidgets.QPushButton("Fit CAD")
        self.fit_trajectory_button = QtWidgets.QPushButton("Fit full trajectory")
        self.reset_camera_button = QtWidgets.QPushButton("Reset camera")
        self.advanced_button = QtWidgets.QPushButton("Advanced settings…")
        for button in (self.fit_cad_button, self.fit_trajectory_button,
                       self.reset_camera_button, self.advanced_button):
            layers_layout.addWidget(button)
        layout.addWidget(layers)
        layout.addStretch()

        self._create_internal_advanced_controls()
        self._apply_tooltips()
        self._connect_signals()
        self._quality_changed()

    def _create_internal_advanced_controls(self) -> None:
        """Compatibility state loaded from YAML; these widgets are never shown."""
        self.standard_laser_rays = self._spin(21)
        self.final_laser_rays = self._spin(101)
        self.lane_spacing = self._double_spin(20.0, 0.01, 10000.0)
        self.overlap = self._double_spin(20.0, 0.0, 99.0)
        self.minimum_face_area = self._double_spin(25.0, 0.0, 1e12)
        self.include_internal = QtWidgets.QCheckBox()
        self.cylinder_strategy = QtWidgets.QComboBox(); self.cylinder_strategy.addItems(["longitudinal", "helical"])
        self.surface_checks = {}
        for name in ("plane", "cylinder", "cone", "sphere", "torus"):
            check = QtWidgets.QCheckBox(); check.setChecked(name in ("plane", "cylinder")); self.surface_checks[name] = check
        self.view_checks = {}
        for name in ("top", "front", "right", "back", "left"):
            check = QtWidgets.QCheckBox(); check.setChecked(True); self.view_checks[name] = check
        self.isometric_elevation = self._double_spin(35.0, 1.0, 89.0)
        self.maximum_recovery_sweeps = self._spin(1)
        self.maximum_views_per_hole = self._spin(2)
        self.candidate_tilt_angles = QtWidgets.QLineEdit("25, 40")
        self.candidate_azimuth_count = self._spin(2)
        self.enable_orbital_hole = QtWidgets.QCheckBox()
        self.enable_blind_bottom = QtWidgets.QCheckBox(); self.enable_blind_bottom.setChecked(True)
        self.enable_bottom_sweep = QtWidgets.QCheckBox(); self.enable_bottom_sweep.setChecked(True)
        self.enable_lower_isometric = QtWidgets.QCheckBox(); self.enable_lower_isometric.setChecked(True)
        self.bottom_clearance = self._double_spin(25.0, 0.0, 10000.0)
        self.lower_isometric_elevation = self._double_spin(35.0, 1.0, 89.0)
        self.maximum_lower_sweeps = self._spin(1)
        self.fixture_exclusion = QtWidgets.QCheckBox(); self.fixture_path = QtWidgets.QLineEdit(); self.fixture_button = QtWidgets.QPushButton()
        self.minimum_candidate_gain = self._double_spin(1.0, 0.0, 100.0)
        self.maximum_optional_sweeps = self._spin(3)
        self.connector_stride = self._spin(10); self.frame_stride = self._spin(20); self.laser_stride = self._spin(20)
        self.region_combo = QtWidgets.QComboBox(); self.face_combo = QtWidgets.QComboBox(); self.surface_combo = QtWidgets.QComboBox(); self.surface_combo.addItems(["All", "Plane", "Cylinder", "Cone", "Sphere"])
        self.only_selected_region = QtWidgets.QCheckBox()

    def _apply_tooltips(self) -> None:
        mapping = {
            self.planning_quality: "planning_quality", self.planning_mode: "planning_mode",
            self.stand_off: "stand_off_mm", self.travel_spacing: "travel_spacing_mm",
            self.laser_width: "laser_width_mm", self.top_sweep: "top_sweep",
            self.side_sweeps: "side_sweeps", self.isometric_recovery: "isometric_recovery",
            self.bottom_coverage: "enable_bottom_sweep",
            self.enable_hole_recovery: "enable_hole_recovery",
            self.hole_target_coverage: "hole_target_coverage_percent",
            self.enable_focused_hole_orbit: "enable_focused_hole_orbit",
            self.total_target_coverage: "total_target_coverage_percent",
        }
        for widget, field in mapping.items():
            widget.setToolTip(tooltip(field))
        action_tip = "Runs this pipeline stage using the current validated configuration."
        for widget in (self.upload_button, self.generate_button, self.coverage_button,
                       self.recovery_button, self.optimise_button, self.validate_button,
                       self.export_csv_button, self.export_json_button, self.cancel_button,
                       self.fit_cad_button, self.fit_trajectory_button,
                       self.reset_camera_button, self.advanced_button):
            widget.setToolTip(action_tip)
        for check in self.layer_checks.values():
            check.setToolTip("Shows or hides this existing viewer layer without replanning.")
        self.hole_tree.setToolTip("Physical holes; expand a row to inspect stepped bore sections.")

    def _connect_signals(self) -> None:
        self.planning_quality.currentIndexChanged.connect(self._quality_changed)
        for widget in (self.planning_mode,):
            widget.currentIndexChanged.connect(self.planning_changed)
        for widget in (self.stand_off, self.travel_spacing, self.laser_width,
                       self.hole_target_coverage, self.total_target_coverage):
            widget.valueChanged.connect(self.planning_changed)
        for widget in (self.top_sweep, self.side_sweeps, self.isometric_recovery,
                       self.bottom_coverage, self.enable_hole_recovery,
                       self.enable_focused_hole_orbit):
            widget.toggled.connect(self.planning_changed)
        self.hole_tree.itemSelectionChanged.connect(self._emit_hole_selection)
        self.advanced_button.clicked.connect(self._show_advanced_summary)

    @staticmethod
    def _double_spin(value, minimum, maximum):
        widget = QtWidgets.QDoubleSpinBox(); widget.setRange(minimum, maximum); widget.setDecimals(2); widget.setValue(value); return widget

    @staticmethod
    def _spin(value):
        widget = QtWidgets.QSpinBox(); widget.setRange(1, 100000); widget.setValue(value); return widget

    def _gui_overrides(self):
        return {
            "planning": {"quality": self.planning_quality.currentData(), "mode": self.planning_mode.currentData()},
            "scanner": {"nominal_standoff_mm": self.stand_off.value(), "travel_spacing_mm": self.travel_spacing.value(), "laser_line_width_mm": self.laser_width.value()},
            "principal_coverage": {"top_sweep": self.top_sweep.isChecked(), "side_sweeps": self.side_sweeps.isChecked(), "upper_isometric_recovery": self.isometric_recovery.isChecked(), "bottom_coverage": self.bottom_coverage.isChecked()},
            "hole_recovery": {"enabled": self.enable_hole_recovery.isChecked(), "target_coverage_percent": self.hole_target_coverage.value(), "focused_conical_orbit": {"enabled": self.enable_focused_hole_orbit.isChecked()}},
            "mission": {"total_target_coverage_percent": self.total_target_coverage.value()},
        }

    def planning_config(self, step_path, output_dir=None):
        config = pipeline_config_from_yaml(step_path, self.planning_quality.currentData(), gui_overrides=self._gui_overrides(), output_dir=output_dir)
        return replace(config, include_internal_surfaces=self.include_internal.isChecked(),
                       bottom_clearance_mm=self.bottom_clearance.value(),
                       fixture_exclusion_enabled=self.fixture_exclusion.isChecked(),
                       fixture_mesh_path=(Path(self.fixture_path.text()).expanduser() if self.fixture_path.text().strip() else None))

    def _quality_changed(self) -> None:
        config = pipeline_config_from_yaml(Path("model.step"), self.planning_quality.currentData())
        self.standard_laser_rays.setValue(config.standard_laser_ray_count); self.final_laser_rays.setValue(config.final_laser_ray_count)
        self.maximum_views_per_hole.setValue(config.maximum_views_per_hole)
        self.candidate_tilt_angles.setText(", ".join(f"{value:g}" for value in config.focused_orbit_cone_angles_deg))
        self.minimum_candidate_gain.setValue(config.minimum_candidate_gain_percent); self.maximum_optional_sweeps.setValue(config.maximum_optional_sweeps)
        self.hole_target_coverage.setValue(config.hole_target_coverage_percent); self.total_target_coverage.setValue(config.total_target_coverage_percent)
        self.enable_focused_hole_orbit.setChecked(config.enable_focused_hole_orbit)
        self.planning_changed.emit(); self._refresh_candidate_estimate()

    def apply_fast_recovery_defaults(self) -> None:
        self.planning_quality.setCurrentIndex(self.planning_quality.findData("fast_preview")); self._quality_changed()

    def update_candidate_estimate(self, features, reports=()):
        try:
            estimate = estimate_candidate_complexity(features, reports,
                tilt_count=len(self.focused_orbit_cone_angles_deg()), azimuth_count=1,
                orbital_mode="adaptive" if self.enable_focused_hole_orbit.isChecked() else "disabled",
                blind_bottom_enabled=self.enable_blind_bottom.isChecked(),
                bottom_sweep_enabled=self.bottom_coverage.isChecked(),
                lower_isometric_enabled=self.bottom_coverage.isChecked(),
                maximum_views_per_hole=self.maximum_views_per_hole.value())
            self.candidate_estimate_label.setText(f"Estimated candidates: {estimate.estimated_candidates} (after pruning ≤ {estimate.pruned_candidate_limit})")
            return estimate
        except Exception:
            self.candidate_estimate_label.setText("Estimated candidates: —"); return None

    def focused_orbit_cone_angles_deg(self):
        return tuple(float(value.strip()) for value in self.candidate_tilt_angles.text().split(",") if value.strip())

    def _refresh_candidate_estimate(self, *_args):
        self.update_candidate_estimate(self._estimated_features, self._estimated_reports)

    def set_selection_values(self, regions, faces):
        for combo, values in ((self.region_combo, regions), (self.face_combo, faces)):
            combo.clear(); combo.addItem("All", None)
            for value in values: combo.addItem(str(value), int(value))

    def set_hole_features(self, features, before_reports, after_reports, evaluations):
        self._estimated_features = tuple(features); self._estimated_reports = tuple(after_reports or before_reports); self.hole_tree.clear()
        before = {row.feature_id: row for row in before_reports}; after = {row.feature_id: row for row in after_reports}
        selected = {row.candidate_id: row for row in evaluations if row.selected}
        for feature in features:
            prior, final = before.get(feature.feature_id), after.get(feature.feature_id)
            strategies = sorted({row.strategy for row in selected.values() if f"hole_{feature.feature_id}_" in row.name})
            item = QtWidgets.QTreeWidgetItem([str(feature.feature_id), f"{2 * feature.radius_mm:.2f}", f"{feature.depth_mm:.2f}", f"{prior.cylindrical_wall_coverage_percent:.1f}%" if prior else "—", "Yes" if final and not final.complete else "No", ", ".join(strategies) or "None", f"{final.cylindrical_wall_coverage_percent:.1f}%" if final else "—"])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, feature.feature_id); self.hole_tree.addTopLevelItem(item)
            for index, section in enumerate(feature.sections):
                item.addChild(QtWidgets.QTreeWidgetItem([f"Section {index + 1}", f"{section.diameter_mm:.2f}", f"{section.start_depth_mm:.2f}–{section.end_depth_mm:.2f}", "", "", section.section_type, "faces " + ", ".join(map(str, section.face_ids))]))
        self.update_candidate_estimate(features, after_reports or before_reports)

    def _emit_hole_selection(self):
        items = self.hole_tree.selectedItems(); self.hole_selection_changed.emit(int(items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)) if items and items[0].parent() is None else -1)

    def _show_advanced_summary(self):
        try: text = advanced_summary(self.planning_config(Path("model.step")))
        except Exception as exc: text = str(exc)
        dialog = QtWidgets.QMessageBox(self); dialog.setWindowTitle("Advanced settings"); dialog.setText("Advanced settings are loaded from config/*.yaml."); dialog.setDetailedText(text); dialog.exec()
