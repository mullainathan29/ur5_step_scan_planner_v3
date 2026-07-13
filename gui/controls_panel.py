"""Standard Qt controls for planning and viewer state."""

from __future__ import annotations

from pathlib import Path

from PySide6 import QtCore, QtWidgets

from app.config import PipelineConfig, recovery_quality_profile
from recovery import estimate_candidate_complexity


LAYER_LABELS = {
    "cad": "CAD model",
    "coverage": "Coverage",
    "nominal": "Nominal surface paths",
    "scanner": "Scanner paths",
    "transition": "Approach/retract",
    "connectors": "Connectors",
    "frames": "Scanner frames",
    "laser": "Laser footprints",
    "labels": "Face labels",
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
        for widget in (self.upload_button, self.filename_label,
                       self.face_statistics_label, self.dimensions_label):
            model_layout.addWidget(widget)
        layout.addWidget(model)

        planning = QtWidgets.QGroupBox("Planning")
        form = QtWidgets.QFormLayout(planning)
        self.planning_quality = QtWidgets.QComboBox()
        self.planning_quality.addItem("Fast preview", "fast_preview")
        self.planning_quality.addItem("Standard", "standard")
        self.planning_quality.addItem("Final validation", "final_validation")
        self.standard_laser_rays = self._spin(21)
        self.final_laser_rays = self._spin(101)
        form.addRow("Planning quality", self.planning_quality)
        form.addRow("Standard laser rays", self.standard_laser_rays)
        form.addRow("Final laser rays", self.final_laser_rays)
        self.planning_mode = QtWidgets.QComboBox()
        self.planning_mode.addItem("Simple view-based sweeps", "simple_view_based")
        self.planning_mode.addItem("Advanced face-based planning", "advanced_face_based")
        form.addRow("Planning mode", self.planning_mode)
        self.surface_checks = {}
        surface_row = QtWidgets.QWidget()
        surface_layout = QtWidgets.QVBoxLayout(surface_row)
        surface_layout.setContentsMargins(0, 0, 0, 0)
        for name in ("plane", "cylinder", "cone", "sphere", "torus"):
            check = QtWidgets.QCheckBox(name.title())
            check.setChecked(name in ("plane", "cylinder"))
            if name == "torus":
                check.setEnabled(False)
                check.setToolTip("Torus planning is not implemented yet.")
            check.toggled.connect(self.planning_changed)
            self.surface_checks[name] = check
            surface_layout.addWidget(check)
        form.addRow("Surface types", surface_row)
        self.lane_spacing = self._double_spin(20.0, 0.01, 10000.0)
        self.travel_spacing = self._double_spin(5.0, 0.01, 10000.0)
        self.stand_off = self._double_spin(310.0, 0.01, 10000.0)
        self.laser_width = self._double_spin(200.0, 0.01, 10000.0)
        self.overlap = self._double_spin(20.0, 0.0, 99.0)
        self.minimum_face_area = self._double_spin(25.0, 0.0, 1e12)
        self.include_internal = QtWidgets.QCheckBox()
        self.cylinder_strategy = QtWidgets.QComboBox()
        self.cylinder_strategy.addItems(["longitudinal", "helical"])
        for label, widget in (
            ("Lane spacing (mm)", self.lane_spacing),
            ("Travel spacing (mm)", self.travel_spacing),
            ("Stand-off (mm)", self.stand_off),
            ("Laser-line width (mm)", self.laser_width),
            ("Sweep overlap (%)", self.overlap),
            ("Minimum face area (mm²)", self.minimum_face_area),
            ("Include internal surfaces", self.include_internal),
            ("Cylinder strategy", self.cylinder_strategy),
        ):
            form.addRow(label, widget)
        self.view_checks = {}
        views = QtWidgets.QWidget()
        views_layout = QtWidgets.QVBoxLayout(views)
        views_layout.setContentsMargins(0, 0, 0, 0)
        for name in ("top", "front", "right", "back", "left"):
            check = QtWidgets.QCheckBox(f"{name.title()} sweep")
            check.setChecked(True)
            check.toggled.connect(self.planning_changed)
            self.view_checks[name] = check
            views_layout.addWidget(check)
        form.addRow("Principal views", views)
        self.isometric_recovery = QtWidgets.QCheckBox()
        self.isometric_recovery.setChecked(True)
        self.isometric_elevation = self._double_spin(35.0, 1.0, 89.0)
        self.maximum_recovery_sweeps = self._spin(1)
        form.addRow("Isometric recovery", self.isometric_recovery)
        form.addRow("Isometric elevation (deg)", self.isometric_elevation)
        form.addRow("Maximum recovery sweeps", self.maximum_recovery_sweeps)
        for widget in (self.lane_spacing, self.travel_spacing, self.stand_off,
                       self.laser_width, self.overlap, self.isometric_elevation,
                       self.minimum_face_area):
            widget.valueChanged.connect(self.planning_changed)
        self.maximum_recovery_sweeps.valueChanged.connect(self.planning_changed)
        self.isometric_recovery.toggled.connect(self.planning_changed)
        self.planning_mode.currentIndexChanged.connect(self.planning_changed)
        self.planning_quality.currentIndexChanged.connect(self._quality_changed)
        self.standard_laser_rays.valueChanged.connect(self.planning_changed)
        self.final_laser_rays.valueChanged.connect(self.planning_changed)
        self.include_internal.toggled.connect(self.planning_changed)
        self.cylinder_strategy.currentIndexChanged.connect(self.planning_changed)
        layout.addWidget(planning)

        recovery = QtWidgets.QGroupBox("Feature Recovery")
        recovery_form = QtWidgets.QFormLayout(recovery)
        self.enable_hole_recovery = QtWidgets.QCheckBox(); self.enable_hole_recovery.setChecked(True)
        self.hole_target_coverage = self._double_spin(85.0, 0.0, 100.0)
        self.maximum_views_per_hole = self._spin(2)
        self.candidate_tilt_angles = QtWidgets.QLineEdit("25, 40")
        self.candidate_azimuth_count = self._spin(2)
        self.enable_orbital_hole = QtWidgets.QCheckBox(); self.enable_orbital_hole.setChecked(False)
        self.enable_blind_bottom = QtWidgets.QCheckBox(); self.enable_blind_bottom.setChecked(True)
        self.enable_bottom_sweep = QtWidgets.QCheckBox(); self.enable_bottom_sweep.setChecked(True)
        self.enable_lower_isometric = QtWidgets.QCheckBox(); self.enable_lower_isometric.setChecked(True)
        self.bottom_clearance = self._double_spin(25.0, 0.0, 10000.0)
        self.lower_isometric_elevation = self._double_spin(35.0, 1.0, 89.0)
        self.maximum_lower_sweeps = self._spin(1)
        self.fixture_exclusion = QtWidgets.QCheckBox()
        fixture_row = QtWidgets.QWidget()
        fixture_layout = QtWidgets.QHBoxLayout(fixture_row)
        fixture_layout.setContentsMargins(0, 0, 0, 0)
        self.fixture_path = QtWidgets.QLineEdit()
        self.fixture_button = QtWidgets.QPushButton("Browse")
        self.fixture_button.clicked.connect(self._choose_fixture)
        fixture_layout.addWidget(self.fixture_path); fixture_layout.addWidget(self.fixture_button)
        self.total_target_coverage = self._double_spin(90.0, 0.0, 100.0)
        self.minimum_candidate_gain = self._double_spin(1.0, 0.0, 100.0)
        self.maximum_optional_sweeps = self._spin(3)
        self.candidate_estimate_label = QtWidgets.QLabel("Estimated candidates: 0")
        for label, widget in (
            ("Enable hole recovery", self.enable_hole_recovery),
            ("Hole target coverage (%)", self.hole_target_coverage),
            ("Maximum views per hole", self.maximum_views_per_hole),
            ("Candidate tilt angles", self.candidate_tilt_angles),
            ("Candidate azimuth count", self.candidate_azimuth_count),
            ("Enable orbital hole sweep", self.enable_orbital_hole),
            ("Enable blind-hole bottom recovery", self.enable_blind_bottom),
            ("Enable bottom sweep", self.enable_bottom_sweep),
            ("Enable lower isometric orbit", self.enable_lower_isometric),
            ("Bottom clearance (mm)", self.bottom_clearance),
            ("Lower isometric elevation", self.lower_isometric_elevation),
            ("Maximum lower recovery sweeps", self.maximum_lower_sweeps),
            ("Fixture exclusion enabled", self.fixture_exclusion),
            ("Fixture mesh", fixture_row),
            ("Total target coverage (%)", self.total_target_coverage),
            ("Minimum candidate gain (%)", self.minimum_candidate_gain),
            ("Maximum optional sweeps", self.maximum_optional_sweeps),
        ):
            recovery_form.addRow(label, widget)
        self.hole_tree = QtWidgets.QTreeWidget()
        self.hole_tree.setHeaderLabels([
            "Hole", "Diameter", "Depth", "Before", "Recovery", "Strategy", "Final"
        ])
        recovery_form.addRow("Detected holes", self.hole_tree)
        recovery_form.addRow("Complexity", self.candidate_estimate_label)
        self.hole_tree.itemSelectionChanged.connect(self._emit_hole_selection)
        for widget in (
            self.enable_hole_recovery, self.hole_target_coverage,
            self.maximum_views_per_hole, self.candidate_tilt_angles,
            self.candidate_azimuth_count, self.enable_orbital_hole,
            self.enable_blind_bottom, self.enable_bottom_sweep,
            self.enable_lower_isometric, self.bottom_clearance,
            self.lower_isometric_elevation, self.maximum_lower_sweeps,
            self.fixture_exclusion, self.total_target_coverage,
            self.minimum_candidate_gain, self.maximum_optional_sweeps,
        ):
            if isinstance(widget, QtWidgets.QAbstractButton):
                widget.toggled.connect(self.planning_changed)
                widget.toggled.connect(self._refresh_candidate_estimate)
            elif isinstance(widget, QtWidgets.QLineEdit):
                widget.editingFinished.connect(self.planning_changed)
                widget.editingFinished.connect(self._refresh_candidate_estimate)
            elif hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self.planning_changed)
                widget.valueChanged.connect(self._refresh_candidate_estimate)
        layout.addWidget(recovery)

        selection = QtWidgets.QGroupBox("Selection")
        selection_form = QtWidgets.QFormLayout(selection)
        self.region_combo = QtWidgets.QComboBox()
        self.face_combo = QtWidgets.QComboBox()
        self.surface_combo = QtWidgets.QComboBox()
        self.surface_combo.addItems(["All", "Plane", "Cylinder", "Cone", "Sphere"])
        self.only_selected_region = QtWidgets.QCheckBox()
        selection_form.addRow("Region", self.region_combo)
        selection_form.addRow("Face", self.face_combo)
        selection_form.addRow("Surface type", self.surface_combo)
        selection_form.addRow("Only selected region", self.only_selected_region)
        for widget in (self.region_combo, self.face_combo, self.surface_combo):
            widget.currentIndexChanged.connect(self.selection_changed)
        self.only_selected_region.toggled.connect(self.selection_changed)
        layout.addWidget(selection)

        layers = QtWidgets.QGroupBox("Viewer layers")
        layers_layout = QtWidgets.QVBoxLayout(layers)
        defaults = {"cad", "coverage", "nominal", "scanner"}
        self.layer_checks = {}
        for key, label in LAYER_LABELS.items():
            check = QtWidgets.QCheckBox(label)
            check.setChecked(key in defaults)
            check.toggled.connect(lambda value, layer=key: self.layer_changed.emit(layer, value))
            self.layer_checks[key] = check
            layers_layout.addWidget(check)
        layout.addWidget(layers)

        display = QtWidgets.QGroupBox("Display settings")
        display_form = QtWidgets.QFormLayout(display)
        self.connector_stride = self._spin(10)
        self.frame_stride = self._spin(20)
        self.laser_stride = self._spin(20)
        display_form.addRow("Connector stride", self.connector_stride)
        display_form.addRow("Frame stride", self.frame_stride)
        display_form.addRow("Laser footprint stride", self.laser_stride)
        self.fit_cad_button = QtWidgets.QPushButton("Fit CAD")
        self.fit_trajectory_button = QtWidgets.QPushButton("Fit full trajectory")
        self.reset_camera_button = QtWidgets.QPushButton("Reset camera")
        for button in (self.fit_cad_button, self.fit_trajectory_button,
                       self.reset_camera_button):
            display_form.addRow(button)
        for widget in (self.connector_stride, self.frame_stride, self.laser_stride):
            widget.valueChanged.connect(self.stride_changed)
        layout.addWidget(display)

        actions = QtWidgets.QGroupBox("Actions")
        actions_layout = QtWidgets.QVBoxLayout(actions)
        self.generate_button = QtWidgets.QPushButton("Generate paths")
        self.coverage_button = QtWidgets.QPushButton("Run coverage")
        self.recovery_button = QtWidgets.QPushButton("Generate feature recovery")
        self.optimise_button = QtWidgets.QPushButton("Optimise mission")
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_json_button = QtWidgets.QPushButton("Export JSON")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        for button in (self.generate_button, self.coverage_button,
                       self.recovery_button, self.optimise_button,
                       self.validate_button, self.export_csv_button,
                       self.export_json_button, self.cancel_button):
            actions_layout.addWidget(button)
        layout.addWidget(actions)
        layout.addStretch()

    @staticmethod
    def _double_spin(value: float, minimum: float, maximum: float):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(2)
        widget.setValue(value)
        return widget

    @staticmethod
    def _spin(value: int):
        widget = QtWidgets.QSpinBox()
        widget.setRange(1, 100000)
        widget.setValue(value)
        return widget

    def planning_config(self, step_path, output_dir=None) -> PipelineConfig:
        from planners.base import CylinderStrategy
        try:
            tilt_angles = tuple(float(value.strip()) for value in
                                self.candidate_tilt_angles.text().split(",") if value.strip())
        except ValueError as exc:
            raise ValueError("Candidate tilt angles must be comma-separated numbers.") from exc
        return PipelineConfig(
            step_path=step_path,
            planning_mode=self.planning_mode.currentData(),
            planning_quality=self.planning_quality.currentData(),
            standard_laser_ray_count=self.standard_laser_rays.value(),
            final_laser_ray_count=self.final_laser_rays.value(),
            surface_types=tuple(k for k, v in self.surface_checks.items() if v.isChecked()),
            lane_spacing_mm=self.lane_spacing.value(),
            travel_spacing_mm=self.travel_spacing.value(),
            stand_off_mm=self.stand_off.value(),
            laser_width_mm=self.laser_width.value(),
            sweep_overlap_fraction=self.overlap.value() / 100.0,
            top_sweep=self.view_checks["top"].isChecked(),
            front_sweep=self.view_checks["front"].isChecked(),
            right_sweep=self.view_checks["right"].isChecked(),
            back_sweep=self.view_checks["back"].isChecked(),
            left_sweep=self.view_checks["left"].isChecked(),
            isometric_recovery=self.isometric_recovery.isChecked(),
            isometric_elevation_deg=self.isometric_elevation.value(),
            maximum_isometric_sweeps=self.maximum_recovery_sweeps.value(),
            enable_hole_recovery=self.enable_hole_recovery.isChecked(),
            hole_target_coverage_percent=self.hole_target_coverage.value(),
            maximum_views_per_hole=self.maximum_views_per_hole.value(),
            hole_candidate_tilt_angles_deg=tilt_angles,
            hole_candidate_azimuth_count=self.candidate_azimuth_count.value(),
            enable_orbital_hole_sweep=self.enable_orbital_hole.isChecked(),
            orbital_hole_mode=(
                "disabled" if not self.enable_orbital_hole.isChecked()
                else "adaptive" if self.planning_quality.currentData() == "standard"
                else "enabled"
            ),
            enable_blind_hole_bottom_recovery=self.enable_blind_bottom.isChecked(),
            enable_bottom_sweep=self.enable_bottom_sweep.isChecked(),
            enable_lower_isometric_recovery=self.enable_lower_isometric.isChecked(),
            bottom_clearance_mm=self.bottom_clearance.value(),
            lower_isometric_elevation_deg=self.lower_isometric_elevation.value(),
            maximum_lower_recovery_sweeps=self.maximum_lower_sweeps.value(),
            fixture_exclusion_enabled=self.fixture_exclusion.isChecked(),
            fixture_mesh_path=(Path(self.fixture_path.text()).expanduser()
                               if self.fixture_path.text().strip() else None),
            total_target_coverage_percent=self.total_target_coverage.value(),
            minimum_candidate_gain_percent=self.minimum_candidate_gain.value(),
            maximum_optional_sweeps=self.maximum_optional_sweeps.value(),
            minimum_face_area_mm2=self.minimum_face_area.value(),
            include_internal_surfaces=self.include_internal.isChecked(),
            cylinder_strategy=CylinderStrategy(self.cylinder_strategy.currentText()),
            display=False,
            output_dir=output_dir,
            connector_stride=self.connector_stride.value(),
            frame_stride=self.frame_stride.value(),
            laser_footprint_stride=self.laser_stride.value(),
        )

    def set_selection_values(self, regions, faces) -> None:
        for combo, values in ((self.region_combo, regions), (self.face_combo, faces)):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All", None)
            for value in values:
                combo.addItem(str(value), int(value))
            combo.blockSignals(False)

    def set_hole_features(self, features, before_reports, after_reports, evaluations) -> None:
        self._estimated_features = tuple(features)
        self._estimated_reports = tuple(after_reports or before_reports)
        self.hole_tree.clear()
        before = {row.feature_id: row for row in before_reports}
        after = {row.feature_id: row for row in after_reports}
        selected = {row.candidate_id: row for row in evaluations if row.selected}
        for feature in features:
            prior = before.get(feature.feature_id)
            final = after.get(feature.feature_id)
            strategies = sorted({row.strategy for row in selected.values()
                                 if f"hole_{feature.feature_id}_" in row.name})
            item = QtWidgets.QTreeWidgetItem([
                str(feature.feature_id), f"{2 * feature.radius_mm:.2f}",
                f"{feature.depth_mm:.2f}",
                f"{prior.cylindrical_wall_coverage_percent:.1f}%" if prior else "—",
                "Yes" if final and not final.complete else "No",
                ", ".join(strategies) or "None",
                f"{final.cylindrical_wall_coverage_percent:.1f}%" if final else "—",
            ])
            item.setData(0, QtCore.Qt.ItemDataRole.UserRole, feature.feature_id)
            self.hole_tree.addTopLevelItem(item)
            for section_index, section in enumerate(feature.sections):
                child = QtWidgets.QTreeWidgetItem([
                    f"Section {section_index + 1}",
                    f"{section.diameter_mm:.2f}",
                    f"{section.start_depth_mm:.2f}–{section.end_depth_mm:.2f}",
                    "", "", section.section_type,
                    "faces " + ", ".join(map(str, section.face_ids)),
                ])
                item.addChild(child)
        self.update_candidate_estimate(features, after_reports or before_reports)

    def _emit_hole_selection(self) -> None:
        items = self.hole_tree.selectedItems()
        self.hole_selection_changed.emit(
            int(items[0].data(0, QtCore.Qt.ItemDataRole.UserRole)) if items else -1
        )

    def _choose_fixture(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select fixture mesh", "",
            "Mesh files (*.stl *.obj *.ply *.off);;All files (*)",
        )
        if filename:
            self.fixture_path.setText(filename)
            self.planning_changed.emit()

    def _quality_changed(self) -> None:
        quality = self.planning_quality.currentData()
        profile = recovery_quality_profile(quality)
        self.hole_target_coverage.setValue(profile["hole_target_coverage_percent"])
        self.maximum_views_per_hole.setValue(profile["maximum_views_per_hole"])
        self.candidate_tilt_angles.setText(", ".join(
            f"{value:g}" for value in profile["hole_candidate_tilt_angles_deg"]
        ))
        self.candidate_azimuth_count.setValue(profile["hole_candidate_azimuth_count"])
        self.enable_orbital_hole.setChecked(profile["orbital_hole_mode"] != "disabled")
        self.maximum_optional_sweeps.setValue(profile["maximum_optional_sweeps"])
        self.minimum_candidate_gain.setValue(profile["minimum_candidate_gain_percent"])
        self.total_target_coverage.setValue(profile["total_target_coverage_percent"])
        self.planning_changed.emit()

    def apply_fast_recovery_defaults(self) -> None:
        self.planning_quality.setCurrentIndex(
            self.planning_quality.findData("fast_preview")
        )
        self._quality_changed()

    def update_candidate_estimate(self, features, reports=()):
        try:
            tilt_count = len([value for value in self.candidate_tilt_angles.text().split(",")
                              if value.strip()])
            mode = (
                "disabled" if not self.enable_orbital_hole.isChecked()
                else "adaptive" if self.planning_quality.currentData() == "standard"
                else "enabled"
            )
            estimate = estimate_candidate_complexity(
                features, reports, tilt_count=tilt_count,
                azimuth_count=self.candidate_azimuth_count.value(),
                orbital_mode=mode,
                blind_bottom_enabled=self.enable_blind_bottom.isChecked(),
                bottom_sweep_enabled=self.enable_bottom_sweep.isChecked(),
                lower_isometric_enabled=self.enable_lower_isometric.isChecked(),
                maximum_views_per_hole=self.maximum_views_per_hole.value(),
            )
            self.candidate_estimate_label.setText(
                f"Estimated candidates: {estimate.estimated_candidates} "
                f"(after pruning ≤ {estimate.pruned_candidate_limit})"
            )
            return estimate
        except Exception:
            self.candidate_estimate_label.setText("Estimated candidates: —")
            return None

    def _refresh_candidate_estimate(self, *_args) -> None:
        self.update_candidate_estimate(
            self._estimated_features, self._estimated_reports
        )
