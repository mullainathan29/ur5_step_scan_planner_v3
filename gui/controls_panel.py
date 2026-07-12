"""Standard Qt controls for planning and viewer state."""

from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from app.config import PipelineConfig


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
}


class ControlsPanel(QtWidgets.QScrollArea):
    planning_changed = QtCore.Signal()
    selection_changed = QtCore.Signal()
    layer_changed = QtCore.Signal(str, bool)
    stride_changed = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
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
        self.include_internal.toggled.connect(self.planning_changed)
        self.cylinder_strategy.currentIndexChanged.connect(self.planning_changed)
        layout.addWidget(planning)

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
        self.validate_button = QtWidgets.QPushButton("Validate")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_json_button = QtWidgets.QPushButton("Export JSON")
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        for button in (self.generate_button, self.coverage_button,
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
        return PipelineConfig(
            step_path=step_path,
            planning_mode=self.planning_mode.currentData(),
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
