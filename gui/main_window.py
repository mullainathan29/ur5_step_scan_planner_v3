"""Main PySide6 window for the unified B-Rep pipeline."""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path
from time import perf_counter

from PySide6 import QtCore, QtWidgets

from gui.controls_panel import ControlsPanel
from gui.viewer_widget import ViewerWidget
from app.staged_pipeline import GeometryCache, StageTimings
from gui.workers import (
    CoverageWorker, FeatureRecoveryWorker, OptimisationWorker, PlanningWorker,
    StepLoadWorker, ValidationWorker, WorkerResult,
)
from exporter import export_staged_trajectory


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("UR5 STEP B-Rep Scan Planner")
        self.resize(1400, 900)
        self.controls = ControlsPanel()
        self.controls.setMinimumWidth(330)
        self.viewer = ViewerWidget()
        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.controls)
        splitter.addWidget(self.viewer)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setMaximumWidth(260)
        self.progress_bar.hide()
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.step_path: Path | None = None
        self.model = None
        self.result = None
        self.config = None
        self.planning_dirty = False
        self._thread = None
        self._worker = None
        self.geometry_cache = GeometryCache()
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        log_dock = QtWidgets.QDockWidget("Pipeline timings", self)
        log_dock.setWidget(self.log_view)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        self._create_menus()
        self._connect_controls()

    def _create_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Upload STEP", self.choose_step)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)
        menu.addMenu("Generate").addAction("Generate paths", self.generate_paths)
        menu.addMenu("Coverage").addAction("Run coverage", self.run_coverage)
        export_menu = menu.addMenu("Export")
        export_menu.addAction("Export CSV", self.export_csv)
        export_menu.addAction("Export JSON", self.export_json)

    def _connect_controls(self) -> None:
        c = self.controls
        c.upload_button.clicked.connect(self.choose_step)
        c.generate_button.clicked.connect(self.generate_paths)
        c.coverage_button.clicked.connect(self.run_coverage)
        c.recovery_button.clicked.connect(self.generate_feature_recovery)
        c.optimise_button.clicked.connect(self.optimise_mission)
        c.validate_button.clicked.connect(self.validate_geometry)
        c.export_csv_button.clicked.connect(self.export_csv)
        c.export_json_button.clicked.connect(self.export_json)
        c.cancel_button.clicked.connect(self.cancel_worker)
        c.layer_changed.connect(self.viewer.set_layer_visibility)
        c.selection_changed.connect(self.apply_selection)
        c.planning_changed.connect(self.mark_planning_dirty)
        c.stride_changed.connect(self.refresh_stride_geometry)
        c.fit_cad_button.clicked.connect(self.viewer.fit_cad)
        c.fit_trajectory_button.clicked.connect(self.viewer.fit_trajectory)
        c.reset_camera_button.clicked.connect(self.viewer.reset_camera)
        c.hole_selection_changed.connect(self.viewer.select_hole)

    @QtCore.Slot()
    def choose_step(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open STEP model", "", "STEP files (*.step *.stp)"
        )
        if filename:
            self.load_step(Path(filename))

    def load_step(self, path: Path) -> None:
        self.step_path = path
        self.geometry_cache.set_path(path)
        self.controls.filename_label.setText(path.name)
        self._start_worker(StepLoadWorker(path, self.geometry_cache), self._model_loaded)

    def _model_loaded(self, payload: WorkerResult) -> None:
        model = payload.value
        self._log_timings(payload.timings)
        self.model = model
        self.result = None
        counts = Counter(record.geometry_type.value for record in model.faces)
        stats = ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        self.controls.face_statistics_label.setText(f"Faces: {len(model.faces)} ({stats})")
        dimensions = model.validation_mesh.mesh.extents
        self.controls.dimensions_label.setText(
            f"Dimensions: {dimensions[0]:.2f} × {dimensions[1]:.2f} × {dimensions[2]:.2f} mm"
        )
        self.controls.set_selection_values([], [face.face_id for face in model.faces])
        self.viewer.show_model(model)
        self.planning_dirty = True
        self.statusBar().showMessage(f"Loaded {model.source_path.name}", 5000)

    @QtCore.Slot()
    def generate_paths(self) -> None:
        if self.step_path is None:
            self._message("Load a STEP model before generating paths.")
            return
        try:
            config = self.controls.planning_config(self.step_path)
            config.validate()
        except ValueError as exc:
            self._message(str(exc))
            return
        self.config = config
        self._start_worker(PlanningWorker(config, self.geometry_cache), self._planning_complete)

    def _planning_complete(self, payload: WorkerResult) -> None:
        result = payload.value
        self._log_timings(payload.timings)
        self.result = result
        self.planning_dirty = False
        regions = sorted({pose.region_id for pose in result.poses if pose.scan_enabled})
        self.controls.set_selection_values(regions, [face.face_id for face in result.model.faces])
        started = perf_counter()
        self.viewer.show_result(result, self.config)
        viewer_timings = StageTimings(); viewer_timings.record(
            "viewer update", perf_counter() - started
        )
        self._log_timings(viewer_timings)
        if result.hole_features:
            optimisation = result.optimisation_report
            self.controls.set_hole_features(
                result.hole_features,
                optimisation.hole_reports_before if optimisation else (),
                result.hole_coverage_reports,
                optimisation.candidate_evaluations if optimisation else (),
            )
        self._sync_layer_controls()
        self.statusBar().showMessage(
            f"Generated {result.summary.generated_scan_segments} segments, "
            f"{result.summary.scan_pose_count} scan poses", 8000)

    @QtCore.Slot()
    def run_coverage(self) -> None:
        if self.result is None or self.config is None:
            self._message("Generate paths before running coverage.")
            return
        self.config = self.controls.planning_config(self.step_path)
        self._start_worker(CoverageWorker(
            self.result, self.config, self.geometry_cache
        ), self._result_updated)

    @QtCore.Slot()
    def generate_feature_recovery(self) -> None:
        if self.result is None or self.config is None:
            self._message("Generate paths and run coverage first.")
            return
        self.config = self.controls.planning_config(self.step_path)
        estimate = self.controls.update_candidate_estimate(
            self.result.hole_features, self.result.hole_coverage_reports
        )
        if (estimate is not None
                and self.config.planning_quality == "fast_preview"
                and estimate.estimated_candidates > 100):
            message = QtWidgets.QMessageBox(self)
            message.setIcon(QtWidgets.QMessageBox.Icon.Warning)
            message.setWindowTitle("High recovery complexity")
            message.setText(
                f"Estimated recovery candidates: {estimate.estimated_candidates}."
            )
            message.setInformativeText(
                "Apply reduced Fast-preview settings, continue with the "
                "current configuration, or cancel."
            )
            apply_button = message.addButton(
                "Apply recommended settings", QtWidgets.QMessageBox.ButtonRole.AcceptRole
            )
            continue_button = message.addButton(
                "Continue", QtWidgets.QMessageBox.ButtonRole.DestructiveRole
            )
            message.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
            message.exec()
            clicked = message.clickedButton()
            if clicked is apply_button:
                self.controls.apply_fast_recovery_defaults()
                self.config = self.controls.planning_config(self.step_path)
            elif clicked is not continue_button:
                return
        self._start_worker(
            FeatureRecoveryWorker(self.result, self.config), self._result_updated
        )

    @QtCore.Slot()
    def optimise_mission(self) -> None:
        if self.result is None or self.config is None:
            self._message("Generate paths, run coverage, and generate recovery first.")
            return
        self.config = self.controls.planning_config(self.step_path)
        self._start_worker(
            OptimisationWorker(self.result, self.config), self._result_updated
        )

    @QtCore.Slot()
    def validate_geometry(self) -> None:
        if self.result is None:
            self._message("Generate paths before validating geometry.")
            return
        self._start_worker(ValidationWorker(
            self.result, self.config, self.geometry_cache
        ), self._result_updated)

    def _result_updated(self, payload: WorkerResult) -> None:
        result = payload.value
        self._log_timings(payload.timings)
        self.result = result
        started = perf_counter()
        self.viewer.show_result(result, self.config)
        timings = StageTimings(); timings.record("viewer update", perf_counter() - started)
        self._log_timings(timings)
        if result.hole_features:
            optimisation = result.optimisation_report
            self.controls.set_hole_features(
                result.hole_features,
                (optimisation.hole_reports_before if optimisation else result.hole_coverage_reports),
                result.hole_coverage_reports,
                (optimisation.candidate_evaluations if optimisation else ()),
            )
        self._sync_layer_controls()
        self.apply_selection()

    @QtCore.Slot()
    def mark_planning_dirty(self) -> None:
        self.planning_dirty = True
        self.statusBar().showMessage("Planning parameters changed; generate paths to apply.")

    @QtCore.Slot()
    def refresh_stride_geometry(self) -> None:
        if self.result is None or self.config is None:
            return
        from dataclasses import replace
        self.config = replace(
            self.config,
            connector_stride=self.controls.connector_stride.value(),
            frame_stride=self.controls.frame_stride.value(),
            laser_footprint_stride=self.controls.laser_stride.value(),
        )
        self.viewer.show_result(self.result, self.config)
        self._sync_layer_controls()
        self.apply_selection()

    @QtCore.Slot()
    def apply_selection(self) -> None:
        surface = self.controls.surface_combo.currentText().lower()
        self.viewer.set_selection(
            self.controls.region_combo.currentData(),
            self.controls.face_combo.currentData(),
            None if surface == "all" else surface,
            self.controls.only_selected_region.isChecked(),
        )

    def _sync_layer_controls(self) -> None:
        for layer, check in self.controls.layer_checks.items():
            self.viewer.set_layer_visibility(layer, check.isChecked())

    def _start_worker(self, worker, completion) -> None:
        if self._thread is not None:
            self._message("Another task is already running.")
            return
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._worker_progress)
        worker.completed.connect(completion)
        worker.failed.connect(self._worker_failed)
        worker.cancelled.connect(lambda: self.statusBar().showMessage("Task cancelled", 5000))
        worker.finished.connect(thread.quit)
        worker.finished.connect(self._worker_finished)
        thread.finished.connect(thread.deleteLater)
        self._thread, self._worker = thread, worker
        self.controls.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.progress_bar.show()
        thread.start()

    @QtCore.Slot(int, str)
    def _worker_progress(self, value: int, message: str) -> None:
        self.progress_bar.setValue(value)
        self.statusBar().showMessage(message)

    @QtCore.Slot(str)
    def _worker_failed(self, details: str) -> None:
        message = details.splitlines()[0]
        self.statusBar().showMessage(f"Error: {message}")
        QtWidgets.QMessageBox.critical(self, "Task failed", details)

    @QtCore.Slot()
    def _worker_finished(self) -> None:
        self.progress_bar.hide()
        self.controls.cancel_button.setEnabled(False)
        self._worker = None
        self._thread = None

    @QtCore.Slot()
    def cancel_worker(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self.statusBar().showMessage("Cancellation requested…")

    def export_csv(self) -> None:
        self._copy_export("trajectory_csv", "CSV files (*.csv)")

    def export_json(self) -> None:
        self._copy_export("trajectory_json", "JSON files (*.json)")

    def _copy_export(self, key: str, file_filter: str) -> None:
        if self.result is None:
            self._message("Generate paths before exporting.")
            return
        if key not in self.result.summary.output_files:
            suffix = ".csv" if key == "trajectory_csv" else ".json"
            destination, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Export staged trajectory", f"trajectory{suffix}", file_filter
            )
            if destination:
                export_staged_trajectory(destination, self.config, self.result.poses)
                self.statusBar().showMessage(f"Exported {destination}", 5000)
            return
        source = self.result.summary.output_files[key]
        destination, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export trajectory", source.name, file_filter
        )
        if destination:
            shutil.copyfile(source, destination)
            self.statusBar().showMessage(f"Exported {destination}", 5000)

    def _message(self, text: str) -> None:
        self.statusBar().showMessage(text, 5000)

    def _log_timings(self, timings: StageTimings) -> None:
        for stage, seconds in timings.seconds.items():
            line = f"{stage}: {seconds:.3f} s"
            self.log_view.appendPlainText(line)
            print(line)

    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()
        self.viewer.interactor.close()
        super().closeEvent(event)
