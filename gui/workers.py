"""QThread worker objects around existing non-GUI algorithms."""

from __future__ import annotations

import traceback
from dataclasses import replace

from PySide6 import QtCore

from app.pipeline import run_pipeline
from cad_model import load_step_model
from coverage import CoverageSettings, validate_coverage
from geometry_validation import validate_pose_geometry


class WorkerCancelled(RuntimeError):
    pass


class TaskWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    completed = QtCore.Signal(object)
    failed = QtCore.Signal(str)
    cancelled = QtCore.Signal()
    finished = QtCore.Signal()

    def __init__(self) -> None:
        super().__init__()
        self._cancel_requested = False

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def check_cancelled(self) -> None:
        if self._cancel_requested:
            raise WorkerCancelled()

    def execute(self):
        raise NotImplementedError

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self.check_cancelled()
            value = self.execute()
            self.check_cancelled()
            self.completed.emit(value)
        except WorkerCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(f"{exc}\n{traceback.format_exc()}")
        finally:
            self.finished.emit()


class StepLoadWorker(TaskWorker):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path

    def execute(self):
        self.progress.emit(10, "Loading STEP model")
        model = load_step_model(self.path)
        self.progress.emit(100, "STEP model loaded")
        return model


class PlanningWorker(TaskWorker):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def execute(self):
        self.progress.emit(5, "Planning exact B-Rep paths")
        result = run_pipeline(self.config)
        self.progress.emit(100, "Planning complete")
        return result


class CoverageWorker(TaskWorker):
    def __init__(self, result, config) -> None:
        super().__init__()
        self.result, self.config = result, config

    def execute(self):
        self.progress.emit(10, "Casting coverage rays")
        settings = CoverageSettings(
            laser_width_mm=self.config.laser_width_mm,
            laser_ray_count=self.config.laser_ray_count,
            min_range_mm=self.config.min_sensor_range_mm,
            max_range_mm=self.config.max_sensor_range_mm,
            max_incidence_angle_deg=self.config.max_incidence_angle_deg,
            include_internal_surfaces=self.config.include_internal_surfaces,
        )
        coverage = validate_coverage(self.result.model, self.result.poses, settings)
        self.progress.emit(100, "Coverage complete")
        return replace(self.result, coverage=coverage)


class ValidationWorker(TaskWorker):
    def __init__(self, result) -> None:
        super().__init__()
        self.result = result

    def execute(self):
        self.progress.emit(20, "Validating scanner geometry")
        validation = validate_pose_geometry(self.result.poses)
        self.progress.emit(100, "Validation complete")
        return replace(self.result, geometry_validation=validation)
