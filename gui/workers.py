"""Cancellable QThread stages for the responsive desktop workflow."""

from __future__ import annotations

import traceback
from dataclasses import dataclass, replace
from time import perf_counter

from PySide6 import QtCore

from app.staged_pipeline import (
    GeometryCache, StageTimings, generate_fast_paths,
    generate_recovery_candidates,
)
from coverage import CoverageSettings, validate_coverage
from features import analyse_hole_coverage
from geometry_validation import validate_pose_geometry
from mission_optimizer import optimise_scan_mission
from recovery import RecoveryCandidate


class WorkerCancelled(RuntimeError):
    pass


@dataclass(frozen=True)
class WorkerResult:
    value: object
    timings: StageTimings


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
    def __init__(self, path, cache: GeometryCache) -> None:
        super().__init__()
        self.path, self.cache = path, cache

    def execute(self):
        timings = StageTimings()
        model = self.cache.ensure_preview(
            self.path, timings, self.progress.emit, self.check_cancelled
        )
        self.progress.emit(100, "STEP preview loaded")
        return WorkerResult(model, timings)


class PlanningWorker(TaskWorker):
    def __init__(self, config, cache: GeometryCache) -> None:
        super().__init__()
        self.config, self.cache = config, cache

    def execute(self):
        result, timings = generate_fast_paths(
            self.cache, self.config, self.progress.emit, self.check_cancelled
        )
        self.progress.emit(100, "Path preview complete (optical simulation skipped)")
        return WorkerResult(result, timings)


class CoverageWorker(TaskWorker):
    def __init__(self, result, config, cache: GeometryCache) -> None:
        super().__init__()
        self.result, self.config, self.cache = result, config, cache

    def execute(self):
        timings = StageTimings()
        if self.config.planning_quality == "final_validation":
            model = self.cache.ensure_full(
                self.config.step_path, timings, self.progress.emit,
                self.check_cancelled,
            )
            ray_count = self.config.final_laser_ray_count
        else:
            model = self.cache.ensure_preview(
                self.config.step_path, timings, self.progress.emit,
                self.check_cancelled,
            )
            ray_count = self.config.standard_laser_ray_count
        self.progress.emit(35, f"Coverage simulation ({ray_count} rays/profile)")
        started = perf_counter()
        settings = CoverageSettings(
            laser_width_mm=self.config.laser_width_mm,
            laser_ray_count=ray_count,
            min_range_mm=self.config.min_sensor_range_mm,
            max_range_mm=self.config.max_sensor_range_mm,
            max_incidence_angle_deg=self.config.max_incidence_angle_deg,
            include_internal_surfaces=(self.config.include_internal_surfaces
                                       or self.config.enable_hole_recovery),
        )
        coverage = validate_coverage(
            model, self.result.poses, settings,
            cancellation_check=self.check_cancelled,
            progress_callback=lambda done, total: self.progress.emit(
                35 + int(60 * done / max(total, 1)),
                f"Coverage poses {done}/{total}",
            ),
        )
        timings.record("coverage simulation", perf_counter() - started)
        reports = tuple(analyse_hole_coverage(
            model, hole, coverage, self.config.hole_target_coverage_percent
        ) for hole in self.result.hole_features)
        self.progress.emit(100, "Coverage complete")
        return WorkerResult(replace(
            self.result, model=model, coverage=coverage,
            hole_coverage_reports=reports,
        ), timings)


class FeatureRecoveryWorker(TaskWorker):
    def __init__(self, result, config) -> None:
        super().__init__()
        self.result, self.config = result, config

    def execute(self):
        result, timings = generate_recovery_candidates(
            self.result, self.config, self.progress.emit, self.check_cancelled
        )
        self.progress.emit(100, f"Generated {len(result.recovery_candidates)} recovery candidates")
        return WorkerResult(result, timings)


class OptimisationWorker(TaskWorker):
    def __init__(self, result, config) -> None:
        super().__init__()
        self.result, self.config = result, config

    def execute(self):
        if self.result.coverage is None:
            raise ValueError("Run coverage before optimising the mission.")
        timings = StageTimings()
        mandatory = tuple(sweep for sweep in self.result.sweeps if sweep.view_type in {
            "top", "front", "right", "back", "left"
        })
        optional = [RecoveryCandidate(
            sweep.sweep_id, sweep.sweep_name, sweep.view_type, sweep
        ) for sweep in self.result.sweeps if sweep not in mandatory]
        optional.extend(self.result.recovery_candidates)
        settings = CoverageSettings(
            laser_width_mm=self.config.laser_width_mm,
            laser_ray_count=(self.config.final_laser_ray_count
                             if self.config.planning_quality == "final_validation"
                             else self.config.standard_laser_ray_count),
            min_range_mm=self.config.min_sensor_range_mm,
            max_range_mm=self.config.max_sensor_range_mm,
            max_incidence_angle_deg=self.config.max_incidence_angle_deg,
            include_internal_surfaces=True,
        )
        started = perf_counter()
        optimisation = optimise_scan_mission(
            self.result.model, mandatory, tuple(optional),
            self.result.hole_features, settings,
            target_total_coverage_percent=self.config.total_target_coverage_percent,
            target_hole_coverage_percent=self.config.hole_target_coverage_percent,
            minimum_gain_percent=self.config.minimum_candidate_gain_percent,
            maximum_optional_sweeps=self.config.maximum_optional_sweeps,
            floating_part_assumption=not self.config.fixture_exclusion_enabled,
            cancellation_check=self.check_cancelled,
            progress_callback=lambda done, total: self.progress.emit(
                10 + int(80 * done / max(total, 1)),
                f"Scoring candidate {done + 1}/{total}",
            ),
        )
        timings.record("mission optimisation", perf_counter() - started)
        poses = tuple(pose for sweep in optimisation.sweeps for pose in sweep.poses)
        result = replace(
            self.result, sweeps=optimisation.sweeps, poses=poses,
            coverage=optimisation.coverage,
            hole_coverage_reports=optimisation.hole_reports_after,
            optimisation_report=optimisation,
        )
        self.progress.emit(100, "Mission optimisation complete")
        return WorkerResult(result, timings)


class ValidationWorker(TaskWorker):
    def __init__(self, result, config=None, cache: GeometryCache | None = None) -> None:
        super().__init__()
        self.result, self.config, self.cache = result, config, cache

    def execute(self):
        timings = StageTimings()
        result = self.result
        if (self.config is not None and self.cache is not None
                and self.config.planning_quality == "final_validation"):
            model = self.cache.ensure_full(
                self.config.step_path, timings, self.progress.emit,
                self.check_cancelled,
            )
            result = replace(result, model=model)
        self.progress.emit(20, "Geometry validation")
        started = perf_counter()
        self.check_cancelled()
        validation = validate_pose_geometry(result.poses)
        self.check_cancelled()
        timings.record("geometry validation", perf_counter() - started)
        self.progress.emit(100, "Validation complete")
        return WorkerResult(replace(
            result, geometry_validation=validation
        ), timings)
