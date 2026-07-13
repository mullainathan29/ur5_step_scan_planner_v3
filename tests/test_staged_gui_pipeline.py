from __future__ import annotations

import tempfile
from dataclasses import replace
from pathlib import Path

import cadquery as cq

from app.config import PipelineConfig
from app.staged_pipeline import GeometryCache, generate_fast_paths
from gui.workers import PlanningWorker, WorkerResult


def _step(root: Path, name: str, size: float = 20.0) -> Path:
    path = root / name
    cq.exporters.export(
        cq.Workplane("XY").box(size, size * 0.8, size * 0.5).val(),
        str(path), exportType="STEP",
    )
    return path


def test_generate_paths_is_optical_free_and_returns_preview() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = _step(Path(temporary), "preview.step")
        cache = GeometryCache()
        config = PipelineConfig(
            step_path=path, planning_mode="simple_view_based", display=False,
            planning_quality="fast_preview",
        )
        result, timings = generate_fast_paths(cache, config)

        assert result.coverage is None
        assert result.optimisation_report is None
        assert result.recovery_candidates == ()
        assert result.poses
        assert cache.full_model is None
        assert {
            "STEP load", "face extraction", "validation mesh creation",
            "simple sweep generation", "hole detection", "pose generation",
        } <= timings.seconds.keys()
        assert "coverage simulation" not in timings.seconds
        assert "recovery generation" not in timings.seconds
        assert "mission optimisation" not in timings.seconds


def test_geometry_cache_survives_sweep_setting_changes() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = _step(Path(temporary), "cached.step")
        cache = GeometryCache()
        config = PipelineConfig(step_path=path, display=False)
        first, _ = generate_fast_paths(cache, config)
        brep = cache.brep
        mesh = cache.preview_model.validation_mesh

        second, timings = generate_fast_paths(
            cache, replace(config, stand_off_mm=320.0, sweep_overlap_fraction=0.3)
        )

        assert cache.brep is brep
        assert cache.preview_model.validation_mesh is mesh
        assert first.model is second.model
        assert timings.seconds["STEP load"] == 0.0
        assert timings.seconds["face extraction"] == 0.0
        assert timings.seconds["validation mesh creation"] == 0.0


def test_new_step_invalidates_all_geometry_cache_entries() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        first_path = _step(root, "first.step", 20)
        second_path = _step(root, "second.step", 30)
        cache = GeometryCache()
        generate_fast_paths(cache, PipelineConfig(step_path=first_path, display=False))
        first_brep = cache.brep
        generate_fast_paths(cache, PipelineConfig(step_path=second_path, display=False))
        assert cache.brep is not first_brep
        assert cache.path == second_path.resolve()
        assert cache.full_model is None


def test_planning_worker_reports_staged_result() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        path = _step(Path(temporary), "worker.step")
        worker = PlanningWorker(
            PipelineConfig(step_path=path, display=False), GeometryCache()
        )
        completed = []
        worker.completed.connect(completed.append)
        worker.run()
        assert isinstance(completed[0], WorkerResult)
        assert completed[0].value.coverage is None
