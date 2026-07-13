from pathlib import Path

from PySide6 import QtWidgets

from gui.controls_panel import ControlsPanel
from gui.viewer_widget import ActorRecord, LayerController
from gui.workers import TaskWorker


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


class _Actor:
    def __init__(self):
        self.visible = None

    def SetVisibility(self, value):
        self.visible = bool(value)


def test_control_defaults() -> None:
    _application()
    controls = ControlsPanel()
    config = controls.planning_config(Path("model.step"))
    assert config.planning_mode == "simple_view_based"
    assert config.planning_quality == "fast_preview"
    assert config.standard_laser_ray_count == 21
    assert config.final_laser_ray_count == 101
    assert config.hole_target_coverage_percent == 85.0
    assert config.maximum_views_per_hole == 2
    assert config.hole_candidate_tilt_angles_deg == (25.0, 40.0)
    assert config.hole_candidate_azimuth_count == 2
    assert config.orbital_hole_mode == "disabled"
    assert config.minimum_candidate_gain_percent == 1.0
    assert config.maximum_optional_sweeps == 3
    assert config.total_target_coverage_percent == 90.0


def test_standard_quality_recovery_profile() -> None:
    _application()
    controls = ControlsPanel()
    controls.planning_quality.setCurrentIndex(
        controls.planning_quality.findData("standard")
    )
    config = controls.planning_config(Path("model.step"))
    assert config.hole_target_coverage_percent == 90.0
    assert config.maximum_views_per_hole == 4
    assert config.hole_candidate_tilt_angles_deg == (20.0, 35.0, 50.0)
    assert config.hole_candidate_azimuth_count == 4
    assert config.orbital_hole_mode == "adaptive"
    assert config.maximum_optional_sweeps == 5
    assert config.minimum_candidate_gain_percent == 0.5
    assert config.surface_types == ("plane", "cylinder")
    assert config.lane_spacing_mm == 20.0
    assert config.travel_spacing_mm == 5.0
    assert config.stand_off_mm == 310.0
    assert config.minimum_face_area_mm2 == 25.0
    assert config.include_internal_surfaces is False
    assert controls.connector_stride.value() == 10
    assert controls.frame_stride.value() == 20


def test_region_filtering_does_not_replan() -> None:
    controller = LayerController()
    first, second = _Actor(), _Actor()
    controller.add(ActorRecord(first, "nominal", 1, frozenset({4}), "plane"))
    controller.add(ActorRecord(second, "nominal", 2, frozenset({5}), "cylinder"))
    controller.set_filter(region_id=2, only_selected_region=True)
    assert first.visible is False
    assert second.visible is True


def test_layer_visibility() -> None:
    controller = LayerController()
    actor = _Actor()
    controller.add(ActorRecord(actor, "scanner", 1, frozenset({4}), "plane"))
    controller.set_layer("scanner", False)
    assert actor.visible is False
    controller.set_layer("scanner", True)
    assert actor.visible is True


class _SuccessfulWorker(TaskWorker):
    def execute(self):
        return "done"


class _FailingWorker(TaskWorker):
    def execute(self):
        raise RuntimeError("worker exploded")


def test_worker_completion() -> None:
    worker = _SuccessfulWorker()
    completed, finished = [], []
    worker.completed.connect(completed.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert completed == ["done"]
    assert finished == [True]


def test_worker_error_handling() -> None:
    worker = _FailingWorker()
    errors, finished = [], []
    worker.failed.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert "worker exploded" in errors[0]
    assert finished == [True]


def test_no_pyvista_in_scene_controls_remain() -> None:
    root = Path(__file__).resolve().parents[1]
    sources = [root / "viewer.py", root / "app" / "pipeline.py"]
    sources.extend((root / "gui").glob("*.py"))
    forbidden = (
        "add_text_slider_widget",
        "add_slider_widget",
        "add_checkbox_button_widget",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in sources)
    assert not any(name in text for name in forbidden)
