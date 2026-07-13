from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6 import QtWidgets

from app.result import PipelineSummary
from gui.advanced_settings import ADVANCED_TABS, AdvancedSettingsDialog
from gui.controls_panel import ControlsPanel
from gui.main_window import MainWindow
from gui.workers import WorkerResult


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _visible_text(widget: QtWidgets.QWidget) -> str:
    labels = []
    for klass in (QtWidgets.QLabel, QtWidgets.QAbstractButton):
        labels.extend(child.text() for child in widget.findChildren(klass) if not child.isHidden())
    labels.extend(child.title() for child in widget.findChildren(QtWidgets.QGroupBox) if not child.isHidden())
    return " ".join(labels)


def test_main_gui_contains_only_operator_level_controls() -> None:
    _application()
    controls = ControlsPanel()
    text = _visible_text(controls.widget())

    for expected in (
        "Upload STEP",
        "Planning quality",
        "Stand-off distance",
        "Top scan",
        "Side scans",
        "Hole recovery",
        "Focused hole orbit",
        "Generate mission",
        "Validate mission",
        "Export CSV",
        "Export JSON",
        "CAD model",
        "Free-travel path",
        "Advanced settings...",
    ):
        assert expected in text

    for internal in (
        "Generate paths",
        "Run coverage",
        "Generate recovery",
        "Optimise mission",
        "Build continuous mission",
        "Transition waypoints",
        "Rejected smoothing candidates",
        "Minimum-clearance markers",
        "Face IDs",
    ):
        assert internal not in text


def test_advanced_settings_dialog_exposes_technical_tabs() -> None:
    _application()
    controls = ControlsPanel()
    dialog = AdvancedSettingsDialog(controls)
    tab_names = {dialog.tabs.tabText(index) for index in range(dialog.tabs.count())}

    assert set(ADVANCED_TABS) <= tab_names
    for field in (
        "standard_laser_ray_count",
        "lane_spacing_mm",
        "focused_orbit_cone_angles_deg",
        "tool_envelope_radius_mm",
        "free_travel_smoothing_method",
        "connector_stride",
    ):
        assert field in dialog.field_widgets


def test_quality_profile_switching_preserves_explicit_gui_edits_when_declined(monkeypatch: pytest.MonkeyPatch) -> None:
    _application()
    controls = ControlsPanel()
    controls.stand_off.setValue(333.0)
    monkeypatch.setattr(QtWidgets.QMessageBox, "exec", lambda self: QtWidgets.QMessageBox.StandardButton.No)

    controls.planning_quality.setCurrentIndex(controls.planning_quality.findData("standard"))

    assert controls.stand_off.value() == pytest.approx(333.0)
    assert controls.standard_laser_rays.value() == 51


def test_generate_mission_uses_full_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    _application()
    window = MainWindow()
    window.step_path = Path("model.step")
    started = []

    def fake_start(worker, completion):
        started.append(worker.__class__.__name__)

    monkeypatch.setattr(window, "_start_worker", fake_start)
    window.generate_mission()

    assert started == ["GenerateMissionWorker"]


def test_viewer_visibility_change_does_not_regenerate(monkeypatch: pytest.MonkeyPatch) -> None:
    _application()
    window = MainWindow()
    calls = []
    monkeypatch.setattr(window, "generate_mission", lambda: calls.append("generate"))

    window.controls.layer_checks["coverage"].setChecked(False)

    assert calls == []


def test_final_export_gating() -> None:
    _application()
    window = MainWindow()
    assert not window.controls.export_csv_button.isEnabled()
    assert not window.controls.export_json_button.isEnabled()

    summary = PipelineSummary(
        total_step_faces=0,
        selected_planar_faces=0,
        selected_cylindrical_faces=0,
        selected_conical_faces=0,
        selected_spherical_faces=0,
        skipped_faces_by_reason={},
        generated_scan_regions=0,
        generated_scan_segments=0,
        scan_pose_count=0,
        total_scan_path_length_mm=0.0,
        coverage_percent=0.0,
        invalid_segment_count=0,
        output_files={"mission_validation_report": {"valid": True}},
    )
    window.result = SimpleNamespace(
        poses=(SimpleNamespace(global_pose_id=0, scan_status=0, transition_strategy="direct"),),
        geometry_validation=object(),
        summary=summary,
    )
    window._update_export_state()

    assert window.controls.export_csv_button.isEnabled()
    assert window.controls.export_json_button.isEnabled()


def test_debug_layers_hidden_by_default() -> None:
    _application()
    controls = ControlsPanel()
    for key in (
        "mission_smoothing_control_points",
        "rejected_direct_transitions",
        "min_clearance_markers",
        "labels",
    ):
        assert key in controls.layer_checks
        assert not controls.layer_checks[key].isVisible()
        assert not controls.layer_checks[key].isChecked()
