from __future__ import annotations

from pathlib import Path

import pytest
from PySide6 import QtWidgets

from app.configuration import ConfigurationError, pipeline_config_from_yaml
from gui.controls_panel import ControlsPanel


def _application():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_configuration_precedence_and_advanced_yaml(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text(
        "scanner:\n  nominal_standoff_mm: 300\n  travel_spacing_mm: 7\n"
        "planning:\n  mode: simple_view_based\n", encoding="utf-8"
    )
    (tmp_path / "standard.yaml").write_text(
        "scanner:\n  nominal_standoff_mm: 305\n", encoding="utf-8"
    )
    (tmp_path / "user_overrides.yaml").write_text(
        "scanner:\n  nominal_standoff_mm: 308\n", encoding="utf-8"
    )
    config = pipeline_config_from_yaml(
        "model.step", "standard", config_root=tmp_path,
        gui_overrides={"scanner": {"nominal_standoff_mm": 312}},
    )
    assert config.stand_off_mm == 312
    assert config.travel_spacing_mm == 7
    assert config.planning_quality == "standard"


def test_invalid_yaml_value_reports_exact_field(tmp_path: Path) -> None:
    (tmp_path / "default.yaml").write_text(
        "scanner:\n  preferred_range_mm: [190, 330]\n"
        "  absolute_range_mm: [200, 420]\n", encoding="utf-8"
    )
    (tmp_path / "fast_preview.yaml").write_text("{}\n", encoding="utf-8")
    (tmp_path / "user_overrides.yaml").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="scanner.preferred_range_mm"):
        pipeline_config_from_yaml("model.step", config_root=tmp_path)


def test_yaml_is_commented_and_gui_contains_only_basic_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("default.yaml", "fast_preview.yaml", "standard.yaml",
                 "final_validation.yaml", "user_overrides.yaml"):
        text = (root / "config" / name).read_text(encoding="utf-8")
        assert "#" in text
    _application()
    controls = ControlsPanel()
    text_widgets = controls.widget().findChildren(QtWidgets.QLabel)
    text_widgets += controls.widget().findChildren(QtWidgets.QAbstractButton)
    visible_text = " ".join(widget.text() for widget in text_widgets)
    for advanced in ("Standard laser rays", "Lane spacing", "Sweep overlap",
                     "Candidate tilt angles", "Bottom clearance",
                     "Connector stride"):
        assert advanced not in visible_text
    assert "Focused hole orbit" in visible_text


def test_every_visible_control_has_schema_tooltip_and_profile_updates() -> None:
    _application()
    controls = ControlsPanel()
    descendants = controls.widget().findChildren(QtWidgets.QWidget)
    interactive = [widget for widget in descendants if isinstance(widget, (
        QtWidgets.QAbstractButton, QtWidgets.QComboBox,
        QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox, QtWidgets.QTreeWidget,
    ))]
    assert interactive
    assert all(widget.toolTip() for widget in interactive)
    controls.planning_quality.setCurrentIndex(
        controls.planning_quality.findData("standard")
    )
    config = controls.planning_config(Path("model.step"))
    assert config.standard_laser_ray_count == 51
    assert config.focused_orbit_cone_angles_deg == (20.0, 35.0, 50.0)
    assert config.maximum_views_per_hole == 4
