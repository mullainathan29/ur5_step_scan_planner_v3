"""Documented YAML configuration loading and schema metadata."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from app.config import PipelineConfig
from planners.base import CylinderStrategy


CONFIG_ROOT = Path(__file__).resolve().parents[1] / "config"
QUALITY_PROFILES = frozenset({"fast_preview", "standard", "final_validation"})
PLANNING_MODES = frozenset({"simple_view_based", "advanced_face_based"})


@dataclass(frozen=True)
class ParameterMetadata:
    path: str
    config_field: str
    display_name: str
    tooltip: str
    units: str = ""
    valid_range: tuple[float, float] | None = None
    classification: str = "advanced"


PARAMETERS = {
    row.config_field: row for row in (
        ParameterMetadata("planning.quality", "planning_quality", "Planning quality",
                          "Selects validation detail and recovery effort.", classification="basic"),
        ParameterMetadata("planning.mode", "planning_mode", "Planning mode",
                          "Uses principal projected sweeps or exact face-based planning.", classification="basic"),
        ParameterMetadata("scanner.nominal_standoff_mm", "stand_off_mm", "Stand-off",
                          "Preferred distance from the scanner reference point to the visible part surface. Recommended value for LLT30x0-200: 310 mm.",
                          "mm", (1.0, 10000.0), "basic"),
        ParameterMetadata("scanner.travel_spacing_mm", "travel_spacing_mm", "Travel spacing",
                          "Distance between successive robot poses during scanning. Smaller values increase profile overlap and trajectory size.",
                          "mm", (0.01, 10000.0), "basic"),
        ParameterMetadata("scanner.laser_line_width_mm", "laser_width_mm", "Laser-line width",
                          "Usable width of one X-Z profile at nominal stand-off. If the part projection fits inside this width, one sweep is sufficient.",
                          "mm", (0.01, 10000.0), "basic"),
        ParameterMetadata("principal_coverage.top_sweep", "top_sweep", "Top sweep",
                          "Generates the principal sweep above the upright model.", classification="basic"),
        ParameterMetadata("principal_coverage.side_sweeps", "side_sweeps", "Side sweeps",
                          "Enables front, right, back and left principal sweeps together.", classification="basic"),
        ParameterMetadata("principal_coverage.upper_isometric_recovery", "isometric_recovery", "Upper isometric recovery",
                          "Adds upper angled views for exterior regions missed by principal sweeps.", classification="basic"),
        ParameterMetadata("principal_coverage.bottom_coverage", "enable_bottom_sweep", "Bottom/underside coverage",
                          "Allows underside and lower recovery sweeps.", classification="basic"),
        ParameterMetadata("hole_recovery.enabled", "enable_hole_recovery", "Enable hole recovery",
                          "Enables feature-specific recovery for physical holes below target coverage.", classification="basic"),
        ParameterMetadata("hole_recovery.target_coverage_percent", "hole_target_coverage_percent", "Hole target coverage",
                          "Required cylindrical-wall coverage for each detected physical hole before feature recovery stops.",
                          "%", (0.0, 100.0), "basic"),
        ParameterMetadata("hole_recovery.focused_conical_orbit.enabled", "enable_focused_hole_orbit", "Focused hole orbit",
                          "Moves the scanner along an angled arc around a hole while continuously aiming toward a focus point inside the bore.", classification="basic"),
        ParameterMetadata("mission.total_target_coverage_percent", "total_target_coverage_percent", "Total target coverage",
                          "Target exterior-surface coverage used to stop optional mission recovery.", "%", (0.0, 100.0), "basic"),
        ParameterMetadata("transition_planning.enable_safe_traverse", "enable_safe_traverse", "Enable safe traverse",
                          "Routes free-travel motion around the part safety envelope instead of connecting scan blocks directly.", classification="basic"),
        ParameterMetadata("transition_planning.safety_clearance_mm", "safety_clearance_mm", "Safety clearance",
                          "Minimum free-travel separation between the scanner/tool envelope and the part safety envelope.",
                          "mm", (0.0, 10000.0), "basic"),
        ParameterMetadata("free_travel_smoothing.enabled", "free_travel_smoothing_enabled", "Smooth free travel",
                          "Blends collision-safe transition waypoints into continuous curves while preserving obstacle clearance.",
                          classification="basic"),
        ParameterMetadata("validation.standard_laser_ray_count", "standard_laser_ray_count", "Standard laser rays",
                          "Number of simulated profile rays used for Standard coverage validation."),
        ParameterMetadata("validation.final_laser_ray_count", "final_laser_ray_count", "Final laser rays",
                          "Number of simulated profile rays used for Final validation."),
        ParameterMetadata("validation.maximum_incidence_angle_deg", "max_incidence_angle_deg", "Maximum incidence angle",
                          "Rejects visible hits where the laser ray meets the surface too obliquely.", "deg"),
        ParameterMetadata("planning.lane_spacing_mm", "lane_spacing_mm", "Lane spacing",
                          "Lane separation used by exact face-based raster planners.", "mm"),
        ParameterMetadata("planning.minimum_face_area_mm2", "minimum_face_area_mm2", "Minimum face area",
                          "Faces smaller than this threshold are ignored by exact face-based planning.", "mm2"),
        ParameterMetadata("planning.surface_types", "surface_types", "Surface taxonomy",
                          "Exact CAD surface types eligible for face-based planning."),
        ParameterMetadata("planning.cylinder_strategy", "cylinder_strategy", "Cylinder strategy",
                          "Longitudinal or helical strategy for exact cylindrical faces."),
        ParameterMetadata("principal_coverage.sweep_overlap_fraction", "sweep_overlap_fraction", "Sweep overlap",
                          "Fractional overlap between adjacent projected sweep lanes."),
        ParameterMetadata("principal_coverage.isometric_azimuth_step_deg", "isometric_azimuth_step_deg", "Isometric azimuth spacing",
                          "Angular spacing for upper isometric recovery candidates.", "deg"),
        ParameterMetadata("hole_recovery.maximum_views_per_hole", "maximum_views_per_hole", "Maximum recovery views",
                          "Maximum feature-recovery views retained for a physical hole."),
        ParameterMetadata("hole_recovery.candidate_azimuth_count", "hole_candidate_azimuth_count", "Candidate azimuth count",
                          "Azimuth count for legacy cross-hole recovery candidates."),
        ParameterMetadata("hole_recovery.minimum_candidate_gain_percent", "minimum_candidate_gain_percent", "Minimum candidate gain",
                          "Minimum coverage improvement needed for an optional candidate.", "%"),
        ParameterMetadata("hole_recovery.blind_bottom_enabled", "enable_blind_hole_bottom_recovery", "Blind bottom recovery",
                          "Enables bottom-specific candidates for blind holes."),
        ParameterMetadata("hole_recovery.lower_isometric_enabled", "enable_lower_isometric_recovery", "Lower isometric recovery",
                          "Enables lower angled views around the underside."),
        ParameterMetadata("hole_recovery.maximum_lower_recovery_sweeps", "maximum_lower_recovery_sweeps", "Maximum lower sweeps",
                          "Maximum lower recovery sweeps generated before optimisation."),
        ParameterMetadata("hole_recovery.focused_conical_orbit.candidate_cone_angles_deg", "focused_orbit_cone_angles_deg", "Candidate cone angles",
                          "Candidate scanner angles for focused conical hole orbits.", "deg"),
        ParameterMetadata("hole_recovery.focused_conical_orbit.minimum_arc_deg", "focused_orbit_minimum_arc_deg", "Minimum orbit arc",
                          "Minimum focused-orbit arc length considered useful.", "deg"),
        ParameterMetadata("hole_recovery.focused_conical_orbit.maximum_arc_deg", "focused_orbit_maximum_arc_deg", "Maximum orbit arc",
                          "Maximum focused-orbit arc before full-orbit behavior would be needed.", "deg"),
        ParameterMetadata("mission.maximum_optional_sweeps", "maximum_optional_sweeps", "Maximum optional sweeps",
                          "Maximum optional recovery sweeps selected during mission optimisation."),
        ParameterMetadata("transition_planning.envelope_type", "envelope_type", "Envelope type",
                          "Safety envelope type: OBB or AABB."),
        ParameterMetadata("transition_planning.tool_envelope_radius_mm", "tool_envelope_radius_mm", "Tool-envelope radius",
                          "Approximate scanner/tool radius added to free-travel clearance.", "mm"),
        ParameterMetadata("transition_planning.vertical_clearance_mm", "vertical_clearance_mm", "Vertical clearance",
                          "Additional global-Z clearance used for top/bottom traverse routes.", "mm"),
        ParameterMetadata("transition_planning.corner_penalty", "corner_penalty", "Corner penalty",
                          "Cost penalty applied to routed safety-envelope graph turns."),
        ParameterMetadata("transition_planning.vertical_motion_penalty", "vertical_motion_penalty", "Vertical motion penalty",
                          "Cost penalty for vertical travel in free-space routing."),
        ParameterMetadata("transition_planning.proximity_penalty", "proximity_penalty", "Proximity penalty",
                          "Cost penalty that biases routes away from the CAD mesh."),
        ParameterMetadata("transition_planning.collision_sampling_interval_mm", "collision_sampling_interval_mm", "Collision sample spacing",
                          "Distance between samples used while validating transition segment clearance.", "mm"),
        ParameterMetadata("transition_planning.graph_waypoint_density", "graph_waypoint_density", "Graph waypoint density",
                          "Density hint for safety-envelope waypoint routing."),
        ParameterMetadata("free_travel_smoothing.method", "free_travel_smoothing_method", "Spline method",
                          "Preferred curve fitting method for free-travel smoothing."),
        ParameterMetadata("free_travel_smoothing.nominal_blend_radius_mm", "nominal_blend_radius_mm", "Blend radius",
                          "Nominal radius for corner blending before automatic reduction.", "mm"),
        ParameterMetadata("free_travel_smoothing.minimum_blend_radius_mm", "minimum_blend_radius_mm", "Minimum blend radius",
                          "Smallest blend radius attempted when clearance is tight.", "mm"),
        ParameterMetadata("free_travel_smoothing.sample_spacing_mm", "sample_spacing_mm", "Smoothing sample spacing",
                          "Target pose spacing for smoothed free-travel curves.", "mm"),
        ParameterMetadata("free_travel_smoothing.maximum_retries", "maximum_retries", "Smoothing retries",
                          "Maximum number of reduced-radius smoothing attempts."),
        ParameterMetadata("orientation_smoothing.maximum_step_deg", "orientation_smoothing_maximum_step_deg", "Orientation step",
                          "Maximum orientation change between adjacent free-travel poses.", "deg"),
        ParameterMetadata("orientation_smoothing.interpolation_window_fraction", "orientation_interpolation_window_fraction", "Interpolation window",
                          "Fraction of a free-travel route over which orientation SLERP is active."),
        ParameterMetadata("viewer.connector_stride", "connector_stride", "Connector stride",
                          "Draw every Nth connector line in debug viewer mode."),
        ParameterMetadata("viewer.frame_stride", "frame_stride", "Frame stride",
                          "Draw every Nth scanner coordinate frame in debug viewer mode."),
        ParameterMetadata("viewer.laser_footprint_stride", "laser_footprint_stride", "Laser footprint stride",
                          "Draw every Nth laser footprint in debug viewer mode."),
        ParameterMetadata("free_travel.enabled", "free_travel_enabled", "Enable free travel",
                          "Enables collision-free routing and smoothing for free travel.", classification="basic"),
        ParameterMetadata("free_travel.forbidden_region_margin_mm", "forbidden_region_margin_mm", "Forbidden margin",
                          "Margin added around the CAD part to form the forbidden region.", "mm"),
        ParameterMetadata("free_travel.tool_envelope_radius_mm", "free_travel_tool_envelope_radius_mm", "Tool envelope radius",
                          "Effective radius representing the scanner/tool envelope.", "mm"),
        ParameterMetadata("free_travel.effective_minimum_clearance_mm", "effective_minimum_clearance_mm", "Effective minimum clearance",
                          "Effective minimum clearance threshold for mission validation.", "mm"),
        ParameterMetadata("free_travel.tool_radius_mm", "tool_radius_mm", "Tool radius",
                          "Effective radius representing the scanner/tool envelope.", "mm"),
        ParameterMetadata("free_travel.collision_sample_spacing_mm", "collision_sample_spacing_mm", "Collision sample spacing",
                          "Distance between segment samples for collision/clearance validation of free travel.", "mm"),
        ParameterMetadata("free_travel.maximum_curvature_1_per_mm", "maximum_curvature_1_per_mm", "Max curvature",
                          "Maximum allowed curvature for robot-friendly free travel.", "1/mm"),
        ParameterMetadata("free_travel.maximum_control_points", "maximum_control_points", "Max control points",
                          "Maximum allowed control points for transition curves."),
    )
}


class ConfigurationError(ValueError):
    """A configuration value failed schema validation."""


def tooltip(config_field: str) -> str:
    return PARAMETERS[config_field].tooltip


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"Invalid YAML root in {path}: expected a mapping.")
    return value


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            output.update(_flatten(child, path))
        else:
            output[path] = child
    return output


PATH_TO_FIELD = {
    "planning.mode": "planning_mode",
    "planning.quality": "planning_quality",
    "planning.lane_spacing_mm": "lane_spacing_mm",
    "planning.minimum_face_area_mm2": "minimum_face_area_mm2",
    "planning.cylinder_strategy": "cylinder_strategy",
    "planning.surface_types": "surface_types",
    "scanner.nominal_standoff_mm": "stand_off_mm",
    "scanner.travel_spacing_mm": "travel_spacing_mm",
    "scanner.laser_line_width_mm": "laser_width_mm",
    "scanner.preferred_range_mm": "preferred_range_mm",
    "scanner.absolute_range_mm": "absolute_range_mm",
    "validation.standard_laser_ray_count": "standard_laser_ray_count",
    "validation.final_laser_ray_count": "final_laser_ray_count",
    "validation.maximum_incidence_angle_deg": "max_incidence_angle_deg",
    "principal_coverage.top_sweep": "top_sweep",
    "principal_coverage.side_sweeps": "side_sweeps",
    "principal_coverage.upper_isometric_recovery": "isometric_recovery",
    "principal_coverage.bottom_coverage": "bottom_coverage",
    "principal_coverage.sweep_overlap_fraction": "sweep_overlap_fraction",
    "principal_coverage.isometric_azimuth_step_deg": "isometric_azimuth_step_deg",
    "hole_recovery.enabled": "enable_hole_recovery",
    "hole_recovery.target_coverage_percent": "hole_target_coverage_percent",
    "hole_recovery.maximum_views_per_hole": "maximum_views_per_hole",
    "hole_recovery.candidate_azimuth_count": "hole_candidate_azimuth_count",
    "hole_recovery.orbital_mode": "orbital_hole_mode",
    "hole_recovery.minimum_candidate_gain_percent": "minimum_candidate_gain_percent",
    "hole_recovery.focused_conical_orbit.enabled": "enable_focused_hole_orbit",
    "hole_recovery.focused_conical_orbit.candidate_cone_angles_deg": "focused_orbit_cone_angles_deg",
    "hole_recovery.focused_conical_orbit.focus_depth_fraction": "focused_orbit_focus_depth_fraction",
    "hole_recovery.focused_conical_orbit.angular_spacing_deg": "focused_orbit_angular_spacing_deg",
    "hole_recovery.focused_conical_orbit.minimum_arc_deg": "focused_orbit_minimum_arc_deg",
    "hole_recovery.focused_conical_orbit.maximum_arc_deg": "focused_orbit_maximum_arc_deg",
    "hole_recovery.focused_conical_orbit.maximum_orbits_per_hole": "maximum_focused_orbits_per_hole",
    "hole_recovery.blind_bottom_enabled": "enable_blind_hole_bottom_recovery",
    "hole_recovery.lower_isometric_enabled": "enable_lower_isometric_recovery",
    "hole_recovery.maximum_lower_recovery_sweeps": "maximum_lower_recovery_sweeps",
    "mission.total_target_coverage_percent": "total_target_coverage_percent",
    "mission.maximum_optional_sweeps": "maximum_optional_sweeps",
    "viewer.connector_stride": "connector_stride",
    "viewer.frame_stride": "frame_stride",
    "viewer.laser_footprint_stride": "laser_footprint_stride",
    "transition_planning.safety_clearance_mm": "safety_clearance_mm",
    "transition_planning.vertical_clearance_mm": "vertical_clearance_mm",
    "transition_planning.tool_envelope_radius_mm": "tool_envelope_radius_mm",
    "transition_planning.envelope_type": "envelope_type",
    "transition_planning.enable_safe_traverse": "enable_safe_traverse",
    "transition_planning.prefer_top_clearance": "prefer_top_clearance",
    "transition_planning.show_safety_envelope": "show_safety_envelope",
    "transition_planning.orientation_weight": "orientation_weight",
    "transition_planning.vertical_motion_penalty": "vertical_motion_penalty",
    "transition_planning.corner_penalty": "corner_penalty",
    "transition_planning.proximity_penalty": "proximity_penalty",
    "transition_planning.graph_waypoint_density": "graph_waypoint_density",
    "transition_planning.collision_sampling_interval_mm": "collision_sampling_interval_mm",
    "free_travel_smoothing.enabled": "free_travel_smoothing_enabled",
    "free_travel_smoothing.method": "free_travel_smoothing_method",
    "free_travel_smoothing.nominal_blend_radius_mm": "nominal_blend_radius_mm",
    "free_travel_smoothing.minimum_blend_radius_mm": "minimum_blend_radius_mm",
    "free_travel_smoothing.sample_spacing_mm": "sample_spacing_mm",
    "free_travel_smoothing.maximum_route_deviation_mm": "maximum_route_deviation_mm",
    "free_travel_smoothing.maximum_retries": "maximum_retries",
    "orientation_smoothing.enabled": "orientation_smoothing_enabled",
    "orientation_smoothing.maximum_step_deg": "orientation_smoothing_maximum_step_deg",
    "orientation_smoothing.interpolation_window_fraction": "orientation_interpolation_window_fraction",
    "free_travel.enabled": "free_travel_enabled",
    "free_travel.forbidden_region_margin_mm": "forbidden_region_margin_mm",
    "free_travel.tool_envelope_radius_mm": "free_travel_tool_envelope_radius_mm",
    "free_travel.effective_minimum_clearance_mm": "effective_minimum_clearance_mm",
    "free_travel.tool_radius_mm": "tool_radius_mm",
    "free_travel.collision_sample_spacing_mm": "collision_sample_spacing_mm",
    "free_travel.maximum_curvature_1_per_mm": "maximum_curvature_1_per_mm",
    "free_travel.maximum_control_points": "maximum_control_points",
}


def load_configuration_values(
    quality: str,
    *,
    config_root: Path = CONFIG_ROOT,
    gui_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if quality not in QUALITY_PROFILES:
        raise ConfigurationError(
            f"planning.quality={quality!r}; allowed values: {sorted(QUALITY_PROFILES)}"
        )
    merged = _read_yaml(config_root / "default.yaml")
    merged = _deep_merge(merged, _read_yaml(config_root / f"{quality}.yaml"))
    merged = _deep_merge(merged, _read_yaml(config_root / "user_overrides.yaml"))
    if gui_overrides:
        merged = _deep_merge(merged, gui_overrides)
    return merged


def pipeline_config_from_yaml(
    step_path: str | Path,
    quality: str = "fast_preview",
    *,
    config_root: Path = CONFIG_ROOT,
    gui_overrides: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> PipelineConfig:
    values = load_configuration_values(
        quality, config_root=config_root, gui_overrides=gui_overrides
    )
    flat = _flatten(values)
    kwargs: dict[str, Any] = {"step_path": Path(step_path), "output_dir": output_dir}
    valid_fields = {row.name for row in fields(PipelineConfig)}
    for path, value in flat.items():
        field = PATH_TO_FIELD.get(path)
        if field is None:
            continue
        if field == "preferred_range_mm":
            try:
                kwargs["preferred_min_range_mm"], kwargs["preferred_max_range_mm"] = map(float, value)
            except Exception as exc:
                raise ConfigurationError(
                    f"{path}={value!r}; expected [minimum, maximum]."
                ) from exc
        elif field == "absolute_range_mm":
            try:
                kwargs["min_sensor_range_mm"], kwargs["max_sensor_range_mm"] = map(float, value)
            except Exception as exc:
                raise ConfigurationError(
                    f"{path}={value!r}; expected [minimum, maximum]."
                ) from exc
        elif field == "side_sweeps":
            kwargs.update({name: bool(value) for name in (
                "front_sweep", "right_sweep", "back_sweep", "left_sweep"
            )})
        elif field == "bottom_coverage":
            kwargs["enable_bottom_sweep"] = bool(value)
            kwargs["enable_lower_isometric_recovery"] = bool(value)
        elif field == "cylinder_strategy":
            kwargs[field] = CylinderStrategy(value)
        elif field in {"surface_types", "focused_orbit_cone_angles_deg"}:
            kwargs[field] = tuple(value)
            if field == "focused_orbit_cone_angles_deg":
                kwargs["hole_candidate_tilt_angles_deg"] = tuple(value)
        elif field == "orbital_hole_mode":
            kwargs[field] = value
            kwargs["enable_orbital_hole_sweep"] = value != "disabled"
        elif field in valid_fields:
            kwargs[field] = value
    kwargs["planning_quality"] = quality
    try:
        config = PipelineConfig(**kwargs)
        config.validate()
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(str(exc)) from exc
    if config.planning_mode not in PLANNING_MODES:
        raise ConfigurationError(
            f"planning.mode={config.planning_mode!r}; allowed values: {sorted(PLANNING_MODES)}"
        )
    for path, value, low, high in (
        ("hole_recovery.target_coverage_percent", config.hole_target_coverage_percent, 0, 100),
        ("mission.total_target_coverage_percent", config.total_target_coverage_percent, 0, 100),
    ):
        if not low <= value <= high:
            raise ConfigurationError(f"{path}={value!r}; allowed range: {low}–{high}.")
    return config


def advanced_summary(config: PipelineConfig) -> str:
    rows = []
    for field in fields(config):
        metadata = PARAMETERS.get(field.name)
        if metadata is None or metadata.classification == "advanced":
            rows.append(f"{field.name}: {getattr(config, field.name)}")
    return "\n".join(rows)
