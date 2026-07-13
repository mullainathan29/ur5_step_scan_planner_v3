"""Configuration for the unified B-Rep planning application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from planners.base import CylinderStrategy


VALID_SURFACE_TYPES = frozenset({"plane", "cylinder", "cone", "sphere"})


RECOVERY_QUALITY_PROFILES = {
    "fast_preview": {
        "hole_target_coverage_percent": 85.0,
        "maximum_views_per_hole": 2,
        "hole_candidate_tilt_angles_deg": (25.0, 40.0),
        "hole_candidate_azimuth_count": 2,
        "orbital_hole_mode": "disabled",
        "maximum_optional_sweeps": 3,
        "minimum_candidate_gain_percent": 1.0,
        "total_target_coverage_percent": 90.0,
    },
    "standard": {
        "hole_target_coverage_percent": 90.0,
        "maximum_views_per_hole": 4,
        "hole_candidate_tilt_angles_deg": (20.0, 35.0, 50.0),
        "hole_candidate_azimuth_count": 4,
        "orbital_hole_mode": "adaptive",
        "maximum_optional_sweeps": 5,
        "minimum_candidate_gain_percent": 0.5,
        "total_target_coverage_percent": 95.0,
    },
    "final_validation": {
        "hole_target_coverage_percent": 90.0,
        "maximum_views_per_hole": 8,
        "hole_candidate_tilt_angles_deg": (20.0, 35.0, 50.0),
        "hole_candidate_azimuth_count": 4,
        "orbital_hole_mode": "enabled",
        "maximum_optional_sweeps": 8,
        "minimum_candidate_gain_percent": 0.5,
        "total_target_coverage_percent": 95.0,
    },
}


def recovery_quality_profile(quality: str) -> dict:
    try:
        return dict(RECOVERY_QUALITY_PROFILES[quality])
    except KeyError as exc:
        raise ValueError(f"Unsupported planning quality: {quality}.") from exc


@dataclass(frozen=True)
class PipelineConfig:
    step_path: Path
    planning_mode: str = "advanced_face_based"
    surface_types: tuple[str, ...] = ("plane", "cylinder")
    lane_spacing_mm: float = 20.0
    travel_spacing_mm: float = 5.0
    stand_off_mm: float = 310.0
    minimum_face_area_mm2: float = 25.0
    include_internal_surfaces: bool = False
    cylinder_strategy: CylinderStrategy = CylinderStrategy.LONGITUDINAL
    display: bool = True
    geometry_validation: bool = False
    connector_stride: int = 10
    frame_stride: int = 20
    laser_footprint_stride: int = 20
    show_connectors: bool = False
    show_face_labels: bool = False
    planar_normal_tolerance_deg: float = 2.0
    plane_distance_tolerance_mm: float = 0.1
    cylinder_axis_angle_tolerance_deg: float = 2.0
    cylinder_axis_distance_tolerance_mm: float = 0.1
    cylinder_radius_tolerance_mm: float = 0.1
    output_dir: Path | None = None
    laser_width_mm: float = 200.0
    laser_ray_count: int = 21
    min_sensor_range_mm: float = 200.0
    max_sensor_range_mm: float = 420.0
    max_incidence_angle_deg: float = 60.0
    sweep_overlap_fraction: float = 0.20
    top_sweep: bool = True
    front_sweep: bool = True
    right_sweep: bool = True
    back_sweep: bool = True
    left_sweep: bool = True
    isometric_recovery: bool = True
    isometric_elevation_deg: float = 35.0
    isometric_azimuth_step_deg: float = 10.0
    maximum_isometric_sweeps: int = 1
    minimum_recovery_gain_percent: float = 0.5
    enable_hole_recovery: bool = True
    hole_target_coverage_percent: float = 85.0
    maximum_views_per_hole: int = 2
    hole_candidate_tilt_angles_deg: tuple[float, ...] = (25.0, 40.0)
    hole_candidate_azimuth_count: int = 2
    enable_orbital_hole_sweep: bool = False
    orbital_hole_mode: str = "disabled"
    enable_blind_hole_bottom_recovery: bool = True
    enable_bottom_sweep: bool = True
    enable_lower_isometric_recovery: bool = True
    bottom_clearance_mm: float = 25.0
    lower_isometric_elevation_deg: float = 35.0
    maximum_lower_recovery_sweeps: int = 1
    fixture_exclusion_enabled: bool = False
    fixture_mesh_path: Path | None = None
    total_target_coverage_percent: float = 90.0
    minimum_candidate_gain_percent: float = 1.0
    maximum_optional_sweeps: int = 3
    planning_quality: str = "fast_preview"
    standard_laser_ray_count: int = 21
    final_laser_ray_count: int = 101

    def validate(self) -> None:
        if self.planning_mode not in {"simple_view_based", "advanced_face_based"}:
            raise ValueError(f"Unsupported planning mode: {self.planning_mode}.")
        requested = set(self.surface_types)
        unsupported = requested - VALID_SURFACE_TYPES
        if unsupported:
            raise ValueError(
                f"Unsupported surface types: {', '.join(sorted(unsupported))}."
            )
        if not requested:
            raise ValueError("At least one surface type must be selected.")
        if self.lane_spacing_mm <= 0.0:
            raise ValueError("lane_spacing_mm must be greater than zero.")
        if self.travel_spacing_mm <= 0.0:
            raise ValueError("travel_spacing_mm must be greater than zero.")
        if self.stand_off_mm <= 0.0:
            raise ValueError("stand_off_mm must be greater than zero.")
        if (
            self.planning_mode == "simple_view_based"
            and not self.min_sensor_range_mm <= self.stand_off_mm <= self.max_sensor_range_mm
        ):
            raise ValueError(
                "View-based stand-off must be within the configured sensor range."
            )
        if self.minimum_face_area_mm2 < 0.0:
            raise ValueError("minimum_face_area_mm2 must not be negative.")
        if min(
            self.connector_stride,
            self.frame_stride,
            self.laser_footprint_stride,
        ) < 1:
            raise ValueError("viewer strides must be at least 1.")
        if not 0.0 <= self.sweep_overlap_fraction < 1.0:
            raise ValueError("sweep_overlap_fraction must be in [0, 1).")
        if self.maximum_isometric_sweeps < 0:
            raise ValueError("maximum_isometric_sweeps must not be negative.")
        if not 0.0 <= self.hole_target_coverage_percent <= 100.0:
            raise ValueError("hole_target_coverage_percent must be between 0 and 100.")
        if not 0.0 <= self.total_target_coverage_percent <= 100.0:
            raise ValueError("total_target_coverage_percent must be between 0 and 100.")
        if min(self.maximum_views_per_hole, self.hole_candidate_azimuth_count,
               self.maximum_optional_sweeps) < 1:
            raise ValueError("Feature-recovery candidate limits must be at least 1.")
        if self.bottom_clearance_mm < 0.0:
            raise ValueError("bottom_clearance_mm must not be negative.")
        if self.fixture_exclusion_enabled and self.fixture_mesh_path is None:
            raise ValueError(
                "fixture_mesh_path is required when fixture exclusion is enabled."
            )
        if self.planning_quality not in {"fast_preview", "standard", "final_validation"}:
            raise ValueError(f"Unsupported planning quality: {self.planning_quality}.")
        if min(self.standard_laser_ray_count, self.final_laser_ray_count) < 3:
            raise ValueError("Optical laser ray counts must be at least 3.")
        if self.orbital_hole_mode not in {"disabled", "adaptive", "enabled"}:
            raise ValueError(f"Unsupported orbital hole mode: {self.orbital_hole_mode}.")
