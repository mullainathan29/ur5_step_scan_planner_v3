"""Configuration for the unified B-Rep planning application."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from planners.base import CylinderStrategy


VALID_SURFACE_TYPES = frozenset({"plane", "cylinder", "cone", "sphere"})


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
