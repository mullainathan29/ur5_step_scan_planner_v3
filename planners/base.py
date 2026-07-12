"""Shared typed records for exact B-Rep scan planning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


Vector3 = NDArray[np.float64]


class PlanningError(RuntimeError):
    """Raised when exact surface planning cannot produce a valid result."""


class MotionType(str, Enum):
    APPROACH = "approach"
    LEAD_IN = "lead_in"
    SCAN = "scan"
    LANE_TRANSITION = "lane_transition"
    LEAD_OUT = "lead_out"
    RETRACT = "retract"


class CylinderStrategy(str, Enum):
    LONGITUDINAL = "longitudinal"
    HELICAL = "helical"


class ConeStrategy(str, Enum):
    LONGITUDINAL = "longitudinal"
    CIRCUMFERENTIAL = "circumferential"
    HELICAL = "helical"


class SphereStrategy(str, Enum):
    LATITUDE = "latitude"
    LONGITUDE = "longitude"
    AUTOMATIC = "automatic"


@dataclass(frozen=True)
class PlannerSettings:
    """Planar raster and scanner-pose configuration in millimetres."""

    lane_spacing_mm: float = 160.0
    travel_spacing_mm: float = 5.0
    stand_off_mm: float = 310.0
    approach_clearance_mm: float = 120.0
    lead_distance_mm: float = 20.0
    geometric_tolerance_mm: float = 1.0e-7

    def validate(self) -> None:
        for name in (
            "lane_spacing_mm",
            "travel_spacing_mm",
            "stand_off_mm",
            "geometric_tolerance_mm",
        ):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{name} must be greater than zero.")
        if self.approach_clearance_mm < 0.0:
            raise ValueError("approach_clearance_mm must not be negative.")
        if self.lead_distance_mm < 0.0:
            raise ValueError("lead_distance_mm must not be negative.")


@dataclass(frozen=True)
class PlaneFrame:
    """Exact planar coordinate system and scanner axes."""

    origin: Vector3
    laser_axis: Vector3
    travel_axis: Vector3
    outward_normal: Vector3


@dataclass(frozen=True)
class BoundaryCurve2D:
    """One exact edge p-curve in the native plane parameter space."""

    edge: object
    curve: object
    first_parameter: float
    last_parameter: float


@dataclass(frozen=True)
class BoundaryWire2D:
    """One outer or inner exact boundary loop."""

    is_outer: bool
    curves: tuple[BoundaryCurve2D, ...]


@dataclass(frozen=True)
class SurfaceSample:
    """An exact nominal point on a STEP face before scanner offset."""

    face_id: int
    lane_index: int
    section_index: int
    sample_index: int
    uv: tuple[float, float]
    point: Vector3
    normal: Vector3


@dataclass(frozen=True)
class RasterSegment:
    """One connected clipped raster section; never joins another section."""

    face_id: int
    lane_index: int
    section_index: int
    laser_coordinate: float
    samples: tuple[SurfaceSample, ...]


@dataclass(frozen=True)
class Pose:
    """Scanner pose using quaternion orientation internally."""

    face_id: int
    segment_id: int
    pose_id: int
    motion_type: MotionType
    scan_enabled: bool
    surface_point: Vector3
    surface_normal: Vector3
    position: Vector3
    quaternion_xyzw: tuple[float, float, float, float]
    region_id: int = -1
    member_face_ids: tuple[int, ...] = ()
    sweep_id: int = -1
    sweep_name: str = ""
    view_type: str = "face"
    laser_line_direction: tuple[float, float, float] | None = None
    travel_direction: tuple[float, float, float] | None = None
    predicted_stand_off_mm: float | None = None
    predicted_visible_laser_width_mm: float | None = None


@dataclass(frozen=True)
class ScanMotionSegment:
    """A complete approach-to-retract motion for one clipped section."""

    segment_id: int
    raster_segment: RasterSegment
    poses: tuple[Pose, ...]

    @property
    def scan_poses(self) -> tuple[Pose, ...]:
        return tuple(pose for pose in self.poses if pose.scan_enabled)

    @property
    def transition_poses(self) -> tuple[Pose, ...]:
        return tuple(pose for pose in self.poses if not pose.scan_enabled)

    @property
    def approach_pose(self) -> Pose:
        return self.poses[0]

    @property
    def retract_pose(self) -> Pose:
        return self.poses[-1]


@dataclass(frozen=True)
class PlanarRasterPlan:
    """Exact planar rasters and corresponding offset scanner motions."""

    face_id: int
    frame: PlaneFrame
    boundary_wires_2d: tuple[BoundaryWire2D, ...]
    raster_segments: tuple[RasterSegment, ...]
    motion_segments: tuple[ScanMotionSegment, ...]


@dataclass(frozen=True)
class CylinderGeometry:
    """Exact analytic cylinder parameters in the CAD coordinate frame."""

    axis_origin: Vector3
    axis_direction: Vector3
    x_direction: Vector3
    y_direction: Vector3
    radius_mm: float
    angular_bounds: tuple[float, float]
    axial_bounds: tuple[float, float]
    periodic: bool


@dataclass(frozen=True)
class CylindricalPlan:
    """Exact cylindrical paths and their scanner motion segments."""

    face_id: int
    strategy: CylinderStrategy
    geometry: CylinderGeometry
    path_segments: tuple[RasterSegment, ...]
    motion_segments: tuple[ScanMotionSegment, ...]


@dataclass(frozen=True)
class ConeGeometry:
    apex: Vector3
    axis_direction: Vector3
    x_direction: Vector3
    y_direction: Vector3
    reference_radius_mm: float
    semi_angle_rad: float
    angular_bounds: tuple[float, float]
    axial_bounds: tuple[float, float]


@dataclass(frozen=True)
class ConicalPlan:
    face_id: int
    strategy: ConeStrategy
    geometry: ConeGeometry
    path_segments: tuple[RasterSegment, ...]
    motion_segments: tuple[ScanMotionSegment, ...]


@dataclass(frozen=True)
class SphereGeometry:
    center: Vector3
    radius_mm: float
    angular_bounds: tuple[float, float]
    latitude_bounds: tuple[float, float]


@dataclass(frozen=True)
class SphericalPlan:
    face_id: int
    strategy: SphereStrategy
    geometry: SphereGeometry
    path_segments: tuple[RasterSegment, ...]
    motion_segments: tuple[ScanMotionSegment, ...]
