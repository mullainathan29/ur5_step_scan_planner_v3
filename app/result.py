"""Results returned by the unified B-Rep pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cad_model import CadModel
from coverage import CoverageResult
from geometry_validation import GeometryValidationResult
from planners.base import ConicalPlan, CylindricalPlan, PlanarRasterPlan, Pose, SphericalPlan
from planners.view_based import ViewSweep


@dataclass(frozen=True)
class PipelineSummary:
    total_step_faces: int
    selected_planar_faces: int
    selected_cylindrical_faces: int
    selected_conical_faces: int
    selected_spherical_faces: int
    skipped_faces_by_reason: dict[str, tuple[int, ...]]
    generated_scan_regions: int
    generated_scan_segments: int
    scan_pose_count: int
    total_scan_path_length_mm: float
    coverage_percent: float
    invalid_segment_count: int
    output_files: dict[str, Path]


@dataclass(frozen=True)
class PipelineResult:
    model: CadModel
    planar_plans: tuple[PlanarRasterPlan, ...]
    cylindrical_plans: tuple[CylindricalPlan, ...]
    conical_plans: tuple[ConicalPlan, ...]
    spherical_plans: tuple[SphericalPlan, ...]
    poses: tuple[Pose, ...]
    coverage: CoverageResult
    baseline_coverage: CoverageResult
    geometry_validation: GeometryValidationResult | None
    summary: PipelineSummary
    sweeps: tuple[ViewSweep, ...] = ()
