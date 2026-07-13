"""Fast, cached GUI stages separated from optical validation."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from time import perf_counter

import numpy as np

from app.config import PipelineConfig
from app.result import PipelineResult, PipelineSummary
from cad_model import (
    CadBRep, CadModel, cad_model_from_brep, load_step_brep,
    tessellate_for_validation,
)
from features import analyse_hole_coverage, detect_hole_features
from coverage import CoverageSettings, validate_coverage
from planners.view_based import ViewSweepSettings, generate_preview_sweeps
from recovery import (
    RecoveryCandidate, candidate_accessible, generate_bottom_sweep,
    generate_focused_conical_orbits, generate_hole_recovery_candidates,
    generate_lower_diagonal_sweeps,
    generate_lower_isometric,
)


@dataclass
class StageTimings:
    seconds: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, duration: float) -> None:
        self.seconds[stage] = self.seconds.get(stage, 0.0) + float(duration)


class GeometryCache:
    """Cache B-Rep/faces and independent preview/full validation meshes."""

    def __init__(self) -> None:
        self.path: Path | None = None
        self.brep: CadBRep | None = None
        self.preview_model: CadModel | None = None
        self.full_model: CadModel | None = None
        self.model_bounds: tuple[np.ndarray, np.ndarray] | None = None

    def invalidate(self) -> None:
        self.path = None
        self.brep = None
        self.preview_model = None
        self.full_model = None
        self.model_bounds = None

    def set_path(self, path: Path) -> None:
        resolved = path.expanduser().resolve()
        if self.path != resolved:
            self.invalidate()
            self.path = resolved

    def ensure_brep(self, path: Path, timings: StageTimings,
                    progress=None, cancellation_check=None) -> CadBRep:
        self.set_path(path)
        progress = progress or (lambda *_: None)
        cancellation_check = cancellation_check or (lambda: None)
        if self.brep is None:
            progress(8, "STEP load")
            def stage_done(stage, duration):
                timings.record(stage, duration)
                progress(18 if stage == "STEP load" else 25, stage)
            self.brep = load_step_brep(
                path, stage_callback=stage_done,
                cancellation_check=cancellation_check,
            )
        else:
            timings.record("STEP load", 0.0)
            timings.record("face extraction", 0.0)
        cancellation_check()
        if self.model_bounds is None:
            progress(26, "Model bounds and upright-axis analysis")
            started = perf_counter()
            box = self.brep.shape.BoundingBox()
            self.model_bounds = (
                np.asarray([box.xmin, box.ymin, box.zmin]),
                np.asarray([box.xmax, box.ymax, box.zmax]),
            )
            timings.record("model bounds and upright-axis analysis",
                           perf_counter() - started)
        else:
            timings.record("model bounds and upright-axis analysis", 0.0)
        return self.brep

    def ensure_preview(self, path: Path, timings: StageTimings,
                       progress=None, cancellation_check=None) -> CadModel:
        progress = progress or (lambda *_: None)
        cancellation_check = cancellation_check or (lambda: None)
        brep = self.ensure_brep(path, timings, progress, cancellation_check)
        if self.preview_model is None:
            progress(28, "Preview validation mesh creation")
            started = perf_counter()
            diagonal = float(np.linalg.norm(self.model_bounds[1] - self.model_bounds[0]))
            tolerance = max(diagonal / 20.0, 0.5)
            angular = 0.5
            mesh = None
            # Start coarse and refine toward 10k–30k. Simple analytic solids
            # may legitimately contain fewer triangles even at fine tolerance.
            for _ in range(4):
                cancellation_check()
                mesh = tessellate_for_validation(
                    brep.topods_faces, tolerance, angular,
                    cancellation_check=cancellation_check,
                )
                count = len(mesh.mesh.faces)
                if 10_000 <= count <= 30_000:
                    break
                if count < 10_000:
                    factor = max(1.4, np.sqrt(10_000 / max(count, 1)))
                    tolerance = max(0.05, tolerance / factor)
                    angular = max(0.08, angular / factor)
                else:
                    break
            self.preview_model = cad_model_from_brep(brep, mesh)
            timings.record("validation mesh creation", perf_counter() - started)
        else:
            timings.record("validation mesh creation", 0.0)
        return self.preview_model

    def ensure_full(self, path: Path, timings: StageTimings,
                    progress=None, cancellation_check=None) -> CadModel:
        progress = progress or (lambda *_: None)
        cancellation_check = cancellation_check or (lambda: None)
        brep = self.ensure_brep(path, timings, progress, cancellation_check)
        if self.full_model is None:
            progress(20, "Full validation mesh creation")
            started = perf_counter()
            mesh = tessellate_for_validation(
                brep.topods_faces, 0.25, 0.1,
                cancellation_check=cancellation_check,
            )
            self.full_model = cad_model_from_brep(brep, mesh)
            timings.record("validation mesh creation", perf_counter() - started)
        else:
            timings.record("validation mesh creation", 0.0)
        return self.full_model


def _view_settings(config: PipelineConfig) -> ViewSweepSettings:
    return ViewSweepSettings(
        stand_off_mm=config.stand_off_mm,
        travel_spacing_mm=config.travel_spacing_mm,
        laser_width_mm=config.laser_width_mm,
        overlap_fraction=config.sweep_overlap_fraction,
        top_sweep=config.top_sweep, front_sweep=config.front_sweep,
        right_sweep=config.right_sweep, back_sweep=config.back_sweep,
        left_sweep=config.left_sweep,
        isometric_recovery=config.isometric_recovery,
        isometric_elevation_deg=config.isometric_elevation_deg,
        isometric_azimuth_step_deg=config.isometric_azimuth_step_deg,
        maximum_isometric_sweeps=config.maximum_isometric_sweeps,
        minimum_recovery_gain_percent=config.minimum_recovery_gain_percent,
    )


def generate_fast_paths(cache: GeometryCache, config: PipelineConfig,
                        progress=None, cancellation_check=None):
    timings = StageTimings()
    progress = progress or (lambda *_: None)
    cancellation_check = cancellation_check or (lambda: None)
    model = cache.ensure_preview(config.step_path, timings, progress,
                                 cancellation_check)
    cancellation_check()
    progress(48, "Simple sweep generation")
    started = perf_counter()
    settings = _view_settings(config)
    sweeps = list(generate_preview_sweeps(
        model, settings, cancellation_check=cancellation_check
    ))
    next_id = max((s.sweep_id for s in sweeps), default=-1) + 1
    if config.enable_lower_isometric_recovery:
        for index in range(config.maximum_lower_recovery_sweeps):
            sweeps.append(generate_lower_isometric(
                model, settings, next_id,
                min(80.0, config.lower_isometric_elevation_deg + 5 * index),
            ).sweep)
            next_id += 1
        sweeps.extend(candidate.sweep for candidate in generate_lower_diagonal_sweeps(
            model, settings, next_id, config.lower_isometric_elevation_deg
        )); next_id += 2
    if config.enable_bottom_sweep:
        sweeps.append(generate_bottom_sweep(model, settings, next_id).sweep)
    timings.record("simple sweep generation", perf_counter() - started)
    cancellation_check()
    progress(72, "Lightweight hole detection")
    started = perf_counter()
    holes = detect_hole_features(
        model, lightweight=True, cancellation_check=cancellation_check
    ) if config.enable_hole_recovery else ()
    timings.record("hole detection", perf_counter() - started)
    started = perf_counter()
    poses = tuple(pose for sweep in sweeps for pose in sweep.poses)
    timings.record("pose generation", perf_counter() - started)
    summary = PipelineSummary(
        total_step_faces=len(model.faces), selected_planar_faces=0,
        selected_cylindrical_faces=0, selected_conical_faces=0,
        selected_spherical_faces=0, skipped_faces_by_reason={},
        generated_scan_regions=len(sweeps), generated_scan_segments=len(sweeps),
        scan_pose_count=sum(p.scan_enabled for p in poses),
        total_scan_path_length_mm=0.0, coverage_percent=0.0,
        invalid_segment_count=0, output_files={},
    )
    result = PipelineResult(
        model=model, planar_plans=(), cylindrical_plans=(), conical_plans=(),
        spherical_plans=(), poses=poses, coverage=None,
        baseline_coverage=None, geometry_validation=None, summary=summary,
        sweeps=tuple(sweeps), hole_features=holes,
    )
    progress(90, "Fast paths ready")
    return result, timings


def generate_recovery_candidates(result: PipelineResult, config: PipelineConfig,
                                 progress=None, cancellation_check=None):
    timings = StageTimings()
    progress = progress or (lambda *_: None)
    cancellation_check = cancellation_check or (lambda: None)
    if result.coverage is None:
        raise ValueError("Run coverage before generating feature recovery.")
    started = perf_counter()
    settings = _view_settings(config)
    candidates = []
    next_id = max((s.sweep_id for s in result.sweeps), default=-1) + 1
    reports = []
    for index, hole in enumerate(result.hole_features):
        cancellation_check()
        progress(int(10 + 75 * (index + 1) / max(len(result.hole_features), 1)),
                 f"Recovery candidates for hole {hole.feature_id}")
        report = analyse_hole_coverage(
            result.model, hole, result.coverage,
            config.hole_target_coverage_percent,
        )
        reports.append(report)
        focused = generate_focused_conical_orbits(
            hole, report, settings, model=result.model,
            candidate_id_start=next_id,
            cone_angles_deg=config.focused_orbit_cone_angles_deg,
            focus_depth_fraction=config.focused_orbit_focus_depth_fraction,
            minimum_arc_deg=config.focused_orbit_minimum_arc_deg,
            maximum_arc_deg=config.focused_orbit_maximum_arc_deg,
            angular_spacing_deg=config.focused_orbit_angular_spacing_deg,
            maximum_orbits=config.maximum_focused_orbits_per_hole,
            laser_ray_count=config.standard_laser_ray_count,
        ) if config.enable_focused_hole_orbit else ()
        remaining_views = max(0, config.maximum_views_per_hole - len(focused))
        legacy = generate_hole_recovery_candidates(
            hole, report, settings, candidate_id_start=next_id + len(focused),
            tilt_angles_deg=config.hole_candidate_tilt_angles_deg,
            azimuth_count=config.hole_candidate_azimuth_count,
            maximum_views=remaining_views,
            enable_orbital=config.enable_orbital_hole_sweep,
            enable_blind_bottom=config.enable_blind_hole_bottom_recovery,
            orbital_mode=("disabled" if config.enable_focused_hole_orbit
                          else config.orbital_hole_mode),
        ) if remaining_views else ()
        generated = (*focused, *legacy)
        shallow = hole.depth_mm <= 4.0 * hole.radius_mm
        primary = next((row for row in generated
                        if row.strategy == "tilted_cross_hole"), None)
        if shallow and primary is not None and len(generated) < config.maximum_views_per_hole:
            progress(int(20 + 70 * (index + 1) / max(len(result.hole_features), 1)),
                     f"Testing shallow-hole primary view {hole.feature_id}")
            optical = validate_coverage(
                result.model,
                (*result.poses, *primary.sweep.poses),
                CoverageSettings(
                    laser_width_mm=config.laser_width_mm,
                    laser_ray_count=config.standard_laser_ray_count,
                    min_range_mm=config.min_sensor_range_mm,
                    max_range_mm=config.max_sensor_range_mm,
                    max_incidence_angle_deg=config.max_incidence_angle_deg,
                    include_internal_surfaces=True,
                ),
                cancellation_check=cancellation_check,
            )
            after_primary = analyse_hole_coverage(
                result.model, hole, optical, config.hole_target_coverage_percent
            )
            if not after_primary.complete:
                expanded = generate_hole_recovery_candidates(
                    hole, report, settings,
                    candidate_id_start=next_id + len(generated),
                    tilt_angles_deg=config.hole_candidate_tilt_angles_deg,
                    azimuth_count=config.hole_candidate_azimuth_count,
                    maximum_views=config.maximum_views_per_hole,
                    enable_orbital=False,
                    enable_blind_bottom=False,
                    orbital_mode="disabled",
                    shallow_primary_only=False,
                )
                second = next((row for row in expanded
                               if row.strategy == "tilted_cross_hole"
                               and row.name != primary.name), None)
                if second is not None:
                    generated = (*generated, second)
        candidates.extend(generated); next_id += len(generated)
    fixture = config.fixture_mesh_path if config.fixture_exclusion_enabled else None
    candidates = tuple(candidate for candidate in candidates if candidate_accessible(
        candidate, result.model, fixture, config.bottom_clearance_mm
    ))
    timings.record("recovery generation", perf_counter() - started)
    return replace(result, recovery_candidates=candidates,
                   hole_coverage_reports=tuple(reports)), timings
