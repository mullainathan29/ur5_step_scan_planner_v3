"""Greedy optical candidate selection and low-motion mission ordering."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np
from scipy.spatial.transform import Rotation

from cad_model import CadModel
from coverage import CoverageSettings, TriangleStatus, detect_exterior_face_ids, validate_coverage
from features import HoleFeature, analyse_hole_coverage
from planners.view_based import ViewSweep
from recovery import RecoveryCandidate


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: int
    name: str
    strategy: str
    selected: bool
    rejection_reason: str
    score: float
    total_coverage_gain_percent: float
    newly_covered_area_mm2: float
    newly_covered_hole_area_mm2: float
    transition_distance_mm: float
    orientation_change_deg: float


@dataclass(frozen=True)
class MissionOptimisationResult:
    sweeps: tuple[ViewSweep, ...]
    coverage: object
    candidate_evaluations: tuple[CandidateEvaluation, ...]
    hole_reports_before: tuple[object, ...]
    hole_reports_after: tuple[object, ...]
    floating_part_assumption: bool


def _scan_poses(sweeps):
    return tuple(pose for sweep in sweeps for pose in sweep.poses)


def _transition_metrics(selected, candidate):
    if not selected or not selected[-1].scan_poses or not candidate.sweep.scan_poses:
        return 0.0, 0.0
    first = candidate.sweep.scan_poses[0]
    last = selected[-1].scan_poses[-1]
    distance = float(np.linalg.norm(first.position - last.position))
    a = Rotation.from_quat(last.quaternion_xyzw)
    b = Rotation.from_quat(first.quaternion_xyzw)
    angle = float(np.degrees((a.inv() * b).magnitude()))
    return distance, angle


def _ordered(sweeps: list[ViewSweep]) -> tuple[ViewSweep, ...]:
    rank = {"top": 0, "isometric": 1, "front": 2, "right": 3,
            "back": 4, "left": 5, "hole_recovery": 6,
            "lower_isometric": 7, "bottom": 8}
    return tuple(sorted(sweeps, key=lambda sweep: (rank.get(sweep.view_type, 6),
                                                   sweep.sweep_id)))


def optimise_scan_mission(
    model: CadModel,
    mandatory_sweeps: tuple[ViewSweep, ...],
    optional_candidates: tuple[RecoveryCandidate, ...],
    holes: tuple[HoleFeature, ...],
    coverage_settings: CoverageSettings,
    *,
    target_total_coverage_percent: float = 95.0,
    target_hole_coverage_percent: float = 90.0,
    minimum_gain_percent: float = 0.5,
    maximum_optional_sweeps: int = 8,
    floating_part_assumption: bool = True,
    cancellation_check=None,
    progress_callback=None,
) -> MissionOptimisationResult:
    cancellation_check = cancellation_check or (lambda: None)
    progress_callback = progress_callback or (lambda *_: None)
    exterior = detect_exterior_face_ids(model)
    # Hole recovery must make internal hole triangles eligible.
    optical_settings = replace(coverage_settings, include_internal_surfaces=True)
    validate = lambda sweeps: validate_coverage(
        model, _scan_poses(sweeps), optical_settings,
        exterior_face_ids=exterior,
    )
    selected = list(mandatory_sweeps)
    current = validate(selected)
    before = tuple(analyse_hole_coverage(
        model, hole, current, target_hole_coverage_percent
    ) for hole in holes)
    remaining = list(optional_candidates)
    evaluations: list[CandidateEvaluation] = []
    selected_count = 0
    areas = np.asarray(model.validation_mesh.mesh.area_faces)
    mapping = model.validation_mesh.triangle_face_ids

    while remaining and selected_count < maximum_optional_sweeps:
        cancellation_check()
        current_valid = current.triangle_status == TriangleStatus.VALID_COVERED
        best = None
        best_payload = None
        for candidate_index, candidate in enumerate(remaining):
            cancellation_check()
            if candidate.strategy == "adaptive_orbital_hole" and any(
                other.target_hole_id == candidate.target_hole_id
                and other.strategy == "tilted_cross_hole"
                for other in remaining
            ):
                continue
            progress_callback(candidate_index, len(remaining))
            updated = validate([*selected, candidate.sweep])
            candidate_only = validate([candidate.sweep])
            updated_valid = updated.triangle_status == TriangleStatus.VALID_COVERED
            new_mask = updated_valid & ~current_valid
            new_area = float(areas[new_mask].sum())
            hole_ids = next((hole.cylindrical_face_ids + hole.conical_face_ids
                             + ((hole.bottom_face_id,) if hole.bottom_face_id is not None else ())
                             for hole in holes if hole.feature_id == candidate.target_hole_id), ())
            new_hole_area = float(areas[new_mask & np.isin(mapping, hole_ids)].sum())
            old_percent = current.report["total_exterior_surface"].coverage_percent
            new_percent = updated.report["total_exterior_surface"].coverage_percent
            gain = new_percent - old_percent
            transition, orientation = _transition_metrics(selected, candidate)
            duplicate_penalty = 1000.0 if not np.any(new_mask) else 0.0
            scan_time_penalty = 0.02 * len(candidate.sweep.scan_poses)
            score = (new_area + 2.0 * new_hole_area - 0.02 * transition
                     - 0.2 * orientation - duplicate_penalty - scan_time_penalty
                     - 0.1 * candidate_only.invalid_incidence_hit_count)
            payload = (candidate, updated, gain, new_area, new_hole_area,
                       transition, orientation, score)
            if best is None or score > best:
                best, best_payload = score, payload
        candidate, updated, gain, new_area, new_hole_area, transition, orientation, score = best_payload
        remaining.remove(candidate)
        if gain < minimum_gain_percent and new_hole_area <= 0.0:
            evaluations.append(CandidateEvaluation(
                candidate.candidate_id, candidate.name, candidate.strategy,
                False, "minimum gain not reached", score, gain, new_area,
                new_hole_area, transition, orientation,
            ))
            if score <= 0:
                if not any(row.strategy == "adaptive_orbital_hole"
                           for row in remaining):
                    break
            continue
        before_hole = next((row.cylindrical_wall_coverage_percent for row in before
                            if row.feature_id == candidate.target_hole_id), 0.0)
        after_report = next((analyse_hole_coverage(
            model, hole, updated, target_hole_coverage_percent
        ) for hole in holes if hole.feature_id == candidate.target_hole_id), None)
        after_hole = after_report.cylindrical_wall_coverage_percent if after_report else 0.0
        annotated = replace(candidate.sweep, poses=tuple(replace(
            pose, candidate_score=score, coverage_gain=gain,
            hole_coverage_before=before_hole, hole_coverage_after=after_hole,
            bottom_coverage_gain=(after_report.bottom_face_coverage_percent if after_report else 0.0),
            mandatory_or_optional="optional",
        ) for pose in candidate.sweep.poses))
        selected.append(annotated)
        selected_count += 1
        current = updated
        evaluations.append(CandidateEvaluation(
            candidate.candidate_id, candidate.name, candidate.strategy, True,
            "", score, gain, new_area, new_hole_area, transition, orientation,
        ))
        hole_complete = all(analyse_hole_coverage(
            model, hole, current, target_hole_coverage_percent
        ).complete for hole in holes)
        if (current.report["total_exterior_surface"].coverage_percent
                >= target_total_coverage_percent and hole_complete):
            break
    for candidate in remaining:
        evaluations.append(CandidateEvaluation(
            candidate.candidate_id, candidate.name, candidate.strategy, False,
            "not required or optional sweep limit reached", 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ))
    ordered = _ordered(selected)
    final = validate(ordered)
    after = tuple(analyse_hole_coverage(
        model, hole, final, target_hole_coverage_percent
    ) for hole in holes)
    return MissionOptimisationResult(
        ordered, final, tuple(evaluations), before, after,
        floating_part_assumption,
    )


def optimisation_payload(result: MissionOptimisationResult) -> dict:
    return {
        "selected_sweep_order": [sweep.sweep_name for sweep in result.sweeps],
        "floating_part_assumption": result.floating_part_assumption,
        "candidates": [asdict(row) for row in result.candidate_evaluations],
    }
