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


def line_collision_with_mesh(mesh, p1: np.ndarray, p2: np.ndarray) -> bool:
    direction = p2 - p1
    distance = np.linalg.norm(direction)
    if distance < 1e-6:
        return False
    dir_norm = direction / distance
    eps = 1.0
    if distance <= 2 * eps:
        return False
    origin = p1 + eps * dir_norm
    locations, _, _ = mesh.ray.intersects_location(
        ray_origins=np.array([origin]),
        ray_directions=np.array([dir_norm]),
        multiple_hits=False
    )
    if len(locations) > 0:
        hit_dist = np.linalg.norm(locations[0] - origin)
        if hit_dist < (distance - 2 * eps):
            return True
    return False


def _heuristic_ordered(sweeps: list[ViewSweep], model: CadModel) -> tuple[ViewSweep, ...]:
    if not sweeps:
        return ()
    
    # 1. Classify principal sweeps and optional/recovery sweeps
    principal_types = {"top", "front", "right", "back", "left"}
    principal_sweeps = [s for s in sweeps if s.view_type in principal_types]
    recovery_sweeps = [s for s in sweeps if s.view_type not in principal_types]
    
    # 2. Associate each recovery sweep with the nearest principal sweep
    def sweep_centroid(s: ViewSweep) -> np.ndarray:
        scan_poses = [p for p in s.poses if p.scan_enabled]
        if not scan_poses:
            scan_poses = list(s.poses)
        if not scan_poses:
            return np.zeros(3)
        return np.mean([p.position for p in scan_poses], axis=0)
    
    principal_centroids = {s.sweep_id: sweep_centroid(s) for s in principal_sweeps}
    
    groups: dict[int, list[ViewSweep]] = {s.sweep_id: [] for s in principal_sweeps}
    unassociated = []
    
    for r in recovery_sweeps:
        r_centroid = sweep_centroid(r)
        if principal_sweeps:
            best_p = min(principal_sweeps, key=lambda p: np.linalg.norm(principal_centroids[p.sweep_id] - r_centroid))
            groups[best_p.sweep_id].append(r)
        else:
            unassociated.append(r)
            
    mesh = model.validation_mesh.mesh if model else None
    
    def transition_cost(s_a: ViewSweep, s_b: ViewSweep) -> float:
        if not s_a.poses or not s_b.poses:
            return 0.0
        p_a = np.asarray(s_a.poses[-1].position)
        p_b = np.asarray(s_b.poses[0].position)
        dist = float(np.linalg.norm(p_b - p_a))
        
        q_a = np.asarray(s_a.poses[-1].quaternion_xyzw)
        q_b = np.asarray(s_b.poses[0].quaternion_xyzw)
        angle = float(np.degrees((Rotation.from_quat(q_a).inv() * Rotation.from_quat(q_b)).magnitude()))
        
        penalty = 0.0
        if mesh is not None:
            if line_collision_with_mesh(mesh, p_a, p_b):
                penalty = 500.0
                
        return dist + 2.0 * angle + penalty

    cost_matrix = {}
    all_sweeps = list(sweeps)
    for s_a in all_sweeps:
        cost_matrix[s_a.sweep_id] = {}
        for s_b in all_sweeps:
            cost_matrix[s_a.sweep_id][s_b.sweep_id] = transition_cost(s_a, s_b)

    # 3. Order recovery sweeps within each group using Nearest-Neighbour
    ordered_groups = {}
    for p_id, recs in groups.items():
        if not recs:
            ordered_groups[p_id] = []
            continue
        p_sweep = next(s for s in principal_sweeps if s.sweep_id == p_id)
        current = p_sweep
        group_order = []
        remaining_recs = list(recs)
        while remaining_recs:
            next_rec = min(remaining_recs, key=lambda r: cost_matrix[current.sweep_id][r.sweep_id])
            remaining_recs.remove(next_rec)
            group_order.append(next_rec)
            current = next_rec
        if len(group_order) >= 3:
            improved = True
            while improved:
                improved = False
                for i in range(1, len(group_order) - 1):
                    for j in range(i + 1, len(group_order) + 1):
                        if j - i == 1:
                            continue
                        new_order = group_order[:i] + group_order[i:j][::-1] + group_order[j:]
                        old_cost = (
                            cost_matrix[p_sweep.sweep_id][group_order[0].sweep_id] +
                            sum(cost_matrix[group_order[k].sweep_id][group_order[k+1].sweep_id] for k in range(len(group_order)-1))
                        )
                        new_cost = (
                            cost_matrix[p_sweep.sweep_id][new_order[0].sweep_id] +
                            sum(cost_matrix[new_order[k].sweep_id][new_order[k+1].sweep_id] for k in range(len(new_order)-1))
                        )
                        if new_cost < old_cost - 1e-4:
                            group_order = new_order
                            improved = True
        ordered_groups[p_id] = group_order

    # 4. Order principal sweeps using Nearest-Neighbour
    if principal_sweeps:
        p_ranks = {"top": 0, "front": 1, "right": 2, "back": 3, "left": 4}
        sorted_p = sorted(principal_sweeps, key=lambda s: p_ranks.get(s.view_type, 5))
        
        ordered_p = [sorted_p.pop(0)]
        while sorted_p:
            current = ordered_p[-1]
            current_end_sweep = ordered_groups[current.sweep_id][-1] if ordered_groups[current.sweep_id] else current
            next_p = min(sorted_p, key=lambda p: cost_matrix[current_end_sweep.sweep_id][p.sweep_id])
            sorted_p.remove(next_p)
            ordered_p.append(next_p)
            
        if len(ordered_p) >= 3:
            improved = True
            while improved:
                improved = False
                for i in range(1, len(ordered_p) - 1):
                    for j in range(i + 1, len(ordered_p) + 1):
                        if j - i == 1:
                            continue
                        new_p_order = ordered_p[:i] + ordered_p[i:j][::-1] + ordered_p[j:]
                        def group_seq_cost(p_seq):
                            total = 0.0
                            for k in range(len(p_seq) - 1):
                                p_a = p_seq[k]
                                p_b = p_seq[k+1]
                                end_a = ordered_groups[p_a.sweep_id][-1] if ordered_groups[p_a.sweep_id] else p_a
                                total += cost_matrix[end_a.sweep_id][p_b.sweep_id]
                            return total
                        if group_seq_cost(new_p_order) < group_seq_cost(ordered_p) - 1e-4:
                            ordered_p = new_p_order
                            improved = True
    else:
        ordered_p = []

    final_sweeps = []
    for p in ordered_p:
        final_sweeps.append(p)
        final_sweeps.extend(ordered_groups[p.sweep_id])
    
    final_sweeps.extend(unassociated)
    return tuple(final_sweeps)


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
    ordered = _heuristic_ordered(selected, model)
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
