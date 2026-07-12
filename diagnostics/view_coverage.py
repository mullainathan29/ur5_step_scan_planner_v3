"""Plan implemented surfaces, validate coverage, and show triangle status."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv

from cad_model import CadModelError, load_step_model
from coverage import CoverageSettings, coverage_rgb, validate_coverage
from planners.base import PlannerSettings
from planners.cylinder import plan_cylindrical_faces
from planners.plane import plan_planar_faces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate mesh coverage from exact planar/cylindrical poses."
    )
    parser.add_argument("model", type=Path)
    parser.add_argument("--lane-spacing", type=float, default=160.0)
    parser.add_argument("--travel-spacing", type=float, default=5.0)
    parser.add_argument("--laser-width", type=float, default=200.0)
    parser.add_argument("--laser-rays", type=int, default=21)
    parser.add_argument("--min-range", type=float, default=200.0)
    parser.add_argument("--max-range", type=float, default=420.0)
    parser.add_argument("--max-incidence", type=float, default=60.0)
    parser.add_argument(
        "--include-internal-surfaces",
        action="store_true",
        help="Include internal surfaces in coverage denominators and colors.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Print the report without opening the PyVista window.",
    )
    return parser


def _all_scan_poses(model, settings: PlannerSettings):
    planar = plan_planar_faces(model.topods_faces, model.faces, settings)
    cylindrical = plan_cylindrical_faces(model.topods_faces, model.faces, settings)
    return tuple(
        pose
        for plan in (*planar, *cylindrical)
        for motion in plan.motion_segments
        for pose in motion.scan_poses
    )


def _print_report(result) -> None:
    labels = {
        "planar_faces": "Planar faces",
        "cylindrical_faces": "Cylindrical faces",
        "conical_faces": "Conical faces",
        "freeform_faces": "Freeform faces",
        "total_exterior_surface": "Total exterior surface",
    }
    print(
        f"Scan poses: {result.scan_pose_count}, rays: {result.ray_count}, "
        f"first hits: {result.first_hit_count}"
    )
    print(
        f"Exterior faces: {len(result.exterior_face_ids)}, "
        f"internal faces excluded: {len(result.internal_face_ids)}"
    )
    for key, label in labels.items():
        row = result.report[key]
        print(
            f"{label:<24} {row.coverage_percent:7.2f}%  "
            f"covered {row.covered_area_mm2:12.2f} / "
            f"{row.eligible_area_mm2:12.2f} mm^2  "
            f"yellow {row.visible_invalid_area_mm2:10.2f} mm^2"
        )


def _show(model, result) -> None:
    mesh = model.validation_mesh.mesh
    cells = np.column_stack(
        (
            np.full(len(mesh.faces), 3, dtype=np.int64),
            np.asarray(mesh.faces, dtype=np.int64),
        )
    ).ravel()
    poly = pv.PolyData(np.asarray(mesh.vertices), cells)
    poly.cell_data["coverage_rgb"] = coverage_rgb(result)
    plotter = pv.Plotter()
    plotter.add_mesh(poly, scalars="coverage_rgb", rgb=True, show_edges=False)
    plotter.add_text(
        "Green: valid covered   Red: uncovered   Yellow: visible invalid",
        font_size=11,
    )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(title="STEP scan coverage validation")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = load_step_model(args.model)
        planner_settings = PlannerSettings(
            lane_spacing_mm=args.lane_spacing,
            travel_spacing_mm=args.travel_spacing,
        )
        poses = _all_scan_poses(model, planner_settings)
        result = validate_coverage(
            model,
            poses,
            CoverageSettings(
                laser_width_mm=args.laser_width,
                laser_ray_count=args.laser_rays,
                min_range_mm=args.min_range,
                max_range_mm=args.max_range,
                max_incidence_angle_deg=args.max_incidence,
                include_internal_surfaces=args.include_internal_surfaces,
            ),
        )
    except (CadModelError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    _print_report(result)
    if not args.no_display:
        _show(model, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
