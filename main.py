#!/usr/bin/env python3
"""Unified command-line entry point for the current B-Rep pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from app.config import PipelineConfig
from app.pipeline import run_pipeline
from planners.base import CylinderStrategy


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run STEP B-Rep extraction, exact planar/cylindrical planning, "
            "scanner pose generation, coverage validation, display, and export."
        )
    )
    parser.add_argument("step_file", type=Path)
    parser.add_argument(
        "--planning-mode",
        choices=("simple_view_based", "advanced_face_based"),
        default="simple_view_based",
        help="Use minimal principal-view sweeps (default) or exact face paths.",
    )
    parser.add_argument(
        "--surface-types",
        nargs="+",
        choices=("plane", "cylinder", "cone", "sphere"),
        default=("plane", "cylinder"),
    )
    parser.add_argument("--lane-spacing", type=float, default=20.0)
    parser.add_argument("--travel-spacing", type=float, default=5.0)
    parser.add_argument("--stand-off", type=float, default=310.0)
    parser.add_argument("--laser-line-width", type=float, default=200.0)
    parser.add_argument("--overlap", type=float, default=20.0,
                        help="Adjacent laser-line overlap in percent.")
    parser.add_argument("--isometric-elevation", type=float, default=35.0)
    parser.add_argument("--isometric-azimuth-step", type=float, default=10.0)
    parser.add_argument("--maximum-isometric-sweeps", type=int, default=1)
    parser.add_argument("--minimum-recovery-gain", type=float, default=0.5,
                        help="Minimum incremental exterior coverage in percentage points.")
    parser.add_argument("--isometric-recovery", action=argparse.BooleanOptionalAction,
                        default=True)
    parser.add_argument("--minimum-face-area", type=float, default=25.0)
    parser.add_argument(
        "--include-internal-surfaces",
        action="store_true",
        help="Include bore and cavity faces in planning and coverage.",
    )
    parser.add_argument(
        "--cylinder-strategy",
        choices=tuple(strategy.value for strategy in CylinderStrategy),
        default=CylinderStrategy.LONGITUDINAL.value,
    )
    display = parser.add_mutually_exclusive_group()
    display.add_argument("--display", dest="display", action="store_true")
    display.add_argument("--no-display", dest="display", action="store_false")
    parser.set_defaults(display=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--geometry-validation",
        action="store_true",
        help=(
            "Print stand-off/continuity metrics and add toggleable scanner "
            "offset connectors to the display."
        ),
    )
    parser.add_argument("--connector-stride", type=int, default=10)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--laser-footprint-stride", type=int, default=20)
    connectors = parser.add_mutually_exclusive_group()
    connectors.add_argument(
        "--show-connectors", dest="show_connectors", action="store_true"
    )
    connectors.add_argument(
        "--hide-connectors", dest="show_connectors", action="store_false"
    )
    labels = parser.add_mutually_exclusive_group()
    labels.add_argument(
        "--show-face-labels", dest="show_face_labels", action="store_true"
    )
    labels.add_argument(
        "--hide-face-labels", dest="show_face_labels", action="store_false"
    )
    parser.set_defaults(show_connectors=False, show_face_labels=False)
    return parser


def _print_summary(summary) -> None:
    print("\nUnified B-Rep pipeline summary")
    print(f"  Total STEP faces:              {summary.total_step_faces}")
    print(f"  Selected planar faces:         {summary.selected_planar_faces}")
    print(
        f"  Selected cylindrical faces:    {summary.selected_cylindrical_faces}"
    )
    print(f"  Selected conical faces:        {summary.selected_conical_faces}")
    print(f"  Selected spherical faces:      {summary.selected_spherical_faces}")
    print("  Skipped faces and reasons:")
    if not summary.skipped_faces_by_reason:
        print("    none")
    for reason, face_ids in summary.skipped_faces_by_reason.items():
        preview = ", ".join(str(face_id) for face_id in face_ids[:12])
        suffix = ", ..." if len(face_ids) > 12 else ""
        print(f"    {reason}: {len(face_ids)} [{preview}{suffix}]")
    print(f"  Generated scan regions:        {summary.generated_scan_regions}")
    print(f"  Generated scan segments:       {summary.generated_scan_segments}")
    print(f"  Scan pose count:                {summary.scan_pose_count}")
    print(
        f"  Total scan-path length:         "
        f"{summary.total_scan_path_length_mm:.2f} mm"
    )
    print(f"  Exterior coverage:              {summary.coverage_percent:.2f}%")
    print(f"  Invalid segment count:          {summary.invalid_segment_count}")
    print("  Output files:")
    for name, path in summary.output_files.items():
        print(f"    {name}: {path}")
    print(
        "\nSafety: optical mission selection is not robot feasibility. No UR5 "
        "IK, joint-limit, MoveIt2, ROS 2, or complete collision checking has "
        "been performed."
    )


def _print_geometry_validation(validation) -> None:
    print("\nGeometry Validation mode")
    print("  Per-pose stand-off distances:")
    for metric in validation.pose_metrics:
        print(
            f"    segment={metric.segment_id:4d} "
            f"face={metric.face_id:4d} pose={metric.pose_id:4d} "
            f"stand-off={metric.stand_off_mm:9.3f} mm "
            f"connector={metric.connector_status.value}"
        )
    print("  Statistics:")
    print(
        f"    Minimum stand-off:                  "
        f"{validation.minimum_stand_off_mm:.3f} mm"
    )
    print(
        f"    Maximum stand-off:                  "
        f"{validation.maximum_stand_off_mm:.3f} mm"
    )
    print(
        f"    Mean stand-off:                     "
        f"{validation.mean_stand_off_mm:.3f} mm"
    )
    print(
        f"    Stand-off standard deviation:       "
        f"{validation.stand_off_standard_deviation_mm:.6f} mm"
    )
    print(
        f"    Maximum consecutive orientation change: "
        f"{validation.maximum_orientation_change_deg:.3f} deg"
    )
    print(
        f"    Maximum consecutive position change:    "
        f"{validation.maximum_position_change_mm:.3f} mm"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PipelineConfig(
        step_path=args.step_file,
        planning_mode=args.planning_mode,
        surface_types=tuple(args.surface_types),
        lane_spacing_mm=args.lane_spacing,
        travel_spacing_mm=args.travel_spacing,
        stand_off_mm=args.stand_off,
        laser_width_mm=args.laser_line_width,
        sweep_overlap_fraction=args.overlap / 100.0,
        isometric_recovery=args.isometric_recovery,
        isometric_elevation_deg=args.isometric_elevation,
        isometric_azimuth_step_deg=args.isometric_azimuth_step,
        maximum_isometric_sweeps=args.maximum_isometric_sweeps,
        minimum_recovery_gain_percent=args.minimum_recovery_gain,
        minimum_face_area_mm2=args.minimum_face_area,
        include_internal_surfaces=args.include_internal_surfaces,
        cylinder_strategy=CylinderStrategy(args.cylinder_strategy),
        display=args.display,
        output_dir=args.output_dir,
        geometry_validation=args.geometry_validation,
        connector_stride=args.connector_stride,
        frame_stride=args.frame_stride,
        laser_footprint_stride=args.laser_footprint_stride,
        show_connectors=args.show_connectors,
        show_face_labels=args.show_face_labels,
    )
    try:
        result = run_pipeline(config)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    _print_summary(result.summary)
    if result.geometry_validation is not None:
        _print_geometry_validation(result.geometry_validation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
