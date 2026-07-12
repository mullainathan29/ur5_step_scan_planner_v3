"""Inspect STEP B-Rep faces from the command line."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Sequence

from cad_model import CadModelError, load_step_model
from face_analysis import SurfaceType


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect authoritative STEP B-Rep faces and surface types."
    )
    parser.add_argument("model", type=Path, help="Path to a .step or .stp model")
    parser.add_argument(
        "--linear-tolerance",
        type=float,
        default=0.25,
        help="Rendering/validation tessellation tolerance in mm (default: 0.25)",
    )
    parser.add_argument(
        "--angular-tolerance",
        type=float,
        default=0.1,
        help="Rendering/validation angular tolerance in rad (default: 0.1)",
    )
    return parser


def format_summary(model_path: Path, face_count: int, counts: Counter) -> str:
    lines = [
        f"STEP model: {model_path}",
        f"Total B-Rep faces: {face_count}",
        "Surface summary:",
    ]
    for surface_type in SurfaceType:
        count = counts.get(surface_type, 0)
        if count:
            lines.append(f"  {surface_type.value:<10} {count:>6}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        model = load_step_model(
            args.model,
            linear_tolerance=args.linear_tolerance,
            angular_tolerance=args.angular_tolerance,
        )
    except (CadModelError, ValueError) as exc:
        print(f"error: {exc}")
        return 2

    counts = Counter(record.geometry_type for record in model.faces)
    print(format_summary(model.source_path, len(model.faces), counts))
    print(
        "Validation mesh: "
        f"{len(model.validation_mesh.mesh.vertices)} vertices, "
        f"{len(model.validation_mesh.mesh.faces)} triangles"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
