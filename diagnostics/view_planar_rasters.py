"""Interactive debug view for exact planar rasters before scanner offsets."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pyvista as pv

from cad_model import CadModelError, load_step_model
from planners.base import PlannerSettings
from planners.plane import plan_planar_faces


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Display exact planar raster paths before robot offsets."
    )
    parser.add_argument("model", type=Path, help="Path to a .step or .stp model")
    parser.add_argument("--lane-spacing", type=float, default=160.0)
    parser.add_argument("--travel-spacing", type=float, default=5.0)
    return parser


def _validation_polydata(vertices: np.ndarray, triangles: np.ndarray) -> pv.PolyData:
    cells = np.column_stack(
        (np.full(len(triangles), 3, dtype=np.int64), triangles)
    ).ravel()
    return pv.PolyData(vertices, cells)


def show_planar_rasters(
    model_path: Path,
    lane_spacing_mm: float,
    travel_spacing_mm: float,
) -> None:
    """Show nominal surface paths only; no stand-off or transition poses."""

    model = load_step_model(model_path)
    settings = PlannerSettings(
        lane_spacing_mm=lane_spacing_mm,
        travel_spacing_mm=travel_spacing_mm,
    )
    plans = plan_planar_faces(model.topods_faces, model.faces, settings)

    plotter = pv.Plotter()
    mesh = model.validation_mesh.mesh
    plotter.add_mesh(
        _validation_polydata(
            np.asarray(mesh.vertices),
            np.asarray(mesh.faces),
        ),
        color="lightgray",
        opacity=0.35,
        show_edges=False,
    )
    for plan in plans:
        for raster in plan.raster_segments:
            points = np.asarray([sample.point for sample in raster.samples])
            if len(points) >= 2:
                plotter.add_mesh(
                    pv.lines_from_points(points, close=False),
                    color="blue",
                    line_width=4,
                )
    plotter.add_axes()
    plotter.show_grid()
    plotter.show(
        title=(
            f"Planar raster debug: {len(plans)} faces, "
            f"{sum(len(plan.raster_segments) for plan in plans)} sections"
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        show_planar_rasters(
            args.model,
            lane_spacing_mm=args.lane_spacing,
            travel_spacing_mm=args.travel_spacing,
        )
    except (CadModelError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
