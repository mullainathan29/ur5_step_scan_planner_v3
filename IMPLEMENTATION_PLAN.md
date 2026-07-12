# STEP B-Rep Planner Refactor Plan

## Scope of this phase

This document records the current nominal-planning dependency on the tessellated
mesh and defines the modular replacement. The current application is preserved
unchanged at `legacy/ur5_step_scan_planner_v3.py`. No B-Rep planner is implemented
in this phase.

## Current architecture and mesh-dependent nominal path generation

The application is a single module containing CAD import, tessellation, nominal
planning, pose generation, validation, display, GUI, and export.

The complete STEP-to-nominal-pose call chain is:

1. `MainWindow.load_model_dialog()` calls `load_model()` and stores only the
   returned `trimesh.Trimesh`; it does not retain the CadQuery/OCC shape.
2. `load_model()` calls `cadquery_shape_from_step()` and then
   `tessellate_cadquery_shape()` for STEP input.
3. `cadquery_shape_from_step()` imports the STEP B-Rep, but its result survives
   only long enough to be tessellated.
4. `tessellate_cadquery_shape()` calls `shape.Faces()` and
   `face.tessellate()` for every face. It records only face index, broad geometry
   type, and area; UV bounds, wires, exact normals, orientation, and adjacency
   are not retained. Mesh cleanup can change triangle indices, after which the
   triangle-to-STEP-face map is replaced with `-1` values.
5. `MainWindow.generate_plan()` passes only the mesh to `generate_scan_plan()`.
6. `generate_scan_plan()` iterates hard-coded principal/isometric view
   definitions and calls `plan_view_fixed()` for each view.
7. `plan_view_fixed()` is the nominal mesh planner. It uses
   `view_projection_extents()` on mesh vertices to form a rectangular view grid,
   then calls `ray_first_hits()` to choose nominal surface points. It reads
   `mesh.face_normals[triangle_index]` as nominal normals and offsets poses in a
   fixed view direction. Thus neither lane clipping nor point generation uses
   trimmed B-Rep face parameter space.
8. `plan_view_fixed()` also calls `simulate_laser_line_coverage()` before
   accepting each nominal center. This function casts more mesh rays through
   `ray_first_hits()`, reads triangle normals, checks range/incidence, and returns
   triangle coverage. Although conceptually validation, it is currently an
   acceptance gate in nominal generation.
9. `split_fixed_view_samples()` and `create_fixed_view_motion_segment()` consume
   the mesh-derived samples. They do not tessellate directly, but their segments,
   surface points, normals, and fixed orientations all inherit that geometry.
10. `generate_scan_plan()` computes area coverage from `mesh.area_faces` and
    returns the triangle mask. This is valid as approximate coverage validation,
    but it must not determine the nominal face paths.

Related mesh-only functions that may remain on the validation/visualization side
are `trimesh_to_pyvista()`, `_show_mesh()`, and `_show_plan()`.

The generated JSON in the workspace corroborates the failure mode: 315 STEP
faces are reduced to 181,150 triangles; only 20 scan poses in four isometric
segments survive, with 1.930655% triangle-area coverage. All five principal
views produce zero segments.

## Target module boundaries

### `cad_model.py`

- Import STEP and retain the authoritative CadQuery/OCC shape.
- Define typed `CadModel` and stable per-load face identity records.
- Extract all `TopoDS_Face` objects without tessellating them for planning.
- Build a separate validation/display mesh with an explicit, preserved
  triangle-to-face mapping.
- Reject unsupported nominal STL planning with a descriptive exception (STL can
  remain a validation/display input only if that workflow is explicitly kept).

### `face_analysis.py`

- Define `SurfaceType` and typed `FaceInfo` records.
- Record face ID, OCC face, orientation, surface type, area, UV bounds, outer and
  inner boundary wires, representative exact normal, and adjacent face IDs.
- Classify plane, cylinder, cone, sphere, torus, B-spline, and other surfaces.
- Centralize UV-in-trim classification and face-orientation-aware exact normal
  evaluation.

### `planners/base.py`

- Define planner protocol/abstract base class and shared typed values such as
  `UVSample`, `SurfaceSample`, `ScanLane`, `ScanSegment`, and `PlannerSettings`.
- Dispatch by `SurfaceType`.
- Provide shared sampling, trim testing, endpoint handling, serpentine ordering,
  continuity checks, and descriptive unsupported-surface errors.
- Never connect unrelated faces. Output face-local nominal scan lanes only.

### `planners/plane.py`

- Build a stable in-plane 2D frame from exact plane geometry.
- Generate parallel raster lanes, clip them to outer/inner trimmed boundaries,
  sample at travel spacing, and serpentine-order the resulting intervals.

### `planners/cylinder.py`

- Use angular and axial parameters within trimmed UV limits.
- Support longitudinal lanes first; add helical mode only after trimmed periodic
  behavior is tested.
- Handle angular seams and face orientation explicitly.

### `planners/cone.py`

- Generate continuous angular/axial paths in conical UV coordinates.
- Account for changing physical lane width and exclude the singular apex region.

### `planners/sphere.py`

- Generate trimmed latitude or longitude paths.
- Adapt parameter steps to physical spacing and avoid pole singularities.

### `planners/torus.py`

- Compare usable physical U/V spans and choose iso-U or iso-V lanes.
- Handle both periodic seams and trimmed islands.

### `planners/bspline.py`

- Generate trimmed iso-parametric U and V candidates.
- Score candidates using physical length, curvature/normal change, discontinuity
  risk, and robot-motion smoothness; choose the better direction.

### `pose_generation.py`

- Convert accepted UV samples to exact 3D points and orientation-aware normals.
- Construct scanner frames with local +Z toward the surface, local +X along the
  laser line, and local +Y along travel.
- Use normalized, sign-continuous quaternions internally.
- Offset at nominal stand-off, enforce the 200–420 mm range policy, and split at
  boundaries, discontinuities, excessive position/normal/orientation changes,
  occlusion, and invalid range.
- Add approach, lead-in, scan, lead-out, and retract poses while keeping scan and
  transition trajectories distinct.

### `coverage.py`

- Own tessellation, ray intersection, first-surface visibility, incidence/range
  checks, laser-line simulation, collision approximation, and triangle-area
  coverage reporting.
- Validate B-Rep-generated poses; never create their nominal centerline or lane
  order.
- Report coverage by triangle and, through the preserved map, by STEP face.

### `exporter.py`

- Own CSV/JSON schemas and serialization.
- Export XYZ, quaternion, rotation vector, and RPY; clearly mark RPY as export
  only and include face/segment/lane identity and scan-vs-transition type.

### `gui.py`

- Own Qt widgets and PyVista presentation only.
- Orchestrate model load, face analysis, planner dispatch, pose generation,
  validation, and export through module APIs.
- Keep expensive work out of UI logic and retain both B-Rep model and validation
  mesh in application state.

## Recommended implementation sequence

1. Add typed domain models, descriptive exceptions, and package structure.
2. Implement STEP loading, face extraction, metadata, adjacency, exact UV point
   and normal evaluation, and independent validation tessellation.
3. Add unit tests with synthetic planar and cylindrical trimmed faces before
   changing the GUI path.
4. Implement the base planner plus plane raster planning and validate holes,
   concave trims, serpentine order, spacing, and face containment.
5. Implement cylinder planning and validate partial/full periodic cylinders,
   seams, reversed faces, and axial trims.
6. Implement cone, sphere, torus, and B-spline planners with focused geometry
   tests.
7. Implement quaternion pose generation, continuity splitting, and explicit
   transition poses.
8. Move existing mesh-ray coverage into `coverage.py` and apply it only after
   nominal B-Rep paths exist.
9. Move exports and GUI into their modules, then switch the entry point from the
   legacy pipeline to the new orchestrator.
10. Add integration tests against representative STEP models and compare
    per-face coverage, gaps, segment continuity, and deterministic output.

At each substantial step, run syntax checks and the full test suite. Robot
safety must remain explicitly unclaimed until IK, joint limits, calibrated
frames, and robot/workcell collision validation exist.

## CadQuery/OCP APIs to verify before implementation

The installed environment currently contains CadQuery 2.8.0 and OCP 7.9.3.1.
The following bindings exist locally, but semantics, overloads, periodic behavior,
units, tolerance handling, and reversed-face behavior must be verified with small
executable probes and tests:

- STEP import and topology: `cadquery.importers.importStep()`,
  `cq.Shape.Faces()`, `cq.Face.wrapped`, `OCP.TopExp.TopExp_Explorer`, and
  `OCP.TopoDS.TopoDS.Face_s()`.
- CadQuery face convenience APIs: `cq.Face.geomType()`, `Area()`, `uvBounds()`,
  `positionAt()`, `normalAt()`, `outerWire()`, and `innerWires()`. Verify whether
  normalized-vs-native UV conventions make the convenience evaluators unsuitable
  for planner internals.
- Exact surface adaptation/classification: `OCP.BRepAdaptor.BRepAdaptor_Surface`
  with `GetType()`, `FirstUParameter()`, `LastUParameter()`,
  `FirstVParameter()`, `LastVParameter()`, `Value()`, `D0()`, `D1()`, and
  `Plane()`, `Cylinder()`, `Cone()`, `Sphere()`, `Torus()`, `BSpline()`.
- Surface type enums: `OCP.GeomAbs.GeomAbs_SurfaceType` values for plane,
  cylinder, cone, sphere, torus, B-spline, Bezier, revolution, extrusion, offset,
  and other surfaces.
- Trim bounds and classification: `OCP.BRepTools.BRepTools.UVBounds_s()`,
  `OCP.BRepClass.BRepClass_FaceClassifier`, `Perform()`, `State()`,
  `OCP.gp.gp_Pnt2d`, and `OCP.TopAbs.TopAbs_State`. Verify boundary tolerance,
  holes, multiple wires, and periodic seam points.
- Boundary traversal and p-curves: `cq.Face.outerWire()`, `innerWires()`,
  `OCP.BRepTools.BRepTools_WireExplorer`, and the relevant
  `BRep_Tool.CurveOnSurface_s()` overload. Verify edge orientation and parameter
  intervals in the face's native UV domain.
- Area/properties: `OCP.GProp.GProp_GProps` and
  `OCP.BRepGProp.BRepGProp.SurfaceProperties_s()`, including whether location and
  units are already applied.
- Exact normals: `BRepAdaptor_Surface.D1()` cross products and/or
  `OCP.BRepLProp.BRepLProp_SLProps` / `OCP.GeomLProp.GeomLProp_SLProps`.
  Verify normal sign correction for `TopAbs_REVERSED`, degeneracies, and
  continuity at seams.
- Adjacency: `OCP.TopExp.TopExp.MapShapesAndAncestors_s()` with
  `OCP.TopTools.TopTools_IndexedDataMapOfShapeListOfShape`, mapping edges to
  faces. Verify shape hashing/equality and non-manifold edges.
- Tessellation for validation only: `cq.Face.tessellate()` or OCC
  `BRepMesh_IncrementalMesh` plus `BRep_Tool.Triangulation_s()`. Verify that
  triangle-to-face identity is captured before any mesh merge/deduplication and
  remains valid afterward.
- Curve sampling/length where needed: OCC adaptor and GCPnts APIs (notably
  `BRepAdaptor_Curve`, `GCPnts_AbscissaPoint`, and uniform-abscissa sampling).
  Verify behavior for p-curves versus 3D curves and closed/periodic intervals.

## Acceptance gates for the rewrite phase

- Every nominal scan pose carries a STEP face ID and native UV coordinate.
- Every nominal surface point and normal is evaluated from that B-Rep face.
- No nominal lane center or ordering is derived from mesh vertices, triangle
  centroids, random samples, or mesh-ray hits.
- Validation can reject/split a B-Rep lane for visibility/range/coverage, but it
  cannot invent nominal lanes.
- Plane and cylinder unit tests pass before GUI integration.
- Scan and transition trajectories are separately identifiable, quaternion
  continuity is tested, and no unrelated faces are directly connected.
