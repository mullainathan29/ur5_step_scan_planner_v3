# UR5 STEP Scan Planner

## Project objective

Build an offline Cartesian scan-path planner for a UR5 carrying a
Micro-Epsilon scanCONTROL LLT30x0-200 profile scanner.

The input is a STEP file. STEP B-Rep faces must be used as the primary
trajectory-planning geometry.

The triangle mesh may be used only for:

- visualization;
- ray intersection;
- visibility verification;
- collision approximation;
- coverage verification.

Do not generate nominal scan paths from random mesh samples or triangle
centroids.

## Scanner model

Micro-Epsilon scanCONTROL LLT30x0-200.

Default parameters:

- nominal stand-off: 310 mm;
- sensor Z range: 200–420 mm;
- nominal laser-line width at 310 mm: 200 mm;
- scanner local +Z points toward the surface;
- scanner local +X is the laser-line direction;
- scanner local +Y is the travel direction.

## Required planning architecture

1. Load STEP using CadQuery/OpenCascade.
2. Extract every B-Rep face.
3. Record:
   - face ID;
   - surface type;
   - area;
   - UV bounds;
   - boundary wires;
   - surface normal;
   - adjacency.
4. Classify surfaces:
   - plane;
   - cylinder;
   - cone;
   - sphere;
   - torus;
   - B-spline;
   - other.
5. Generate trajectories directly in face parameter space.
6. Convert UV samples to 3D surface points and exact normals.
7. Offset scanner poses while respecting stand-off and scanner direction.
8. Split scan segments at:
   - face boundaries;
   - discontinuities;
   - excessive position changes;
   - excessive normal changes;
   - occlusion;
   - invalid sensor range.
9. Simulate laser-line coverage using a tessellated validation mesh.
10. Export XYZ, quaternion, rotation vector and RPY.

## Surface-specific planning

### Plane

- Build a local 2D face frame.
- Generate parallel raster lanes clipped to the trimmed face boundary.
- Use serpentine ordering.
- Sample travel points at configurable spacing.

### Cylinder

- Use axial and angular parameters.
- Generate longitudinal lanes or a helical path.
- Respect trimmed angular and axial limits.

### Cone

- Generate paths in conical UV coordinates.
- Maintain continuous angular and axial ordering.

### Sphere

- Generate latitude or longitude paths inside trimmed UV limits.

### Torus

- Generate iso-U or iso-V paths according to the larger usable span.

### B-spline

- Generate iso-parametric U or V curves.
- Select the direction that produces smoother and shorter robot motion.

## Robot-oriented constraints

- No random-point trajectory.
- No direct connection between unrelated faces.
- A segment must have continuous position and orientation.
- Use quaternions internally.
- RPY is export-only.
- Add approach, lead-in, scan, lead-out and retract poses.
- Keep scan and transition trajectories separate.
- Do not claim robot safety without IK and collision validation.

## Code requirements

- Keep geometry, planning, validation, GUI and export code in separate modules.
- Add type hints.
- Add descriptive exceptions.
- Add unit tests for plane and cylinder trajectory generation.
- Avoid changing working environment files unless necessary.
- Run syntax checks and tests after every substantial change.