# UR5 STEP Scan Planner V3

This revision corrects the large scanner-space jumps found in V2.

## Main corrections

### 1. Fixed-view scan segments

Each scan segment belongs to one selected view:

- top
- front
- right
- back
- left
- four upper isometric views

The scanner position is now calculated as:

```text
scanner position = surface hit + stand-off × selected view direction
```

The scanner no longer follows every local triangle normal while crossing a
surface. This prevents large arcs caused by rapidly changing normals.

### 2. Automatic segment splitting

A scan segment is split when either condition is exceeded:

- maximum consecutive scanner-position movement;
- maximum consecutive surface-normal change.

Default limits:

```text
Maximum scanner step:  12 mm
Maximum normal change: 15 degrees
```

### 3. Fixed orientation inside a segment

Within a segment:

- scanner `+Z` points toward the part;
- scanner `+X` is the laser-line direction;
- scanner `+Y` is the travel direction.

The orientation does not oscillate with tessellated triangle normals.

### 4. Laser-line coverage simulation

At every accepted scan centre, the software simulates multiple rays distributed
over the laser line. Each ray is checked for:

- first-surface visibility;
- sensor range;
- incidence angle.

The default uses 21 rays over the 200 mm laser line.

### 5. Coverage map

The CAD display is coloured:

- green: geometrically covered;
- red: not covered.

Coverage percentage is calculated using tessellated triangle area.

### 6. Safer orientation export

The output still includes roll, pitch and yaw, but now also exports:

```text
qx, qy, qz, qw
rx, ry, rz
```

Use quaternion or rotation-vector orientation for robot interpolation. Do not
interpolate Euler angles near pitch ±90 degrees.

## Installation

### Existing virtual environment

```bash
cd ~/ur5_step_scan_planner_v3
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Conda

```bash
conda env create -f environment.yml
conda activate ur5_step_scan_v3
```

## Run

```bash
python ur5_step_scan_planner_v3.py
```

## Inspect STEP B-Rep faces

Phase 1 of the B-Rep rewrite provides a command-line diagnostic that loads the
authoritative STEP topology, extracts every face, and prints counts grouped by
surface type:

```bash
python -m diagnostics.inspect_step model.step
```

The extracted B-Rep faces are kept separate from the tessellated mesh used for
rendering and future coverage validation.

To inspect exact planar raster paths on the B-Rep surfaces before scanner
stand-off and transition offsets, run:

```bash
python -m diagnostics.view_planar_rasters "479 -Part model.stp" \
  --lane-spacing 160 \
  --travel-spacing 5
```

Phase 3 also provides exact cylindrical planning through
`planners.cylinder.plan_cylindrical_face`. The default strategy is
`CylinderStrategy.LONGITUDINAL`; `CylinderStrategy.HELICAL` selects a
trim-aware helical path.

To validate coverage using the tessellated mesh only after B-Rep planning:

```bash
python -m diagnostics.view_coverage "479 -Part model.stp" \
  --lane-spacing 20 \
  --travel-spacing 5
```

Add `--include-internal-surfaces` to include bore and cavity surfaces in the
coverage denominator. Use `--no-display` for a report-only run.

## Unified B-Rep application

The current phase 1–4 implementation is exercised without the legacy planner by:

```bash
python main.py "479 -Part model.stp"
```

For a report/export-only run:

```bash
python main.py "479 -Part model.stp" \
  --surface-types plane cylinder \
  --lane-spacing 20 \
  --travel-spacing 5 \
  --stand-off 310 \
  --no-display \
  --output-dir output
```

Geometry Validation mode prints every scan pose's measured stand-off and
continuity statistics. With display enabled it adds colour-coded surface-to-
scanner connectors and a **Show scanner offset connectors** checkbox:

```bash
python main.py "479 -Part model.stp" --geometry-validation --display
```

## Recommended initial LLT30x0-200 parameters

```text
Stand-off distance:             310 mm
Laser-line width:               200 mm
Lane overlap:                   20%
Travel pose spacing:            5 mm
Minimum sensor range:           200 mm
Maximum sensor range:           420 mm
Laser coverage rays:            21
Maximum incidence angle:        60 degrees
Maximum scanner step:           12 mm
Maximum normal change:          15 degrees
Maximum orientation step:       5 degrees
Minimum valid laser fraction:   0.20
Approach/retract clearance:     120 mm
Lead-in/out distance:           20 mm
```

## Output motion types

Each scan segment contains:

```text
approach
lead_in
scan
lead_out
retract
```

Only rows where:

```text
motion_type = scan
scan_enabled = true
```

should trigger profile acquisition.

## Remaining requirements before real UR5 execution

This remains a Cartesian scan-planning prototype. Before sending a path to the
physical robot, add:

1. CAD/part frame to UR5 base-frame calibration.
2. UR5 flange to scanner optical-frame calibration.
3. Inverse kinematics with a continuous seed solution.
4. Joint-limit and wrist-flip checks.
5. Robot, scanner, fixture and workpiece collision checking.
6. Velocity and acceleration time parameterization.
7. Transition planning between retract and approach poses.
8. Simulation and low-speed dry-run validation.
9. Comparison of predicted and acquired coverage.

The displayed coverage is geometric and approximate. It does not replace the
manufacturer's optical calibration model or real measurement validation.
