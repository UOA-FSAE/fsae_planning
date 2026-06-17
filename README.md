# fsae_planning

Autonomous path-planning and control stack for Formula Student Driverless, developed and tested against the [FSDS simulator](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator).

---

## Quick Start

**Terminal 1 — Simulator**
```bash
cd ~/repo/fsds-v2.2.0-linux
./FSDS.sh
```

**Terminal 2 — ROS2 Bridge**
```bash
cd ~/ros2_fsd
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py UDP_control:=false
```

**Terminal 3 — Autonomous Stack**
```bash
cd ~/ros2_fsd
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch fsae_planning launch_planning.py
```

**Build**
```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select fsae_planning
```

---

## Codebase Structure

```
fsae_planning/
├── launch/
│   └── launch_planning.py          # Launches all three nodes
│
└── fsae_planning/
    ├── track_utils/
    │   ├── perception_node.py      # Fake perception — sliding FOV window over oracle map
    │   ├── control_node.py         # Stanley path-tracking controller
    │   └── control_utils.py        # StanleyController class + legacy pure-pursuit helper
    │
    ├── planner_node.py             # Planning node — orchestrates the pipeline
    ├── boundary.py                 # Cone-wall mesh planner + ft-fsd trace-sort planner
    ├── cone_sorting.py             # Colour separation, NN sorting, pairing, FOV filtering
    ├── path_utils.py               # Centreline, spline smoothing, lookahead, speed, direction
    ├── cone_map.py                 # Persistent cone accumulator
    ├── localisation.py             # Loop-closure detection + closed-loop centreline
    └── viz_utils.py                # Non-blocking Matplotlib ego-view
```

---

## Node Architecture

```
[FSDS Simulator]
      │  /fsds/testing_only/track  (oracle cone map, latched)
      │  /fsds/testing_only/odom   (ground-truth odometry)
      │  /fsds/signal/go           (race start signal)
      ▼
┌─────────────────┐
│  PerceptionNode │  Publishes only cones inside a 25 m × 20 m forward window,
│  (perception)   │  simulating the FOV of a real perception stack.
└────────┬────────┘
         │  /FusionCones  (Track)
         ▼
┌─────────────────┐      /fsds/planned_path    (nav_msgs/Path)
│  PlannerNode    │ ───────────────────────────────────────────▶ ┌──────────────┐
│  (centreline_   │      /fsds/desired_speed   (Float32)    ───▶ │  ControlNode │
│   planner)      │      /fsds/lookahead_target (PointStamped)   │  (controller)│
└─────────────────┘                                              └──────┬───────┘
                                                                        │  /fsds/control_command
                                                                        ▼
                                                                 [FSDS Simulator]
```

### Coordinate Frame

FSDS uses **ENU** throughout:

| Axis | Direction |
|------|-----------|
| `x`  | Forward   |
| `y`  | Left      |
| `z`  | Up        |

Blue cones = left boundary, yellow cones = right boundary.
Steering output: `+1` = right, `−1` = left.

---

## Path Planning Algorithm

The active planner is `build_path_walls` in [boundary.py](fsae_planning/boundary.py).

### 1 — Persistent Cone Map (`cone_map.py`)

```
Each /FusionCones frame
        │
        ▼
  ConeMap.update()
  ├── new obs within 0.8 m of existing cone → running-average position update
  └── new obs beyond 0.8 m of all existing cones → appended as new entry
        │
        ▼
  Accumulated map (never forgets — cones persist after leaving sensor FOV)
```

Retaining historical cones ensures that cone walls behind the car remain active barriers even after the raw sensor window has moved forward past them.

### 2 — Cone-Wall Mesh

All same-colour cone pairs within **9 m** of each other are connected into a wall segment mesh — separately for blue (left) and yellow (right).

```
Blue cones:    B1 ── B2 ── B3
                  ╲    ╱
                   B2─B3        (all pairs ≤ 6 m form segments)

Yellow cones:  Y1 ── Y2 ── Y3
```

Wall cones are drawn from a window that extends **5 m behind** the car (`min_ahead = −5 m`) so that recently-passed cones still contribute as barriers.

### 3 — Candidate Midpoints

Midpoints are generated between blue-yellow pairs within **10 m**, subject to a validity filter:

> The blue cone must be **laterally to the left** of the yellow cone in the car's current frame (`lat_blue > lat_yellow`).

This eliminates midpoints that would land inside a boundary wall — they arise when a same-colour cone from an adjacent parallel track is incorrectly paired with a cone on the wrong side.

### 4 — Greedy Forward Walk

Midpoints are sorted by forward distance from the car. A greedy walk chains them in that order with the following step cost:

```
cost = distance
     + 5000 × (number of wall segments crossed)    ← cross-track guard
     + 2.0 × heading_change (radians)             ← direction consistency
```

**Sorting by forward distance before the walk enforces monotone forward progression** — no backward zigzag is possible because the walk can only advance to higher-index (further-ahead) midpoints.

The 500 m crossing penalty makes any step that passes through a cone wall effectively infinite cost, blocking the path from jumping to an adjacent parallel track.

### 5 — Spline Smoothing

The raw midpoint chain is smoothed via `smooth_centreline`:

1. Drop duplicate consecutive points
2. Remove direction reversals > 120° (bad pairings)
3. Fit a cubic spline with **chord-length parameterisation** (arc length as knot parameter — prevents backwards tangents at unevenly spaced midpoints)
4. Resample at uniform parameter spacing

**Fallback chain:** `build_path_walls` → `build_local_path` (simple NN-pair midpoints) → `None` (brake)

---

## Localisation Mode

`localisation.py` lets the planner switch from the rolling sensor window to
**closed-loop planning on the full accumulated map** once a complete lap has
been driven. It is gated by a single master switch in [planner_node.py](fsae_planning/planner_node.py):

```python
ENABLE_LOCAL_MODE = True   # set False to disable localisation entirely
```

While `False`, the planner behaves exactly as before — pure rolling-window
mapping, no loop-closure tracking.

### Why two modes?

The fake perception pipeline only ever publishes cones inside a 25 m × 20 m
forward window, so during the first lap the planner sees the track one corner
at a time and plans a *local* centreline. After a full lap the `ConeMap` holds
the entire track, so the path can be closed into a loop and the car can localise
against the known map instead of the cropped sensor frame.

### Loop-closure detection (`LoopClosureDetector`)

Closure is detected from the **path the car drives**, not from the cone borders.
As the car laps, its position is accumulated into a path polyline (one point
every `TRAJ_MIN_STEP` = 0.5 m). Local mode engages only when **all three**
requirements hold on the same frame:

| # | Requirement | Check |
|---|-------------|-------|
| 1 | Car back at start | Car departed > `DEPART_DIST` (12 m) from the start pose, then returned within `RETURN_RADIUS` (5 m) |
| 2 | Old cones re-detected | ≥ `MIN_REDETECT` (3) live cones lie within `MATCH_DIST` (1 m) of the cones first snapshotted at the start line |
| 3 | Path closes on itself | At least `MIN_TRAJ_PTS` (20) path points accumulated, so returning to the start closes the driven path into a full loop |

The detector latches once closed. The start pose and start-cone snapshot (live
cones within `START_RADIUS` = 12 m) are captured on the first frame after GO.

### Closed-loop planning (`build_completed_path`)

On closure, the accumulated driven path **is** the completed track centreline —
it is already in track order, so it is fitted **once** with a *periodic* cubic
spline to remove odometry jitter and close the seam between its end (back at the
start) and its beginning:

```
driven path polyline  ──→  periodic cubic spline  ──→  closed loop (cached)
```

Each subsequent frame, `roll_loop_to_car` reorients the cached loop so the
segment ahead of the car comes first (with the loop tail wrapped back on for a
seamless lookahead at the seam), then the usual lookahead + speed logic runs on
the **full completed loop**. The wall-mesh / direction-check / midpoint
machinery is bypassed in this mode.

### Drift monitoring (`compute_drift`)

The path is planned from the **map**, but perception is still consumed to watch
for odometry drift: for every live cone the distance to its nearest same-colour
map cone is measured. A mean above `DRIFT_WARN_DIST` (1.5 m) means the live
frame no longer overlays the map and is logged as a warning (and shown on the
plot).

---

## Speed Control

`compute_desired_speed` in [path_utils.py](fsae_planning/path_utils.py):

```
v_target = safety × √(a_lat_max / κ_peak)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `a_lat_max` | 4.0 m/s² | Maximum lateral acceleration |
| `safety` | 0.75 | Margin factor — compensates for spline underestimating true curvature |
| `scan_end` | 14 m | Scan window length |

**Short-path cap:** when the visible path is shorter than `scan_end`, `v_max` is scaled down proportionally:

```
v_max_eff = v_max × min(1.0, path_length / scan_end)
```

This prevents the car from accelerating into an unseen corner when only a short path is available.

Curvature `κ` is estimated via **Menger curvature** at step-spaced triplets along the path.

---

## Stanley Controller

`StanleyController` in [control_utils.py](fsae_planning/track_utils/control_utils.py):

```
δ = θ_e + arctan(k_cte · e / (v + k_soft)) − k_d · ω
```

| Term | Description |
|------|-------------|
| `θ_e` | Heading error — path tangent angle minus car yaw |
| `e` | Cross-track error — signed distance from front axle to nearest path point |
| `k_d · ω` | Yaw-rate damper — counters oscillation by opposing rapid heading changes |

Control point is projected to the **front axle** (`wheelbase = 1.5 m`). Output is clamped to `[−1, 1]` and normalised by `MAX_STEER_RAD = 25°`.

---

## Visualiser

`Visualizer` in [viz_utils.py](fsae_planning/viz_utils.py) renders a live non-blocking Matplotlib ego-view at 3 Hz showing:

| Layer | Colour |
|-------|--------|
| Blue cone-wall segments | Faint blue lines |
| Yellow cone-wall segments | Faint gold lines |
| Candidate midpoints | Grey dots |
| Accumulated cone map | Blue / gold filled circles |
| Planned centreline | Green dashed line (mapping) / solid line (local mode) |
| Start / loop-closure point | Red star |
| Car | Black filled triangle |

The view is car-relative (+y up = ahead, +x right = right of car).

**In localisation mode** the plot switches to a closed-loop view: the axes
auto-scale to fit the entire loop, the centreline is drawn as a solid line, the
title reads `LOCALISATION MODE — closed-loop tracking`, and the live
perception-vs-map drift (mean / max) is annotated in the top-left corner.

**Scroll to zoom:** scrolling the mouse wheel over the plot zooms in/out
(persists across frame redraws), so you can pull back to see more of the track
in either mode.

---

## Key Tuning Parameters

| Constant | File | Default | Effect |
|----------|------|---------|--------|
| `_WALL_MAX_DIST` | `boundary.py` | 7 m | Max distance to link same-colour cones into wall segments |
| `_WALL_MID_DIST` | `boundary.py` | 4 m | Max blue-yellow pair distance for midpoint generation |
| `_WALL_CROSS_PENALTY` | `boundary.py` | 5000 | Cost per wall segment crossed — raise to tighten cross-track guard |
| `_WALL_PATH_MAX_STEP` | `boundary.py` | 10 m | Max step between consecutive midpoints in the path |
| `MERGE_DIST` | `cone_map.py` | 0.8 m | Radius for merging repeated cone detections |
| `safety` | `path_utils.py` | 1.0 | Curvature-speed safety factor — lower to slow through corners |
| `a_lat_max` | `path_utils.py` | 4.0 m/s² | Max lateral acceleration for speed calculation |
| `V_MAX` / `V_MIN` | `planner_node.py` | 15 / 1.5 m/s | Speed envelope |
| `LOOKAHEAD_DIST` | `planner_node.py` | 4.0 m | Pure-pursuit lookahead distance (passed to Stanley target) |
| `ENABLE_LOCAL_MODE` | `planner_node.py` | True | Master switch for localisation / closed-loop mode |
| `DRIFT_WARN_DIST` | `planner_node.py` | 1.5 m | Mean perception-vs-map drift that triggers a warning |
| `DEPART_DIST` / `RETURN_RADIUS` | `localisation.py` | 12 / 5 m | Lap depart distance / return radius for path closure |
| `TRAJ_MIN_STEP` | `localisation.py` | 0.5 m | Spacing of accumulated driven-path points |

---

## Git Convention

### Branch Naming
```
feat/xyz  |  fix/xyz  |  chore/xyz  |  docs/xyz  |  release/v1.2.3
```
Use kebab-case.

**ALWAYS DELETE SHORTLIVED FIX/FEATURE BRANCH AFTER MERGING**

### Commit Format
```
<type>(<scope>)/<subject>

<body>
```

| Type | Use |
|------|-----|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no logic change |
| `refactor` | Code restructure (no fix/feat) |
| `perf` | Performance improvement |
| `test` | Tests |
| `chore` | Build / tooling |

**Rules:** subject ≤ 50 chars, all lowercase unless quoting; body wrapped at 72 chars explaining what and why.

**Examples:**
```
feat/add cone map localisation
fix(planner)/prevent cross-track midpoint pairing
docs/update README with wall-barrier algorithm
```
