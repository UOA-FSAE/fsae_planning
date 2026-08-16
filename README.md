# fsae_planning (simulator stack)

FSDS-simulator testbed for the Formula Student Driverless planning stack, tested against the
[FSDS simulator](https://github.com/FS-Driverless/Formula-Student-Driverless-Simulator).

This repo mirrors the **structure and ROS interface of the car monorepo** (`fsae_autonomous`):
same folder layout, same `fsae_interfaces` messages, same `/fsae/*` topic names, same central
parameter config. The planners developed and tuned here speak the exact interface the car stack
consumes, so they can move onto the real car with no I/O changes. The simulator-specific parts are
confined to two thin bridge nodes that adapt FSDS ↔ the `/fsae/*` interface.

---

## Quick Start

**Terminal 1 — Simulator**
```bash
cd ~/repo/fsds-v2.2.0-linux
./FSDS.sh
```

**Terminal 2 — ROS 2 bridge**
```bash
cd ~/ros2_fsd
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch fsds_ros2_bridge fsds_ros2_bridge.launch.py UDP_control:=false
```

**Terminal 3 — Autonomous stack**
```bash
cd ~/ros2_fsd
source /opt/ros/jazzy/setup.bash && source install/setup.bash
ros2 launch fsae_bringup sim.launch.py                              # mpc_standalone (default)
ros2 launch fsae_bringup sim.launch.py planner:=skidpad_planner
ros2 launch fsae_bringup sim.launch.py controller:=stanley          # use the Stanley controller
```

The **planner is selected at launch** (`planner:=…`); the mode wires the rest automatically
(skidpad publishes its own command and needs the full cone map, so `sim.launch.py` skips the
Stanley controller and sets `sim_perception full_track:=true` in that mode).

| `planner` | Package / node | Description |
|-----------|----------------|-------------|
| `centerline_planner` *(default)* | `planning/fsae_planning` | Cone-wall centreline planner: follows a rolling first-lap centreline every lap, no localisation |
| `skidpad_planner` | `planning/fsae_planning/special_utils` | Figure-8 characterisation: laps a precomputed figure-8 at a ramping speed and logs the spin-off (sim-only extension) |

The **controller is selected independently** (`controller:=…`):

| `controller` | Node | Description |
|-----------|----------------|-------------|
| `mpc_standalone` *(default)* | `control/fsae_control` | LTV-QP MPC (`mpc_core.py`), commands throttle/brake directly — bypasses `fsds_bridge`'s separate speed P-loop so the MPC's own longitudinal tuning actually reaches the car/sim |
| `mpc` | `control/fsae_control` | Same LTV-QP MPC, routes a speed target through `fsds_bridge`'s P-loop instead — matches the car stack's split more closely |
| `stanley` | `control/fsae_control` | Original Stanley controller (`cmd_vel`) |

Both `mpc`/`mpc_standalone` can run either of two optimisers under the hood, picked with
`use_nmpc:=true|false` (default `false`) — see "MPC tuning" below and
[control/fsae_control/fsae_control/nmpc_core.py](control/fsae_control/fsae_control/nmpc_core.py).
`record_cones:=false` skips the `cone_recorder` node that otherwise launches alongside every mode
to log one completed lap's boundary cones for offline reuse (default on;
`cone_out_path:=''` → `~/fsae_logs/cone_map_<timestamp>.json`).

**Build**
```bash
source /opt/ros/jazzy/setup.bash
# one-time dependency (control interface uses ackermann_msgs):
sudo apt install ros-jazzy-ackermann-msgs
cd ~/ros2_fsd && colcon build
```

---

## Package structure

Mirrors `fsae_autonomous`: folders group packages by subsystem (colcon discovers packages
recursively). Vendored/bridge packages are the only sim-specific additions.

```
fsae_planning/
├── common/
│   ├── fsae_interfaces/        # vendored msgs (Track, ConeDetection, …) — matches the car repo
│   └── fsae_bringup/           # central fsae_params.yaml + launch composition (sim.launch.py)
├── perception/
│   └── fsae_sim_perception/    # sim stand-in for camera+SLAM: FSDS oracle+odom → /fsae/* inputs
├── planning/
│   └── fsae_planning/          # centerline_planner, skidpad_planner + utils
└── control/
    └── fsae_control/           # controller (Stanley → cmd_vel) + fsds_bridge (cmd_vel → FSDS)
```

---

## ROS interface

The core pipeline speaks only `/fsae/*` topics and `fsae_interfaces` / standard messages — identical
to the car. Two bridge nodes isolate everything FSDS-specific.

```
[FSDS]  /fsds/testing_only/track (oracle, latched)   ┐
        /fsds/testing_only/odom  (ground truth)      ├─▶ sim_perception ─┐
                                                      ┘                   │  /fsae/slam/car_position   (PoseStamped)
                                                                          ├─ /fsae/slam/car_odom       (Odometry)
                                                                          ├─ /fsae/slam/left_track     (Track, blue)
                                                                          └─ /fsae/slam/right_track    (Track, yellow)
                                                                             /fsae/perception/cone_detection (ConeDetection)
                                                                                       │
                                                                                       ▼
                                                                     centerline_planner
                                                                                       │  /fsae/planning/selected_trajectory (PoseArray)
                                                                                       ▼
                                                                                  controller
                                                                                       │  /fsae/control/cmd_vel (AckermannDriveStamped)
                                                                                       ▼
                                            fsds_bridge ─▶ /fsds/control_command (fs_msgs/ControlCommand) ─▶ [FSDS]
```

| Topic | Type | Producer → Consumer |
|-------|------|---------------------|
| `/fsae/slam/car_position` | `geometry_msgs/PoseStamped` | sim_perception → planners, controller. x,y in `pose.position`; **yaw (rad) in `pose.orientation.w`** (car-stack convention). `header.stamp` carries the odom's own measurement time, for delay compensation (see MPC core below) — was a bare `Pose` until this was needed. Drives the plan/control loops. |
| `/fsae/slam/car_odom` | `nav_msgs/Odometry` | sim_perception → controller. Position/yaw and speed/yaw-rate from one atomic snapshot. Exists because reading pose and twist off separately-timed topics gave no guarantee a given control tick's pair actually came from the same sample; the controller nodes read this instead of `/fsds/testing_only/odom` directly. |
| `/fsae/slam/left_track` | `fsae_interfaces/Track` | sim_perception → planners. Blue (left) boundary, global frame (`Point[] cones`). |
| `/fsae/slam/right_track` | `fsae_interfaces/Track` | sim_perception → planners. Yellow (right) boundary, global frame. |
| `/fsae/perception/cone_detection` | `fsae_interfaces/ConeDetection` | sim_perception → fsds_bridge (proximity brake). Local-frame detections + embedded `car_pose`. |
| `/fsae/planning/selected_trajectory` | `geometry_msgs/PoseArray` | planners → controller. Centreline waypoints (positions only). |
| `/fsae/control/cmd_vel` | `ackermann_msgs/AckermannDriveStamped` | controller (or skidpad_planner) → fsds_bridge. `speed` + `steering_angle`. |

**Simulator boundary (FSDS `fs_msgs`, sim-only):** `sim_perception` subscribes the latched oracle
map `/fsds/testing_only/track` and `/fsds/testing_only/odom`; `fsds_bridge` subscribes
`/fsds/testing_only/odom` (speed feedback) and `/fsds/signal/go` (start gating) and publishes
`/fsds/control_command`. The skidpad planner also reads `/fsds/testing_only/odom` for ground-truth
speed logging.

### Coordinate frame

FSDS uses **ENU**: `x` forward, `y` left, `z` up. Blue cones = left boundary, yellow = right.
`cmd_vel.steering_angle` is radians (positive = left); `fsds_bridge` normalises it to FSDS
`ControlCommand.steering` (`+1` = right).

---

## CSV telemetry logs

Set the `log_csv` parameter on a controller node (`log_dir` defaults to `~/fsae_logs`) to
write two CSVs per run, via `fsae_control/telemetry_logger.py`:

- `<tag>_control_<stamp>.csv` — one row per control step (20 Hz)
- `<tag>_path_<stamp>.csv` — planned-path snapshots (~1 Hz), long form

**Frame.** Everything positional is in the **global ENU frame** described above, written
exactly as received — the logger performs no coordinate conversion, so control rows and path
rows overlay directly on one plot with no transform. The two error signals are the exception:
they are Frenet-style, measured **at the front axle** relative to the nearest path segment,
not relative to the world.

**Time.** `t` is **seconds since the run started** (first logged step = `0.0`), not a ROS
epoch stamp — so it reads directly as lap time. Both CSVs share one origin. The epoch of that
origin is preserved in the header as `t0_epoch_s` if you need to line a run up against a rosbag.

### `<tag>_control_<stamp>.csv`

| Column | Unit | Meaning |
|---|---|---|
| `t` | s | Run-relative time, `0.0` at first control step |
| `car_x` | m | Global ENU east position |
| `car_y` | m | Global ENU north position |
| `car_yaw` | rad | Global ENU heading, right-handed, `0` = +x/east |
| `v_actual` | m/s | Measured forward speed |
| `v_desired` | m/s | Target speed (post-filtering) |
| `steer_deg` | deg | Commanded **roadwheel** angle, +ve = left |
| `e_y` | m | Lateral error at front axle, +ve = car is **left** of path |
| `e_psi_deg` | deg | Heading error vs path tangent, +ve = CCW/left |
| `yaw_rate` | rad/s | Measured yaw rate, +ve = CCW/left |
| `delta_cmd` | rad | MPC steering command (`u[0]`) — `steer_deg` in radians |
| `a_cmd` | m/s² | MPC longitudinal command (`u[1]`), −ve = braking |
| `solver_failed` | 0/1 | MPC solve failed this step |
| `inaccurate` | 0/1 | Solver returned `OPTIMAL_INACCURATE` |
| `pose_age_s` | s | Age of the pose this solve used (now − pose header stamp) |
| `path_age_s` | s | Age of the planner path this solve used |
| `n_delay` | steps | Rollforward depth the controller chose to compensate the lag |
| `solve_ms` | ms | QP solve wall time |
| `cmd_latency_ms` | ms | Control-loop entry → command published |

### Latency diagnostics

The last five columns exist to answer one question: **does the real latency
chain match what the offline simulator assumes?** `fsae_MPCTest` models a fixed
`DELAY_STEPS = 1` (50 ms) of actuation lag on a perfectly uniform 20 Hz loop;
these columns let a live run be checked against that assumption instead of it
being taken on trust.

Read them together — they separate the failure modes:

- **`pose_age_s` >> 50 ms** → the controller is steering toward where the car
  *was*. This is the leading hypothesis for the heading-error gap.
- **`path_age_s` large but `pose_age_s` fine** → the reference is stale, not the
  state. Expected to sit near ~1 s since the planner publishes at ~1 Hz against
  a 20 Hz control loop; that alone is not a fault.
- **`solve_ms` ≈ `cmd_latency_ms`** → the QP dominates the loop budget.
- **`cmd_latency_ms` >> `solve_ms`** → the time is going somewhere other than
  the solver.
- **`n_delay` changing tick to tick** → the compensation depth is dithering,
  which is itself a source of oscillation (the hysteresis in `mpc_core` exists
  to prevent this; this column verifies it works).

All five are optional in `log_control()` and written as empty cells when the
caller doesn't supply them, so the Stanley controller still logs normally.

`delta_cmd`/`a_cmd` are logged in the MPC's own units rather than normalised FSDS command
units so the score can be recomputed from the file without re-deriving any scaling.

> **Units gotcha.** `log_control()` takes **radians**. Passing the normalised
> `ControlCommand.steering` (`[-1, 1]`) instead silently inflates `steer_deg` by
> ~2.3× and still looks plausible — this bug once hid live slew-rate saturation
> for a whole tuning cycle. Scale by `MAX_STEER_RAD` first.

### `<tag>_path_<stamp>.csv`

| Column | Unit | Meaning |
|---|---|---|
| `t` | s | Run-relative time of this snapshot |
| `idx` | — | Waypoint index within the snapshot |
| `x`, `y` | m | Global ENU waypoint position (same frame as `car_x`/`car_y`) |

### Score header

On close, the control CSV is rewritten with a `#`-commented block holding the run's composite
score and every component metric, computed by `fsae_control/scoring.py` — a verbatim copy of
the offline tuner's `sim/scoring.py`. A live score is therefore directly comparable to an
offline one. `pandas.read_csv(path, comment='#')` parses the file unchanged; numpy needs
`skip_header=<number of '#' lines>` because `genfromtxt` won't skip comments when locating the
`names=True` row.

When a precomputed speed profile is loaded (`map_path` set — the normal live-driving
setup), `telemetry_logger.py`'s `LapProgressTracker` derives real `progress`/`reached_end`/
`time_bonus` from the car's position against that path, integrating `ds / v_target` over
the profile for the time bound. The car still can't measure `offtrack` (needs
ground-truth track edges), and a run against the live planner topic instead of a
precomputed profile has no known path end either — either case leaves those terms at
zero/`False` and the header records `score_is_partial=1`. The weighted-metric component
is directly comparable regardless.

---

## How the pieces map to the car

| Sim node | Car-stack counterpart it stands in for |
|----------|----------------------------------------|
| `sim_perception` | ZED `cone_detection_node` + `cone_mapper` (SLAM) — produces `car_position` + `left/right_track`. **Stands in for range only, not accuracy** — see the limitation note below. |
| `centerline_planner` | upstream `centerline_planner` (same topics; our cone-wall implementation) |
| `skidpad_planner` | sim-only extension (no car-stack equivalent) |
| `controller` | `stanley_controller` (`cmd_vel`) |
| `fsds_bridge` | `ack_to_can_node` — turns `cmd_vel` into the vehicle command bus |

### Simulator fidelity limits

`sim_perception` replaces the camera + SLAM front-end, but it only models
**limited range** — not sensing error. Specifically:

- **Pose is exact.** It copies FSDS's ground-truth `/fsds/testing_only/odom`
  verbatim. No noise, no drift, no estimation lag. The real car's pose comes
  from ZED visual odometry + `cone_mapper` and has all three.
- **Cones are an oracle.** The map is FSDS's exact cone list, cropped to a
  forward window and radius. No false positives/negatives, no position error,
  no colour confusion.

So a clean FSDS run does **not** certify real-car behaviour. The offline tuner
can model the localisation half of this (`SLAM_NOISE_ENABLED` in
`fsae_MPCTest/settings.py`, default off since FSDS has no such error); the
cone-detection half is not modelled anywhere yet.

One rate constraint worth knowing: **`pose_rate` must be >= the controller's
rate** (`CONTROL_HZ` = 20 Hz). Pose and cones previously shared a single 10 Hz
timer, so the 20 Hz MPC re-solved against an unchanged pose on 50.5% of control
steps, which drove steering oscillation. They are now separate timers
(`pose_rate` 20 Hz, `cone_rate` 10 Hz).

---

## Parameters

Central config: [common/fsae_bringup/config/fsae_params.yaml](common/fsae_bringup/config/fsae_params.yaml),
keyed by node name (ROS matches params by node name, so `name == executable == key`).

| Node | Key params |
|------|-----------|
| `sim_perception` | `look_ahead` (25 m), `look_wide` (10 m), `min_ahead` (0.5 m), `look_radius` (25 m), `full_track` (false), `pose_rate` (20 Hz), `cone_rate` (10 Hz) |
| `centerline_planner` | `smooth` (0.015), `look_radius` (25 m), `plan_horizon` (25 m), `path_blend` (0.4) |
| `skidpad_planner` | `v_start` (3 m/s), `ramp_accel` (0.25 m/s²), `v_cap` (25 m/s) |
| `controller` | `v_max` (15 m/s), `v_min` (1.5 m/s), `stanley_gain` (1.0), plus every MPC tuning field below |

### MPC tuning (`mpc` / `mpc_standalone` only)

Single source of truth: [control/fsae_control/fsae_control/mpc_params.py](control/fsae_control/fsae_control/mpc_params.py)'s
`MPCParams` dataclass — every field there (Q/R/R_rate weights, adaptive-gain
shape constants, feature flags; ~56 total) has a matching key in
`fsae_params.yaml`'s `controller:` block with the same default, and a
matching `DeclareLaunchArgument` in `control.launch.py`/`sim.launch.py`
(generated from `MPCParams` itself, not hand-written, so the three can't
drift against each other). Override any of them the same way as
`v_max`/`v_min` above, e.g.:

```
ros2 launch fsae_bringup sim.launch.py q_e_y:=6.5 adaptive_r_rate_during_floor:=0.55
```

`ros2/launch_all.sh` also exposes a small shortlist of the most commonly
retuned fields (`MPC_Q_E_Y`, `MPC_Q_E_PSI`, `MPC_R_DELTA`,
`MPC_ADAPTIVE_R_RATE_DURING_FLOOR`, `MPC_ADAPTIVE_R_RATE_ENTERING_FLOOR`) as
commented-out shell variables near the top of that script — uncomment and
set one to override it for that launch without touching `fsae_params.yaml`.

Must be kept numerically identical to `fsae_MPCTest/settings.py`'s matching
constants (see the outer repo's `CLAUDE.md` for the parity rule) — that repo
cannot import this one, so the two are kept in sync by hand, the same as
`Q_diag`/`R_diag`/`R_rate_diag` always have been. The weights are NOT
unit-normalised (e.g. `q_e_y`, on metres², and `q_e_psi`, on radians², are
not on a comparable scale) — see `mpc_params.py`'s own field comments for
each constant's unit and tuning history.

### NMPC — second controller (`use_nmpc`, `mpc` / `mpc_standalone` only)

`use_nmpc:=true` swaps in
[control/fsae_control/fsae_control/nmpc_core.py](control/fsae_control/fsae_control/nmpc_core.py)
(`NMPCController`, Frenet-frame nonlinear MPC — Gauss-Newton SQP, condensed QP solved by OSQP) in
place of `mpc_core.py`'s LTV-QP. Default `false`; `mpc_core.py`/`mpc_params.py` are byte-unchanged
when off. Every field lives in `nmpc_params.py`'s `NMPCParams` dataclass, wired through
`fsae_params.yaml`/`control.launch.py`/`sim.launch.py` the same mechanical way as `MPCParams` above
— same override syntax, e.g. `use_nmpc:=true nmpc_horizon:=25`.

Exists because the LTV-QP's linear prediction has no term for the path itself bending
(`e_psi_dot = r`, missing `- kappa(s)*s_dot`) — a car dead on-line approaching a corner is predicted
to stay on-line forever (measured, not assumed: exactly 0.000° commanded across 8 synthetic test
states). The NMPC instead tracks arc length `s` as a state and looks up `kappa(s)` directly, so a
bend ahead is part of the dynamics rather than bolted onto the cost. Offline closed-loop A/B
(`comp_test_map_3`, identical weights, same simulated plant): steering saturation 12.5% → 0.8%,
turns in earlier on 7/7 corners tested (median 25.6 m earlier). **Live-tested, matched same-day
pair**: steering saturation 6.45% → 0.58%, lap 54.72s → 52.35s, composite score 0.695 → 0.532,
|e_psi| mean 7.85° → 5.06°.

Reproduce the offline A/B with no ROS/FSDS needed:
[control/fsae_control/test/nmpc_offline_check.py](control/fsae_control/test/nmpc_offline_check.py)
(optionally cross-checks against `fsae_MPCTest`'s closed-loop rollout if that repo is checked out
alongside; degrades to synthetic-state checks only if it isn't). Must be kept numerically identical
to `fsae_MPCTest/settings.py`'s NMPC constants, same parity rule as `MPCParams` above.

### Precomputed-map launch args (`sim.launch.py` / `control.launch.py`)

For a track that's already been mapped, the `mpc`/`mpc_standalone`
controllers can bypass parts of the live planner/perception pipeline and
track a precomputed path/speed pair instead:

| Launch arg | Default | Effect |
|------|---------|--------|
| `map_path` | `tracks/comp_test_map_3/speed_profile.csv` (this repo's own `tracks/`, under `ros2/src/fsae_planning/`) | CSV to read for `use_precomputed_speed` |
| `use_precomputed_speed` | `true` | Look up target speed from `map_path`'s oracle profile instead of live `curvature_speed()` per tick |
| `path_map_path` | `tracks/comp_test_map_3/raceline.csv` | CSV to read for `use_precomputed_path` (same `x,y,psi,v_target` format, different file — see below) |
| `use_precomputed_path` | `true` | Track `path_map_path`'s precomputed path instead of subscribing to `centerline_planner.py`'s `/fsae/planning/selected_trajectory` — removes the live planner from the control loop entirely, to isolate controller/plant tracking error from planner-induced path error. On by default so `mpc`/`mpc_standalone` track the precomputed oracle path/speed pair; override with `use_precomputed_path:=false` for the planner-vs-controller isolation / live-planner-in-loop experiment mode. |

Example: `ros2 launch fsae_bringup sim.launch.py use_precomputed_path:=false`

`comp_test_map_3` (cone map + both exported CSVs) is committed under this
repo's own `tracks/comp_test_map_3/`, so a checkout of FSDS + `fsae_planning`
alone can drive it with no other repo cloned. **To switch to a different
already-committed track, edit `ros2/launch_all.sh`'s `TRACK=` variable** —
it expands to both launch args above from this repo's `tracks/<TRACK>/`, so
the per-controller defaults here only matter for a bare `ros2 launch`
outside that script.

**To record and export a NEW track**, see the separate `fsae_MPCTest` repo's
developer guide, section "Recording, exporting and driving a track"
(`docs/developer_guide.md` in that repo) — that repo's tools write directly
into this repo's `tracks/<name>/` when both are checked out side by side.
Copy the resulting directory in manually if you're exporting from elsewhere.
`fsae_MPCTest` is only needed to *produce* a new track; nothing here needs
it to *drive* one that already exists. This repo has no fixed relative path
to `fsae_MPCTest` — it does not need to be nested under the same root, or
even present, to build/drive `fsae_planning` on its own.

---

## Planning algorithm

The active planner (`build_path_walls` in
[planning/fsae_planning/fsae_planning/boundary.py](planning/fsae_planning/fsae_planning/boundary.py))
connects same-colour boundary cones into a wall mesh, then generates midpoints by anchoring on
whichever boundary has more cones in view and matching each anchor cone to its single nearest
unclaimed opposite-colour cone (within a bounded pairing distance) — an exclusive one-to-one match
rather than pairing every cone within range, which avoids one cone on the sparser side fanning out
into several conflicting midpoints on tight corners. Those midpoints are chained with a greedy walk
that penalises steps crossing the wall mesh, then fit with a cubic spline. Boundary cones arrive
already colour-separated and in the global frame on `left_track` / `right_track`, and are
accumulated into a persistent `ConeMap`. The chain is clamped to an arc-length horizon
(`plan_horizon`) and temporally blended frame-to-frame (`path_blend`) so the published centreline
stays stable as the car drives. This is a **first-lap centreline follower** — it replans the same
rolling window every lap and does not build or optimise a raceline.

Speed is **curvature-limited in the controller** (`curvature_speed`,
`v = safety·√(a_lat_max / κ_peak)` with a short-path cap) rather than published by the planner —
the trajectory interface (PoseArray) carries no speed channel, matching the car stack.

---

## Git convention

Branch names: `feat/… | fix/… | chore/… | docs/… | refactor/…` (kebab-case).
**Delete short-lived branches after merging.** Commit subject ≤ 50 chars, imperative; wrap the
body at 72 chars explaining what and why.
