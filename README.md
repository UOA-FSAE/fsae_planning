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
ros2 launch fsae_bringup sim.launch.py                              # centerline_planner (default)
ros2 launch fsae_bringup sim.launch.py planner:=raceline_planner
ros2 launch fsae_bringup sim.launch.py planner:=skidpad_planner
```

The **planner is selected at launch** (`planner:=…`); the mode wires the rest automatically
(skidpad publishes its own command and needs the full cone map, so `sim.launch.py` skips the
Stanley controller and sets `sim_perception full_track:=true` in that mode).

| `planner` | Package / node | Description |
|-----------|----------------|-------------|
| `centerline_planner` *(default)* | `planning/fsae_planning` | Barebone cone-wall centreline planner, no localisation |
| `raceline_planner` | `planning/fsae_planning` | Centreline mapping that closes the loop and switches to closed-loop raceline planning once a lap completes (sim-only extension) |
| `skidpad_planner` | `planning/fsae_planning/special_utils` | Figure-8 characterisation: laps a precomputed figure-8 at a ramping speed and logs the spin-off (sim-only extension) |

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
│   └── fsae_planning/          # centerline_planner, raceline_planner, skidpad_planner + utils
└── control/
    └── fsae_control/           # controller (Stanley → cmd_vel) + fsds_bridge (cmd_vel → FSDS)
```

`raceline_planner` subclasses `centerline_planner` — it reuses all the cone-wall mapping,
publishing and visualisation machinery and only adds loop-closure + closed-loop planning.

---

## ROS interface

The core pipeline speaks only `/fsae/*` topics and `fsae_interfaces` / standard messages — identical
to the car. Two bridge nodes isolate everything FSDS-specific.

```
[FSDS]  /fsds/testing_only/track (oracle, latched)   ┐
        /fsds/testing_only/odom  (ground truth)      ├─▶ sim_perception ─┐
                                                      ┘                   │  /fsae/slam/car_position   (Pose)
                                                                          ├─ /fsae/slam/left_track     (Track, blue)
                                                                          └─ /fsae/slam/right_track    (Track, yellow)
                                                                             /fsae/perception/cone_detection (ConeDetection)
                                                                                       │
                                                                                       ▼
                                                                   centerline_/raceline_planner
                                                                                       │  /fsae/planning/selected_trajectory (PoseArray)
                                                                                       ▼
                                                                                  controller
                                                                                       │  /fsae/control/cmd_vel (AckermannDriveStamped)
                                                                                       ▼
                                            fsds_bridge ─▶ /fsds/control_command (fs_msgs/ControlCommand) ─▶ [FSDS]
```

| Topic | Type | Producer → Consumer |
|-------|------|---------------------|
| `/fsae/slam/car_position` | `geometry_msgs/Pose` | sim_perception → planners, controller. x,y in `position`; **yaw (rad) in `orientation.w`** (car-stack convention). Drives the plan/control loops. |
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

## How the pieces map to the car

| Sim node | Car-stack counterpart it stands in for |
|----------|----------------------------------------|
| `sim_perception` | ZED `cone_detection_node` + `cone_mapper` (SLAM) — produces `car_position` + `left/right_track` |
| `centerline_planner` | upstream `centerline_planner` (same topics; our cone-wall implementation) |
| `raceline_planner`, `skidpad_planner` | sim-only extensions (no car-stack equivalent) |
| `controller` | `stanley_controller` (`cmd_vel`) |
| `fsds_bridge` | `ack_to_can_node` — turns `cmd_vel` into the vehicle command bus |

---

## Parameters

Central config: [common/fsae_bringup/config/fsae_params.yaml](common/fsae_bringup/config/fsae_params.yaml),
keyed by node name (ROS matches params by node name, so `name == executable == key`).

| Node | Key params |
|------|-----------|
| `sim_perception` | `look_ahead` (25 m), `look_wide` (10 m), `min_ahead` (0.5 m), `full_track` (false) |
| `centerline_planner` / `raceline_planner` | `plot` (matplotlib ego-view, default false) |
| `skidpad_planner` | `plot`, `v_start` (3 m/s), `ramp_accel` (0.25 m/s²), `v_cap` (25 m/s) |
| `controller` | `v_max` (15 m/s), `v_min` (1.5 m/s), `stanley_gain` (1.0) |

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
accumulated into a persistent `ConeMap`.

`raceline_planner` additionally watches the driven path for loop closure
([localisation.py](planning/fsae_planning/fsae_planning/localisation.py)); once a lap closes it
plans on the full completed loop (the raceline) and monitors perception-vs-map drift.

Speed is **curvature-limited in the controller** (`curvature_speed`,
`v = safety·√(a_lat_max / κ_peak)` with a short-path cap) rather than published by the planner —
the trajectory interface (PoseArray) carries no speed channel, matching the car stack.

---

## Git convention

Branch names: `feat/… | fix/… | chore/… | docs/… | refactor/…` (kebab-case).
**Delete short-lived branches after merging.** Commit subject ≤ 50 chars, imperative; wrap the
body at 72 chars explaining what and why.
