# Planning/control stack: speed planning, live scoring, telemetry

Improves the Stanley control loop and the planner/perception it depends on,
and adds live scoring so a real run is directly comparable to an offline
rollout. Full detail grouped by subsystem below.

## Perception (`sim_perception.py`)

- Split pose and cone publishing onto two independent timers (`pose_rate`
  20 Hz, `cone_rate` 10 Hz, both new params). Pose previously shared the
  cone map's slower 10 Hz timer; since the control loop is directly
  triggered by each car_position arrival, that capped it at 10 Hz too.
- `car_position` (`geometry_msgs/Pose`) is now `PoseStamped`, with
  `header.stamp` carrying the odom's own measurement time. All subscribers
  (`centerline_planner.py`, `skidpad_planner.py`, `stanley_controller.py`)
  updated to match.
- `look_radius` raised 18 m → 25 m to stay ahead of the planner's own
  lookahead increase (below).

## Planning (`boundary.py`, `cone_map.py`)

- `_WALL_PLAN_HORIZON` / `look_radius` raised 15/18 m → 25 m, to stay ahead
  of the controller's braking-distance scan (see `curvature_speed()`
  below) — a corner had to be visible far enough out to plan a speed
  profile that can actually slow down for it in time.
- Path-seed selection now rejects only midpoints clearly *behind* the car
  (small negative margin) instead of requiring a positive forward
  projection. The old cutoff could discard the single nearest midpoint as
  the car's heading rotated through a corner, forcing the seed to jump to a
  much farther point and deleting a real corner from the published path
  right as the car needed to react to it.
- `ConeMap._absorb()` now also checks new detections against each other
  within the same batch, not just against already-stored cones. A cone
  seen for the first time by two detections in the same frame previously
  became two permanent duplicate entries.

## Control — speed planning (`control_utils.py`)

- `curvature_speed()` rewritten to propagate a braking-distance limit: each
  corner in the scan window is converted to the fastest speed the car can
  still be going now and brake for that corner given its distance, and the
  most restrictive wins. Previously the function returned a single
  worst-curvature target regardless of how far away that corner was, which
  could demand more deceleration than the car can produce.
  `scan_start`/`scan_end` widened to 0–24 m to match.
- Curvature estimation now denoises the scan window (dense resample +
  moving average) before measuring Menger curvature, and reduces over the
  smoothed series with a max rather than a raw per-triple max — the
  planner's frame-to-frame centreline refit was injecting sub-corner noise
  that made the speed target oscillate on straights.
- Added `load_speed_profile_csv()` / `load_path_profile_csv()` /
  `precomputed_speed_at()`: optional CSV-backed speed/path lookup for a
  track that's already been mapped (see `map_path`/`path_map_path` in
  `stanley_controller.py`), bypassing the live planner's re-derivation
  entirely.

## Control — Stanley node (`stanley_controller.py`)

- Added `map_path`/`path_map_path` params to swap in a precomputed speed
  profile and/or path for an already-mapped track, bypassing
  `curvature_speed()` and/or the live planner subscription.
- Updated to subscribe to `car_position` as `PoseStamped` (see perception
  changes above).

## New nodes/files

- `scoring.py`: verbatim copy of `fsae_MPCTest/sim/scoring.py`'s composite
  score, so a live run and an offline tuner rollout are graded identically.
  `telemetry_logger.py` now computes and prepends this score to the
  control CSV on close.
- `cone_recorder.py` (+ `cone_recorder.launch.py`): records one lap's cone
  map to a file once the car completes a loop back to its start, for later
  reconstruction/reuse by an offline tuner.
- `target_speed_viz.py`: debug node that republishes `curvature_speed()`'s
  per-waypoint target as a visualisable profile, for offline speed-target
  diagnosis without touching the 20 Hz control loop.

## Telemetry (`telemetry_logger.py`)

- New CSV columns: `delta_cmd`, `a_cmd`, `solver_failed`, `inaccurate`, and
  latency-diagnostic columns (`pose_age_s`, `path_age_s`, `n_delay`,
  `solve_ms`, `cmd_latency_ms`) — optional, written empty by controllers
  that don't have them.
- `close()` now prepends a `#`-commented score header (see `scoring.py`).
- **Fixed: every live run's `composite_score` was pinned at `13.0`.** The
  controller node called `telemetry.close()` with no arguments, so
  `progress` defaulted to `0.0` and `reached_end` to `None` —
  `compute_composite_score()` reads that as "the run never finished" and
  always returns `CONSTRAINT_FLOOR + DNF_PENALTY = 13.0`, no matter how well
  the car actually drove. Added `LapProgressTracker`: tracks the car's
  position against the precomputed track path to compute real
  `progress`/`reached_end`, and integrates `ds / v_target` over the
  already-loaded speed profile for a `time_bonus` (an `optimal_time` bound,
  without needing `fsae_MPCTest`'s `speed_profile.optimal_lap_time()`, which
  isn't on the live node's `PYTHONPATH`). The controller node now passes the
  tracker's output into `close()`; the header also gains
  `lap_time_s`/`optimal_time_s`. Only takes effect when a precomputed speed
  profile is loaded (`map_path` set) — a run against the live planner topic
  still has no known path end, so `progress`/`reached_end` fall back to the
  old defaults in that mode.

## Interface/config

- `fsae_params.yaml`: `look_radius`/`plan_horizon`/`pose_rate`/`cone_rate`
  updated to match the values above.
- `setup.py` (all four packages): `zip_safe=False` (works around a stale
  `colcon build` issue); new entry point for `cone_recorder`.
- `centerline_planner.py`/`skidpad_planner.py`: updated to subscribe to
  `car_position` as `PoseStamped` instead of `Pose`.
