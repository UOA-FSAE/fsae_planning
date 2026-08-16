# Planning/control stack: delay compensation, NMPC, centralized tuning, live scoring

Closes the gap between offline-tuned weights and what the car actually runs:
adds pose-age delay compensation and a second, curvature-aware NMPC
controller to the control loop, centralizes every MPC weight/gain/flag into
one dataclass shared by both controller nodes, and fixes live runs being
scored identically to offline tuner rollouts (previously pinned at the DNF
floor regardless of driving quality). Also tightens planner/perception
timing (split pose/cone publish rates, atomic pose+twist snapshot) and fixes
a cone-map duplication bug. Full detail grouped by subsystem below; the
`fsae_MPCTest` repo's `docs/planning_control_sync.md` and
`docs/logs/late_turn_in_investigation.md` carry the offline validation this
was tuned against.

## Perception (`sim_perception.py`)

- Split pose and cone publishing onto two independent timers (`pose_rate`
  20 Hz, `cone_rate` 10 Hz, both new params). Pose previously shared the
  cone map's slower 10 Hz timer, so the 20 Hz MPC sometimes re-solved
  against a stale, unchanged pose.
- Added `/fsae/slam/car_odom` (`nav_msgs/Odometry`): position/yaw and
  speed/yaw-rate from the *same* odom snapshot, published atomically. The
  MPC controllers now read this instead of subscribing to
  `/fsds/testing_only/odom` directly, which had no guarantee the pose and
  twist it read on a given tick came from the same underlying sample.
- `car_position` (`geometry_msgs/Pose`) is now `PoseStamped`, with
  `header.stamp` carrying the odom's own measurement time — needed for the
  delay-compensation feature below. All four subscribers
  (`centerline_planner.py`, `skidpad_planner.py`, `stanley_controller.py`,
  `mpc_controller.py`) updated to match.
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
- Added `tracking_error_speed_gate()`: scales the speed target down when
  lateral or heading tracking error is large, with a floor so the car
  always retains enough speed to steer itself back. `curvature_speed()`
  alone only looks at path shape and has no way to know the car isn't
  actually near that path.
- Added `load_speed_profile_csv()` / `load_path_profile_csv()` /
  `precomputed_speed_at()`: optional CSV-backed speed/path lookup for a
  track that's already been mapped (see `map_path`/`path_map_path` below),
  bypassing the live planner's re-derivation entirely.

## Control — MPC core (`mpc_core.py`)

- Added pose-age-based delay compensation: `compute()` now takes
  `pose_age_s` and rolls the error state forward through recently issued
  commands (`predict_ahead()`) before solving, so the MPC plans against the
  state it will actually face rather than a stale measurement. The step
  count is low-pass filtered and hysteresis-gated (`_update_n_delay()`) so
  ordinary control-loop jitter doesn't flip it between adjacent values and
  inject a step disturbance into the QP.
- `e_yd` (lateral error rate) now includes the body-frame lateral velocity
  term (`car_vy * cos(e_psi)`), not just `car_speed * sin(e_psi)` — the
  previous formula silently dropped this term whenever the car had real
  sideslip.
- Lateral error `e_y` is now the signed perpendicular projection onto the
  path segment, not the full Euclidean distance to the nearest path point
  (the latter only equals the correct value when that point happens to sit
  exactly abeam of the car).
- Added a hard steering-rate limit expressed as deg/s × dt (180 deg/s)
  rather than a fixed per-step angle, so it survives a change of `dt`.
- MPC horizon `N` raised 25 → 35 steps; `MAX_BRAKE` reduced 9 → 7 m/s²;
  `Cf`/`Cr` and `Q`/`R`/`R_rate` weights retuned (kept numerically identical
  to `fsae_MPCTest/settings.py` and the `fsds_simulator` mirror per this
  repo's parity rule).
- Added (disabled-by-default) `_adaptive_Q_scaling()` and reference-heading
  rate limiting, both mirroring `fsae_MPCTest`'s equivalents and gated
  behind flags for future re-evaluation.
- Added `terminal_scale` (extra terminal-state cost weight, currently a
  1.0 no-op) for parity with the offline tuner's `TERMINAL_Q_SCALE`.
- **Removed the entire forward-scanning lookahead gain-scheduling family**
  (~15 functions: lookahead approach/exit boosts, demand normalisation, the
  U-turn detector, straight-line adjustments, curvature forcing, and the
  precomputed `CornerMap`/`use_precomputed_corner_map` fast path added
  earlier this session) and replaced it with `_corner_factor`/
  `_low_speed_corner_boost`: one continuous CURRENT-curvature-only fraction
  blending four weights between a straight and a corner endpoint, plus an
  independent, always-on heading-error-driven accel/brake asymmetry
  (`epsi_ra_*`). The forward-scanning family reweighted today's (usually
  near-zero) cost based on a future corner, which doesn't change what the
  QP's own horizon predicts once the car gets there — see
  `fsae_MPCTest/docs/planning_control_sync.md`'s "Corner-factor scheduler
  rewrite" section for the full reasoning and mapping. Mirrored the same day
  into `fsae_MPCTest/fsds_simulator/` and `controller/model_utils.py`.

## Control — nodes (`mpc_controller.py`, `stanley_controller.py`)

- `mpc_controller.py`: added `map_path`/`path_map_path` params to swap in a
  precomputed speed profile and/or path for an already-mapped track,
  bypassing `curvature_speed()` and/or the live planner subscription.
- Added the tracking-error speed gate and a speed-target rise-rate limiter
  (braking is never delayed, only the "speed up" direction is capped) to
  both `mpc_controller.py` and `mpc_controller_standalone.py`, so a car
  that's badly off-line is told to slow down instead of speed up.
- Both controller nodes now slice the tracked path from the car's nearest
  point before measuring curvature, instead of always measuring from the
  path's start — fixes an inconsistency where the two nodes measured
  curvature differently.
- Updated to subscribe to `car_position` as `PoseStamped` and use
  `/fsae/slam/car_odom` for speed/yaw-rate (see perception changes above).

## New nodes/files

- `mpc_controller_standalone.py`: an alternative MPC controller node that
  sends the MPC's own throttle/brake commands directly instead of routing
  a speed target through `fsds_bridge`'s separate P-loop — the offline
  tuner's longitudinal tuning otherwise never reaches the real car.
- `scoring.py`: verbatim copy of `fsae_MPCTest/sim/scoring.py`'s composite
  score, so a live run and an offline tuner rollout are graded identically.
  `telemetry_logger.py` now computes and prepends this score to the
  control CSV on close.
- `cone_recorder.py` (+ `cone_recorder.launch.py`): records one lap's cone
  map to a file once the car completes a loop back to its start, for later
  reconstruction/reuse by the offline tuner.
- `nmpc_core.py` + `nmpc_params.py`: a SECOND controller, a Frenet-frame
  nonlinear MPC (Gauss-Newton SQP, condensed QP solved by OSQP), selected by
  a new node parameter `use_nmpc` (default `false` — `mpc_core.py`/
  `mpc_params.py` are byte-unchanged when off). Closes the structural gap the
  linear QP has: its prediction model has no term for the path itself
  bending (`e_psi_dot = r`, missing `- kappa(s)*s_dot`), so with the car
  dead on-line approaching a corner it predicts staying on-line forever —
  measured, not assumed, at exactly 0.000 deg commanded across 8 synthetic
  test states. The NMPC instead tracks arc length `s` as a state and looks
  up `kappa(s)` directly, so a bend ahead is part of the dynamics rather
  than bolted onto the cost (three earlier attempts at the latter all
  produced a wrong-direction steering transient — see
  `fsae_MPCTest/docs/logs/late_turn_in_investigation.md` Parts 2/7/15).
  Offline closed-loop A/B (`comp_test_map_3`, identical weights, same
  simulated plant): steering saturation 12.5% → 0.8%, |e_y| p90 1.45 → 0.69m,
  lap 43.1 → 42.0s, turns in earlier on 7/7 corners tested (median 25.6m).
  Cross-checked against an independent CasADi+IPOPT solve (not used in the
  shipped code — not installable into this ROS interpreter without
  `--break-system-packages`). Reproduce everything with
  `control/fsae_control/test/nmpc_offline_check.py` (no ROS/FSDS needed).
  **Live-tested, matched same-day pair on `comp_test_map_3`**: steering
  saturation 6.45% → 0.58%, lap 54.72s → 52.35s, composite score 0.695 →
  0.532, |e_psi| mean 7.85° → 5.06° — same direction as the offline A/B.
  Full design record and offline validation:
  `fsae_MPCTest/docs/logs/late_turn_in_investigation.md` Part 16.
  Mirrored into `fsae_MPCTest/fsds_simulator/` and documented in that repo's
  `planning_control_sync.md`, `tuning.md` §4.5d, `architecture.md`, and
  `docs/logs/changes_2026-08-13_nmpc.md`.

## Telemetry (`telemetry_logger.py`)

- New CSV columns: `delta_cmd`, `a_cmd`, `solver_failed`, `inaccurate`, and
  five latency-diagnostic columns (`pose_age_s`, `path_age_s`, `n_delay`,
  `solve_ms`, `cmd_latency_ms`) to check the delay-compensation feature
  above against reality.
- `close()` now prepends a `#`-commented score header (see `scoring.py`).
- **Fixed: every live run's `composite_score` was pinned at `13.0`.** Both
  controller nodes called `telemetry.close()` with no arguments, so
  `progress` defaulted to `0.0` and `reached_end` to `None` —
  `compute_composite_score()` reads that as "the run never finished" and
  always returns `CONSTRAINT_FLOOR + DNF_PENALTY = 13.0`, no matter how well
  the car actually drove. Added `LapProgressTracker`: tracks the car's
  position against the precomputed track path to compute real
  `progress`/`reached_end`, and integrates `ds / v_target` over the
  already-loaded speed profile for a `time_bonus` (an `optimal_time` bound,
  without needing `fsae_MPCTest`'s `speed_profile.optimal_lap_time()`, which
  isn't on the live node's `PYTHONPATH`). Both controller nodes now pass the
  tracker's output into `close()`; the header also gains
  `lap_time_s`/`optimal_time_s`. Only takes effect when a precomputed speed
  profile is loaded (`map_path` set) — a run against the live planner topic
  still has no known path end, so `progress`/`reached_end` fall back to the
  old defaults in that mode.
- 8 new `nmpc_*` columns appended to `ADAPTIVE_COLUMNS` (empty on every
  LTV-QP run, exactly like the `m_*` columns are empty on Stanley runs):
  solver iterations/status/cost, the car's Frenet arc-length position, and
  the prediction's own terminal `e_y`/`e_psi`/curvature. Lets a log
  distinguish "the NMPC's model/solver disagreed with reality" from "the
  weights are wrong," which the LTV-QP's telemetry can't do.
- **Fixed: `ADAPTIVE_COLUMNS` still listed the deleted lookahead family's
  columns (`kappa_max_abs`, `m_Q_ey_approach`, `uturn_severity`,
  `kappa_horizon_end`, etc. — ~23 columns), which `mpc_core.py` no longer
  writes, so they silently logged empty on every run.** Meanwhile the
  corner-factor rewrite's own telemetry (`corner_factor`, `corner_frac`,
  `Q_ey_base`, `Rrate_steer_corner_blend`, etc. — 9 keys) was written to
  the `adapt` dict every tick but had no matching column, so it was
  silently dropped from every CSV instead. `ADAPTIVE_COLUMNS` now lists
  exactly what `compute()` currently writes. Mirrored same-day into
  `fsae_MPCTest/fsds_simulator/`.

## Control — centralized MPC tuning (`mpc_params.py`)

- New `MPCParams` dataclass: every `mpc_core.py` weight/gain/flag that used
  to be a hardcoded module-level constant or inline list literal (Q/R/R_rate
  weights, adaptive-gain shape constants, delay-compensation limits, the
  a_lat-ceiling law, feature-enable flags — 44 fields as of the corner-factor
  rewrite above, down from an original ~56; that rewrite deleted more fields
  than the later NMPC work added) now lives in one place, with the docstring
  cross-referencing the matching
  `fsae_MPCTest/settings.py` constant for each field. `mpc_core.py`'s
  `MPCController` takes an `MPCParams` instance (defaulting to
  `DEFAULT_MPC_PARAMS`) instead of reading bare constants. Pure mechanical
  relocation — no default changes prior behaviour.
- `mpc_controller.py`/`mpc_controller_standalone.py` now `declare_mpc_params()`
  every field as a ROS2 parameter (default = `DEFAULT_MPC_PARAMS`) and build
  the `MPCParams` passed to `MPCController` from the live values
  (`mpc_params_from_node()`), so every weight is retunable at launch time
  without a code change.
- `control.launch.py`/`sim.launch.py` generate one `DeclareLaunchArgument`
  per `MPCParams` field mechanically from `MPC_PARAM_FIELDS` (field name,
  default, metadata) instead of 35+ hand-written near-identical blocks —
  removes the drift risk of the launch args silently diverging from the
  dataclass defaults.
- `fsae_params.yaml`'s `controller:` block gains matching YAML defaults for
  every field, at full float precision to stay identical to `mpc_params.py`.

## Control — NMPC MPCC-inspired additions (`nmpc_core.py`, `nmpc_params.py`)

Compared the NMPC against Alexander Liniger's MPCC (Model Predictive
Contouring Control) for transplantable ideas. Full MPCC (progress `θ` as a
maximised decision variable) was assessed and rejected — `θ̇`-maximisation is
a more aggressive version of the "exogenous, schedulable future obligation"
failure mode `kappa(s)`-as-state already exists to avoid (see the forward-scanning-family
removal above). Three narrower ideas were adopted instead, NMPC-only
(`mpc_core.py`/LTV-QP untouched), mirrored identically into
`fsae_MPCTest/controller/nmpc_optimiser.py`:

- `nmpc_spline_reference_enabled` (default **true**): analytic
  `CubicSpline`-derived `kappa(s)`/`psi_ref(s)`, replacing the moving-average
  + finite-difference pipeline. Targets the known open centreline-curvature-spike
  defect. Live-tested (implicitly, on by default): steering saturation
  0.21%, |e_y| mean 0.213 m on `comp_test_map_3` — tighter than the
  pre-existing NMPC baseline (0.58% saturation).
- `nmpc_horizon_speed_profile_enabled` (default **false**, experimental):
  samples a precomputed speed profile at each horizon stage's own predicted
  arc length, mirroring `kappa(s)`'s non-schedulable pattern. **Live-tested
  and rejected**: `v_actual` hit ~16.7 m/s against a `v_desired` of
  ~3.3–5 m/s for ~2 s approaching a corner, car went off-track by up to
  3.6 m. Cause: the QP sums `v_x − v_ref(s_k)` across all 20 horizon stages,
  so a high `v_ref` at a later stage (post-corner straight) can offset a low
  `v_ref` at an earlier stage (the corner itself) within the same solve —
  the non-schedulability property does not transfer to a *summed* cost the
  way it does to a single state-coupled term. Left in the code, disabled.
- `nmpc_friction_circle_enabled` (default **false**, experimental): hard
  per-axle `|F_yf|,|F_yr| <= F_max` QP constraint, additive to (not
  replacing) the existing soft `alat_ceiling` tanh saturation. **Live-tested
  and rejected, more severely than the speed profile**: SQP failed to solve
  on 77.5% of all 614 ticks, steering pinned at the mechanical ±25° lock on
  30.8% of ticks starting at t=0.65s, run ended in a full stall (4.94 m
  off-track, heading error −52°). Cause: the hard force bound has no slack
  variable (unlike the soft tanh saturation beside it), so when ordinary
  cornering geometry conflicts with it — which happens under normal driving
  on this track, not just extreme conditions — the subproblem goes
  infeasible and steering stops updating. `F_max = m·ceiling(v_x)/2` per
  axle is measurably tighter than normal cornering needs. Left in the code,
  disabled.

**Fixed, same day: NMPC steered hard-right to the ±25° mechanical lock for
the first ~0.5–0.7s of every run from a standing start.** Not caused by any
of the three features above (reproduces with only the spline reference
active) and not track/spawn geometry (LTV-QP and Stanley see the identical
initial `e_y≈0.16°` at the same spawn point with no equivalent snap). Cause:
the tyre slip-angle formula `alpha_f = arctan((v_y+lf*r)/v_safe) - d` floors
its denominator (`v_safe = max(|v_x|, v_blend_hi)`) to avoid a divide-by-zero
at `v_x=0`, but this makes a *stationary* tyre's slip angle track the
commanded steering directly (`alpha_f ≈ -d`), producing a large fictitious
cornering force from steering alone at zero forward speed — backwards from a
real tyre. Fixed by scaling `F_yf`/`F_yr` by the existing `blend` factor
(already used for the kinematic/dynamic mix) immediately after computing
them. Verified: parity holds (1e-13/1e-14), full `nmpc_offline_check` suite
passes, no closed-loop regression. Not yet live-retested.

Also fixed in this pass: `nmpc_fyf_max_abs`/`nmpc_fyr_max_abs` were added to
`last_telemetry` but never added to `telemetry_logger.py`'s
`ADAPTIVE_COLUMNS`, so they silently never reached any CSV.

See `fsae_MPCTest/docs/planning_control_sync.md`'s "Three MPCC-inspired
additions" and "Which settings affect which controller" sections for the
full mechanism and field-by-field controller-scope map.

## Interface/config

- `fsae_params.yaml`: `look_radius`/`plan_horizon`/`pose_rate`/`cone_rate`
  updated to match the values above.
- `setup.py` (all four packages): `zip_safe=False` (works around a stale
  `colcon build` issue); new entry points for `mpc_controller_standalone`
  and `cone_recorder`.
- `centerline_planner.py`/`skidpad_planner.py`: updated to subscribe to
  `car_position` as `PoseStamped` instead of `Pose`.
