# Planner sync from `fsae_autonomous`

Ports the planner work done on the integration-testing bench
(`fsae_autonomous`, branch `test/integration-testing-for-planning-tommy`)
back into this repo. That branch took this repo's planner across at
`7bf246b` ("sim planner migration for integration testing") and then fixed
it against real rosbag runs over three commits; `c1594ec` here already
back-ported part of the first one. This picks up the rest.

Upstream commits covered:

| commit | what |
|--------|------|
| `15fe218` | Delaunay cone walls, minimum midpoint distance, truncate-not-invalidate path sanitisation |
| `e533c55` | Delaunay triangulation for midpoint matching (the corridor graph) |
| `8751593` | path-disappearing patch: corridor walk bounded by arc length, blend truncation, `_seg_intersect` fixes |

## Planning core (`boundary.py`, `path_utils.py`)

These files are platform-neutral numpy and were taken from `fsae_autonomous`
wholesale. Two comments were kept at this repo's wording, because they point
at things only this repo has: `_WALL_PLAN_HORIZON`'s derivation from
`control_utils.curvature_speed`'s scan window, and `path_utils`' pointer to
`StanleyController`'s sign-convention docstring. `cone_sorting.py` was
already identical.

### New: Delaunay corridor midpoints (`midpoint_method`)

The previous midpoint builder (`_gen_midpoints`, exclusive nearest-neighbour
matching) is still there and still selectable as `midpoint_method: 'nn'`, but
the default is now `'delaunay'`:

- Both boundaries are triangulated together. Each **mixed** triangle (its
  three vertices split 2-1 by colour) contributes its two cross-colour edges
  as a corridor "gate pair" (`_build_corridor_graph`, `_walk_corridor`,
  `_build_corridor_path`).
- Chaining gates this way is wall-safe *by construction* — a triangle's
  interior cannot reach any other triangle's edges — so the greedy walk's
  per-step wall-crossing check and its `_WALL_CROSS_PENALTY` cost term are
  not needed at all on this path.
- The walk terminates on real **arc length** (`max_arc_length`, fed from
  `plan_horizon`) rather than a gate count. Dense cone spacing produces much
  shorter gate-to-gate hops than `_WALL_PATH_MAX_STEP`, so the old shared
  count cap (`_WALL_PATH_MAX_WALK = 18`) could exhaust itself well inside
  25 m and truncate the path for no geometric reason.
  `_WALL_CORRIDOR_MAX_WALK = 120` remains only as a runaway-loop backstop.

Measured on `tracks/comp_test_map_3` (1000 raceline poses, cones cropped to
`look_radius` exactly as `sim_perception` crops them), `delaunay` is both
faster and better-behaved than `nn`: 1.3 ms vs 2.3 ms mean tick, 23.2 m vs
21.4 m mean published arc length, and it respects `plan_horizon` (max 25.2 m).

### `mid_max_gate` had to be retuned for this repo — read this before tuning

`boundary._WALL_MID_MAX_DELAUNAY` defaults to **5.0 m**, measured upstream on
the real stack's track (cone-spacing p99 ~4.2 m). **That default does not
transfer to this repo's simulator maps**, and the failure is silent.

A mixed triangle's gate edges include the *diagonal*, roughly
`hypot(track_width, cone_spacing)`. On `tracks/comp_test_map_3` — 3.50 m wide,
same-colour spacing up to 4.94 m — the mixed-triangle cross-colour edges run
to **5.35 m**, so `mid_max_gate = 5.0` rejects **47%** of all gates and the
corridor walk finds *no gates at all* on 172 of 200 sampled poses. When that
happens `build_path_walls` falls through to `build_local_path`'s naive
nearest-cone pairing, which is not clamped to `plan_horizon` — the observed
result was published "paths" of 84 m to 208 m of arc length inside a 25 m
cone window. Nothing logs an error; it just quietly stops being the planner
you configured.

`fsae_params.yaml` therefore sets `mid_max_gate: 6.5` for
`centerline_planner`, clearing the measured 5.35 m maximum with ~20% margin
while staying under `_WALL_MAX_DIST` (7.0). **Re-measure this for any track
with different cone spacing** — `probe`-style check: take the mixed-triangle
cross-colour edge lengths over the cone map and put the bound above p100.

### `path_utils`

- `_resample_forward` now returns `(samples, total)`. Callers need `total` to
  tell a real sample from one that only exists because it was clamped to the
  path's last point.
- `blend_paths` truncates the blended output to the *fresh* path's own
  validated arc length. Previously a short fresh path's clamped tail was
  averaged against a longer previous path's genuine further-out samples,
  pulling the blend past the point the freshly-validated path was willing to
  vouch for, into territory only the stale plan supported.
- `build_local_path` takes `max_pair_dist`, so the fallback's width policy can
  be aligned with the primary planner's gate band.

### `_seg_intersect`

Two genuine misses fixed. The old strict `0.0 < t < 1.0` test never counted a
crossing that lands exactly *on* a sampled waypoint — both adjacent segments
see it at their own `t=0`/`t=1` boundary, so a wall contact at a waypoint
escaped detection from both sides at once. And the `abs(denom) < 1e-10`
early-out discarded collinear overlap entirely (`denom` is 0 for any parallel
pair, including two segments on the same line). Now epsilon-tolerant, with
collinear overlap checked explicitly.

## Planner node (`centerline_planner.py`)

Upstream renamed its node to `wall_centerline_planner` and retired
`centerline_planner` to `legacy/`. This repo keeps the existing node name,
executable, `fsae_params.yaml` key and launch wiring — the hardening was
folded into `centerline_planner.py` in place, so nothing downstream changed.

Ported:

- **Input staleness gates** (`max_pose_age`, `max_track_age`,
  `max_track_skew`). Ages are measured from *local arrival time*, not message
  stamps: `Track.msg` carries no header at all, and `car_position`'s
  `PoseStamped` stamp is the upstream odom's own measurement time, which need
  not share a clock domain with this node — comparing it against
  `get_clock().now()` would report a constant offset as permanent staleness.
  The skew check exists because both tracks can individually pass
  `max_track_age` while having arrived a full `max_track_age` apart, which
  plans a centreline from a left/right pair that never coexisted.
- **Output sanitisation** (`_sanitize`). Finite coordinates, no oversized
  point-to-point jumps, initial direction agreeing with car heading within
  the same `_WALL_MAX_TURN_COS` allowance the walk itself uses. A wall
  crossing or a `min_cone_clearance` violation **truncates** at the first bad
  index rather than discarding the path — previously one bad segment out near
  the horizon blanked the good near-field part too. `min_usable_path_length`
  then rejects a stub outright.
- **Blend re-validation.** The blended path is sanitised again, not just the
  fresh one: two individually valid paths can blend into one that crosses a
  wall in a nonconvex corridor.
- **`_hold_reachable`.** Before republishing a held path, check the car hasn't
  drifted off it (`max_hold_deviation`) and that it doesn't cross the *current*
  wall mesh.
- **Cone input hardening.** `cones_to_array` drops non-finite cones and merges
  coincident ones — scipy's Delaunay treats coincident points as a degenerate
  simplex and can raise or return an unstable triangulation.
- **Parameter validation** at construction, and an unrecognised
  `midpoint_method` now errors and forces `'delaunay'` instead of silently
  falling through to the `'nn'` branch.
- **Whole-tick exception guard** and a slow-tick warning
  (`_SLOW_TICK_WARN_SEC`) — the tick runs synchronously inside the pose
  callback on a single-threaded executor, so a slow `build_path_walls` blocks
  every other callback on the node.
- **Waypoint orientations** on the published `PoseArray`, from the local path
  heading. `geometry_msgs/Quaternion` defaults to `(0,0,0,0)`, a zero-norm
  quaternion that RViz and tf reject. `stanley_controller` derives its own
  target yaw from consecutive points and is unaffected.
- The wall mesh is now rebuilt independently in `_compute_path`'s exception
  fallback rather than cleared. Clearing it silently disabled every collision
  check in `_sanitize` for exactly the path that had already gone wrong.

**Deliberately not ported** (real-stack-only, and they come as a set):

- The **explicit-empty-`PoseArray` fail-safe.** Upstream publishes an empty
  trajectory when it has no valid path because the real Stanley controller
  caches its last `tx`/`ty` indefinitely and only clears them on a fresh empty
  message — silence alone would not stop the car. This repo's
  `stanley_controller._control_step` already returns early on a short path, so
  the node stays silent here, as before.
- The **watchdog timer.** Its only job upstream is to force that empty publish
  when poses stop arriving entirely. It is useless without the empty publish;
  if that controller behaviour ever changes, port both together.
- `shadow_mode`, the RViz triangulation `MarkerArray`
  (`debug_triangulation_edges` itself is still in `boundary.py`, just
  unpublished), `cone_mapper_shim.py` and `car_position_bridge.py`.
- `skidpad_planner.py` needed no change — its only upstream difference is the
  `PoseStamped` → `Pose` subscription, which is a real-stack adaptation, not a
  fix.

## Known issue, not introduced here

`build_local_path` — the fallback `build_path_walls` uses when the midpoint
build yields nothing — is not clamped to `plan_horizon`, so it can return an
arbitrarily long path. It is bounded in practice by the node's
`_sanitize`/`max_point_jump` checks, but the clamp belongs in
`build_path_walls`. Present upstream too.

## Verification

- `colcon build --packages-select fsae_planning` clean.
- Offline sweep of `build_path_walls` over `tracks/comp_test_map_3`: behaviour
  is byte-identical to the `fsae_autonomous` source for the same inputs,
  confirming a faithful port.
- With `mid_max_gate: 6.5`, 996/1000 raceline poses produce a path over
  `min_usable_path_length`; the 4 that don't are isolated single samples,
  comfortably inside the 0.3 s `path_hold_timeout` bridge.
- Node-level run (real `CenterlinePlanner` spun against replayed `Track` +
  `PoseStamped` over 300 poses): published on 100% of ticks, 44–60 waypoints
  per message, no empty messages, no warnings or errors.
- **Not yet driven in the simulator itself.**
