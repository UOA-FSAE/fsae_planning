# Wall-barrier planner: false wall segments from mislabelled cones

Status: **superseded — all three real-stack changes are now in this repo.**
See [CHANGES.md](CHANGES.md) for the full sync, which also brought across the
later Delaunay *corridor* midpoint builder (`e533c55`) and the
path-disappearing patch (`8751593`) that this document predates.

This document is kept for the root-cause analysis below, which is still the
clearest write-up of *why* the wall builder was changed.

Two corrections to what this document originally said. First, the truncating
`_sanitize` no longer has "no equivalent to port" — `centerline_planner.py`
now has a full `_sanitize` of its own, along with input-staleness gating and
the rest of the node hardening from upstream's `wall_centerline_planner.py`,
folded into the existing node rather than added as a second one. Second, two
of upstream's node behaviours genuinely are real-stack-only and were **not**
ported: the explicit-empty-`PoseArray` fail-safe and the watchdog timer that
exists to trigger it. CHANGES.md explains why they come as a set.

## Symptom

On the real (narrower, physical) track, placing a single cone of the
*opposite* colour just outside a track boundary — e.g. a stray/mislabelled
yellow cone sitting just past the blue line — caused the published
trajectory (`/fsae/planning/selected_trajectory`) to go completely empty in
RViz. Moving that cone ~8-9 m away from the nearest same-colour cone made
the path come back. Moving it far down-track (but still geometrically close
to some other in-view cone) still blanked the *entire* path, not just the
part near the rogue cone — that last part didn't match the initial
hypothesis and needed a second root cause.

## Root cause 1 — false wall segments (`boundary.py`)

`build_wall_segments()` linked **every pair of same-coloured cones within
7 m (`_WALL_MAX_DIST`)** into a "wall" segment, with no concept of what
physically lies between them — purely a same-colour, within-distance
all-pairs graph. A stray/mislabelled cone of colour X sitting close to
colour X's cones *across* the track links straight into that wall mesh,
producing a wall segment that cuts laterally across the drivable corridor.

That phantom segment feeds two places:

- `_build_wall_path`'s greedy walk cost (`_WALL_CROSS_PENALTY = 100000` per
  crossing) — discourages, but does not forbid, stepping through it.
- `WallCenterlinePlanner._sanitize()` — a **hard, binary** reject: if any
  segment of the final published centreline crosses any wall segment, the
  whole path was rejected (see root cause 2).

The wall-cone window (`look_radius + 4`, omnidirectional radius OR forward
box, `build_path_walls`' old `blue_wall`/`yellow_wall`) bounded how far away
a contributing cone could be, but did nothing to stop the false link
*within* that window — and a track with a loop/hairpin/pinch can bring two
sections that are far apart *along* the track close together in a straight
line, so "far down the track" did not mean "outside the window."

## Root cause 2 — all-or-nothing sanitize (`wall_centerline_planner.py`)

```python
walls = self._blue_segs + self._yellow_segs
if walls and any(
    segment_crosses_walls(pts[i], pts[i + 1], walls)
    for i in range(len(pts) - 1)
):
    return False
```

This scanned every segment of the whole published spline and rejected the
**entire path** if even one segment anywhere — including right out at the
lookahead horizon — crossed a wall. There was no truncation: a single bad
tail segment discarded the good near-field path directly in front of the
car too. Once rejected every tick for longer than `path_hold_timeout`
(0.3 s default), `_planning_loop` clears `_centreline`/`_prev_centreline`
and publishes an explicit **empty** `PoseArray` (deliberate fail-safe so the
Stanley controller's cached target doesn't drive on a stale plan forever —
see that node's module docstring) — this is the "path disappears in RViz"
symptom.

## Fix — four changes

### 1. Same-colour wall segments now come from a Delaunay triangulation, not all-pairs

New `build_wall_segments_delaunay(blue, yellow, max_dist=_WALL_MAX_DIST)` in
`boundary.py`:

- Triangulate **every known cone of both colours together**
  (`scipy.spatial.Delaunay` — already a project dependency via
  `path_utils.py`'s `splprep`/`splev`).
- Collect all triangle edges, keep only edges where both endpoints are the
  same colour.
- `max_dist` (still `_WALL_MAX_DIST = 7.0`) is kept as a **safety net**, not
  the primary filter: it trims abnormally long edges a plain triangulation
  can produce at the convex hull / in sparse regions of a long, thin point
  cloud — exactly the shape of a track boundary.
- Falls back to the old per-colour `build_wall_segments()` if triangulation
  itself can't run (fewer than 3 total cones, or a near-degenerate/collinear
  point set — both raise `QhullError`) — publishing the old method's walls
  is preferable to publishing none that tick.

**Why this actually fixes it, not just narrows the odds:** Delaunay's
empty-circumcircle property means a direct edge between two same-coloured
cones on opposite sides of the track essentially can't survive if there's
any cone (either colour) physically between them — the circumcircle of that
long edge would have to avoid enclosing every intervening cone. A stray
cone next to the *other* colour's line ends up edged mostly to its
*nearest* neighbours (the opposite-colour cones next to it), which are
discarded by the same-colour filter, not edged across the track to a
same-coloured cone on the far side. This is a structural fix (this is the
same construction used in AMZ/KA-RaceIng-style FSD stacks for boundary
extraction), not a threshold tune — a tighter `_WALL_MAX_DIST` was
considered and rejected because it would just as readily break genuine,
sparser cone spacing without addressing the actual failure mode.

`build_wall_segments()` itself is **untouched** and still used as-is by
`special_utils/skidpad_planner.py` (different geometry — two tight
concentric rings, where a full triangulation would create unwanted interior
chords; out of scope for this pass per explicit instruction, to be
revisited separately if needed).

### 2. Wall building no longer uses a car-relative window

`build_path_walls` no longer computes `blue_wall`/`yellow_wall` via
`filter_cones_window(..., radius=look_radius + 4.0, ...)` before wall
building — it now passes the full, unfiltered `blue_cones`/`yellow_cones`
straight into `build_wall_segments_delaunay`. The triangulation is what
keeps a false link from forming; the window was never actually solving
that, only bounding its blast radius. Unbounded wall input is safe now that
the linking method itself is topology-aware.

**This does not touch the midpoint/centreline window** — `blue_fwd`/
`yellow_fwd` (still `filter_cones_window(..., radius=look_radius, ...)`) and
`_gen_midpoints` are unchanged, per the plan to leave the centreline method
itself intact.

Performance note: this was **not** benchmarked against real cone-count data
from `cone_mapper`/SLAM or measured on the Jetson. `scipy.spatial.Delaunay`
over a few hundred points is normally sub-millisecond, but this should be
confirmed against real `/fsae/slam/left_track` + `/fsae/slam/right_track`
sizes and the actual `car_position` tick rate before trusting it at speed —
per the "verify with real measurements" rule, `ros2 topic hz` under-reports
on this Jetson, so use rosbag2 message counts / timestamps if measuring.

### 3. `_sanitize()` truncates instead of invalidating the whole path

`WallCenterlinePlanner._sanitize()` now returns the path **truncated at the
first wall-crossing segment** (scanning from the car/index 0 outward)
instead of a bare `bool`. The finite/`max_point_jump`/initial-heading checks
are unchanged and still reject the whole path outright — those are
whole-path shape problems, not "the path becomes untrustworthy from some
point on." A crossing found early enough that truncation would leave fewer
than 2 points falls back to `None`, which is exactly the old full-reject
behaviour and flows into the existing hold-timeout/empty-trajectory logic
unchanged.

`_planning_loop`'s call site was updated to use the (possibly shorter)
returned path instead of branching on a bool.

### 4. `_gen_midpoints()` now rejects implausibly close opposite-colour pairs

Added `_WALL_MIN_MID_DIST = 1.5` (metres) in `boundary.py`. In `_gen_midpoints`,
an anchor cone's nearest opposite-colour candidate is now rejected (no
midpoint produced for that anchor this tick) if it is closer than
`min_dist`, in addition to the existing `max_dist` upper bound:

```python
dists      = np.linalg.norm(other[cand_idx] - a, axis=1)
best_local = int(np.argmin(dists))
if dists[best_local] > max_dist or dists[best_local] < min_dist:
    continue
```

**Why:** a genuine blue-yellow pair spans roughly the track width. A pair
much closer than that (well under any real track's minimum width) is far
more likely to be a real cone sitting next to a stray/mislabelled
opposite-colour cone than a real narrow section of track. Pairing it in
anyway pulls that midpoint — and the smoothed path through it — off toward
the stray cone instead of following the actual corridor. This is a
different failure mode from root causes 1-2 above: those produced a false
*wall* segment that could blank or truncate the path; this one produces a
*valid-looking but wrong* midpoint that drags the path off-line without
tripping any wall-crossing check at all, since no wall segment is
necessarily involved.

Unlike the upper-bound (`max_dist`) rejection, this does **not** fall
through to the anchor's next-nearest candidate — an anchor whose only
in-range candidate is implausibly close produces no midpoint that tick,
same as if no candidate were in range at all. A real narrow track losing a
midpoint at one gap is recovered by neighbouring midpoints and smoothing,
same as any other single dropped midpoint; guessing a second-best pairing
instead risks accepting a *different* wrong cone.

This does not catch every possible false pairing (e.g. a rogue cone spaced
at a plausible track-width distance from the wrong-side cone can still
pair in, if it also clears the existing left/right local-direction validity
check) — it specifically targets the "stray cone sitting right next to a
real one" case, which is the case a minimum-width sanity bound can actually
distinguish from a genuine gap.

## Testing performed this session

No physical track / rosbag access in this session — validated with
synthetic cone layouts against the actual `fsae_planning.boundary` and
`fsae_planning.wall_centerline_planner` modules (imported directly, not
mocked):

- **Near-boundary case** (the original bug report): straight 12-cone-pair
  track, rogue yellow cone placed 0.3 m outside a blue cone. Old
  `build_wall_segments` produces a wall segment a corridor-crossing probe
  segment intersects (reproduces the bug); new
  `build_wall_segments_delaunay` produces none that the same probe
  intersects.
- **Loop-pinch case** (the outlier from testing): rogue cone placed near
  the *start* of the track to simulate a track-distant-but-geometrically-
  close cone — new method still produces no corridor-crossing wall.
- **End-to-end**: `build_path_walls` with the rogue cone present still
  returns a non-empty, roughly-centred centreline (not dragged toward the
  rogue cone).
- **Degenerate inputs**: fewer than 3 total cones, and near-collinear cones
  (a fresh straight) — both fall back to the old per-colour method without
  raising.
- **`_sanitize` truncation**: constructed a `WallCenterlinePlanner` instance
  directly, injected a synthetic wall segment mid-path — confirmed it
  truncates to the pre-crossing points (not a full reject), passes a
  non-crossing path through unchanged, and still falls back to full-reject
  (`None`) when the crossing is close enough that truncation would leave
  under 2 points.
- **`_gen_midpoints` minimum pair distance**: a single blue/yellow pair
  0.4 m apart (car-relative, no other cones) produces no midpoint; the same
  pair at 3 m and at exactly 1.5 m both produce the expected midpoint —
  confirms the lower bound rejects only implausibly-close pairs and is
  inclusive at the threshold, not exclusive of normal spacing.

None of this replaces driving it — it only confirms the code does what the
design says it does on paper. **Still needs**: a real run reproducing both
original bug reports (near-boundary and far/loop-pinch placements), and a
skidpad run to confirm no regression there (`build_wall_segments` itself
wasn't touched, but confirm anyway since `_sanitize`'s truncation behaviour
change affects any planner path that flows through it).

## Files changed (real stack: `~/ros2_ws/src/fsae_autonomous/planning/fsae_planning/fsae_planning/`)

- `boundary.py`: added `build_wall_segments_delaunay()`; `build_path_walls`
  now calls it instead of `build_wall_segments()` and no longer builds
  `blue_wall`/`yellow_wall`; docstrings updated. `build_wall_segments()`
  and `_WALL_MAX_DIST` kept (still used by `skidpad_planner.py` and by
  `_gen_midpoints`'s `local_dir()`).
- `wall_centerline_planner.py`: `_sanitize()` returns a (possibly
  truncated) path or `None` instead of `bool`; `_planning_loop`'s call site
  updated to match.
- `boundary.py`: added `_WALL_MIN_MID_DIST = 1.5`; `_gen_midpoints` gained a
  `min_dist` parameter and now rejects an anchor's best candidate if it's
  closer than that, in addition to the existing `max_dist` rejection.

## Porting into this repo (simulator stack) — done, apart from the window removal

Two of the four real-stack changes have been ported onto this repo's
`boundary.py` (same root cause, same fix, same reasoning as the real-stack
section above):

- `build_wall_segments_delaunay()` — added verbatim (this repo has no scipy
  dependency issue; `path_utils.py` already uses `scipy.interpolate`).
  `build_path_walls` now calls it instead of `build_wall_segments()`.
- `_gen_midpoints`'s minimum pair distance (`_WALL_MIN_MID_DIST = 1.5`) —
  added verbatim. The stray-cone-next-to-a-real-one failure mode it targets
  is a property of the midpoint-matching logic, not of the real stack's
  SLAM-bounded perception, so it applies equally to the simulator's
  ground-truth cones.

The truncating `_sanitize` change was **not ported** — not by choice, but
because there is nothing here to port it onto. This repo's
`centerline_planner.py` has no post-`build_path_walls` sanitize/reject step
at all (see the status note at the top); `_compute_path` publishes whatever
`build_path_walls` returns as-is. If this repo later grows an equivalent
whole-path validity check, apply the same truncate-instead-of-reject
principle to it then.

**Did not port the wall-window removal (change 2 / "the lookout window
thing").** `build_path_walls` here still builds `blue_wall`/`yellow_wall` via
the `look_radius + 4` window before triangulating them, unlike the real
stack (which now triangulates the full unwindowed cone arrays). The working
theory is that `look_radius`'s wall-window role
exists here specifically because the simulator's perception is
synchronous ground truth with effectively unlimited range (unlike the real
stack's actual SLAM-bounded view), so an unwindowed wall build here could
behave very differently (and possibly much more expensively — the
simulator's cone map may have no equivalent of the real stack's "latest
frame only" cone_mapper boundedness). Keep this repo's existing wall-cone
windowing as-is unless a specific reason to change it turns up in testing.

### Files changed (this repo: `planning/fsae_planning/fsae_planning/`)

- `boundary.py`: added `from scipy.spatial import Delaunay` import,
  `build_wall_segments_delaunay()`, and `_WALL_MIN_MID_DIST = 1.5`.
  `build_path_walls` now calls `build_wall_segments_delaunay(blue_wall,
  yellow_wall)` (still windowed, unlike the real stack) instead of
  `build_wall_segments()` per colour. `_gen_midpoints` gained a `min_dist`
  parameter, defaulting to `_WALL_MIN_MID_DIST`, checked alongside the
  existing `max_dist`. `build_wall_segments()` and `_WALL_MAX_DIST` kept
  (still used as the degenerate-input fallback and by `_gen_midpoints`'s
  `local_dir()`). Docstrings updated to match.
- `centerline_planner.py`: unchanged.

Verified with `python3 -m py_compile` on both files, plus the same style of
synthetic-data sanity checks as the real-stack section above (near-boundary
false-link reproduction under the old method vs. none under the new one,
degenerate-input fallback, minimum-pair-distance rejection, and an
end-to-end `build_path_walls` call producing a sane non-empty centreline)
run directly against this repo's `fsae_planning.boundary` module. No
simulator run performed.
