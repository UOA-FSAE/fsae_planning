"""
Lap localisation: detect when the planned path closes on itself and switch to
following that full completed path.

In the default (mapping) state the planner only ever sees a forward-cropped
window of cones from the fake perception pipeline, so it plans a *local*
centreline a few midpoints at a time and the path only reaches as far as the
visible cones.  As the car drives, the centreline it follows is accumulated
into a running path.  Once that path loops back on itself — the car is back at
its starting position, re-seeing the cones it first saw there — the lap is
complete and the accumulated path *is* the full track centreline.  From that
point the planner localises against it and follows the whole closed loop
instead of the rolling window.

This module provides:

LoopClosureDetector
    Accumulates the driven path and fires once the three loop-closure
    requirements are met: (1) the car is back at its starting position, (2) it
    is re-detecting the old cones it first saw at the start, and (3) the
    accumulated path has closed on itself into a full loop.

build_completed_path
    Smooths the accumulated path into a seamless closed centreline.  Computed
    once on loop closure.

roll_loop_to_car / compute_drift
    Per-frame helpers for local mode: orient the cached loop so the segment
    ahead of the car comes first, and measure perception-vs-map drift.
"""
import math

import numpy as np
from scipy.interpolate import splev, splprep

from fsae_planning.path_utils import DEFAULT_SMOOTH_PER_PT

# --- Loop-closure tuning ---------------------------------------------------
DEPART_DIST   = 12.0  # m — car must leave this radius from start before a return counts
RETURN_RADIUS = 5.0   # m — back within this of the start = path has closed on itself
START_RADIUS  = 12.0  # m — perception cones this close to start define the start snapshot
MATCH_DIST    = 1.0   # m — a live cone this close to a start cone = re-detected
MIN_REDETECT  = 3     # min re-detected start cones to confirm old-cone sighting
TRAJ_MIN_STEP = 0.5   # m — min car movement before a new path point is recorded
MIN_TRAJ_PTS  = 20    # min accumulated path points before a closure can be declared


class LoopClosureDetector:
    """
    Detects completion of a full lap by tracking the path the car drives.

    The car's position is accumulated into a path polyline (one point every
    TRAJ_MIN_STEP metres).  Closure fires (update() returns True, latched) only
    when all three requirements hold on the same frame:

      1. at_start    — the car has departed the start (> DEPART_DIST) and has
                       returned within RETURN_RADIUS of it.
      2. old_cones   — at least MIN_REDETECT of the cones first seen at the
                       start are being re-detected by perception right now.
      3. path_closes — enough path has been accumulated (>= MIN_TRAJ_PTS) that
                       returning to the start closes the path into a loop.
    """

    def __init__(self) -> None:
        self._start_pos: np.ndarray | None = None
        self._start_cones: np.ndarray = np.empty((0, 2))
        self._traj: list[np.ndarray] = []
        self._departed = False
        self._closed = False

    @property
    def start_pos(self) -> np.ndarray | None:
        return self._start_pos

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def trajectory(self) -> np.ndarray:
        """The accumulated driven path as an (N, 2) array."""
        return np.array(self._traj, dtype=np.float64) if self._traj else np.empty((0, 2))

    def reopen(self) -> None:
        """Clear the latched closure so detection retries on the next lap pass."""
        self._closed = False

    def update(
        self,
        car_pos: np.ndarray,
        perc_blue: np.ndarray,
        perc_yellow: np.ndarray,
    ) -> bool:
        """Feed one frame; return True once (and forever after) the path loops."""
        if self._closed:
            return True

        car_pos = np.asarray(car_pos, dtype=np.float64)

        # First frame after GO: latch the starting pose and seed the path.
        if self._start_pos is None:
            self._start_pos = car_pos.copy()
            self._traj.append(car_pos.copy())

        # Snapshot the cones around the start line as soon as they are visible.
        if len(self._start_cones) == 0:
            self._start_cones = _cones_near(
                _stack(perc_blue, perc_yellow), self._start_pos, START_RADIUS
            )

        # Accumulate the driven path, one point every TRAJ_MIN_STEP metres.
        if float(np.linalg.norm(car_pos - self._traj[-1])) > TRAJ_MIN_STEP:
            self._traj.append(car_pos.copy())

        dist = float(np.linalg.norm(car_pos - self._start_pos))
        if not self._departed:
            if dist > DEPART_DIST:
                self._departed = True
            return False

        at_start    = dist < RETURN_RADIUS
        path_closes = len(self._traj) >= MIN_TRAJ_PTS
        old_cones   = (
            _count_redetected(_stack(perc_blue, perc_yellow), self._start_cones)
            >= MIN_REDETECT
        )

        if at_start and path_closes and old_cones:
            self._closed = True
        return self._closed


# ---------------------------------------------------------------------------
# Completed-path construction
# ---------------------------------------------------------------------------

def build_completed_path(
    trajectory: np.ndarray,
    n_out_per_pt: int = 3,
    smooth_per_pt: float = DEFAULT_SMOOTH_PER_PT,
) -> np.ndarray | None:
    """
    Smooth the accumulated driven path into a seamless closed centreline.

    The path the car drove is already in track order.  Before fitting, the
    lead-in is trimmed so the loop closes on the path *itself* (end → nearest
    earlier point) rather than on the arbitrary start pose (end → start); the
    latter stitches two non-coincident points together and leaves a kink at the
    start line.  The trimmed path is then fitted with a *periodic* cubic spline
    to remove odometry jitter and close the seam seamlessly.

    Returns an (N, 2) loop (last point == first point) or None if the path is
    too short to form a loop.
    """
    traj = np.asarray(trajectory, dtype=np.float64)
    if len(traj) < MIN_TRAJ_PTS:
        return None
    traj = _close_on_self(traj)
    return _smooth_loop(traj, n_out=max(200, len(traj) * n_out_per_pt),
                        smooth_per_pt=smooth_per_pt)


def roll_loop_to_car(
    loop: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    ahead: float = 35.0,
    wall_segs: list[tuple[np.ndarray, np.ndarray]] | None = None,
    tangent_entry: bool = False,
) -> np.ndarray:
    """
    Reorder a closed loop so the segment ahead of the car comes first.

    Finds the loop point nearest the car, rolls the loop to start there, orients
    it in the car's heading direction, and wraps `ahead` metres of the loop tail
    back onto the end so downstream lookahead / speed scans never run off the
    array at the wrap seam.  car_pos is prepended as the near anchor (matching
    the convention used by build_local_path / build_path_walls).

    When `wall_segs` (same-colour cone-wall segments, see boundary.build_wall_
    segments) is given, the entry point is the nearest loop point whose straight
    connection from the car crosses no wall.  This stops the car latching onto a
    loop point on the far side of a cone wall — e.g. on a skidpad it must reach
    the figure-8 through the opening rather than cutting across the cone rings.

    With `tangent_entry`, the entry point is not the *nearest* point but the
    reachable point ahead whose loop tangent is most aligned with the approach
    direction.  The car therefore merges onto the circle along a tangent — a
    smooth join into the turn — instead of driving at the closest point and
    cornering hard onto it.  This is a mode the caller turns on only while the
    car is still approaching (see the skidpad planner, which drops it once the
    car reaches the crossing); it should be off during normal loop following, or
    it would keep steering the car back toward the tangent target.
    """
    pts = np.asarray(loop, dtype=np.float64)
    n = len(pts)
    if n < 3:
        return pts.copy()

    # Drop the duplicate closing point so rolling does not repeat it.
    if float(np.linalg.norm(pts[0] - pts[-1])) < 1e-6:
        pts = pts[:-1]
        n -= 1

    car = np.asarray(car_pos, dtype=np.float64)
    heading = np.array([math.cos(car_yaw), math.sin(car_yaw)])
    rel = pts - car
    dist = np.linalg.norm(rel, axis=1)
    order = np.argsort(dist)

    reachable = None
    idx = int(order[0])
    if wall_segs:
        from fsae_planning.boundary import segment_crosses_walls
        reachable = np.array(
            [not segment_crosses_walls(car, pts[i], wall_segs) for i in range(n)]
        )
        for cand in order:                       # nearest reachable point (via opening)
            if reachable[cand]:
                idx = int(cand)
                break

    # Tangent entry: join the circle where the approach direction is tangent to
    # it (smooth merge) rather than at the nearest point.  The caller enables
    # this only while approaching; near the skidpad crossing a circle arc passes
    # within a lane-width of the entry lane, so a distance test cannot tell an
    # approaching car from a following one — the mode is owned by the planner.
    if tangent_entry:
        with np.errstate(invalid='ignore'):
            direction = rel / dist[:, None]
        # Loop tangent (central difference), oriented toward the car's heading.
        tang = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
        tang /= np.linalg.norm(tang, axis=1, keepdims=True) + 1e-12
        tang *= np.sign(tang @ heading)[:, None]
        align = np.einsum('ij,ij->i', direction, tang)     # 1 = tangent, 0 = radial
        eligible = (rel @ heading) > 0.0                    # ahead of the car
        if reachable is not None:
            eligible &= reachable
        if np.any(eligible):
            align = np.where(eligible, align, -np.inf)
            idx = int(np.argmax(align))

    rolled = np.vstack([pts[idx:], pts[:idx]])

    # Orient in travel direction: if the next point is behind the car relative
    # to its heading, the loop is wound the wrong way — reverse it.
    if float(np.dot(rolled[1] - rolled[0], heading)) < 0.0:
        rolled = np.vstack([rolled[:1], rolled[1:][::-1]])

    # Wrap `ahead` metres of the loop back onto the tail for seamless lookahead.
    seg = np.linalg.norm(np.diff(rolled, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    m = int(np.searchsorted(cum, ahead)) + 1
    tail = rolled[: min(m, n)]

    return np.vstack([np.asarray(car_pos, dtype=np.float64).reshape(1, 2), rolled, tail])


def compute_drift(
    perc_blue: np.ndarray,
    perc_yellow: np.ndarray,
    map_blue: np.ndarray,
    map_yellow: np.ndarray,
) -> dict:
    """
    Measure how far the live perception cones sit from the accumulated map.

    For every live cone, the distance to its nearest map cone of the same colour
    is taken; a large mean/max signals odometry drift (the live frame no longer
    overlays the map it was built from).  Returns {'mean', 'max', 'n'} in metres.
    """
    dists: list[float] = []
    for perc, mp in ((perc_blue, map_blue), (perc_yellow, map_yellow)):
        if len(perc) == 0 or len(mp) == 0:
            continue
        for p in perc:
            dists.append(float(np.min(np.linalg.norm(mp - p, axis=1))))

    if not dists:
        return {'mean': 0.0, 'max': 0.0, 'n': 0}
    arr = np.asarray(dists)
    return {'mean': float(arr.mean()), 'max': float(arr.max()), 'n': int(arr.size)}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _stack(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vertically stack two (N, 2) arrays, tolerating empties."""
    parts = [p for p in (a, b) if len(p) > 0]
    return np.vstack(parts) if parts else np.empty((0, 2))


def _cones_near(cones: np.ndarray, centre: np.ndarray, radius: float) -> np.ndarray:
    if len(cones) == 0:
        return np.empty((0, 2))
    return cones[np.linalg.norm(cones - centre, axis=1) < radius]


def _count_redetected(live: np.ndarray, reference: np.ndarray) -> int:
    """Count live cones lying within MATCH_DIST of any reference (start) cone."""
    if len(live) == 0 or len(reference) == 0:
        return 0
    return int(sum(
        np.min(np.linalg.norm(reference - p, axis=1)) < MATCH_DIST for p in live
    ))


def _close_on_self(traj: np.ndarray) -> np.ndarray:
    """
    Trim the lead-in so the loop closes on the driven path itself.

    The trajectory starts at the GO pose — which may sit off the racing line and
    include a short lead-in onto it — and ends where the car returned, within
    RETURN_RADIUS of the start.  Joining end → start (two non-coincident points
    on different parts of the track) leaves a kink the car reads as a left/right
    sway at the start line.  Instead, find the earliest trajectory point closest
    to the end point and start the loop there: the closing seam then joins two
    near-coincident points already on the driven line, so the loop closes
    smoothly.  The dropped lead-in is redundant — the loop already covers that
    region near the seam.
    """
    n = len(traj)
    end = traj[-1]
    # Search only the first half so the match is the lead-in near the start,
    # never a point adjacent to the end.
    search_n = max(1, n // 2)
    j = int(np.argmin(np.linalg.norm(traj[:search_n] - end, axis=1)))
    return traj if j == 0 else traj[j:]


def _smooth_loop(pts: np.ndarray, n_out: int,
                 smooth_per_pt: float = DEFAULT_SMOOTH_PER_PT) -> np.ndarray | None:
    """
    Fit a periodic cubic *approximating* spline through the points → seamless
    closed loop.

    smooth_per_pt scales splprep's s (s = smooth_per_pt * n_points).  With the
    old s=0.0 the spline interpolated every driven-path point, freezing the
    mapping-lap odometry jitter (and every wobble the car drove) permanently into
    the raceline; a positive value actually removes it, as the docstring intends.
    """
    pts = np.asarray(pts, dtype=np.float64)
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], gaps > 1e-4])]
    if len(pts) < 4:
        return None

    # Periodic splprep expects the sample sequence to wrap, so close it.
    if float(np.linalg.norm(pts[0] - pts[-1])) > 1e-6:
        pts = np.vstack([pts, pts[0]])

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smooth_per_pt * len(pts),
                         per=1, k=3)
    except Exception:
        return pts
    u = np.linspace(0.0, 1.0, n_out)
    x, y = splev(u, tck)
    return np.column_stack([x, y])
