import math

import numpy as np
from scipy.interpolate import splev, splprep

from fs_msgs.msg import Cone

# Typical FS track width upper bound used to reject implausible pairs
MAX_PAIR_DIST = 7.0  # metres


def separate_cones_by_color(track_msg):
    """
    Split a Track message into blue (left boundary) and yellow (right boundary)
    cone arrays, discarding orange and unknown cones.
    Returns (blue_cones, yellow_cones) as float64 arrays of shape (N, 2).
    """
    blue, yellow = [], []
    for cone in track_msg.track:
        pt = [cone.location.x, cone.location.y]
        if cone.color == Cone.BLUE:
            blue.append(pt)
        elif cone.color == Cone.YELLOW:
            yellow.append(pt)
    return np.array(blue, dtype=np.float64), np.array(yellow, dtype=np.float64)


def sort_cones_nn(cones, start=None):
    """
    Order a (N, 2) cone array into track sequence using a greedy nearest-neighbour
    walk starting from the cone closest to `start` (default: map origin = car start).
    Returns a reordered (N, 2) array.
    """
    if len(cones) == 0:
        return cones.copy()

    if start is None:
        start = np.zeros(2)

    remaining = list(range(len(cones)))
    seed = int(np.argmin(np.linalg.norm(cones - start, axis=1)))
    remaining.remove(seed)
    ordered = [seed]

    while remaining:
        last = cones[ordered[-1]]
        dists = np.linalg.norm(cones[remaining] - last, axis=1)
        nearest = remaining[int(np.argmin(dists))]
        ordered.append(nearest)
        remaining.remove(nearest)

    return cones[ordered]


def pair_cones_nn(left_cones, right_cones, max_dist=MAX_PAIR_DIST):
    """
    Match each left cone (sorted track order) to its nearest unpaired right cone
    within max_dist metres.
    Returns a list of (left_pt, right_pt) pairs as 1-D float64 arrays.
    """
    if len(left_cones) == 0 or len(right_cones) == 0:
        return []

    right_remaining = list(range(len(right_cones)))
    pairs = []

    for lc in left_cones:
        if not right_remaining:
            break
        candidates = right_cones[right_remaining]
        dists = np.linalg.norm(candidates - lc, axis=1)
        best_local = int(np.argmin(dists))
        if dists[best_local] <= max_dist:
            pairs.append((lc, right_cones[right_remaining[best_local]]))
            right_remaining.pop(best_local)

    return pairs


def compute_centreline(pairs):
    """
    Compute midpoints of (left, right) cone pairs.
    Returns (N, 2) float64 array.
    """
    if not pairs:
        return np.empty((0, 2), dtype=np.float64)
    return np.array([(l + r) * 0.5 for l, r in pairs], dtype=np.float64)


def _remove_reversals(pts: np.ndarray, min_cos: float = -0.5,
                      max_removals: int = 3) -> np.ndarray:
    """
    Remove midpoints that cause sharp direction reversals (dot < min_cos ≈ 120°).
    Capped at max_removals to prevent cascading elimination of a legitimate corner.
    """
    for _ in range(max_removals):
        if len(pts) < 3:
            break
        segs = np.diff(pts, axis=0)
        norms = np.linalg.norm(segs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        d = segs / norms
        dots = np.einsum('ij,ij->i', d[:-1], d[1:])
        worst_local = int(np.argmin(dots))
        if dots[worst_local] >= min_cos:
            break
        pts = np.delete(pts, worst_local + 1, axis=0)
    return pts


def smooth_centreline(waypoints, n_out=None, smooth=0.0):
    """
    Fit a parametric cubic spline through cone-pair midpoints and resample.

    Pipeline:
      1. Drop duplicate consecutive points.
      2. Remove midpoints that cause direction reversals > 120° — these
         come from bad pairings and would otherwise force the spline into
         a U-turn.
      3. Fit a cubic spline using chord-length parameterisation (arc-length
         as the knot parameter).  This prevents backwards tangents at
         unevenly-spaced midpoints, which is the other common U-turn source.
      4. Resample at n_out uniform parameter values.

    smooth=0 gives an interpolating spline (passes exactly through every
    surviving midpoint).
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    if len(pts) < 2:
        return pts.copy()

    # 1. Drop duplicates
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    pts = pts[np.concatenate([[True], gaps > 1e-4])]

    # 2. Remove U-turn midpoints
    pts = _remove_reversals(pts)

    n = len(pts)
    if n_out is None:
        n_out = n * 8

    if n < 4:
        u_in  = np.linspace(0.0, 1.0, n)
        u_out = np.linspace(0.0, 1.0, n_out)
        return np.column_stack([
            np.interp(u_out, u_in, pts[:, 0]),
            np.interp(u_out, u_in, pts[:, 1]),
        ])

    # 3. Chord-length parameterisation — arc length as knot parameter
    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    if arc[-1] < 1e-6:
        return pts.copy()
    u_knots = arc / arc[-1]

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], u=u_knots, s=smooth, k=3)
        u_out = np.linspace(0.0, 1.0, n_out)
        x_new, y_new = splev(u_out, tck)
        return np.column_stack([x_new, y_new])
    except Exception:
        u_out = np.linspace(0.0, 1.0, n_out)
        return np.column_stack([
            np.interp(u_out, u_knots, pts[:, 0]),
            np.interp(u_out, u_knots, pts[:, 1]),
        ])


def filter_cones_forward(cones, car_pos, car_yaw,
                          min_ahead=0.5, max_ahead=25.0, max_lateral=6.0):
    """Return cones within the car's forward window."""
    if len(cones) == 0:
        return cones.copy()
    cos_y = math.cos(car_yaw)
    sin_y = math.sin(car_yaw)
    rel = cones - car_pos
    x_car =  rel[:, 0] * cos_y + rel[:, 1] * sin_y
    y_car = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
    mask = (x_car > min_ahead) & (x_car < max_ahead) & (np.abs(y_car) < max_lateral)
    return cones[mask]


def build_local_path(blue_cones, yellow_cones, car_pos, car_yaw,
                     max_ahead=25.0, max_lateral=10.0):
    """
    Build a smooth local centreline each control tick.

    Only ahead cones are paired so the NN sort is monotonically forward and
    cannot zigzag back across the car.  car_pos is prepended as the near
    anchor so the spline starts at the car's current position.

    Returns a smoothed (N, 2) array or None if there are not enough cones.
    """
    blue_fwd   = filter_cones_forward(blue_cones,   car_pos, car_yaw,
                                       min_ahead=0.5,
                                       max_ahead=max_ahead, max_lateral=max_lateral)
    yellow_fwd = filter_cones_forward(yellow_cones, car_pos, car_yaw,
                                       min_ahead=0.5,
                                       max_ahead=max_ahead, max_lateral=max_lateral)

    if len(blue_fwd) < 1 or len(yellow_fwd) < 1:
        return None

    blue_sorted   = sort_cones_nn(blue_fwd,   start=car_pos)
    yellow_sorted = sort_cones_nn(yellow_fwd, start=car_pos)
    pairs = pair_cones_nn(blue_sorted, yellow_sorted)

    if not pairs:
        return None

    raw = compute_centreline(pairs)
    # Prepend car_pos so the spline is anchored at the car's current position
    # without relying on behind-cones (which caused zigzag sort ordering).
    anchored = np.vstack([car_pos.reshape(1, 2), raw])
    return smooth_centreline(anchored, n_out=max(20, len(raw) * 5))


# ---------------------------------------------------------------------------
# ft-fsd-inspired trace-sort path planner
# ---------------------------------------------------------------------------

# Same-colour adjacency graph
_TS_K_NEIGHBOURS  = 5      # max k-NN per cone (same colour only)
_TS_MAX_EDGE_M    = 6.5    # metres — max edge length in the graph
_TS_MAX_WALK      = 14     # max cones to chain per boundary

# Ellipse-gate matching (ft-fsd §4.2 / cone_matching)
_TS_ELLIPSE_MAJOR = 7.5    # metres — major axis along inward direction
_TS_ELLIPSE_MINOR = 3.0    # metres — minor axis ⊥ to inward (≈ min track width)


def _build_same_color_adj(cones: np.ndarray) -> list[list[int]]:
    """
    Build a k-NN same-colour adjacency list restricted to _TS_MAX_EDGE_M.

    Only same-colour cones are in the graph so opposite-colour cones
    (and adjacent-track cones of the same colour that are too far away)
    can never be reached through graph edges.
    """
    n = len(cones)
    adj: list[list[int]] = [[] for _ in range(n)]
    if n < 2:
        return adj
    diff  = cones[:, None, :] - cones[None, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    for i in range(n):
        within = np.where(dists[i] <= _TS_MAX_EDGE_M)[0]
        if len(within):
            adj[i] = within[np.argsort(dists[i, within])][:_TS_K_NEIGHBOURS].tolist()
    return adj


def _local_tangent(wall: np.ndarray, idx: int) -> np.ndarray:
    """Unit tangent at wall[idx] via chord between its immediate neighbours."""
    n = len(wall)
    if n < 2:
        return np.array([1.0, 0.0])
    if idx == 0:
        t = wall[1] - wall[0]
    elif idx == n - 1:
        t = wall[-1] - wall[-2]
    else:
        t = wall[idx + 1] - wall[idx - 1]
    length = float(np.linalg.norm(t))
    return t / length if length > 1e-6 else np.array([1.0, 0.0])


def _sort_boundary(
    cones: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    opposite_cones: np.ndarray,
    is_left: bool,
) -> np.ndarray:
    """
    Order same-colour boundary cones with a greedy walk on the same-colour
    adjacency graph, using a two-term step cost:

        angle_cost  — heading change from current direction (prefer straight)
        cross_cost  — fraction of nearby opposite-colour cones on the wrong
                      lateral side within 6 m (cross-track guard)

    FSDS ENU convention: x = forward, y = left.
    For the left (blue) wall, yellow cones should appear to the RIGHT of the
    travel direction.  When the walk drifts toward an adjacent track's blue
    wall, this invariant breaks — the yellow cones of our track begin
    appearing to the LEFT — and cross_cost rises to suppress that candidate.
    """
    n = len(cones)
    if n == 0:
        return cones.copy()

    adj     = _build_same_color_adj(cones)
    cos_y   = math.cos(car_yaw)
    sin_y   = math.sin(car_yaw)

    def x_fwd(pt: np.ndarray) -> float:
        rel = pt - car_pos
        return float(rel[0] * cos_y + rel[1] * sin_y)

    # Seed: nearest forward cone to car_pos
    fwd    = np.array([x_fwd(cones[i]) for i in range(n)])
    pool   = np.where(fwd > 0.5)[0]
    if not len(pool):
        pool = np.arange(n)
    seed   = int(pool[np.argmin(np.linalg.norm(cones[pool] - car_pos, axis=1))])

    ordered = [seed]
    visited = {seed}
    d0      = cones[seed] - car_pos
    cur_dir = d0 / (np.linalg.norm(d0) + 1e-9)

    for _ in range(_TS_MAX_WALK - 1):
        current    = ordered[-1]
        candidates = [nb for nb in adj[current] if nb not in visited]
        if not candidates:
            break

        best_nb, best_score = None, math.inf
        for nb in candidates:
            step     = cones[nb] - cones[current]
            step_len = float(np.linalg.norm(step))
            if step_len < 1e-6:
                continue
            step_dir = step / step_len

            # Reject hard reversal (> ~107° from current heading)
            if float(np.dot(step_dir, cur_dir)) < -0.3:
                continue

            angle_cost = math.acos(float(np.clip(np.dot(cur_dir, step_dir), -1.0, 1.0)))

            # Cross-track guard
            # right_dir = 90° CW of step_dir; lat > 0 ↔ cone is to the right
            cross_cost = 0.0
            if len(opposite_cones):
                right_dir = np.array([step_dir[1], -step_dir[0]])
                rel_opp   = opposite_cones - cones[nb]
                near      = rel_opp[np.linalg.norm(rel_opp, axis=1) < 6.0]
                if len(near):
                    lat   = np.dot(near, right_dir)
                    # left wall (blue): yellow should be RIGHT (lat > 0); wrong if lat < 0
                    # right wall (yellow): blue should be LEFT (lat < 0); wrong if lat > 0
                    wrong = int(np.sum(lat < 0)) if is_left else int(np.sum(lat > 0))
                    cross_cost = wrong / len(near)

            score = angle_cost + 2.0 * cross_cost
            if score < best_score:
                best_score = score
                best_nb    = nb

        if best_nb is None:
            break

        step    = cones[best_nb] - cones[ordered[-1]]
        cur_dir = step / (np.linalg.norm(step) + 1e-9)
        ordered.append(best_nb)
        visited.add(best_nb)

    return cones[ordered]


def _match_cones_ellipse(
    left_wall: np.ndarray,
    right_wall: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Match ordered left (blue) to right (yellow) cones with an oriented
    ellipse gate and a strict monotonicity constraint.

    For each left cone at index li:
      inward direction  = rightward perpendicular to local tangent
      ellipse major axis = _TS_ELLIPSE_MAJOR m along inward direction
      ellipse minor axis = _TS_ELLIPSE_MINOR m perpendicular (along track)

    Only right-wall cones in the inward half-space (along > 0) AND inside
    the ellipse are candidates; the closest wins.

    Monotonicity: the matched right-cone index must not decrease, so the
    same right-wall cone cannot be re-matched after the walk has moved past
    it, and no backward cross-track jump is possible.

    The narrow minor axis (≈ minimum track width) is the primary cross-track
    guard: a cone from a parallel adjacent track that is laterally offset by
    more than _TS_ELLIPSE_MINOR metres from the inward search direction fails
    the gate automatically.
    """
    if not len(left_wall) or not len(right_wall):
        return []

    n_right = len(right_wall)
    pairs   = []
    last_ri = 0

    for li in range(len(left_wall)):
        lc      = left_wall[li]
        tang    = _local_tangent(left_wall, li)
        # inward = right of travel direction (toward yellow boundary)
        inward  = np.array([ tang[1], -tang[0]])
        # perpendicular to inward = along-track direction
        perp_in = np.array([-inward[1], inward[0]])

        best_dist, best_ri = math.inf, None

        for ri in range(last_ri, n_right):
            rel    = right_wall[ri] - lc
            along  = float(np.dot(rel, inward))
            if along <= 0.0:
                continue                            # not in the inward half-space
            across = float(np.dot(rel, perp_in))
            if (along  / _TS_ELLIPSE_MAJOR) ** 2 + \
               (across / _TS_ELLIPSE_MINOR) ** 2 > 1.0:
                continue                            # outside ellipse

            dist = float(np.linalg.norm(rel))
            if dist < best_dist:
                best_dist = dist
                best_ri   = ri

        if best_ri is not None:
            pairs.append((lc.copy(), right_wall[best_ri].copy()))
            last_ri = best_ri   # monotonicity

    return pairs


def build_path_trace(
    blue_cones: np.ndarray,
    yellow_cones: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    max_ahead: float = 25.0,
    max_lateral: float = 10.0,
) -> np.ndarray | None:
    """
    Build a centreline using the ft-fsd trace-sort approach.

    Pipeline
    --------
    1. Forward-filter cones to the planning window.
    2. Sort each boundary using a same-colour-only adjacency graph with an
       integrated cross-track guard.  The graph ensures the walk can never
       jump to an opposite-colour cone; the guard penalises paths where
       opposite-colour cones appear on the wrong lateral side.
    3. Match left↔right with an oriented ellipse gate (major axis = inward
       direction at _TS_ELLIPSE_MAJOR m; minor axis = _TS_ELLIPSE_MINOR m
       ≈ minimum track width) plus monotonicity.  The narrow minor axis
       rejects cones from parallel adjacent tracks that lie outside the
       cross-track search corridor.
    4. Midpoints of matched pairs → cubic spline centreline.

    Falls back to the simple NN pairing via build_local_path() if either
    boundary cannot be sorted or no ellipse matches are found.
    """
    blue_fwd = filter_cones_forward(
        blue_cones,   car_pos, car_yaw,
        min_ahead=0.5, max_ahead=max_ahead, max_lateral=max_lateral,
    )
    yellow_fwd = filter_cones_forward(
        yellow_cones, car_pos, car_yaw,
        min_ahead=0.5, max_ahead=max_ahead, max_lateral=max_lateral,
    )

    if len(blue_fwd) < 1 or len(yellow_fwd) < 1:
        return build_local_path(
            blue_cones, yellow_cones, car_pos, car_yaw, max_ahead, max_lateral
        )

    blue_sorted   = _sort_boundary(blue_fwd,   car_pos, car_yaw, yellow_fwd, is_left=True)
    yellow_sorted = _sort_boundary(yellow_fwd, car_pos, car_yaw, blue_fwd,   is_left=False)

    if not len(blue_sorted) or not len(yellow_sorted):
        return build_local_path(
            blue_cones, yellow_cones, car_pos, car_yaw, max_ahead, max_lateral
        )

    pairs = _match_cones_ellipse(blue_sorted, yellow_sorted)

    if not pairs:
        # Ellipse gate found no matches; fall back to simple NN pairing on
        # the adjacency-sorted walls (still better than unsorted NN pairing)
        pairs = pair_cones_nn(blue_sorted, yellow_sorted)

    if not pairs:
        return build_local_path(
            blue_cones, yellow_cones, car_pos, car_yaw, max_ahead, max_lateral
        )

    raw      = compute_centreline(pairs)
    anchored = np.vstack([car_pos.reshape(1, 2), raw])
    return smooth_centreline(anchored, n_out=max(20, len(raw) * 5))


# ---------------------------------------------------------------------------
# Cone-wall barrier path planner
# ---------------------------------------------------------------------------

_WALL_MAX_DIST      = 7.0    # metres — max dist to link same-colour cones into wall
_WALL_MID_DIST      = 4.0   # metres — max blue-yellow dist for midpoint candidates
_WALL_CROSS_PENALTY = 5000.0  # cost added per wall segment crossed by a path step
_WALL_PATH_MAX_STEP = 10.0   # metres — max step between consecutive path midpoints
_WALL_PATH_MAX_WALK = 18     # max midpoints in the constructed path


def _build_wall_segments(
    cones: np.ndarray,
    max_dist: float = _WALL_MAX_DIST,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (p1, p2) segments connecting every same-colour cone pair within max_dist."""
    n = len(cones)
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(cones[i] - cones[j])) <= max_dist:
                segs.append((cones[i], cones[j]))
    return segs


def _seg_intersect(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
) -> bool:
    """True if segment a1→a2 properly intersects segment b1→b2 (endpoints excluded)."""
    d1 = a2 - a1
    d2 = b2 - b1
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-10:
        return False
    diff = b1 - a1
    t = float(diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = float(diff[0] * d1[1] - diff[1] * d1[0]) / denom
    return 0.0 < t < 1.0 and 0.0 < u < 1.0


def _gen_midpoints(
    blue: np.ndarray,
    yellow: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    max_dist: float = _WALL_MID_DIST,
) -> np.ndarray:
    """
    Return midpoints of valid blue-yellow pairs within max_dist metres.

    Validity filter: the blue cone must be laterally to the LEFT of the yellow
    cone in the car's current frame (lat_blue > lat_yellow).  This eliminates
    midpoints that land inside a boundary wall — they arise when a same-colour
    cone from an adjacent parallel track forms a pair with a cone on the wrong
    side.
    """
    if len(blue) == 0 or len(yellow) == 0:
        return np.empty((0, 2), dtype=np.float64)

    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)

    def lat(pt: np.ndarray) -> float:
        rel = pt - car_pos
        return float(-rel[0] * sin_y + rel[1] * cos_y)  # positive = left of car

    mids = []
    for b in blue:
        b_lat = lat(b)
        for y in yellow:
            if lat(y) >= b_lat:                          # yellow must be right of blue
                continue
            if float(np.linalg.norm(b - y)) <= max_dist:
                mids.append((b + y) * 0.5)

    return np.array(mids, dtype=np.float64) if mids else np.empty((0, 2), dtype=np.float64)


def _build_wall_path(
    midpoints: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    wall_segs: list[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """
    Sort midpoints by forward distance from the car, then chain them greedily.

    Step cost:
        distance  +  _WALL_CROSS_PENALTY × crossings  +  2.0 × heading_change(rad)

    Sorting by forward distance before the walk enforces monotonic forward
    progression (no backward zigzag).  The heading-change term prefers steps
    that continue the current travel direction, smoothing the raw chain before
    spline fitting.  The wall-crossing penalty blocks jumps to adjacent tracks.
    """
    n = len(midpoints)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)

    def x_fwd(pt: np.ndarray) -> float:
        rel = pt - car_pos
        return float(rel[0] * cos_y + rel[1] * sin_y)

    # Sort all midpoints by forward distance
    fwd = np.array([x_fwd(m) for m in midpoints])
    sort_idx = np.argsort(fwd)
    midpoints = midpoints[sort_idx]
    fwd = fwd[sort_idx]

    # Seed: first midpoint that is forward of the car
    forward_start = int(np.searchsorted(fwd, 0.3))
    if forward_start >= n:
        return np.empty((0, 2), dtype=np.float64)

    pool_fwd = midpoints[forward_start:]
    seed = forward_start + int(np.argmin(np.linalg.norm(pool_fwd - car_pos, axis=1)))

    ordered  = [seed]
    cur_dir  = np.array([cos_y, sin_y], dtype=np.float64)  # initial = car heading

    for _ in range(_WALL_PATH_MAX_WALK - 1):
        curr_idx = ordered[-1]
        curr     = midpoints[curr_idx]
        best_nb, best_cost = None, math.inf

        # Only search indices AHEAD in the sorted-by-fwd array (monotone forward)
        for idx in range(curr_idx + 1, n):
            cand = midpoints[idx]
            d = float(np.linalg.norm(cand - curr))
            if d > _WALL_PATH_MAX_STEP:
                continue
            step_dir = (cand - curr) / (d + 1e-9)
            angle    = math.acos(float(np.clip(np.dot(cur_dir, step_dir), -1.0, 1.0)))
            n_cross  = sum(1 for (w1, w2) in wall_segs if _seg_intersect(curr, cand, w1, w2))
            cost     = d + _WALL_CROSS_PENALTY * n_cross + 2.0 * angle
            if cost < best_cost:
                best_cost = cost
                best_nb   = idx

        if best_nb is None:
            break
        step    = midpoints[best_nb] - curr
        cur_dir = step / (float(np.linalg.norm(step)) + 1e-9)
        ordered.append(best_nb)

    return midpoints[ordered]


def build_path_walls(
    blue_cones: np.ndarray,
    yellow_cones: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    max_ahead: float = 25.0,
    max_lateral: float = 10.0,
) -> tuple[np.ndarray | None,
           list[tuple[np.ndarray, np.ndarray]],
           list[tuple[np.ndarray, np.ndarray]],
           np.ndarray]:
    """
    Build a centreline using cone-wall segments as a path barrier.

    Same-colour cones within _WALL_MAX_DIST (9 m) are connected into a wall
    mesh.  Candidate midpoints are generated between all blue-yellow pairs
    within _WALL_MID_DIST.  A greedy walk picks the cheapest chain through
    those midpoints, where every wall-segment crossing adds _WALL_CROSS_PENALTY
    to the step cost.

    Returns
    -------
    centreline  : (N, 2) smoothed path, or None on failure
    blue_segs   : wall segments from blue cones  (for visualisation)
    yellow_segs : wall segments from yellow cones (for visualisation)
    midpoints   : (M, 2) all candidate midpoints  (for visualisation)
    """
    # Wall mesh: extend 5 m behind the car so recently-passed cones still
    # contribute as barriers (prevents the wall from dropping away as cones leave
    # the forward window).
    blue_wall = filter_cones_forward(
        blue_cones, car_pos, car_yaw,
        min_ahead=-5.0, max_ahead=max_ahead, max_lateral=max_lateral,
    )
    yellow_wall = filter_cones_forward(
        yellow_cones, car_pos, car_yaw,
        min_ahead=-5.0, max_ahead=max_ahead, max_lateral=max_lateral,
    )

    # Midpoints only from strictly forward cones so the path starts ahead.
    blue_fwd = filter_cones_forward(
        blue_cones, car_pos, car_yaw,
        min_ahead=0.5, max_ahead=max_ahead, max_lateral=max_lateral,
    )
    yellow_fwd = filter_cones_forward(
        yellow_cones, car_pos, car_yaw,
        min_ahead=0.5, max_ahead=max_ahead, max_lateral=max_lateral,
    )

    blue_segs   = _build_wall_segments(blue_wall)
    yellow_segs = _build_wall_segments(yellow_wall)
    all_segs    = blue_segs + yellow_segs

    midpoints = _gen_midpoints(blue_fwd, yellow_fwd, car_pos, car_yaw)

    if len(midpoints) < 2:
        cl = build_local_path(blue_cones, yellow_cones, car_pos, car_yaw,
                               max_ahead, max_lateral)
        return cl, blue_segs, yellow_segs, midpoints

    ordered = _build_wall_path(midpoints, car_pos, car_yaw, all_segs)

    if len(ordered) < 2:
        cl = build_local_path(blue_cones, yellow_cones, car_pos, car_yaw,
                               max_ahead, max_lateral)
        return cl, blue_segs, yellow_segs, midpoints

    anchored = np.vstack([car_pos.reshape(1, 2), ordered])
    cl = smooth_centreline(anchored, n_out=max(20, len(ordered) * 5))
    return cl, blue_segs, yellow_segs, midpoints


def get_lookahead_waypoint(waypoints, car_pos, car_yaw,
                            lookahead_dist=5.0, min_ahead=1.0):
    """
    Project the car onto the nearest path segment, then walk lookahead_dist
    forward along the path from that projection.

    min_ahead: the returned target is guaranteed to have at least this many
    metres of forward (x_car) component.  If the arc-length walk ends up with
    less, the lookahead is extended until the constraint is satisfied or the
    path is exhausted.

    Returns a (2,) array, or None if waypoints is empty.
    """
    n = len(waypoints)
    if n == 0:
        return None
    if n == 1:
        return waypoints[0].copy()

    cos_y = math.cos(car_yaw)
    sin_y = math.sin(car_yaw)

    def _x_fwd(pt):
        rel = pt - car_pos
        return rel[0] * cos_y + rel[1] * sin_y

    def _walk(dist):
        # --- find the segment closest to the car ---
        best_seg = 0
        best_t   = 0.0
        best_d2  = np.inf

        for i in range(n - 1):
            ab  = waypoints[i + 1] - waypoints[i]
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-12:
                continue
            t = float(np.dot(car_pos - waypoints[i], ab)) / ab2
            t = max(0.0, min(1.0, t))
            proj = waypoints[i] + t * ab
            d2   = float(np.dot(car_pos - proj, car_pos - proj))
            if d2 < best_d2:
                best_d2  = d2
                best_seg = i
                best_t   = t

        # --- walk forward dist from the projection ---
        remaining = dist
        for i in range(best_seg, n - 1):
            a       = waypoints[i]
            b       = waypoints[i + 1]
            seg     = b - a
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-6:
                continue
            t0        = best_t if i == best_seg else 0.0
            available = (1.0 - t0) * seg_len
            if remaining <= available:
                return a + (t0 + remaining / seg_len) * seg
            remaining -= available

        # Past end — use a longer baseline so a curved spline tail doesn't
        # produce a backward extrapolation direction.
        n_back = min(max(1, n // 8), n - 1)
        last_dir = waypoints[-1] - waypoints[-1 - n_back]
        last_len = float(np.linalg.norm(last_dir))
        if last_len > 1e-6:
            return waypoints[-1] + (remaining / last_len) * last_dir
        return waypoints[-1].copy()

    target = _walk(lookahead_dist)

    # Enforce minimum forward component — extend lookahead if needed
    step = 1.0
    while _x_fwd(target) < min_ahead and lookahead_dist < 50.0:
        lookahead_dist += step
        target = _walk(lookahead_dist)

    # Fallback: if still behind (extrapolation failed), return the most-forward
    # waypoint on the path instead of a point behind the car.
    if _x_fwd(target) < min_ahead:
        fwds = np.array([_x_fwd(wp) for wp in waypoints])
        return waypoints[int(np.argmax(fwds))].copy()

    return target


def compute_desired_speed(waypoints, v_max=5.0, v_min=1.5, a_lat_max=4.0,
                           scan_start=1.5, scan_end=14.0, step=2.0,
                           safety: float = 0.75):
    """
    Estimate the peak curvature of the path over the next scan_end metres and
    return a safe target speed.

    Two conservative limits are applied beyond the base physics formula:

    Short-path cap: when the visible path is shorter than scan_end the car
    cannot see what is coming.  v_max is scaled down linearly so it does not
    accelerate into an unseen corner (e.g. 8 m path / 14 m scan = 57 % of
    v_max allowed).

    Safety multiplier: the curvature-derived speed is multiplied by `safety`
    (default 0.75) to compensate for spline smoothing through sparse midpoints
    producing a lower-than-true curvature estimate at tight corners.

    waypoints[0] is assumed to be the car's current position (prepended by
    build_local_path / build_path_walls), so scan_start / scan_end are
    relative to the car.
    """
    n = len(waypoints)
    if n < 3:
        return float(v_max)

    # Cumulative arc lengths
    segs  = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc   = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]

    # Short-path cap: scale v_max down when the path is shorter than scan_end.
    v_max_eff = max(v_min, v_max * min(1.0, total / scan_end))

    if total < scan_start + step:
        return float(v_max_eff)

    # Sample positions at step-metre intervals within the scan window
    sample_arcs = np.arange(scan_start, min(scan_end, total), step)
    if len(sample_arcs) < 3:
        return float(v_max_eff)

    sx  = np.interp(sample_arcs, arc, waypoints[:, 0])
    sy  = np.interp(sample_arcs, arc, waypoints[:, 1])
    pts = np.column_stack([sx, sy])

    # Menger curvature: κ = 2|cross(p2-p1, p3-p1)| / (d12 * d23 * d31)
    max_kappa = 0.0
    for i in range(1, len(pts) - 1):
        p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1]
        d12 = float(np.linalg.norm(p2 - p1))
        d23 = float(np.linalg.norm(p3 - p2))
        d31 = float(np.linalg.norm(p1 - p3))
        denom = d12 * d23 * d31
        if denom < 1e-9:
            continue
        v1    = p2 - p1
        v2    = p3 - p1
        cross = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
        kappa = 2.0 * cross / denom
        if kappa > max_kappa:
            max_kappa = kappa

    if max_kappa < 1e-4:
        return float(v_max_eff)

    v_target = safety * math.sqrt(a_lat_max / max_kappa)
    return float(max(v_min, min(v_max_eff, v_target)))


# Narrow window for direction checking — kept deliberately small so that
# outer-arc cones at corners (which appear on the wrong side at distance)
# are not mistaken for a wrong-way violation.
_DIR_LOOK_AHEAD = 8.0   # metres
_DIR_LOOK_WIDE  = 5.0   # metres lateral half-width


def check_direction(
    car_pos,
    car_yaw,
    blue_cones,
    yellow_cones,
    min_cones: int = 4,
) -> bool:
    """
    Return True when the car is travelling in the correct direction.

    Checks that blue cones are predominantly LEFT and yellow predominantly RIGHT
    within a short forward window (_DIR_LOOK_AHEAD / _DIR_LOOK_WIDE).  The
    narrow window avoids false positives at corners where outer-arc cones
    appear on the wrong side at distance.

      min_cones=4   — abstain when fewer than 4 of either colour are visible.
      threshold=2/3 — both colours must have a clear two-thirds majority on the
                      correct side before a stop is triggered.
    """
    cos_y = math.cos(car_yaw)
    sin_y = math.sin(car_yaw)

    def in_window(cones):
        if len(cones) == 0:
            return np.empty((0, 2))
        rel = cones - car_pos
        x_car =  rel[:, 0] * cos_y + rel[:, 1] * sin_y
        y_car = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
        mask = (x_car > 0.5) & (x_car < _DIR_LOOK_AHEAD) & (np.abs(y_car) < _DIR_LOOK_WIDE)
        return np.column_stack([x_car[mask], y_car[mask]]) if mask.any() else np.empty((0, 2))

    blue_view   = in_window(blue_cones)
    yellow_view = in_window(yellow_cones)

    if len(blue_view) < min_cones or len(yellow_view) < min_cones:
        return True  # not enough cones to make a confident call — proceed

    blue_on_left    = int(np.sum(blue_view[:, 1] > 0))
    yellow_on_right = int(np.sum(yellow_view[:, 1] < 0))

    # Require a clear two-thirds majority on the correct side before stopping
    return (blue_on_left    >= math.ceil(len(blue_view)   * 2 / 3) and
            yellow_on_right >= math.ceil(len(yellow_view) * 2 / 3))


