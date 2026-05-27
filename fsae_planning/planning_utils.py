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
                           scan_start=1.5, scan_end=14.0, step=2.0):
    """
    Estimate the peak curvature of the path over the next scan_end metres and
    return a safe target speed.

    Pipeline:
      1. Compute cumulative arc lengths along the waypoint array.
      2. Sample the path at `step`-metre intervals in [scan_start, scan_end]
         using linear interpolation on the arc-length parameterisation.
      3. Compute Menger curvature at every interior triplet of samples.
      4. Map the peak curvature to a speed via v = sqrt(a_lat_max / kappa),
         clamped to [v_min, v_max].

    waypoints[0] is assumed to be the car's current position (prepended by
    build_local_path), so scan_start / scan_end are relative to the car.
    """
    n = len(waypoints)
    if n < 3:
        return float(v_max)

    # Cumulative arc lengths
    segs = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]
    if total < scan_start + step:
        return float(v_max)

    # Sample positions at step-metre intervals within the scan window
    sample_arcs = np.arange(scan_start, min(scan_end, total), step)
    if len(sample_arcs) < 3:
        return float(v_max)

    sx = np.interp(sample_arcs, arc, waypoints[:, 0])
    sy = np.interp(sample_arcs, arc, waypoints[:, 1])
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
        v1 = p2 - p1
        v2 = p3 - p1
        cross = abs(float(v1[0] * v2[1] - v1[1] * v2[0]))
        kappa = 2.0 * cross / denom
        if kappa > max_kappa:
            max_kappa = kappa

    if max_kappa < 1e-4:
        return float(v_max)
    v_target = math.sqrt(a_lat_max / max_kappa)
    return float(max(v_min, min(v_max, v_target)))


def check_direction(
    car_pos,
    car_yaw,
    blue_cones,
    yellow_cones,
    look_ahead: float = 8.0,
    look_wide: float = 5.0,
    min_cones: int = 4,
) -> bool:
    """
    Return True when the car is travelling in the correct direction.

    Checks that blue cones are predominantly LEFT and yellow predominantly RIGHT
    within a short forward window.  Defaults are tuned to avoid false positives
    at corners:

      look_ahead=8m  — only nearby cones are checked; outer-arc corner cones
                       (which appear on the wrong side at distance) are ignored.
      min_cones=4    — abstain when fewer than 4 of either colour are visible,
                       which covers tight corners where cones thin out.
      threshold=2/3  — both colours must have a clear two-thirds majority on the
                       correct side before a stop is triggered; a bare majority
                       (50 %) is not enough.
    """
    cos_y = math.cos(car_yaw)
    sin_y = math.sin(car_yaw)

    def in_window(cones):
        if len(cones) == 0:
            return np.empty((0, 2))
        rel = cones - car_pos
        x_car =  rel[:, 0] * cos_y + rel[:, 1] * sin_y
        y_car = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
        mask = (x_car > 0.5) & (x_car < look_ahead) & (np.abs(y_car) < look_wide)
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


