import math

import numpy as np

# FSDS: max steering angle 25 degrees per ros-bridge.md
MAX_STEER_RAD = math.radians(25.0)

# Planning-level braking capability (m/s^2, positive magnitude) used by
# curvature_speed()'s braking-distance propagation. Deliberately below the
# vehicle's true limit (~9 m/s^2): the tyres also need lateral force for the
# corner being braked into (friction circle), plus margin for model error and
# actuation lag. Mirrors a_brake_max in fsae_MPCTest/sim/speed_profile.py.
A_BRAKE_PLAN = 5.0


def _heading_error(car_pos, car_yaw, target_global) -> float:
    """Return heading error in radians: positive when target is left of car."""
    dx = target_global[0] - car_pos[0]
    dy = target_global[1] - car_pos[1]
    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)
    x_car =  dx * cos_y + dy * sin_y
    y_car = -dx * sin_y + dy * cos_y
    return math.atan2(y_car, x_car)


def compute_steering(car_pos, car_yaw, target_global) -> float:
    """Pure-proportional steering (legacy helper, kept for reference)."""
    return float(np.clip(-_heading_error(car_pos, car_yaw, target_global)
                         / MAX_STEER_RAD, -1.0, 1.0))


class StanleyController:
    """
    Stanley path-tracking steering controller (Thrun et al., DARPA 2005).

    δ = θ_e + arctan(k_cte · e / (v + k_soft))

    θ_e — heading error: path tangent angle minus car yaw (rad).
          Positive when path turns left relative to the car.
    e   — cross-track error: signed lateral distance from the front axle
          to the nearest path point (m), positive when the axle is to
          the RIGHT of the path.
    v   — car speed (m/s); k_soft prevents division by zero at standstill.

    Sign convention (FSDS ENU: x forward, y left):
      output > 0  → steer right
      output < 0  → steer left
      output ∈ [-1, 1]

    Tuning:
      k_cte     — cross-track gain.  Higher values correct lateral error faster
                  but cause oscillation on a high-speed straight.
      k_soft    — speed softening (m/s).  Set to ~walking speed so the CTE
                  term doesn't saturate the steering at low speeds.
      k_d       — yaw-rate damper gain.  Subtracts k_d·ω from the Stanley
                  angle before normalising, opposing rapid heading changes.
                  This is the primary fix for left-right sway: the CTE term
                  alone has no memory of how fast the heading is already
                  changing, so it overshoots; k_d·ω counters each swing.
      wheelbase — distance from rear to front axle (m).  Used to project
                  the control point to the front axle, which is where Stanley
                  measures cross-track error.
      arc_window — half-width (m) of the arc-length window the segment search
                  is restricted to around the PREVIOUS tick's match (see
                  compute()'s docstring on why this isn't a plain global
                  nearest-point search).

    Nearest-point search:
      The front axle is projected onto the nearest PATH SEGMENT (continuous,
      clamped projection), not just the nearest waypoint — a plain nearest-
      vertex search picks a needlessly coarse line when waypoint spacing is
      coarse relative to the turn radius, and (worse) on a path that loops
      back near itself (e.g. a teardrop corner's pinch) a pure Euclidean
      search can jump to a spatially-close segment on the WRONG leg, flipping
      e_psi by ~180° tick-to-tick. To prevent that jump, the search only
      considers segments within `arc_window` metres (by arc length from the
      path start) of the PREVIOUS tick's matched point — the true match can
      only move a bounded, continuous distance per tick, whereas a wrong-leg
      segment is normally arc-length-far even when it's Euclidean-close. The
      first tick (or any tick where the window contains nothing, e.g. the
      path just got much shorter) falls back to a global search.
    """

    _ARC_WINDOW_DEFAULT = 4.0  # metres; see arc_window in the class docstring

    def __init__(
        self,
        k_cte: float = 1.0,
        k_soft: float = 1.0,
        k_d: float = 0.1,
        wheelbase: float = 1.5,
        arc_window: float = _ARC_WINDOW_DEFAULT,
    ):
        self.k_cte     = k_cte
        self.k_soft    = k_soft
        self.k_d       = k_d
        self.wheelbase = wheelbase
        self.arc_window = arc_window
        # Last tick's tracking error (+ve = left/CCW), matching
        # telemetry_logger.py's CSV convention — see compute()'s sign flip
        # on `e` below. Exposed for ControlLogger; not used internally by
        # compute() itself.
        self.last_e_y: float = 0.0
        self.last_e_psi: float = 0.0
        # Arc length (m) of the last matched point along the last path
        # searched; seeds next tick's local search window. None = search
        # the whole path (first tick, or after reset()/a failed match).
        self._last_s: float | None = None

    def reset(self) -> None:
        """Forget the remembered match, forcing a full-path search next tick."""
        self._last_s = None

    def _search_window(self, s_cum: np.ndarray) -> np.ndarray:
        """Segment indices to search: local window around self._last_s, or
        every segment if there's no previous match (or it no longer fits)."""
        n_segs = len(s_cum) - 1
        if self._last_s is not None:
            lo = self._last_s - self.arc_window
            hi = self._last_s + self.arc_window
            mask = (s_cum[:-1] <= hi) & (s_cum[1:] >= lo)
            idxs = np.nonzero(mask)[0]
            if len(idxs) > 0:
                return idxs
        return np.arange(n_segs)

    @staticmethod
    def _nearest_segment(path, seg_vecs, seg_lens, fa, candidates):
        """
        Closest point among `candidates` segments to `fa`, via a clamped
        projection onto each segment (not just its nearest endpoint).

        Returns (idx, proj_m) — the segment's start index and the arc length
        (m) from that start to the closest point — or (None, 0.0) if every
        candidate segment is degenerate (zero length).
        """
        best_dist = math.inf
        best_idx = None
        best_proj_m = 0.0
        for i in candidates:
            seg_len = seg_lens[i]
            if seg_len < 1e-6:
                continue
            seg = seg_vecs[i]
            proj = float(np.dot(fa - path[i], seg) / (seg_len * seg_len))
            proj_clamped = min(1.0, max(0.0, proj))
            closest = path[i] + proj_clamped * seg
            dist = float(np.linalg.norm(fa - closest))
            if dist < best_dist:
                best_dist = dist
                best_idx = int(i)
                best_proj_m = proj_clamped * seg_len
        return best_idx, best_proj_m

    def compute(
        self,
        path: np.ndarray,
        car_pos: np.ndarray,
        car_yaw: float,
        car_speed: float,
        car_yaw_rate: float = 0.0,
    ) -> float:
        """
        Return a steering command in [-1, 1].

        path         — (N, 2) array of waypoints in map frame (must have N ≥ 2)
        car_pos      — (2,) car position in map frame
        car_yaw      — car heading in radians
        car_speed    — car speed in m/s
        car_yaw_rate — yaw rate in rad/s (positive = left / CCW); used by the
                       damper term to oppose rapid heading changes

        Also updates self.last_e_y / self.last_e_psi (see their docstring).
        """
        if len(path) < 2:
            self._last_s = None
            return 0.0

        # Project control point to front axle
        fa = car_pos + self.wheelbase * np.array([math.cos(car_yaw), math.sin(car_yaw)])

        seg_vecs = np.diff(path, axis=0)
        seg_lens = np.linalg.norm(seg_vecs, axis=1)
        s_cum = np.concatenate([[0.0], np.cumsum(seg_lens)])  # arc length at each waypoint

        candidates = self._search_window(s_cum)
        idx, proj_m = self._nearest_segment(path, seg_vecs, seg_lens, fa, candidates)
        if idx is None:
            self._last_s = None
            return 0.0

        t = seg_vecs[idx] / seg_lens[idx]
        self._last_s = float(s_cum[idx] + proj_m)

        # Heading error: path_yaw - car_yaw, normalised to (-π, π)
        path_yaw = math.atan2(t[1], t[0])
        theta_e = math.atan2(
            math.sin(path_yaw - car_yaw),
            math.cos(path_yaw - car_yaw),
        )

        # Cross-track error: right-normal of path, positive = axle right of path.
        # right_n ⊥ t, so this is the perpendicular distance to the infinite
        # line through the segment — invariant to where along that segment
        # path[idx] sits, so no separate "use the clamped closest point"
        # term is needed here.
        right_n = np.array([t[1], -t[0]])   # 90° CW rotation of tangent
        e = float(np.dot(fa - path[idx], right_n))

        # Expose this tick's error with +ve = left/CCW — e above is +ve
        # RIGHT, the opposite — matching telemetry_logger.py's CSV
        # convention. theta_e already matches (+ve = path turns left of car).
        self.last_e_y = -e
        self.last_e_psi = theta_e

        # Stanley angle — positive = left turn (standard convention).  Damper
        # subtracts k_d·ω: when the car is already swinging left (ω > 0), this
        # reduces δ so the next tick steers less left, preventing overshoot.
        delta = (theta_e
                 + math.atan2(self.k_cte * e, car_speed + self.k_soft)
                 - self.k_d * car_yaw_rate)

        # Return the steering ANGLE in radians (positive = left), clamped to the
        # physical limit.  FSDS normalisation (+1 = right) is done by fsds_bridge.
        return float(np.clip(delta, -MAX_STEER_RAD, MAX_STEER_RAD))


def curvature_speed(waypoints, v_max=15.0, v_min=1.5, a_lat_max=4.0,
                    scan_start=0.0, scan_end=24.0, step=2.0, safety=0.9):
    """
    Curvature-limited target speed over the next scan_end metres of the path.

    Ported from the planner's speed logic so the controller can set cmd_vel.speed
    without a cross-package import. v_target = safety·√(a_lat_max / κ_peak) for
    each corner in the scan window, propagated for braking distance (see below)
    and reduced to the most restrictive value. A short-path cap (scales v_max
    down when the visible path is shorter than scan_end) applies only when
    there isn't enough path to measure curvature at all — once curvature has
    been measured, the short-path cap is not reapplied on top of v_target.
    waypoints[0] is assumed to be the car's current position.

    scan_end=24 m is sized so a tight hairpin (~2 m radius, v_target ~2.7 m/s)
    approached at v_max is visible far enough out to brake for at a realistic
    deceleration; a shorter scan sees the corner too late, saturating steering
    at corner entry. Kept in sync with the planner's own lookahead.

    Braking-distance propagation: each corner in the scan window is converted
    to the fastest speed from which that corner is still reachable at
    A_BRAKE_PLAN, and the most restrictive wins — otherwise a corner 24 m
    ahead and the same corner 2 m ahead give the same target, which can demand
    a deceleration the car cannot produce.

    scan_start=0.0: curvature measurement starts at the car's own position
    rather than ahead of it, so a short, tight corner isn't skipped by a dead
    zone (see the apex-blind-spot comment below).

    safety=0.8 (was 1.0): the live Stanley path has no other margin layered on
    top of this — at safety=1.0 the corner speed
    is the raw a_lat_max-limited value with zero derating, which overspent the
    tyre model's actual grip on a sharp corner and caused a loss of control
    on the teardrop apex. 0.8 gives a real margin without over-slowing normal
    corners.
    """
    n = len(waypoints)
    if n < 3:
        return float(v_max)

    segs  = np.linalg.norm(np.diff(waypoints, axis=0), axis=1)
    arc   = np.concatenate([[0.0], np.cumsum(segs)])
    total = arc[-1]

    v_max_eff = max(v_min, v_max * min(1.0, total / scan_end))
    if total < scan_start + step:
        return float(v_max_eff)

    hi = min(scan_end, total)
    # The planner re-fits the centreline every frame, so the published path
    # carries a few cm of per-point lateral wiggle even on a straight. Taking
    # the raw MAX Menger curvature over ~2 m triples turns that noise into a
    # spurious kappa and makes v_target oscillate frame-to-frame. Fix: densely
    # resample the scan window and moving-average denoise it before measuring
    # curvature — a real corner is a sustained bend that survives smoothing,
    # only the cm-scale wiggle is removed.
    #
    # pts_s0 is the arc distance ahead of the car of pts[0]. A width-w 'valid'
    # moving average places its first output at the centre of the first
    # window, (w-1)/2 * dense_step further along than scan_start, so this is
    # tracked explicitly to keep the braking propagation below honest.
    #
    # A short, tight corner's curvature can be sustained over only ~2-3 m of
    # arc, so scan_start=0.0, a fine dense_step (0.5) and a narrow smoothing
    # window (w<=3, no decimation) keep the effective measurement start close
    # to the car and leave enough samples inside a short apex zone to see it.
    pts = None
    pts_s0 = scan_start
    dense_step = 0.5
    dense = np.arange(scan_start, hi, dense_step)
    if len(dense) >= 7:                       # room to smooth and still leave >=3 triples
        dx = np.interp(dense, arc, waypoints[:, 0])
        dy = np.interp(dense, arc, waypoints[:, 1])
        w  = min(3, len(dense) - 4)           # 'valid' conv keeps len-w+1 >= 3 points
        ker = np.ones(w) / w
        sx = np.convolve(dx, ker, mode='valid')
        sy = np.convolve(dy, ker, mode='valid')
        pts = np.column_stack([sx, sy])       # no decimation -- keep every smoothed sample
        pts_s0 = scan_start + (w - 1) / 2.0 * dense_step
    if pts is None or len(pts) < 3:
        # Short scan window: no headroom to denoise — fall back to coarse sampling.
        sample_arcs = np.arange(scan_start, hi, step)
        if len(sample_arcs) < 3:
            return float(v_max_eff)
        sx  = np.interp(sample_arcs, arc, waypoints[:, 0])
        sy  = np.interp(sample_arcs, arc, waypoints[:, 1])
        pts = np.column_stack([sx, sy])
        pts_s0 = scan_start

    # Collect every triple's Menger curvature, then reduce below rather than
    # taking the raw max, which would let a single bad triple (a frame-to-frame
    # centreline-fit artifact, not a real corner) set the speed for the whole
    # window. kappa_at[j] records which pts index kappas[j] is centred on —
    # needed because a degenerate triple is skipped, so the index can't be
    # assumed from position alone.
    kappas = []
    kappa_at = []
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
        kappas.append(2.0 * cross / denom)
        kappa_at.append(i)

    if not kappas:
        return float(v_max_eff)

    k = np.asarray(kappas, dtype=float)
    k_at = np.asarray(kappa_at, dtype=int)
    # A genuine corner spans several consecutive triples, so a short running
    # mean survives it while an isolated fit artifact gets averaged down.
    # Reduce with a max over the smoothed series rather than a percentile of
    # the raw one — the scan window only yields ~7 triples, so a p75/p90 is
    # both noisy and biased toward speeds faster than the raw max would give.
    # The 'valid' 3-point mean drops one entry at each end, so the surviving
    # centres are k_at[1:-1]; tracked alongside k rather than assumed fixed.
    if len(k) >= 3:
        k = np.convolve(k, np.ones(3) / 3.0, mode='valid')
        k_at = k_at[1:len(k) + 1]

    # ── Braking-distance propagation ──────────────────────────────────────
    # A single max over the window ignores WHERE the corner is: a hairpin
    # 24 m ahead and the same hairpin 2 m ahead would produce an identical
    # target, demanding a deceleration the car cannot physically produce as
    # it gets closer. Instead, for each sampled corner at distance d ahead
    # with its own corner-speed limit v_corner, the fastest we may travel now
    # and still brake to it is
    #     v_allowed = sqrt(v_corner^2 + 2 * a_brake * d)
    # (from v_f^2 = v_i^2 - 2*a*d). Take the most restrictive over the window;
    # a corner far enough away imposes no limit since v_allowed then exceeds
    # v_max_eff.
    #
    # Distances: each surviving entry k[j] is centred on pts[k_at[j]], whose
    # distance ahead of the car is pts_s0 (where pts[0] actually sits,
    # including the moving-average centre shift) plus the arc length along
    # pts to there.
    #
    # entry_margin: a curvature sample describes the middle of a bend, but the
    # car must already be at corner speed by the bend's entry, which is
    # earlier — braking to the sample's own distance would be too late. Sized
    # at one triple half-width (the arc a single curvature sample spans) so
    # it scales with the sampling density.
    k_safe = np.maximum(k, 1e-9)
    v_corner = safety * np.sqrt(a_lat_max / k_safe)
    if len(pts) > 1:
        pts_arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        )
        entry_margin = float(np.median(np.diff(pts_arc))) if len(pts_arc) > 1 else 0.0
        d_ahead = (pts_s0
                   + pts_arc[np.clip(k_at, 0, len(pts_arc) - 1)]
                   - entry_margin)
        d_ahead = np.maximum(d_ahead, 0.0)
    else:
        d_ahead = np.full(len(k), pts_s0)

    v_allowed = np.sqrt(v_corner ** 2 + 2.0 * A_BRAKE_PLAN * d_ahead)
    v_target = float(np.min(v_allowed))

    # v_max_eff (the short-path-scaled ceiling) is not reapplied here — it
    # only applies to the early-return "not enough path to measure curvature"
    # cases above. v_max itself is still re-clamped: on a straight approach,
    # before a corner enters the scan window, kappa is measured near zero, so
    # v_target can be arbitrarily large otherwise.
    return float(np.clip(v_target, v_min, v_max))


def curvature_speed_profile(waypoints, v_max=15.0, v_min=1.5, a_lat_max=4.0,
                             scan_start=0.0, scan_end=24.0, step=2.0, safety=0.8):
    """
    Per-waypoint target speed: curvature_speed() evaluated as if the car were
    sitting at each waypoint in turn, scanning the path ahead of it from there.

    Visualisation-only (see target_speed_viz.py) — the live control loop only
    ever needs curvature_speed()'s single current-position value and calls
    that directly; this is O(n^2) (one curvature_speed() scan per point) and
    is not meant to run in that hot path. Reuses curvature_speed() verbatim
    rather than re-deriving the profile some cheaper way, so the visualised
    target always matches exactly what the controller would command. Default
    safety kept identical to curvature_speed()'s so the visualisation doesn't
    silently drift from what's actually commanded.
    """
    n = len(waypoints)
    if n == 0:
        return np.empty(0)
    return np.array([
        curvature_speed(waypoints[i:], v_max=v_max, v_min=v_min, a_lat_max=a_lat_max,
                         scan_start=scan_start, scan_end=scan_end, step=step, safety=safety)
        for i in range(n)
    ])


def _load_profile_csv(csv_path: str):
    """
    Shared reader for fsae_MPCTest's tuner/export_speed_profile.py /
    tuner/tools/raceline_optimizer.py CSVs. Accepts two header shapes:
      "x,y,psi,v_target"              (4 columns, speed_profile.csv and
                                        older raceline.csv exports)
      "x,y,psi,psi_target,v_target"   (5 columns, raceline_optimizer.py
                                        exports since the shaped
                                        heading-lead reference was added —
                                        see late_turn_in_investigation.md
                                        Part 8/9)
    Comment lines starting with '#' are skipped. Column count is detected
    per-file from the first data row, not the header text, so a caller
    doesn't need to know in advance which shape a given CSV has.

    Deliberately a plain reader with no scipy/centreline-reconstruction
    dependency — that runs once, offline, when the CSV is exported.

    Returns
    -------
    (path_X, path_Y, path_Psi, path_PsiTarget, path_V) : tuple of
        np.ndarray, shape (n,). path_PsiTarget equals path_Psi
        (the geometric tangent) for a 4-column file — i.e. an old/
        speed-profile CSV with no shaped-heading column behaves exactly
        as if psi_target had been exported equal to psi, a genuine no-op
        for every caller that doesn't ask for it explicitly.

    Raises
    ------
    FileNotFoundError, ValueError : bad csv_path or malformed contents.
    """
    xs, ys, psis, psi_targets, vs = [], [], [], [], []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('x,y'):
                continue
            fields = line.split(',')
            x_s, y_s, psi_s = fields[0], fields[1], fields[2]
            if len(fields) >= 5:
                psi_target_s, v_s = fields[3], fields[4]
            else:
                psi_target_s, v_s = psi_s, fields[3]
            xs.append(float(x_s))
            ys.append(float(y_s))
            psis.append(float(psi_s))
            psi_targets.append(float(psi_target_s))
            vs.append(float(v_s))
    if len(xs) < 2:
        raise ValueError(f"{csv_path}: fewer than 2 valid rows")
    return (np.asarray(xs), np.asarray(ys), np.asarray(psis),
            np.asarray(psi_targets), np.asarray(vs))


def load_speed_profile_csv(csv_path: str):
    """
    Load a pre-computed (x, y, v_target) speed profile exported by
    fsae_MPCTest's tuner/export_speed_profile.py.

    For a track that's already been mapped, this replaces curvature_speed()'s
    per-tick re-derivation with a lookup against an oracle profile computed
    once, offline, from the whole recorded map — the live-built centreline is
    frequently shorter than curvature_speed()'s own scan_end (perception FOV
    clips laterally on a corner before its forward window does), so this
    bypasses that shortfall for a track that's already known.

    Parameters
    ----------
    csv_path : str   Path to a CSV with header "x,y,psi,v_target" (comment
                      lines starting with '#' are skipped).

    Returns
    -------
    (path_X, path_Y, path_V) : tuple of np.ndarray, shape (n,)

    Raises
    ------
    FileNotFoundError, ValueError : bad csv_path or malformed contents.
    """
    path_X, path_Y, _path_Psi, _path_PsiTarget, path_V = _load_profile_csv(csv_path)
    return path_X, path_Y, path_V


def load_path_profile_csv(csv_path: str):
    """
    Load a pre-computed (x, y) path exported by fsae_MPCTest's
    tuner/export_speed_profile.py, for use as a drop-in replacement of the
    live planner's /fsae/planning/selected_trajectory centreline.

    For a track that's already been mapped, this removes the live planner
    (centerline_planner.py / boundary.py) from the control loop entirely,
    isolating controller/plant tracking error from planner-induced path
    error.

    psi is exported but not returned here: StanleyController already derives
    path heading from consecutive waypoints (atan2 of the segment direction,
    the same convention centerline_planner.py's published PoseArray uses),
    so the (n,2) array below is a direct substitute for the live topic's
    path with no interface change.

    Parameters
    ----------
    csv_path : str   Path to a CSV with header "x,y,psi,v_target" (comment
                      lines starting with '#' are skipped).

    Returns
    -------
    path : np.ndarray, shape (n, 2)   [x, y] waypoints, global frame.

    Raises
    ------
    FileNotFoundError, ValueError : bad csv_path or malformed contents.
    """
    path_X, path_Y, _path_Psi, _path_PsiTarget, _path_V = _load_profile_csv(csv_path)
    return np.column_stack([path_X, path_Y])


def precomputed_speed_at(car_pos, path_X, path_Y, path_V) -> float:
    """
    Nearest-point lookup into a pre-computed speed profile (see
    load_speed_profile_csv()).

    Deliberately a plain nearest-point search, not a Frenet/arc-length
    projection: the profile is dense (see export script, default 1000 pts
    over a lap), so nearest-point error is small and this avoids needing the
    heading/tangent bookkeeping a proper Frenet projection would add for a
    speed lookup that only needs to be "close enough," unlike e_y/e_psi
    tracking error, which does need that precision.

    Parameters
    ----------
    car_pos : array-like, shape (2,)   Car's current [x, y] (global frame).
    path_X, path_Y, path_V : np.ndarray, shape (n,)   From load_speed_profile_csv().

    Returns
    -------
    float : v_target at the nearest profile point to car_pos.
    """
    d2 = (path_X - car_pos[0]) ** 2 + (path_Y - car_pos[1]) ** 2
    return float(path_V[int(np.argmin(d2))])
