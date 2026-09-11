"""
Barebone centreline planner node.

Plans a centreline a few midpoints at a time using the cone-wall barrier planner
(see boundary.build_path_walls): cones are Delaunay-triangulated, same-colour
edges become a wall mesh, and each mixed triangle's two cross-colour edges
become a corridor "gate pair" that the walk chains into a path.  There is NO
localisation — the car plans purely from the boundary cones it currently has,
lap after lap.  This is a pure first-lap centreline follower; the same rolling-
window plan is republished every lap.

Interface (matches the fsae_autonomous car stack):

    in   /fsae/slam/left_track    fsae_interfaces/Track     blue (left) boundary, global frame
    in   /fsae/slam/right_track   fsae_interfaces/Track     yellow (right) boundary, global frame
    in   /fsae/slam/car_position  geometry_msgs/PoseStamped x,y in position; yaw in orientation.w
    out  /fsae/planning/selected_trajectory  geometry_msgs/PoseArray   centreline waypoints

The plan loop is triggered by each car_position update (upstream convention),
using whatever left_track/right_track cones are cached at that moment. An
earlier revision gated this behind a "new cone data has arrived" flag to
avoid re-testing an unchanged cone frame against an ever-more-rotated pose;
that halved the effective plan rate (cones publish at half the pose rate),
which lengthened blend_paths' effective time constant enough to reintroduce
path lag/flip symptoms at corners, so it was reverted. If the disappearing-
path-at-a-corner symptom that fix targeted reappears, see path_hold_timeout
and git history for that approach rather than re-deriving it from scratch.

Divergence from the real stack's wall_centerline_planner (fsae_autonomous):
that node also publishes an EXPLICIT EMPTY PoseArray when it has no valid
path, and runs a pose-independent watchdog timer whose sole job is to force
that empty publish if poses stop arriving.  Both exist because the real
Stanley controller caches its last tx/ty indefinitely and only clears them on
receiving a fresh empty PoseArray, so silence alone would not stop the car.
This repo's stanley_controller already treats an absent plan correctly
(_control_step returns early on a short path), so neither is ported here —
the node stays silent when it has nothing to say.  If that controller
behaviour ever changes, port both together; the watchdog is useless without
the empty publish.
"""
import math
import time

import numpy as np
import rclpy
from rclpy.node import Node

from fsae_interfaces.msg import Track
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from std_msgs.msg import Empty

from fsae_planning.boundary import (
    build_path_walls,
    build_wall_segments_delaunay,
    segment_crosses_walls,
    _WALL_MAX_TURN_COS,
    _WALL_MID_MIN_DELAUNAY,
    _WALL_MID_MAX_DELAUNAY,
)
from fsae_planning.cone_sorting import MAX_PAIR_DIST
from fsae_planning.path_utils import (
    blend_paths,
    build_local_path,
    DEFAULT_SMOOTH_PER_PT,
)

# _planning_loop runs synchronously inside the pose-subscription callback, on
# rclpy's default SingleThreadedExecutor (see main()) -- there is no
# background thread, so a single slow build_path_walls() call blocks EVERY
# other callback (pose, left_track, right_track) for its entire duration.
# That's a much smaller budget than it sounds: max_pose_age (0.15s) is the
# gate that decides whether the NEXT tick even considers this data fresh, so
# a compute this node itself takes 50+ ms on is already eating into its own
# freshness margin. Logged so a slow tick shows up in the node's own log
# instead of only being visible as an unexplained gap in the path.
_SLOW_TICK_WARN_SEC = 0.05

_CONE_DEDUP_DIST = 0.01   # metres — merge-distance for coincident/near-duplicate cones


def cones_to_array(cones, logger=None, label: str = 'cones') -> np.ndarray:
    """
    geometry_msgs/Point[] → (N, 2) float64 array of x,y.

    Drops non-finite points (a NaN/Inf cone would otherwise reach
    build_path_walls' Delaunay triangulation and either raise or, worse,
    silently corrupt the wall mesh -- see _sanitize's own finite check, which
    only covers the PATH, not the raw cone inputs feeding its geometry) and
    merges near-duplicate coincident points (two cones within
    _CONE_DEDUP_DIST of each other -- a repeated detection of the same real
    cone -- are collapsed to one; scipy's Delaunay treats coincident points
    as a degenerate simplex, which can either raise a QhullError or produce
    an unstable triangulation).
    """
    if not cones:
        return np.empty((0, 2), dtype=np.float64)
    arr = np.array([[p.x, p.y] for p in cones], dtype=np.float64)
    finite = np.all(np.isfinite(arr), axis=1)
    n_bad = int(len(arr) - finite.sum())
    arr = arr[finite]
    if n_bad and logger is not None:
        logger.warn(f'{label}: dropped {n_bad} non-finite cone(s)', throttle_duration_sec=2.0)

    if len(arr) < 2:
        return arr

    kept: list = []
    for pt in arr:
        if not any(float(np.linalg.norm(pt - k)) < _CONE_DEDUP_DIST for k in kept):
            kept.append(pt)
    if len(kept) != len(arr) and logger is not None:
        logger.warn(
            f'{label}: merged {len(arr) - len(kept)} near-duplicate cone(s)',
            throttle_duration_sec=2.0)
    return np.array(kept, dtype=np.float64)


def _direction_from_path(
    path: np.ndarray | None,
    origin: np.ndarray,
    lookahead: float = 1.5,
) -> np.ndarray | None:
    """
    Unit direction from `origin` toward the path point ~lookahead metres of arc
    length along `path` (clamped to the last point on a shorter path).

    Used to seed next tick's wrong-leg guard (see boundary.build_path_walls'
    prev_dir) with a stable heading estimate -- the immediately-adjacent point
    (index 1) sits too close to `origin` on a densely-resampled path for its
    direction to be numerically reliable.
    """
    if path is None or len(path) < 2:
        return None
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    idx = min(int(np.searchsorted(arc, lookahead)), len(path) - 1)
    d = path[idx] - origin
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-6 else None


class CenterlinePlanner(Node):
    def __init__(self, node_name: str = 'centerline_planner'):
        super().__init__(node_name)

        # Per-point spline smoothing budget (splprep s = smooth * n_points).
        # 0.0 → interpolating spline (reproduces every cone-pairing kink);
        # a small positive value approximates the midpoints for a clean line.
        self.declare_parameter('smooth', DEFAULT_SMOOTH_PER_PT)

        # Arc-length horizon (m) the published centreline is clamped to.  Keeps
        # the near path in front of the car invariant to how far the lookahead
        # reaches (extra far midpoints no longer reshape it) and stops distant
        # apex points dragging the corner line inward. Must match
        # boundary._WALL_PLAN_HORIZON — see that constant's comment for why
        # 25.0 (braking-distance math), not repeated here.
        self.declare_parameter('plan_horizon', 25.0)

        # Temporal path blend weight toward each freshly-planned path (EMA in the
        # map frame).  The planner rebuilds the path from scratch every tick, so
        # without blending successive paths jump and the controller jerks.
        # 1.0 disables blending (pure new path); smaller = smoother/laggier.
        self.declare_parameter('path_blend', 0.4)

        # Cone visibility radius for planning (m).  Crops the latest
        # boundary-frame cones for the 'nn' midpoint method and for the
        # fallback planner; the default 'delaunay' corridor build
        # triangulates the full cone set (see boundary._build_corridor_path)
        # and bounds itself by plan_horizon instead.
        # 25.0 — kept >= plan_horizon so the midpoint mesh actually extends as
        # far as the path is allowed to; see plan_horizon.
        self.declare_parameter('look_radius', 25.0)

        # Bounded hold on the last known-good path when a tick fails to produce
        # one (a thin perception frame, momentary occlusion at a tight corner).
        # Bridges a short gap without accumulating any cone state -- if the gap
        # outlasts this, the planner gives up and stops publishing rather than
        # trusting an increasingly outdated path. See _finalize.
        self.declare_parameter('path_hold_timeout', 0.3)

        # 'delaunay' (default): each mixed triangle of a Delaunay
        # triangulation of both boundaries contributes its 2 cross-colour
        # edges directly as a centreline segment (see
        # boundary._build_corridor_path) -- provably can't cross a wall
        # segment by construction (a triangle's interior can't reach any
        # other triangle's edges), so no per-step wall-crossing check is
        # needed to chain them. 'nn' is the older exclusive-nearest-
        # neighbour matching (boundary._gen_midpoints), kept for
        # rollback/comparison.
        self.declare_parameter('midpoint_method', 'delaunay')

        # Cross-colour gate length band for the 'delaunay' corridor method
        # (see boundary._build_corridor_graph) -- exposed so a track's actual
        # cone spacing can be tuned without a code change; a fixed band can
        # reject a geometrically consistent corridor whose triangle diagonals
        # exceed it even though the true cross-track width does not.
        self.declare_parameter('mid_min_gate', _WALL_MID_MIN_DELAUNAY)
        self.declare_parameter('mid_max_gate', _WALL_MID_MAX_DELAUNAY)

        # Max left/right pairing distance for the exception-path fallback
        # planner (build_local_path) -- kept separately configurable from
        # mid_max_gate so the fallback's width policy can be aligned with it
        # instead of always using the wide cone_sorting.MAX_PAIR_DIST default.
        self.declare_parameter('fallback_max_pair_dist', MAX_PAIR_DIST)

        # -- input staleness gates. Ported from the real stack, where a
        # camera/SLAM pipeline can stall or drop frames. The simulator's
        # perception is synchronous ground truth so these rarely fire here,
        # but they are what turns a stalled or half-updated input into "no
        # plan this tick" instead of a centreline built from a boundary pair
        # that never coexisted. Ages are measured from LOCAL ARRIVAL TIME,
        # not message stamps: Track.msg carries no header at all, and
        # car_position's PoseStamped stamp is the upstream odom's own
        # measurement time, which need not share a clock domain with this
        # node -- comparing it against get_clock().now() would report a
        # constant offset as permanent staleness.
        self.declare_parameter('max_pose_age', 0.15)
        self.declare_parameter('max_track_age', 0.50)
        # Max allowed arrival skew between the left/right track updates (both
        # can individually pass max_track_age while having arrived a full
        # max_track_age apart -- e.g. one side stalls one cycle -- which
        # plans a centreline from a boundary pair that never coexisted).
        self.declare_parameter('max_track_skew', 0.20)

        # -- output sanity gates (see _sanitize).
        self.declare_parameter('max_point_jump', 10.0)
        # Minimum clearance (m) any published waypoint must keep from every
        # known cone -- segment_crosses_walls only catches a path crossing
        # the WALL MESH between cones, not a path that threads directly
        # through/beside a single cone without crossing any wall segment
        # (e.g. clipping a corner cone). Should be tuned as at least
        # half the vehicle track width + expected tracking error +
        # localisation uncertainty; 0 disables the check.
        self.declare_parameter('min_cone_clearance', 0.4)
        # Minimum remaining arc length (m) a candidate path must have (after
        # any wall/clearance truncation) to be published/held at all. A
        # 2-point stub technically satisfies every other check but gives the
        # controller no real stopping distance and no shape to judge
        # curvature feasibility from -- reject it outright.
        self.declare_parameter('min_usable_path_length', 2.0)
        # How far the car may drift from a HELD path (see path_hold_timeout)
        # before the hold is no longer trusted -- guards against republishing
        # a stale path unchanged after a localisation jump or map update, see
        # _hold_reachable.
        self.declare_parameter('max_hold_deviation', 2.0)

        self._smooth_per_pt      = self.get_parameter('smooth').get_parameter_value().double_value
        self._plan_horizon       = self.get_parameter('plan_horizon').get_parameter_value().double_value
        self._path_blend         = self.get_parameter('path_blend').get_parameter_value().double_value
        self._look_radius        = self.get_parameter('look_radius').get_parameter_value().double_value
        self._path_hold_timeout  = self.get_parameter('path_hold_timeout').get_parameter_value().double_value
        self._midpoint_method    = self.get_parameter('midpoint_method').get_parameter_value().string_value
        self._mid_min_gate       = self.get_parameter('mid_min_gate').get_parameter_value().double_value
        self._mid_max_gate       = self.get_parameter('mid_max_gate').get_parameter_value().double_value
        self._fallback_max_pair_dist = \
            self.get_parameter('fallback_max_pair_dist').get_parameter_value().double_value
        self._max_pose_age       = self.get_parameter('max_pose_age').get_parameter_value().double_value
        self._max_track_age      = self.get_parameter('max_track_age').get_parameter_value().double_value
        self._max_track_skew     = self.get_parameter('max_track_skew').get_parameter_value().double_value
        self._max_point_jump     = self.get_parameter('max_point_jump').get_parameter_value().double_value
        self._min_cone_clearance = self.get_parameter('min_cone_clearance').get_parameter_value().double_value
        self._min_usable_path_length = \
            self.get_parameter('min_usable_path_length').get_parameter_value().double_value
        self._max_hold_deviation = self.get_parameter('max_hold_deviation').get_parameter_value().double_value

        # -- parameter sanity. An unrecognised midpoint_method used to
        # silently fall through to the 'nn' branch in
        # boundary.build_path_walls (its `if 'delaunay' ... else` gate) --
        # clamp and warn here instead of behaving differently from what was
        # actually configured.
        if self._midpoint_method not in ('delaunay', 'nn'):
            self.get_logger().error(
                f"unrecognised midpoint_method '{self._midpoint_method}' "
                "(expected 'delaunay' or 'nn') -- forcing 'delaunay'")
            self._midpoint_method = 'delaunay'
        for name, val in (
            ('max_pose_age', self._max_pose_age),
            ('max_track_age', self._max_track_age),
            ('max_track_skew', self._max_track_skew),
            ('path_hold_timeout', self._path_hold_timeout),
            ('plan_horizon', self._plan_horizon),
            ('look_radius', self._look_radius),
        ):
            if not (math.isfinite(val) and val > 0.0):
                raise ValueError(f'{name} must be a finite positive number, got {val!r}')
        if not (math.isfinite(self._path_blend) and 0.0 < self._path_blend <= 1.0):
            raise ValueError(f'path_blend must be in (0, 1], got {self._path_blend!r}')
        for name, val in (
            ('max_point_jump', self._max_point_jump),
            ('max_hold_deviation', self._max_hold_deviation),
            ('min_cone_clearance', self._min_cone_clearance),
            ('min_usable_path_length', self._min_usable_path_length),
            ('smooth', self._smooth_per_pt),
        ):
            if not (math.isfinite(val) and val >= 0.0):
                raise ValueError(f'{name} must be a finite non-negative number, got {val!r}')

        self.create_subscription(Track, '/fsae/slam/left_track',  self._left_cb,  10)
        self.create_subscription(Track, '/fsae/slam/right_track', self._right_cb, 10)
        self.create_subscription(PoseStamped, '/fsae/slam/car_position', self._pose_cb, 10)

        # Debug hook: the planner has no cone memory of its own (see _compute_path),
        # but blend_paths() still eases between the last published path and each
        # fresh one, and a bounded hold (path_hold_timeout) can republish the
        # last known-good path for a tick or two. An Empty here drops all of
        # that state so the next tick publishes a clean, unblended plan from
        # the current boundary frame.
        self.create_subscription(Empty, '/fsae/planning/reset_map', self._reset_cb, 10)

        self.pub_traj = self.create_publisher(PoseArray, '/fsae/planning/selected_trajectory', 10)

        # Debug: expose the wall mesh and candidate midpoints that build_path_walls
        # computes but otherwise keeps to itself.  Off by default — this is pure
        # instrumentation for the external path analyser.
        self.declare_parameter('debug_viz', False)
        self._debug_viz = self.get_parameter('debug_viz').get_parameter_value().bool_value
        self._dbg_pubs: dict = {}
        if self._debug_viz:
            self._dbg_pubs = {
                'midpoints':    self.create_publisher(PoseArray, '/fsae/planning/debug/midpoints', 10),
                'blue_walls':   self.create_publisher(PoseArray, '/fsae/planning/debug/blue_walls', 10),
                'yellow_walls': self.create_publisher(PoseArray, '/fsae/planning/debug/yellow_walls', 10),
            }
            self.get_logger().info('debug_viz on — publishing /fsae/planning/debug/*')

        # Latest boundary frame only — no persistent accumulation across ticks.
        # A stale or misassociated cone from an earlier frame would otherwise
        # sit in the map forever (a real problem once perception is a noisy
        # real SLAM stack rather than sim ground truth). The planner
        # re-derives the path each tick purely from whatever
        # /fsae/slam/*_track publishes right now.
        self._blue_cones:   np.ndarray = np.empty((0, 2))
        self._yellow_cones: np.ndarray = np.empty((0, 2))
        self._car_pos   = np.zeros(2)
        self._car_yaw   = 0.0
        self._have_pose = False
        self._centreline: np.ndarray | None = None
        self._prev_centreline: np.ndarray | None = None   # last published, for blending
        self._last_valid_centreline: np.ndarray | None = None   # for the bounded hold
        self._last_valid_time = None                            # rclpy.time.Time, set alongside it
        self._prev_path_dir: np.ndarray | None = None   # wrong-leg guard, see boundary.build_path_walls
        self._blue_segs:   list = []
        self._yellow_segs: list = []
        self._midpoints:   np.ndarray = np.empty((0, 2))

        # Local reception clocks — see the max_pose_age/max_track_age block
        # above for why arrival time and not message stamps.
        self._left_track_time = None
        self._right_track_time = None
        self._pose_time = None

        self.get_logger().info(f'{node_name} ready — waiting for car_position.')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _left_cb(self, msg: Track) -> None:
        self._blue_cones = cones_to_array(msg.cones, self.get_logger(), 'left_track')
        self._left_track_time = self.get_clock().now()

    def _right_cb(self, msg: Track) -> None:
        self._yellow_cones = cones_to_array(msg.cones, self.get_logger(), 'right_track')
        self._right_track_time = self.get_clock().now()

    def _pose_cb(self, msg: PoseStamped) -> None:
        # x,y in position; yaw (rad) is stuffed into orientation.w (upstream
        # convention). Converted immediately to plain numpy/float so the
        # planning core stays free of ROS message types.
        x, y, yaw = msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.w
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            # Don't let a NaN/Inf pose reach math.cos/np.linalg.norm downstream
            # (both the main corridor build and the exception fallback use car
            # yaw directly) -- treat it the same as no pose at all this tick.
            self.get_logger().warn(
                'pose_cb: rejected non-finite pose', throttle_duration_sec=2.0)
            return
        self._car_pos   = np.array([x, y])
        self._car_yaw   = float(yaw)
        self._pose_time = self.get_clock().now()
        self._have_pose = True
        self._planning_loop()

    def _reset_cb(self, _msg: Empty) -> None:
        """Drop blend/path state (debug/testing only)."""
        self._blue_cones   = np.empty((0, 2))
        self._yellow_cones = np.empty((0, 2))
        self._centreline   = None
        self._prev_centreline = None
        self._last_valid_centreline = None
        self._last_valid_time = None
        self._prev_path_dir = None
        self._blue_segs    = []
        self._yellow_segs  = []
        self._midpoints    = np.empty((0, 2))
        # Also drop pose/freshness state -- otherwise a stale-but-still-fresh
        # pose/track arrival time from before the reset could let the very next
        # tick silently rebuild a plan from pre-reset inputs instead of
        # genuinely starting clean.
        self._have_pose = False
        self._pose_time = None
        self._left_track_time = None
        self._right_track_time = None
        self.get_logger().info('planner state reset by external request')

    # ------------------------------------------------------------------
    # Input staleness
    # ------------------------------------------------------------------

    def _inputs_fresh(self) -> bool:
        now = self.get_clock().now()

        def age_ok(t, max_age: float) -> bool:
            if t is None:
                return False
            return (now - t).nanoseconds * 1e-9 <= max_age

        fresh = (
            age_ok(self._pose_time,            self._max_pose_age)
            and age_ok(self._left_track_time,  self._max_track_age)
            and age_ok(self._right_track_time, self._max_track_age)
        )
        if not fresh:
            return False

        # Both tracks can individually be "fresh enough" while having
        # arrived a full max_track_age apart (e.g. one side stalled a cycle),
        # which plans a centreline from a left/right pair that never actually
        # coexisted as a map snapshot. Reject that skew explicitly.
        skew = abs((self._left_track_time - self._right_track_time).nanoseconds) * 1e-9
        if skew > self._max_track_skew:
            self.get_logger().warn(
                f'left/right track update skew {skew:.3f}s > max_track_skew={self._max_track_skew} '
                '— no plan this tick',
                throttle_duration_sec=2.0,
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Output sanitisation
    # ------------------------------------------------------------------

    def _sanitize(self, path: np.ndarray) -> np.ndarray | None:
        """
        Returns a safe-to-publish path (possibly truncated), or None if
        nothing of it is usable.

        Finite/jump/initial-heading are whole-path shape checks — a failure
        there means the path itself is malformed, not that part of it is
        fine, so they still reject outright. A wall crossing is different: it
        only means the path becomes untrustworthy from that point on (the
        walk stepped through a real or phantom barrier), so it truncates at
        the first crossing instead of discarding the whole path — previously
        a single bad segment out near the horizon blanked the entire
        trajectory, including the good near-field part directly in front of
        the car.
        """
        pts = np.asarray(path, dtype=np.float64)
        if len(pts) < 2 or not np.all(np.isfinite(pts)):
            self.get_logger().warn(
                'sanitize: rejected (path too short or non-finite)', throttle_duration_sec=2.0)
            return None

        seg_lens = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if np.any(seg_lens > self._max_point_jump):
            self.get_logger().warn(
                f'sanitize: rejected (point-to-point jump > max_point_jump={self._max_point_jump})',
                throttle_duration_sec=2.0)
            return None

        # Initial direction agrees with the vehicle heading within the SAME
        # per-step turn allowance the wall-walk itself uses (_WALL_MAX_TURN_COS,
        # ~120 deg) rather than a stricter 90 deg cutoff -- the walk can accept
        # a first hop that sharp, so rejecting it here again would blank a
        # path the walk itself considered valid.
        d = pts[1] - pts[0]
        if float(np.linalg.norm(d)) < 1e-6 and len(pts) > 2:
            d = pts[-1] - pts[0]
        heading = np.array([math.cos(self._car_yaw), math.sin(self._car_yaw)])
        d_norm = float(np.linalg.norm(d))
        if d_norm > 1e-6 and float(d @ heading) / d_norm < _WALL_MAX_TURN_COS:
            self.get_logger().warn(
                'sanitize: rejected (initial heading disagrees with car heading)',
                throttle_duration_sec=2.0)
            return None

        # Two independent "becomes untrustworthy from here on" checks, same
        # truncate-not-discard treatment as each other: a wall crossing
        # (segment_crosses_walls only sees the reconstructed WALL MESH
        # between cones -- a path that passes directly through/beside a
        # single cone without its segment crossing any wall edge slips past
        # it entirely) and a cone-clearance violation (catches exactly that
        # case: the car's reference-point path coming within
        # min_cone_clearance of any known cone, approximating vehicle
        # footprint + tracking error + localisation uncertainty). Whichever
        # happens first wins -- both are evaluated and the path is cut at the
        # earliest bad index so neither check can be bypassed by the other
        # firing later.
        cut_idx = len(pts)   # exclusive upper bound; len(pts) = "no cut"

        walls = self._blue_segs + self._yellow_segs
        if walls:
            for i in range(len(pts) - 1):
                if segment_crosses_walls(pts[i], pts[i + 1], walls):
                    cut_idx = min(cut_idx, i + 1)
                    break

        if self._min_cone_clearance > 0.0:
            cones = np.vstack([self._blue_cones, self._yellow_cones]) \
                if (len(self._blue_cones) + len(self._yellow_cones)) > 0 else None
            if cones is not None and len(cones) > 0:
                d = np.linalg.norm(pts[:, None, :] - cones[None, :, :], axis=2)
                too_close = np.where(d.min(axis=1) < self._min_cone_clearance)[0]
                if len(too_close) > 0:
                    cut_idx = min(cut_idx, int(too_close[0]))

        if cut_idx < len(pts):
            if cut_idx < 2:
                self.get_logger().warn(
                    'sanitize: rejected (wall/cone-clearance violation at the very first segment)',
                    throttle_duration_sec=2.0)
                return None
            self.get_logger().info(
                f'sanitize: truncated at wall/cone-clearance violation ({cut_idx}/{len(pts)} pts kept)',
                throttle_duration_sec=2.0)
            pts = pts[:cut_idx]

        # A short-but-technically-valid stub gives the controller no real
        # stopping distance or shape to judge curvature from -- reject it
        # outright rather than publish/hold something driveable in name only.
        usable_len = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if usable_len < self._min_usable_path_length:
            self.get_logger().warn(
                f'sanitize: rejected (usable path length {usable_len:.2f}m < '
                f'min_usable_path_length={self._min_usable_path_length})',
                throttle_duration_sec=2.0)
            return None

        return pts

    # ------------------------------------------------------------------
    # Planning loop (template — subclasses override the hooks)
    # ------------------------------------------------------------------

    def _planning_loop(self) -> None:
        """
        Pose-driven tick: compute (if inputs are fresh) -> sanitise -> blend
        -> hold/expire -> publish. Wrapped so an unexpected exception
        ANYWHERE in the tick -- sanitisation, blending, publishing, debug --
        drops to a clean no-plan state instead of leaving stale state in
        place or killing the node (previously only _compute_path's own
        geometry call was guarded; a failure in blend_paths/publishing was
        not).
        """
        if not self._have_pose:
            return
        try:
            self._run_planning_tick()
        except Exception as exc:
            self._fail_safe(exc)

    def _fail_safe(self, exc: Exception) -> None:
        self.get_logger().error(
            f'planning tick failed unexpectedly ({exc!r}) — no trajectory this tick',
            throttle_duration_sec=2.0,
        )
        self._centreline = None
        self._prev_centreline = None
        self._prev_path_dir = None

    def _run_planning_tick(self) -> None:
        if self._inputs_fresh():
            self._compute_path()
            if not self._inputs_fresh():
                # An input aged past its limit WHILE _compute_path was
                # running (see _SLOW_TICK_WARN_SEC) -- the plan we just built
                # was computed from a snapshot that is no longer considered
                # fresh enough to publish/cache, so discard it rather than
                # trusting a result whose input freshness we can no longer
                # vouch for.
                self.get_logger().warn(
                    'inputs went stale during planning — discarding this tick\'s result',
                    throttle_duration_sec=2.0,
                )
                self._centreline = None
        else:
            self._centreline = None
            self.get_logger().warn(
                'stale input (pose/track age/skew exceeded max_pose_age/max_track_age/'
                'max_track_skew) — no plan this tick',
                throttle_duration_sec=2.0,
            )

        sanitized = None
        if self._centreline is not None and len(self._centreline) >= 2:
            sanitized = self._sanitize(self._centreline)

        self._finalize(sanitized)

    def _finalize(self, sanitized: np.ndarray | None) -> None:
        """Hold/expire/publish tail: blend, re-validate, or fall back to the hold."""
        now = self.get_clock().now()

        if sanitized is not None:
            # Temporally blend the fresh path with the last one so the published
            # trajectory eases between frames instead of jumping (which the
            # controller would track as a steering jerk).  Recursive EMA in the map
            # frame; resets itself when the path genuinely diverges (see blend_paths).
            blended = blend_paths(
                self._prev_centreline, sanitized, self._car_pos,
                alpha=self._path_blend, horizon=self._plan_horizon,
            )
            # Re-validate AFTER blending, not just before it: two individually
            # valid paths can blend into one that crosses a wall in a
            # nonconvex corridor (sanitising only the fresh path never sees
            # the blended result at all). Fall back to the validated fresh
            # path if the blend itself doesn't pass.
            checked = self._sanitize(blended) if blended is not None and len(blended) >= 2 else None
            final = checked if checked is not None else sanitized

            self._centreline = final
            self._prev_centreline = final
            self._last_valid_centreline = final
            self._last_valid_time = now
            self._prev_path_dir = _direction_from_path(final, self._car_pos)
        elif (
            self._last_valid_centreline is not None
            and self._last_valid_time is not None
            and (now - self._last_valid_time).nanoseconds * 1e-9 < self._path_hold_timeout
            and self._hold_reachable(self._last_valid_centreline)
        ):
            # This tick's frame was too thin to plan from (see module docstring)
            # but the gap is still within the bounded hold -- republish the last
            # known-good path instead of going silent, once _hold_reachable
            # confirms the car hasn't drifted away from it and it doesn't now
            # cross an updated wall mesh. _last_valid_time is left untouched so
            # the hold has a hard deadline from the last GOOD plan, rather than
            # resetting on every held tick and holding forever.
            self._centreline = self._last_valid_centreline
            self.get_logger().info(
                'holding last path (no fresh valid plan this tick)',
                throttle_duration_sec=1.0,
            )
        else:
            self._centreline = None
            self._prev_centreline = None
            self._prev_path_dir = None

        self._publish_trajectory()
        self._publish_debug()

        if self._centreline is None:
            self.get_logger().warn(
                'No usable plan — no trajectory published',
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().info(
                f'trajectory published: {len(self._centreline)} pts  '
                f'car=({self._car_pos[0]:.1f},{self._car_pos[1]:.1f})',
                throttle_duration_sec=1.0,
            )

    def _hold_reachable(self, path: np.ndarray) -> bool:
        """
        Before republishing a held (previous-tick) path unchanged, check that
        the car hasn't drifted away from it (localisation jump, or simply
        enough real motion that the cached path no longer describes where
        the car is) and that it doesn't now cross the latest wall mesh
        (newly observed cones since the path was cached).

        A held path was already validated in full when it was first computed
        (it's exactly a previous `final` from _finalize), so this only needs
        to re-check what could have changed SINCE then: the car's position
        and the wall mesh -- not initial heading or point-jump shape, which
        the path's own construction already satisfied.
        """
        pts = np.asarray(path, dtype=np.float64)
        if len(pts) < 2:
            return False

        seg_a, seg_b = pts[:-1], pts[1:]
        ab = seg_b - seg_a
        ab2 = np.einsum('ij,ij->i', ab, ab)
        ab2_safe = np.where(ab2 < 1e-12, 1.0, ab2)
        t = np.clip(np.einsum('ij,ij->i', self._car_pos - seg_a, ab) / ab2_safe, 0.0, 1.0)
        proj = seg_a + t[:, None] * ab
        dist_to_path = float(np.min(np.linalg.norm(proj - self._car_pos, axis=1)))
        if dist_to_path > self._max_hold_deviation:
            self.get_logger().warn(
                f'hold rejected: car is {dist_to_path:.2f}m from the held path '
                f'(> max_hold_deviation={self._max_hold_deviation})',
                throttle_duration_sec=1.0,
            )
            return False

        walls = self._blue_segs + self._yellow_segs
        if walls:
            for i in range(len(pts) - 1):
                if segment_crosses_walls(pts[i], pts[i + 1], walls):
                    self.get_logger().warn(
                        'hold rejected: held path crosses the current wall mesh',
                        throttle_duration_sec=1.0,
                    )
                    return False

        return True

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    def _compute_path(self) -> None:
        """Cone-wall mesh planner over the latest boundary-frame cones."""
        t0 = time.perf_counter()
        try:
            self._centreline, self._blue_segs, self._yellow_segs, self._midpoints = \
                build_path_walls(
                    self._blue_cones, self._yellow_cones,
                    self._car_pos, self._car_yaw,
                    smooth_per_pt=self._smooth_per_pt,
                    look_radius=self._look_radius,
                    plan_horizon=self._plan_horizon,
                    prev_dir=self._prev_path_dir,
                    midpoint_method=self._midpoint_method,
                    mid_min_gate=self._mid_min_gate,
                    mid_max_gate=self._mid_max_gate,
                )
            dt = time.perf_counter() - t0
            if dt > _SLOW_TICK_WARN_SEC:
                n_blue, n_yellow = len(self._blue_cones), len(self._yellow_cones)
                self.get_logger().warn(
                    f'slow planning tick: {dt * 1000:.1f} ms (blue={n_blue}, yellow={n_yellow} '
                    f'cones, {len(self._blue_segs) + len(self._yellow_segs)} wall segments) -- '
                    'this blocks every other callback on this node for its duration (see '
                    '_SLOW_TICK_WARN_SEC)',
                )
        except Exception as exc:
            self.get_logger().warn(
                f'Wall-barrier planner failed ({exc!r}), falling back to simple pairing',
                throttle_duration_sec=5.0,
            )
            self._centreline = build_local_path(
                self._blue_cones, self._yellow_cones,
                self._car_pos, self._car_yaw,
                max_pair_dist=self._fallback_max_pair_dist,
            )
            # Rebuild the wall mesh independently rather than clearing it: the
            # exception above was raised by the corridor/midpoint build, not
            # by the wall-mesh build, so real wall geometry is very likely
            # still obtainable here -- clearing blue_segs/yellow_segs would
            # silently disable EVERY collision check in _sanitize for this
            # fallback path (wall-crossing truncation has nothing to check
            # against), accepting a fallback plan under weaker scrutiny than
            # the primary planner instead of equivalent validation.
            try:
                self._blue_segs, self._yellow_segs = build_wall_segments_delaunay(
                    self._blue_cones, self._yellow_cones,
                )
            except Exception:
                self._blue_segs, self._yellow_segs = [], []
            self._midpoints = np.empty((0, 2))

    # ------------------------------------------------------------------
    # Publishing / visualisation
    # ------------------------------------------------------------------

    def _publish_trajectory(self) -> None:
        """
        Publish the current centreline, or nothing at all when there isn't one
        (see the module docstring on why this node stays silent rather than
        publishing an explicit empty PoseArray like the real stack's).

        Each waypoint's orientation is set from the local path heading
        (forward difference to the next point; the last point reuses the
        previous segment's heading) rather than left at the message
        default -- geometry_msgs/Quaternion defaults to (0,0,0,0), a
        zero-norm quaternion that is not a valid orientation (RViz/tf reject
        it). stanley_controller does not currently read this orientation (it
        derives its own target yaw from consecutive path points), so this is
        purely a correctness fix for the message contract and other
        consumers, not a behaviour change to the control loop.
        """
        if self._centreline is None or len(self._centreline) == 0:
            return
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        pts = self._centreline
        n = len(pts)
        for i, wp in enumerate(pts):
            pose = Pose()
            pose.position.x = float(wp[0])
            pose.position.y = float(wp[1])
            # Forward difference to the next point; the last point reuses
            # the incoming segment's direction (there is no "next").
            if i + 1 < n:
                dx, dy = float(pts[i + 1][0] - wp[0]), float(pts[i + 1][1] - wp[1])
            elif i > 0:
                dx, dy = float(wp[0] - pts[i - 1][0]), float(wp[1] - pts[i - 1][1])
            else:
                dx, dy = 0.0, 0.0   # single-point path -- no direction to infer
            yaw = math.atan2(dy, dx) if (abs(dx) > 1e-9 or abs(dy) > 1e-9) else 0.0
            pose.orientation.z = math.sin(yaw * 0.5)
            pose.orientation.w = math.cos(yaw * 0.5)
            msg.poses.append(pose)
        self.pub_traj.publish(msg)

    def _pose_array(self, points) -> PoseArray:
        """(N, 2) points -> PoseArray in the map frame (positions only)."""
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for pt in points:
            pose = Pose()
            pose.position.x = float(pt[0])
            pose.position.y = float(pt[1])
            pose.orientation.w = 1.0   # RViz rejects a 0-norm quaternion
            msg.poses.append(pose)
        return msg

    def _publish_debug(self) -> None:
        """
        Publish the wall mesh and candidate midpoints.

        Wall segments go out as a flat PoseArray of consecutive endpoint PAIRS
        (poses 0-1 are one segment, 2-3 the next, ...).  Reusing PoseArray keeps this
        patch free of new interface definitions and rebuilds of fsae_interfaces.
        """
        if not self._dbg_pubs:
            return
        self._dbg_pubs['midpoints'].publish(self._pose_array(self._midpoints))
        for key, segs in (('blue_walls', self._blue_segs), ('yellow_walls', self._yellow_segs)):
            flat = [pt for seg in segs for pt in seg]
            self._dbg_pubs[key].publish(self._pose_array(flat))


def main(args=None):
    rclpy.init(args=args)
    node = CenterlinePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
