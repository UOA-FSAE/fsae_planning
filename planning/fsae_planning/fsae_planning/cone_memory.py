#!/usr/bin/env python3
"""
cone_memory — a longer-lived local cone map, in its own process.

cone_mapper_shim (the kart's planner input) forgets a cone persistence_sec
(1.0 s) after its last detection. In a corner that is exactly the wrong
cone to lose: the inner apex cones leave the camera's ~110 deg view as the
car yaws, and are dropped a second later while they are still right beside
the car and are the only evidence of the inner boundary. And when the
camera's both-colours gate starves the shim, every cone ages out together.

This node keeps cones longer, without touching the shim or the SLAM mapper
(cone_landmark_mapper): it reads the same detection frames and publishes its
own topics. The planner reads them only when its inputs are remapped there
(planning.launch.py's planner_cone_source:=memory), so it can be switched in
and out, and later dropped next to the shim on the kart unchanged.

Store (per colour), same merge as the shim: a detection within match_radius
of an entry refreshes it, its published position is the mean of its last
filter_window detections; otherwise it starts a new entry. On top of that:
  * min_hits: an entry is published only once seen in this many frames, so
    a one-frame phantom never reaches the planner;
  * kept while it is recent (seen within persistence_sec, like the shim), OR
    within keep_radius of the car and seen within max_age. So cones beside
    and just behind the car survive leaving the camera's view, and anything
    left behind on the track, or not seen for max_age, is forgotten;
  * missed when it should have been seen: an entry inside the camera's view
    (view_hfov_deg, view_range, less a small edge margin) on a frame that
    did not detect it gains a miss; max_misses consecutive misses drop it
    (a phantom, or a cone that is not there any more). A frame that never
    arrives (the both-colours gate) counts nothing;
  * a car pose jump of more than reset_jump between consecutive poses
    (localisation reset / teleport) clears the whole store.

Interface:
    in   /cone_detection_fused    fsae_interfaces/ConeDetection   (the shim's input; in the
                                  sim, remapped to /fsae/perception/cone_detection)
    in   /fsae/slam/car_position  geometry_msgs/PoseStamped       yaw in orientation.w
    out  /fsae/cone_memory/left_track          fsae_interfaces/Track  blue,   global frame
    out  /fsae/cone_memory/right_track         fsae_interfaces/Track  yellow, global frame
    out  /fsae/cone_memory/big_orange_track    fsae_interfaces/Track
    out  /fsae/cone_memory/small_orange_track  fsae_interfaces/Track
Published on a fixed timer (publish_rate), like the shim, so the planner's
max_track_age never sees a gap even while detections stop.
"""
import math
from collections import deque

from fsae_interfaces.msg import ConeDetection, Track
from geometry_msgs.msg import Point, PoseStamped
import rclpy
from rclpy.node import Node

_COLOURS = ('blue', 'yellow', 'big_orange', 'small_orange')

# view test margins: a cone this close to the edge of the camera's view
# (deg of bearing, m of range) is not counted as "should have been seen" --
# detections there are unreliable, and a cone that drifts out of view as the
# car yaws would otherwise collect misses it doesn't deserve.
_VIEW_EDGE_DEG = 5.0
_VIEW_EDGE_M = 1.0
_VIEW_MIN_X = 0.5      # m ahead of the car origin


class _Entry:
    __slots__ = ('x', 'y', 'hist', 'last_seen', 'hits', 'misses')

    def __init__(self, x: float, y: float, now: float, window: int):
        self.x, self.y = x, y
        self.hist = deque([(x, y)], maxlen=window)
        self.last_seen = now
        self.hits = 1
        self.misses = 0


class ConeMemoryStore:
    """The node's state and rules, free of ROS so it can be tested directly."""

    def __init__(self, match_radius=0.8, filter_window=5, persistence_sec=1.0,
                 keep_radius=10.0, max_age=30.0, min_hits=2, view_hfov_deg=110.0,
                 view_range=16.0, max_misses=3, reset_jump=5.0):
        self.match_radius = match_radius
        self.filter_window = max(1, int(filter_window))
        self.persistence_sec = persistence_sec
        self.keep_radius = keep_radius
        self.max_age = max_age
        self.min_hits = max(1, int(min_hits))
        self.view_half = math.radians(0.5 * view_hfov_deg - _VIEW_EDGE_DEG)
        self.view_range = view_range - _VIEW_EDGE_M
        self.max_misses = max(1, int(max_misses))
        self.reset_jump = reset_jump
        self.store = {c: [] for c in _COLOURS}
        self.car = None          # (x, y, yaw) of the latest pose
        self.resets = 0

    def clear(self) -> None:
        self.store = {c: [] for c in _COLOURS}

    def update_pose(self, x: float, y: float, yaw: float) -> bool:
        """Track the car. Returns True if a pose jump cleared the store."""
        jumped = (self.car is not None
                  and math.hypot(x - self.car[0], y - self.car[1]) > self.reset_jump)
        self.car = (x, y, yaw)
        if jumped:
            self.clear()
            self.resets += 1
        return jumped

    def _in_view(self, e: _Entry, pose) -> bool:
        px, py, yaw = pose
        dx, dy = e.x - px, e.y - py
        fx = dx * math.cos(yaw) + dy * math.sin(yaw)
        fy = -dx * math.sin(yaw) + dy * math.cos(yaw)
        return (fx > _VIEW_MIN_X and math.hypot(fx, fy) <= self.view_range
                and abs(math.atan2(fy, fx)) <= self.view_half)

    def add_frame(self, frame: dict, pose, now: float) -> None:
        """
        Merge one detection frame and count misses for entries it missed.

        frame maps colour -> list of global (x, y); pose is the (x, y, yaw)
        the frame was taken from.
        """
        for colour in _COLOURS:
            entries = self.store[colour]
            matched = set()
            for x, y in frame.get(colour, ()):
                best, best_d = None, self.match_radius
                for e in entries:
                    d = math.hypot(e.x - x, e.y - y)
                    if d < best_d:
                        best, best_d = e, d
                if best is None:
                    best = _Entry(x, y, now, self.filter_window)
                    entries.append(best)
                else:
                    best.hist.append((x, y))
                    best.x = sum(p[0] for p in best.hist) / len(best.hist)
                    best.y = sum(p[1] for p in best.hist) / len(best.hist)
                    best.last_seen = now
                    best.hits += 1
                best.misses = 0
                matched.add(id(best))
            kept = []
            for e in entries:
                if id(e) not in matched and self._in_view(e, pose):
                    e.misses += 1
                    if e.misses >= self.max_misses:
                        continue
                kept.append(e)
            self.store[colour] = kept

    def expire(self, now: float) -> None:
        for colour in _COLOURS:
            self.store[colour] = [e for e in self.store[colour] if self._keep(e, now)]

    def _keep(self, e: _Entry, now: float) -> bool:
        age = now - e.last_seen
        if age <= self.persistence_sec:
            return True
        if age > self.max_age or self.car is None:
            return False
        return math.hypot(e.x - self.car[0], e.y - self.car[1]) <= self.keep_radius

    def published(self, colour: str) -> list:
        return [(e.x, e.y) for e in self.store[colour] if e.hits >= self.min_hits]


def _to_global(points, car_pose) -> list:
    """Car-frame Points -> global (x, y), yaw in orientation.w (as the shim)."""
    theta = car_pose.orientation.w
    c, s = math.cos(theta), math.sin(theta)
    ox, oy = car_pose.position.x, car_pose.position.y
    return [(ox + p.x * c - p.y * s, oy + p.x * s + p.y * c) for p in points]


def _to_track(points) -> Track:
    track = Track()
    track.cones = [Point(x=float(x), y=float(y), z=0.0) for x, y in points]
    return track


class ConeMemory(Node):
    def __init__(self):
        super().__init__('cone_memory')
        self.declare_parameter('invert_cones', False)
        self.declare_parameter('match_radius', 0.8)       # m, same as the shim
        self.declare_parameter('filter_window', 5)        # detections averaged per cone, as the shim
        self.declare_parameter('persistence_sec', 1.0)    # s, always kept this long (the shim's whole life)
        self.declare_parameter('keep_radius', 10.0)       # m, then kept while this close to the car...
        self.declare_parameter('max_age', 30.0)           # s, ...up to this long since last seen
        self.declare_parameter('min_hits', 2)             # frames before an entry is published
        self.declare_parameter('view_hfov_deg', 110.0)    # the camera's view, for "should have seen it"
        self.declare_parameter('view_range', 16.0)        # m
        self.declare_parameter('max_misses', 3)           # consecutive in-view misses before dropping
        self.declare_parameter('reset_jump', 5.0)         # m, pose jump that clears the store
        self.declare_parameter('publish_rate', 20.0)      # Hz, fixed expiry/publish tick

        def p(name):
            return self.get_parameter(name).value

        self._invert = bool(p('invert_cones'))
        self._mem = ConeMemoryStore(
            match_radius=float(p('match_radius')), filter_window=int(p('filter_window')),
            persistence_sec=float(p('persistence_sec')), keep_radius=float(p('keep_radius')),
            max_age=float(p('max_age')), min_hits=int(p('min_hits')),
            view_hfov_deg=float(p('view_hfov_deg')), view_range=float(p('view_range')),
            max_misses=int(p('max_misses')), reset_jump=float(p('reset_jump')),
        )
        rate = float(p('publish_rate'))
        if not rate > 0.0:
            self.get_logger().warn(f'publish_rate must be > 0 (got {rate}); using 20.0 Hz')
            rate = 20.0

        self.create_subscription(ConeDetection, '/cone_detection_fused', self._det_cb, 10)
        self.create_subscription(PoseStamped, '/fsae/slam/car_position', self._pose_cb, 10)
        self._pubs = {c: self.create_publisher(Track, f'/fsae/cone_memory/{t}', 10)
                      for c, t in (('blue', 'left_track'), ('yellow', 'right_track'),
                                   ('big_orange', 'big_orange_track'),
                                   ('small_orange', 'small_orange_track'))}
        self.create_timer(1.0 / rate, self._tick)
        m = self._mem
        self.get_logger().info(
            f'cone_memory ready: keep {m.persistence_sec:g} s, then within {m.keep_radius:g} m '
            f'up to {m.max_age:g} s; min_hits {m.min_hits}, drop after {m.max_misses} in-view misses '
            f'({math.degrees(2 * m.view_half) + 2 * _VIEW_EDGE_DEG:g} deg × '
            f'{m.view_range + _VIEW_EDGE_M:g} m view).')

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _pose_cb(self, msg: PoseStamped) -> None:
        x, y, yaw = msg.pose.position.x, msg.pose.position.y, msg.pose.orientation.w
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            return
        if self._mem.update_pose(x, y, yaw):
            self.get_logger().warn(
                f'car pose jumped > {self._mem.reset_jump:g} m -- cone memory cleared')

    def _det_cb(self, msg: ConeDetection) -> None:
        pose = msg.car_pose
        frame = {c: _to_global(getattr(msg, c), pose) for c in _COLOURS}
        self._mem.add_frame(frame, (pose.position.x, pose.position.y, pose.orientation.w), self._now())

    def _tick(self) -> None:
        """Fixed-rate expiry + publish (empty tracks included), as the shim's _tick."""
        self._mem.expire(self._now())
        left, right = ('yellow', 'blue') if self._invert else ('blue', 'yellow')
        self._pubs['blue'].publish(_to_track(self._mem.published(left)))
        self._pubs['yellow'].publish(_to_track(self._mem.published(right)))
        self._pubs['big_orange'].publish(_to_track(self._mem.published('big_orange')))
        self._pubs['small_orange'].publish(_to_track(self._mem.published('small_orange')))


def main(args=None):
    rclpy.init(args=args)
    node = ConeMemory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
