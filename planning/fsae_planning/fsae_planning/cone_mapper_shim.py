#!/usr/bin/env python3
"""
cone_mapper_shim — temporary, minimal stand-in for perception's cone_mapper.

TEMPORARY BRIDGE: perception is replacing fsae_slam's Python cone_mapper.py
with a C++ GTSAM factor-graph mapper (cone_landmark_mapper.cpp), still
Stage 1 and not publishing yet (see fsae_slam/CMakeLists.txt). Until that
lands, this node keeps /fsae/slam/left_track and /fsae/slam/right_track
alive so wall_centerline_planner doesn't need to change or wait.

Reads /cone_detection_fused (LiDAR+camera fusion), not the raw ZED-only
/fsae/perception/cone_detection. Same ConeDetection message shape, so this
is a straight topic swap.

Persistence model — kept deliberately dumb:
  Each seen cone is held in a plain list as (x, y, last_seen_time) and
  re-published as long as it was seen within `persistence_sec`. A new
  detection within `match_radius` of an existing entry refreshes that
  entry's position and timestamp; otherwise it's appended as new. Expiry
  and publishing run on a fixed timer, NOT in the callback (see below).
  That's it — no landmark IDs, no
  Kalman filter, no external data structure (just a Python list, linearly
  scanned). This is a direct reaction to the old cone_mapper.py
  occasionally publishing phantom cones (suspected stale/mismatched state
  in its KD-tree); the fix here is a cache simple enough to read in one
  sitting, with every entry on a hard, bounded expiry.

  This is real persistence, not a full session map: gaps of more than
  persistence_sec still drop cones, and it does not replace an accumulated
  full-track map. It exists so a few dropped detection frames don't blank
  out left_track/right_track and collapse the planner's path.

Why publishing is on a timer, not the callback:
  This node used to expire and publish inside `_cb`, which made both
  depend on input arriving. When /cone_detection_fused went quiet — the
  detection gate in cone_detection.cpp only fires on frames holding BOTH a
  blue and a yellow cone, see docs/MIGRATION_BUGS.md — the shim stopped
  publishing entirely. Two consequences, both observed on the car:
    1. left_track/right_track went stale rather than empty, so
       wall_centerline_planner aged them out against max_track_age (0.5s)
       and fell back to its hold-then-explicit-empty path, blanking the
       trajectory in RViz.
    2. On the FIRST message after a gap, the whole backlog of expiry was
       applied in one step — every cone not re-seen within persistence_sec
       vanished at once, collapsing a full track to a single frame's worth
       of cones and then regrowing. That's the non-smooth snap.
  Running expiry + publish on a fixed timer fixes both: cones age out
  continuously whether or not input is flowing, and downstream always sees
  a fresh message. The tracks still empty when detections genuinely stop —
  they just decay there over persistence_sec instead of teleporting.

Interface:
    in   /cone_detection_fused    fsae_interfaces/ConeDetection
    out  /fsae/slam/left_track    fsae_interfaces/Track   blue/left,  global frame
    out  /fsae/slam/right_track   fsae_interfaces/Track   yellow/right, global frame
    out  /fsae/slam/big_orange_track    fsae_interfaces/Track  big orange (start/finish line), global frame
    out  /fsae/slam/small_orange_track  fsae_interfaces/Track  small orange, global frame

Orange cones get the same store/merge/expiry as blue and yellow, on their own
topics: they mark the start line rather than an edge, so they never join
left/right_track, and invert_cones doesn't touch them (wall_centerline_planner
works out which side each one is on). The camera puts every non-blue/yellow
label in big_orange, so small_orange_track is normally empty on the car.

/fsae/slam/car_position is untouched by any of this — published directly by
fsae_camera's cone_detection_node, not by cone_mapper.

Delete this file once cone_landmark_mapper.cpp is finished and publishing,
and point slam.launch.py back at it.
"""
import math
from collections import deque

from fsae_interfaces.msg import ConeDetection, Track
from geometry_msgs.msg import Point, Pose
import rclpy
from rclpy.node import Node


def _to_global(points, car_pose: Pose) -> list:
    """Camera-body-frame Points -> list of (x, y) global-frame tuples.

    Same (x, y, yaw-in-orientation.w) convention used everywhere else in the
    real stack — orientation.w carries yaw directly, it is not a real
    quaternion component.
    """
    theta = car_pose.orientation.w
    c, s = math.cos(theta), math.sin(theta)
    ox, oy = car_pose.position.x, car_pose.position.y
    return [(ox + p.x * c - p.y * s, oy + p.x * s + p.y * c) for p in points]


def _merge(store: list, new_points: list, now: float, match_radius: float, filter_window: int) -> None:
    """In-place: refresh matched entries' position+time, append unmatched ones.

    store entries are [x, y, last_seen_time, history] lists (mutable in place).
    Published (x, y) is a moving average over `history` (a bounded deque of raw
    detections) rather than the latest raw hit, to damp per-frame position
    jitter/shifting on an otherwise-stationary cone.
    """
    for x, y in new_points:
        best, best_d = None, match_radius
        for entry in store:
            d = math.hypot(entry[0] - x, entry[1] - y)
            if d < best_d:
                best, best_d = entry, d
        if best is not None:
            best[3].append((x, y))
            best[0] = sum(p[0] for p in best[3]) / len(best[3])
            best[1] = sum(p[1] for p in best[3]) / len(best[3])
            best[2] = now
        else:
            store.append([x, y, now, deque([(x, y)], maxlen=filter_window)])


def _to_track(store: list) -> Track:
    track = Track()
    track.cones = [Point(x=e[0], y=e[1], z=0.0) for e in store]
    return track


class ConeMapperShim(Node):
    def __init__(self):
        super().__init__('cone_mapper_shim')
        self.declare_parameter('invert_cones', False)
        self.declare_parameter('match_radius', 0.8)      # m, same-cone tolerance
        self.declare_parameter('persistence_sec', 0.5)   # how long a cone survives with no re-detection
        self.declare_parameter('filter_window', 5)       # detections averaged per cone (medium average filter)
        # Output rate. Decoupled from the input rate on purpose: expiry and publishing
        # run here, so they must keep running through input gaps (see module docstring).
        # Must stay comfortably above wall_centerline_planner's max_track_age (0.5s) or
        # the planner ages the tracks out between ticks; 20 Hz leaves 10x margin and is
        # cheap (the callback is a list scan over a handful of cones).
        self.declare_parameter('publish_rate', 20.0)     # Hz, fixed publish/expiry tick
        self._invert = self.get_parameter('invert_cones').get_parameter_value().bool_value
        self._match_radius = self.get_parameter('match_radius').get_parameter_value().double_value
        self._persistence_sec = self.get_parameter('persistence_sec').get_parameter_value().double_value
        self._filter_window = self.get_parameter('filter_window').get_parameter_value().integer_value
        self._publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value
        if not (self._publish_rate > 0.0):
            self.get_logger().warn(
                f"publish_rate must be > 0 (got {self._publish_rate}); falling back to 20.0 Hz")
            self._publish_rate = 20.0

        self._blue: list = []    # [x, y, last_seen_time, history] entries
        self._yellow: list = []
        self._big_orange: list = []
        self._small_orange: list = []

        self.create_subscription(ConeDetection, '/cone_detection_fused', self._cb, 10)
        self.left_pub = self.create_publisher(Track, '/fsae/slam/left_track', 10)
        self.right_pub = self.create_publisher(Track, '/fsae/slam/right_track', 10)
        self.big_orange_pub = self.create_publisher(Track, '/fsae/slam/big_orange_track', 10)
        self.small_orange_pub = self.create_publisher(Track, '/fsae/slam/small_orange_track', 10)
        self.create_timer(1.0 / self._publish_rate, self._tick)

        self.get_logger().warn(
            "cone_mapper_shim running — TEMPORARY, reads /cone_detection_fused, "
            f"{self._persistence_sec}s persistence, {self._publish_rate}Hz publish. "
            "Switch to cone_landmark_mapper "
            "once it publishes.")

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _cb(self, msg: ConeDetection) -> None:
        """Merge one detection frame into the store. Does NOT expire or publish.

        Kept to pure ingestion so the output rate is set by _tick alone and stays
        steady through input gaps — see the module docstring for why.
        """
        now = self._now()
        _merge(self._blue, _to_global(msg.blue, msg.car_pose), now, self._match_radius, self._filter_window)
        _merge(self._yellow, _to_global(msg.yellow, msg.car_pose), now, self._match_radius, self._filter_window)
        _merge(self._big_orange, _to_global(msg.big_orange, msg.car_pose), now,
               self._match_radius, self._filter_window)
        _merge(self._small_orange, _to_global(msg.small_orange, msg.car_pose), now,
               self._match_radius, self._filter_window)

    def _tick(self) -> None:
        """Fixed-rate expiry + publish. Runs whether or not detections are arriving.

        Publishing unconditionally (including empty tracks) is deliberate: downstream
        reads track freshness by arrival time, so a steady stream of honestly-empty
        tracks is strictly better than silence — it tells wall_centerline_planner
        'no cones', which it handles, instead of 'no data', which it has to age out.
        """
        cutoff = self._now() - self._persistence_sec
        self._blue[:] = [e for e in self._blue if e[2] >= cutoff]
        self._yellow[:] = [e for e in self._yellow if e[2] >= cutoff]
        self._big_orange[:] = [e for e in self._big_orange if e[2] >= cutoff]
        self._small_orange[:] = [e for e in self._small_orange if e[2] >= cutoff]

        left, right = (self._yellow, self._blue) if self._invert else (self._blue, self._yellow)
        self.left_pub.publish(_to_track(left))
        self.right_pub.publish(_to_track(right))
        self.big_orange_pub.publish(_to_track(self._big_orange))
        self.small_orange_pub.publish(_to_track(self._small_orange))


def main(args=None):
    rclpy.init(args=args)
    node = ConeMapperShim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == '__main__':
    main()
