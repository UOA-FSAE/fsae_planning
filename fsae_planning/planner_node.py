import csv
import math
import os

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from fs_msgs.msg import GoSignal, Track
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32

from fsae_planning.boundary import build_path_walls
from fsae_planning.cone_map import ConeMap
from fsae_planning.cone_sorting import separate_cones_by_color
from fsae_planning.localisation import (
    LoopClosureDetector,
    build_completed_path,
    compute_drift,
    roll_loop_to_car,
)
from fsae_planning.path_utils import (
    build_local_path,
    check_direction,
    compute_desired_speed,
    get_lookahead_waypoint,
)
from fsae_planning.skidpad import build_figure8, path_deviation
from fsae_planning.viz_utils import Visualizer

LOOKAHEAD_DIST = 4.0  # metres ahead for pure-pursuit target
V_MAX          = 15.0  # m/s — top speed on straights
V_MIN          = 1.5  # m/s — minimum speed through tight corners

# Master switch: when True the planner watches for a completed lap and, once the
# track loop closes, switches from the rolling sensor window to closed-loop
# planning on the full accumulated cone map.  Set False to disable entirely.
ENABLE_LOCAL_MODE = True
DRIFT_WARN_DIST   = 1.5  # m — mean perception-vs-map drift above this is warned

# --- Skidpad (figure-8) characterisation mode -----------------------------
# When True the planner ignores the lap/localisation logic above and instead
# reconstructs the figure-8 from the full broadcast track and PRECOMPUTES the
# centreline once, then laps it at a steadily rising speed until the car slides
# off the lane.  The speed is logged every second and the exact (time, speed)
# of departure is recorded — this is the data used to compare simulation against
# the real vehicle.  Set False for normal autocross / trackdrive planning.
#
# The whole track is taken from the latched oracle topic /fsds/testing_only/track
# (every cone, broadcast once) rather than the rolling perception window, so the
# path is ready before the car moves and is independent of where the car starts.
ENABLE_SKIDPAD_MODE = True

SKID_V_START      = 3.0    # m/s — speed at the start of the ramp
SKID_RAMP_ACCEL   = 0.25   # m/s per second — how fast the target speed rises
SKID_V_CAP        = 25.0   # m/s — hard ceiling on the ramp
SKID_AHEAD        = 18.0   # m — forward planning window along the figure-8
SKID_OFFTRACK_MARGIN = 0.5  # m — slop beyond the lane half-width before "off track"
SKID_MIN_CENTRE_SEP  = 8.0  # m — reject fits whose two circles are too close
SKID_LOG_PATH = os.path.expanduser('~/skidpad_speed_log.csv')


class PlannerNode(Node):
    def __init__(self):
        super().__init__('centreline_planner')

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(Track, '/FusionCones', self._track_cb, 10)
        self.create_subscription(Odometry, '/fsds/testing_only/odom', self._odom_cb, sensor_qos)
        self.create_subscription(GoSignal, '/fsds/signal/go', self._go_cb, 10)

        # Skidpad mode precomputes its path from the full broadcast track, which
        # the simulator latches once on /fsds/testing_only/track.  Matching the
        # TRANSIENT_LOCAL QoS lets a late-joining subscriber still receive it.
        if ENABLE_SKIDPAD_MODE:
            oracle_qos = QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.create_subscription(
                Track, '/fsds/testing_only/track', self._oracle_track_cb, oracle_qos
            )

        self.pub_path   = self.create_publisher(Path,         '/fsds/planned_path',     10)
        self.pub_target = self.create_publisher(PointStamped, '/fsds/lookahead_target', 10)
        self.pub_speed  = self.create_publisher(Float32,      '/fsds/desired_speed',    10)

        self._go_received = False
        self._cone_map    = ConeMap()          # accumulated historical cone map
        self._blue_cones:   np.ndarray = np.empty((0, 2))   # latest sensor frame (viz)
        self._yellow_cones: np.ndarray = np.empty((0, 2))
        self._car_pos = np.zeros(2)
        self._car_yaw = 0.0
        self._centreline: np.ndarray | None = None
        self._blue_segs:   list = []
        self._yellow_segs: list = []
        self._midpoints:   np.ndarray = np.empty((0, 2))

        # --- Localisation / closed-loop mode ---
        self._loop_detector = LoopClosureDetector()
        self._local_mode    = False
        self._global_loop: np.ndarray | None = None   # cached closed centreline
        self._drift = {'mean': 0.0, 'max': 0.0, 'n': 0}

        # --- Skidpad characterisation mode ---
        self._car_speed     = 0.0          # actual speed from odometry twist
        self._skid_track    = None         # cached Figure8Track once reconstructed
        self._skid_t0       = None         # rclpy.time.Time — ramp start
        self._skid_v_target = 0.0          # latest ramped target speed
        self._skid_dev      = 0.0          # latest deviation from the centreline
        self._skid_on_lane  = False        # car has reached the lane at least once
        self._skid_off_track = False       # spin-off recorded (terminal)
        self._skid_csv_ready = False

        self.create_timer(0.05, self._planning_loop)
        if ENABLE_SKIDPAD_MODE:
            self.create_timer(1.0, self._skid_log_loop)   # speed log @ 1 Hz

        self._viz = Visualizer()
        self.create_timer(1 / 3.0, self._viz_loop)

        mode = 'SKIDPAD characterisation' if ENABLE_SKIDPAD_MODE else 'normal'
        self.get_logger().info(
            f'Planner node ready ({mode} mode) — waiting for GO signal.'
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _go_cb(self, msg: GoSignal) -> None:
        if not self._go_received:
            self.get_logger().info(
                f'GO received: mission={msg.mission!r}, track={msg.track!r}'
            )
            self._go_received = True

    def _track_cb(self, msg: Track) -> None:
        blue, yellow = separate_cones_by_color(msg)
        self._blue_cones   = blue
        self._yellow_cones = yellow
        self._cone_map.update(blue, yellow)
        self.get_logger().info(
            f'Track update: {len(blue)} blue + {len(yellow)} yellow this frame  '
            f'| map: {len(self._cone_map.blue)}b + {len(self._cone_map.yellow)}y total',
            throttle_duration_sec=2.0,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._car_pos = np.array([p.x, p.y])

        v = msg.twist.twist.linear
        self._car_speed = math.hypot(v.x, v.y)

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._car_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # Planning loop
    # ------------------------------------------------------------------

    def _planning_loop(self) -> None:
        if not self._go_received:
            self.get_logger().info('Waiting for GO signal...', throttle_duration_sec=2.0)
            return

        if ENABLE_SKIDPAD_MODE:
            self._plan_skidpad()
            return

        self._update_localisation()

        if self._local_mode:
            self._plan_closed_loop()
        else:
            self._plan_local_window()
        self._publish_path()

        if self._centreline is None:
            self.get_logger().warn(
                'No forward cones visible — no target published',
                throttle_duration_sec=2.0,
            )
            return

        # In local mode the path is the known closed loop, already oriented in
        # the travel direction, so the live-frame direction guard is skipped.
        if not self._local_mode and not check_direction(
            self._car_pos, self._car_yaw,
            self._blue_cones, self._yellow_cones,   # use live frame for direction check
        ):
            self.get_logger().warn(
                'Direction check failed: blue not left / yellow not right',
                throttle_duration_sec=1.0,
            )
            return

        target = get_lookahead_waypoint(
            self._centreline, self._car_pos, self._car_yaw, LOOKAHEAD_DIST
        )

        if target is None:
            return

        msg = PointStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'fsds/map'
        msg.point.x = float(target[0])
        msg.point.y = float(target[1])
        msg.point.z = 0.0
        self.pub_target.publish(msg)

        v_target = compute_desired_speed(self._centreline, v_max=V_MAX, v_min=V_MIN)
        speed_msg = Float32()
        speed_msg.data = v_target
        self.pub_speed.publish(speed_msg)

        self.get_logger().info(
            f'Target: ({float(target[0]):.1f}, {float(target[1]):.1f})  '
            f'v_target={v_target:.2f} m/s  '
            f'car=({self._car_pos[0]:.1f},{self._car_pos[1]:.1f})',
            throttle_duration_sec=1.0,
        )

    # ------------------------------------------------------------------
    # Localisation / planning strategies
    # ------------------------------------------------------------------

    def _update_localisation(self) -> None:
        """Watch for loop closure; on the first close, cache the global loop."""
        if not ENABLE_LOCAL_MODE or self._local_mode:
            return

        closed = self._loop_detector.update(
            self._car_pos,
            self._blue_cones, self._yellow_cones,
        )
        if not closed:
            return

        loop = build_completed_path(self._loop_detector.trajectory)
        if loop is None:
            # Path reported closed but was too short to fit — stay in mapping
            # mode and retry on the next lap pass.
            self._loop_detector.reopen()
            return

        self._global_loop = loop
        self._local_mode  = True
        self.get_logger().info(
            f'PATH CLOSED — entering localisation mode '
            f'(completed loop: {len(loop)} pts from '
            f'{len(self._loop_detector.trajectory)} driven path points).'
        )

    def _plan_local_window(self) -> None:
        """Default mapping planner: cone-wall mesh over the rolling sensor window."""
        try:
            self._centreline, self._blue_segs, self._yellow_segs, self._midpoints = \
                build_path_walls(
                    self._cone_map.blue, self._cone_map.yellow,
                    self._car_pos, self._car_yaw,
                )
        except Exception as exc:
            self.get_logger().warn(
                f'Wall-barrier planner failed ({exc!r}), falling back to simple pairing',
                throttle_duration_sec=5.0,
            )
            self._centreline = build_local_path(
                self._cone_map.blue, self._cone_map.yellow,
                self._car_pos, self._car_yaw,
            )
            self._blue_segs   = []
            self._yellow_segs = []
            self._midpoints   = np.empty((0, 2))

    def _plan_closed_loop(self) -> None:
        """
        Localisation planner: plan on the cached closed loop built from the full
        cone map.  Perception is no longer used for the path — only to monitor
        drift between the live frame and the map the loop was built from.
        """
        self._drift = compute_drift(
            self._blue_cones, self._yellow_cones,
            self._cone_map.blue, self._cone_map.yellow,
        )
        if self._drift['mean'] > DRIFT_WARN_DIST:
            self.get_logger().warn(
                f'Perception drift vs map: mean={self._drift["mean"]:.2f} m '
                f'max={self._drift["max"]:.2f} m over {self._drift["n"]} cones',
                throttle_duration_sec=2.0,
            )

        self._centreline  = roll_loop_to_car(self._global_loop, self._car_pos, self._car_yaw)
        # Wall mesh / candidate midpoints are mapping-mode artefacts only.
        self._blue_segs   = []
        self._yellow_segs = []
        self._midpoints   = np.empty((0, 2))

    # ------------------------------------------------------------------
    # Skidpad characterisation
    # ------------------------------------------------------------------

    def _plan_skidpad(self) -> None:
        """
        Figure-8 characterisation planner.

        The figure-8 centreline is precomputed once from the full broadcast track
        (see _oracle_track_cb), so here it is only followed: a forward window is
        published together with a target speed that ramps linearly with time.  As
        the speed rises the car eventually exceeds lateral grip and slides off the
        lane; the moment it does is detected (deviation from the centreline) and
        the (time, speed) recorded.
        """
        # 1. The path is precomputed from the oracle track — wait for it.
        if self._skid_track is None:
            self.get_logger().info(
                'Skidpad: waiting for the broadcast track to precompute the figure-8...',
                throttle_duration_sec=2.0,
            )
            return

        # Start the speed ramp on the first cycle after GO (not when the track
        # was received, which may be well before the car is released).
        if self._skid_t0 is None:
            self._skid_t0 = self.get_clock().now()
            self._init_skid_csv()
            self.get_logger().info(
                f'Skidpad run starting — ramping from {SKID_V_START:.1f} m/s '
                f'at {SKID_RAMP_ACCEL:.2f} m/s².'
            )

        loop = self._skid_track.loop

        # 2. Forward planning window along the figure-8 (reuse the loop roller).
        self._centreline = roll_loop_to_car(
            loop, self._car_pos, self._car_yaw, ahead=SKID_AHEAD
        )
        self._blue_segs   = []
        self._yellow_segs = []
        self._midpoints   = np.empty((0, 2))
        self._publish_path()

        # 3. Track deviation → arm and watch for the spin-off.
        self._skid_dev = path_deviation(loop, self._car_pos)
        off_limit = self._skid_track.half_width + SKID_OFFTRACK_MARGIN
        if not self._skid_on_lane and self._skid_dev <= self._skid_track.half_width:
            self._skid_on_lane = True
        if (self._skid_on_lane and not self._skid_off_track
                and self._skid_dev > off_limit):
            self._record_spinoff()

        # 4. Ramped target speed (zero once the car has left the track).
        if self._skid_off_track:
            self._skid_v_target = 0.0
        else:
            elapsed = self._skid_elapsed()
            self._skid_v_target = min(
                SKID_V_CAP, SKID_V_START + SKID_RAMP_ACCEL * elapsed
            )

        speed_msg = Float32()
        speed_msg.data = float(self._skid_v_target)
        self.pub_speed.publish(speed_msg)

        # 5. Lookahead target (for visualisation / downstream consumers).
        target = get_lookahead_waypoint(
            self._centreline, self._car_pos, self._car_yaw, LOOKAHEAD_DIST
        )
        if target is not None:
            msg = PointStamped()
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'fsds/map'
            msg.point.x = float(target[0])
            msg.point.y = float(target[1])
            self.pub_target.publish(msg)

    def _oracle_track_cb(self, msg: Track) -> None:
        """
        Precompute the figure-8 path once from the full broadcast track.

        The simulator latches every cone on /fsds/testing_only/track, so a single
        message holds the whole map.  The figure-8 is reconstructed from it once
        and cached; the path is then ready before the car is released and does not
        depend on the car's start position (roll_loop_to_car orients it per frame).
        """
        if self._skid_track is not None:
            return

        blue, yellow = separate_cones_by_color(msg)
        track = build_figure8(blue, yellow)
        if track is None:
            self.get_logger().warn(
                f'Skidpad: could not fit a figure-8 to the broadcast track '
                f'({len(blue)} blue + {len(yellow)} yellow cones).'
            )
            return
        sep = float(np.linalg.norm(track.centres[0] - track.centres[1]))
        if sep < SKID_MIN_CENTRE_SEP:
            self.get_logger().warn(
                f'Skidpad: rejected fit — circles only {sep:.1f} m apart '
                f'(expected a figure-8 of two separated circles).'
            )
            return

        self._skid_track = track
        # Seed the cone map so the visualiser shows the whole track immediately.
        self._cone_map.update(blue, yellow)
        self.get_logger().info(
            f'Skidpad figure-8 precomputed from {len(blue) + len(yellow)} cones: '
            f'circles {sep:.1f} m apart, lane radius {track.lane_radius:.2f} m, '
            f'half-width {track.half_width:.2f} m.'
        )

    def _skid_elapsed(self) -> float:
        if self._skid_t0 is None:
            return 0.0
        return (self.get_clock().now() - self._skid_t0).nanoseconds * 1e-9

    def _record_spinoff(self) -> None:
        """Latch the exact time and speed at which the car left the lane."""
        self._skid_off_track = True
        t = self._skid_elapsed()
        v = self._car_speed
        self.get_logger().error(
            f'*** SPIN-OFF *** car left the lane after {t:.2f} s at '
            f'{v:.2f} m/s (deviation {self._skid_dev:.2f} m '
            f'> {self._skid_track.half_width + SKID_OFFTRACK_MARGIN:.2f} m). '
            f'Logged to {SKID_LOG_PATH}'
        )
        self._append_skid_csv(t, self._skid_v_target, v, self._skid_dev,
                              event='SPIN_OFF')

    def _skid_log_loop(self) -> None:
        """Log the speed once per second while characterising (req: every second)."""
        if self._skid_track is None or self._skid_off_track:
            return
        t = self._skid_elapsed()
        self.get_logger().info(
            f'Skidpad t={t:6.1f} s  v_target={self._skid_v_target:5.2f} m/s  '
            f'v_actual={self._car_speed:5.2f} m/s  dev={self._skid_dev:.2f} m'
        )
        self._append_skid_csv(t, self._skid_v_target, self._car_speed,
                              self._skid_dev, event='run')

    def _init_skid_csv(self) -> None:
        try:
            with open(SKID_LOG_PATH, 'w', newline='') as f:
                csv.writer(f).writerow(
                    ['time_s', 'v_target_mps', 'v_actual_mps', 'deviation_m', 'event']
                )
            self._skid_csv_ready = True
        except OSError as exc:
            self.get_logger().warn(f'Could not open skidpad log {SKID_LOG_PATH}: {exc!r}')
            self._skid_csv_ready = False

    def _append_skid_csv(self, t: float, v_target: float, v_actual: float,
                         dev: float, event: str) -> None:
        if not self._skid_csv_ready:
            return
        try:
            with open(SKID_LOG_PATH, 'a', newline='') as f:
                csv.writer(f).writerow(
                    [f'{t:.3f}', f'{v_target:.3f}', f'{v_actual:.3f}',
                     f'{dev:.3f}', event]
                )
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def _viz_loop(self) -> None:
        # Show the full accumulated cone map so historical cones remain visible.
        self._viz.update(
            self._car_pos,
            self._car_yaw,
            self._cone_map.blue,
            self._cone_map.yellow,
            self._centreline,
            blue_segs=self._blue_segs,
            yellow_segs=self._yellow_segs,
            midpoints=self._midpoints,
            local_mode=self._local_mode,
            start_pos=self._loop_detector.start_pos,
            drift=self._drift if self._local_mode else None,
        )

    def _publish_path(self) -> None:
        if self._centreline is None or len(self._centreline) == 0:
            return
        path = Path()
        path.header.frame_id = 'fsds/map'
        path.header.stamp    = self.get_clock().now().to_msg()
        for wp in self._centreline:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(wp[0])
            ps.pose.position.y = float(wp[1])
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub_path.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
