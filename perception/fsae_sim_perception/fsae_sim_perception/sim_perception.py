"""
Simulator perception stand-in (FSDS → fsae_autonomous interface bridge).

On the real car the perception front-end is a ZED camera node (cone detection +
visual-odometry pose) followed by a SLAM cone_mapper that produces the boundary
tracks.  The FSDS simulator instead broadcasts a latched oracle cone map plus
ground-truth odometry.  This node stands in for the whole camera+SLAM front-end:
it crops the oracle map to a forward FOV window (to mimic a real sensor) and
republishes it on the exact topics/messages the car stack expects.

    in   /fsds/testing_only/track  fs_msgs/Track       latched oracle cone map
    in   /fsds/testing_only/odom   nav_msgs/Odometry   ground-truth pose

    out  /fsae/slam/car_position   geometry_msgs/PoseStamped     x,y in position; yaw in orientation.w;
                                                                  header.stamp = odom's own measurement time
    out  /fsae/slam/car_odom       nav_msgs/Odometry             SAME snapshot as car_position (position/
                                                                  yaw in pose, speed/yaw-rate in twist) --
                                                                  see "Speed/yaw-rate" below. Consumed by
                                                                  external tooling (e.g. the ros2_vis_dashboard,
                                                                  which colours the driven trail by speed), not
                                                                  by anything in this repo.
    (the four *_track topics are window mode only; in camera mode cone_mapper_shim publishes them)
    out  /fsae/slam/left_track     fsae_interfaces/Track         blue (left) boundary, global frame
    out  /fsae/slam/right_track    fsae_interfaces/Track         yellow (right) boundary, global frame
    out  /fsae/slam/big_orange_track    fsae_interfaces/Track    big orange cones (start/finish line), global frame
    out  /fsae/slam/small_orange_track  fsae_interfaces/Track    small orange cones (e.g. accel braking zone), global frame
    out  /fsae/perception/cone_detection  fsae_interfaces/ConeDetection  local-frame detections + car_pose
                                                                  (blue, yellow, big_orange, small_orange)

Orange cones: FSDS's oracle map carries them (ORANGE_BIG = the 4-cone start/
finish gate on every map; Acceleration also has ORANGE_SMALL along the braking
zone). They are kept on their own topics, never mixed into left/right_track,
because they mark the start/finish line rather than a track edge -- the same
split the real car's SLAM makes. Cropped to the same FOV window as blue/yellow,
so the start gate drops out of view after launch and reappears on the next lap.

Set the `full_track` parameter true to publish the entire map every frame (used by
the skidpad planner, which reconstructs the whole figure-8 up front); the default
(false) publishes only the forward window, matching normal driving.

Camera mode (`fov_mode: 'camera'`, default 'window'): the window above is
much more generous than the real kart's sensing, so the sim never shows the
one-colour views the kart sees in sparse corners. Camera mode instead
emulates the kart's camera, and ONLY the camera: this node then publishes
cone_detection frames (and the pose) but no *_track topics. The tracks come
from cone_mapper_shim (fsae_planning, the kart's own node, unchanged), fed
these frames by perception.launch.py -- the same camera -> shim -> planner
chain as on the kart, retention included. The camera:
  * a forward wedge, `camera_hfov_deg` wide and `camera_range` deep from the
    car origin. The kart's ZED 2i (2.1 mm lens) is 110 deg horizontal with
    0.3-20 m depth;
  * `camera_rate`: frames per second. The kart's fused detections run at
    8.0-8.5 Hz (ZED grabbing at 15 Hz, then YOLO and the LiDAR fusion);
  * `camera_dropout`: each cone is independently missed per frame with this
    probability;
  * `camera_fixed_miss`: this fraction of the map's blue/yellow cones is
    never detected at all (chosen once, seeded by camera_seed). FSDS maps are
    all 3.5 m wide with ~4 m spacing, so the wedge alone always sees both
    colours; permanent misses open the multi-metre gaps and one-colour
    stretches of the kart's sparser tracks, which is what makes the planner's
    gate walk dead-end. Cones within _FIXED_MISS_START_CLEAR of the map
    origin (the car's start pose in FSDS) are never dropped, so the car can
    always launch -- the start gate is not what this emulates;
  * `camera_noise_std`: Gaussian position noise per detection, this many
    metres (per axis) at 10 m range and growing linearly with range (stereo
    depth error grows with distance). The shim averages it out per cone;
  * `camera_both_colours_gate`: the kart's cone_detection publishes a frame
    only when it holds >= 1 blue AND >= 1 yellow cone; with this on, a
    one-colour frame is not published at all, so the shim only ages its
    cones. fsds_bridge's cone e-brake (car-frame) then keeps reading the
    last frame it got.
Every non-blue/yellow cone is reported as big_orange, as the kart's camera
does (it never fills small_orange).

Two publish rates (`pose_rate` 20 Hz, `cone_rate` 10 Hz): pose and cones are
published on separate timers. `pose_rate` must be >= the controller's rate --
stanley_controller.py's control step is directly triggered by each
car_position arrival, so a slower pose_rate here directly slows the control
loop and can produce catch-up jumps rather than smooth motion. The FSDS
bridge itself publishes odom at 250 Hz, so this node's own publish rate is
the only possible bottleneck. Cones stay at 10 Hz — cropping the oracle map
and building three messages is the expensive part of this node, and the
planner gains nothing from running it at the control rate.

Known limitation: this node is not a SLAM stand-in for accuracy, only for
range. The pose it publishes is FSDS ground truth, copied verbatim: no noise,
no drift, no estimation lag. The offline tuner models this gap explicitly via
`SLAM_NOISE_ENABLED` in fsae_MPCTest/settings.py (default off, since FSDS has
no such error).

Speed/yaw-rate: car_odom's twist (speed/yaw-rate) is set in the SAME
_odom_cb call as car_position's pose, so a consumer reading both from one
_publish_pose() tick always gets a mutually consistent snapshot rather than
racing two separately-timed reads of the raw 250 Hz /fsds/testing_only/odom
topic.
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from fs_msgs.msg import Cone, Track as FsTrack
from fsae_interfaces.msg import ConeDetection, Track
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry


# Camera mode: camera_fixed_miss never drops cones this close (m) to the
# map origin, which is where FSDS puts the car at the start.
_FIXED_MISS_START_CLEAR = 10.0


class SimPerception(Node):
    def __init__(self):
        super().__init__('sim_perception')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('look_ahead', 25.0),   # m ahead to include in the forward box
                ('look_wide', 10.0),    # m lateral half-width of the forward box
                ('min_ahead', 0.5),     # m: box ignores cones behind / at the car
                # 25.0 — kept >= centerline_planner's look_radius/
                # plan_horizon so cones for a tight corner reach the planner
                # far enough out to detect and brake for it in time.
                ('look_radius', 25.0),  # m: omni-directional visibility radius
                ('full_track', False),  # publish the whole map instead of the window
                # Publish rates (Hz). Split deliberately — see the module
                # docstring. pose_rate MUST be >= the controller's rate, since
                # the control loop is directly triggered by car_position.
                ('pose_rate', 20.0),
                ('cone_rate', 10.0),
                # Camera mode -- see the module docstring. 'window' keeps the
                # look_radius/box crop above.
                ('fov_mode', 'window'),
                ('camera_hfov_deg', 110.0),
                ('camera_range', 10.0),
                ('camera_rate', 8.0),
                ('camera_dropout', 0.0),
                ('camera_fixed_miss', 0.0),
                ('camera_noise_std', 0.0),
                ('camera_both_colours_gate', False),
                ('camera_seed', 0),     # dropout/noise RNG seed, for repeatable runs
            ],
        )
        self._look_ahead  = self.get_parameter('look_ahead').get_parameter_value().double_value
        self._look_wide   = self.get_parameter('look_wide').get_parameter_value().double_value
        self._min_ahead   = self.get_parameter('min_ahead').get_parameter_value().double_value
        self._look_radius = self.get_parameter('look_radius').get_parameter_value().double_value
        self._full_track  = self.get_parameter('full_track').get_parameter_value().bool_value
        self._pose_rate   = self.get_parameter('pose_rate').get_parameter_value().double_value
        self._cone_rate   = self.get_parameter('cone_rate').get_parameter_value().double_value
        # Guard against a zero/negative rate turning into a divide-by-zero or
        # a timer that never fires.
        self._pose_rate = self._pose_rate if self._pose_rate > 0.0 else 20.0
        self._cone_rate = self._cone_rate if self._cone_rate > 0.0 else 10.0

        self._fov_mode = self.get_parameter('fov_mode').get_parameter_value().string_value
        if self._fov_mode not in ('window', 'camera'):
            self.get_logger().error(
                f"unrecognised fov_mode '{self._fov_mode}' (expected 'window' or 'camera') "
                "-- using 'window'")
            self._fov_mode = 'window'
        self._cam_half_fov = math.radians(
            0.5 * self.get_parameter('camera_hfov_deg').get_parameter_value().double_value)
        self._cam_range   = self.get_parameter('camera_range').get_parameter_value().double_value
        self._cam_dropout = self.get_parameter('camera_dropout').get_parameter_value().double_value
        self._cam_fixed_miss = self.get_parameter('camera_fixed_miss').get_parameter_value().double_value
        self._cam_gate    = self.get_parameter('camera_both_colours_gate').get_parameter_value().bool_value
        self._cam_noise   = self.get_parameter('camera_noise_std').get_parameter_value().double_value
        self._cam_rate    = self.get_parameter('camera_rate').get_parameter_value().double_value
        self._cam_rate = self._cam_rate if self._cam_rate > 0.0 else 8.0
        self._camera = self._fov_mode == 'camera' and not self._full_track
        self._rng = np.random.default_rng(
            self.get_parameter('camera_seed').get_parameter_value().integer_value)
        # Camera mode: per colour, the oracle cones the camera never detects
        # (camera_fixed_miss). Index-aligned with *_all.
        self._never_seen: dict = {}

        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.create_subscription(FsTrack, '/fsds/testing_only/track', self._track_cb, latched_qos)
        self.create_subscription(Odometry, '/fsds/testing_only/odom', self._odom_cb, sensor_qos)

        self.pub_pose  = self.create_publisher(PoseStamped,   '/fsae/slam/car_position',        10)
        self.pub_odom  = self.create_publisher(Odometry,      '/fsae/slam/car_odom',            10)
        # Camera mode leaves the tracks to cone_mapper_shim (see the module
        # docstring), so it doesn't even advertise them.
        if not self._camera:
            self.pub_left  = self.create_publisher(Track,         '/fsae/slam/left_track',          10)
            self.pub_right = self.create_publisher(Track,         '/fsae/slam/right_track',         10)
            self.pub_big_orange   = self.create_publisher(Track, '/fsae/slam/big_orange_track',   10)
            self.pub_small_orange = self.create_publisher(Track, '/fsae/slam/small_orange_track', 10)
        self.pub_det   = self.create_publisher(ConeDetection, '/fsae/perception/cone_detection', 10)

        # Oracle map, split by colour into global (N, 2) arrays once received.
        self._blue_all:   np.ndarray = np.empty((0, 2))
        self._yellow_all: np.ndarray = np.empty((0, 2))
        self._big_orange_all:   np.ndarray = np.empty((0, 2))
        self._small_orange_all: np.ndarray = np.empty((0, 2))
        self._have_map = False
        self._have_odom = False
        self._car_x = 0.0
        self._car_y = 0.0
        self._car_yaw = 0.0
        # Speed/yaw-rate, set alongside car_x/y/yaw in the SAME _odom_cb call
        # so car_odom's twist and pose always originate from one atomic
        # snapshot -- see the module docstring's "Speed/yaw-rate" note.
        self._car_vx = 0.0
        self._car_vy = 0.0
        self._car_yaw_rate = 0.0
        self._odom_stamp = None

        # Two timers, deliberately at different rates — see the module
        # docstring's "Two publish rates" note. Pose must keep up with the
        # controller; the cone map does not need to.
        self.create_timer(1.0 / self._pose_rate, self._publish_pose)
        self.create_timer(1.0 / (self._cam_rate if self._camera else self._cone_rate),
                          self._publish_cones)

        if self._full_track:
            mode = 'full map'
        elif self._camera:
            mode = (f'camera {math.degrees(2 * self._cam_half_fov):g} deg × {self._cam_range:g} m '
                    f'@ {self._cam_rate:g} Hz, dropout {self._cam_dropout:g}, '
                    f'fixed miss {self._cam_fixed_miss:g}, noise {self._cam_noise:g} m @ 10 m, '
                    f'both-colours gate {self._cam_gate}; tracks from cone_mapper_shim')
        else:
            mode = f'radius {self._look_radius} m ∪ box {self._look_ahead}×{2 * self._look_wide} m'
        self.get_logger().info(
            f'sim_perception ready ({mode}); '
            f'pose @ {self._pose_rate:g} Hz, cones @ '
            f'{self._cam_rate if self._camera else self._cone_rate:g} Hz.'
        )
        if self._pose_rate < 20.0:
            self.get_logger().warn(
                f'pose_rate is {self._pose_rate:g} Hz but the controller runs at 20 Hz — '
                'the control loop will run slower than intended, which can '
                'induce steering oscillation. See this node\'s docstring.'
            )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _track_cb(self, msg: FsTrack) -> None:
        blue, yellow, big_orange, small_orange = [], [], [], []
        for cone in msg.track:
            pt = [cone.location.x, cone.location.y]
            if cone.color == Cone.BLUE:
                blue.append(pt)
            elif cone.color == Cone.YELLOW:
                yellow.append(pt)
            elif cone.color == Cone.ORANGE_BIG:
                big_orange.append(pt)
            elif cone.color == Cone.ORANGE_SMALL:
                small_orange.append(pt)
        self._blue_all   = self._as_array(blue)
        self._yellow_all = self._as_array(yellow)
        self._big_orange_all   = self._as_array(big_orange)
        self._small_orange_all = self._as_array(small_orange)
        # Camera mode: cones the camera never detects (camera_fixed_miss).
        # Blue/yellow only -- the start gate's orange cones stay visible.
        miss_rng = np.random.default_rng(
            self.get_parameter('camera_seed').get_parameter_value().integer_value + 1000)
        self._never_seen = {
            key: (miss_rng.random(len(arr)) < self._cam_fixed_miss
                  if key in ('blue', 'yellow') else np.zeros(len(arr), dtype=bool))
            for key, arr in self._all_cones().items()
        }
        for key, arr in self._all_cones().items():
            if len(arr):
                self._never_seen[key] &= np.hypot(arr[:, 0], arr[:, 1]) > _FIXED_MISS_START_CLEAR
        self._have_map = True
        self.get_logger().info(
            f'Oracle map received: {len(blue)} blue + {len(yellow)} yellow + '
            f'{len(big_orange)} big orange + {len(small_orange)} small orange.', once=True
        )

    @staticmethod
    def _as_array(pts: list) -> np.ndarray:
        return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2))

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._car_x = p.x
        self._car_y = p.y
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._car_yaw = math.atan2(siny_cosp, cosy_cosp)
        # Set in the SAME callback invocation as position/yaw above, so
        # car_odom's pose and twist always come from one atomic odom sample.
        v = msg.twist.twist.linear
        self._car_vx = v.x
        self._car_vy = v.y
        self._car_yaw_rate = msg.twist.twist.angular.z
        self._odom_stamp = msg.header.stamp
        self._have_odom = True

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def _visible(self, cones: np.ndarray) -> np.ndarray:
        """
        Crop cones to the sensor FOV (car frame), or pass all through.

        FOV = within look_radius of the car (omni — mimics a 360° lidar and keeps
        the cones around a bend that a forward-only box loses when the track
        curves away) OR inside the forward box (longer-range preview straight
        ahead).  Cropping still happens so the planner's cone map fills in track
        order over a lap rather than being handed the whole oracle map at once.
        """
        if self._full_track or len(cones) == 0:
            return cones
        cos_y, sin_y = math.cos(self._car_yaw), math.sin(self._car_yaw)
        rel = cones - np.array([self._car_x, self._car_y])
        dist = np.hypot(rel[:, 0], rel[:, 1])
        x_car =  rel[:, 0] * cos_y + rel[:, 1] * sin_y
        y_car = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
        box = (x_car > self._min_ahead) & (x_car < self._look_ahead) & \
              (np.abs(y_car) < self._look_wide)
        return cones[(dist < self._look_radius) | box]

    def _all_cones(self) -> dict:
        return {
            'blue': self._blue_all, 'yellow': self._yellow_all,
            'big_orange': self._big_orange_all, 'small_orange': self._small_orange_all,
        }

    def _camera_detect(self, cones: np.ndarray) -> np.ndarray:
        """
        Boolean mask of the cones the camera detects this frame: inside the
        forward wedge (camera_hfov_deg, camera_range from the car origin),
        each then independently missed with probability camera_dropout.
        """
        if len(cones) == 0:
            return np.zeros(0, dtype=bool)
        cos_y, sin_y = math.cos(self._car_yaw), math.sin(self._car_yaw)
        rel = cones - np.array([self._car_x, self._car_y])
        x_car =  rel[:, 0] * cos_y + rel[:, 1] * sin_y
        y_car = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
        seen = ((x_car > 0.0) & (np.hypot(x_car, y_car) <= self._cam_range)
                & (np.abs(np.arctan2(y_car, x_car)) <= self._cam_half_fov))
        if self._cam_dropout > 0.0:
            seen &= self._rng.random(len(cones)) >= self._cam_dropout
        return seen

    def _camera_frame(self) -> dict | None:
        """
        One camera frame: global positions of this frame's detections per
        colour (orange all as big_orange, as the kart's camera labels them),
        or None when the both-colours gate (if enabled) drops the frame.
        """
        cones = self._all_cones()
        det = {}
        for key, arr in cones.items():
            pts = arr[self._camera_detect(arr) & ~self._never_seen[key]]
            if self._cam_noise > 0.0 and len(pts):
                rng_m = np.hypot(pts[:, 0] - self._car_x, pts[:, 1] - self._car_y)
                sigma = self._cam_noise * rng_m / 10.0
                pts = pts + self._rng.normal(0.0, 1.0, pts.shape) * sigma[:, None]
            det[key] = pts
        det['big_orange'] = np.vstack([det['big_orange'], det['small_orange']])
        det['small_orange'] = np.empty((0, 2))
        if self._cam_gate and not (len(det['blue']) and len(det['yellow'])):
            return None
        return det

    def _car_pose_msg(self) -> PoseStamped:
        msg = PoseStamped()
        # Odom's own timestamp, not now() — this is the time the pose was
        # actually measured, which is what downstream pose-age estimation
        # needs to be measured against.
        msg.header.stamp = self._odom_stamp
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self._car_x)
        msg.pose.position.y = float(self._car_y)
        # Upstream convention: yaw (rad) is repurposed into orientation.w.
        msg.pose.orientation.w = float(self._car_yaw)
        return msg

    def _car_odom_msg(self) -> Odometry:
        """
        Same snapshot as _car_pose_msg() (both read _car_x/_car_y/_car_yaw/
        _car_vx/_car_vy/_car_yaw_rate, all set together in _odom_cb) -- see
        the module docstring's "Speed/yaw-rate" note. Unlike car_position's
        repurposed orientation.w convention, this uses a real quaternion
        (nav_msgs/Odometry consumers expect one).
        """
        msg = Odometry()
        msg.header.stamp = self._odom_stamp
        msg.header.frame_id = 'map'
        msg.pose.pose.position.x = float(self._car_x)
        msg.pose.pose.position.y = float(self._car_y)
        half_yaw = 0.5 * self._car_yaw
        msg.pose.pose.orientation.z = float(math.sin(half_yaw))
        msg.pose.pose.orientation.w = float(math.cos(half_yaw))
        msg.twist.twist.linear.x = float(self._car_vx)
        msg.twist.twist.linear.y = float(self._car_vy)
        msg.twist.twist.angular.z = float(self._car_yaw_rate)
        return msg

    @staticmethod
    def _global_track(cones: np.ndarray) -> Track:
        msg = Track()
        for c in cones:
            msg.cones.append(Point(x=float(c[0]), y=float(c[1]), z=0.0))
        return msg

    def _local_points(self, cones: np.ndarray) -> list:
        """Transform global cones into the car frame for cone_detection."""
        pts = []
        cos_y, sin_y = math.cos(self._car_yaw), math.sin(self._car_yaw)
        for c in cones:
            dx = c[0] - self._car_x
            dy = c[1] - self._car_y
            x_local =  dx * cos_y + dy * sin_y
            y_local = -dx * sin_y + dy * cos_y
            pts.append(Point(x=float(x_local), y=float(y_local), z=0.0))
        return pts

    def _publish_pose(self) -> None:
        """
        Pose-only tick, run at pose_rate (default 20 Hz to match the
        controller).  Deliberately does NOT touch the cone map — see the
        class docstring's "Two publish rates" note for why these are split.

        Publishes car_position and car_odom back-to-back from the SAME
        _car_x/_car_y/_car_yaw/_car_vx/_car_vy/_car_yaw_rate snapshot -- see
        the module docstring's "Speed/yaw-rate" note. Do not call _odom_cb
        or otherwise refresh state between these two publishes.
        """
        if not self._have_odom:
            return
        self.pub_pose.publish(self._car_pose_msg())
        self.pub_odom.publish(self._car_odom_msg())

    def _publish_cones(self) -> None:
        """
        Cone/track tick, run at cone_rate (default 10 Hz; camera_rate in
        camera mode, which publishes cone_detection only).  Cropping the
        oracle map and building three messages is the expensive part of this
        node, and the planner gains nothing from it running at the control
        rate.
        """
        if not self._have_odom or not self._have_map:
            return

        # ConeDetection embeds the pose the detections were taken from, so
        # this tick builds its own (current) pose rather than reusing one
        # from the faster pose timer — the two must be consistent with the
        # cone transforms computed immediately below.
        pose = self._car_pose_msg()

        if self._camera:
            det_cones = self._camera_frame()
            if det_cones is None:
                return   # gated: the kart's camera publishes nothing this frame
        else:
            det_cones = {key: self._visible(arr) for key, arr in self._all_cones().items()}
            self.pub_left.publish(self._global_track(det_cones['blue']))
            self.pub_right.publish(self._global_track(det_cones['yellow']))
            self.pub_big_orange.publish(self._global_track(det_cones['big_orange']))
            self.pub_small_orange.publish(self._global_track(det_cones['small_orange']))

        det = ConeDetection()
        det.header.stamp    = self.get_clock().now().to_msg()
        det.header.frame_id = 'base_link'
        det.car_pose = pose.pose   # ConeDetection.car_pose is geometry_msgs/Pose, not Stamped
        det.blue   = self._local_points(det_cones['blue'])
        det.yellow = self._local_points(det_cones['yellow'])
        det.big_orange   = self._local_points(det_cones['big_orange'])
        det.small_orange = self._local_points(det_cones['small_orange'])
        self.pub_det.publish(det)


def main(args=None):
    rclpy.init(args=args)
    node = SimPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
