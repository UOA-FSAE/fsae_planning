import math

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from fs_msgs.msg import Track
from nav_msgs.msg import Odometry

# Forward sensor window — mimics the FOV of a real perception stack.
# Matched to build_local_path(max_ahead, max_lateral) so the planner sees
# the same number of cone pairs it did when subscribing to the oracle map.
LOOK_AHEAD = 25.0  # metres ahead to include in the published cone map
LOOK_WIDE  = 10.0  # metres lateral half-width of the sensor window
MIN_AHEAD  = 0.5   # metres: ignore cones already behind / at the car


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception')

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

        # Oracle map (published once by the simulator, latched)
        self.create_subscription(
            Track, '/fsds/testing_only/track', self._track_cb, latched_qos
        )
        self.create_subscription(
            Odometry, '/fsds/testing_only/odom', self._odom_cb, sensor_qos
        )

        self.pub_cones = self.create_publisher(Track, '/FusionCones', 10)

        self._track_msg: Track | None = None
        self._car_x   = 0.0
        self._car_y   = 0.0
        self._car_yaw = 0.0

        self.create_timer(0.1, self._publish_visible_cones)  # 10 Hz

        self.get_logger().info(
            f'Perception node ready '
            f'(look_ahead={LOOK_AHEAD} m, look_wide={LOOK_WIDE} m).'
        )

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _track_cb(self, msg: Track) -> None:
        self._track_msg = msg
        self.get_logger().info(
            f'Oracle map received: {len(msg.track)} cones total.',
            once=True,
        )

    def _odom_cb(self, msg: Odometry) -> None:
        p = msg.pose.pose.position
        self._car_x = p.x
        self._car_y = p.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self._car_yaw = math.atan2(siny_cosp, cosy_cosp)

    # ------------------------------------------------------------------
    # Perception pipeline
    # ------------------------------------------------------------------

    def _publish_visible_cones(self) -> None:
        if self._track_msg is None:
            return

        cos_y = math.cos(self._car_yaw)
        sin_y = math.sin(self._car_yaw)

        visible = []
        for cone in self._track_msg.track:
            dx = cone.location.x - self._car_x
            dy = cone.location.y - self._car_y
            x_car =  dx * cos_y + dy * sin_y
            y_car = -dx * sin_y + dy * cos_y
            if MIN_AHEAD < x_car < LOOK_AHEAD and abs(y_car) < LOOK_WIDE:
                visible.append(cone)

        out = Track()
        out.track = visible
        self.pub_cones.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
