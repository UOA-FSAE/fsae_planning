"""
Target-speed-profile visualiser (debug/viz only, not part of the control loop).

Republishes /fsae/planning/selected_trajectory's curvature-based target speed,
one value per waypoint, so a dashboard can colour the planned path by target
speed the same way it colours the driven trail by actual speed -- letting you
compare, at the same point in space (e.g. a corner the car is drifting on),
what speed the controller *should* be commanding there against what the car
*actually* did. That comparison is the reason this exists: it separates a
planner/speed-target problem (the profile itself is wrong for the corner)
from a controller/tracking problem (the profile is right but the car doesn't
hit it).

    in   /fsae/planning/selected_trajectory   geometry_msgs/PoseArray        planned path
    out  /fsae/planning/target_speed_profile  std_msgs/Float64MultiArray     one target
                                               speed per pose, same order/length as the
                                               trajectory that produced it

Reuses control_utils.curvature_speed_profile(), which itself calls the exact
curvature_speed() the live controller calls -- so the visualised number is
never a re-derived approximation, only ever the real thing. Pass this node
the SAME v_max/v_min as the live controller (they are launch params here too)
so the two match; there's no shared source for that today outside
fsae_params.yaml, so a mismatch here would visualise the wrong target.

Deliberately its own node rather than a change to stanley_controller.py:
curvature_speed_profile() is O(n^2) (a curvature_speed() scan from every
waypoint), and has no place in the 20 Hz control loop's tick budget. Not
part of any default launch file -- run it alongside a normal sim session
only while debugging a speed-target question:
    ros2 run fsae_control target_speed_viz
"""
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import PoseArray
from std_msgs.msg import Float64MultiArray

from fsae_control.control_utils import curvature_speed_profile


class TargetSpeedViz(Node):
    def __init__(self):
        super().__init__('target_speed_viz')

        self.declare_parameters(
            namespace='',
            parameters=[
                # Keep these equal to stanley_controller.py's own v_max/v_min
                # -- see the module docstring.
                ('v_max', 15.0),
                ('v_min', 1.5),
            ],
        )
        self._v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self._v_min = self.get_parameter('v_min').get_parameter_value().double_value

        self.create_subscription(
            PoseArray, '/fsae/planning/selected_trajectory', self._traj_cb, 10
        )
        self.pub_profile = self.create_publisher(
            Float64MultiArray, '/fsae/planning/target_speed_profile', 10
        )

        self.get_logger().info(
            f'target_speed_viz ready (v_max={self._v_max:g}, v_min={self._v_min:g}) -- '
            'debug/viz only, not part of the control loop.'
        )

    def _traj_cb(self, msg: PoseArray) -> None:
        if not msg.poses:
            return
        waypoints = [[p.position.x, p.position.y] for p in msg.poses]
        profile = curvature_speed_profile(waypoints, v_max=self._v_max, v_min=self._v_min)
        out = Float64MultiArray()
        out.data = [float(v) for v in profile]
        self.pub_profile.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TargetSpeedViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
