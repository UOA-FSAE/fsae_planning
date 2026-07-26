"""
Barebone centreline planner node.

Plans a centreline a few midpoints at a time using the cone-wall barrier planner
(see boundary.build_path_walls): same-colour cones are connected into a wall mesh
and midpoints are chained with a greedy walk that penalises steps crossing the
mesh.  There is NO localisation — the car plans purely from the boundary cones it
currently has, lap after lap.

Interface (matches the fsae_autonomous car stack):

    in   /fsae/slam/left_track    fsae_interfaces/Track     blue (left) boundary, global frame
    in   /fsae/slam/right_track   fsae_interfaces/Track     yellow (right) boundary, global frame
    in   /fsae/slam/car_position  geometry_msgs/Pose        x,y in position; yaw in orientation.w
    out  /fsae/planning/selected_trajectory  geometry_msgs/PoseArray   centreline waypoints

The plan loop is triggered by each car_position update (upstream convention).  This
node is also the base for the localisation-aware raceline planner
(raceline_planner.RacelinePlanner), which reuses the mapping, publishing and
visualisation machinery here and only swaps in a closed-loop path once a lap closes.
"""
import numpy as np
import rclpy
from rclpy.node import Node

from fsae_interfaces.msg import Track
from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Empty

from fsae_planning.boundary import build_path_walls
from fsae_planning.cone_map import ConeMap
from fsae_planning.path_utils import build_local_path


def cones_to_array(cones) -> np.ndarray:
    """geometry_msgs/Point[] → (N, 2) float64 array of x,y."""
    if not cones:
        return np.empty((0, 2))
    return np.array([[p.x, p.y] for p in cones], dtype=np.float64)


class CenterlinePlanner(Node):
    def __init__(self, node_name: str = 'centerline_planner'):
        super().__init__(node_name)

        self.declare_parameter('plot', False)
        self._plot = self.get_parameter('plot').get_parameter_value().bool_value

        self.create_subscription(Track, '/fsae/slam/left_track',  self._left_cb,  10)
        self.create_subscription(Track, '/fsae/slam/right_track', self._right_cb, 10)
        self.create_subscription(Pose,  '/fsae/slam/car_position', self._pose_cb, 10)

        # Debug hook: the accumulated ConeMap never forgets a cone (see cone_map.py),
        # so an external tool that edits the track has no way to retract one.  An
        # Empty here drops the map; the next boundary frame rebuilds it from scratch.
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

        self._cone_map    = ConeMap()          # accumulated historical cone map
        self._blue_cones:   np.ndarray = np.empty((0, 2))   # latest boundary frame
        self._yellow_cones: np.ndarray = np.empty((0, 2))
        self._car_pos   = np.zeros(2)
        self._car_yaw   = 0.0
        self._have_pose = False
        self._centreline: np.ndarray | None = None
        self._blue_segs:   list = []
        self._yellow_segs: list = []
        self._midpoints:   np.ndarray = np.empty((0, 2))

        # Visualisation hooks — always off for the barebone planner; the raceline
        # subclass sets these once it enters closed-loop mode.
        self._local_mode = False
        self._start_pos: np.ndarray | None = None
        self._drift: dict | None = None

        self._viz = None
        if self._plot:
            from fsae_planning.viz_utils import Visualizer
            self._viz = Visualizer()
            self.create_timer(1 / 3.0, self._viz_loop)

        self.get_logger().info(f'{node_name} ready — waiting for car_position.')

    # ------------------------------------------------------------------
    # Subscribers
    # ------------------------------------------------------------------

    def _left_cb(self, msg: Track) -> None:
        self._blue_cones = cones_to_array(msg.cones)

    def _right_cb(self, msg: Track) -> None:
        self._yellow_cones = cones_to_array(msg.cones)

    def _pose_cb(self, msg: Pose) -> None:
        # x,y in position; yaw (rad) is stuffed into orientation.w (upstream convention).
        self._car_pos   = np.array([msg.position.x, msg.position.y])
        self._car_yaw   = float(msg.orientation.w)
        self._have_pose = True
        self._planning_loop()

    def _reset_cb(self, _msg: Empty) -> None:
        """Drop the accumulated cone map (debug/testing only)."""
        self._cone_map.reset()
        self._blue_cones   = np.empty((0, 2))
        self._yellow_cones = np.empty((0, 2))
        self._centreline   = None
        self._blue_segs    = []
        self._yellow_segs  = []
        self._midpoints    = np.empty((0, 2))
        self.get_logger().info('cone map reset by external request')

    # ------------------------------------------------------------------
    # Planning loop (template — subclasses override the hooks)
    # ------------------------------------------------------------------

    def _planning_loop(self) -> None:
        if not self._have_pose:
            return

        # Accumulate the current boundary frame into the persistent map.
        self._cone_map.update(self._blue_cones, self._yellow_cones)

        self._compute_path()
        self._publish_trajectory()
        self._publish_debug()

        if self._centreline is None:
            self.get_logger().warn(
                'No forward cones visible — no trajectory published',
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().info(
                f'trajectory published: {len(self._centreline)} pts  '
                f'car=({self._car_pos[0]:.1f},{self._car_pos[1]:.1f}) '
                f'{"[LOCALISED]" if self._local_mode else ""}',
                throttle_duration_sec=1.0,
            )

    # ------------------------------------------------------------------
    # Override hooks
    # ------------------------------------------------------------------

    def _compute_path(self) -> None:
        """Cone-wall mesh planner over the accumulated boundary cones."""
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

    # ------------------------------------------------------------------
    # Publishing / visualisation
    # ------------------------------------------------------------------

    def _publish_trajectory(self) -> None:
        if self._centreline is None or len(self._centreline) == 0:
            return
        msg = PoseArray()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        for wp in self._centreline:
            pose = Pose()
            pose.position.x = float(wp[0])
            pose.position.y = float(wp[1])
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

    def _viz_loop(self) -> None:
        if self._viz is None:
            return
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
            start_pos=self._start_pos,
            drift=self._drift if self._local_mode else None,
        )


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
