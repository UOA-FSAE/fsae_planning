"""
Localisation-aware raceline planner node.

Starts identically to the barebone centreline planner — mapping the track a few
midpoints at a time with the cone-wall planner.  It additionally watches the
driven path for loop closure (localisation.LoopClosureDetector); once a full lap
has been completed the whole track is known, so the accumulated path is closed
into a single loop and the car switches to CLOSED-LOOP planning on that completed
centreline (the raceline), localising against the map instead of the latest
boundary frame.  Perception is then used only to monitor odometry drift.

This is a sim-only extension — the upstream car stack has no localisation planner.
Everything except the localisation switch is inherited from CenterlinePlanner;
this class only overrides the planning-path hook (and reads car pose from the same
/fsae/slam/car_position topic).

Interface: identical to centerline_planner (subscribes /fsae/slam/{left,right}_track
+ /fsae/slam/car_position, publishes /fsae/planning/selected_trajectory).
"""
import numpy as np
import rclpy

from fsae_planning.centerline_planner import CenterlinePlanner
from fsae_planning.localisation import (
    LoopClosureDetector,
    build_completed_path,
    compute_drift,
    roll_loop_to_car,
)

DRIFT_WARN_DIST = 1.5  # m — mean perception-vs-map drift above this is warned


class RacelinePlanner(CenterlinePlanner):
    def __init__(self, node_name: str = 'raceline_planner'):
        super().__init__(node_name=node_name)

        self._loop_detector = LoopClosureDetector()
        self._global_loop: np.ndarray | None = None   # cached closed centreline
        self._drift = {'mean': 0.0, 'max': 0.0, 'n': 0}

    # ------------------------------------------------------------------
    # Overridden planning hook
    # ------------------------------------------------------------------

    def _compute_path(self) -> None:
        self._update_localisation()
        # Expose the loop-closure start point to the visualiser once seeded.
        self._start_pos = self._loop_detector.start_pos

        if self._local_mode:
            self._plan_closed_loop()
        else:
            super()._compute_path()   # reuse the cone-wall mapping planner

    # ------------------------------------------------------------------
    # Localisation / closed-loop planning
    # ------------------------------------------------------------------

    def _update_localisation(self) -> None:
        """Watch for loop closure; on the first close, cache the global loop."""
        if self._local_mode:
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

    def _plan_closed_loop(self) -> None:
        """
        Plan on the cached closed loop built from the full cone map.  Perception
        is no longer used for the path — only to monitor drift between the latest
        boundary frame and the map the loop was built from.
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


def main(args=None):
    rclpy.init(args=args)
    node = RacelinePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
