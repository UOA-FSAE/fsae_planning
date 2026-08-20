#!/usr/bin/env python3
"""
Lap timer / race-stats node.

Uses the BIG ORANGE cones as the start/finish gate:
  * timing for lap 1 starts when the car first pulls away from the orange gate,
  * every time the car returns to the orange gate a lap is completed,
  * average speed for each lap = distance travelled / lap time.

Results are written to  ~/.fsds_race_stats.txt  (plain text, read live by the
in-sim F9 tuning window's RACE STATS panel) and  ~/.fsds_race_stats.json.

Orange-cone positions come from the ground-truth track topic
(/fsds/testing_only/track, latched) which reliably carries cone colours.
"""
import json
import math
import os

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy,
                       QoSHistoryPolicy, qos_profile_sensor_data)

from nav_msgs.msg import Odometry
from fs_msgs.msg import Track, Cone

TXT_PATH  = os.path.expanduser('~/.fsds_race_stats.txt')
JSON_PATH = os.path.expanduser('~/.fsds_race_stats.json')

NEAR_M        = 6.0    # within this distance of an orange cone => "at the gate"
FAR_M         = 12.0   # must get this far from the gate before it can re-trigger
MIN_LAP_TIME  = 5.0    # s, ignore gate re-triggers faster than this (debounce)


class LapTimer(Node):
    def __init__(self):
        super().__init__('lap_timer')

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(Track, '/fsds/testing_only/track', self._track_cb, latched)
        self.create_subscription(Odometry, '/fsds/testing_only/odom', self._odom_cb,
                                 qos_profile_sensor_data)

        self._orange = []          # [(x, y), ...] big-orange cone positions
        self._have_gate = False
        self._at_gate = True       # car starts on the gate
        self._running = False
        self._lap_start = 0.0
        self._dist = 0.0           # distance travelled in current lap [m]
        self._last_pos = None
        self._laps = []            # [{lap, time_s, avg_speed_mps, avg_speed_kmh}, ...]

        self.get_logger().info(f'Lap timer running. Stats -> {TXT_PATH}')
        self._write()

    # ------------------------------------------------------------------
    def _track_cb(self, msg: Track) -> None:
        pts = [(c.location.x, c.location.y)
               for c in msg.track if c.color == Cone.ORANGE_BIG]
        if pts:
            self._orange = pts
            if not self._have_gate:
                self._have_gate = True
                self.get_logger().info(f'Start/finish gate found: {len(pts)} orange cones.')

    # ------------------------------------------------------------------
    def _odom_cb(self, msg: Odometry) -> None:
        if not self._have_gate:
            return

        p = msg.pose.pose.position
        pos = (p.x, p.y)

        # accumulate travelled distance while a lap is being timed
        if self._running and self._last_pos is not None:
            self._dist += math.hypot(pos[0] - self._last_pos[0],
                                     pos[1] - self._last_pos[1])
        self._last_pos = pos

        d = min(math.hypot(pos[0] - ox, pos[1] - oy) for ox, oy in self._orange)
        now = self.get_clock().now().nanoseconds / 1e9

        if self._at_gate and d > FAR_M:
            # car has left the gate
            self._at_gate = False
            if not self._running:
                self._running = True
                self._lap_start = now
                self._dist = 0.0
                self.get_logger().info('Track START — timing lap 1.')
                self._write()

        elif (not self._at_gate) and d < NEAR_M:
            # car has returned to the gate -> lap complete
            self._at_gate = True
            if self._running:
                self._complete_lap(now)

    # ------------------------------------------------------------------
    def _complete_lap(self, now: float) -> None:
        t = now - self._lap_start
        if t < MIN_LAP_TIME:
            return  # debounce spurious re-trigger
        avg = (self._dist / t) if t > 0 else 0.0
        lap = {
            'lap': len(self._laps) + 1,
            'time_s': round(t, 2),
            'avg_speed_mps': round(avg, 2),
            'avg_speed_kmh': round(avg * 3.6, 2),
        }
        self._laps.append(lap)
        self.get_logger().info(
            f"Lap {lap['lap']} complete: {lap['time_s']} s, "
            f"avg {lap['avg_speed_kmh']} km/h ({lap['avg_speed_mps']} m/s).")
        # immediately start timing the next lap
        self._lap_start = now
        self._dist = 0.0
        self._write()

    # ------------------------------------------------------------------
    def _write(self) -> None:
        # JSON (machine-readable)
        data = {
            'running': self._running,
            'current_lap': (len(self._laps) + 1) if self._running else 0,
            'laps': self._laps,
        }
        try:
            with open(JSON_PATH, 'w') as f:
                json.dump(data, f)
        except OSError as e:
            self.get_logger().warn(f'stats json write failed: {e}')

        # Plain text (shown verbatim by the F9 RACE STATS panel)
        lines = ['RACE STATS']
        if self._running:
            lines.append(f'Status: RUNNING  (on lap {len(self._laps) + 1})')
        elif self._laps:
            lines.append('Status: finished')
        else:
            lines.append('Status: waiting at start gate')
        lines.append('')
        if self._laps:
            best = min(self._laps, key=lambda l: l['time_s'])
            for l in self._laps:
                star = '  *best' if l is best else ''
                lines.append(
                    f"Lap {l['lap']}:  {l['time_s']:6.2f} s   "
                    f"avg {l['avg_speed_kmh']:5.1f} km/h{star}")
        else:
            lines.append('(no completed laps yet)')
        try:
            with open(TXT_PATH, 'w') as f:
                f.write('\n'.join(lines) + '\n')
        except OSError as e:
            self.get_logger().warn(f'stats txt write failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = LapTimer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
