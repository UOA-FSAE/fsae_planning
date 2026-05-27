import math

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """Non-blocking overhead ego-view: car, cones, centreline, lookahead target.

    Coordinate convention for the plot:
        +y  = ahead of car  (up on screen)
        +x  = right of car  (right on screen)
    Blue cones (left boundary) therefore appear on the LEFT, yellow on the RIGHT.
    """

    def __init__(self, window_half: float = 25.0):
        plt.ion()
        self._fig, self._ax = plt.subplots(figsize=(6, 8))
        self._w = window_half
        self._fig.tight_layout()
        plt.show(block=False)

    def update(
        self,
        car_pos: np.ndarray,
        car_yaw: float,
        blue_cones: np.ndarray,
        yellow_cones: np.ndarray,
        centreline: np.ndarray | None,
        target: np.ndarray | None = None,
    ) -> None:
        ax = self._ax
        ax.cla()

        cos_y = math.cos(car_yaw)
        sin_y = math.sin(car_yaw)

        def to_plot(pts: np.ndarray) -> np.ndarray:
            """Global (N, 2) → plot frame (+x right-of-car, +y ahead)."""
            pts = np.atleast_2d(np.asarray(pts, dtype=float))
            if len(pts) == 0:
                return np.empty((0, 2))
            rel = pts - car_pos
            x_fwd  =  rel[:, 0] * cos_y + rel[:, 1] * sin_y
            y_left = -rel[:, 0] * sin_y + rel[:, 1] * cos_y
            # negate y_left so left-of-car → negative plot_x → left side of screen
            return np.column_stack([-y_left, x_fwd])

        if len(blue_cones) > 0:
            bc = to_plot(blue_cones)
            ax.scatter(bc[:, 0], bc[:, 1], c='dodgerblue', s=30, zorder=3, label='Blue')

        if len(yellow_cones) > 0:
            yc = to_plot(yellow_cones)
            ax.scatter(yc[:, 0], yc[:, 1], c='gold', edgecolors='darkorange',
                       linewidths=0.5, s=30, zorder=3, label='Yellow')

        if centreline is not None and len(centreline) > 0:
            cl = to_plot(centreline)
            ax.plot(cl[:, 0], cl[:, 1], 'g--', lw=1.5, label='Centreline', zorder=2)

        if target is not None:
            tgt = to_plot(np.atleast_2d(target))
            ax.scatter(tgt[:, 0], tgt[:, 1], c='red', s=150,
                       marker='*', zorder=5, label='Target')

        # Car as filled triangle pointing up (= forward)
        ax.add_patch(plt.Polygon(
            [[0, 2.0], [-1.0, -0.8], [1.0, -0.8]], color='black', zorder=6
        ))

        w = self._w
        ax.set_xlim(-w * 0.5, w * 0.5)
        ax.set_ylim(-w * 0.15, w * 0.85)
        ax.set_aspect('equal')
        ax.set_xlabel('← left of car   |   right of car →')
        ax.set_ylabel('distance ahead (m)')
        ax.set_title('Centreline Planner — Ego View')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.pause(0.001)
