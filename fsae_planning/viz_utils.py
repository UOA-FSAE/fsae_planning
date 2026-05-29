import math

import numpy as np
import matplotlib.pyplot as plt


class Visualizer:
    """Non-blocking overhead ego-view: car, cones, walls, midpoints, centreline.

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
        blue_segs: list | None = None,
        yellow_segs: list | None = None,
        midpoints: np.ndarray | None = None,
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
            return np.column_stack([-y_left, x_fwd])

        # --- Wall segments (drawn first, lowest zorder) ---
        if blue_segs:
            for (p1, p2) in blue_segs:
                pts = to_plot(np.array([p1, p2]))
                ax.plot(pts[:, 0], pts[:, 1],
                        color='dodgerblue', alpha=0.30, lw=0.9, zorder=1)

        if yellow_segs:
            for (p1, p2) in yellow_segs:
                pts = to_plot(np.array([p1, p2]))
                ax.plot(pts[:, 0], pts[:, 1],
                        color='gold', alpha=0.30, lw=0.9, zorder=1)

        # --- Candidate midpoints ---
        if midpoints is not None and len(midpoints) > 0:
            mc = to_plot(midpoints)
            ax.scatter(mc[:, 0], mc[:, 1],
                       c='lightgrey', edgecolors='grey', linewidths=0.4,
                       s=14, zorder=2, label='Midpoints')

        # --- Cones ---
        if len(blue_cones) > 0:
            bc = to_plot(blue_cones)
            ax.scatter(bc[:, 0], bc[:, 1],
                       c='dodgerblue', s=30, zorder=3, label='Blue')

        if len(yellow_cones) > 0:
            yc = to_plot(yellow_cones)
            ax.scatter(yc[:, 0], yc[:, 1],
                       c='gold', edgecolors='darkorange', linewidths=0.5,
                       s=30, zorder=3, label='Yellow')

        # --- Centreline ---
        if centreline is not None and len(centreline) > 0:
            cl = to_plot(centreline)
            ax.plot(cl[:, 0], cl[:, 1], 'g--', lw=1.5, label='Centreline', zorder=4)

        # --- Car triangle (pointing up = forward) ---
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
