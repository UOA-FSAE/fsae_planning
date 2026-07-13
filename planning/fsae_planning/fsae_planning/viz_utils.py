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
        self._zoom = 1.0   # user scroll-zoom multiplier (>1 = zoomed out)
        self._fig.canvas.mpl_connect('scroll_event', self._on_scroll)
        self._fig.tight_layout()
        plt.show(block=False)

    def _on_scroll(self, event) -> None:
        """Scroll up = zoom in, scroll down = zoom out (persists across frames)."""
        if event.button == 'up':
            self._zoom *= 0.9
        elif event.button == 'down':
            self._zoom *= 1.1
        self._zoom = float(min(max(self._zoom, 0.1), 20.0))

    def _set_limits(self, ax, cx: float, cy: float, hx: float, hy: float) -> None:
        """Set axis limits centred at (cx, cy) with half-extents scaled by zoom."""
        hx *= self._zoom
        hy *= self._zoom
        ax.set_xlim(cx - hx, cx + hx)
        ax.set_ylim(cy - hy, cy + hy)

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
        local_mode: bool = False,
        start_pos: np.ndarray | None = None,
        drift: dict | None = None,
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

        # --- Centreline (dashed while mapping, solid closed loop in local mode) ---
        if centreline is not None and len(centreline) > 0:
            cl = to_plot(centreline)
            if local_mode:
                ax.plot(cl[:, 0], cl[:, 1], 'g-', lw=2.0,
                        label='Closed loop', zorder=4)
            else:
                ax.plot(cl[:, 0], cl[:, 1], 'g--', lw=1.5,
                        label='Centreline', zorder=4)

        # --- Start / loop-closure marker ---
        if start_pos is not None:
            sp = to_plot(start_pos)
            ax.scatter(sp[:, 0], sp[:, 1], marker='*', s=180,
                       c='red', edgecolors='black', linewidths=0.6,
                       zorder=5, label='Start')

        # --- Car triangle (pointing up = forward) ---
        ax.add_patch(plt.Polygon(
            [[0, 2.0], [-1.0, -0.8], [1.0, -0.8]], color='black', zorder=6
        ))

        if local_mode:
            # Closed-loop view: fit the whole loop so the full track is visible.
            self._fit_to_loop(ax, centreline, to_plot)
            title = 'LOCALISATION MODE — closed-loop tracking'
            if drift is not None:
                ax.text(
                    0.02, 0.98,
                    f'drift  mean={drift["mean"]:.2f} m  max={drift["max"]:.2f} m',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8,
                    bbox=dict(boxstyle='round', fc='white', ec='grey', alpha=0.8),
                )
        else:
            w = self._w
            self._set_limits(ax, 0.0, w * 0.35, w * 0.5, w * 0.5)
            title = 'Centreline Planner — Ego View'

        ax.set_aspect('equal')
        ax.set_xlabel('← left of car   |   right of car →')
        ax.set_ylabel('distance ahead (m)')
        ax.set_title(f'{title}   (scroll to zoom)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.pause(0.001)

    def _fit_to_loop(self, ax, centreline: np.ndarray | None, to_plot) -> None:
        """Auto-scale the axes to fit the whole closed loop, with a margin."""
        if centreline is None or len(centreline) == 0:
            w = self._w
            self._set_limits(ax, 0.0, 0.0, w * 0.5, w * 0.5)
            return
        cl = to_plot(centreline)
        margin = 5.0
        cx = float((cl[:, 0].min() + cl[:, 0].max()) * 0.5)
        cy = float((cl[:, 1].min() + cl[:, 1].max()) * 0.5)
        hx = float((cl[:, 0].max() - cl[:, 0].min()) * 0.5) + margin
        hy = float((cl[:, 1].max() - cl[:, 1].min()) * 0.5) + margin
        self._set_limits(ax, cx, cy, hx, hy)
