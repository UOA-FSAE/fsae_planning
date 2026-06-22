"""
Skidpad (figure-8) track geometry.

The FS skidpad is two circles driven as a figure-8.  This module reconstructs
that geometry from the accumulated cone map and builds a single closed,
tangent-continuous centreline the car can lap repeatedly:

build_figure8
    Cluster the cones into the two circles (2-means), fit a circle to each
    (algebraic Kåsa fit), and stitch them into one figure-8 loop that crosses
    cleanly at the point between the two circle centres.

path_deviation
    Shortest distance from the car to the centreline — used to decide when the
    car has slid off the lane ("spun off course").

The figure-8 loop is returned in track order (last point ≈ first point) so it
drops straight into localisation.roll_loop_to_car for a forward planning window.
"""
import numpy as np

# Default lane half-width (centreline → cone line) used only if the cone radial
# spread is degenerate.  Real skidpad lane width is ~3 m → ~1.5 m half-width.
_DEFAULT_HALF_WIDTH = 1.5
_MIN_CONES          = 8     # need at least this many cones to attempt a fit


class Figure8Track:
    """A reconstructed figure-8 centreline plus the geometry it was built from."""

    __slots__ = ('loop', 'centres', 'lane_radius', 'half_width')

    def __init__(self, loop, centres, lane_radius, half_width):
        self.loop        = loop          # (N, 2) closed centreline, track order
        self.centres     = centres       # (2, 2) the two circle centres
        self.lane_radius = lane_radius   # mean centreline radius (m)
        self.half_width  = half_width    # centreline → cone line distance (m)


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Algebraic (Kåsa) least-squares circle fit.

    Solves x² + y² = 2·a·x + 2·b·y + c for (a, b, c); the centre is (a, b) and
    the radius is sqrt(c + a² + b²).  Returns (centre[2], radius).
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(points))])
    rhs = x * x + y * y
    sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
    centre = sol[:2]
    radius = float(np.sqrt(max(0.0, sol[2] + centre[0] ** 2 + centre[1] ** 2)))
    return centre, radius


def _two_means(points: np.ndarray, iters: int = 20) -> list[np.ndarray]:
    """
    Split points into two clusters (the two skidpad circles) with Lloyd's
    algorithm, seeded by the two points that are farthest apart.
    """
    d = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    i, j = np.unravel_index(int(np.argmax(d)), d.shape)
    c = np.array([points[i], points[j]], dtype=np.float64)

    labels = np.zeros(len(points), dtype=int)
    for _ in range(iters):
        d0 = np.linalg.norm(points - c[0], axis=1)
        d1 = np.linalg.norm(points - c[1], axis=1)
        new_labels = (d1 < d0).astype(int)
        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels
        for k in (0, 1):
            members = points[labels == k]
            if len(members):
                c[k] = members.mean(axis=0)

    return [points[labels == 0], points[labels == 1]]


def _arc(centre: np.ndarray, radius: float,
         start_ang: float, sweep: float, n: int) -> np.ndarray:
    """Sample n points along a circular arc from start_ang sweeping `sweep` rad."""
    ang = start_ang + np.linspace(0.0, sweep, n, endpoint=False)
    return centre + radius * np.column_stack([np.cos(ang), np.sin(ang)])


def build_figure8(
    blue: np.ndarray,
    yellow: np.ndarray,
    n_per_circle: int = 180,
) -> Figure8Track | None:
    """
    Reconstruct the figure-8 centreline from the cone map.

    Cone colour is irrelevant to the geometry (both circles use both colours as
    inner/outer boundaries), so blue and yellow are pooled.  The cones are split
    into the two circles, each circle is fitted, and the two are stitched into a
    single figure-8 that crosses at the point between the centres.

    The crossing is built tangent-continuous: circle 0 is traversed CCW starting
    from the point facing circle 1, and circle 1 is traversed CW starting from
    the point facing circle 0, so travel direction through the crossing matches
    on both passes (the defining property of a figure-8).

    Returns a Figure8Track, or None if there are too few cones to fit.
    """
    cones = [c for c in (blue, yellow) if len(c) > 0]
    if not cones:
        return None
    pts = np.vstack(cones)
    if len(pts) < _MIN_CONES:
        return None

    clusters = _two_means(pts)
    if any(len(c) < 3 for c in clusters):
        return None

    centres, radii, half_widths = [], [], []
    for cl in clusters:
        centre, _ = fit_circle(cl)
        radial = np.linalg.norm(cl - centre, axis=1)
        centres.append(centre)
        radii.append(float(radial.mean()))                       # centreline radius
        half_widths.append(float((radial.max() - radial.min()) * 0.5))

    # Order the two circles left→right so the loop winding is deterministic.
    order = np.argsort([c[0] for c in centres])
    c0, c1 = centres[order[0]], centres[order[1]]
    r0, r1 = radii[order[0]],   radii[order[1]]

    u = c1 - c0
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-6:
        return None
    u = u / u_norm

    ang0 = float(np.arctan2(u[1], u[0]))          # c0 → point facing c1
    ang1 = ang0 + np.pi                            # c1 → point facing c0

    # Circle 0 CCW (+sweep), circle 1 CW (-sweep): travel through the crossing
    # is the same direction on both, giving a true figure-8 rather than an O.
    arc0 = _arc(c0, r0, ang0, 2.0 * np.pi,  n_per_circle)
    arc1 = _arc(c1, r1, ang1, -2.0 * np.pi, n_per_circle)
    loop = np.vstack([arc0, arc1, arc0[:1]])       # close back to the start

    half_width = float(np.mean(half_widths))
    if not np.isfinite(half_width) or half_width < 1e-3:
        half_width = _DEFAULT_HALF_WIDTH

    return Figure8Track(
        loop=loop,
        centres=np.array([c0, c1]),
        lane_radius=float(np.mean([r0, r1])),
        half_width=half_width,
    )


def path_deviation(loop: np.ndarray, car_pos: np.ndarray) -> float:
    """Shortest distance from car_pos to the centreline polyline (metres)."""
    if loop is None or len(loop) < 2:
        return float('inf')

    a = loop[:-1]
    b = loop[1:]
    ab = b - a
    ab2 = np.einsum('ij,ij->i', ab, ab)
    ab2 = np.where(ab2 < 1e-12, 1.0, ab2)
    t = np.einsum('ij,ij->i', car_pos - a, ab) / ab2
    t = np.clip(t, 0.0, 1.0)
    proj = a + t[:, None] * ab
    return float(np.min(np.linalg.norm(car_pos - proj, axis=1)))
