r"""
Track mesh: ONE triangulation shared by the cone-wall mesh and the corridor.

build_track_mesh() Delaunay-triangulates every known cone (both colours
together) and then repairs the result so that each colour's boundary line is
an actual edge of the mesh, before anything reads walls or gates off it.
boundary.build_path_walls takes both its wall segments (same-colour edges)
and its corridor gates (cross-colour edges of mixed triangles) from this same
object, so the two can never disagree.

Why a repair step is needed
---------------------------
Delaunay is purely geometric: it has no notion of "each colour forms a line".
Where two track legs run close together with the SAME colour facing each
other (a narrow infield between two blue rows), a cone of the neighbouring
leg can sit inside the circumcircle of the real boundary edge B1-B2:

        B1 ------ B2        <- real outer wall of the corner (4.7 m)
           \    /
             A              <- neighbouring leg's blue cone, 1.2 m behind it

Delaunay then drops B1-B2 and connects B1-A-B2 instead. Nothing is "wrong"
colour-wise (all blue), so the wall mesh simply loses B1-B2, and the corridor
gets gates Y-A that reach past the real wall, pulling their midpoints (and the
path) towards the other leg. comp_test_map_3 has this at b33-b32-b30.

The repair decides, per colour, which cones form each boundary line (every
cone lies on one line; a line turns smoothly and has roughly even spacing),
and when the line skips a cone that demonstrably belongs to ANOTHER line, it
inserts the skipped edge ("bridge") into the mesh by constrained edge
insertion. Gates that crossed it disappear, and the region is re-triangulated
from the real wall cones.

A bridge is only inserted when all of these hold:
  * the skipped cone A has at least 4 same-colour mesh neighbours (its own
    line's two plus the bridge's two ends);
  * the line decomposition (_decompose_lines) picked the bridge as part of
    the B1/B2 line;
  * the same decomposition put A in the middle of a DIFFERENT line (two wall
    neighbours, neither of them B1 or B2), so A is claimed by another
    boundary, not left orphaned;
  * the bridge only crosses mesh edges that run from A to the other colour.
    In other words it only slides past A. It never crosses a wall, another
    bridge, or any edge that doesn't touch A.
On a track where no two same-colour rows are close enough to interleave,
nothing qualifies, and the mesh is plain Delaunay (identical to the old
behaviour).
"""
import math

import numpy as np
from scipy.spatial import Delaunay

# Same meaning as boundary._WALL_MAX_DIST (defined there, passed in here) --
# repeated only as this module's default.
_DEFAULT_MAX_DIST = 7.0

# --- boundary-line decomposition (_decompose_lines) --------------------------
# Largest direction change a line may make at one cone. Generous on purpose:
# the inner cones of a hairpin legitimately turn 60-80 deg each. An infield
# "rung" between two parallel rows turns ~90 deg at both ends, so it rarely
# wins against a cone's own along-row neighbours anyway (costs below).
_LINE_MAX_TURN = math.radians(80.0)
# Cost scales for one cone's (prev, cone, next) triple: turn angle in radians
# over _LINE_TURN_SCALE, plus |ln(spacing ratio)| over _LINE_RATIO_SCALE. A
# straight, evenly spaced row costs ~0. Doubling or halving the spacing (ln 2)
# costs about as much as a ~30 deg turn.
_LINE_TURN_SCALE = 0.5
_LINE_RATIO_SCALE = 0.7
# Extra cost per bridge edge in a triple, so a real mesh edge wins a tie.
_LINE_BRIDGE_COST = 0.25
# A cone must have at least this many same-colour mesh neighbours to be
# skipped by a bridge: two for its own line and two for the bridge ends.
_BRIDGE_MIN_SKIP_DEGREE = 4

_EPS = 1e-9


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Twice the signed area of triangle abc (> 0 when counter-clockwise)."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _in_circumcircle(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    """True if d lies strictly inside the circumcircle of triangle abc (any winding)."""
    ax, ay = a[0] - d[0], a[1] - d[1]
    bx, by = b[0] - d[0], b[1] - d[1]
    cx, cy = c[0] - d[0], c[1] - d[1]
    det = ((ax * ax + ay * ay) * (bx * cy - cx * by)
           - (bx * bx + by * by) * (ax * cy - cx * ay)
           + (cx * cx + cy * cy) * (ax * by - bx * ay))
    return det * _orient(a, b, c) > _EPS


class TrackMesh:
    """
    Triangulated cone set: points, per-point colour, and a triangle list that
    may differ from plain Delaunay by the inserted bridges (see module
    docstring).

    Exposes `simplices` and `find_simplex(pt)` with the same meaning as
    scipy.spatial.Delaunay, so corridor code written against a Delaunay object
    works on it unchanged.
    """

    def __init__(self, points: np.ndarray, is_blue: np.ndarray, simplices: np.ndarray):
        self.points = points
        self.is_blue = is_blue
        self.simplices = np.asarray(simplices, dtype=np.int64).reshape(-1, 3)
        # Inserted bridges as (u, w, skipped) vertex indices -- debug/tests.
        self.bridges: list[tuple[int, int, int]] = []
        self._locate_cache = None
        self._edge_cache = None

    # -- queries -------------------------------------------------------------

    def edges(self) -> set[tuple[int, int]]:
        """Every mesh edge as a sorted (i, j) vertex-index pair."""
        if self._edge_cache is None:
            s = self.simplices
            e = np.sort(np.vstack([s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]]]), axis=1)
            self._edge_cache = set(map(tuple, np.unique(e, axis=0).tolist()))
        return self._edge_cache

    def wall_segments(
        self, max_dist: float,
    ) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
        """Same-colour mesh edges no longer than max_dist, as (blue_segs, yellow_segs)."""
        blue_segs: list = []
        yellow_segs: list = []
        pts = self.points
        for i, j in self.edges():
            if self.is_blue[i] != self.is_blue[j]:
                continue
            if float(np.linalg.norm(pts[i] - pts[j])) > max_dist:
                continue
            (blue_segs if self.is_blue[i] else yellow_segs).append((pts[i], pts[j]))
        return blue_segs, yellow_segs

    def find_simplex(self, pt) -> int:
        """Index of a triangle containing pt (boundary inclusive), or -1."""
        hits = self.find_simplices(pt)
        return hits[0] if hits else -1

    def find_simplices(self, pt) -> list[int]:
        """
        Every triangle containing pt, boundary inclusive: two when pt sits on
        a shared edge, more on a shared vertex. Callers that care which side
        of an edge they start on (the corridor walk) choose among these
        instead of trusting an arbitrary first hit.
        """
        if self._locate_cache is None:
            tri = self.points[self.simplices]              # (K, 3, 2)
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            area = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
                    - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
            self._locate_cache = (a, b, c, np.sign(area))
        a, b, c, sgn = self._locate_cache
        p = np.asarray(pt, dtype=np.float64).reshape(2)

        def side(u, v):
            return ((v[:, 0] - u[:, 0]) * (p[1] - u[:, 1])
                    - (v[:, 1] - u[:, 1]) * (p[0] - u[:, 0])) * sgn

        tol = -1e-9
        inside = (side(a, b) >= tol) & (side(b, c) >= tol) & (side(c, a) >= tol) & (sgn != 0)
        return [int(h) for h in np.nonzero(inside)[0]]

    # -- constrained edge insertion -------------------------------------------

    def insert_bridge(self, u: int, w: int, skipped: int) -> bool:
        """
        Force edge u-w into the mesh, provided it only slides past `skipped`:
        every mesh edge the open segment u-w crosses must join `skipped` to a
        cone of the OTHER colour. Otherwise the mesh is left untouched and
        False is returned.

        The triangles the segment passes through are removed, and each side of
        the new edge is re-triangulated as a constrained-Delaunay
        pseudo-polygon (Anglada 1997). The result stays a valid planar
        triangulation, so the corridor's wall-safety argument (a triangle's
        interior can't reach any other triangle's edges) still holds.
        """
        pts = self.points
        pu, pw = pts[u], pts[w]
        d = pw - pu
        colour = bool(self.is_blue[u])
        if bool(self.is_blue[w]) != colour or bool(self.is_blue[skipped]) != colour:
            return False
        if (min(u, w), max(u, w)) in self.edges():
            return True

        # Every mesh edge not touching u or w, tested against the open
        # segment u-w in one vectorised pass.
        e = np.array([ij for ij in self.edges() if u not in ij and w not in ij], dtype=np.int64)
        if len(e) == 0:
            return False
        pi, pj = pts[e[:, 0]], pts[e[:, 1]]

        def orient(a, b, c):
            return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - \
                (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])

        o1 = orient(pu, pw, pi)
        o2 = orient(pu, pw, pj)
        o3 = orient(pi, pj, pu)
        o4 = orient(pi, pj, pw)

        # A cone lying on the open segment itself blocks the bridge outright.
        on_line = np.concatenate([np.abs(o1) < _EPS, np.abs(o2) < _EPS])
        if on_line.any():
            q = np.concatenate([pi, pj])[on_line]
            t = ((q - pu) @ d) / float(d @ d)
            if ((t > 0.0) & (t < 1.0)).any():
                return False

        proper = ((o1 > _EPS) & (o2 < -_EPS)) | ((o1 < -_EPS) & (o2 > _EPS))
        proper &= ((o3 > _EPS) & (o4 < -_EPS)) | ((o3 < -_EPS) & (o4 > _EPS))
        crossed = []   # (t along u->w, left vertex, right vertex)
        for idx in np.nonzero(proper)[0]:
            i, j = int(e[idx, 0]), int(e[idx, 1])
            # Every crossed edge must run from `skipped` to the other colour.
            if skipped not in (i, j):
                return False
            other = j if i == skipped else i
            if bool(self.is_blue[other]) == colour:
                return False
            t = float(o3[idx] / (o3[idx] - o4[idx]))
            left, right = (i, j) if o1[idx] > 0 else (j, i)
            crossed.append((t, left, right))

        if not crossed:
            return False
        crossed.sort()
        crossed_set = {(min(a, b), max(a, b)) for _, a, b in crossed}

        left_chain: list[int] = []
        right_chain: list[int] = []
        for _, lv, rv in crossed:
            if not left_chain or left_chain[-1] != lv:
                left_chain.append(lv)
            if not right_chain or right_chain[-1] != rv:
                right_chain.append(rv)

        keep = []
        removed = 0
        for tri in self.simplices:
            i, j, k = int(tri[0]), int(tri[1]), int(tri[2])
            tri_edges = ((min(i, j), max(i, j)), (min(j, k), max(j, k)), (min(i, k), max(i, k)))
            if any(e in crossed_set for e in tri_edges):
                removed += 1
            else:
                keep.append((i, j, k))

        new = _triangulate_pseudo_polygon(pts, u, w, left_chain) + \
            _triangulate_pseudo_polygon(pts, u, w, right_chain)
        # Sanity: removing k crossed edges frees k+1 triangles; the two
        # re-triangulated sides must fill exactly that region.
        if removed != len(crossed) + 1 or len(new) != removed:
            return False

        self.simplices = np.array(keep + new, dtype=np.int64).reshape(-1, 3)
        self.bridges.append((u, w, skipped))
        self._locate_cache = None
        self._edge_cache = None
        return True


def _triangulate_pseudo_polygon(
    pts: np.ndarray, a: int, b: int, chain: list[int],
) -> list[tuple[int, int, int]]:
    """
    Constrained-Delaunay triangulation of the polygon a, chain..., b (closed
    by the edge b-a). Picks the chain vertex whose circle through a, b holds
    no other chain vertex, then recurses on both sides.
    """
    if not chain:
        return []
    c = 0
    for i in range(1, len(chain)):
        if _in_circumcircle(pts[a], pts[b], pts[chain[c]], pts[chain[i]]):
            c = i
    tris = [(a, chain[c], b)]
    tris += _triangulate_pseudo_polygon(pts, a, chain[c], chain[:c])
    tris += _triangulate_pseudo_polygon(pts, chain[c], b, chain[c + 1:])
    return tris


def _turn(p_prev, p, p_next) -> float:
    """Direction change (rad) at p along prev -> p -> next; 0 for a straight line."""
    v1x, v1y = p[0] - p_prev[0], p[1] - p_prev[1]
    v2x, v2y = p_next[0] - p[0], p_next[1] - p[1]
    n1 = math.hypot(v1x, v1y)
    n2 = math.hypot(v2x, v2y)
    if n1 < 1e-9 or n2 < 1e-9:
        return math.pi
    c = (v1x * v2x + v1y * v2y) / (n1 * n2)
    return math.acos(max(-1.0, min(1.0, c)))


def _decompose_lines(
    pts: list,
    neigh: dict[int, set[int]],
    centres: set[int],
    bridge_ends: set[tuple[int, int]],
) -> dict[int, set[int]]:
    """
    Split one colour's cones into boundary lines: give every cone at most two
    line neighbours, chosen best-first by how straight and evenly spaced the
    resulting (prev, cone, next) triple is.

    pts:         cone positions as plain (x, y) sequences.
    neigh:       candidate neighbours per cone (mesh edges plus bridges).
    centres:     cones whose own triples are considered (the neighbourhood
                 of the candidate bridges); other cones can only be line ends.
    bridge_ends: the candidate edges that are bridges, as sorted pairs.

    Returns each cone's accepted line neighbours. A triple is accepted only if
    its centre cone has no triple yet, and each end either already links to
    the centre or can still take one more neighbour without turning too
    sharply. Cones on a straight, evenly spaced row claim their neighbours
    first, which is what lets a real row keep its own cone when another row
    runs right next to it.
    """
    triples = []
    for v in centres:
        nb = sorted(neigh[v])
        pv = pts[v]
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                a, b = nb[x], nb[y]
                turn = _turn(pts[a], pv, pts[b])
                if turn > _LINE_MAX_TURN:
                    continue
                la = math.hypot(pts[a][0] - pv[0], pts[a][1] - pv[1])
                lb = math.hypot(pts[b][0] - pv[0], pts[b][1] - pv[1])
                if la < 1e-6 or lb < 1e-6:
                    continue
                n_bridge = ((min(a, v), max(a, v)) in bridge_ends) + \
                    ((min(b, v), max(b, v)) in bridge_ends)
                cost = (turn / _LINE_TURN_SCALE
                        + abs(math.log(la / lb)) / _LINE_RATIO_SCALE
                        + _LINE_BRIDGE_COST * n_bridge)
                triples.append((cost, v, a, b))
    triples.sort()

    line: dict[int, set[int]] = {v: set() for v in neigh}
    centred: set[int] = set()

    def can_take(x: int, v: int) -> bool:
        """Can end cone x gain line neighbour v?"""
        if v in line[x]:
            return True
        if x in centred or len(line[x]) >= 2:
            return False
        if len(line[x]) == 1:
            (other,) = tuple(line[x])
            return _turn(pts[other], pts[x], pts[v]) <= _LINE_MAX_TURN
        return True

    for _, v, a, b in triples:
        if v in centred or not line[v] <= {a, b}:
            continue
        if not (can_take(a, v) and can_take(b, v)):
            continue
        for x in (a, b):
            line[v].add(x)
            line[x].add(v)
            if len(line[x]) == 2:
                centred.add(x)
        centred.add(v)

    return line


def _plan_bridges(mesh: TrackMesh, max_dist: float) -> list[tuple[float, int, int, int]]:
    """
    Find the bridges build_track_mesh should insert, as (length, u, w, skipped).
    See the module docstring for the conditions.
    """
    pts = mesh.points.tolist()
    is_blue = mesh.is_blue.tolist()
    edges = mesh.edges()

    def dist(i: int, j: int) -> float:
        return math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1])

    same: dict[int, set[int]] = {v: set() for v in range(len(pts))}
    for i, j in edges:
        if is_blue[i] == is_blue[j] and dist(i, j) <= max_dist:
            same[i].add(j)
            same[j].add(i)

    # Candidate bridges: two same-colour neighbours U, W of a crowded cone V
    # that are close enough to be a wall edge themselves but aren't joined.
    candidates: dict[tuple[int, int], int] = {}
    for v, nbs in same.items():
        if len(nbs) < _BRIDGE_MIN_SKIP_DEGREE:
            continue
        nb = sorted(nbs)
        for x in range(len(nb)):
            for y in range(x + 1, len(nb)):
                u, w = nb[x], nb[y]
                if (u, w) not in edges and dist(u, w) <= max_dist:
                    candidates.setdefault((u, w), v)
    if not candidates:
        return []

    neigh = {v: set(nbs) for v, nbs in same.items()}
    for u, w in candidates:
        neigh[u].add(w)
        neigh[w].add(u)
    bridge_ends = set(candidates)

    # Only the neighbourhood of the candidates can change the outcome: two
    # hops out from every cone a candidate touches.
    centres = set()
    for (u, w), v in candidates.items():
        centres.update((u, w, v))
    for _ in range(2):
        centres |= {n for c in centres for n in neigh[c]}

    # Colours never share a mesh neighbour set, so one pass decomposes both.
    line = _decompose_lines(pts, neigh, centres, bridge_ends)

    plans = []
    for (u, w), v in candidates.items():
        # Only skip V if the bridge made it into a line and another line has
        # claimed V outright.
        if w in line[u] and len(line[v]) == 2 and not (line[v] & {u, w}):
            plans.append((dist(u, w), u, w, v))
    plans.sort()
    return plans


def build_track_mesh(
    blue: np.ndarray,
    yellow: np.ndarray,
    max_dist: float = _DEFAULT_MAX_DIST,
    repair: bool = True,
) -> TrackMesh | None:
    """
    Triangulate every cone of both colours once, then insert the boundary
    bridges described in the module docstring (unless repair=False, which
    gives plain Delaunay).

    Returns None when no triangulation exists: fewer than 3 cones in total,
    or a degenerate (e.g. collinear) set that makes Qhull raise.
    """
    blue = np.asarray(blue, dtype=np.float64).reshape(-1, 2)
    yellow = np.asarray(yellow, dtype=np.float64).reshape(-1, 2)
    points = np.vstack([blue, yellow])
    is_blue = np.concatenate([
        np.ones(len(blue), dtype=bool),
        np.zeros(len(yellow), dtype=bool),
    ])
    if len(points) < 3:
        return None
    try:
        tri = Delaunay(points)
    except Exception:
        return None

    mesh = TrackMesh(points, is_blue, tri.simplices)
    if repair:
        for _, u, w, v in _plan_bridges(mesh, max_dist):
            mesh.insert_bridge(u, w, v)
    return mesh
