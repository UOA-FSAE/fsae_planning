"""
Boundary detection: cone-wall mesh centreline planner.

build_path_walls  — the planner.  Delaunay-triangulates every known cone and
                    keeps the same-colour edges as a wall mesh (see
                    build_wall_segments_delaunay); each MIXED triangle (one
                    colour split 2-1) contributes its 2 cross-colour edges
                    directly as a centreline "gate pair" (see
                    _build_corridor_path), which is wall-safe by
                    construction -- a triangle's interior can't reach any
                    other triangle's edges, so no runtime wall-crossing
                    check is needed to chain them. The chain is clamped to a
                    fixed arc-length horizon and smoothed into a centreline.
"""
import math

import numpy as np
from scipy.spatial import Delaunay

from fsae_planning.cone_sorting import filter_cones_window
from fsae_planning.path_utils import (
    build_local_path,
    DEFAULT_SMOOTH_PER_PT,
    smooth_centreline,
)

# ---------------------------------------------------------------------------
# Cone-wall barrier planner
# ---------------------------------------------------------------------------

# _WALL_MAX_DIST/_WALL_MID_DIST: wide enough that a normal cone gap never
# breaks the link/pairing (missing a genuine link truncates the wall at that
# gap), but well under a typical track width so a wall never bridges across
# to the opposite boundary or a midpoint pairs with the wrong-side cone.
#
# _WALL_MAX_DIST is no longer the primary thing stopping a wall from bridging
# across the track (see build_wall_segments_delaunay) -- the triangulation
# does that structurally. It now only trims abnormally long edges that a
# Delaunay triangulation can produce at the convex hull / in sparse regions
# of a long thin point cloud (exactly the shape of a track boundary), and
# it's still what build_wall_segments (used as-is by the skidpad planner)
# uses as its sole distance cutoff.
_WALL_MAX_DIST      = 7.0      # metres — max dist to link same-colour cones into wall
_WALL_MID_DIST      = 4.0      # metres — max blue-yellow dist for midpoint candidates
# A genuine opposite-colour pair spans roughly the track width. A pair much
# closer than that is far more likely to be a real cone paired with a stray/
# mislabelled cone sitting right next to it than a real narrow section of
# track -- accepting it anyway pulls the midpoint (and the path) toward that
# stray cone instead of along the actual corridor.
_WALL_MIN_MID_DIST  = 1.5      # metres — min blue-yellow dist for midpoint candidates
# Corridor-graph gate generation (_build_corridor_path) widens the
# blue-yellow distance bounds relative to _gen_midpoints' exclusive-NN
# matching above. _build_corridor_path (the 'delaunay' method) doesn't need
# a perpendicularity gate to disambiguate competing candidates -- a mixed
# triangle's 2 cross-colour edges are used directly, no selection needed --
# so these bounds are its only sanity net: an edge shorter than mid_min is
# almost always a stray/mislabelled cone next to a genuine one, and one
# longer than mid_max means the triangulation bridged a gap with a missing
# intermediate cone. 5.0 m covers the real cone-spacing p99 (~4.2 m,
# measured from a recorded track run); 1.2 m still rejects an implausibly
# narrow gap.
_WALL_MID_MIN_DELAUNAY = 1.2    # metres
_WALL_MID_MAX_DELAUNAY = 5.0    # metres
# Large enough that no distance/angle saving in the greedy walk ever makes
# crossing into the wall mesh worth it — acts as a hard constraint expressed
# as a cost, not a real magnitude to be weighed against other terms.
_WALL_CROSS_PENALTY = 100000.0   # cost per wall segment crossed by a path step
_WALL_PATH_MAX_STEP = 10.0     # metres — max step between consecutive path midpoints
# Max midpoints in the 'nn' method's cost-weighted greedy walk (_build_wall_path,
# kept for rollback/comparison -- see build_path_walls). 18 * _WALL_PATH_MAX_STEP
# comfortably exceeds _WALL_PLAN_HORIZON (25 m) assuming near-_WALL_PATH_MAX_STEP
# hops throughout.
_WALL_PATH_MAX_WALK = 18       # max midpoints in the constructed path
# Hard cap on gates walked by the default 'delaunay' corridor walk
# (_walk_corridor). Unlike _WALL_PATH_MAX_WALK above, this is NOT how that
# walk is meant to reach _WALL_PLAN_HORIZON -- it now terminates by actual arc
# length (see _walk_corridor's max_arc_length) because dense cone spacing
# produces much shorter gate-to-gate hops than _WALL_PATH_MAX_STEP, so a
# count-only cap (the old shared _WALL_PATH_MAX_WALK = 18) could exhaust
# itself well inside 25 m of actual arc length and truncate early for no
# geometric reason. This cap exists only as a runaway-loop backstop, sized
# generously past any plausible plan_horizon / typical-hop-length ratio.
_WALL_CORRIDOR_MAX_WALK = 120
# Softest per-step turn the walk will accept, as cos(max turn).  The old walk
# used a hard 0.0 (a 90° per-step ceiling): at a tight hairpin every next
# midpoint sits >90° off the current travel direction, so the walk stalled and
# the path truncated into the corner (car then drove straight off).  -0.5 (~120°)
# lets the chain follow a genuine hairpin; the angle cost + wall-cross penalty
# still keep it from doubling back or hopping to a parallel track.
#
# This was set below -1.0 for a time to test corner-truncation on a 60-90 deg
# turn (a cosine can never go that low, so the `fwd_dot <= _WALL_MAX_TURN_COS`
# gate never fired and the walk no longer stopped on turn-angle grounds at
# all) -- that left ONLY the wall-cross penalty guarding against the walk
# doubling back on itself, and it doubled back (path pointing behind the car
# at a teardrop pinch). Restored to -0.5.
_WALL_MAX_TURN_COS  = -0.5
# Default arc-length horizon (m) the published centreline is clamped to before
# smoothing.  Far midpoints beyond this are dropped so the near path in front of
# the car does not change as the lookahead grows, and the global spline is not
# dragged by distant apex points.  Kept >= the controller's ~24 m speed scan
# (control_utils.curvature_speed's scan_end) — a tight hairpin (~2 m radius,
# v_target ~2.7 m/s) approached at v_max=15 m/s needs ~24 m to brake for at a
# realistic achieved deceleration (~4.5 m/s2, well under the 9 m/s2 hard limit
# once the controller's own speed-request low-pass and rate limits are accounted for);
# the previous 15 m horizon only revealed such a corner a couple of car-lengths
# before the car needed to already be nearly stopped, causing steering
# saturation and a spin-out.
_WALL_PLAN_HORIZON  = 25.0


def build_wall_segments(
    cones: np.ndarray,
    max_dist: float = _WALL_MAX_DIST,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (p1, p2) segments connecting every same-colour cone pair within max_dist."""
    n = len(cones)
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if float(np.linalg.norm(cones[i] - cones[j])) <= max_dist:
                segs.append((cones[i], cones[j]))
    return segs


def segment_crosses_walls(
    p1: np.ndarray,
    p2: np.ndarray,
    wall_segs: list[tuple[np.ndarray, np.ndarray]],
) -> bool:
    """True if the segment p1→p2 crosses any cone-wall segment."""
    return any(_seg_intersect(p1, p2, w1, w2) for (w1, w2) in wall_segs)


_SEG_EPS = 1e-9   # tolerance on the [0, 1] parametric range -- see _seg_intersect


def _seg_intersect(
    a1: np.ndarray, a2: np.ndarray,
    b1: np.ndarray, b2: np.ndarray,
) -> bool:
    """
    True if segment a1->a2 intersects segment b1->b2, INCLUDING endpoint
    contact and collinear overlap.

    The old strict `0.0 < t < 1.0 and 0.0 < u < 1.0` test excluded a crossing
    that lands exactly on one of the sampled waypoints: a path vertex sitting
    ON a wall line makes both of its adjacent segments see the crossing at
    their own t=0 or t=1 boundary, which the strict inequality never counts,
    so a wall contact at a waypoint escaped detection from both sides at
    once. Widening to a small epsilon catches that without materially
    widening genuine near-misses (t/u are dimensionless segment fractions,
    not metres, so 1e-9 is not a physical clearance -- see min_cone_clearance
    in wall_centerline_planner for the actual physical safety margin).

    The old `abs(denom) < 1e-10: return False` branch also missed a
    collinear-overlap crossing entirely (denom is 0 whenever the two segments
    are parallel, including when they lie on the same line and overlap) --
    that case is now checked explicitly.
    """
    d1 = a2 - a1
    d2 = b2 - b1
    denom = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(denom) < 1e-10:
        # Parallel. Collinear (not just parallel) iff b1 lies on the line
        # through a1/a2 -- test via the same cross product with (b1 - a1).
        diff0 = b1 - a1
        cross = float(diff0[0] * d1[1] - diff0[1] * d1[0])
        d1_len = float(np.linalg.norm(d1))
        if d1_len < 1e-12 or abs(cross) > 1e-9 * max(1.0, d1_len):
            return False   # parallel but offset -- never touches
        d1_sq = float(np.dot(d1, d1))
        t0 = float(np.dot(b1 - a1, d1)) / d1_sq
        t1 = float(np.dot(b2 - a1, d1)) / d1_sq
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        return hi >= -_SEG_EPS and lo <= 1.0 + _SEG_EPS
    diff = b1 - a1
    t = float(diff[0] * d2[1] - diff[1] * d2[0]) / denom
    u = float(diff[0] * d1[1] - diff[1] * d1[0]) / denom
    return -_SEG_EPS <= t <= 1.0 + _SEG_EPS and -_SEG_EPS <= u <= 1.0 + _SEG_EPS


def build_wall_segments_delaunay(
    blue: np.ndarray,
    yellow: np.ndarray,
    max_dist: float = _WALL_MAX_DIST,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    """
    Build same-colour wall segments from a Delaunay triangulation of every
    known cone (both colours together), instead of build_wall_segments' old
    all-pairs same-colour distance linking.

    Why: build_wall_segments links ANY two same-coloured cones within
    max_dist, with no notion of what physically lies between them. A single
    mislabelled or genuinely off-track cone sitting close to the *opposite*
    boundary is enough to link into that boundary's wall mesh and draw a
    false wall segment straight across the track -- a real cone placed just
    outside the track next to the other colour's line reproduces this
    directly, and it isn't limited to nearby placements: build_wall_segments
    is fed by an unbounded car-relative window (formerly this function's
    look_radius+4 argument, since removed -- see build_path_walls), so a
    same-coloured cone anywhere in that window can complete the same false
    link regardless of how far along the track it actually is (e.g. a loop
    pinch bringing two track-distant sections close together in a straight
    line).

    Triangulating BOTH colours together fixes this structurally rather than
    by tuning a threshold: a cone genuinely between two same-coloured cones
    (regardless of ITS colour) makes Delaunay's empty-circumcircle property
    very unlikely to keep a direct edge between them, because the circle
    through any such long edge would have to avoid enclosing every cone lying
    near that line. A tighter max_dist could not achieve the same thing
    without also breaking real, sparser cone spacing.

    max_dist remains as a safety net, not the primary filter: a plain
    Delaunay triangulation of a long thin point cloud (exactly the shape of a
    track boundary) can still produce occasional abnormally long edges at the
    convex hull or in low-density regions, so any triangulation edge longer
    than max_dist is dropped regardless of colour.

    Falls back to the old all-pairs build_wall_segments (run separately per
    colour) if the triangulation itself cannot run at all -- fewer than 3
    total cones, or a degenerate/near-collinear point set, both of which
    scipy raises a QhullError for. Publishing the old method's walls that
    tick is preferable to publishing none.

    Returns (blue_segs, yellow_segs), matching build_wall_segments' shape so
    callers don't need to know which method produced them.
    """
    blue = np.asarray(blue, dtype=np.float64).reshape(-1, 2)
    yellow = np.asarray(yellow, dtype=np.float64).reshape(-1, 2)
    points = np.vstack([blue, yellow])
    is_blue = np.concatenate([
        np.ones(len(blue), dtype=bool),
        np.zeros(len(yellow), dtype=bool),
    ])

    if len(points) < 3:
        return build_wall_segments(blue, max_dist), build_wall_segments(yellow, max_dist)

    try:
        tri = Delaunay(points)
    except Exception:
        # Degenerate input (e.g. near-collinear cones on a fresh straight) --
        # triangulation unavailable this tick.
        return build_wall_segments(blue, max_dist), build_wall_segments(yellow, max_dist)

    edges = set()
    for simplex in tri.simplices:
        i, j, k = int(simplex[0]), int(simplex[1]), int(simplex[2])
        edges.add((min(i, j), max(i, j)))
        edges.add((min(j, k), max(j, k)))
        edges.add((min(i, k), max(i, k)))

    blue_segs: list = []
    yellow_segs: list = []
    for i, j in edges:
        if is_blue[i] != is_blue[j]:
            continue  # cross-colour edge -- centreline midpoints come from _gen_midpoints, not here
        if float(np.linalg.norm(points[i] - points[j])) > max_dist:
            continue  # hull / low-density triangulation artifact -- safety net
        seg = (points[i], points[j])
        (blue_segs if is_blue[i] else yellow_segs).append(seg)

    return blue_segs, yellow_segs


def debug_triangulation_edges(
    blue: np.ndarray,
    yellow: np.ndarray,
    max_dist: float = _WALL_MAX_DIST,
    mid_min: float = _WALL_MID_MIN_DELAUNAY,
    mid_max: float = _WALL_MID_MAX_DELAUNAY,
) -> list[tuple[np.ndarray, np.ndarray, str]]:
    """
    Return every edge of the same Delaunay triangulation
    build_wall_segments_delaunay/_build_corridor_path compute, each tagged
    with why it was kept or dropped. Debug/viz only -- not called by the
    planner's path building; it exists so a wall- or corridor-build problem
    can be inspected directly (was the edge never in the triangulation at
    all, or was it triangulated and then filtered out, and why) instead of
    only seeing the final filtered wall mesh / corridor.

    Tags:
      'wall'  — same colour, kept as a wall segment.
      'long'  — same colour, dropped for exceeding max_dist.
      'mid'   — cross colour, and a "gate" of at least one mixed triangle
                (see _build_corridor_path) whose other cross-colour edge is
                also within [mid_min, mid_max] -- usable as part of the
                corridor.
      'cross' — cross colour, but every mixed triangle it belongs to failed
                that length check (an implausibly short/long gate, usually
                from a missing intermediate cone or a stray/mislabelled one).

    Returns an empty list on degenerate input (fewer than 3 cones total, or a
    triangulation-raising point set) -- same conditions
    build_wall_segments_delaunay falls back on, but there's nothing
    meaningful to tag without a triangulation to draw from.
    """
    blue = np.asarray(blue, dtype=np.float64).reshape(-1, 2)
    yellow = np.asarray(yellow, dtype=np.float64).reshape(-1, 2)
    points = np.vstack([blue, yellow])
    is_blue = np.concatenate([
        np.ones(len(blue), dtype=bool),
        np.zeros(len(yellow), dtype=bool),
    ])

    if len(points) < 3:
        return []

    try:
        tri = Delaunay(points)
    except Exception:
        return []

    edges = set()
    gate_ok: dict = {}
    for simplex in tri.simplices:
        verts = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        i, j, k = verts
        for e in ((min(i, j), max(i, j)), (min(j, k), max(j, k)), (min(i, k), max(i, k))):
            edges.add(e)

        cross_edges = _classify_simplex(verts, is_blue)
        if cross_edges is None:
            continue
        lens = [float(np.linalg.norm(points[e[0]] - points[e[1]])) for e in cross_edges]
        ok = not any(length < mid_min or length > mid_max for length in lens)
        for e in cross_edges:
            gate_ok[e] = gate_ok.get(e, False) or ok

    tagged: list[tuple[np.ndarray, np.ndarray, str]] = []
    for i, j in edges:
        if is_blue[i] != is_blue[j]:
            tag = 'mid' if gate_ok.get((i, j), False) else 'cross'
        elif float(np.linalg.norm(points[i] - points[j])) > max_dist:
            tag = 'long'
        else:
            tag = 'wall'
        tagged.append((points[i], points[j], tag))

    return tagged


def _local_tangent(
    points: np.ndarray,
    same_colour: np.ndarray,
    idx: int,
    fwd,
    car_dir: np.ndarray,
    max_dist: float = _WALL_MAX_DIST,
) -> np.ndarray:
    """
    Along-track unit direction at points[idx], estimated from its two
    spatially nearest same-colour neighbours (within max_dist) -- the chord
    between them gives the tangent line; falls back to car_dir when the
    cone has fewer than one same-colour neighbour in range.

    Used by _gen_midpoints' left/right validity test (the 'nn' midpoint
    method, kept for rollback/comparison -- see its docstring for why the
    tangent is built from spatially-nearest neighbours rather than
    forward-sort order: a tight corner's forward sort mixes cones from both
    legs of the bend, pointing the tangent across the track instead of
    along it).

    `fwd(pt)` is the car-frame forward projection, used only to orient a
    two-neighbour chord consistently (near-to-far along the direction of
    travel); `car_dir` is the fallback orientation for a lone neighbour.
    """
    same_idx = np.where(same_colour)[0]
    same_idx = same_idx[same_idx != idx]
    if len(same_idx) == 0:
        return car_dir

    a = points[idx]
    d = np.linalg.norm(points[same_idx] - a, axis=1)
    order = np.argsort(d)
    near = [int(same_idx[pos]) for pos in order[:2] if d[pos] <= max_dist]
    if not near:
        return car_dir

    if len(near) == 1:
        v = points[near[0]] - a
        if float(v @ car_dir) < 0.0:
            v = -v
    else:
        k0, k1 = near
        if fwd(points[k0]) > fwd(points[k1]):
            k0, k1 = k1, k0
        v = points[k1] - points[k0]

    n = float(np.linalg.norm(v))
    return v / n if n > 1e-6 else car_dir


def _gen_midpoints(
    blue: np.ndarray,
    yellow: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    max_dist: float = _WALL_MID_DIST,
    min_dist: float = _WALL_MIN_MID_DIST,
) -> np.ndarray:
    """
    Return midpoints from an exclusive nearest-neighbour match between blue
    and yellow cones between min_dist and max_dist metres apart.

    The lower bound rejects an anchor cone's nearest opposite-colour
    candidate when it sits implausibly close (closer than a real track is
    ever narrow) -- almost always a stray/mislabelled cone next to a genuine
    one rather than a real gap, and pairing it in anyway would pull the
    midpoint (and the path) toward it. Unlike the upper bound, this does not
    fall through to the next-nearest candidate: an anchor with only an
    implausibly-close candidate in range produces no midpoint that tick
    rather than a guessed one, consistent with a real narrow track simply
    losing a midpoint at that gap.

    The denser side (more cones in view) is used as the anchor: every anchor
    cone claims its single nearest still-unclaimed opposite-colour cone, so no
    opposite-colour cone can be shared between multiple midpoints.  This
    matters most in corners, where the tighter-radius boundary has cones
    closer together than the outer one — under plain all-pairs distance
    matching, one outer cone falls within range of several inner cones and
    fans out into several midpoints pulling in different directions, which
    stays jagged even after spline smoothing.  Anchor cones are matched in
    forward order (nearest to the car first) so exclusivity resolves in
    favour of the more immediately relevant matches.

    Validity filter: the blue cone must be to the LEFT of the yellow cone
    relative to the LOCAL track direction (not the car's instantaneous heading).
    This eliminates midpoints that would land inside a boundary wall, which arise
    when a same-colour cone from an adjacent parallel track is incorrectly paired.

    Using the local track direction rather than the car heading is what keeps
    CORNER midpoints: around a bend the boundary rotates away from the car's
    current heading, so a car-frame left/right test wrongly rejected the very
    apex cones and left the spline to cut a wide line across the gap.

    The local direction is estimated per anchor cone from its two SPATIALLY
    nearest same-colour neighbours (not its neighbours in the forward-sorted
    order).  This matters at tight corners with a narrow infield (hairpins,
    chicanes): the forward-sort mixes cones from the two legs of the bend, so a
    sort-order neighbour can be a cone on the OPPOSITE leg several metres across
    the infield.  The tangent then points across the track instead of along it,
    the left/right cross-product test inverts, and cross-infield cone pairs are
    wrongly accepted — producing midpoints that cut diagonally across the track
    (car drives off the map at the apex).  Spatially-nearest neighbours are
    always on the same leg, so the tangent stays along-track and the test holds.
    """
    if len(blue) == 0 or len(yellow) == 0:
        return np.empty((0, 2), dtype=np.float64)

    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)

    def fwd(pt: np.ndarray) -> float:
        rel = pt - car_pos
        return float(rel[0] * cos_y + rel[1] * sin_y)

    blue_is_anchor = len(blue) >= len(yellow)
    anchor, other  = (blue, yellow) if blue_is_anchor else (yellow, blue)

    anchor_order = np.argsort([fwd(pt) for pt in anchor])
    n_anchor     = len(anchor_order)
    car_dir      = np.array([cos_y, sin_y], dtype=np.float64)
    claimed      = np.zeros(len(other), dtype=bool)

    _anchor_all = np.ones(n_anchor, dtype=bool)

    def local_dir(idx: int) -> np.ndarray:
        """Along-track direction at anchor cone `idx` -- see _local_tangent."""
        if n_anchor < 2:
            return car_dir
        return _local_tangent(anchor, _anchor_all, idx, fwd, car_dir)

    mids = []
    for pos, idx in enumerate(anchor_order):
        a  = anchor[idx]
        ld = local_dir(idx)

        # Signed cross product of the local track direction with (cone - anchor):
        # > 0 → cone is left of the track, < 0 → right.
        rel   = other - a
        cross = ld[0] * rel[:, 1] - ld[1] * rel[:, 0]
        if blue_is_anchor:
            valid = (~claimed) & (cross < 0.0)   # yellow must be right of blue
        else:
            valid = (~claimed) & (cross > 0.0)   # blue must be left of yellow

        cand_idx = np.where(valid)[0]
        if len(cand_idx) == 0:
            continue

        dists      = np.linalg.norm(other[cand_idx] - a, axis=1)
        best_local = int(np.argmin(dists))
        if dists[best_local] > max_dist or dists[best_local] < min_dist:
            continue

        best_idx = cand_idx[best_local]
        claimed[best_idx] = True
        mids.append((a + other[best_idx]) * 0.5)

    return np.array(mids, dtype=np.float64) if mids else np.empty((0, 2), dtype=np.float64)



def _classify_simplex(
    verts: tuple[int, int, int],
    is_blue: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Classify one Delaunay triangle by its 3 vertices' colours.

    A triangle with vertices split 2-1 between colours ("mixed") always has
    EXACTLY 2 cross-colour edges (each connecting the lone minority-colour
    vertex to one of the two majority-colour vertices) and 1 same-colour
    edge (connecting the two majority-colour vertices -- a wall edge). This
    is the geometric fact _build_corridor_path relies on: because a
    triangulation's triangles have pairwise-disjoint interiors, and a
    triangle's closed region is convex, any segment connecting two points on
    a mixed triangle's boundary (in particular, its two cross-colour-edge
    midpoints) stays entirely inside that triangle -- it can touch the
    triangle's own wall edge only at a shared vertex, and cannot reach any
    OTHER triangle's edges at all. So a path built by connecting each mixed
    triangle's 2 gate midpoints directly can never cross a wall edge,
    without needing a runtime segment-intersection check.

    Returns the triangle's 2 cross-colour edges (each as a sorted vertex-
    index pair), or None if the triangle is all one colour (0 cross-colour
    edges -- not part of the corridor).
    """
    colours = [bool(is_blue[v]) for v in verts]
    n_blue = sum(colours)
    if n_blue == 0 or n_blue == 3:
        return None

    minority_is_blue = (n_blue == 1)
    minority_pos = colours.index(minority_is_blue)
    majority_pos = [k for k in range(3) if k != minority_pos]
    return (
        tuple(sorted((verts[minority_pos], verts[majority_pos[0]]))),
        tuple(sorted((verts[minority_pos], verts[majority_pos[1]]))),
    )


def _build_corridor_graph(
    points: np.ndarray,
    is_blue: np.ndarray,
    tri: Delaunay,
    mid_min: float,
    mid_max: float,
) -> tuple[dict, dict, dict, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Classify every simplex of `tri` and assemble the corridor graph.

    Returns:
      gate_mid       : dict edge -> (2,) midpoint, one entry per accepted
                        cross-colour "gate" edge (both of its mixed
                        triangle's cross-colour edges are within
                        [mid_min, mid_max]).
      simplex_gates   : dict simplex_idx -> (edge_a, edge_b), the 2 gate
                        edges of each ACCEPTED mixed simplex.
      gate_simplices  : dict edge -> list of simplex_idx (len 1 or 2) --
                        which accepted mixed simplices use this edge as a
                        gate. A cross-colour edge always borders only mixed
                        triangles (see _classify_simplex), and at most 2
                        (1 if it's on the convex hull) -- this is what makes
                        the corridor graph a simple chain (max degree 2) at
                        every gate, no branching search needed to walk it.
      wall_segs       : same-colour edges of this SAME triangulation, as
                        (p1, p2) point pairs -- used only for the one-off
                        entry check in _walk_corridor when the car sits
                        outside the corridor mesh entirely (see there).
    """
    gate_mid: dict = {}
    simplex_gates: dict = {}
    gate_simplices: dict = {}
    wall_edges: set = set()

    for s_idx, simplex in enumerate(tri.simplices):
        verts = (int(simplex[0]), int(simplex[1]), int(simplex[2]))
        i, j, k = verts
        for e in ((min(i, j), max(i, j)), (min(j, k), max(j, k)), (min(i, k), max(i, k))):
            if is_blue[e[0]] == is_blue[e[1]]:
                wall_edges.add(e)

        cross_edges = _classify_simplex(verts, is_blue)
        if cross_edges is None:
            continue
        lens = [float(np.linalg.norm(points[e[0]] - points[e[1]])) for e in cross_edges]
        if any(length < mid_min or length > mid_max for length in lens):
            continue

        simplex_gates[s_idx] = cross_edges
        for e in cross_edges:
            gate_mid.setdefault(e, (points[e[0]] + points[e[1]]) * 0.5)
            gate_simplices.setdefault(e, []).append(s_idx)

    wall_segs = [(points[e[0]], points[e[1]]) for e in wall_edges]
    return gate_mid, simplex_gates, gate_simplices, wall_segs


def _walk_corridor(
    car_pos: np.ndarray,
    car_yaw: float,
    tri: Delaunay,
    gate_mid: dict,
    simplex_gates: dict,
    gate_simplices: dict,
    wall_segs: list[tuple[np.ndarray, np.ndarray]],
    prev_dir: np.ndarray | None = None,
    max_steps: int = _WALL_CORRIDOR_MAX_WALK,
    max_step_dist: float = _WALL_PATH_MAX_STEP,
    max_arc_length: float = _WALL_PLAN_HORIZON,
) -> np.ndarray:
    """
    Chain the corridor graph (see _build_corridor_graph) into a path,
    starting from the car's position.

    Since every gate has at most 2 neighbouring (accepted, mixed) simplices,
    once the walk has entered the graph there is at most one way to
    continue at each gate (the "other side" of the triangle just entered) --
    no per-step candidate search or wall-crossing cost needed: the walk
    steps deterministically along the chain until a dead end (a gate on the
    convex hull, or a triangle whose neighbour's gates failed the length
    gate), a already-visited simplex (loop-safety on a genuine closed-loop
    triangulation), max_arc_length worth of path walked, max_steps (a
    generous computational backstop, not the normal stopping condition --
    see _WALL_CORRIDOR_MAX_WALK), or a too-sharp turn (a degenerate/sliver
    triangle producing a nonsensical kink -- reusing _WALL_MAX_TURN_COS as a
    single O(1) sanity check per step, not a filter over many candidates).

    max_arc_length is the walk's real stopping distance (car -> last point,
    summed along the chain). build_path_walls also re-clamps the returned
    chain to plan_horizon before smoothing, so this only needs to be AT LEAST
    plan_horizon -- it is not itself the source of truth for how far the
    published path reaches, just what stops the walk from doing wasted work
    (or, before this existed, stopping short of plan_horizon on dense cones
    purely because it ran out of gate COUNT -- see _WALL_CORRIDOR_MAX_WALK).

    Entering the graph:
      - If the car's position lies inside a mixed (accepted) triangle
        (tri.find_simplex), the walk starts there and heads to whichever of
        its 2 gates is better aligned with prev_dir/heading -- both options
        are provably wall-safe (car_pos and either gate's midpoint are both
        within that same convex triangle).
      - Otherwise (car outside the corridor mesh entirely -- common at the
        very start of a run, before enough cones are behind/around the car)
        the walk falls back to the nearest gate roughly ahead, with a
        ONE-OFF wall-crossing check for just that single bridging segment:
        car_pos is not part of the triangulation, so it isn't covered by
        the "stays inside its own triangle" guarantee the rest of this walk
        relies on. This is the only wall-crossing check left anywhere in
        this function, and it runs at most once per tick, not once per
        candidate per step.

    Returns an (K, 2) array of chained midpoints (car_pos is NOT included --
    same contract as the old _build_wall_path), or an empty array if no
    entry point is reachable.
    """
    if not simplex_gates:
        return np.empty((0, 2), dtype=np.float64)

    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)
    seed_dir = prev_dir if prev_dir is not None else np.array([cos_y, sin_y], dtype=np.float64)

    current_gate = None
    visited: set = set()

    s0 = int(tri.find_simplex(car_pos))
    if s0 in simplex_gates:
        ga, gb = simplex_gates[s0]
        da, db = gate_mid[ga] - car_pos, gate_mid[gb] - car_pos
        na, nb = float(np.linalg.norm(da)), float(np.linalg.norm(db))
        cos_a = float(da @ seed_dir) / na if na > 1e-6 else -2.0
        cos_b = float(db @ seed_dir) / nb if nb > 1e-6 else -2.0
        current_gate = ga if cos_a >= cos_b else gb
        visited = {s0}
    else:
        candidates = []
        for gate, mid in gate_mid.items():
            d = mid - car_pos
            n = float(np.linalg.norm(d))
            if n < 1e-6 or n > max_step_dist:
                continue
            cosang = float(d @ seed_dir) / n
            if cosang <= 0.0:
                continue
            candidates.append((n, gate))
        candidates.sort(key=lambda c: c[0])
        for _, gate in candidates[:8]:
            mid = gate_mid[gate]
            if wall_segs and segment_crosses_walls(car_pos, mid, wall_segs):
                continue
            current_gate = gate
            break

        # Unlike the in-hull branch above, car_pos is not itself part of any
        # simplex here (the car sits outside the mesh entirely), so there is
        # no simplex "behind us" to seed `visited` with -- the bordering
        # simplex/simplices of `current_gate` are ALL still unwalked. Marking
        # gate_simplices[current_gate][0] visited up front (the previous
        # behaviour) silently discarded the only walkable neighbour whenever
        # this gate happened to be a convex-hull edge (exactly the common
        # case: the fallback gate is picked because it's the nearest one
        # roughly ahead of the car, which is often the near/hull edge of the
        # locally-triangulated corridor) -- the main loop's first iteration
        # then found nothing left to visit and returned a 1-point path even
        # though the corridor plainly continued.
        #
        # When two simplices border this gate (an interior, not hull, entry
        # point) we still need to pick one to treat as "behind us" so the
        # walk doesn't need to try both -- discard whichever one's OTHER gate
        # points least along seed_dir, keeping the one that actually heads
        # further into the corridor.
        if current_gate is not None:
            bordering = gate_simplices[current_gate]
            if len(bordering) > 1:
                def _other_gate_cos(s_idx: int) -> float:
                    ga, gb = simplex_gates[s_idx]
                    other = gb if ga == current_gate else ga
                    d = gate_mid[other] - gate_mid[current_gate]
                    n = float(np.linalg.norm(d))
                    return float(d @ seed_dir) / n if n > 1e-6 else -2.0
                behind_sim = min(bordering, key=_other_gate_cos)
                visited = {behind_sim}

    if current_gate is None:
        return np.empty((0, 2), dtype=np.float64)

    path = [gate_mid[current_gate]]
    step = path[0] - car_pos
    n = float(np.linalg.norm(step))
    cur_dir = step / n if n > 1e-6 else seed_dir
    arc_len = n   # cumulative car -> path[-1] distance -- the walk's real stopping metric

    if arc_len >= max_arc_length:
        return np.array(path, dtype=np.float64)

    for _ in range(max_steps - 1):
        nxt = [s for s in gate_simplices.get(current_gate, ()) if s not in visited]
        if not nxt:
            break
        next_sim = nxt[0]
        ga, gb = simplex_gates[next_sim]
        other_gate = gb if ga == current_gate else ga

        step = gate_mid[other_gate] - path[-1]
        d = float(np.linalg.norm(step))
        if d < 1e-6 or d > max_step_dist:
            break
        step_dir = step / d
        if float(step_dir @ cur_dir) <= _WALL_MAX_TURN_COS:
            break

        path.append(gate_mid[other_gate])
        visited.add(next_sim)
        current_gate = other_gate
        cur_dir = step_dir
        arc_len += d

        # Stop once we've walked far enough -- the count-based max_steps
        # backstop above is not meant to be what normally ends this loop
        # (dense cone spacing makes it exhaust well short of max_arc_length
        # by count alone; see _WALL_CORRIDOR_MAX_WALK).
        if arc_len >= max_arc_length:
            break

    return np.array(path, dtype=np.float64)


def _build_corridor_path(
    blue_cones: np.ndarray,
    yellow_cones: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    prev_dir: np.ndarray | None = None,
    mid_min: float = _WALL_MID_MIN_DELAUNAY,
    mid_max: float = _WALL_MID_MAX_DELAUNAY,
    max_arc_length: float = _WALL_PLAN_HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a wall-safe-by-construction centreline chain from a Delaunay
    triangulation of every known cone (both colours together) -- replaces
    _gen_midpoints' exclusive-nearest-neighbour matching AND _build_wall_
    path's cost-weighted greedy walk for the 'delaunay' midpoint_method.

    Why: the old approach picked "the best" cross-colour edge per cone by
    distance (or, briefly, distance/perpendicularity), then chained the
    resulting independent midpoints with a greedy walk that had to pay
    _WALL_CROSS_PENALTY to check every remaining candidate against every
    wall segment at every step -- an O(steps x candidates x wall_segments)
    cost that scales badly as the cone count grows. This function instead
    uses each mixed Delaunay triangle's 2 cross-colour edges directly as a
    corridor "gate pair" (see _classify_simplex for why that's provably
    wall-safe) and walks the resulting graph, which has at most one way to
    continue at each step (see _walk_corridor) -- no wall-crossing check
    inside the walk at all, and no perpendicularity heuristic needed to
    disambiguate competing candidates, because a mixed triangle's 2 gates
    aren't competing with anything: they're just used.

    Triangulates the FULL unbounded blue_cones/yellow_cones (same point set
    build_wall_segments_delaunay uses), not a look_radius-windowed subset --
    windowing risked hiding a real intermediate cone right at the window
    edge (the same failure mode the unbounded wall mesh was built to avoid,
    see build_wall_segments_delaunay). How far the published path reaches
    is instead governed entirely by the walk itself (dead end / max_steps)
    and, same as before, build_path_walls' post-walk plan_horizon clamp.

    Returns (ordered, all_gate_midpoints):
      ordered            : (K, 2) chained corridor path (car_pos excluded --
                            same contract as the old _build_wall_path).
      all_gate_midpoints : (M, 2) every accepted gate midpoint in the
                            triangulation, not just the ones used in
                            `ordered` -- for /fsae/planning/debug/midpoints,
                            so a tick with a short/empty path still shows
                            what candidates existed.

    Both arrays are empty if there are too few cones (<1 of either colour,
    or <3 total) or the triangulation itself fails (degenerate/near-
    collinear point set) -- build_path_walls' existing build_local_path
    fallback takes over in that case, same as when the old midpoint
    generator came back empty.
    """
    blue = np.asarray(blue_cones, dtype=np.float64).reshape(-1, 2)
    yellow = np.asarray(yellow_cones, dtype=np.float64).reshape(-1, 2)

    if len(blue) == 0 or len(yellow) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    points = np.vstack([blue, yellow])
    is_blue = np.concatenate([
        np.ones(len(blue), dtype=bool),
        np.zeros(len(yellow), dtype=bool),
    ])

    if len(points) < 3:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    try:
        tri = Delaunay(points)
    except Exception:
        # Degenerate input (e.g. near-collinear cones on a fresh straight).
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    gate_mid, simplex_gates, gate_simplices, wall_segs = _build_corridor_graph(
        points, is_blue, tri, mid_min, mid_max,
    )
    if not simplex_gates:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    ordered = _walk_corridor(
        car_pos, car_yaw, tri, gate_mid, simplex_gates, gate_simplices, wall_segs,
        prev_dir=prev_dir, max_arc_length=max_arc_length,
    )
    all_mids = np.array(list(gate_mid.values()), dtype=np.float64)
    return ordered, all_mids


_SEED_DIR_MIN_ALIGN = 0.3   # cos of ~72 deg -- see prev_dir note in _build_wall_path


def _build_wall_path(
    midpoints: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    wall_segs: list[tuple[np.ndarray, np.ndarray]],
    prev_dir: np.ndarray | None = None,
) -> np.ndarray:
    """
    Chain midpoints into a path by walking from the car's own position.

    Step cost (identical for every hop, including the first, car → first
    midpoint):
        distance  +  _WALL_CROSS_PENALTY × crossings  +  2.0 × heading_change(rad)

    At each step, picks the cheapest unvisited midpoint whose bearing is
    within _WALL_MAX_TURN_COS of the current travel direction, where cur_dir
    starts at the car's heading and rotates as the walk turns.

    Starting the walk AT the car rather than seeding it with a separately
    (and more loosely) chosen "nearest midpoint ahead" means the car → first
    point segment — the one the controller weights most heavily — goes
    through the same wall-crossing and turn-angle checks as every other step,
    instead of bypassing the barrier entirely. The old seed rule only tested
    whether a midpoint was "not clearly behind" the car (a loose distance-
    along-heading gate with no lateral or wall-crossing check at all), which
    let it flip onto the wrong leg of a self-intersecting corner: the
    already-driven return leg can sit physically closer to the car than the
    correct next midpoint while still passing that gate, and nothing in the
    old seed rule penalised the resulting near-180° turn, so the walk from
    there ran backward relative to true track progress.

    This is also the key to not truncating at corners: an earlier version
    sorted midpoints by forward distance in the *car's fixed heading frame*
    and only chained to ever-greater forward distance, so as soon as the
    track curved away from the initial heading the chain stalled (the
    apex/exit midpoints have smaller heading-frame forward distance) —
    producing a stub path into corners. Following cur_dir instead lets the
    chain turn with the track.

    The per-step gate is a relaxed turn allowance (_WALL_MAX_TURN_COS, ~120°) not
    a hard 90° forward test: at an extreme bend the next along-track midpoint can
    sit well past 90° from the current heading, and a 90° ceiling dropped it and
    truncated the path into the corner.  The angle cost still favours straighter
    chains and the wall-crossing penalty still blocks jumps to adjacent parallel
    tracks, so relaxing the gate wraps hairpins without doubling back.

    prev_dir: unit direction (car → first point) of the path this function
    produced last tick, if any. On the FIRST hop only, candidates whose
    direction from the car disagrees with prev_dir (cosine <=
    _SEED_DIR_MIN_ALIGN) are excluded before the cost comparison above, unless
    that would exclude everything — an extra guard against the wrong-leg
    flip described above surviving the turn-angle/wall-cross checks (e.g. a
    return-leg midpoint within the ~120° per-step allowance). Later hops are
    unaffected.
    """
    n = len(midpoints)
    if n == 0:
        return np.empty((0, 2), dtype=np.float64)

    cos_y, sin_y = math.cos(car_yaw), math.sin(car_yaw)
    heading = np.array([cos_y, sin_y], dtype=np.float64)

    ordered  = []
    visited  = set()
    cur_dir  = heading.copy()
    curr     = car_pos

    for step_i in range(_WALL_PATH_MAX_WALK):
        cand_idx = np.array([idx for idx in range(n) if idx not in visited], dtype=int)

        if step_i == 0 and prev_dir is not None and len(cand_idx) > 0:
            rel      = midpoints[cand_idx] - car_pos
            rel_norm = np.linalg.norm(rel, axis=1)
            rel_norm[rel_norm < 1e-6] = 1e-6
            align    = (rel / rel_norm[:, None]) @ prev_dir
            aligned  = cand_idx[align > _SEED_DIR_MIN_ALIGN]
            if len(aligned) > 0:
                cand_idx = aligned

        best_nb, best_cost = None, math.inf
        for idx in cand_idx:
            cand = midpoints[idx]
            step = cand - curr
            d = float(np.linalg.norm(step))
            if d < 1e-6 or d > _WALL_PATH_MAX_STEP:
                continue
            step_dir = step / d
            fwd_dot  = float(np.dot(cur_dir, step_dir))
            if fwd_dot <= _WALL_MAX_TURN_COS:   # reject only sharp doublings-back
                continue
            angle    = math.acos(max(-1.0, min(1.0, fwd_dot)))
            n_cross  = sum(1 for (w1, w2) in wall_segs if _seg_intersect(curr, cand, w1, w2))
            cost     = d + _WALL_CROSS_PENALTY * n_cross + 2.0 * angle
            if cost < best_cost:
                best_cost = cost
                best_nb   = int(idx)

        if best_nb is None:
            break
        cur_dir = (midpoints[best_nb] - curr)
        cur_dir = cur_dir / (float(np.linalg.norm(cur_dir)) + 1e-9)
        visited.add(best_nb)
        ordered.append(best_nb)
        curr = midpoints[best_nb]

    return midpoints[ordered]


def build_path_walls(
    blue_cones: np.ndarray,
    yellow_cones: np.ndarray,
    car_pos: np.ndarray,
    car_yaw: float,
    max_ahead: float = 25.0,
    max_lateral: float = 10.0,
    smooth_per_pt: float = DEFAULT_SMOOTH_PER_PT,
    look_radius: float = 25.0,   # kept >= _WALL_PLAN_HORIZON — see that constant's comment
    plan_horizon: float = _WALL_PLAN_HORIZON,
    prev_dir: np.ndarray | None = None,
    midpoint_method: str = 'delaunay',
    mid_min_gate: float = _WALL_MID_MIN_DELAUNAY,
    mid_max_gate: float = _WALL_MID_MAX_DELAUNAY,
) -> tuple[np.ndarray | None,
           list[tuple[np.ndarray, np.ndarray]],
           list[tuple[np.ndarray, np.ndarray]],
           np.ndarray]:
    """
    Build a centreline using cone-wall segments as a path barrier.

    Wall segments come from a Delaunay triangulation of every known cone,
    filtered down to same-colour edges (see build_wall_segments_delaunay) --
    NOT a car-relative window. A window bounded the old all-pairs linker's
    blast radius, but it never addressed the actual failure mode (a same-
    coloured cone linking straight across the track), and pairing it with the
    triangulation is redundant now that the triangulation itself is what
    keeps a false link from forming, regardless of how far away the cone
    responsible for it is.

    `midpoint_method` selects how cones are turned into a chained midpoint
    path:

      'delaunay' (default) — each mixed (2-1 vertex-colour split) triangle
        of a Delaunay triangulation of both boundaries contributes its 2
        cross-colour edges directly as a corridor "gate pair" (see
        _build_corridor_path). This is provably wall-safe by construction
        (a triangle's interior can't reach any other triangle's edges), so
        no per-step wall-crossing check or cost function is needed to chain
        them -- also triangulates the full unbounded cone set, not a
        look_radius-windowed subset (see _build_corridor_path for why).
      'nn' — the older exclusive-nearest-neighbour matching (see
        _gen_midpoints) restricted to a look_radius/max_ahead forward window
        (blue_fwd/yellow_fwd below), chained by a greedy walk that costs in
        a _WALL_CROSS_PENALTY per wall-segment crossing. Kept for rollback/
        comparison.

    prev_dir: unit direction (car → first path point) from the last tick's
    result, passed straight through to the walk's seed selection to stop it
    flipping onto the wrong leg at a pinch (see _walk_corridor's / the 'nn'
    path's _build_wall_path's docstring). Caller is responsible for updating
    it from the returned centreline after a successful tick and clearing it
    after a failure/reset.

    mid_min_gate/mid_max_gate: min/max cross-colour edge length accepted as a
    'delaunay' corridor gate (see _build_corridor_graph). Exposed here (rather
    than only as the _WALL_MID_MIN_DELAUNAY/_WALL_MID_MAX_DELAUNAY module
    constants) so an operator can retune them for a track's actual cone
    spacing without a code change -- a fixed [1.2, 5.0] m band can reject a
    geometrically consistent corridor whose triangle diagonals exceed
    mid_max_gate even though its true cross-track width does not (a straight
    3 m-wide corridor with cones spaced 5 m apart has ~5.83 m diagonals).

    Returns
    -------
    centreline  : (N, 2) smoothed path, or None on failure
    blue_segs   : wall segments from blue cones  (for visualisation)
    yellow_segs : wall segments from yellow cones (for visualisation)
    midpoints   : (M, 2) all candidate midpoints  (for visualisation)
    """
    blue_segs, yellow_segs = build_wall_segments_delaunay(blue_cones, yellow_cones)
    all_segs = blue_segs + yellow_segs

    if midpoint_method == 'delaunay':
        ordered, midpoints = _build_corridor_path(
            blue_cones, yellow_cones, car_pos, car_yaw, prev_dir=prev_dir,
            mid_min=mid_min_gate, mid_max=mid_max_gate, max_arc_length=plan_horizon,
        )
    else:
        # Radius (omni) OR forward box.  The radius keeps the cones around a bend
        # (which a heading-aligned box drops as the track curves away), so the path
        # no longer truncates at corners; the box keeps long-range preview straight
        # ahead.  Only the 'nn' method needs this window -- see the docstring above.
        blue_fwd = filter_cones_window(
            blue_cones, car_pos, car_yaw, radius=look_radius,
            min_ahead=0.5, max_ahead=max_ahead, max_lateral=max_lateral,
        )
        yellow_fwd = filter_cones_window(
            yellow_cones, car_pos, car_yaw, radius=look_radius,
            min_ahead=0.5, max_ahead=max_ahead, max_lateral=max_lateral,
        )
        midpoints = _gen_midpoints(blue_fwd, yellow_fwd, car_pos, car_yaw)
        ordered = (
            _build_wall_path(midpoints, car_pos, car_yaw, all_segs, prev_dir=prev_dir)
            if len(midpoints) >= 1 else np.empty((0, 2), dtype=np.float64)
        )

    # A single ordered midpoint is enough: anchored to the car below it becomes a
    # short but ON-TRACK forward segment, which the controller happily extrapolates.
    # Only fall back to build_local_path when the walk is genuinely empty.  The old
    # `< 2` gate was the corner-cut culprit: at a loop's pinch (the descending leg
    # passing close to the returning leg) most midpoints sit BEHIND the car on the
    # return leg, so the forward-seeded walk yields exactly one midpoint — and the
    # `< 2` gate then discarded it for build_local_path, whose naive nearest-cone
    # pairing links straight across the pinch (a ~3 m cross-infield chord that
    # mowed down the apex cones).  Keeping the one-point walk avoids that entirely.
    if len(ordered) < 1:
        cl = build_local_path(blue_cones, yellow_cones, car_pos, car_yaw,
                               max_ahead, max_lateral)
        return cl, blue_segs, yellow_segs, midpoints

    # Clamp the chain to a fixed arc-length horizon (measured from the car through
    # the midpoints) before smoothing.  This makes the near path in front of the
    # car independent of how far the lookahead reaches — extra far midpoints no
    # longer extend the chain or drag the global spline — and keeps the corner
    # line on the true centreline instead of letting distant apex points pull it
    # inward.  At least three points are retained so the spline stays well-posed.
    anchored = np.vstack([car_pos.reshape(1, 2), ordered])
    seg = np.linalg.norm(np.diff(anchored, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    keep = arc <= plan_horizon
    if int(keep.sum()) < 3:
        keep[:min(3, len(anchored))] = True
    anchored = anchored[keep]

    cl = smooth_centreline(anchored, n_out=max(20, (len(anchored) - 1) * 5),
                           smooth_per_pt=smooth_per_pt)
    return cl, blue_segs, yellow_segs, midpoints
