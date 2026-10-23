"""Orange start-line classification, entry-skip guard, held-path clearance."""
import itertools
import math

import numpy as np
import pytest

from fsae_planning.boundary import (
    _build_corridor_path,
    _seg_intersect,
    _START_LINE_MAX_OFFSET,
    assign_orange_sides,
    build_path_walls,
    first_close_segment,
    path_ahead_too_close,
)

HALF_W = 1.75
GATE = np.array([[11.35, HALF_W], [11.35, -HALF_W], [12.65, HALF_W], [12.65, -HALF_W]])


def _straight(spacing=3.0, half_w=HALF_W, x_max=30.0, skip=(12.0,)):
    xs = [x for x in np.arange(0.0, x_max + 1e-9, spacing) if all(abs(x - s) > 1e-9 for s in skip)]
    return np.array([[x, half_w] for x in xs]), np.array([[x, -half_w] for x in xs])


def _sides_ok(blue_aug, yellow_aug, n_blue, n_yellow):
    """Every orange appended to blue sits at +y, every one appended to yellow at -y."""
    return bool(np.all(blue_aug[n_blue:, 1] > 0.0) and np.all(yellow_aug[n_yellow:, 1] < 0.0))


def _walls_cross_centreline(segs, x_lo=-5.0, x_hi=40.0):
    c1, c2 = np.array([x_lo, 0.0]), np.array([x_hi, 0.0])
    return any(_seg_intersect(c1, c2, p, q) for p, q in segs)


# -- P1: same-side pairs ------------------------------------------------------

def test_p1_one_blue_one_yellow_abstains():
    # One blue and one yellow cone give no row direction that doesn't pass
    # through an orange cone itself, so the gate is left unclassified (safe:
    # that frame plans as if orange were off) rather than risk a wrong side.
    blue = np.array([[9.0, 1.75]])
    yellow = np.array([[13.5, -1.75]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, GATE)
    assert pairs == [] and len(blue_aug) == 1 and len(yellow_aug) == 1


def test_p1_sparse_with_a_row_direction_pairs_across_not_along():
    # One blue + two yellow (the kart's usual view): the yellow row's
    # direction is known, so the gate is classified -- across, never along.
    blue = np.array([[9.0, 1.75]])
    yellow = np.array([[10.5, -1.75], [13.5, -1.75]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, GATE)
    assert len(pairs) >= 2
    for b_end, y_end in pairs:
        assert b_end[1] > 0.0 and y_end[1] < 0.0     # across the track, never along it
    assert len(blue_aug) - 1 + len(yellow_aug) - 2 == 4
    assert _sides_ok(blue_aug, yellow_aug, 1, 2)


def test_same_row_oranges_with_one_cone_per_colour_are_not_walled_across():
    # Codex validation repro: both oranges are on the yellow row (y=-1.75).
    # With only chords as evidence, (5,-1.75) used to be classified blue and
    # walled to (7.5,1.75) across the track.
    blue = np.array([[7.5, 1.75]])
    yellow = np.array([[1.0, -1.75]])
    orange = np.array([[5.0, -1.75], [1.4, -1.75]])
    blue_aug, yellow_aug, _ = assign_orange_sides(blue, yellow, orange)
    assert _sides_ok(blue_aug, yellow_aug, 1, 1)
    cl, b_segs, y_segs, _ = build_path_walls(
        blue, yellow, np.array([1.0, 0.0]), 0.0, mid_max_gate=5.0, orange_cones=orange)
    assert not _walls_cross_centreline(b_segs + y_segs)


@pytest.mark.parametrize('k', [1, 2, 3, 4])
@pytest.mark.parametrize('perception', ['full', 'sparse', 'one_each'])
def test_every_subset_of_the_four_cone_gate(k, perception):
    if perception == 'full':
        blue, yellow = _straight()
    elif perception == 'sparse':      # 1 blue + 2 yellow: the kart's usual view
        blue, yellow = np.array([[9.0, 1.75]]), np.array([[10.5, -1.75], [13.5, -1.75]])
    else:                             # 1 + 1: no row direction -> abstain, never wrong
        blue, yellow = np.array([[9.0, 1.75]]), np.array([[13.5, -1.75]])
        for idx in itertools.combinations(range(4), k):
            blue_aug, yellow_aug, _ = assign_orange_sides(blue, yellow, GATE[list(idx)])
            assert _sides_ok(blue_aug, yellow_aug, 1, 1), idx
        return
    for idx in itertools.combinations(range(4), k):
        sub = GATE[list(idx)]
        blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, sub)
        assert _sides_ok(blue_aug, yellow_aug, len(blue), len(yellow)), idx
        across = any(GATE[i][0] == GATE[j][0] for i, j in itertools.combinations(idx, 2))
        diagonal = any(GATE[i][0] != GATE[j][0] and GATE[i][1] != GATE[j][1]
                       for i, j in itertools.combinations(idx, 2))
        # A lone diagonal seen with one blue and one yellow cone far along the
        # track can't be told apart -- left unpaired (safe), never swapped.
        if across or (diagonal and perception == 'full'):
            assert len(pairs) >= 1, idx
        if k == 3:
            # All three classified: through a valid pair, or along the row
            # from a classified neighbour.
            assert len(blue_aug) - len(blue) + len(yellow_aug) - len(yellow) == 3, idx
        if k == 4:
            assert len(blue_aug) - len(blue) == 2 and len(yellow_aug) - len(yellow) == 2, idx
        if k == 1:
            assert pairs == [], idx

        if perception == 'full':
            cl, b_segs, y_segs, _ = build_path_walls(
                blue, yellow, np.array([1.0, 0.2]), 0.0, mid_max_gate=5.0, orange_cones=sub)
            assert not _walls_cross_centreline(b_segs + y_segs), idx


# -- run-4 kart start: 2-cone gate, sparse short-range perception --------------

def _kart_track(width=3.2, spacing=3.4, x_max=40.0):
    xs = np.arange(-6.8, x_max, spacing)
    blue = np.c_[xs, np.full_like(xs, width / 2)]
    yellow = np.c_[xs + spacing / 2, np.full_like(xs, -width / 2)]   # staggered rows
    return blue, yellow


def test_run4_two_cone_gate_from_spawn():
    blue, yellow = _kart_track()
    gate = np.array([[2.0, 1.2], [2.0, -1.2]])          # 2.4 m gate, 2 m ahead of spawn
    car = np.array([0.0, 0.0])
    near = lambda c: c[np.linalg.norm(c - car, axis=1) < 5.0]   # ~5 m kart range
    b, y = near(blue), near(yellow)
    assert len(b) <= 3 and len(y) <= 3
    blue_aug, yellow_aug, pairs = assign_orange_sides(b, y, gate)
    assert len(pairs) == 1
    assert pairs[0][0][1] > 0.0 and pairs[0][1][1] < 0.0

    cl, b_segs, y_segs, _ = build_path_walls(b, y, car, 0.0, mid_max_gate=5.0, orange_cones=gate)
    assert cl is not None and cl[:, 0].max() > 2.0
    at_gate = cl[np.argmin(np.abs(cl[:, 0] - 2.0))]
    assert abs(at_gate[1]) < 1.2                       # passes between the two orange cones
    assert not _walls_cross_centreline(b_segs + y_segs, -1.0, 5.0)


# -- strays ---------------------------------------------------------------------

@pytest.mark.parametrize('stray', [
    (12.8, 0.5),     # inside the track, beside the gate
    (14.5, -0.3),    # inside the track, past the gate
    (12.0, 3.5),     # outside, straight behind the blue gate cone
    (16.0, -3.0),    # outside, away from the gate
])
def test_strays_never_make_a_false_wall(stray):
    blue, yellow = _straight()
    orange = np.vstack([GATE, [stray]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, orange)
    for g in GATE:                                   # every gate cone still classified
        assert tuple(g) in {tuple(p) for p in np.vstack([blue_aug, yellow_aug])}
    assert _sides_ok(blue_aug, yellow_aug, len(blue), len(yellow))
    # No one-partner rule: a stray that passes the side test is classified and
    # walled like any blue/yellow cone. Only one on (within
    # _START_LINE_MAX_OFFSET of) its boundary row can be -- (12.8, 0.5) is
    # 1.25 m inside the blue row and is; one farther inside never is.
    in_mesh = tuple(stray) in {tuple(p) for p in np.vstack([blue_aug, yellow_aug])}
    if HALF_W - abs(stray[1]) > _START_LINE_MAX_OFFSET:
        assert not in_mesh
    cl, b_segs, y_segs, _ = build_path_walls(
        blue, yellow, np.array([1.0, 0.2]), 0.0, mid_max_gate=5.0, orange_cones=orange)
    assert not _walls_cross_centreline(b_segs + y_segs)
    assert cl[:, 0].max() > 20.0
    if in_mesh:                                      # walled off, so the path clears it
        assert np.min(np.linalg.norm(cl - np.array(stray), axis=1)) > 0.4


def test_lone_stray_inside_track_is_ignored():
    blue, yellow = _straight(skip=())
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, np.array([[12.0, 0.2]]))
    assert pairs == [] and len(blue_aug) == len(blue) and len(yellow_aug) == len(yellow)


# -- P2: entry skip must not cut through a wall -----------------------------------

def test_p2_entry_point_kept_when_skipping_would_cross_a_wall():
    blue = np.array([[-2.0, 2.0], [0.0, 0.0], [3.0, 1.0]])
    yellow = np.array([[0.0, -2.0]])
    ordered, _ = _build_corridor_path(blue, yellow, np.array([-1.0, 0.4]), math.pi / 4,
                                      mid_max=6.5)
    assert len(ordered) >= 1
    assert np.allclose(ordered[0], [0.0, -1.0])


# -- held path vs a newly seen cone --------------------------------------------------

def test_held_path_rejected_for_a_cone_ahead_only():
    path = np.c_[np.linspace(0.0, 20.0, 21), np.zeros(21)]
    car = np.array([5.0, 0.1])
    near = lambda x, y, c=0.4: path_ahead_too_close(path, car, np.array([[x, y]]), c)
    assert near(10.5, 0.2)            # on the path, ahead
    assert near(5.6, -0.3)            # just ahead of the car
    assert not near(2.0, 0.1)         # behind the car
    assert not near(10.5, 1.0)        # clear of the path
    assert not near(10.5, 0.1, 0.0)   # check disabled
    assert not path_ahead_too_close(path, car, np.empty((0, 2)), 0.4)


# -- genuine orange cones right next to border cones --------------------------------

@pytest.mark.parametrize('gap', [0.27, 0.30, 0.49, 0.51, 0.62, 0.76])
def test_gate_cones_close_to_border_cones_are_used(gap):
    # 4-cone gate on the border rows (FS rules), the outer pair `gap` metres
    # past the x=12 border cones. 0.27 m is about where two cone bases touch.
    blue, yellow = _straight(skip=())
    gate = np.array([[12.0 + gap, HALF_W], [12.0 + gap, -HALF_W],
                     [13.3 + gap, HALF_W], [13.3 + gap, -HALF_W]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, gate)
    assert len(pairs) >= 2
    assert len(blue_aug) - len(blue) == 2 and len(yellow_aug) - len(yellow) == 2
    assert _sides_ok(blue_aug, yellow_aug, len(blue), len(yellow))
    cl, b_segs, y_segs, _ = build_path_walls(
        blue, yellow, np.array([1.0, 0.2]), 0.0, mid_max_gate=5.0, orange_cones=gate)
    assert not _walls_cross_centreline(b_segs + y_segs)
    assert cl[:, 0].max() > 20.0


# -- width invariance: a narrower track gives the same result as a wider one --------

def _width_result(width, mid_max):
    half = width / 2.0
    blue, yellow = _straight(half_w=half, skip=(12.0,))
    gate = np.array([[11.35, half], [11.35, -half], [12.65, half], [12.65, -half]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, gate)
    cl, b_segs, y_segs, _ = build_path_walls(
        blue, yellow, np.array([1.0, 0.2]), 0.0, mid_max_gate=mid_max, orange_cones=gate)
    # Valid-pair count is not compared: the gate diagonals drop out past
    # start_line_max_width on the widest tracks, the classification doesn't.
    return dict(
        n_blue=len(blue_aug) - len(blue), n_yellow=len(yellow_aug) - len(yellow),
        sides_ok=_sides_ok(blue_aug, yellow_aug, len(blue), len(yellow)),
        false_wall=_walls_cross_centreline(b_segs + y_segs),
        reaches_horizon=bool(cl[:, 0].max() > 24.0),
        centred=bool(np.all(np.abs(cl[:, 1] - 0.2 * (cl[:, 0] < 3.0)) < 0.5)),
    )


@pytest.mark.parametrize('mid_max, widths', [
    (5.0, [3.0, 3.5, 3.9]),          # 3 m spacing: diagonals stay under 5.0 m up to ~4 m wide
    (6.5, [3.0, 3.5, 4.0, 4.4, 5.0]),        # up to the 5.0 m start-line width bound
])
def test_same_result_across_track_widths(mid_max, widths):
    results = [_width_result(w, mid_max) for w in widths]
    for w, r in zip(widths, results):
        assert r == results[0], (w, r, results[0])
    assert results[0] == dict(n_blue=2, n_yellow=2, sides_ok=True,
                              false_wall=False, reaches_horizon=True, centred=True)


def test_gate_wider_than_the_width_bound_is_not_paired():
    # The 5.0 m start-line width bound is deliberate: a wider orange "pair" is
    # left out of the mesh rather than risk pairing mislabelled objects.
    r = _width_result(5.5, 6.5)
    assert r['n_blue'] == 0 and r['n_yellow'] == 0


# -- voting: no one-partner rule ------------------------------------------------------

def _fake_sides(monkeypatch, blue_end_of):
    """
    Replace _gate_sides with a lookup.

    blue_end_of[(i, j)] is the index (i or j) of the blue end for orange cones
    i < j; pairs not listed are invalid.
    """
    import fsae_planning.boundary as b
    pts = {}

    def fake(o1, o2, blue, yellow, radius):
        i, j = pts[tuple(o1)], pts[tuple(o2)]
        key = (min(i, j), max(i, j))
        if key not in blue_end_of:
            return None
        return (o1, o2) if blue_end_of[key] == i else (o2, o1)

    monkeypatch.setattr(b, '_gate_sides', fake)
    return pts


def _classify(monkeypatch, orange, blue_end_of):
    pts = _fake_sides(monkeypatch, blue_end_of)
    pts.update({tuple(p): k for k, p in enumerate(orange)})
    blue, yellow = _straight(skip=())
    ba, ya, pairs = assign_orange_sides(blue, yellow, orange)
    in_b = {tuple(p) for p in ba[len(blue):]}
    in_y = {tuple(p) for p in ya[len(yellow):]}
    side = ['b' if tuple(p) in in_b else 'y' if tuple(p) in in_y else None for p in orange]
    return side, pairs


# Four cones spread along x, all 2.5 m apart pairwise or more, inside min_w..max_w.
_V = np.array([[10.0, 1.75], [10.0, -1.75], [12.5, 1.75], [12.5, -1.75]])


def test_majority_vote_overrides_a_single_conflicting_pair(monkeypatch):
    # Cone 0 is the blue end of two pairs and the yellow end of one: blue.
    side, pairs = _classify(monkeypatch, _V, {(0, 1): 0, (0, 3): 0, (0, 2): 2})
    assert side[0] == 'b'
    # Cone 2 wins its only vote (blue) even though cone 0 outvoted that pair.
    assert side[2] == 'b'
    # The (2 blue, 0 yellow) pair contradicts cone 0's final side: not a start pair.
    keys = {(tuple(a), tuple(b)) for a, b in pairs}
    assert (tuple(_V[2]), tuple(_V[0])) not in keys
    assert (tuple(_V[0]), tuple(_V[1])) in keys and (tuple(_V[0]), tuple(_V[3])) in keys


def test_tied_votes_leave_a_cone_unclassified(monkeypatch):
    # Cone 0: blue in (0, 1), yellow in (0, 2) -> tie -> out of the mesh, even
    # though classified cones sit right next to it on its row.
    side, pairs = _classify(monkeypatch, _V, {(0, 1): 0, (0, 2): 2})
    assert side[0] is None
    assert side[1] == 'y' and side[2] == 'b'
    assert pairs == []        # both pairs contradict cone 0's (missing) side


# -- fresh-path clearance is measured along segments, not just at vertices -------

def test_clearance_catches_a_cone_between_waypoints():
    # Codex validation repro: the planned path passes 0.395 m from an
    # unclassified orange while its nearest WAYPOINT is 0.413 m away, so the
    # old vertex-only check (clearance 0.4) let it through.
    blue = np.array([[4.6, 1.75]])
    yellow = np.array([[4.6, -1.75]])
    orange = np.array([[2.5421052631578944, 0.395]])
    cl, *_ = build_path_walls(blue, yellow, np.array([0.0, 0.0]), 0.0,
                              mid_max_gate=5.0, orange_cones=orange)
    cones = np.vstack([blue, yellow, orange])
    assert np.linalg.norm(cl - orange[0], axis=1).min() >= 0.4     # vertices look clear
    assert first_close_segment(cl, cones, 0.4) >= 0                # a segment is not


def test_first_close_segment_index():
    path = np.c_[np.arange(0.0, 6.0), np.zeros(6)]              # segments 0..4 along y=0
    assert first_close_segment(path, np.array([[2.5, 0.3]]), 0.4) == 2   # mid-segment cone
    # A cone at a vertex: the segment ending there is the first too close.
    assert first_close_segment(path, np.array([[3.0, 0.3]]), 0.4) == 2
    assert first_close_segment(path, np.array([[2.5, 0.5]]), 0.4) == -1  # clear
    assert first_close_segment(path, np.array([[0.1, 0.0]]), 0.4) == 0   # right at the start
    assert first_close_segment(path, np.array([[2.5, 0.3]]), 0.0) == -1  # check disabled
    assert first_close_segment(path, np.empty((0, 2)), 0.4) == -1
