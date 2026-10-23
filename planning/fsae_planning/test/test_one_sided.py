"""
One-sided offset trace tests.

boundary.build_path_walls' one_sided, _side_trace and _one_sided_extension.

Synthetic tracks: a centreline of straight / arc / straight, cones offset
+-width/2 (blue on the left). "Visible" cones are those a forward camera
would see from the car; a case then removes one colour, or thins one side,
to make the view one-sided. Offsets are the go-kart defaults (0.9 / 1.3 /
1.15 m, half width 0.5 m) on a 3.0 m track, mid_max_gate at its 5.0 m
default.
"""
import math

import numpy as np
import pytest

from fsae_planning.boundary import (
    _ONE_SIDED_CAR_HALF_WIDTH,
    _ONE_SIDED_OFFSET_DEFAULT,
    _ONE_SIDED_OFFSET_MAX,
    _side_trace,
    build_path_walls,
)

WIDTH = 3.0


def _centreline(l1, turn_deg, radius, l2, ds=0.05):
    """Straight l1 along +x from the origin, an arc (+ = left), then straight l2."""
    pts = [np.array([0.0, 0.0])]
    yaw = 0.0
    n1 = int(round(l1 / ds))
    for _ in range(n1):
        pts.append(pts[-1] + ds * np.array([math.cos(yaw), math.sin(yaw)]))
    arc = abs(math.radians(turn_deg)) * radius
    sign = 1.0 if turn_deg >= 0 else -1.0
    for _ in range(int(round(arc / ds))):
        yaw += sign * ds / radius
        pts.append(pts[-1] + ds * np.array([math.cos(yaw), math.sin(yaw)]))
    for _ in range(int(round(l2 / ds))):
        pts.append(pts[-1] + ds * np.array([math.cos(yaw), math.sin(yaw)]))
    return np.array(pts)


def _side(cl, offset, spacing):
    """Cones every `spacing` metres of arc along the centreline offset by `offset` (+ = left)."""
    t = np.gradient(cl, axis=0)
    t /= np.linalg.norm(t, axis=1)[:, None]
    line = cl + offset * np.column_stack([-t[:, 1], t[:, 0]])
    seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    s = np.arange(0.0, arc[-1], spacing)
    return np.column_stack([np.interp(s, arc, line[:, 0]), np.interp(s, arc, line[:, 1])])


def _track(l1=12.0, turn=90.0, radius=6.0, l2=15.0, blue_sp=3.5, yellow_sp=3.5):
    cl = _centreline(l1, turn, radius, l2)
    return cl, _side(cl, WIDTH / 2, blue_sp), _side(cl, -WIDTH / 2, yellow_sp)


def _pose(cl, s):
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    i = int(np.searchsorted(arc, s))
    d = cl[min(i + 1, len(cl) - 1)] - cl[max(i - 1, 0)]
    return cl[i].copy(), math.atan2(d[1], d[0])


def _visible(cones, pos, yaw, rng=9.0, hfov=110.0, behind=2.0):
    """Forward wedge plus a short memory of cones just beside/behind (the shim's retention)."""
    if len(cones) == 0:
        return cones
    rel = cones - pos
    x = rel[:, 0] * math.cos(yaw) + rel[:, 1] * math.sin(yaw)
    y = -rel[:, 0] * math.sin(yaw) + rel[:, 1] * math.cos(yaw)
    wedge = (x > 0) & (np.hypot(x, y) <= rng) & (np.abs(np.arctan2(y, x)) <= math.radians(hfov / 2))
    near = (x > -behind) & (x <= 0) & (np.abs(y) < WIDTH)
    return cones[wedge | near]


def _plan(blue, yellow, pos, yaw, one_sided=True):
    info = {}
    cl, bs, ys, _ = build_path_walls(blue, yellow, pos, yaw, one_sided=one_sided, info=info)
    return cl, info, bs + ys


def _arc(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _seg_dist(pts, line):
    """Distance from each point to the polyline `line`, and the arc position of its foot."""
    a, b = line[:-1], line[1:]
    ab = b - a
    ab2 = np.maximum(np.einsum('ij,ij->i', ab, ab), 1e-12)
    arc0 = np.concatenate([[0.0], np.cumsum(np.sqrt(ab2))])
    out_d, out_s = [], []
    for p in pts:
        t = np.clip(np.einsum('ij,ij->i', p - a, ab) / ab2, 0, 1)
        foot = a + t[:, None] * ab
        d = np.linalg.norm(foot - p, axis=1)
        k = int(np.argmin(d))
        out_d.append(d[k])
        out_s.append(arc0[k] + t[k] * math.sqrt(ab2[k]))
    return np.array(out_d), np.array(out_s)


def _assert_on_track(path, cl, blue_all, yellow_all, car_s):
    """
    Assert the path stays on the track and moves forward along it.

    On the track: inside it, and the car's half width clear of both walls.

    The walls are the real cones joined in order (what the cones define --
    a tight bend's inner arc between two cones is a chord), so the body may
    not cross a cone-to-cone line, nor come within half width of a cone.
    """
    d, s = _seg_dist(path, cl)
    assert d.max() <= WIDTH / 2, d.max()
    for wall in (blue_all, yellow_all):
        dw, _ = _seg_dist(path, wall)
        assert dw.min() >= _ONE_SIDED_CAR_HALF_WIDTH - 0.05, dw.min()
    # Never jumps to another part of the track (the other leg of a hairpin).
    assert np.all(np.diff(s) > -0.3), np.diff(s).min()
    assert s[-1] > car_s


# ---------------------------------------------------------------------------
# Corners seen from one side
# ---------------------------------------------------------------------------

CORNERS = [
    ('left90', 90.0, 6.0),
    ('right90', -90.0, 6.0),
    ('left_hairpin', 180.0, 4.5),
    ('right_hairpin', -180.0, 4.5),
]


@pytest.mark.parametrize('name,turn,radius', CORNERS)
@pytest.mark.parametrize('keep', ['outer', 'inner'])
@pytest.mark.parametrize('car_s', [6.0, 10.0, 14.0, 18.0])
def test_corner_one_colour_only(name, turn, radius, keep, car_s):
    cl, blue, yellow = _track(turn=turn, radius=radius)
    pos, yaw = _pose(cl, car_s)
    vb, vy = _visible(blue, pos, yaw), _visible(yellow, pos, yaw)
    # Left turn: blue is the inner side.
    inner_is_blue = turn > 0
    keep_blue = (keep == 'inner') == inner_is_blue
    vb, vy = (vb, vy[:0]) if keep_blue else (vb[:0], vy)
    seen = vb if keep_blue else vy
    fwd = (seen - pos) @ np.array([math.cos(yaw), math.sin(yaw)])
    if np.sum(fwd > 0.0) < 2:
        pytest.skip('fewer than 2 cones of the kept colour ahead (no single-cone mode)')

    off, _, _ = _plan(vb, vy, pos, yaw, one_sided=False)
    assert off is None                               # today: no path at all

    path, info, _ = _plan(vb, vy, pos, yaw)
    assert info['one_sided'] and info['side'] == ('blue' if keep_blue else 'yellow')
    assert path is not None and _arc(path) >= 1.5
    _assert_on_track(path, cl, blue, yellow, car_s)
    # Never past the last cone seen (projected on the centreline).
    seen = vb if keep_blue else vy
    _, s_path = _seg_dist(path[-1:], cl)
    _, s_cones = _seg_dist(seen, cl)
    assert s_path[0] <= s_cones.max() + 0.5           # one resample step (ds)


@pytest.mark.parametrize('name,turn,radius', CORNERS[:2])
def test_corner_sparse_outer_side(name, turn, radius):
    """Outer side at 7 m spacing: the gates' diagonals exceed mid_max_gate and the walk dies."""
    sp = dict(blue_sp=7.0, yellow_sp=2.5) if turn < 0 else dict(blue_sp=2.5, yellow_sp=7.0)
    cl, blue, yellow = _track(turn=turn, radius=radius, **sp)
    for car_s in (8.0, 11.0, 14.0):
        pos, yaw = _pose(cl, car_s)
        vb, vy = _visible(blue, pos, yaw), _visible(yellow, pos, yaw)
        off, _, _ = _plan(vb, vy, pos, yaw, one_sided=False)
        path, info, _ = _plan(vb, vy, pos, yaw)
        assert path is not None
        _assert_on_track(path, cl, blue, yellow, car_s)
        if info['one_sided']:
            assert off is None or _arc(path) > _arc(off)


# ---------------------------------------------------------------------------
# Hairpin with the other leg in view
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('turn', [180.0, -180.0])
@pytest.mark.parametrize('radius', [3.0, 3.5])   # 3.0: FS minimum (9 m outer diameter)
@pytest.mark.parametrize('keep', ['inner', 'outer'])
def test_hairpin_other_leg_not_latched(turn, radius, keep):
    """
    Stay on the entry leg of a hairpin and round the apex.

    Narrow infield (2 * radius - WIDTH between the two inner rows): the exit
    leg's cones of the same colour are in view, and nearer than our own next
    cone.
    """
    cl, blue, yellow = _track(l1=14.0, turn=turn, radius=radius, l2=14.0, blue_sp=4.0, yellow_sp=4.0)
    inner_is_blue = turn > 0
    keep_blue = (keep == 'inner') == inner_is_blue
    for car_s in (4.0, 7.0, 10.0):
        pos, yaw = _pose(cl, car_s)
        # Wide view: everything within 12 m, both legs included.
        own = blue if keep_blue else yellow
        seen = own[np.linalg.norm(own - pos, axis=1) <= 12.0]
        vb, vy = (seen, yellow[:0]) if keep_blue else (blue[:0], seen)
        path, info, walls = _plan(vb, vy, pos, yaw)
        if path is None:
            continue
        assert info['one_sided']
        _assert_on_track(path, cl, blue, yellow, car_s)
        # Every trace cone is on the entry leg or the apex, in order.
        _, s_tr = _seg_dist(info['trace'], cl)
        assert np.all(np.diff(s_tr) > 0.0), s_tr


def test_trace_prefers_own_row_over_close_other_leg():
    """Direct check of the turn cost: other row 2 m across, 1 m ahead; own row 5 m ahead."""
    own = np.array([[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]])
    other = np.array([[1.0, 2.0], [-3.0, 2.0], [-7.0, 2.0]])
    trace = _side_trace(np.vstack([own, other]), np.array([0.0, -1.0]), np.array([1.0, 0.0]))
    assert np.allclose(trace, own)


# ---------------------------------------------------------------------------
# Straights, transitions, noise, degenerate input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('keep_blue', [True, False])
def test_straight_one_side_offset(keep_blue):
    cl, blue, yellow = _track(l1=40.0, turn=0.0, radius=1.0, l2=0.0)
    pos, yaw = _pose(cl, 5.0)
    vb, vy = _visible(blue, pos, yaw), _visible(yellow, pos, yaw)
    vb, vy = (vb, vy[:0]) if keep_blue else (vb[:0], vy)
    path, info, _ = _plan(vb, vy, pos, yaw)
    assert info['one_sided']
    # After the taper, the path runs at offset_default from the seen boundary.
    far = path[path[:, 0] > pos[0] + 4.0]
    assert len(far)
    seen_y = WIDTH / 2 if keep_blue else -WIDTH / 2
    lat = np.abs(far[:, 1] - seen_y)
    assert np.all(np.abs(lat - _ONE_SIDED_OFFSET_DEFAULT) < 0.05), lat
    # Starts at the car (centred, 1.5 m from the seen side -> clamped to 1.3).
    assert abs(path[0, 1] - pos[1]) < 0.05
    near = path[(path[:, 0] > pos[0] + 0.4) & (path[:, 0] < pos[0] + 1.0)]
    assert np.all(np.abs(np.abs(near[:, 1] - seen_y) - _ONE_SIDED_OFFSET_MAX) < 0.25)


def test_alternating_sides_no_lateral_jump():
    """Drive a straight, losing one colour then the other: the near path stays put."""
    cl, blue, yellow = _track(l1=60.0, turn=0.0, radius=1.0, l2=0.0)
    prev = None
    for k, s in enumerate(np.arange(4.0, 40.0, 0.5)):
        pos, yaw = _pose(cl, s)
        vb, vy = _visible(blue, pos, yaw), _visible(yellow, pos, yaw)
        phase = (k // 8) % 3          # both, blue only, yellow only
        if phase == 1:
            vy = vy[:0]
        elif phase == 2:
            vb = vb[:0]
        path, info, _ = _plan(vb, vy, pos, yaw)
        assert path is not None, (s, phase)
        assert info['one_sided'] == (phase != 0)
        y1 = float(np.interp(pos[0] + 1.0, path[:, 0], path[:, 1]))
        if prev is not None:
            # Changing sides moves the path by at most twice the gap between
            # the track centre and offset_max (on a 3.0 m track the car, centred,
            # is 1.5 m from either side but is offset at most 1.3 m).
            assert abs(y1 - prev) < 2 * (WIDTH / 2 - _ONE_SIDED_OFFSET_MAX) + 0.1, (s, phase, y1, prev)
        prev = y1


def test_noisy_duplicate_cones():
    rng = np.random.default_rng(3)
    cl, blue, yellow = _track(turn=90.0, radius=6.0)
    for car_s in (8.0, 12.0, 16.0):
        pos, yaw = _pose(cl, car_s)
        vy = _visible(yellow, pos, yaw)
        noisy = vy + rng.normal(0.0, 0.1, vy.shape)
        dup = noisy + rng.normal(0.0, 0.05, noisy.shape)
        seen = np.vstack([noisy, dup])
        path, info, _ = _plan(blue[:0], seen, pos, yaw)
        assert path is not None and info['one_sided']
        d, _ = _seg_dist(path, cl)
        assert d.max() < WIDTH / 2 - 0.3


def test_one_cone_gives_nothing_two_cones_give_a_path():
    pos, yaw = np.array([0.0, 0.0]), 0.0
    one = np.array([[4.0, 1.5]])
    path, info, _ = _plan(one, np.empty((0, 2)), pos, yaw)
    assert path is None and not info['one_sided']        # no single-cone mode
    two = np.array([[2.0, 1.5], [6.0, 1.5]])
    path, info, _ = _plan(two, np.empty((0, 2)), pos, yaw)
    assert info['one_sided'] and info['side'] == 'blue'
    assert path[-1, 0] <= 6.0 + 1e-6                     # ends at the last cone
    assert np.all(path[1:, 1] < 1.5 - 0.9 + 1e-6)        # right of blue by >= offset_min


def test_car_outside_the_seen_wall_gets_nothing():
    """Car to the LEFT of a blue row: that row cannot be its left wall."""
    blue = np.array([[2.0, -1.5], [6.0, -1.5], [10.0, -1.5]])
    path, info, _ = _plan(blue, np.empty((0, 2)), np.array([0.0, 0.0]), 0.0)
    assert not info['one_sided']


def test_fold_stops_before_tight_inner_curve():
    """Offset toward the centre of a curve tighter than 1.2 x offset: the path stops first."""
    ang = np.radians(np.arange(-90.0, 91.0, 25.0))
    r = 1.2
    # Yellow (right) wall curving left around a 1.2 m radius: its track side
    # (left) is the inside of the curve.
    yellow = np.column_stack([8.0 + r * np.cos(ang - np.pi / 2) * 0 + r * np.sin(ang + np.pi / 2) * 0
                              + r * np.cos(ang), r * np.sin(ang)])
    yellow = np.vstack([[[2.0, -r], [5.0, -r]], yellow + [0.0, 0.0]])
    path, info, _ = _plan(np.empty((0, 2)), yellow, np.array([0.0, 0.0]), 0.0)
    if info['one_sided']:
        assert path[:, 0].max() < 8.0          # never reaches round the 1.2 m bend


# ---------------------------------------------------------------------------
# Two-sided regression: identical output when both sides are seen
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('turn,radius', [(0.0, 1.0), (90.0, 6.0), (-90.0, 6.0),
                                         (180.0, 4.5), (-180.0, 4.5), (45.0, 12.0)])
def test_two_sided_identical(turn, radius):
    cl, blue, yellow = _track(l1=12.0, turn=turn, radius=radius, l2=15.0)
    total = _arc(cl)
    for car_s in np.arange(1.0, total - 10.0, 1.3):
        pos, yaw = _pose(cl, car_s)
        for vis in ((blue, yellow),
                    (_visible(blue, pos, yaw, rng=20.0), _visible(yellow, pos, yaw, rng=20.0))):
            off, _, w_off = _plan(*vis, pos, yaw, one_sided=False)
            on, info, w_on = _plan(*vis, pos, yaw, one_sided=True)
            assert not info['one_sided'], car_s
            assert (off is None and on is None) or np.array_equal(off, on)
            assert len(w_off) == len(w_on)
            assert (off is None and info['two_sided'] is None) or np.array_equal(info['two_sided'], off)


# ---------------------------------------------------------------------------
# planner_tester layouts (screenshots, 2026-10-24), cones read off the images:
# x forward, y left, car at the origin heading +x. Kart planner settings.
# ---------------------------------------------------------------------------

_KART = dict(smooth_per_pt=0.05, look_radius=12.0, plan_horizon=10.0,
             mid_min_gate=1.2, mid_max_gate=5.0)
# Left hairpin; the inner (blue) row has an 8.9 m gap on the entry straight.
_SHOT1_YELLOW = [(14.56, 4.59), (13.97, 7.77), (13.54, 1.51), (11.93, 10.23), (10.98, -0.79),
                 (8.98, 11.48), (6.16, -1.51), (4.07, 11.54), (2.03, -1.51), (0.03, 11.54)]
_SHOT1_BLUE = [(11.25, 6.49), (11.57, 4.79), (10.16, 7.84), (10.75, 3.08), (8.56, 8.49),
               (4.03, 8.52), (2.03, 1.51), (0.03, 8.52)]
# Car inside a hairpin whose inner apex cones are missing: only the yellow
# arc ahead, blue only beside/behind the car.
_SHOT2_YELLOW = [(8.16, 4.62), (7.18, 7.67), (7.52, 1.47), (4.81, 9.89), (5.49, -1.05),
                 (2.29, -2.26), (0.23, 11.69), (-2.44, -1.20), (-3.50, 13.12), (-6.28, 0.26)]
_SHOT2_BLUE = [(-0.83, 8.87), (-2.07, 2.07), (-4.55, 10.30), (-5.19, 3.08)]


def _yellow_offsets(path, yellow_line):
    d, _ = _seg_dist(path, np.array(yellow_line))
    return d


@pytest.mark.parametrize('blue,yellow,yellow_line', [
    (_SHOT1_BLUE, _SHOT1_YELLOW,
     [(0.03, 11.54), (4.07, 11.54), (8.98, 11.48), (11.93, 10.23), (13.97, 7.77), (14.56, 4.59),
      (13.54, 1.51), (10.98, -0.79), (6.16, -1.51), (2.03, -1.51)][::-1]),
    (_SHOT2_BLUE, _SHOT2_YELLOW,
     [(-6.28, 0.26), (-2.44, -1.20), (2.29, -2.26), (5.49, -1.05), (7.52, 1.47), (8.16, 4.62),
      (7.18, 7.67), (4.81, 9.89), (0.23, 11.69), (-3.50, 13.12)]),
])
def test_planner_tester_layouts_follow_yellow(blue, yellow, yellow_line):
    blue, yellow = np.array(blue), np.array(yellow)
    car = np.zeros(2)
    info = {}
    path, *_ = build_path_walls(blue, yellow, car, 0.0, one_sided=True, info=info, **_KART)
    assert info['one_sided'] and info['side'] == 'yellow'
    assert _arc(path) > 8.0
    d0 = path[min(5, len(path) - 1)] - path[0]
    assert d0[0] > 0.0                              # leaves forward, not sideways/back
    far = path[np.linalg.norm(path - car, axis=1) > 3.0]
    off = _yellow_offsets(far, yellow_line)
    assert off.min() > 0.85 and off.max() < 1.35, (off.min(), off.max())
    # Never closer than half a kart width to any cone.
    cones = np.vstack([blue, yellow])
    assert min(float(np.min(np.linalg.norm(cones - p, axis=1))) for p in path) > 0.5


def test_planner_tester_shot2_today_has_no_path():
    """Without one_sided this layout gives nothing at all (what planner_tester showed)."""
    path, *_ = build_path_walls(np.array(_SHOT2_BLUE), np.array(_SHOT2_YELLOW), np.zeros(2), 0.0, **_KART)
    assert path is None


def test_planner_tester_shot1_flags_the_fallback():
    """
    Flag shot 1's two-sided path as build_local_path's fallback.

    The walk finds no gates, so the two-sided path is build_local_path's
    pairing straight across the hairpin's infield. info says so, and the node
    then takes the one-sided path at once instead of keeping that one.
    """
    info = {}
    build_path_walls(np.array(_SHOT1_BLUE), np.array(_SHOT1_YELLOW), np.zeros(2), 0.0,
                     one_sided=True, info=info, **_KART)
    assert info['one_sided'] and info['two_sided_fallback']
    assert info['two_sided'] is not None
