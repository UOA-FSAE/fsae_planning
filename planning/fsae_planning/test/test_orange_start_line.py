"""Start-line orange cones in the wall/corridor planner (boundary.assign_orange_sides)."""
import math

import numpy as np

from fsae_planning.boundary import (
    assign_orange_sides,
    build_path_walls,
    debug_triangulation_edges,
)

HALF_W = 1.75     # 3.5 m wide track
SPACING = 3.0     # same-colour cone spacing


def _straight(gate_xs=(12.0,), x_max=30.0):
    """Straight track along +x, blue at +y. Cones at gate_xs are replaced by orange."""
    xs = [x for x in np.arange(0.0, x_max + 1e-9, SPACING)
          if not any(abs(x - g) < 1e-9 for g in gate_xs)]
    blue = np.array([[x, HALF_W] for x in xs])
    yellow = np.array([[x, -HALF_W] for x in xs])
    orange = np.array([[g, s * HALF_W] for g in gate_xs for s in (1.0, -1.0)])
    return blue, yellow, orange


def _arc_len(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def _plan(blue, yellow, orange, car, yaw):
    cl, *_ = build_path_walls(blue, yellow, np.asarray(car, dtype=float), yaw,
                              mid_max_gate=5.0, orange_cones=orange)
    return cl


def _sets(arr):
    return {tuple(np.round(p, 6)) for p in arr}


def test_straight_start_line_no_longer_dead_ends():
    blue, yellow, orange = _straight()
    car = (1.0, 0.2)
    dropped = _plan(blue, yellow, None, car, 0.0)
    assert dropped[:, 0].max() < 10.0          # today: the walk stops at x=9

    with_orange = _plan(blue, yellow, orange, car, 0.0)
    assert with_orange[:, 0].max() > 24.0      # runs on to the plan_horizon
    assert np.all(np.abs(with_orange[:, 1]) < 0.5)

    _, _, pairs = assign_orange_sides(blue, yellow, orange)
    assert len(pairs) == 1
    b_end, y_end = pairs[0]
    assert b_end[1] > 0.0 and y_end[1] < 0.0


def test_fsds_style_four_cone_gate():
    # FSDS: 2 orange per side, 1.30 m apart along the track (inside the
    # default min_w..max_w band) -- the same-side pairs must not be taken.
    blue, yellow, _ = _straight()
    orange = np.array([[11.35, HALF_W], [11.35, -HALF_W], [12.65, -HALF_W], [12.65, HALF_W]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, orange)
    assert len(pairs) == 2
    for b_end, y_end in pairs:
        assert b_end[1] > 0.0 and y_end[1] < 0.0
        assert abs(b_end[0] - y_end[0]) < 1e-9      # straight across, not diagonal
    assert len(blue_aug) == len(blue) + 2 and len(yellow_aug) == len(yellow) + 2

    cl = _plan(blue, yellow, orange, (1.0, 0.2), 0.0)
    assert cl[:, 0].max() > 24.0

    tags = [t for *_, t in debug_triangulation_edges(blue, yellow, orange_cones=orange)]
    assert tags.count('start') == 2


def _bend(radius=40.0, s_max=30.0, gate_s=12.0):
    """Left-hand arc centred at (0, radius), blue inside. Cones at gate_s become orange."""
    def at(s, off):
        th = s / radius
        r = radius - off
        return [r * math.sin(th), radius - r * math.cos(th)]
    ss = [s for s in np.arange(0.0, s_max + 1e-9, SPACING) if abs(s - gate_s) > 1e-9]
    blue = np.array([at(s, HALF_W) for s in ss])
    yellow = np.array([at(s, -HALF_W) for s in ss])
    orange = np.array([at(gate_s, HALF_W), at(gate_s, -HALF_W)])
    return blue, yellow, orange


def test_start_line_in_gentle_bend():
    blue, yellow, orange = _bend()
    _, _, pairs = assign_orange_sides(blue, yellow, orange)
    assert len(pairs) == 1
    b_end, _ = pairs[0]
    assert np.allclose(b_end, orange[0])        # inner (left) cone is blue

    car = (1.0, 0.2)
    assert _arc_len(_plan(blue, yellow, None, car, 0.0)) < 10.0
    assert _arc_len(_plan(blue, yellow, orange, car, 0.0)) > 20.0


def test_start_line_seen_from_opposite_direction():
    blue, yellow, orange = _straight()
    fwd = assign_orange_sides(blue, yellow, orange)
    rev = assign_orange_sides(blue[::-1], yellow[::-1], orange[::-1])
    assert _sets(fwd[0]) == _sets(rev[0])
    assert _sets(fwd[1]) == _sets(rev[1])

    # Driving the other way (heading -x) also gets through the gate.
    cl = _plan(blue, yellow, orange, (29.0, -0.2), math.pi)
    assert cl[:, 0].min() < 6.0


def test_lone_orange_is_ignored():
    blue, yellow, _ = _straight(gate_xs=())
    lone = np.array([[12.0, HALF_W + 1.0]])
    blue_aug, yellow_aug, pairs = assign_orange_sides(blue, yellow, lone)
    assert pairs == []
    assert np.array_equal(blue_aug, blue) and np.array_equal(yellow_aug, yellow)
    assert _plan(blue, yellow, lone, (1.0, 0.2), 0.0) is not None


def test_stray_orange_next_to_boundary_cone_is_dropped():
    blue, yellow, orange = _straight()
    stray = np.array([[15.3, HALF_W]])          # 0.3 m from the blue cone at x=15
    blue_aug, _, pairs = assign_orange_sides(blue, yellow, np.vstack([orange, stray]))
    assert len(pairs) == 1
    assert tuple(stray[0]) not in _sets(blue_aug)

    # On its own with a partner across the track, a stray still forms no gate.
    lonely = np.array([[15.3, HALF_W], [15.3, -HALF_W - 1.0]])
    blue2, yellow2, _ = _straight(gate_xs=())
    assert assign_orange_sides(blue2, yellow2, lonely)[2] == []


def test_none_matches_todays_output():
    blue, yellow, _ = _straight()
    car, yaw = np.array([1.0, 0.2]), 0.0
    base = build_path_walls(blue, yellow, car, yaw)
    for orange in (None, np.empty((0, 2))):
        out = build_path_walls(blue, yellow, car, yaw, orange_cones=orange)
        assert np.array_equal(base[0], out[0])
        assert np.array_equal(base[3], out[3])
        assert len(base[1]) == len(out[1]) and len(base[2]) == len(out[2])
    tags_base = [t for *_, t in debug_triangulation_edges(blue, yellow)]
    tags_none = [t for *_, t in debug_triangulation_edges(blue, yellow, orange_cones=None)]
    assert tags_base == tags_none and 'start' not in tags_base
