"""Corridor walk entry when the car sits on/just past a gate midpoint (boundary._walk_corridor)."""
from fsae_planning.boundary import _build_corridor_path, build_path_walls
import numpy as np
import pytest

MID_MAX = 6.5     # the sim's mid_max_gate (FSDS diagonals are ~5.3 m)


def _straight(spacing, half_w, x_max=70.0):
    """Straight track along +x, blue at +y, cones paired straight across."""
    xs = np.arange(0.0, x_max + 1e-9, spacing)
    blue = np.c_[xs, np.full_like(xs, half_w)]
    yellow = np.c_[xs, np.full_like(xs, -half_w)]
    return blue, yellow


def _arc_len(path):
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


# FSDS-like spacing: gate midpoints every 2 m on y=0 (rungs at x=0,4,8,...,
# diagonals at x=2,6,10,...).
@pytest.mark.parametrize('gate_x', [8.0, 10.0, 12.0, 14.0])
@pytest.mark.parametrize('dx', [-0.1, 0.1, 0.2])
@pytest.mark.parametrize('dy', [-0.3, -0.2, 0.0, 0.2, 0.3])
def test_car_next_to_gate_midpoint_reaches_horizon(gate_x, dx, dy):
    blue, yellow = _straight(4.0, 1.75)
    car = np.array([gate_x + dx, dy])

    ordered, _ = _build_corridor_path(blue, yellow, car, 0.0, mid_max=MID_MAX)
    assert len(ordered) > 1
    assert ordered[0, 0] > car[0]                  # no point at/behind the car
    assert ordered[-1, 0] >= car[0] + 24.0

    cl, *_ = build_path_walls(blue, yellow, car, 0.0, mid_max_gate=MID_MAX)
    assert _arc_len(cl) > 20.0                     # clamped to plan_horizon (25 m)
    assert np.all(np.diff(cl[:, 0]) > 0.0)         # no hook back at the start


def test_car_behind_diagonal_but_past_its_midpoint():
    # 5 m spacing, 3 m wide: the diagonal (5,1.5)-(10,-1.5) has its midpoint
    # at (7.5, 0) but crosses y=-0.3 at x~7.9. At (7.91, -0.3) the car is
    # still behind that gate yet 0.51 m past its midpoint.
    blue, yellow = _straight(5.0, 1.5)
    car = np.array([7.91, -0.3])
    ordered, _ = _build_corridor_path(blue, yellow, car, 0.0, mid_max=MID_MAX)
    assert len(ordered) > 1
    assert ordered[0, 0] > car[0]
    assert ordered[-1, 0] >= car[0] + 24.0


def test_prev_dir_seeds_the_entry():
    # Same pose, heading slightly off but prev_dir along the track.
    blue, yellow = _straight(4.0, 1.75)
    car = np.array([6.1, 0.2])
    ordered, _ = _build_corridor_path(blue, yellow, car, 0.3, mid_max=MID_MAX,
                                      prev_dir=np.array([1.0, 0.0]))
    assert len(ordered) > 1
    assert ordered[-1, 0] >= car[0] + 24.0
