"""cone_memory's store rules (ConeMemoryStore), without ROS."""
import pytest

from fsae_planning.cone_memory import ConeMemoryStore

CAR = (0.0, 0.0, 0.0)   # at the origin, facing +x


def _mem(**kw):
    m = ConeMemoryStore(**kw)
    m.update_pose(*CAR)
    return m


def test_min_hits_hides_one_frame_phantoms():
    m = _mem(min_hits=2)
    m.add_frame({'blue': [(5.0, 1.5)]}, CAR, 0.0)
    assert m.published('blue') == []
    m.add_frame({'blue': [(5.1, 1.5)]}, CAR, 0.1)
    assert len(m.published('blue')) == 1


def test_merge_averages_and_separates():
    m = _mem(min_hits=1, filter_window=5)
    m.add_frame({'blue': [(5.0, 1.5), (9.0, 1.5)]}, CAR, 0.0)
    m.add_frame({'blue': [(5.2, 1.5), (9.0, 1.5)]}, CAR, 0.1)
    pub = sorted(m.published('blue'))
    assert len(pub) == 2
    assert pub[0] == pytest.approx((5.1, 1.5))


def test_cone_beside_the_car_outlives_the_shim():
    """Out of view (beside the car) and within keep_radius: kept past persistence_sec."""
    m = _mem(min_hits=1, persistence_sec=1.0, keep_radius=10.0, max_age=30.0)
    m.add_frame({'blue': [(0.0, 1.5)]}, CAR, 0.0)
    m.expire(5.0)
    assert m.published('blue') == [(0.0, 1.5)]
    m.expire(31.0)                                   # max_age
    assert m.published('blue') == []


def test_cone_left_behind_is_forgotten():
    m = _mem(min_hits=1, persistence_sec=1.0, keep_radius=10.0)
    m.add_frame({'blue': [(0.0, 1.5)]}, CAR, 0.0)
    m.update_pose(4.0, 0.0, 0.0)
    m.update_pose(8.0, 0.0, 0.0)
    m.update_pose(12.0, 0.0, 0.0)
    m.expire(0.5)
    assert len(m.published('blue')) == 1            # still within persistence_sec
    m.expire(2.0)
    assert m.published('blue') == []                # 12 m away, not seen for 2 s


def test_in_view_misses_drop_a_phantom():
    m = _mem(min_hits=1, max_misses=3, view_hfov_deg=110.0, view_range=16.0)
    m.add_frame({'yellow': [(6.0, 0.0)]}, CAR, 0.0)
    for k in range(1, 3):
        m.add_frame({'yellow': []}, CAR, 0.1 * k)
        assert len(m.published('yellow')) == 1
    m.add_frame({'yellow': []}, CAR, 0.3)
    assert m.published('yellow') == []


def test_out_of_view_cone_collects_no_misses():
    m = _mem(min_hits=1, max_misses=1)
    m.add_frame({'blue': [(0.0, 2.0), (-3.0, 1.5), (20.0, 0.0)]}, CAR, 0.0)
    m.add_frame({'blue': []}, CAR, 0.1)
    assert len(m.published('blue')) == 3            # beside, behind, past view_range


def test_a_redetection_resets_misses():
    m = _mem(min_hits=1, max_misses=2)
    m.add_frame({'blue': [(6.0, 1.0)]}, CAR, 0.0)
    m.add_frame({'blue': []}, CAR, 0.1)
    m.add_frame({'blue': [(6.0, 1.0)]}, CAR, 0.2)
    m.add_frame({'blue': []}, CAR, 0.3)
    assert len(m.published('blue')) == 1


def test_pose_jump_clears_the_store():
    m = _mem(min_hits=1, reset_jump=5.0)
    m.add_frame({'blue': [(5.0, 1.5)]}, CAR, 0.0)
    assert m.update_pose(0.5, 0.0, 0.0) is False
    assert m.update_pose(20.0, 0.0, 0.0) is True
    assert m.published('blue') == []


def test_colours_are_kept_apart():
    m = _mem(min_hits=1)
    m.add_frame({'blue': [(5.0, 1.5)], 'yellow': [(5.0, 1.6)]}, CAR, 0.0)
    assert len(m.published('blue')) == 1 and len(m.published('yellow')) == 1
