"""Tests for candidate_fusion.py. No ROS needed."""
import math

import numpy as np
import pytest

from equipo_jar.candidate_fusion import DEFAULTS, CandidateFusion


def new(**overrides):
    params = dict(DEFAULTS)
    params.update(overrides)
    return CandidateFusion(params)


def two_viewpoints(k):
    """Robot poses that alternate between two spots 0.5 m apart (baseline satisfied)."""
    return (0.0, 0.0) if k % 2 == 0 else (0.5, 0.0)


def test_nearby_observations_join_the_same_candidate():
    f = new()
    for k, z in enumerate([(2.0, 1.0), (2.1, 1.05), (1.95, 0.9)]):
        f.observe(z, 0.1, two_viewpoints(k), k * 0.2)
    assert len(f.candidates) == 1 and f.candidates[0].n_obs == 3


def test_distant_observations_open_separate_candidates():
    f = new()
    f.observe((2.0, 1.0), 0.1, (0.0, 0.0), 0.0)
    f.observe((3.5, 1.0), 0.1, (0.0, 0.0), 0.1)
    assert len(f.candidates) == 2
    assert [c.id for c in f.candidates] == [1, 2]


def test_weighted_mean_reduces_error_and_covariance_shrinks():
    rng = np.random.default_rng(0)
    f = new()
    truth = np.array([2.0, -1.0])
    for k in range(30):
        z = truth + rng.normal(0, 0.1, 2)
        f.observe(tuple(z), 0.1, two_viewpoints(k), k * 0.2)
    c = f.candidates[0]
    assert np.linalg.norm(c.mu - truth) < 0.05
    std = c.max_std(DEFAULTS['min_baseline_m'], DEFAULTS['single_view_sigma_m'])
    # Information part: 0.1 / sqrt(30) ≈ 0.018; sample part adds a similar amount.
    assert 0.01 < std < 0.06


def test_close_observations_weigh_more_than_far_ones():
    f = new()
    f.observe((2.0, 0.0), 0.4, (0.0, 0.0), 0.0)     # sigma 0.4 -> weight 6.25
    f.observe((2.3, 0.0), 0.1, (0.0, 0.0), 0.1)     # sigma 0.1 -> weight 100
    mu = f.candidates[0].mu
    assert mu[0] == pytest.approx((6.25 * 2.0 + 100 * 2.3) / 106.25)


def test_single_viewpoint_keeps_the_covariance_floor():
    f = new(single_view_sigma_m=0.15, min_baseline_m=0.3)
    for k in range(10):
        f.observe((2.0, 1.0), 0.05, (0.0, 0.0), k * 0.1)   # always from the same spot
    c = f.candidates[0]
    assert c.baseline() == 0.0
    assert c.max_std(0.3, 0.15) == pytest.approx(0.15)
    # Mission's filter (std <= 0.10) must NOT pass this one.
    assert c.max_std(0.3, 0.15) > 0.10


def test_second_viewpoint_lifts_the_floor():
    f = new(single_view_sigma_m=0.15, min_baseline_m=0.3)
    for k in range(10):
        f.observe((2.0, 1.0), 0.05, two_viewpoints(k), k * 0.1)
    c = f.candidates[0]
    assert c.baseline() == pytest.approx(0.5)
    assert c.max_std(0.3, 0.15) < 0.05


def test_observation_without_robot_pose_counts_for_position_only():
    f = new()
    f.observe((2.0, 1.0), 0.1, None, 0.0)
    f.observe((2.0, 1.0), 0.1, None, 0.1)
    c = f.candidates[0]
    assert c.n_obs == 2 and c.robot_poses == [] and c.baseline() == 0.0


def test_sample_covariance_reflects_an_elongated_cloud():
    # Wide association so every sample joins the same candidate; spread 0.3 in x, 0.02 in y.
    f = new(min_baseline_m=0.0, association_dist_m=1.5, merge_dist_m=0.1)
    rng = np.random.default_rng(2)
    for k in range(200):
        z = (2.0 + rng.normal(0, 0.3), 1.0 + rng.normal(0, 0.02))
        f.observe(z, 0.1, None, k * 0.1)
    assert len(f.candidates) == 1
    S = f.candidates[0].sample_cov()
    assert S[0, 0] == pytest.approx(0.09, rel=0.3)
    assert S[1, 1] == pytest.approx(0.0004, rel=0.3)
    cov = f.candidates[0].covariance(0.0, 0.0)
    assert cov[0, 0] > 5 * cov[1, 1]            # x much more uncertain than y
    assert cov[0, 1] == pytest.approx(cov[1, 0])


def test_merge_keeps_the_lower_id_and_sums_observations():
    f = new(association_dist_m=0.4, merge_dist_m=0.3)
    f.observe((2.0, 1.0), 0.1, (0.0, 0.0), 0.0)             # id 1
    f.observe((2.55, 1.0), 0.1, (0.0, 0.0), 0.1)            # id 2, 0.55 m away: separate
    assert len(f.candidates) == 2
    # A very confident observation at 2.28 joins id 2 (closer) and drags it to ~2.28,
    # which is less than merge_dist from id 1: they merge and id 1 survives.
    f.observe((2.28, 1.0), 0.01, (0.0, 0.0), 0.2)
    assert len(f.candidates) == 1
    c = f.candidates[0]
    assert c.id == 1
    assert c.n_obs == 3
    assert c.mu[0] == pytest.approx(2.28, abs=0.05)


def test_purge_drops_weak_stale_candidates_only():
    f = new(stale_timeout_s=5.0, min_obs_to_keep=5)
    for k in range(5):
        f.observe((2.0, 1.0), 0.1, two_viewpoints(k), k * 0.1)   # strong: 5 obs
    f.observe((5.0, 5.0), 0.1, (0.0, 0.0), 0.5)                  # weak: 1 obs
    assert len(f.candidates) == 2
    f.purge(t=3.0)
    assert len(f.candidates) == 2                                # not stale yet
    f.purge(t=10.0)
    assert [c.n_obs for c in f.candidates] == [5]


def test_report_has_the_candidata_fields_sorted_by_id():
    f = new()
    f.observe((5.0, 5.0), 0.1, (0.0, 0.0), 0.0)
    for k in range(3):
        f.observe((2.0, 1.0), 0.1, two_viewpoints(k), 1.0 + k)
    assert f.changed
    rep = f.report()
    assert not f.changed
    assert [r['id'] for r in rep] == [1, 2]
    r = rep[1]
    assert r['n_obs'] == 3 and r['t_last'] == pytest.approx(3.0)
    assert len(r['cov_xy']) == 4 and r['cov_xy'][1] == pytest.approx(r['cov_xy'][2])
    assert r['max_std'] == pytest.approx(
        math.sqrt(max(np.linalg.eigvalsh(np.array(r['cov_xy']).reshape(2, 2)))))


def test_non_finite_observation_raises():
    with pytest.raises(ValueError):
        new().observe((float('nan'), 0.0), 0.1, (0.0, 0.0), 0.0)


def test_live_params_are_read_at_call_time():
    params = dict(DEFAULTS)
    f = CandidateFusion(params)
    f.observe((2.0, 1.0), 0.1, None, 0.0)
    params['association_dist_m'] = 0.05          # changed "live"
    params['merge_dist_m'] = 0.05
    f.observe((2.2, 1.0), 0.1, None, 0.1)
    assert len(f.candidates) == 2                 # 0.2 m apart no longer associates
