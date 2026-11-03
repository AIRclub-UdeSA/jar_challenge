from dataclasses import FrozenInstanceError
import math

import pytest

from equipo_jar.historial_vistas import AttemptedView, ObservedView, append_view
from equipo_jar.mapa import Pose


def append(history, view, **changes):
    settings = dict(max_items=1200, min_interval_s=1.0,
                    min_shift_m=0.10, min_turn_rad=0.30)
    return append_view(history, view, **(settings | changes))


def observed(at_s=0.0, x=0.0, y=0.0, yaw=0.0):
    return ObservedView(Pose(x, y, yaw), at_s)


def test_failed_requested_goal_is_separate_from_actual_pose_history():
    actual = Pose(1.0, 1.0, 0.0)
    requested = Pose(3.0, 1.0, 0.0)
    attempt = AttemptedView(requested, 'failed')
    history = append((), ObservedView(actual, 10.0))
    assert history == (ObservedView(actual, 10.0),)
    assert attempt.pose == requested
    assert all(view.pose != requested for view in history)


@pytest.mark.parametrize('outcome', ['arrived', 'failed', 'rejected', 'cancelled'])
def test_attempt_outcomes_do_not_synthesize_observations(outcome):
    attempt = AttemptedView(Pose(1.0, 2.0, 0.0), outcome)
    with pytest.raises(ValueError):
        append((), attempt)


def test_records_and_pose_are_immutable():
    view = observed()
    attempt = AttemptedView(view.pose, 'arrived')
    with pytest.raises(FrozenInstanceError):
        view.at_s = 10.0
    with pytest.raises(FrozenInstanceError):
        attempt.outcome = 'failed'
    with pytest.raises(FrozenInstanceError):
        view.pose.x = 5.0


@pytest.mark.parametrize(('at_s', 'x', 'y', 'yaw', 'expected'), [
    (0.1, 0.0, 0.0, 0.0, False),
    (0.999, 0.0, 0.0, 0.0, False),
    (1.0, 0.0, 0.0, 0.0, True),
    (1.001, 0.0, 0.0, 0.0, True),
    (0.1, 0.099, 0.0, 0.0, False),
    (0.1, 0.1, 0.0, 0.0, True),
    (0.1, 0.0, -0.1, 0.0, True),
    (0.1, 0.0, 0.0, 0.299, False),
    (0.1, 0.0, 0.0, 0.3, True),
    (0.1, 0.0, 0.0, -0.3, True),
])
def test_sampling_tracks_time_translation_and_heading(at_s, x, y, yaw, expected):
    history = (observed(),)
    view = observed(at_s, x, y, yaw)
    expected_history = history + (view,) if expected else history
    assert append(history, view) == expected_history


def test_small_diagonal_changes_use_euclidean_distance():
    history = (observed(),)
    assert append(history, observed(0.1, x=0.06, y=0.06)) == history
    view = observed(0.1, x=0.08, y=0.08)
    assert append(history, view) == history + (view,)


def test_heading_wrap_does_not_invent_a_large_turn():
    history = (observed(yaw=math.radians(179.0)),)
    assert append(history, observed(0.1, yaw=math.radians(-179.0))) == history
    assert len(append(history, observed(0.1, yaw=math.radians(-150.0)))) == 2


def test_equivalent_heading_after_one_revolution_does_not_grow_history():
    history = (observed(),)
    assert append(history, observed(0.1, yaw=2 * math.pi)) == history


@pytest.mark.parametrize('at_s', [9.0, 10.0])
def test_older_or_equal_timestamp_does_not_add_a_view_even_after_a_shift(at_s):
    history = (observed(10.0),)
    assert append(history, observed(at_s, x=1.0)) == history


def test_history_is_bounded_and_does_not_mutate_the_input():
    history = tuple(observed(float(i)) for i in range(1200))
    result = append(history, observed(1200.0))
    assert len(result) == 1200
    assert result[0].at_s == 1.0
    assert result[-1].at_s == 1200.0
    assert history[0].at_s == 0.0
    assert history[-1].at_s == 1199.0


def test_lowered_live_history_cap_also_applies_when_view_is_not_added():
    history = (observed(0.0), observed(1.0), observed(2.0))
    assert append(history, observed(2.1), max_items=2) == history[-2:]


def test_zero_sampling_thresholds_allow_every_new_timestamp():
    history = (observed(),)
    view = observed(0.001)
    assert append(history, view, min_interval_s=0.0, min_shift_m=0.0,
                  min_turn_rad=0.0) == history + (view,)


@pytest.mark.parametrize('at_s', [-1.0, math.nan, math.inf, None, True, '1', 10 ** 1000])
def test_invalid_observation_timestamp_is_rejected(at_s):
    with pytest.raises(ValueError):
        observed(at_s)


@pytest.mark.parametrize('pose', [None, (1.0, 2.0, 0.0), True])
def test_records_require_a_validated_pose(pose):
    with pytest.raises(ValueError):
        ObservedView(pose, 1.0)
    with pytest.raises(ValueError):
        AttemptedView(pose, 'arrived')


@pytest.mark.parametrize('outcome', ['', 'moving', 'succeeded', None, True, []])
def test_invalid_attempt_outcome_is_rejected(outcome):
    with pytest.raises(ValueError):
        AttemptedView(Pose(0.0, 0.0, 0.0), outcome)


@pytest.mark.parametrize(('key', 'value'), [
    ('max_items', 0), ('max_items', -1), ('max_items', 1.5), ('max_items', True),
    ('min_interval_s', -1.0), ('min_interval_s', math.inf),
    ('min_shift_m', -0.1), ('min_shift_m', None),
    ('min_turn_rad', math.nan), ('min_turn_rad', False),
])
def test_invalid_history_settings_are_rejected(key, value):
    with pytest.raises(ValueError):
        append((), observed(), **{key: value})


@pytest.mark.parametrize('history', [[], (object(),),
                                     (AttemptedView(Pose(0.0, 0.0, 0.0), 'failed'),)])
def test_attempts_and_mutable_or_untyped_history_are_rejected(history):
    with pytest.raises(ValueError):
        append(history, observed(2.0))


def test_nonchronological_history_is_rejected():
    with pytest.raises(ValueError):
        append((observed(2.0), observed(1.0)), observed(3.0))
