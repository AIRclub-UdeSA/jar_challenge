"""Ordering checks on generated observation tasks, outside the shipped tree."""
from concurrent.futures import CancelledError
from dataclasses import replace
import math
import random
from threading import Event

import pytest

from equipo_jar.cobertura_plan import (
    Pose, Settings, _order_checkpoints, improve_order, relocate_order, route_metrics)


class MetricCosts:
    def between(self, first, last):
        return math.dist(first, last)


def view(x, y, mask, yaw=0.0):
    return (x, y), Pose(float(x), float(y), yaw), mask


def settings():
    return replace(Settings(), travel_speed=1.0, turn_speed=1.0, observation_time=1.0)


@pytest.mark.parametrize('cached_turns', [False, True])
def test_route_clock_is_exact_for_directed_costs_and_repeated_views(cached_turns):
    from equipo_jar.cobertura_plan import _RouteClock

    class DirectedCosts:
        def between(self, first, last):
            return math.dist(first, last) + (0.03 if first < last else 0.07)

    class WithTurns(DirectedCosts):
        def turn_between(self, first, last):
            return abs(math.atan2(math.sin(first - last), math.cos(first - last))) + 0.125

    costs = WithTurns() if cached_turns else DirectedCosts()
    config = replace(settings(), travel_speed=0.31, turn_speed=0.67, observation_time=0.7)
    tasks = [view(i % 3, i % 2, 1 << i, yaw) for i, yaw in enumerate(
        (-math.pi - 1e-12, math.pi + 1e-12, 0.0, 0.0, 0.5, -0.5, 3.0, 4.0, 0.5))]
    tasks.append(tasks[2])
    clock = _RouteClock((0, 0), -math.pi, costs, config)
    rng = random.Random(281026)
    for _ in range(80):
        rng.shuffle(tasks)
        route = tasks[:rng.randrange(len(tasks) + 1)]
        assert clock(route) == route_metrics(route, (0, 0), -math.pi, costs, config, 511).duration


def test_duration_prefilter_keeps_the_authoritative_order_with_fewer_full_metrics(monkeypatch):
    from equipo_jar import cobertura_plan as planner

    rng = random.Random(918)
    tasks = [view(rng.randrange(1, 10), rng.randrange(1, 10), 1 << i, rng.random() * math.tau)
             for i in range(14)]
    metrics_calls = []
    original = planner.route_metrics
    def counted(*args, **kwargs):
        metrics_calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(planner, 'route_metrics', counted)
    optimized = planner.improve_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), (1 << 14) - 1)
    optimized_calls = len(metrics_calls)
    metrics_calls.clear()
    # Disable only the new early rejection; all original acceptance guards run.
    monkeypatch.setattr(planner, '_RouteClock', lambda *args: lambda choices: -math.inf)
    reference = planner.improve_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), (1 << 14) - 1)
    assert optimized == reference
    assert optimized_calls < len(metrics_calls)


def test_relocation_shortens_search_without_losing_a_view():
    tasks = [view(10, 0, 1), view(1, 0, 2)]
    result = relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 3, 30.0, 0.0)
    assert result == tasks[::-1]
    before = route_metrics(tasks, (0, 0), 0.0, MetricCosts(), settings(), 3)
    after = route_metrics(result, (0, 0), 0.0, MetricCosts(), settings(), 3)
    assert before.duration == 21.0
    assert after.duration == 12.0
    assert before.mean_first_view_time == 16.0
    assert after.mean_first_view_time == 7.0
    assert after.duration + after.return_distance == before.duration + before.return_distance


def test_clock_guard_rejects_a_mean_improvement_that_delays_early_coverage():
    tasks = [view(10, 0, (1 << 10) - 1), view(1, 0, 1 << 10)]
    coarse = relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 2047, 30.0, 0.0)
    guarded = relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 2047, 11.0, 0.0)
    assert coarse == tasks[::-1]
    assert guarded == tasks


def test_start_delay_aligns_checks_with_the_predicted_clock():
    tasks = [view(10, 0, (1 << 10) - 1), view(1, 0, 1 << 10)]
    # A 12 s checkpoint is at route second 11 when startup takes 1 s.
    result = relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 2047, 12.0, 1.0)
    assert result == tasks


def test_a_lower_mean_does_not_justify_longer_total_search():
    tasks = [view(1, 0, 1 << 10), view(10, 0, (1 << 10) - 1)]
    result = relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 2047, 30.0, 0.0)
    assert result == tasks


@pytest.mark.parametrize('interval,delay', [
    (0.0, 0.0), (0.5, 0.0), (math.nan, 0.0), (math.inf, 0.0),
    (30.0, -1.0), (30.0, math.nan), (30.0, math.inf),
])
def test_invalid_clock_settings_are_rejected(interval, delay):
    with pytest.raises(ValueError):
        relocate_order([], (0, 0), 0.0, MetricCosts(), settings(), 0, interval, delay)


def test_empty_and_single_view_routes_are_preserved():
    for tasks in ([], [view(1, 0, 1)]):
        assert relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 1, 30.0, 8.0) == tasks


def test_cancelled_ordering_stops_even_without_uncached_travel():
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        relocate_order([view(1, 0, 1)], (0, 0), 0.0, MetricCosts(), settings(), 1,
                       30.0, 0.0, cancel)


def test_cleanup_handles_improvements_unlocked_by_later_moves():
    tasks = [view(x, y, mask, yaw) for x, y, mask, yaw in (
        (7, 3, 8903, 1.0), (8, 8, 7220, 0.0), (6, 4, 7050, 1.0),
        (5, 2, 8278, 2.0), (6, 7, 12026, 0.0), (6, 2, 9069, 0.0),
        (5, 6, 8493, 2.0), (1, 5, 9068, 2.0), (5, 7, 8708, 0.0),
        (8, 6, 7567, 0.0))]
    config = replace(settings(), order_check_interval_s=10.0, order_start_delay_s=0.0)
    result = improve_order(tasks, (0, 0), 0.0, MetricCosts(), config, 16383)
    metrics = route_metrics(result, (0, 0), 0.0, MetricCosts(), config, 16383)
    assert sorted(result) == sorted(tasks)
    # A single relocation sweep stalls at 44.23 s on this generated task.
    assert metrics.duration < 42.0
    assert metrics.mean_first_view_time < 9.9
    assert max(count for seconds, count in metrics.coverage_curve if seconds <= 10.0) >= 8
    assert max(count for seconds, count in metrics.coverage_curve if seconds <= 20.0) == 14


def test_cancellation_during_initial_metrics_stops_before_two_opt_search():
    cancel = Event()
    class CancellingCosts(MetricCosts):
        calls = 0
        def between(self, first, last):
            self.calls += 1
            cancel.set()
            return super().between(first, last)
    costs = CancellingCosts()
    tasks = [view(1, 0, 1), view(0, 2, 2), view(3, 0, 4)]
    with pytest.raises(CancelledError):
        improve_order(tasks, (0, 0), 0.0, costs, settings(), 7, cancel)
    assert costs.calls <= len(tasks) + 1


def test_clock_checks_are_bounded_by_view_count_for_long_routes():
    curve = [(0.0, 0), (1000000.0, 5), (2000000.0, 5), (3000000.0, 8)]
    assert _order_checkpoints(curve, 0.001, 0.0) == [(1000000.0, 5), (3000000.0, 8)]


def test_generated_routes_preserve_objectives_and_all_clock_checks():
    rng = random.Random(9182)
    options = [(x, y) for x in range(1, 9) for y in range(1, 9)]
    for _ in range(40):
        tasks = [view(x, y, rng.randrange(1, 1 << 14), rng.uniform(-math.pi, math.pi))
                 for x, y in rng.sample(options, 6)]
        original = list(tasks)
        interval, delay = rng.uniform(3.0, 20.0), rng.uniform(0.0, 8.0)
        result = relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 16383, interval, delay)
        before = route_metrics(tasks, (0, 0), 0.0, MetricCosts(), settings(), 16383)
        after = route_metrics(result, (0, 0), 0.0, MetricCosts(), settings(), 16383)
        assert tasks == original
        assert sorted(result) == sorted(tasks)
        assert after.duration <= before.duration + 1e-9
        assert after.mean_first_view_time <= before.mean_first_view_time + 1e-9
        assert after.duration + after.return_distance <= before.duration + before.return_distance + 1e-9
        for k in range(1, math.ceil((before.duration + delay) / interval) + 1):
            seconds = k * interval - delay
            required = max((count for t, count in before.coverage_curve if t <= seconds + 1e-9), default=0)
            actual = max((count for t, count in after.coverage_curve if t <= seconds + 1e-9), default=0)
            assert actual >= required
        assert result == relocate_order(tasks, (0, 0), 0.0, MetricCosts(), settings(), 16383, interval, delay)
