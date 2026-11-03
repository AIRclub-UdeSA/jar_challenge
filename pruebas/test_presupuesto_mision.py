import math

import pytest

from equipo_jar.mapa import Grid
from equipo_jar.presupuesto import fits_extra, return_due, route_seconds
from equipo_jar.vuelta import steps_to_start, way_back_m


def budget(**changes):
    values = dict(outward_s=20.0, look_s=2.0, home_s=30.0,
                  reserve_s=45.0, return_at_s=480.0, finish_at_s=570.0)
    return values | changes


def estimate(distance):
    return route_seconds(distance, speed_mps=0.2, factor=1.5, overhead_s=8.0)


def corridor(door=False, blocked=False, closed=False):
    width, height = 80, 40
    data = []
    for j in range(height):
        for i in range(width):
            edge = i in (0, width - 1) or j in (0, height - 1)
            wall = j in (19, 20) and (i < 60 or closed)
            opening = door and 4 <= i < 16
            obstacle = blocked and 4 <= i < 16 and 18 <= j <= 21
            rim = blocked and 2 <= i < 18 and 16 <= j <= 23
            data.append(100 if edge or (wall and not opening) or obstacle
                        else 99 if rim else 0)
    return Grid(width, height, 0.05, data)


def test_estimate_includes_slowdown_and_trip_overhead():
    assert estimate(10.0) == pytest.approx(83.0)
    assert estimate(0.0) == pytest.approx(8.0)


@pytest.mark.parametrize('distance', [None, math.nan, math.inf, -math.inf,
                                     -1.0, True, '1', 10 ** 1000])
def test_unknown_or_invalid_distance_has_no_estimate(distance):
    assert estimate(distance) is None


@pytest.mark.parametrize(('key', 'value'), [
    ('speed_mps', 0.0), ('speed_mps', -1.0), ('speed_mps', math.nan),
    ('speed_mps', True), ('factor', 0.99), ('factor', math.inf),
    ('factor', False), ('overhead_s', -1.0), ('overhead_s', None),
])
def test_invalid_estimate_parameters_have_no_estimate(key, value):
    settings = dict(speed_mps=0.2, factor=1.5, overhead_s=8.0)
    settings[key] = value
    assert route_seconds(1.0, **settings) is None


def test_overflow_does_not_produce_a_finite_estimate():
    assert route_seconds(1e308, speed_mps=0.1, factor=2.0, overhead_s=8.0) is None


def test_extra_view_must_fit_its_own_return_cost():
    assert fits_extra(400.0, **budget(home_s=30.0))
    assert not fits_extra(400.0, **budget(home_s=120.0))


@pytest.mark.parametrize(('now', 'changes', 'expected'), [
    (400.0, {'home_s': 102.9}, True),
    (400.0, {'home_s': 103.0}, False),
    (400.0, {'home_s': 103.1}, False),
    (457.9, {}, True),
    (458.0, {}, False),
    (458.1, {}, False),
])
def test_both_deadlines_require_strict_spare_time(now, changes, expected):
    assert fits_extra(now, **budget(**changes)) is expected


@pytest.mark.parametrize(('key', 'value'), [
    ('outward_s', None), ('outward_s', -1.0), ('outward_s', math.inf),
    ('look_s', -1.0), ('look_s', False), ('home_s', None),
    ('home_s', math.nan), ('reserve_s', -1.0), ('reserve_s', True),
    ('return_at_s', math.inf), ('finish_at_s', math.nan),
    ('finish_at_s', 480.0), ('return_at_s', 571.0),
])
def test_invalid_budget_denies_extra_search(key, value):
    assert not fits_extra(400.0, **budget(**{key: value}))


@pytest.mark.parametrize('now', [-1.0, None, math.nan, math.inf, False])
def test_invalid_elapsed_time_denies_extra_search(now):
    assert not fits_extra(now, **budget())


@pytest.mark.parametrize('home', [None, math.nan, math.inf, -1.0, True])
def test_unknown_return_estimate_keeps_hard_cap_without_granting_extra_time(home):
    values = dict(home_s=home, reserve_s=45.0, return_at_s=480.0, finish_at_s=570.0)
    assert not return_due(479.0, **values)
    assert return_due(480.0, **values)
    assert not fits_extra(400.0, **budget(home_s=home))


def test_current_return_reserve_can_advance_the_return():
    values = dict(home_s=100.0, reserve_s=45.0, return_at_s=480.0, finish_at_s=570.0)
    assert not return_due(424.9, **values)
    assert return_due(425.0, **values)
    assert return_due(426.0, **values)


@pytest.mark.parametrize(('key', 'value'), [
    ('reserve_s', -1.0), ('reserve_s', math.nan),
    ('return_at_s', None), ('return_at_s', True),
    ('finish_at_s', math.inf), ('finish_at_s', 480.0),
])
def test_invalid_return_configuration_fails_closed(key, value):
    values = dict(home_s=10.0, reserve_s=45.0, return_at_s=480.0, finish_at_s=570.0)
    values[key] = value
    assert return_due(400.0, **values)


@pytest.mark.parametrize('now', [None, -1.0, math.nan, math.inf, True])
def test_invalid_clock_fails_closed_for_return(now):
    assert return_due(now, home_s=10.0, reserve_s=45.0,
                      return_at_s=480.0, finish_at_s=570.0)


def test_zero_duration_and_margin_are_valid_with_room_before_both_limits():
    assert fits_extra(0.0, **budget(outward_s=0.0, look_s=0.0,
                                  home_s=0.0, reserve_s=0.0))


def test_budget_uses_the_long_return_from_the_proposed_destination():
    grid = corridor()
    start, robot, destination = (0.5, 0.5), (0.9, 0.5), (0.5, 1.5)
    home_field = steps_to_start(grid, *start, 0.15, 25, 0.3)
    outward_field = steps_to_start(grid, *robot, 0.15, 25, 0.3)
    assert grid.cell(*destination) in home_field
    assert grid.cell(*destination) in outward_field
    home_m = way_back_m(grid, home_field, *destination, 0.0)
    outward_m = way_back_m(grid, outward_field, *destination, 0.0)
    current_home_m = way_back_m(grid, home_field, *robot, 0.0)
    assert math.dist(start, destination) == pytest.approx(1.0)
    assert home_m > 6.0
    assert current_home_m < 0.5
    values = budget(outward_s=estimate(outward_m))
    assert not fits_extra(416.0, **(values | {'home_s': estimate(home_m)}))
    assert fits_extra(416.0, **(values | {'home_s': estimate(current_home_m)}))


def test_live_blocked_shortcut_prevents_an_optimistic_return_budget():
    static = corridor(door=True)
    live = corridor(door=True, blocked=True)
    start, destination = (0.5, 0.5), (0.5, 1.5)
    static_field = steps_to_start(static, *start, 0.15, 25, 0.3)
    live_field = steps_to_start(live, *start, 0.0, 98, 0.3)
    assert live.cell(*destination) in live_field
    static_home = estimate(way_back_m(static, static_field, *destination, 0.0))
    live_home = estimate(way_back_m(live, live_field, *destination, 0.0))
    assert static_home < live_home
    assert fits_extra(471.0, **budget(outward_s=0.0, home_s=static_home))
    assert not fits_extra(471.0, **budget(outward_s=0.0, home_s=live_home))


def test_disconnected_destination_does_not_get_a_snapped_route():
    grid = corridor(closed=True)
    destination = (0.5, 1.5)
    field = steps_to_start(grid, 0.5, 0.5, 0.15, 25, 0.3)
    assert grid.cell(*destination) not in field
    assert way_back_m(grid, field, *destination, 0.0) is None
    assert not fits_extra(100.0, **budget(home_s=estimate(None)))
