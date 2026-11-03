from concurrent.futures import CancelledError
import math
from threading import Event

import pytest

from equipo_jar.mapa import Grid, Pose
from equipo_jar import vuelta
from equipo_jar.rutas_presupuesto import build_route_field, estimate_route


def corridor(closed=False):
    width, height = 80, 40
    data = [100 if i in (0, 79) or j in (0, 39)
            or (j in (19, 20) and (i < 60 or closed)) else 0
            for j in range(height) for i in range(width)]
    return Grid(width, height, 0.05, data)


def build(grid=None, at_s=10.0):
    return build_route_field(grid or corridor(), Pose(0.9, 0.5, 0.0),
                             Pose(0.5, 0.5, 0.0), at_s)


def test_estimate_uses_the_routes_from_robot_and_destination_home():
    field = build()
    result = estimate_route(field, Pose(0.5, 1.5, 0.0), 11.0, max_age_s=2.5)
    assert result.outward_m == pytest.approx(5.6)
    assert result.home_m == pytest.approx(6.0)
    assert result.at_s == 10.0
    current = estimate_route(field, Pose(0.9, 0.5, 0.0), 11.0, max_age_s=2.5)
    assert current.outward_m == 0.0
    assert current.home_m < 0.5


def test_disconnected_endpoint_is_not_snapped_across_the_wall():
    field = build(corridor(closed=True))
    assert estimate_route(field, Pose(0.5, 1.5, 0.0), 11.0, max_age_s=2.5) is None


@pytest.mark.parametrize('value', [-1, 99, 100])
def test_unknown_inscribed_and_lethal_goals_are_excluded(value):
    grid = corridor()
    data = list(grid.data)
    cell = grid.cell(0.5, 1.5)
    data[cell[1] * grid.width + cell[0]] = value
    field = build(Grid(grid.width, grid.height, grid.resolution, data))
    assert estimate_route(field, Pose(0.5, 1.5, 0.0), 11.0, max_age_s=2.5) is None


def test_inflation_is_not_applied_twice():
    grid = Grid(7, 3, 0.05, [100] * 7 + [100, 50, 98, 50, 98, 50, 100] + [100] * 7)
    field = build_route_field(grid, Pose(0.075, 0.075, 0.0),
                             Pose(0.275, 0.075, 0.0), 10.0)
    result = estimate_route(field, Pose(0.175, 0.075, 0.0), 10.0, max_age_s=2.5)
    assert result.outward_m == pytest.approx(0.1)
    assert result.home_m == pytest.approx(0.1)


@pytest.mark.parametrize(('robot', 'start'), [
    (Pose(-0.01, 0.5, 0.0), Pose(0.5, 0.5, 0.0)),
    (Pose(0.9, 0.5, 0.0), Pose(0.01, 0.5, 0.0)),
])
def test_invalid_endpoints_do_not_get_a_fabricated_start(robot, start):
    assert build_route_field(corridor(), robot, start, 10.0) is None


@pytest.mark.parametrize(('now', 'age', 'accepted'), [
    (9.9, 2.5, False), (12.5, 2.5, True), (12.6, 2.5, False),
    (11.0, 0.0, False), (math.nan, 2.5, False), (11.0, math.inf, False),
])
def test_cache_freshness_uses_the_calculation_timestamp(now, age, accepted):
    result = estimate_route(build(), Pose(0.9, 0.5, 0.0), now, max_age_s=age)
    assert (result is not None) is accepted


@pytest.mark.parametrize('at_s', [-1.0, math.nan, math.inf, None, True])
def test_invalid_snapshot_time_is_rejected(at_s):
    assert build(at_s=at_s) is None


def test_cancellation_is_observed():
    cancel = Event()
    cancel.set()
    with pytest.raises(CancelledError):
        build_route_field(corridor(), Pose(0.9, 0.5, 0.0),
                          Pose(0.5, 0.5, 0.0), 10.0, cancel=cancel)


def test_cancellation_interrupts_a_traversal_already_in_progress(monkeypatch):
    cancel = Event()
    original = vuelta.neighbours
    expanded = []
    def neighbours(cell):
        expanded.append(cell)
        cancel.set()
        return original(cell)
    monkeypatch.setattr(vuelta, 'neighbours', neighbours)
    with pytest.raises(CancelledError):
        build_route_field(corridor(), Pose(0.9, 0.5, 0.0),
                          Pose(0.5, 0.5, 0.0), 10.0, cancel=cancel)
    assert len(expanded) == 1


def test_worker_result_cannot_mutate_route_fields():
    field = build()
    with pytest.raises(TypeError):
        field.home[(1, 1)] = 0


def test_missing_field_or_outside_destination_has_no_estimate():
    assert estimate_route(None, Pose(0.5, 0.5, 0.0), 10.0, max_age_s=2.5) is None
    assert estimate_route(build(), Pose(-1.0, -1.0, 0.0), 10.0, max_age_s=2.5) is None


def test_snapshot_is_independent_of_the_input_grid():
    grid = corridor()
    field = build(grid)
    grid.data = (100,) * (grid.width * grid.height)
    assert estimate_route(field, Pose(0.5, 1.5, 0.0), 11.0, max_age_s=2.5) is not None


def test_routes_support_rotated_and_shifted_map_origins():
    original = corridor()
    grid = Grid(original.width, original.height, original.resolution,
                original.data, Pose(4.0, -2.0, math.pi / 2))
    robot = Pose(*grid.center((18, 10)), 0.0)
    start = Pose(*grid.center((10, 10)), 0.0)
    goal = Pose(*grid.center((10, 30)), 0.0)
    result = estimate_route(build_route_field(grid, robot, start, 10.0),
                            goal, 11.0, max_age_s=2.5)
    assert result.outward_m == pytest.approx(5.6)
    assert result.home_m == pytest.approx(6.0)
