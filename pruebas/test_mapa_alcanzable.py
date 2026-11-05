"""Shared reachability on generated maps, without ROS or practice geometry."""
from concurrent.futures import CancelledError
import math
from threading import Event
from types import SimpleNamespace

import pytest

from equipo_jar.mapa import Grid, Pose, reachable_cells


def doorway(yaw=0.0, door_value=0):
    width, height, resolution = 40, 30, 0.05
    data = [100 if i == 20 and j != 15 else 0
            for j in range(height) for i in range(width)]
    data[15 * width + 20] = door_value
    origin = SimpleNamespace(
        position=SimpleNamespace(x=-1.0, y=0.4),
        orientation=SimpleNamespace(x=0.0, y=0.0,
                                    z=math.sin(yaw / 2), w=math.cos(yaw / 2)))
    return SimpleNamespace(header=SimpleNamespace(frame_id='map'),
                           info=SimpleNamespace(width=width, height=height,
                                                resolution=resolution, origin=origin),
                           data=data)


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, -math.pi / 2, 0.37])
def test_point_connection_is_not_a_robot_sized_connection(yaw):
    grid = Grid.from_occupancy_grid(doorway(yaw))
    start = grid.center((10, 15))
    other_room = grid.cell(*grid.center((30, 15)))
    assert other_room in grid.reachable_from(*start, radius=0.0, threshold=25)
    for radius in (0.05, 0.12, 0.21, 0.25):
        reached = grid.reachable_from(*start, radius=radius, threshold=25)
        assert (10, 15) in reached
        assert other_room not in reached


@pytest.mark.parametrize('door_value,threshold,connected', [
    (-1, 99, False), (100, 99, False), (26, 25, False),
    (25, 25, True), (26, 26, True), (0, 0, True),
])
def test_unknown_obstacles_and_the_callers_threshold(door_value, threshold, connected):
    grid = Grid.from_occupancy_grid(doorway(door_value=door_value))
    reached = grid.reachable_from(*grid.center((10, 15)), radius=0.0, threshold=threshold)
    assert ((30, 15) in reached) is connected


def test_invalid_start_does_not_snap_across_a_wall():
    grid = Grid.from_occupancy_grid(doorway())
    for start in (grid.center((20, 14)), grid.center((0, 15)),
                  (-100.0, -100.0), (math.nan, 0.0), (math.inf, 0.0)):
        with pytest.raises(ValueError, match='safe free space'):
            grid.reachable_from(*start, radius=0.12, threshold=25)


def test_diagonal_contact_does_not_connect_rooms_and_input_is_preserved():
    safe = frozenset({(0, 0), (1, 1), (2, 1)})
    assert reachable_cells(safe, (0, 0)) == {(0, 0)}
    assert safe == {(0, 0), (1, 1), (2, 1)}


def test_map_snapshot_is_not_changed_by_a_later_message_update():
    message = doorway()
    grid = Grid.from_occupancy_grid(message)
    original = grid.center((10, 15))
    message.data[:] = [100] * len(message.data)
    message.info.origin.position.x += 5.0
    assert (30, 15) in grid.reachable_from(*original, radius=0.0, threshold=25)
    assert grid.center((10, 15)) == original


@pytest.mark.parametrize('field,value', [
    ('radius', -0.1), ('radius', math.nan), ('radius', math.inf),
    ('threshold', -1), ('threshold', 100), ('threshold', 25.5), ('threshold', True),
])
def test_bad_footprint_or_threshold_is_rejected(field, value):
    grid = Grid.from_occupancy_grid(doorway())
    arguments = dict(radius=0.12, threshold=25)
    arguments[field] = value
    with pytest.raises(ValueError):
        grid.reachable_from(*grid.center((10, 15)), **arguments)


@pytest.mark.parametrize('bad_rotation', [
    (0.0, 0.0, 0.0, 0.0), (0.1, 0.0, 0.0, math.sqrt(0.99)),
    (0.0, 0.0, math.nan, 1.0),
])
def test_bad_map_rotation_is_rejected(bad_rotation):
    message = doorway()
    q = message.info.origin.orientation
    q.x, q.y, q.z, q.w = bad_rotation
    with pytest.raises(ValueError):
        Grid.from_occupancy_grid(message)


def test_map_frame_must_match_robot_positions():
    message = doorway()
    message.header.frame_id = 'odom'
    with pytest.raises(ValueError, match='map frame'):
        Grid.from_occupancy_grid(message)


def test_connectivity_can_be_cancelled_during_a_large_component():
    class StopDuringFlood:
        calls = 0

        def is_set(self):
            self.calls += 1
            return self.calls >= 5

    safe = {(i, j) for j in range(30) for i in range(40)}
    with pytest.raises(CancelledError):
        reachable_cells(safe, (10, 15), StopDuringFlood())
    stop = Event()
    stop.set()
    grid = Grid.from_occupancy_grid(doorway())
    with pytest.raises(CancelledError):
        grid.reachable_from(*grid.center((10, 15)), radius=0.12, threshold=25, cancel=stop)


def test_existing_coverage_imports_refer_to_the_shared_implementation():
    from equipo_jar import cobertura_plan

    assert cobertura_plan.Grid is Grid
    assert cobertura_plan.Pose is Pose
    assert cobertura_plan.reachable_cells is reachable_cells
