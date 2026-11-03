"""Map-coordinate regressions; generated geometry stays outside src."""
import math
from types import SimpleNamespace

import pytest

from equipo_jar.inspeccion import viewpoint_for
from equipo_jar.mapa import Grid, cell_value, is_free, line_is_clear


def make_grid(yaw, width=6, height=4, resolution=0.5):
    origin = SimpleNamespace(
        position=SimpleNamespace(x=2.0, y=1.0),
        orientation=SimpleNamespace(
            x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
    )
    info = SimpleNamespace(
        origin=origin, width=width, height=height, resolution=resolution)
    return SimpleNamespace(header=SimpleNamespace(frame_id='map'),
                           info=info, data=list(range(width * height)))


def world_point(grid, local_x, local_y, yaw):
    origin = grid.info.origin.position
    return (origin.x + math.cos(yaw) * local_x - math.sin(yaw) * local_y,
            origin.y + math.sin(yaw) * local_x + math.cos(yaw) * local_y)


@pytest.mark.parametrize('yaw', [0.0, math.pi / 2, -math.pi / 2, math.pi, 0.37])
def test_each_cell_center_uses_translation_and_rotation(yaw):
    grid = make_grid(yaw)
    for j in range(grid.info.height):
        for i in range(grid.info.width):
            x, y = world_point(grid, (i + 0.5) * grid.info.resolution,
                               (j + 0.5) * grid.info.resolution, yaw)
            assert cell_value(grid, x, y) == j * grid.info.width + i


@pytest.mark.parametrize('yaw', [math.pi / 2, -math.pi / 2, 0.37])
def test_bounds_are_checked_in_the_grid_frame(yaw):
    grid = make_grid(yaw)
    width = grid.info.width * grid.info.resolution
    height = grid.info.height * grid.info.resolution
    epsilon = 1e-8
    for local_x, local_y in ((-epsilon, 0.25), (0.25, -epsilon),
                             (width + epsilon, 0.25), (0.25, height + epsilon)):
        assert cell_value(grid, *world_point(grid, local_x, local_y, yaw)) is None


def test_rotated_query_with_missing_data_returns_none():
    grid = make_grid(math.pi / 2)
    grid.data = []
    assert cell_value(grid, *world_point(grid, 0.25, 0.25, math.pi / 2)) is None


def test_rotated_query_preserves_unknown_value():
    grid = make_grid(math.pi / 2)
    grid.data[0] = -1
    assert cell_value(grid, *world_point(grid, 0.25, 0.25, math.pi / 2)) == -1


def test_simple_test_grids_without_orientation_still_work():
    grid = make_grid(0.0)
    del grid.info.origin.orientation
    assert cell_value(grid, *world_point(grid, 0.75, 0.25, 0.0)) == 1


def test_existing_clearance_and_sight_calls_use_the_rotated_wall():
    yaw = math.pi / 2
    grid = make_grid(yaw)
    grid.data = [0] * len(grid.data)
    for j in range(grid.info.height):
        grid.data[j * grid.info.width + 2] = 100
    before = world_point(grid, 0.25, 0.75, yaw)
    wall = world_point(grid, 1.25, 0.75, yaw)
    after = world_point(grid, 2.25, 0.75, yaw)
    assert is_free(grid, *before, 0.0)
    assert not is_free(grid, *wall, 0.0)
    assert not line_is_clear(grid, *before, *after)


def test_mission_inspection_finds_a_view_on_a_rotated_map():
    grid = make_grid(math.pi / 2, width=80, height=80, resolution=0.05)
    grid.data = [0] * len(grid.data)
    cell_grid = Grid.from_occupancy_grid(grid)
    reachable = cell_grid.reachable_from(0.0, 1.5, radius=0.21, threshold=25)
    view = viewpoint_for(
        grid, cx=0.0, cy=3.0, robot_x=0.0, robot_y=1.5,
        distance=0.8, clearance=0.25, n_angles=12, min_shift=0.4, box_margin=0.25,
        cell_grid=cell_grid, reachable=reachable)
    assert view is not None
    assert view == pytest.approx((0.0, 2.2, math.pi / 2))
