"""Closed-segment visibility against an independent square-intersection oracle."""
import math
import random

import pytest

from equipo_jar.mapa import Grid, Pose


def world(grid, point):
    x, y = (value * grid.resolution for value in point)
    return (grid.origin.x + grid.cos * x - grid.sin * y,
            grid.origin.y + grid.sin * x + grid.cos * y)


def square_oracle(width, height, blocked, a, b):
    """Segment/slab intersection, in cell units, no grid traversal or floor target."""
    if any(not (0 < p[0] < width and 0 < p[1] < height) for p in (a, b)):
        return False
    for i, j in blocked:
        enter, leave = 0.0, 1.0
        for axis, low in ((0, i), (1, j)):
            delta = b[axis] - a[axis]
            if delta == 0:
                if a[axis] < low - 1e-12 or a[axis] > low + 1 + 1e-12:
                    enter, leave = 1.0, 0.0
                    break
            else:
                lo, hi = sorted(((low - a[axis]) / delta, (low + 1 - a[axis]) / delta))
                enter, leave = max(enter, lo), min(leave, hi)
        if enter <= leave + 1e-12:
            return False
    return True


@pytest.mark.parametrize('yaw', [0.0, 0.37, math.pi / 2, -math.pi / 2])
def test_empty_map_boundary_endpoint_is_visible_in_both_directions(yaw):
    grid = Grid(70, 50, 0.05, [0] * 3500, Pose(-1.3, 2.1, yaw))
    for a, b in (((20.5, 27.5), (44, 22)), ((44, 22), (44, 22)),
                 ((20, 27.5), (44, 22)), ((44.5, 18.5), (44, 22))):
        assert grid.visible(world(grid, a), world(grid, b), 25)
        assert grid.visible(world(grid, b), world(grid, a), 25)


@pytest.mark.parametrize('obstacle', [100, -1, 26])
@pytest.mark.parametrize('yaw', [0.0, 0.37])
def test_endpoint_corner_and_edge_touches_are_blocked(obstacle, yaw):
    data = [0] * 400
    data[10 * 20 + 10] = obstacle
    grid = Grid(20, 20, 0.05, data, Pose(0.7, -0.2, yaw))
    for a, b in (((5, 10.5), (10, 10.5)), ((5, 5), (10, 10)),
                 ((10, 10), (10, 10)), ((11, 11), (15, 15))):
        assert not grid.visible(world(grid, a), world(grid, b), 25)
        assert not grid.visible(world(grid, b), world(grid, a), 25)


@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('yaw', [0.0, 0.37, -math.pi / 2])
def test_line_along_wall_boundary_checks_both_strips(axis, yaw):
    data = [0] * 400
    data[10 * 20 + 9] = 100
    grid = Grid(20, 20, 0.05, data, Pose(-0.4, 0.3, yaw))
    a, b = ((10, 4), (10, 16)) if axis == 0 else ((4, 10), (16, 10))
    assert not grid.visible(world(grid, a), world(grid, b), 25)
    assert not grid.visible(world(grid, b), world(grid, a), 25)


def test_near_boundary_clear_ray_is_not_rounded_onto_wall():
    data = [0] * 400
    data[10 * 20 + 9] = 100
    grid = Grid(20, 20, 0.05, data)
    assert grid.visible(world(grid, (10 + 1e-8, 4)), world(grid, (10 + 1e-8, 16)), 25)


def test_near_corner_tie_does_not_step_past_a_free_endpoint():
    grid = Grid(20, 20, 1.0, [0] * 400)
    for epsilon in (1e-8, 1e-10, 5e-12, 2e-12, 1e-12, 1e-13):
        a, b = (0.5, 0.5), (10 + epsilon, 10 - epsilon)
        assert grid.visible(a, b, 25)
        assert grid.visible(b, a, 25)


def test_threshold_mask_and_outside_or_nonfinite_endpoints():
    data = [0] * 400
    data[10 * 20 + 10] = 50
    grid = Grid(20, 20, 0.05, data)
    a, b = world(grid, (5.5, 10.5)), world(grid, (15.5, 10.5))
    for threshold in (25, 60, 25):
        assert grid.visible(a, b, threshold) == (threshold == 60)
    for bad in ((math.nan, 0.5), (math.inf, 0.5), (-0.1, 0.5), (1.1, 0.5), (0.0, 0.5)):
        assert not grid.visible(a, bad, 25)
        assert not grid.visible(bad, a, 25)


def test_rotated_random_rays_match_closed_square_oracle():
    rng = random.Random(20261030)
    for _ in range(12):
        width, height = 14, 12
        data = rng.choices((0, 25, 26, 100, -1), weights=(30, 2, 1, 1, 1), k=width * height)
        blocked = [(i, j) for j in range(height) for i in range(width)
                   if not 0 <= data[j * width + i] <= 25]
        grids = [Grid(width, height, 0.05, data, Pose(-1.3, 0.7, yaw))
                 for yaw in (0.0, 0.37, math.pi / 2, -math.pi / 2)]
        for n in range(160):
            def point():
                return tuple(rng.choice((rng.randrange(1, limit), rng.randrange(1, limit) + 0.5,
                                         rng.uniform(0.1, limit - 0.1))) for limit in (width, height))
            a, b = point(), point()
            if n % 8 == 0:
                b = a  # Zero-length rays still touch all endpoint neighbours.
            expected = square_oracle(width, height, blocked, a, b)
            for grid in grids:
                wa, wb = world(grid, a), world(grid, b)
                assert grid.visible(wa, wb, 25) == expected, (grid.origin, a, b)
                assert grid.visible(wb, wa, 25) == expected, (grid.origin, b, a)


def test_all_half_cell_pairs_on_small_maps_match_square_oracle():
    points = [(x / 2, y / 2) for x in range(1, 8) for y in range(1, 8)]
    for occupied in (((1, 1),), ((1, 2), (2, 1)), ((0, 2), (2, 2), (3, 1))):
        data = [0] * 16
        for i, j in occupied:
            data[j * 4 + i] = -1 if (i + j) % 2 else 100
        grids = [Grid(4, 4, resolution, data, Pose(-0.7, 0.2, yaw))
                 for resolution, yaw in ((0.05, 0.0), (0.2, 0.37))]
        for a in points:
            for b in points:
                expected = square_oracle(4, 4, occupied, a, b)
                for grid in grids:
                    assert grid.visible(world(grid, a), world(grid, b), 25) == expected, (a, b, occupied)
