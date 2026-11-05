"""Independent geometry/cache checks on generated maps outside src."""
from concurrent.futures import CancelledError, ThreadPoolExecutor
import math
import random

import pytest

from equipo_jar.cobertura_plan import Grid, Pose


def disk_oracle(grid, radius, threshold):
    """Compare each disk directly with every occupied square and the map edge."""
    blocked = [(i, j) for j in range(grid.height) for i in range(grid.width)
               if not 0 <= grid.data[j * grid.width + i] <= threshold]
    clearance = radius / grid.resolution
    result = set()
    for j in range(grid.height):
        for i in range(grid.width):
            edge = min(i + .5, j + .5, grid.width - i - .5, grid.height - j - .5)
            if (edge <= clearance + 1e-12
                    or math.isclose(edge, clearance, abs_tol=1e-12)):
                continue
            for wi, wj in blocked:
                distance = math.hypot(max(wi - i - .5, i - wi - .5, 0),
                                      max(wj - j - .5, j - wj - .5, 0))
                if distance <= clearance or math.isclose(distance, clearance, abs_tol=1e-12):
                    break
            else:
                result.add((i, j))
    return result


def test_cached_inflation_matches_independent_disks_at_cell_boundaries():
    rng = random.Random(261026)
    for _ in range(12):
        width, height = rng.randrange(2, 12), rng.randrange(2, 10)
        grid = Grid(width, height, .1,
                    rng.choices((0, 10, 50, 100, -1), weights=(30, 2, 1, 1, 1), k=width * height))
        for threshold in (25, 60, 0, 99, 25):
            for radius in (0.0, .01, .05 - 1e-11, .05 - 1e-13, .05, .05 + 1e-13,
                           .10, .15, .24, .30, .7):
                assert grid.safe_cells(radius, threshold) == disk_oracle(grid, radius, threshold)


def test_threshold_changes_apply_to_both_sight_and_inflation():
    values = [0] * 35
    values[2 * 7 + 3] = 50
    values[1 * 7 + 3] = -1
    grid = Grid(7, 5, .1, values, Pose(-.3, .2, .4))
    for threshold in (25, 60, 0, 99, 25):
        assert grid.visible(grid.center((1, 2)), grid.center((5, 2)), threshold) == (threshold >= 50)
        assert not grid.visible(grid.center((1, 1)), grid.center((5, 1)), threshold)
        assert ((3, 2) in grid.safe_cells(0.0, threshold)) == (threshold >= 50)
        assert (3, 1) not in grid.safe_cells(0.0, threshold)


def test_parallel_readers_with_different_thresholds_keep_their_own_mask():
    values = [0] * (101 * 73)
    for j in range(73):
        values[j * 101 + 50] = 50
    grid = Grid(101, 73, .1, values)
    def query(threshold):
        return grid.visible(grid.center((5, 30)), grid.center((95, 30)), threshold)
    thresholds = [25, 60, 0, 99] * 24
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(query, thresholds)) == [value >= 50 for value in thresholds]


def test_cancelled_lookup_build_can_be_retried_without_a_partial_mask():
    class StopDuringBuild:
        calls = 0
        def is_set(self):
            self.calls += 1
            return self.calls >= 4
    grid = Grid(20, 20, .1, [0] * 400)
    with pytest.raises(CancelledError):
        grid.safe_cells(.24, 25, StopDuringBuild())
    assert grid.safe_cells(.24, 25) == disk_oracle(grid, .24, 25)
