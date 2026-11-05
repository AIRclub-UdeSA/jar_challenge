"""Separate observation stand-off from transit on generated maps."""
from concurrent.futures import CancelledError
from dataclasses import replace
import math
from threading import Event

import pytest

from equipo_jar.cobertura_plan import (
    Grid, Pose, Settings, make_plan, observation_cells, reachable_cells)
from test_cobertura_plan import room


def chambers():
    width, height = 40, 20
    data = [100 if (i in (0, width - 1) or j in (0, height - 1)
                    or (15 <= i <= 24 and not 8 <= j <= 12)) else 0
            for j in range(height) for i in range(width)]
    return Grid(width, height, 0.1, data)


def test_safe_observations_keep_access_through_a_narrow_transit_corridor():
    grid = chambers()
    start = Pose(*grid.center((5, 10)), 0.0)
    settings = replace(Settings(), view_clearance=0.30, view_range=0.8)
    transit = grid.safe_cells(0.21, settings.free_threshold)
    reachable = reachable_cells(transit, grid.cell(start.x, start.y))
    stops = observation_cells(grid, reachable, settings)
    inflated_transit = reachable_cells(grid.safe_cells(0.30, settings.free_threshold), (5, 10))
    assert (19, 10) in reachable and (19, 10) not in stops
    assert any(i >= 25 for i, j in stops)
    assert not any(i >= 25 for i, j in inflated_transit)
    plan = make_plan(grid, start, settings, diagnostics=True)
    assert plan.reachable_count == len(reachable)
    assert any(grid.cell(p.x, p.y)[0] >= 25 for p in plan.poses)
    assert all(grid.cell(p.x, p.y) in stops for p in plan.poses)
    assert set(plan.diagnostics.positions) <= stops


def test_transit_safe_start_is_not_forced_into_the_observation_plan():
    grid = room(width=22, height=20)
    cell = (3, 9)
    settings = replace(Settings(), view_clearance=0.30)
    assert cell in grid.safe_cells(0.21, 25)
    assert cell not in grid.safe_cells(0.30, 25)
    start = Pose(*grid.center(cell), 0.0)
    plan = make_plan(grid, start, settings, diagnostics=True)
    assert plan.poses
    assert cell not in plan.diagnostics.positions
    assert all(grid.cell(p.x, p.y) != cell for p in plan.poses)


def test_fine_recovery_obeys_the_same_observation_clearance():
    grid = room(width=35, height=30, blocked=[(15, j) for j in range(1, 17)])
    start = Pose(*grid.center((5, 5)), 0.0)
    settings = replace(Settings(), candidate_spacing=1.2, view_range=0.6, view_clearance=0.30)
    plan = make_plan(grid, start, settings, diagnostics=True)
    assert plan.diagnostics.added_positions > 0
    safe_stops = grid.safe_cells(0.30, settings.free_threshold)
    assert set(plan.diagnostics.positions) <= safe_stops
    assert all(grid.cell(p.x, p.y) in safe_stops for p in plan.poses)


def test_a_corridor_without_safe_stops_does_not_emit_unsafe_goals():
    grid = room(width=30, height=7)
    start = Pose(*grid.center((5, 3)), 0.0)
    plan = make_plan(grid, start, replace(Settings(), view_clearance=0.30), diagnostics=True)
    assert plan.reachable_count > 0
    assert plan.target_count > 0
    assert plan.poses == ()
    assert plan.predicted_count == plan.candidate_count == 0
    assert plan.metrics.duration == 0.0


def test_observation_clearance_cannot_weaken_transit_clearance():
    grid = room(width=20, height=20)
    settings = replace(Settings(), robot_radius=0.25, safety_margin=0.05,
                       view_clearance=0.1, view_preferred_clearance=0.1)
    reachable = reachable_cells(grid.safe_cells(0.30, 25), (5, 5))
    assert observation_cells(grid, reachable, settings) == reachable
    assert observation_cells(grid, reachable, settings, preferred=True) == reachable


def test_zero_extra_clearance_preserves_the_original_candidate_mask():
    grid = room(width=20, height=20)
    settings = replace(Settings(), view_clearance=0.0, view_preferred_clearance=0.0)
    reachable = reachable_cells(grid.safe_cells(0.21, 25), (5, 5))
    assert observation_cells(grid, reachable, settings) == reachable


@pytest.mark.parametrize('value', [-0.1, math.nan, math.inf])
@pytest.mark.parametrize('field', ['view_clearance', 'view_preferred_clearance'])
def test_invalid_observation_clearance_is_rejected(value, field):
    with pytest.raises(ValueError, match=field):
        replace(Settings(), **{field: value})


def test_observation_geometry_can_be_cancelled():
    cancel = Event()
    cancel.set()
    grid = room(width=20, height=20)
    with pytest.raises(CancelledError):
        observation_cells(grid, {(5, 5)}, Settings(), cancel)


def test_translated_rotated_map_keeps_observation_safety():
    grid = room(width=24, height=22, origin=Pose(-1.7, 0.8, 0.63))
    settings = replace(Settings(), view_clearance=0.30)
    plan = make_plan(grid, Pose(*grid.center((5, 5)), 0.63), settings)
    stops = grid.safe_cells(0.30, settings.free_threshold)
    assert plan.poses
    assert all(grid.cell(p.x, p.y) in stops for p in plan.poses)


def test_fallback_views_are_used_when_only_the_hard_clearance_fits():
    grid = room(width=30, height=7)
    settings = Settings()
    start = Pose(*grid.center((5, 3)), 0.0)
    plan = make_plan(grid, start, settings, diagnostics=True)
    hard = grid.safe_cells(settings.view_clearance, settings.free_threshold)
    preferred = grid.safe_cells(settings.view_preferred_clearance, settings.free_threshold)
    assert not preferred
    assert plan.poses
    assert plan.diagnostics.added_positions > 0
    assert all(grid.cell(p.x, p.y) in hard for p in plan.poses)


def test_open_room_prefers_the_larger_clearance_without_fallback():
    grid = room(width=20, height=20)
    settings = Settings()
    plan = make_plan(grid, Pose(*grid.center((5, 5)), 0.0), settings, diagnostics=True)
    preferred = grid.safe_cells(settings.view_preferred_clearance, settings.free_threshold)
    assert plan.poses
    assert all(grid.cell(p.x, p.y) in preferred for p in plan.poses)
