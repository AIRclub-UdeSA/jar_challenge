"""Independent accounting cases for the offline coverage evaluator."""
from dataclasses import replace
import math

import pytest

from equipo_jar.cobertura_plan import Camera, Grid, Pose, Settings, reachable_cells
from metricas_cobertura import (
    at_budget, classify_outside, mask_for_cells, mission_indices, route_trace,
    summarize_trace)


class UnitGrid:
    def cell(self, x, y):
        return x, y


class StraightCosts:
    def between(self, first, last):
        return math.dist(first, last)


def trace(indices=(0, 1), masks=(1, 2), startup=2.0):
    settings = replace(Settings(), travel_speed=1.0, turn_speed=1.0, observation_time=1.0)
    poses = (Pose(1.0, 0.0, 0.0), Pose(2.0, 0.0, 0.0))
    result = route_trace(UnitGrid(), Pose(0.0, 0.0, 0.0), poses, masks, indices,
                         StraightCosts(), settings, startup)
    return result, settings


def test_credit_only_after_arrival_and_observation_time():
    route, _ = trace()
    assert [event.elapsed for event in route.events] == [4.0, 6.0]
    assert at_budget(route, 3.99, target_mask=3, travel_speed=1)['visible'] == 0
    assert at_budget(route, 4.0, target_mask=3, travel_speed=1)['visible'] == 1
    assert at_budget(route, 6.0, target_mask=3, travel_speed=1)['visible'] == 2


def test_integral_counts_only_time_after_each_view():
    route, _ = trace()
    value = at_budget(route, 8.0, target_mask=3, travel_speed=1)
    # 0 cells for 4 s, 1 cell for 2 s, 2 cells for 2 s; 6/(2*8).
    assert value['mean_fraction'] == pytest.approx(0.375)
    assert value['return_from_last_completed_s'] == 2.0


def test_repeated_visibility_is_not_added_twice():
    route, _ = trace(masks=(1, 1))
    assert at_budget(route, 8.0, target_mask=3, travel_speed=1)['visible'] == 1


def test_filtered_denominator_excludes_disconnected_targets():
    route, _ = trace()
    value = at_budget(route, 8.0, target_mask=1, travel_speed=1)
    assert value['visible'] == value['targets'] == 1
    assert value['fraction'] == 1.0
    assert value['mean_fraction'] == 0.5


def test_empty_denominator_is_not_reported_as_full_coverage():
    route, _ = trace()
    value = at_budget(route, 8.0, target_mask=0, travel_speed=1)
    assert value['fraction'] is None
    assert value['mean_fraction'] is None


def test_skipped_goal_changes_travel_and_visibility():
    route, _ = trace(indices=(1,))
    assert route.distance == 2.0
    assert route.events[0].elapsed == 5.0
    assert route.events[0].mask == 2


def test_empty_route_has_no_observations_or_return_distance():
    route, _ = trace(indices=())
    assert route.duration == 2.0
    value = at_budget(route, 8.0, target_mask=3, travel_speed=1)
    assert value['visible'] == value['completed'] == 0
    assert value['return_from_last_completed_s'] == 0.0


def test_turn_cost_uses_the_short_angle_across_pi():
    settings = replace(Settings(), travel_speed=1.0, turn_speed=1.0)
    route = route_trace(UnitGrid(), Pose(0, 0, math.pi - 0.1),
                        (Pose(0, 0, -math.pi + 0.1),), (1,), (0,),
                        StraightCosts(), settings)
    assert route.turning == pytest.approx(0.2)
    assert route.duration == pytest.approx(1.2)


def test_summary_retains_goal_indices_and_all_denominators():
    route, settings = trace()
    result = summarize_trace(route, (4.0, 8.0), {'all': 3, 'connected': 1}, settings)
    assert result['final_visible'] == {'all': 2, 'connected': 1}
    assert result['budgets']['4.0']['all']['visible'] == 1
    assert [event['index'] for event in result['curve']] == [0, 1]


@pytest.mark.parametrize('budget', [0.0, -1.0, math.nan, math.inf])
def test_invalid_time_budget_is_rejected(budget):
    route, _ = trace()
    with pytest.raises(ValueError):
        at_budget(route, budget, target_mask=3, travel_speed=1)


@pytest.mark.parametrize('delay', [-1.0, math.nan, math.inf])
def test_invalid_startup_delay_is_rejected(delay):
    with pytest.raises(ValueError):
        trace(startup=delay)


def test_mission_drops_nearby_same_heading_but_keeps_another_heading():
    poses = (Pose(0, 0, 0), Pose(0.2, 0, 0), Pose(0, 0, math.pi / 2), Pose(0.3, 0, 0))
    assert mission_indices(poses, 0.3, 0.5) == (0, 2, 3)


def test_mission_handles_angle_wrap_and_empty_plan():
    assert mission_indices((), 0.3, 0.5) == ()
    poses = (Pose(0, 0, math.pi - 0.1), Pose(0, 0, -math.pi + 0.1))
    assert mission_indices(poses, 0.3, 0.5) == (0,)


def test_cell_mask_preserves_the_dense_indexing():
    assert mask_for_cells(((0, 0), (1, 0), (2, 0)), {(0, 0), (2, 0)}) == 5


def test_oracle_distinguishes_disconnection_sampling_and_model_limits():
    grid = Grid(7, 3, 1.0, [100 if i == 3 else 0 for j in range(3) for i in range(7)])
    settings = replace(Settings(), robot_radius=1.0, safety_margin=0.0, view_range=1.1)
    free = {(i, j) for j in range(3) for i in range(7) if grid.free((i, j), 25)}
    raw = reachable_cells(free, (1, 1))
    safe = reachable_cells(grid.safe_cells(1.0, 25), (1, 1))
    assert safe == {(1, 1)}
    groups, witnesses = classify_outside(
        grid, ((0, 1), (0, 0), (5, 1)), raw, safe, settings, Camera())
    assert groups == dict(raw_disconnected=[(5, 1)],
                          candidate_sampling_gap=[(0, 1)], no_witness=[(0, 0)])
    assert len(witnesses) == 1


def test_oracle_returns_empty_groups_for_no_missing_targets():
    grid = Grid(3, 3, 1.0, [0] * 9)
    groups, witnesses = classify_outside(grid, (), set(), set(), Settings(), Camera())
    assert not any(groups.values())
    assert witnesses == ()
