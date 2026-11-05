"""Independent accounting and geometry regressions for the execution audit."""
from dataclasses import replace
import math

import pytest

from equipo_jar.cobertura_plan import Camera, Grid, Pose, Settings
from ejecucion_cobertura import (
    GoalRecord, audit_records, predict_goals, prediction_trace, records_from_payload)
from metricas_cobertura import mission_indices


class DirectCosts:
    def between(self, a, b):
        return math.dist(a, b)


MISSION = dict(same_point_distance_m=0.02, same_point_angle_rad=0.1,
               turn_in_place_m=0.20, start_wait_s=4.0, settle_s=0.0)


def test_boxes_skip_selection_but_do_not_become_finished_identity_points():
    poses = (Pose(1, 1, 0), Pose(2, 1, 0))
    assert mission_indices(poses, 0.02, 0.1, zones=((1, 1, 0.5),)) == (1,)
    assert mission_indices(poses, 0.02, 0.1) == (0, 1)


def test_commanded_xy_matches_missions_turn_and_remembers_the_planned_pose():
    grid = Grid(30, 30, 0.1, [0] * 900)
    start = Pose(1.05, 1.05, 0)
    poses = (Pose(1.15, 1.05, 0), Pose(1.15, 1.05, math.pi / 2))
    result = predict_goals(grid, poses, start, DirectCosts(), Settings(), MISSION)
    assert result.shifted == (0, 1)
    assert [r.index for r in result.records] == [0, 1]
    assert all(r.pose.x == start.x for r in result.records)


def test_settle_delays_the_next_trip_and_is_not_planners_fictional_dwell():
    grid = Grid(20, 20, 1.0, [0] * 400)
    poses = (Pose(2.5, 5.5, 0), Pose(3.5, 5.5, 0))
    settings = replace(Settings(), travel_speed=0.5, observation_time=10.0)
    result = predict_goals(grid, poses, Pose(1.5, 5.5, 0), DirectCosts(), settings,
                           {**MISSION, 'settle_s': 1.0}, planning_s=5.0)
    assert [r.elapsed for r in result.records] == [7.0, 10.0]
    assert result.startup == 5.0
    trace = prediction_trace(grid, Pose(1.5, 5.5, 0), result, ((4, 5),), settings, Camera(),
                              DirectCosts())
    assert trace.duration == 10.0
    assert trace.distance == 2.0
    assert [event.elapsed for event in trace.events] == [7.0, 10.0]


def test_fixed_zones_and_identity_skips_are_reported_separately():
    grid = Grid(20, 20, 1.0, [0] * 400)
    poses = (Pose(2.5, 5.5, 0), Pose(2.505, 5.5, 0), Pose(3.5, 5.5, 0))
    result = predict_goals(grid, poses, Pose(1.5, 5.5, 0), DirectCosts(), Settings(),
                           MISSION, zones=((3.5, 5.5, 0.4),))
    assert result.identity_skipped == (1,)
    assert result.box_skipped == (2,)
    assert [r.index for r in result.records] == [0]


def test_failed_and_canceled_trips_cost_time_but_do_not_credit_visibility():
    grid = Grid(20, 20, 1.0, [0] * 400)
    cells = ((4, 5),)
    records = (GoalRecord(0, 3.0, Pose(2.5, 5.5, 0), 'failed'),
               GoalRecord(1, 7.0, Pose(2.5, 5.5, 0), 'canceled'))
    report = audit_records(grid, records, cells, Settings(), Camera(), (4.0, 10.0))
    assert report['nominal'] == report['sampled_robust'] == 0
    assert report['duration'] == 7.0
    assert report['failed'] == report['canceled'] == 1
    assert report['budgets']['10.0']['attempts'] == 2
    assert report['budgets']['10.0']['succeeded'] == 0


def test_arrival_view_is_not_credited_before_its_timestamp():
    grid = Grid(20, 20, 1.0, [0] * 400)
    report = audit_records(grid, (GoalRecord(0, 5, Pose(2.5, 5.5, 0)),),
                           ((4, 5),), Settings(), Camera(), (4, 5))
    assert report['budgets']['4.0']['nominal'] == 0
    assert report['budgets']['5.0']['nominal'] == 1


def test_heading_edge_is_visible_nominally_but_lost_under_rotation_error():
    grid = Grid(100, 100, 0.1, [0] * 10000)
    pose = Pose(5.05, 5.05, 0)
    # This target is near the upper edge of the physical 61-degree camera.
    cells = ((65, 58),)
    report = audit_records(grid, (GoalRecord(0, 1, pose),), cells, Settings(), Camera(), (1,),
                           offsets=((0, 0, 0), (0, 0, math.radians(-10))))
    assert report['nominal'] == 1
    assert report['sampled_robust'] == 0


def test_invalid_localization_cell_gets_no_view_and_is_reported():
    grid = Grid(3, 3, 1.0, [100] + [0] * 8)
    report = audit_records(grid, (GoalRecord(0, 1, Pose(0.5, 0.5, 0)),),
                           ((1, 0),), Settings(), Camera(), (1,))
    assert report['nominal'] == 0
    assert report['invalid_pose_indices'] == [0]


def test_empty_execution_is_not_a_successful_observation():
    grid = Grid(3, 3, 1.0, [0] * 9)
    report = audit_records(grid, (), ((1, 1),), Settings(), Camera(), (300,))
    assert report['duration'] is None
    assert report['nominal'] == report['succeeded'] == 0


@pytest.mark.parametrize('status', ['unknown', 'arrived', ''])
def test_unknown_results_are_not_silently_treated_as_success(status):
    with pytest.raises(ValueError):
        GoalRecord(0, 0, Pose(0, 0, 0), status)


def test_unsorted_clock_values_are_rejected():
    grid = Grid(3, 3, 1.0, [0] * 9)
    with pytest.raises(ValueError, match='ordered'):
        audit_records(grid, (GoalRecord(0, 3, Pose(1.5, 1.5, 0)),
                             GoalRecord(1, 2, Pose(1.5, 1.5, 0))),
                       ((2, 1),), Settings(), Camera(), (3,))


def external_records():
    return dict(frame_id='map', time_origin='/map', clock='simulated',
                pose_source='localized', source='test recording',
                goals=[dict(index=0, elapsed=5.0, pose=dict(x=1.5, y=1.5, yaw=0.0),
                            status='canceled')])


@pytest.mark.parametrize('field', ['frame_id', 'time_origin', 'clock', 'pose_source', 'source'])
def test_records_without_a_clock_or_pose_provenance_are_rejected(field):
    payload = external_records()
    del payload[field]
    with pytest.raises(ValueError):
        records_from_payload(payload)


def test_external_record_never_defaults_a_missing_status_to_success():
    payload = external_records()
    del payload['goals'][0]['status']
    with pytest.raises(ValueError):
        records_from_payload(payload)


def test_canceled_record_retains_its_explicit_pose_provenance_and_status():
    payload = external_records()
    record, = records_from_payload(payload)
    assert record.status == 'canceled'
    assert record.elapsed == 5.0
    assert record.pose == Pose(1.5, 1.5, 0.0)


def test_missing_perturbation_samples_are_not_full_robust_coverage():
    grid = Grid(3, 3, 1.0, [0] * 9)
    with pytest.raises(ValueError):
        audit_records(grid, (), ((1, 1),), Settings(), Camera(), (1,), offsets=())


def test_canceled_after_recoveries_is_not_a_view_and_unknown_is_not_zero():
    grid = Grid(20, 20, 1.0, [0] * 400)
    records = (GoalRecord(0, 3, Pose(2.5, 5.5, 0), 'failed'),
               GoalRecord(1, 7, Pose(2.5, 5.5, 0), 'canceled', recoveries=10))
    report = audit_records(grid, records, ((4, 5),), Settings(), Camera(), (10,))
    assert report['nominal'] == 0
    assert report['duration'] == 7
    assert report['recovery_counts_known'] == 1
    assert report['max_recoveries_known'] == 10
    assert report['recoveries_sum_known'] == 10


def test_absent_recovery_counts_remain_unknown_in_the_report():
    payload = external_records()
    record, = records_from_payload(payload)
    assert record.recoveries is None
    payload['goals'][0]['recoveries'] = 10
    record, = records_from_payload(payload)
    assert record.recoveries == 10
