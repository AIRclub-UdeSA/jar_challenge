from concurrent.futures import Future
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Path
from std_msgs.msg import Bool
import pytest

from equipo_jar.cobertura_repaso import ExtraPlan, ExtraView
from equipo_jar.mapa import Pose
from test_cancelar_inspeccion_reportada import Navigation, candidate
from test_historial_mision import mission, instant, map_message


class Executor:
    def __init__(self):
        self.jobs = []

    def submit(self, fn, *args):
        future = Future()
        self.jobs.append((future, fn, args))
        return future

    def finish(self):
        future, fn, args = self.jobs.pop(0)
        if not future.cancelled():
            try:
                future.set_result(fn(*args))
            except BaseException as error:
                future.set_exception(error)

    def shutdown(self, **kwargs):
        pass


@pytest.fixture
def search(mission, monkeypatch):
    node, clock, transform = mission
    node.p['extra_search_enabled'] = True
    node.start_pose = (0.5, 0.5, 0.0)
    node.plan = Path()
    node.candidatas = []
    navigation = Navigation()
    monkeypatch.setattr(node, 'navegacion', navigation)
    optical = TransformStamped()
    optical.transform.translation.x = 0.1
    q = optical.transform.rotation
    q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    monkeypatch.setattr(node.tf_buffer, 'lookup_transform',
                        lambda target, source, time: optical if source.startswith('cam_') else transform)
    executor = Executor()
    node.extra.executor = executor
    node.extra.provider = lambda *args, **kwargs: ExtraPlan((ExtraView(Pose(2.0, 2.4, 0.0), 10),), 10)
    node.on_costmap(map_message())
    return node, clock, transform, executor, navigation


def advance(search, seconds=0.1):
    node, clock, transform, _, _ = search
    clock.seconds += seconds
    transform.header.stamp = instant(clock.seconds).to_msg()
    node.on_costmap(map_message())


def reach_extra_goal(search):
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert len(navigation.sent) == 1
    return navigation.handles[0]


def test_exhausted_plan_without_a_candidate_dispatches_an_extra_goal(search):
    node, _, _, _, navigation = search
    reach_extra_goal(search)
    assert node.state == 'exploring'
    assert node.extra.phase == 'extra'
    assert node.extra.accepted_count == 1
    assert navigation.sent[0].pose.pose.position.x == 2.0


def test_empty_provider_preserves_pending_but_returns(search):
    node, _, _, executor, navigation = search
    node.extra.provider = lambda *args, **kwargs: ExtraPlan((), 100)
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert node.state == 'returning'
    assert len(navigation.sent) == 1
    assert navigation.sent[0].behavior_tree == node.return_tree


def test_all_reports_skip_extra_navigation(search):
    node, _, _, _, navigation = search
    node.reported_points = [(float(i), 0.0) for i in range(node.cantidad_victimas)]
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_worker_timeout_is_bounded_and_late_result_cannot_dispatch(search):
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    advance(search, 6.0)
    node.step_logic()
    assert node.state == 'returning'
    executor.finish()
    node.step_logic()
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_goal_failure_is_recorded_and_does_not_credit_the_requested_pose(search):
    node, _, _, _, _ = search
    goal = reach_extra_goal(search)
    goal.finish(GoalStatus.STATUS_ABORTED)
    assert node.extra.failed_count == 1
    assert node.extra.attempts[-1].outcome == 'failed'
    assert all(view.pose != Pose(2.0, 2.4, 0.0) for view in node.search_views)


def test_reported_inspection_cancellation_is_not_a_blocked_view(search):
    node, _, _, _, navigation = search
    node.candidatas = [candidate()]
    node.current_inspection = (1, 2.0, 2.0, 2)
    node.inspection_spot = (2.8, 2.0)
    node.go_to(node.make_pose(2.8, 2.0, 0.0))
    node.step_logic()
    navigation.handles[0].finish(GoalStatus.STATUS_CANCELED)
    assert not node.extra.attempts


def test_disabling_during_extra_trip_waits_for_terminal_result_before_return(search):
    node, _, _, _, navigation = search
    goal = reach_extra_goal(search)
    node.p['extra_search_enabled'] = False
    node.step_logic()
    assert node.state == 'returning'
    assert goal.cancel_calls == 1
    assert len(navigation.sent) == 1
    goal.finish(GoalStatus.STATUS_CANCELED)
    node.step_logic()
    assert len(navigation.sent) == 2
    assert navigation.sent[-1].behavior_tree == node.return_tree


def test_costmap_clear_does_not_erase_optical_obstacle_memory(search):
    node, _, _, _, _ = search
    obstacle = map_message()
    obstacle.data[100] = 100
    node.on_costmap(obstacle)
    node.on_costmap(map_message())
    assert node.extra.live.data[100] == 0
    assert node.extra.optics.data[100] == 100


def test_map_change_invalidates_worker_and_history(search):
    node, _, _, executor, navigation = search
    node.step_logic()
    epoch = node.extra.epoch
    changed = map_message()
    changed.data[300] = 100
    node.al_recibir_mapa(changed)
    executor.finish()
    assert node.extra.epoch > epoch
    assert not node.search_views
    assert not navigation.sent


def test_arrival_dwells_then_replans_without_repeating_the_same_request(search):
    node, _, _, executor, navigation = search
    goal = reach_extra_goal(search)
    goal.finish(GoalStatus.STATUS_SUCCEEDED)
    node.step_logic()
    assert len(navigation.sent) == 1
    assert not executor.jobs
    advance(search, 2.1)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert len(navigation.sent) == 1
    assert node.extra.done_reason is not None


def test_rejected_extra_goals_are_bounded_and_recorded(search):
    node, _, _, executor, navigation = search
    node.p['extra_max_failed'] = 1
    original = navigation.send_goal_async
    def reject(goal, **kwargs):
        response = original(goal, **kwargs)
        response.result().accepted = False
        return response
    navigation.send_goal_async = reject
    reach_extra_goal(search)
    assert node.extra.accepted_count == 0
    assert node.extra.failed_count == 1
    assert node.extra.attempts[-1].outcome == 'rejected'
    node.step_logic()
    assert node.state == 'returning'


def test_missing_start_denies_extra_but_finishes_waiting(search):
    node, _, _, executor, navigation = search
    node.start_pose = None
    node.step_logic()
    advance(search, 1.1)
    node.step_logic()
    assert not executor.jobs
    advance(search, 5.1)
    node.step_logic()
    assert node.state == 'stopping'
    assert not navigation.sent


def test_stale_costmap_does_not_become_fresh_when_feature_is_enabled(mission):
    node, clock, transform = mission
    node.on_costmap(map_message())
    clock.seconds += 10
    transform.header.stamp = instant(clock.seconds).to_msg()
    node.p['extra_search_enabled'] = True
    node.extra.tick()
    assert not node.extra.ready()


def test_costmap_change_before_plan_result_requires_another_snapshot(search):
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    changed = map_message()
    changed.data[10] = 100
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert not navigation.sent
    advance(search, 1.3)
    node.on_costmap(changed)
    node.step_logic()
    assert executor.jobs


def test_zero_gain_does_not_dispatch(search):
    node, _, _, executor, navigation = search
    node.extra.provider = lambda *args, **kwargs: ExtraPlan((ExtraView(Pose(2.0, 2.4, 0.0), 0),), 10)
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_route_field_refreshes_during_extra_travel(search):
    node, _, _, executor, _ = search
    reach_extra_goal(search)
    advance(search, 1.5)
    node.step_logic()
    assert executor.jobs


def test_fresh_home_estimate_can_preempt_an_active_exploration_goal(search):
    node, clock, transform, executor, navigation = search
    node.p['finish_time_s'] = 60.0
    node.p['return_time_s'] = 50.0
    node.p['return_reserve_s'] = 45.0
    node.go_to(node.make_pose(2.0, 2.0, 0.0))
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert node.state == 'returning'
    assert navigation.handles[0].cancel_calls == 1
    assert len(navigation.sent) == 1


def test_fake_world_new_view_reports_normally_and_returns(search):
    node, _, _, _, navigation = search
    goal = reach_extra_goal(search)
    goal.finish(GoalStatus.STATUS_SUCCEEDED)
    node.cantidad_victimas = 1
    node.candidatas = [candidate()]
    node.step_logic()
    assert len(node.reported_points) == 1
    assert node.state == 'returning'
    assert len(navigation.sent) == 2
    assert navigation.sent[-1].behavior_tree == node.return_tree


@pytest.mark.parametrize(('key', 'value'), [('extra_max_goals', 0), ('extra_range_m', float('nan')),
                                          ('extra_costmap_max_age_s', -1.0),
                                          ('return_estimate_speed_mps', 0.0)])
def test_invalid_extra_parameters_preserve_primary_reporting_and_bound_extra(search, key, value):
    node, _, _, executor, navigation = search
    node.p[key] = value
    node.step_logic()
    advance(search, 1.1)
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_two_failed_route_builds_do_not_resubmit_at_ten_hertz(search):
    node, _, _, executor, _ = search
    changed = map_message()
    changed.data[2 + 20 * 2] = 100
    node.on_costmap(changed)
    node.step_logic()
    executor.finish()
    advance(search, 0.1)
    node.step_logic()
    assert not executor.jobs


def test_real_provider_dispatches_a_shadow_revealing_view(search):
    from equipo_jar.cobertura_repaso import propose_extra_views
    from equipo_jar.historial_vistas import ObservedView
    from test_repaso_cobertura import room, outside_history, sees
    from equipo_jar.cobertura_repaso import pending_samples, visibility_grid
    from equipo_jar.cobertura_plan import Camera
    node, _, transform, executor, navigation = search
    raw, live = room()
    def message(grid):
        msg = map_message()
        msg.info.width, msg.info.height = grid.width, grid.height
        msg.info.resolution = grid.resolution
        msg.data = list(grid.data)
        return msg
    node.al_recibir_mapa(message(raw))
    node.on_costmap(message(live))
    transform.transform.translation.x = 1.0
    transform.transform.translation.y = 2.0
    node.start_pose = (1.0, 2.0, 0.0)
    node.search_views = tuple(ObservedView(v.pose, float(i)) for i, v in enumerate(outside_history()))
    node.extra.provider = propose_extra_views
    node.step_logic()
    executor.finish()
    advance_time = 1.1
    search[1].seconds += advance_time
    transform.header.stamp = instant(search[1].seconds).to_msg()
    node.on_costmap(message(live))
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert len(navigation.sent) == 1
    assert navigation.sent[0].behavior_tree == ''
    pose = node.extra.requested
    visual = visibility_grid(raw, live)
    pending = pending_samples(raw, live, node.search_views, Camera(0.1, 0, 0), node.extra.settings())
    assert any(sees(visual, pose, target) for target in pending)


def test_blocked_home_during_refresh_stops_extra_without_repeating_routes(search):
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.finish()
    advance(search, 3.0)
    node.step_logic()
    changed = map_message()
    changed.data[2 + 20 * 2] = 100
    node.on_costmap(changed)
    executor.finish()
    advance(search, 1.3)
    node.on_costmap(changed)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert not executor.jobs
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_extra_goal_timeout_cancels_once_and_waits_for_result(search):
    node, _, _, _, navigation = search
    goal = reach_extra_goal(search)
    advance(search, 31.0)
    node.step_logic()
    node.step_logic()
    assert goal.cancel_calls == 1
    assert len(navigation.sent) == 1
    goal.finish(GoalStatus.STATUS_CANCELED)
    assert node.extra.attempts[-1].outcome == 'cancelled'


def test_stop_deadline_still_preempts_extra_navigation(search):
    node, clock, transform, executor, navigation = search
    goal = reach_extra_goal(search)
    advance(search, 570.0)
    node.step_logic()
    assert node.state == 'stopping'
    node.step_logic()
    assert goal.cancel_calls == 1
    assert len(navigation.sent) == 1
    advance(search, 20.0)
    node.step_logic()
    assert node.state == 'stopping'
    node.on_stop_confirmed(Bool(data=True))
    node.step_logic()
    node.step_logic()
    assert node.state == 'finished'
    assert len(navigation.sent) == 1


def test_default_disabled_feature_does_not_create_a_worker(mission):
    node, _, _ = mission
    node.step_logic()
    assert node.extra.executor is None
    assert node.extra.live is None


def test_inspection_discovered_at_last_extra_view_keeps_its_confirmation_wait(search):
    node, _, _, _, navigation = search
    node.p['extra_max_goals'] = 1
    goal = reach_extra_goal(search)
    goal.finish(GoalStatus.STATUS_SUCCEEDED)
    doubtful = candidate()
    doubtful.observaciones = 2
    node.candidatas = [doubtful]
    node.step_logic()
    assert node.current_inspection is not None
    inspection = navigation.handles[-1]
    advance(search, 2.1)
    node.step_logic()
    assert node.state == 'exploring'
    assert inspection.cancel_calls == 0
    inspection.finish(GoalStatus.STATUS_SUCCEEDED)
    node.step_logic()
    assert node.look_until is not None
    advance(search, 1.0)
    node.step_logic()
    assert node.current_inspection is not None


def test_delayed_extra_acceptance_waits_for_cancellation_before_return(search):
    node, _, _, _, navigation = search
    pending = Future()
    original = navigation.send_goal_async
    def delay_first(goal, **kwargs):
        response = original(goal, **kwargs)
        return pending if len(navigation.sent) == 1 else response
    navigation.send_goal_async = delay_first
    goal = reach_extra_goal(search)
    assert node.nav_state == 'sending'
    # Reach the return deadline while acceptance is still within its 5 s budget.
    node.p['return_time_s'] = node.seconds_since_t0() + 1.0
    advance(search, 1.1)
    node.step_logic()
    assert node.state == 'returning'
    assert len(navigation.sent) == 1
    pending.set_result(goal)
    node.step_logic()
    assert goal.cancel_calls == 1
    assert len(navigation.sent) == 1
    goal.finish(GoalStatus.STATUS_CANCELED)
    node.step_logic()
    assert len(navigation.sent) == 2
    assert navigation.sent[-1].behavior_tree == node.return_tree


@pytest.mark.parametrize('obstacle', ['costmap', 'candidate'])
def test_existing_guards_cancel_extra_goals_when_their_endpoint_becomes_unsafe(search, obstacle):
    node, _, _, _, navigation = search
    goal = reach_extra_goal(search)
    if obstacle == 'costmap':
        changed = map_message()
        i, j = node.grid.cell(2.0, 2.4)
        changed.data[j * 20 + i] = 100
        node.on_costmap(changed)
    else:
        node.candidatas = [candidate(x=2.0, y=2.4)]
    node.step_logic()
    node.step_logic()
    assert goal.cancel_calls == 1
    assert len(navigation.sent) == 1


def test_already_running_worker_completion_after_return_is_ignored(search):
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.jobs[0][0].set_running_or_notify_cancel()
    advance(search, 6.0)
    node.step_logic()
    return_goal = node.current_goal
    executor.finish()
    node.step_logic()
    assert node.current_goal is return_goal
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_worker_exception_returns_without_escaping_the_timer(search):
    node, _, _, executor, navigation = search
    def fail(*args, **kwargs):
        raise ValueError('unavailable geometry')
    node.extra.provider = fail
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_initially_unlocked_start_and_empty_route_can_start_extra_search(search):
    node, _, _, executor, navigation = search
    node.start_locked = False
    node.start_pose = None
    node.step_logic()
    advance(search, 1.1)
    node.step_logic()
    assert executor.jobs
    executor.finish()
    node.step_logic()
    assert node.start_locked
    assert node.start_pose == (1.2, 2.3, 0.0)
    assert len(navigation.sent) == 1
    assert navigation.sent[0].behavior_tree == ''


@pytest.mark.parametrize('cost', [1, 50, 98])
def test_equivalent_inflation_update_does_not_discard_a_useful_plan(search, cost):
    node, _, _, executor, navigation = search
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    changed = map_message()
    changed.data[0] = cost
    node.on_costmap(changed)
    executor.finish()
    node.step_logic()
    assert len(navigation.sent) == 1
    assert navigation.sent[0].behavior_tree == ''
    field_stamp = node.extra.field.at_s
    advance(search, 2.6)
    node.on_costmap(changed)
    node.step_logic()
    assert node.extra.field.at_s == field_stamp
    assert node.extra.estimate(Pose(2.0, 2.4, 0.0)) is None


@pytest.mark.parametrize(('key', 'value'), [('extra_history_max', 0),
                                          ('extra_history_interval_s', float('nan')),
                                          ('extra_history_shift_m', -1.0),
                                          ('extra_history_turn_rad', float('inf')),
                                          ('extra_pose_max_age_s', -1.0)])
def test_invalid_history_settings_deny_supplementary_dispatch(search, key, value):
    node, _, _, executor, navigation = search
    node.p[key] = value
    node.step_logic()
    if executor.jobs:
        executor.finish()
    advance(search, 1.1)
    node.step_logic()
    if executor.jobs:
        executor.finish()
    node.step_logic()
    assert node.state == 'returning'
    assert all(goal.behavior_tree == node.return_tree for goal in navigation.sent)


def test_new_optical_obstacle_invalidates_views_waiting_for_a_route_refresh(search):
    node, _, _, executor, navigation = search
    snapshots = []
    provider = node.extra.provider
    def record(raw, live, *args, **kwargs):
        snapshots.append(live.data[8 + 20 * 8])
        return provider(raw, live, *args, **kwargs)
    node.extra.provider = record
    node.step_logic()
    executor.finish()
    advance(search, 1.1)
    node.step_logic()
    executor.finish()
    advance(search, 3.0)
    node.step_logic()
    changed = map_message()
    changed.data[8 + 20 * 8] = 100
    node.on_costmap(changed)
    executor.finish()
    advance(search, 1.3)
    node.on_costmap(changed)
    node.step_logic()
    executor.finish()
    node.step_logic()
    assert len(navigation.sent) == 1
    assert snapshots == [0, 100]
