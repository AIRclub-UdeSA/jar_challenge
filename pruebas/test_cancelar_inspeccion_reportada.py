from concurrent.futures import Future
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from std_msgs.msg import Bool
from equipo_jar_msgs.msg import Candidata
import pytest

from test_historial_mision import mission


class Goal:
    def __init__(self, accepted=True):
        self.accepted = accepted
        self.result = Future()
        self.cancel = Future()
        self.cancel_calls = 0
        self.cancel_error = None

    def cancel_goal_async(self):
        self.cancel_calls += 1
        if self.cancel_error is not None:
            raise self.cancel_error
        return self.cancel

    def get_result_async(self):
        return self.result

    def finish(self, status):
        self.result.set_result(SimpleNamespace(status=status))


class Navigation:
    def __init__(self):
        self.sent = []
        self.handles = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, goal, feedback_callback):
        self.sent.append(goal)
        handle = Goal()
        self.handles.append(handle)
        future = Future()
        future.set_result(handle)
        return future


def candidate(id_=1, x=2.0, y=2.0):
    result = Candidata()
    result.id = id_
    result.centro.x, result.centro.y = x, y
    result.observaciones = 5
    result.covarianza_xy = [0.001, 0.0, 0.0, 0.001]
    return result


@pytest.fixture
def inspecting(mission, monkeypatch):
    node, clock, _ = mission
    navigation = Navigation()
    monkeypatch.setattr(node, 'navegacion', navigation)
    reports = []
    monkeypatch.setattr(node, 'reportar_victima', lambda x, y: reports.append((x, y)))
    node.current_inspection = (1, 2.0, 2.0, 2)
    node.inspection_spot = (2.8, 2.0)
    node.inspected_points.append((2.0, 2.0))
    node.plan = SimpleNamespace(poses=[node.make_pose(0.4, 0.4, 0.0)])
    node.goal_start_time = node.last_motion_time = node.seconds_since_t0()
    node.last_motion_pose = node.odom_pose()
    node.candidatas = []
    goal = Goal()
    accepted = Future()
    accepted.set_result(goal)
    node.on_goal_response(accepted)
    return node, clock, goal, navigation, reports


def test_report_cancels_current_inspection_once_and_waits_for_result(inspecting):
    node, _, goal, navigation, reports = inspecting
    node.candidatas = [candidate()]
    node.step_logic()
    node.step_logic()
    assert reports == [(2.0, 2.0)]
    assert goal.cancel_calls == 1
    assert node.nav_state == 'moving'
    assert navigation.sent == []
    goal.cancel.set_result(SimpleNamespace(goals_canceling=[object()]))
    node.step_logic()
    assert navigation.sent == []
    goal.finish(GoalStatus.STATUS_CANCELED)
    node.step_logic()
    assert node.current_inspection is None
    assert node.look_until is None
    assert node.failed_inspections == {}
    assert node.tried_views == {}
    assert len(navigation.sent) == 1
    assert navigation.sent[0].pose.pose.position.x == 0.4


@pytest.mark.parametrize('status', [GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_ABORTED,
                                  GoalStatus.STATUS_CANCELED])
def test_completed_reported_inspection_is_never_a_failed_attempt(inspecting, status):
    node, _, goal, navigation, _ = inspecting
    node.reported_ids.add(1)
    node.reported_points.append((2.0, 2.0))
    node.step_logic()
    goal.finish(status)
    node.step_logic()
    assert node.failed_inspections == {}
    assert node.tried_views == {}
    assert node.current_inspection is None
    assert node.look_until is None
    assert len(navigation.sent) == 1


def test_report_before_goal_acceptance_waits_for_handle(inspecting):
    node, _, goal, navigation, _ = inspecting
    node.nav_state = 'sending'
    node.current_goal = None
    node.reported_ids.add(1)
    node.step_logic()
    assert goal.cancel_calls == 0
    assert navigation.sent == []
    handle = Future()
    handle.set_result(goal)
    node.on_goal_response(handle)
    node.step_logic()
    assert goal.cancel_calls == 1


def test_reporting_another_candidate_does_not_cancel_inspection(inspecting):
    node, _, goal, navigation, reports = inspecting
    node.candidatas = [candidate(2, 3.5, 0.5)]
    node.step_logic()
    assert reports == [(3.5, 0.5)]
    assert goal.cancel_calls == 0
    assert node.current_inspection is not None
    assert navigation.sent == []


def test_equivalent_reported_position_after_id_merge_cancels_inspection(inspecting):
    node, _, goal, _, _ = inspecting
    node.reported_ids.add(9)
    node.reported_points.append((2.05, 2.0))
    node.step_logic()
    assert goal.cancel_calls == 1


@pytest.mark.parametrize('outcome', ['refused', 'response_error', 'request_error'])
def test_cancel_failure_keeps_real_goal_active_without_request_storm(inspecting, outcome):
    node, clock, goal, navigation, _ = inspecting
    node.reported_ids.add(1)
    if outcome == 'request_error':
        goal.cancel_error = RuntimeError('cancel request failed')
    node.step_logic()
    if outcome == 'refused':
        goal.cancel.set_result(SimpleNamespace(goals_canceling=[]))
    elif outcome == 'response_error':
        goal.cancel.set_exception(RuntimeError('cancel response failed'))
    node.step_logic()
    node.step_logic()
    assert goal.cancel_calls == 1
    assert node.nav_state == 'moving'
    assert navigation.sent == []
    goal.cancel_error = None
    clock.seconds += node.p['goal_timeout_s'] + 1.0
    node.step_logic()
    assert goal.cancel_calls == 2


@pytest.mark.parametrize('outcome', ['accepted', 'refused', 'error'])
def test_late_cancel_response_cannot_affect_the_next_goal(inspecting, outcome):
    node, _, goal, navigation, _ = inspecting
    node.reported_ids.add(1)
    node.reported_points.append((2.0, 2.0))
    node.plan = SimpleNamespace(poses=[])
    node.step_logic()
    goal.finish(GoalStatus.STATUS_SUCCEEDED)
    node.step_logic()
    node.candidatas = [candidate(2, 3.5, 0.5)]
    node.candidatas[0].observaciones = 2
    node.step_logic()
    next_goal = node.current_goal
    assert next_goal is navigation.handles[0]
    assert node.current_inspection[0] == 2
    node.candidatas[0].observaciones = 5
    node.step_logic()
    assert node.inspection_canceling
    assert next_goal.cancel_calls == 1
    if outcome == 'error':
        goal.cancel.set_exception(RuntimeError('late cancellation error'))
    else:
        goal.cancel.set_result(SimpleNamespace(goals_canceling=[object()] if outcome == 'accepted' else []))
    assert node.current_goal is next_goal
    assert node.inspection_canceling
    assert next_goal.cancel_calls == 1


def test_last_report_cancels_once_then_returns_only_after_result(inspecting):
    node, _, goal, navigation, _ = inspecting
    node.cantidad_victimas = 1
    node.candidatas = [candidate()]
    node.step_logic()
    node.step_logic()
    assert node.state == 'returning'
    assert goal.cancel_calls == 1
    assert navigation.sent == []
    goal.finish(GoalStatus.STATUS_CANCELED)
    node.step_logic()
    assert len(navigation.sent) == 1
    assert navigation.sent[0].pose.pose.position.x == 0.0
    assert navigation.sent[0].behavior_tree == node.return_tree


def test_pending_cancel_does_not_bypass_final_stop_deadline(inspecting):
    node, clock, goal, navigation, _ = inspecting
    node.reported_ids.add(1)
    node.step_logic()
    clock.seconds = 5.0 + node.p['finish_time_s']
    node.step_logic()
    assert node.state == 'stopping'
    node.step_logic()
    assert goal.cancel_calls == 2
    clock.seconds = 5.0 + node.p['done_limit_s']
    node.on_stop_confirmed(Bool(data=True))
    node.step_logic()
    assert node.state == 'finishing'
    node.step_logic()
    assert node.state == 'finished'
    goal.cancel.set_result(SimpleNamespace(goals_canceling=[object()]))
    goal.finish(GoalStatus.STATUS_CANCELED)
    node.step_logic()
    assert navigation.sent == []


def test_pending_cancel_keeps_existing_goal_timeout(inspecting):
    node, clock, goal, navigation, _ = inspecting
    node.reported_ids.add(1)
    node.step_logic()
    clock.seconds += node.p['goal_timeout_s'] + 1.0
    node.step_logic()
    assert goal.cancel_calls == 2
    assert node.stuck_canceling
    assert navigation.sent == []


def test_watchdog_cancellation_is_not_duplicated_by_a_report(inspecting):
    node, _, goal, _, _ = inspecting
    node.stuck_canceling = True
    goal.cancel_goal_async()
    node.reported_ids.add(1)
    node.step_logic()
    assert goal.cancel_calls == 1


def test_unreported_inspection_keeps_normal_arrival_wait(inspecting):
    node, _, goal, navigation, _ = inspecting
    node.candidatas = [candidate()]
    node.candidatas[0].observaciones = 2
    goal.finish(GoalStatus.STATUS_SUCCEEDED)
    node.step_logic()
    assert node.current_inspection is not None
    assert node.look_until == node.seconds_since_t0() + node.p['inspection_wait_s']
    assert goal.cancel_calls == 0
    assert navigation.sent == []


def test_unreported_failed_inspection_keeps_failure_accounting(inspecting):
    node, _, goal, _, _ = inspecting
    goal.finish(GoalStatus.STATUS_ABORTED)
    node.step_logic()
    assert node.failed_inspections[1][1] == 1
    assert node.tried_views[1] == [(2.8, 2.0)]
