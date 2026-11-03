"""Terminal navigation failures must close the velocity path before /done."""
from concurrent.futures import Future
from types import SimpleNamespace

from action_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry
import pytest
from std_msgs.msg import Bool

from test_historial_mision import mission
from test_cancelar_inspeccion_reportada import Goal, candidate


class PendingNavigation:
    def __init__(self):
        self.response = Future()
        self.calls = 0

    def server_is_ready(self):
        return True

    def send_goal_async(self, *args, **kwargs):
        self.calls += 1
        return self.response


@pytest.mark.parametrize('state', ['exploring', 'returning'])
def test_lost_acceptance_stops_within_five_seconds(mission, monkeypatch, state):
    node, clock, _ = mission
    nav = PendingNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    node.p['goal_accept_max_timeouts'] = 1
    node.state = state
    assert node.go_to(node.make_pose(1.0, 1.0, 0.0))
    clock.seconds += 5.1
    node.step_logic()
    assert node.state == 'stopping'
    assert nav.calls == 1
    late = Goal()
    nav.response.set_result(late)
    assert late.cancel_calls == 1
    assert node.state == 'stopping'
    assert node.nav_state != 'moving'
    assert not node.go_to(node.make_pose(2.0, 1.0, 0.0))


def test_final_deadline_never_publishes_done_without_stop_confirmation(mission, monkeypatch):
    node, clock, _ = mission
    done = []
    monkeypatch.setattr(node, 'pub_done', SimpleNamespace(publish=done.append))
    node.nav_state = 'moving'
    node.current_goal = Goal()
    node.change_state('stopping')
    clock.seconds = 5.0 + node.p['done_limit_s'] + 1.0
    node.step_logic()
    node.step_logic()
    assert node.state == 'stopping'
    assert done == []


def test_confirmed_velocity_inhibit_can_finish_despite_hung_nav2(mission, monkeypatch):
    node, clock, _ = mission
    done, stop = [], []
    monkeypatch.setattr(node, 'pub_done', SimpleNamespace(publish=done.append))
    monkeypatch.setattr(node, 'pub_stop', SimpleNamespace(publish=stop.append))
    node.current_goal = Goal()
    node.nav_state = 'moving'
    node.change_state('stopping')
    assert stop and stop[-1].data
    node.step_logic()
    assert node.state == 'stopping'
    clock.seconds += node.p['stop_wait_s'] + 0.1
    node.on_stop_confirmed(Bool(data=True))
    node.step_logic()
    node.step_logic()
    assert node.state == 'finished'
    assert len(done) == 1
    node.step_logic()
    assert len(done) == 1


def test_cancel_exception_cannot_prevent_velocity_inhibit(mission, monkeypatch):
    node, _, _ = mission
    stop = []
    monkeypatch.setattr(node, 'pub_stop', SimpleNamespace(publish=stop.append))
    goal = Goal()
    goal.cancel_error = RuntimeError('Nav2 unavailable')
    node.current_goal = goal
    node.nav_state = 'moving'
    node.change_state('stopping')
    node.step_logic()
    node.step_logic()
    assert stop[-1].data
    assert goal.cancel_calls == 1
    assert node.state == 'stopping'


def test_acceptance_just_before_timeout_keeps_normal_navigation(mission, monkeypatch):
    node, clock, _ = mission
    nav = PendingNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    assert node.go_to(node.make_pose(1.0, 1.0, 0.0))
    clock.seconds += 4.9
    goal = Goal()
    nav.response.set_result(goal)
    node.step_logic()
    assert node.state == 'exploring'
    assert node.nav_state == 'moving'
    assert goal.cancel_calls == 0


def test_synchronous_send_failure_closes_navigation_output(mission, monkeypatch):
    node, _, _ = mission
    nav = PendingNavigation()
    def unavailable(*args, **kwargs):
        raise RuntimeError('action transport unavailable')
    nav.send_goal_async = unavailable
    monkeypatch.setattr(node, 'navegacion', nav)
    assert not node.go_to(node.make_pose(1.0, 1.0, 0.0))
    assert node.state == 'stopping'


def test_stale_or_revoked_ack_cannot_publish_done(mission, monkeypatch):
    node, clock, _ = mission
    done = []
    monkeypatch.setattr(node, 'pub_done', SimpleNamespace(publish=done.append))
    node.change_state('stopping')
    node.on_stop_confirmed(Bool(data=True))
    clock.seconds += 2.0
    node.step_logic()
    assert node.state == 'stopping'
    node.on_stop_confirmed(Bool(data=True))
    node.step_logic()
    assert node.state == 'finishing'
    node.on_stop_confirmed(Bool(data=False))
    node.step_logic()
    assert node.state == 'stopping'
    assert done == []


def test_asynchronous_acceptance_error_also_closes_output(mission, monkeypatch):
    node, _, _ = mission
    nav = PendingNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    assert node.go_to(node.make_pose(1.0, 1.0, 0.0))
    nav.response.set_exception(RuntimeError('acceptance response lost'))
    assert node.state == 'stopping'
    assert not node.go_to(node.make_pose(2.0, 1.0, 0.0))


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1.0, 0.0, True, 60.0])
def test_invalid_acceptance_timeout_stops_instead_of_waiting_forever(mission, monkeypatch, invalid):
    node, _, _ = mission
    node.p['goal_accept_timeout_s'] = invalid
    monkeypatch.setattr(node, 'navegacion', PendingNavigation())
    node.go_to(node.make_pose(1.0, 1.0, 0.0))
    node.step_logic()
    assert node.state == 'stopping'


def test_accepted_goal_with_unavailable_result_transport_stops(mission, monkeypatch):
    node, _, _ = mission
    nav = PendingNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    node.go_to(node.make_pose(1.0, 1.0, 0.0))
    goal = Goal()
    def unavailable():
        raise RuntimeError('result transport unavailable')
    goal.get_result_async = unavailable
    nav.response.set_result(goal)
    assert node.state == 'stopping'
    assert not node.go_to(node.make_pose(2.0, 1.0, 0.0))


class FreshNavigation:
    """Each goal gets its own pending acceptance."""

    def __init__(self):
        self.responses = []

    def server_is_ready(self):
        return True

    def send_goal_async(self, *args, **kwargs):
        self.responses.append(Future())
        return self.responses[-1]


def test_one_lost_acceptance_fails_only_that_trip(mission, monkeypatch):
    node, clock, _ = mission
    nav = FreshNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    assert node.go_to(node.make_pose(1.0, 1.0, 0.0))
    clock.seconds += node.p['goal_accept_timeout_s'] + 0.1
    node.step_logic()
    assert node.state == 'exploring'
    assert node.nav_state != 'sending'
    late = Goal()
    nav.responses[0].set_result(late)
    assert late.cancel_calls == 1
    assert node.current_goal is not late
    assert node.nav_state != 'moving'


def test_repeated_lost_acceptances_still_stop(mission, monkeypatch):
    node, clock, _ = mission
    nav = FreshNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    for _ in range(node.p['goal_accept_max_timeouts']):
        assert node.go_to(node.make_pose(1.0, 1.0, 0.0))
        clock.seconds += node.p['goal_accept_timeout_s'] + 0.1
        node.step_logic()
    assert node.state == 'stopping'


def accept(node, goal):
    """Run a goal through go_to's numbering and accept it."""
    node.goal_seq += 1
    node.nav_state = 'sending'
    handle = Future()
    handle.set_result(goal)
    node.on_goal_response(handle, node.goal_seq)
    node.goal_start_time = node.last_motion_time = node.seconds_since_t0()
    node.last_motion_pose = node.odom_pose()


def test_cancel_without_result_gives_the_trip_up(mission, monkeypatch):
    node, _, _ = mission
    wall = SimpleNamespace(t=100.0)
    monkeypatch.setattr(node, 'wall_now', lambda: wall.t)
    goal = Goal()
    accept(node, goal)
    assert node.nav_state == 'moving'
    assert node.cancel_current('test')
    node.step_logic()
    assert node.nav_state == 'moving'
    wall.t += node.p['cancel_timeout_s'] + 0.1
    node.step_logic()
    assert node.nav_state == 'idle'
    assert node.current_goal is None
    goal.finish(GoalStatus.STATUS_CANCELED)     # late result of the abandoned goal
    assert node.nav_state == 'idle'


def odom(vx=0.0, wz=0.0):
    msg = Odometry()
    msg.twist.twist.linear.x = vx
    msg.twist.twist.angular.z = wz
    return msg


def run_for(node, wall, seconds, vx=0.0):
    end = wall.t + seconds
    while wall.t < end:
        wall.t += 0.5
        node.on_odom(odom(vx))
        node.step_logic()


def test_still_guard_cancels_then_reports_before_the_60_s_rule(mission, monkeypatch):
    node, _, _ = mission
    wall = SimpleNamespace(t=100.0)
    monkeypatch.setattr(node, 'wall_now', lambda: wall.t)
    reports = []
    monkeypatch.setattr(node, 'reportar_victima', lambda x, y: reports.append((x, y)))
    goal = Goal()
    accept(node, goal)
    seen_once = candidate()
    seen_once.observaciones = 1     # only the loose final round reports it
    node.candidatas = [seen_once]
    run_for(node, wall, node.p['still_rescue_s'] + 1.0)
    assert goal.cancel_calls == 1
    assert node.state == 'exploring'
    assert reports == []
    run_for(node, wall, node.p['still_finish_s'] - node.p['still_rescue_s'])
    assert reports == [(2.0, 2.0)]
    # No /done: if the robot moves again, the run goes on.
    assert node.state == 'exploring'
    run_for(node, wall, 10.0)
    assert reports == [(2.0, 2.0)]


def test_still_time_before_the_first_goal_does_not_count(mission, monkeypatch):
    # The robot waits still for /map (any time) and while it localizes: seen in the
    # simulator, the guard cancelled the first goal right after it was accepted.
    node, _, _ = mission
    wall = SimpleNamespace(t=100.0)
    monkeypatch.setattr(node, 'wall_now', lambda: wall.t)
    reports = []
    monkeypatch.setattr(node, 'reportar_victima', lambda x, y: reports.append((x, y)))
    seen_once = candidate()
    seen_once.observaciones = 1
    node.candidatas = [seen_once]
    run_for(node, wall, 120.0)
    goal = Goal()
    accept(node, goal)
    run_for(node, wall, node.p['still_rescue_s'] - 1.0)
    assert goal.cancel_calls == 0
    assert reports == []
    assert node.state == 'exploring'
    run_for(node, wall, 2.0)
    assert goal.cancel_calls == 1


def test_still_guard_ignores_a_moving_robot(mission, monkeypatch):
    node, _, _ = mission
    wall = SimpleNamespace(t=100.0)
    monkeypatch.setattr(node, 'wall_now', lambda: wall.t)
    goal = Goal()
    accept(node, goal)
    run_for(node, wall, 70.0, vx=0.05)
    assert goal.cancel_calls == 0
    assert node.state == 'exploring'


def test_still_guard_waits_for_a_first_accepted_goal(mission, monkeypatch):
    node, _, _ = mission
    wall = SimpleNamespace(t=100.0)
    monkeypatch.setattr(node, 'wall_now', lambda: wall.t)
    run_for(node, wall, 70.0)
    assert node.state == 'exploring'


def test_lost_acceptance_keeps_the_plan_point(mission, monkeypatch):
    node, clock, _ = mission
    point = node.make_pose(1.0, 1.0, 0.0)
    node.plan = SimpleNamespace(poses=[point])
    nav = FreshNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    node.step_logic()
    assert len(nav.responses) == 1
    clock.seconds += node.p['goal_accept_timeout_s'] + 0.1
    node.step_logic()                       # acceptance timed out: trip failed
    node.step_logic()
    assert node.visited == []
    clock.seconds += 1.1
    node.step_logic()
    assert len(nav.responses) == 2          # the same point is tried again


def test_rejected_goal_keeps_the_plan_point(mission, monkeypatch):
    node, clock, _ = mission
    point = node.make_pose(1.0, 1.0, 0.0)
    node.plan = SimpleNamespace(poses=[point])
    nav = FreshNavigation()
    monkeypatch.setattr(node, 'navegacion', nav)
    node.step_logic()
    assert len(nav.responses) == 1
    nav.responses[0].set_result(Goal(accepted=False))
    node.step_logic()
    assert node.visited == []
    assert len(nav.responses) == 1          # waits before retrying
    clock.seconds += 1.1
    node.step_logic()
    assert len(nav.responses) == 2


def test_unreachable_robot_cell_counts_as_a_failed_inspection(mission, monkeypatch):
    node, _, _ = mission
    doubtful = candidate()
    doubtful.observaciones = 2
    monkeypatch.setattr(node, 'next_doubtful', lambda: doubtful)

    def unsafe(*args, **kwargs):
        raise ValueError('robot cell is not safe')
    monkeypatch.setattr(node.grid, 'reachable_from', unsafe)
    assert not node.inspect_doubtful()
    assert node.failed_inspections[doubtful.id][1] == 1
