"""Planner watch: stale-path detection and the planner_server restart sequence."""
import ast
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState

from equipo_jar.localizacion import Arranque
from equipo_jar.vigia_planner import VigiaPlanner

P = {'vigia_planner_distancia_m': 0.5, 'vigia_planner_veces': 3,
     'vigia_planner_espera_s': 30.0}


def test_paths_from_the_robot_never_restart():
    vigia = VigiaPlanner(dict(P))
    for t in range(20):
        assert vigia.con_plan(float(t), (1.0, 1.0), (1.3, 1.2)) is None
    assert vigia.reinicios == 0


def test_three_stale_paths_in_a_row_restart_once_per_wait():
    vigia = VigiaPlanner(dict(P))
    assert vigia.con_plan(0.0, (-1.82, -2.72), (-1.0, -2.0)) is None
    assert vigia.con_plan(1.0, (-1.82, -2.72), (-0.9, -2.0)) is None
    assert vigia.con_plan(2.0, (-1.82, -2.72), (-0.8, -2.0)) == pytest.approx(1.249, abs=0.01)
    for t in range(3, 32):                      # still stale, but within the wait
        assert vigia.con_plan(float(t), (-1.82, -2.72), (1.0, 1.0)) is None
    assert vigia.con_plan(33.0, (-1.82, -2.72), (1.0, 1.0)) is not None
    assert vigia.reinicios == 2


def test_one_good_path_resets_the_count():
    vigia = VigiaPlanner(dict(P))
    vigia.con_plan(0.0, (0.0, 0.0), (1.0, 0.0))
    vigia.con_plan(1.0, (0.0, 0.0), (1.0, 0.0))
    vigia.con_plan(2.0, (1.0, 0.0), (1.0, 0.0))
    assert vigia.con_plan(3.0, (0.0, 0.0), (1.0, 0.0)) is None
    assert vigia.seguidos == 1


@pytest.mark.parametrize('limite', [0.0, -1.0])
def test_off(limite):
    vigia = VigiaPlanner(dict(P, vigia_planner_distancia_m=limite))
    for t in range(10):
        assert vigia.con_plan(float(t), (0.0, 0.0), (5.0, 5.0)) is None


def test_nan_pose_is_ignored():
    vigia = VigiaPlanner(dict(P))
    for t in range(5):
        assert vigia.con_plan(float(t), (0.0, 0.0), (float('nan'), 0.0)) is None


class FakeClient:
    """Lifecycle service client of one fake node: answers at once."""

    def __init__(self, node, kind):
        self.node, self.kind = node, kind

    def service_is_ready(self):
        return True

    def call_async(self, request):
        future = Future()
        if self.kind is GetState:
            response = GetState.Response()
            response.current_state.id = self.node.state
            response.current_state.label = str(self.node.state)
        else:
            self.node.transitions.append(request.transition.id)
            self.node.state = {
                Transition.TRANSITION_CONFIGURE: State.PRIMARY_STATE_INACTIVE,
                Transition.TRANSITION_ACTIVATE: State.PRIMARY_STATE_ACTIVE,
                Transition.TRANSITION_DEACTIVATE: State.PRIMARY_STATE_INACTIVE,
                Transition.TRANSITION_CLEANUP: State.PRIMARY_STATE_UNCONFIGURED,
            }[request.transition.id]
            response = ChangeState.Response(success=True)
        self.node.pending.append((future, response))
        return future

    def remove_pending_request(self, future):
        pass


class FakeNav2:
    def __init__(self):
        self.state = State.PRIMARY_STATE_ACTIVE
        self.transitions = []
        self.pending = []
        self.logs = []

    def get_logger(self):
        return SimpleNamespace(info=self.logs.append, warning=self.logs.append)

    def create_client(self, kind, name):
        return FakeClient(self, kind)

    def run(self, limit=50):
        """Deliver answers one at a time, as the executor would."""
        for _ in range(limit):
            if not self.pending:
                return
            future, response = self.pending.pop(0)
            future.set_result(response)


def test_restart_goes_down_to_unconfigured_and_back_up():
    nav2 = FakeNav2()
    arranque = Arranque(nav2, ['planner_server'], 5.0)
    arranque.paso()
    nav2.run()
    assert arranque.faltan == [] and nav2.transitions == []
    assert arranque.reiniciar('planner_server')
    nav2.run()
    assert nav2.transitions == [Transition.TRANSITION_DEACTIVATE, Transition.TRANSITION_CLEANUP,
                                Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE]
    assert nav2.state == State.PRIMARY_STATE_ACTIVE
    assert arranque.faltan == [] and not arranque.bajando


def test_no_restart_while_start_up_is_running_or_for_unknown_nodes():
    nav2 = FakeNav2()
    nav2.state = State.PRIMARY_STATE_UNCONFIGURED
    arranque = Arranque(nav2, ['planner_server'], 5.0)
    assert not arranque.reiniciar('planner_server')     # still starting
    arranque.paso()
    nav2.run()
    assert not arranque.reiniciar('controller_server')
    assert arranque.reiniciar('planner_server')


# --- planner_server crash and respawn ---
# Server runs on 2026-11-02 (3 of 3 real restarts): with the TF view frozen, the restart's
# cleanup hung, a costmap clear request arrived and planner_server died (exit code -11). The
# launch respawns it; localizacion must configure and activate the new process.

class CrashingClient(FakeClient):
    def call_async(self, request):
        if self.node.dead:
            return Future()                        # a dead process never answers
        if (self.kind is ChangeState and self.node.die_on_cleanup
                and request.transition.id == Transition.TRANSITION_CLEANUP):
            self.node.dead = True                  # cleanup hangs, then the process dies
            return Future()
        return super().call_async(request)


class CrashingNav2(FakeNav2):
    def __init__(self):
        super().__init__()
        self.dead = False
        self.die_on_cleanup = False

    def create_client(self, kind, name):
        return CrashingClient(self, kind)

    def respawn(self):
        """The launch starts a new process: unconfigured, answering again."""
        self.dead = False
        self.die_on_cleanup = False
        self.state = State.PRIMARY_STATE_UNCONFIGURED


@pytest.fixture
def clock(monkeypatch):
    now = [100.0]
    monkeypatch.setattr('equipo_jar.localizacion.time.monotonic', lambda: now[0])
    return now


def active(nav2):
    arranque = Arranque(nav2, ['planner_server'], 5.0)
    arranque.paso()
    nav2.run()
    assert arranque.faltan == []
    return arranque


def check(arranque, nav2):
    """Two timer ticks of revisar: one asks the state, the next reads the answer."""
    arranque.revisar('planner_server')
    nav2.run()
    arranque.revisar('planner_server')
    nav2.run()


def test_crash_during_the_restart_is_recovered_after_respawn(clock):
    nav2 = CrashingNav2()
    arranque = active(nav2)
    nav2.die_on_cleanup = True
    assert arranque.reiniciar('planner_server')
    nav2.run()
    assert nav2.dead and arranque.faltan == ['planner_server']
    nav2.respawn()
    clock[0] += 6.0                                # the cleanup request times out
    arranque.paso()
    nav2.run()
    assert nav2.state == State.PRIMARY_STATE_ACTIVE
    assert arranque.faltan == [] and not arranque.bajando
    assert nav2.transitions[-2:] == [Transition.TRANSITION_CONFIGURE,
                                     Transition.TRANSITION_ACTIVATE]


def test_active_planner_is_left_alone(clock):
    nav2 = CrashingNav2()
    arranque = active(nav2)
    for _ in range(3):
        check(arranque, nav2)
    assert nav2.transitions == [] and arranque.faltan == []
    assert not any('should be active' in str(m) for m in nav2.logs)


def test_planner_respawned_at_any_other_time_is_brought_up_again(clock):
    nav2 = CrashingNav2()
    arranque = active(nav2)
    nav2.respawn()                                 # crashed outside a restart
    check(arranque, nav2)
    assert nav2.state == State.PRIMARY_STATE_ACTIVE and arranque.faltan == []
    assert nav2.transitions == [Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE]
    assert any('should be active' in str(m) for m in nav2.logs)


def test_check_of_a_dead_planner_times_out_and_asks_again(clock):
    nav2 = CrashingNav2()
    arranque = active(nav2)
    nav2.dead = True
    arranque.revisar('planner_server')
    arranque.revisar('planner_server')             # no answer yet: keep waiting
    assert arranque.consulta is not None
    clock[0] += 6.0
    arranque.revisar('planner_server')             # drop the unanswered request
    assert arranque.consulta is None and arranque.faltan == []
    nav2.respawn()
    check(arranque, nav2)
    assert nav2.state == State.PRIMARY_STATE_ACTIVE and arranque.faltan == []


def test_check_waits_while_start_up_runs_and_ignores_unknown_nodes(clock):
    nav2 = CrashingNav2()
    nav2.state = State.PRIMARY_STATE_UNCONFIGURED
    arranque = Arranque(nav2, ['planner_server'], 5.0)
    arranque.revisar('planner_server')             # start-up not done yet
    assert arranque.consulta is None
    arranque.paso()
    nav2.run()
    arranque.revisar('controller_server')          # not managed here
    assert arranque.consulta is None


def test_restart_drops_a_pending_check(clock):
    nav2 = CrashingNav2()
    arranque = active(nav2)
    arranque.revisar('planner_server')             # question sent, answer not read yet
    assert arranque.reiniciar('planner_server')
    nav2.run()
    assert arranque.consulta is None and arranque.faltan == []
    assert nav2.state == State.PRIMARY_STATE_ACTIVE


def test_launch_respawns_planner_server():
    launch = (Path(__file__).resolve().parents[1] / 'src' / 'equipo_jar' / 'launch'
              / 'navegacion.launch.py')
    calls = [n for n in ast.walk(ast.parse(launch.read_text())) if isinstance(n, ast.Call)
             and any(isinstance(a, ast.Constant) and a.value == 'planner_server' for a in n.args)]
    assert len(calls) == 1
    options = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(options.get('respawn'), ast.Constant) and options['respawn'].value is True
