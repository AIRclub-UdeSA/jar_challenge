"""Nav2 start-up and revisar_todos: every respawned Nav2 node is brought back to active.

The planner_server-only cases are in pruebas/test_vigia_planner.py."""
from types import SimpleNamespace

import pytest

lifecycle = pytest.importorskip('lifecycle_msgs')
from lifecycle_msgs.msg import State, Transition  # noqa: E402
from lifecycle_msgs.srv import ChangeState, GetState  # noqa: E402

from equipo_jar.localizacion import Arranque  # noqa: E402


class FakeFuture:
    def __init__(self):
        self.value = None
        self.callbacks = []

    def done(self):
        return self.value is not None

    def result(self):
        return self.value

    def add_done_callback(self, callback):
        self.callbacks.append(callback)

    def finish(self, value):
        self.value = value
        for callback in self.callbacks:
            callback(self)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.ready = True

    def service_is_ready(self):
        return self.ready

    def call_async(self, request):
        future = FakeFuture()
        self.calls.append((request, future))
        return future

    def remove_pending_request(self, future):
        pass


class FakeNode:
    def __init__(self):
        self.clients = {}
        self.warnings = []

    def get_logger(self):
        return SimpleNamespace(info=lambda *a, **k: None,
                               warning=lambda msg, **k: self.warnings.append(msg))

    def create_client(self, kind, name):
        self.clients[name] = FakeClient()
        return self.clients[name]


def answer(client, state_id):
    _, future = client.calls[-1]
    future.finish(GetState.Response(current_state=State(id=state_id, label=str(state_id))))


def bring_up(get, change):
    """Answer the usual unconfigured -> configure -> inactive -> activate -> active."""
    answer(get, State.PRIMARY_STATE_UNCONFIGURED)
    change.calls[-1][1].finish(ChangeState.Response(success=True))
    answer(get, State.PRIMARY_STATE_INACTIVE)
    change.calls[-1][1].finish(ChangeState.Response(success=True))
    answer(get, State.PRIMARY_STATE_ACTIVE)


def started(names=('planner_server',)):
    node = FakeNode()
    arranque = Arranque(node, list(names), 5.0)
    for name in names:
        arranque.paso()
        bring_up(node.clients[f'/{name}/get_state'], node.clients[f'/{name}/change_state'])
    assert not arranque.faltan
    return node, arranque


def test_start_up_configures_and_activates():
    node, _ = started()
    change = node.clients['/planner_server/change_state']
    assert [r.transition.id for r, _ in change.calls] == [
        Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE]


def test_every_node_is_checked_in_turn():
    names = ('amcl', 'controller_server', 'planner_server')
    node, arranque = started(names)
    asked = []
    for _ in range(2 * len(names)):
        arranque.revisar_todos()              # ask one node
        name = arranque.consulta[0]
        asked.append(name)
        answer(node.clients[f'/{name}/get_state'], State.PRIMARY_STATE_ACTIVE)
        arranque.revisar_todos()              # read its answer
    assert asked == list(names) * 2
    assert not arranque.faltan and not node.warnings


def test_a_respawned_controller_is_brought_back_to_active():
    names = ('controller_server', 'planner_server')
    node, arranque = started(names)
    get = node.clients['/controller_server/get_state']
    change = node.clients['/controller_server/change_state']
    arranque.revisar_todos()
    assert arranque.consulta[0] == 'controller_server'
    answer(get, State.PRIMARY_STATE_UNCONFIGURED)   # the launch started a new process
    arranque.revisar_todos()
    assert arranque.faltan == ['controller_server'] and node.warnings
    bring_up(get, change)
    assert not arranque.faltan
    assert [r.transition.id for r, _ in change.calls][-2:] == [
        Transition.TRANSITION_CONFIGURE, Transition.TRANSITION_ACTIVATE]


def test_a_dead_process_is_skipped():
    names = ('controller_server', 'planner_server')
    node, arranque = started(names)
    node.clients['/controller_server/get_state'].ready = False
    arranque.revisar_todos()                  # controller: not ready, nothing asked
    assert arranque.consulta is None
    arranque.revisar_todos()                  # next one is asked
    assert arranque.consulta[0] == 'planner_server'
