from types import SimpleNamespace

import pytest

from equipo_jar import mission
from equipo_jar_msgs.msg import EstadoMovimiento


class Log:

    def info(self, msg):
        pass

    error = info


class Driver:
    """The motion node confirms a stop (QUIETO) once it has one."""

    def __init__(self, events):
        self.events, self.stopped = events, False

    def stop(self):
        self.events.append('stop')
        self.stopped = True

    def state(self):
        return EstadoMovimiento.QUIETO if self.stopped else EstadoMovimiento.SIGUIENDO


class Node:

    def __init__(self, events, leftovers_fail=False):
        self.p = dict(mission.PARAMS)
        self.t, self.events, self.reported = 0.0, events, []
        self.leftovers_fail = leftovers_fail
        self.done_pub = SimpleNamespace(publish=lambda msg: events.append('done'))

    def now(self):
        self.t += 0.1
        return self.t

    def elapsed(self):
        return self.now()

    def get_logger(self):
        return Log()

    def report_leftovers(self):
        if self.leftovers_fail:
            raise RuntimeError('leftovers')
        self.events.append('leftovers')


@pytest.fixture(autouse=True)
def no_ros(monkeypatch):
    monkeypatch.setattr(mission.rclpy, 'ok', lambda: True)
    monkeypatch.setattr(mission.time, 'sleep', lambda s: None)


def test_done_comes_after_the_stop(monkeypatch):
    events = []
    monkeypatch.setattr(mission, 'go_home', lambda drv, node, log: events.append('home'))
    mission.finish(Driver(events), Node(events), Log())
    assert events[:2] == ['leftovers', 'home']
    assert events[2:] == ['stop'] + ['done'] * 5


def test_done_even_if_everything_fails(monkeypatch):
    events = []

    def go_home(drv, node, log):
        events.append('home')
        raise RuntimeError('home')

    monkeypatch.setattr(mission, 'go_home', go_home)
    node = Node(events, leftovers_fail=True)
    mission.finish(Driver(events), node, Log())
    assert events.count('home') == node.p['errores_max']
    assert events[-6:] == ['stop'] + ['done'] * 5
