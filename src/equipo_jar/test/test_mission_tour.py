"""The mission's tour bookkeeping without ROS: retour while driving."""
import pytest

mission = pytest.importorskip('equipo_jar.mission')  # needs equipo_jar_msgs (the workspace)

from test_planner import planner  # noqa: E402


class Node(mission.Mission):
    """A Mission without the ROS node: just what retour needs."""

    def __init__(self, objects):
        self.n = 1
        self.objects = objects
        self.reported = []
        self.looks = {}
        self.dead_looks = {}
        self.seen_from = {}
        self.lock = __import__('threading').Lock()
        self.p = dict(mission.PARAMS)
        self.planner_params = dict(mission.PLANNER_DEFAULTS)
        self.tour, self.tour_objects, self.tour_failed, self.prefer = [], set(), [], None
        self.tour_visited = []


BOX = {'id': 1, 'center': (1.3, 2.4), 'p': 0.6, 'radius': 0.12, 'miradores': []}


def test_a_box_seen_behind_the_robot_turns_it_back_now():
    p, _, _ = planner(robot=(2.4, 1.5))
    far = p.tour([dict(BOX, id=2, center=(5.2, 1.5), p=0.3)], n_total=1)[0]
    node = Node([dict(BOX, id=2, center=(5.2, 1.5), p=0.3)])
    node.tour_objects = node.undecided(node.objects, p, [])
    assert not node.retour(p, far)          # nothing new: keep driving
    node.objects = node.objects + [BOX]     # a valuable box shows up behind the robot
    assert node.retour(p, far)
    assert node.prefer is not None and node.prefer != far.xy
    assert not node.retour(p, far)          # the same objects again: decided once


def test_no_switch_when_the_new_tour_still_starts_at_the_target():
    p, _, _ = planner(robot=(1.5, 1.5))
    node = Node([BOX])
    target = p.tour([BOX], n_total=1)[0]
    node.tour_objects = node.undecided(node.objects, p, [])
    node.objects = [BOX, dict(BOX, id=3, center=(5.5, 2.6), p=0.2)]  # far away, worth less
    assert not node.retour(p, target)
    assert node.prefer is None


def test_a_box_that_gets_a_look_spot_on_the_way_turns_it_back():
    # laberinto_1 with the sim realism: the corridor box's first mirador was beside the stop
    # just left, where no look goes; its second, half a second later, gave it a look spot,
    # but the box was already known and the robot drove on for 15 s and came back.
    p, _, _ = planner(robot=(2.4, 1.5))
    other = dict(BOX, id=2, center=(5.2, 1.5), p=0.3)
    far = p.tour([other], n_total=1)[0]
    hidden = dict(BOX, center=(0.4, 0.4))   # in the closed pocket: no line of sight
    node = Node([other, hidden])
    node.tour_objects = node.undecided(node.objects, p, [])
    assert not node.retour(p, far)          # known, nowhere to look from: keep driving
    node.objects = [other, dict(hidden, miradores=[(1.5, 1.0)])]  # the lidar saw it
    assert node.retour(p, far)
    assert node.prefer is not None and node.prefer != far.xy


def test_home_is_the_origin_or_the_start():
    from equipo_jar.mission import home_target
    assert home_target('origen', (0.12, -0.08)) == (0.0, 0.0)
    assert home_target('inicio', (0.12, -0.08)) == (0.12, -0.08)
    with pytest.raises(ValueError):
        home_target('origin', (0.0, 0.0))
