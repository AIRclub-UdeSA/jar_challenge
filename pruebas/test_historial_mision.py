import math
from types import SimpleNamespace

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
import pytest
import rclpy
from rclpy.clock import ClockType
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Bool
from tf2_ros import TransformException

from equipo_jar.historial_vistas import ObservedView
from equipo_jar.mapa import Pose
from equipo_jar.mision import Mision


def instant(seconds):
    return Time(seconds=seconds, clock_type=ClockType.ROS_TIME)


def map_message():
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width = msg.info.height = 20
    msg.info.resolution = 0.2
    msg.info.origin.orientation.w = 1.0
    msg.data = [0] * 400
    return msg


@pytest.fixture
def mission(monkeypatch):
    rclpy.init()
    node = Mision()
    clock = SimpleNamespace(seconds=10.0)
    monkeypatch.setattr(node, 'get_clock', lambda: SimpleNamespace(now=lambda: instant(clock.seconds)))
    transform = TransformStamped()
    transform.header.frame_id = 'map'
    transform.child_frame_id = 'base_footprint'
    transform.header.stamp = instant(10.0).to_msg()
    transform.transform.translation.x = 1.2
    transform.transform.translation.y = 2.3
    transform.transform.rotation.w = 1.0
    monkeypatch.setattr(node.tf_buffer, 'lookup_transform', lambda *args: transform)
    node.al_recibir_mapa(map_message())
    node.t0 = instant(5.0)
    node.state = 'exploring'
    node.start_locked = True
    node.start_pose = (0.0, 0.0, 0.0)
    node.on_localization_ready(Bool(data=True))
    try:
        yield node, clock, transform
    finally:
        node.destroy_node()
        rclpy.shutdown()


def enable(node):
    result = node.set_parameters([Parameter('extra_search_enabled', value=True)])
    assert result[0].successful


def test_disabled_history_does_not_query_tf(mission, monkeypatch):
    node, _, _ = mission
    def unexpected(*args):
        pytest.fail('disabled history queried TF')
    monkeypatch.setattr(node.tf_buffer, 'lookup_transform', unexpected)
    node.update_search_history()
    assert node.search_views == ()


def test_enabled_history_records_actual_pose_instead_of_requested_goal(mission):
    node, _, _ = mission
    enable(node)
    node.current_point = node.make_pose(3.0, 3.0, math.pi)
    node.nav_state = 'failed'
    node.update_search_history()
    assert node.search_views == (ObservedView(Pose(1.2, 2.3, 0.0), 5.0),)


def test_step_collects_actual_views_when_enabled(mission, monkeypatch):
    node, _, _ = mission
    enable(node)
    monkeypatch.setattr(node, 'explore', lambda: None)
    node.step_logic()
    assert node.search_views == (ObservedView(Pose(1.2, 2.3, 0.0), 5.0),)


@pytest.mark.parametrize('state', ['waiting_for_map', 'returning', 'checking',
                                  'stopping', 'finishing', 'finished'])
def test_history_is_only_collected_during_exploration(mission, state):
    node, _, _ = mission
    enable(node)
    node.state = state
    node.update_search_history()
    assert node.search_views == ()


def test_missing_map_does_not_create_an_observation(mission):
    node, _, _ = mission
    enable(node)
    node.grid = None
    node.update_search_history()
    assert node.search_views == ()


@pytest.mark.parametrize('stamp', [0.0, 8.9, 10.1])
def test_zero_stale_or_future_tf_is_not_a_view(mission, stamp):
    node, _, transform = mission
    enable(node)
    transform.header.stamp = instant(stamp).to_msg()
    node.update_search_history()
    assert node.search_views == ()


def test_tf_exactly_at_the_age_limit_is_accepted(mission):
    node, _, transform = mission
    enable(node)
    transform.header.stamp = instant(9.0).to_msg()
    node.update_search_history()
    assert len(node.search_views) == 1


def test_missing_tf_does_not_interrupt_mission(mission, monkeypatch):
    node, _, _ = mission
    enable(node)
    def missing(*args):
        raise TransformException('missing test transform')
    monkeypatch.setattr(node.tf_buffer, 'lookup_transform', missing)
    node.update_search_history()
    assert node.search_views == ()


def test_latest_localization_status_gates_history_without_changing_old_start_latch(mission):
    node, _, _ = mission
    enable(node)
    node.on_localization_ready(Bool(data=False))
    node.update_search_history()
    assert node.localization_ready is True
    assert node.search_views == ()
    node.on_localization_ready(Bool(data=True))
    node.update_search_history()
    assert len(node.search_views) == 1


@pytest.mark.parametrize('invalid', ['quaternion', 'position', 'frame'])
def test_invalid_transform_is_not_a_view(mission, invalid):
    node, _, transform = mission
    enable(node)
    if invalid == 'quaternion':
        transform.transform.rotation.w = 0.0
    elif invalid == 'position':
        transform.transform.translation.x = math.nan
    else:
        transform.header.frame_id = 'odom'
    node.update_search_history()
    assert node.search_views == ()


def test_history_sampling_and_live_toggle(mission):
    node, clock, transform = mission
    enable(node)
    node.update_search_history()
    clock.seconds = 10.1
    transform.header.stamp = instant(10.1).to_msg()
    node.update_search_history()
    assert len(node.search_views) == 1
    transform.transform.translation.x = 1.4
    node.update_search_history()
    assert len(node.search_views) == 2
    node.set_parameters([Parameter('extra_search_enabled', value=False)])
    clock.seconds = 12.0
    transform.header.stamp = instant(12.0).to_msg()
    node.update_search_history()
    assert len(node.search_views) == 2


def test_identical_map_preserves_history_and_changed_geometry_resets_it(mission):
    node, _, _ = mission
    enable(node)
    node.update_search_history()
    previous = node.search_views
    node.al_recibir_mapa(map_message())
    assert node.search_views == previous
    changed = map_message()
    changed.data[30] = 100
    node.al_recibir_mapa(changed)
    assert node.search_views == ()


def test_invalid_map_drops_history(mission):
    node, _, _ = mission
    enable(node)
    node.update_search_history()
    invalid = map_message()
    invalid.header.frame_id = 'odom'
    node.al_recibir_mapa(invalid)
    assert node.grid is None
    assert node.search_views == ()


@pytest.mark.parametrize('enabled', [False, True])
def test_plan_exhaustion_starts_extra_phase_only_when_enabled(mission, monkeypatch, enabled):
    node, clock, transform = mission
    if enabled:
        enable(node)
    node.plan = Path()
    node.candidatas = []
    events = []
    monkeypatch.setattr(node, 'explore', lambda: events.append('explore'))
    monkeypatch.setattr(node, 'return_to_start', lambda: events.append('return'))
    node.step_logic()
    clock.seconds = 12.0
    transform.header.stamp = instant(12.0).to_msg()
    node.step_logic()
    assert events == (['explore', 'explore'] if enabled else ['explore', 'return'])
    assert node.state == ('exploring' if enabled else 'returning')
    assert node.extra.phase == ('extra' if enabled else 'primary')


@pytest.mark.parametrize(('key', 'value'), [('extra_history_max', 0),
                                          ('extra_pose_max_age_s', math.nan)])
def test_invalid_history_settings_do_not_prevent_reporting_or_exploration(mission, monkeypatch, key, value):
    node, _, _ = mission
    enable(node)
    node.p[key] = value
    events = []
    monkeypatch.setattr(node, 'check_reports', lambda *args: events.append('report'))
    monkeypatch.setattr(node, 'explore', lambda: events.append('explore'))
    node.step_logic()
    assert events == ['report', 'explore']
    assert node.search_views == ()
