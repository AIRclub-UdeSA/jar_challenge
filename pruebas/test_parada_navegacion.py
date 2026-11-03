"""Exercise the actual ROS output guard with controlled clocks and messages."""
from types import SimpleNamespace

import pytest
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Empty


@pytest.fixture
def guard(monkeypatch):
    from equipo_jar.parada import Parada
    rclpy.init()
    node = Parada()
    clock = SimpleNamespace(seconds=10.0)
    monkeypatch.setattr(node, 'now', lambda: clock.seconds)
    commands, confirmed = [], []
    monkeypatch.setattr(node, 'pub_cmd', SimpleNamespace(publish=commands.append))
    monkeypatch.setattr(node, 'pub_confirmed', SimpleNamespace(publish=confirmed.append))
    try:
        yield node, clock, commands, confirmed
    finally:
        node.destroy_node()
        rclpy.shutdown()


def command():
    msg = Twist()
    msg.linear.x = 0.3
    return msg


def odom(node, speed=0.0):
    msg = Odometry()
    msg.header.stamp = node.get_clock().now().to_msg()
    msg.twist.twist.linear.x = speed
    node.on_odom(msg)
    return msg


def test_no_motion_before_map_and_stale_commands_are_never_replayed(guard):
    node, clock, commands, _ = guard
    node.on_command(command())
    node.tick()
    assert commands[-1].linear.x == 0.0
    node.on_map(OccupancyGrid())
    node.tick()
    assert commands[-1].linear.x == 0.0
    node.on_command(command())
    node.tick()
    assert commands[-1].linear.x == 0.3
    clock.seconds += 1.0
    node.tick()
    assert commands[-1].linear.x == 0.0


@pytest.mark.parametrize('trigger', ['request', 'done'])
def test_stop_is_permanent_even_if_nav2_keeps_commanding(guard, trigger):
    node, _, commands, confirmed = guard
    node.on_map(OccupancyGrid())
    node.on_command(command())
    if trigger == 'request':
        node.on_stop(Bool(data=True))
    else:
        node.on_done(Empty())
    assert commands[-1].linear.x == 0.0
    node.on_stop(Bool(data=False))
    node.on_map(OccupancyGrid())
    node.on_command(command())
    node.tick()
    assert commands[-1].linear.x == 0.0
    assert not confirmed[-1].data


def test_ack_requires_fresh_continuous_stationary_odometry_after_stop(guard):
    node, clock, _, confirmed = guard
    odom(node)
    node.on_stop(Bool(data=True))
    clock.seconds += 1.0
    node.tick()
    assert not confirmed[-1].data
    for _ in range(7):
        clock.seconds += 0.1
        odom(node)
        node.tick()
    assert confirmed[-1].data
    clock.seconds += 1.0
    node.tick()
    assert not confirmed[-1].data
    odom(node)
    node.tick()
    assert not confirmed[-1].data


def test_motion_or_invalid_odometry_resets_confirmation(guard):
    node, clock, _, confirmed = guard
    node.on_stop(Bool(data=True))
    for speed in (0.0, 0.0, 0.3, float('nan'), 0.0):
        clock.seconds += 0.2
        odom(node, speed)
        node.tick()
        assert not confirmed[-1].data


def test_repeated_or_old_sensor_stamps_do_not_confirm_a_stop(guard):
    node, clock, _, confirmed = guard
    node.on_stop(Bool(data=True))
    msg = odom(node)
    for _ in range(10):
        clock.seconds += 0.1
        node.on_odom(msg)
        node.tick()
    assert not confirmed[-1].data
    msg.header.stamp.sec = 1
    node.on_odom(msg)
    node.tick()
    assert not confirmed[-1].data


@pytest.mark.parametrize('offset_s', [-120.0, -2.0, 5.0])
def test_unsynced_robot_clock_still_confirms_from_still_odometry(guard, offset_s):
    node, clock, _, confirmed = guard
    node.on_stop(Bool(data=True))
    stamp = node.get_clock().now().nanoseconds + int(offset_s * 1e9)
    for _ in range(7):
        clock.seconds += 0.1
        stamp += 100_000_000
        msg = Odometry()
        msg.header.stamp.sec, msg.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        node.on_odom(msg)
        node.tick()
    assert confirmed[-1].data


@pytest.mark.parametrize('speed', [None, float('nan')])
def test_stop_confirms_after_timeout_without_usable_odometry(guard, speed):
    node, clock, commands, confirmed = guard
    node.on_stop(Bool(data=True))
    timeout = node.p['confirm_timeout_s']
    while clock.seconds < 10.0 + timeout - 0.15:
        clock.seconds += 0.1
        if speed is not None:
            odom(node, speed)
        node.tick()
        assert not confirmed[-1].data
    clock.seconds = 10.0 + timeout
    node.tick()
    assert confirmed[-1].data
    assert all(msg.linear.x == 0.0 for msg in commands)


def test_fresh_moving_odometry_vetoes_the_timeout_until_the_cap(guard):
    node, clock, commands, confirmed = guard
    node.on_stop(Bool(data=True))
    cap = node.p['confirm_max_s']
    while clock.seconds < 10.0 + cap - 0.15:
        clock.seconds += 0.1
        odom(node, 0.3)
        node.tick()
        assert not confirmed[-1].data
    clock.seconds = 10.0 + cap
    odom(node, 0.3)
    node.tick()
    assert confirmed[-1].data
    assert all(msg.linear.x == 0.0 for msg in commands)


@pytest.mark.parametrize('after', ['silence', 'nan'])
def test_moving_odometry_that_goes_missing_or_invalid_falls_back_to_the_timeout(guard, after):
    node, clock, _, confirmed = guard
    node.on_stop(Bool(data=True))
    clock.seconds += 0.1
    odom(node, 0.3)
    node.tick()
    if after == 'nan':
        clock.seconds += 0.1
        odom(node, float('nan'))
    clock.seconds = 10.0 + node.p['confirm_timeout_s']
    node.tick()
    assert confirmed[-1].data


def test_moving_odometry_after_the_timeout_revokes_confirmation(guard):
    node, clock, _, confirmed = guard
    node.on_stop(Bool(data=True))
    clock.seconds = 10.0 + node.p['confirm_timeout_s']
    node.tick()
    assert confirmed[-1].data
    odom(node, 0.3)
    node.tick()
    assert not confirmed[-1].data
    for _ in range(7):
        clock.seconds += 0.1
        odom(node)
        node.tick()
    assert confirmed[-1].data


def test_ros_topics_stop_mission_even_with_continuing_nav2_commands():
    import time
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.parameter import Parameter
    from equipo_jar.comun import QOS_MAPA
    from equipo_jar.mision import Mision
    from equipo_jar.parada import Parada
    from test_historial_mision import map_message

    rclpy.init()
    mission = Mision()
    node = Parada()
    probe = Node('stop_probe')
    executor = SingleThreadedExecutor()
    for item in (mission, node, probe):
        executor.add_node(item)
    mission.set_parameters([Parameter('stop_wait_s', value=0.1)])
    node.set_parameters([Parameter('still_time_s', value=0.1)])
    outputs, done = [], []
    probe.create_subscription(Twist, '/cmd_vel', outputs.append, 10)
    probe.create_subscription(Empty, '/done', done.append, 10)
    commands = probe.create_publisher(Twist, '/jar/cmd_vel_smoothed', 10)
    maps = probe.create_publisher(OccupancyGrid, '/map', QOS_MAPA)
    odometry = probe.create_publisher(Odometry, '/odom', 10)

    def spin_until(condition, timeout=3.0):
        deadline = time.monotonic() + timeout
        while not condition() and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert condition()

    def sensors():
        commands.publish(command())  # Nav2 deliberately never stops.
        msg = Odometry()
        msg.header.stamp = probe.get_clock().now().to_msg()
        msg.twist.twist.linear.x = 0.0 if node.blocked else 0.3
        odometry.publish(msg)

    probe.create_timer(0.025, sensors)
    try:
        maps.publish(map_message())
        spin_until(lambda: any(msg.linear.x > 0 for msg in outputs))
        mission.change_state('stopping')
        spin_until(lambda: bool(done))
        assert mission.state == 'finished'
        assert node.blocked
        outputs.clear()
        deadline = time.monotonic() + 0.2
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.01)
        assert outputs and all(msg.linear.x == 0.0 for msg in outputs)
        assert len(done) == 1
        assert len(probe.get_publishers_info_by_topic('/cmd_vel')) == 1
    finally:
        executor.shutdown()
        for item in (mission, node, probe):
            item.destroy_node()
        rclpy.shutdown()
