"""ROS adapter checks with generated messages; no simulator or motors."""
from concurrent.futures import Future
from types import SimpleNamespace
import math
import time

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import OccupancyGrid, Path
import pytest
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter

from equipo_jar.cobertura import Cobertura, camera_from_transform
from equipo_jar.cobertura_plan import Camera, Plan, Pose
from equipo_jar.comun import QOS_ESTADO


def map_message():
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.width = msg.info.height = 18
    msg.info.resolution = 0.1
    msg.info.origin.orientation.w = 1.0
    msg.data = [100 if i in (0, 17) or j in (0, 17) else 0
                for j in range(18) for i in range(18)]
    return msg


def put_tf(node, x=0.55, y=0.55, static=True, yaw=0.0):
    robot = TransformStamped()
    robot.header.frame_id = 'map'
    robot.child_frame_id = 'base_footprint'
    robot.transform.translation.x, robot.transform.translation.y = x, y
    robot.transform.rotation.z = math.sin(yaw / 2)
    robot.transform.rotation.w = math.cos(yaw / 2)
    if static:
        node._tf.set_transform_static(robot, 'coverage_test')
    else:
        robot.header.stamp.sec = 1
        node._tf.set_transform(robot, 'coverage_test')
    camera = TransformStamped()
    camera.header.frame_id = 'base_footprint'
    camera.child_frame_id = 'cam_1_color_optical_frame'
    camera.transform.translation.x = 0.1
    q = camera.transform.rotation
    q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    node._tf.set_transform_static(camera, 'coverage_test')


@pytest.fixture
def ros_pair():
    rclpy.init()
    coverage, peer = Cobertura(), Node('coverage_test_peer')
    executor = SingleThreadedExecutor()
    executor.add_node(coverage)
    executor.add_node(peer)
    try:
        yield coverage, peer, executor
    finally:
        executor.remove_node(coverage)
        executor.remove_node(peer)
        coverage.destroy_node()
        peer.destroy_node()
        executor.shutdown()
        rclpy.shutdown()


def spin_until(executor, condition, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        executor.spin_once(timeout_sec=0.02)
        if condition():
            return
    pytest.fail('timed out waiting for coverage state')


def test_waits_for_map(ros_pair):
    coverage, _, _ = ros_pair
    put_tf(coverage)
    coverage._tick()
    assert coverage._future is None


def test_waits_for_tf(ros_pair):
    coverage, _, _ = ros_pair
    coverage.al_recibir_mapa(map_message())
    coverage._tick()
    assert coverage._future is None
    assert coverage._planned_generation == -1


def test_plan_is_latched_and_parameters_apply_live(ros_pair):
    coverage, peer, executor = ros_pair
    put_tf(coverage)
    coverage.al_recibir_mapa(map_message())
    received = []
    peer.create_subscription(Path, '/jar/plan_cobertura', received.append, QOS_ESTADO)
    spin_until(executor, lambda: bool(received))
    assert received[0].header.frame_id == 'map'
    assert received[0].poses
    for pose in received[0].poses:
        assert pose.header.frame_id == 'map'
        q = pose.pose.orientation
        assert q.z ** 2 + q.w ** 2 == pytest.approx(1.0)

    # A late subscriber must receive the already published plan without a retry.
    late = []
    peer.create_subscription(Path, '/jar/plan_cobertura', late.append, QOS_ESTADO)
    spin_until(executor, lambda: bool(late))
    assert len(late[0].poses) == len(received[0].poses)
    first_generation = coverage._generation
    result = coverage.set_parameters_atomically([
        Parameter('max_views', value=2),
        Parameter('order_check_interval_s', value=30.0),
        Parameter('order_start_delay_s', value=5.0),
        Parameter('view_clearance', value=0.32),
        Parameter('view_preferred_clearance', value=0.35)])
    assert result.successful
    spin_until(executor, lambda: len(received) >= 2)
    assert coverage._generation == first_generation + 1
    assert 0 < len(received[-1].poses) <= 2
    assert coverage.p['order_check_interval_s'] == 30.0
    assert coverage.p['order_start_delay_s'] == 5.0
    assert coverage.p['view_clearance'] == 0.32
    assert coverage.p['view_preferred_clearance'] == 0.35
    assert coverage.pub_plan.topic_name == '/jar/plan_cobertura'
    published = {name for name, _ in coverage.get_publisher_names_and_types_by_node(
        'cobertura', '/')}
    assert not {'/cmd_vel', '/victimas', '/done'} & published


def test_invalid_parameter_transaction_does_not_change_state(ros_pair):
    coverage, _, _ = ros_pair
    old = dict(coverage.p)
    result = coverage.set_parameters_atomically([
        Parameter('robot_radius', value=-0.1), Parameter('max_views', value=1)])
    assert not result.successful
    coverage._tick()
    assert coverage.p == old
    assert coverage.get_parameter('max_views').value == old['max_views']


def test_free_threshold_update_replans_through_newly_free_cells(ros_pair):
    coverage, peer, executor = ros_pair
    put_tf(coverage)
    assert coverage.set_parameters_atomically([
        Parameter('view_range', value=0.45)]).successful
    message = map_message()
    for j in range(1, 17):
        message.data[j * 18 + 9] = 50
    received = []
    peer.create_subscription(Path, '/jar/plan_cobertura', received.append, QOS_ESTADO)
    coverage.al_recibir_mapa(message)
    spin_until(executor, lambda: bool(received))
    assert received[-1].poses
    assert all(p.pose.position.x < 0.9 for p in received[-1].poses)
    assert coverage.set_parameters_atomically([
        Parameter('free_threshold', value=60)]).successful
    spin_until(executor, lambda: len(received) >= 2)
    assert any(p.pose.position.x > 0.9 for p in received[-1].poses)


def test_repeated_map_does_not_reset_route(ros_pair):
    coverage, _, _ = ros_pair
    msg = map_message()
    coverage.al_recibir_mapa(msg)
    generation = coverage._generation
    msg.header.stamp.sec = 99
    coverage.al_recibir_mapa(msg)
    assert coverage._generation == generation


@pytest.mark.parametrize('bad', ['frame', 'quaternion', 'length'])
def test_malformed_map_is_rejected_without_crashing(ros_pair, bad):
    coverage, _, _ = ros_pair
    msg = map_message()
    if bad == 'frame':
        msg.header.frame_id = 'odom'
    elif bad == 'quaternion':
        msg.info.origin.orientation.w = 0.0
    else:
        msg.data = [0]
    coverage.al_recibir_mapa(msg)
    coverage._tick()
    assert coverage._grid is None
    assert coverage._future is None


def test_old_worker_result_is_discarded(ros_pair):
    coverage, _, _ = ros_pair
    coverage.al_recibir_mapa(map_message())
    future = Future()
    future.set_result(Plan((), 1, 0, 1, 1))
    coverage._future = future
    coverage._future_generation = coverage._generation - 1
    coverage._tick()
    assert coverage._planned_generation == -1


def finished_plan(coverage, monkeypatch, **robot):
    coverage.al_recibir_mapa(map_message())
    put_tf(coverage, **robot)
    coverage._planning_pose = Pose(0.55, 0.55, 0.0)
    coverage._planning_camera = Camera(0.1, 0.0, 0.0)
    future = Future()
    future.set_result(Plan((), 1, 0, 1, 1))
    coverage._future = future
    coverage._future_generation = coverage._generation
    published = []
    monkeypatch.setattr(coverage, '_publish', published.append)
    coverage._tick()
    return published


def test_worker_result_from_old_robot_position_is_discarded(ros_pair, monkeypatch):
    coverage, _, _ = ros_pair
    assert not finished_plan(coverage, monkeypatch, x=1.25, y=1.25)
    assert coverage._planned_generation == -1


def test_worker_result_is_kept_when_only_the_heading_changed(ros_pair, monkeypatch):
    # Mission turns in place while it waits for the plan: a heading check would throw
    # away every plan that takes longer than one turn.
    coverage, _, _ = ros_pair
    assert finished_plan(coverage, monkeypatch, yaw=math.pi / 2)
    assert coverage._planned_generation == coverage._generation


def test_map_and_parameter_updates_cancel_old_calculations(ros_pair):
    coverage, _, _ = ros_pair
    coverage._cancel_event.clear()
    coverage.al_recibir_mapa(map_message())
    assert coverage._cancel_event.is_set()
    coverage._cancel_event.clear()
    coverage.set_parameters_atomically([Parameter('max_views', value=2)])
    coverage._sync_parameters()
    assert coverage._cancel_event.is_set()


def test_destroy_node_cancels_a_running_worker():
    from concurrent.futures import CancelledError
    from threading import Event

    rclpy.init()
    coverage = Cobertura()
    started = Event()

    def work():
        started.set()
        if not coverage._cancel_event.wait(timeout=2.0):
            pytest.fail('node destruction did not cancel its worker')
        raise CancelledError()

    future = coverage._worker.submit(work)
    assert started.wait(timeout=2.0)
    try:
        coverage.destroy_node()
        with pytest.raises(CancelledError):
            future.result()
    finally:
        rclpy.shutdown()


def test_failed_start_recovers_after_localization_changes(ros_pair):
    coverage, _, executor = ros_pair
    put_tf(coverage, x=0.05, y=0.05)
    coverage.al_recibir_mapa(map_message())
    spin_until(executor, lambda: coverage._failed_start is not None)
    assert coverage._planned_generation == -1
    put_tf(coverage)
    spin_until(executor, lambda: coverage._planned_generation == coverage._generation)


def test_near_wall_start_publishes_a_translation_goal_with_unchanged_clearance(ros_pair):
    from equipo_jar.exploracion import goal_position
    from equipo_jar.mapa import Grid
    coverage, peer, executor = ros_pair
    put_tf(coverage, x=.26, y=.55)
    message = map_message()
    grid = Grid.from_occupancy_grid(message)
    assert grid.cell(.26, .55) not in grid.safe_cells(.21, 25)
    received = []
    peer.create_subscription(Path, '/jar/plan_cobertura', received.append, QOS_ESTADO)
    coverage.al_recibir_mapa(message)
    spin_until(executor, lambda: bool(received))
    first = received[-1].poses[0].pose.position
    x, y, in_place = goal_position(first.x, first.y, (.26, .55, 0), .20)
    assert not in_place
    assert (x, y) == (first.x, first.y)
    stops = grid.safe_cells(.24, 25)
    assert all(grid.cell(p.pose.position.x, p.pose.position.y) in stops for p in received[-1].poses)


def test_failed_start_retries_after_same_cell_tf_correction_with_bounded_backoff(ros_pair, monkeypatch):
    import equipo_jar.cobertura as adapter
    coverage, _, executor = ros_pair
    clock = [100.0]
    monkeypatch.setattr(adapter, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    put_tf(coverage, x=.101, y=.55)  # Nearest safe center is outside the .20 m limit.
    coverage.al_recibir_mapa(map_message())
    spin_until(executor, lambda: coverage._failed_start is not None)
    cell = coverage._grid.cell(.101, .55)
    put_tf(coverage, x=.19, y=.55)
    assert coverage._grid.cell(.19, .55) == cell
    clock[0] = 101.9
    coverage._tick()
    assert coverage._future is None
    clock[0] = 102.1
    spin_until(executor, lambda: coverage._planned_generation == coverage._generation)


def test_unchanged_pose_retries_transient_worker_failure(ros_pair, monkeypatch):
    import equipo_jar.cobertura as adapter
    coverage, _, executor = ros_pair
    clock, calls = [100.0], []
    original = adapter.make_plan
    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError('controlled transient failure')
        return original(*args, **kwargs)
    monkeypatch.setattr(adapter, 'make_plan', flaky)
    monkeypatch.setattr(adapter, 'time', SimpleNamespace(monotonic=lambda: clock[0]))
    put_tf(coverage)
    coverage.al_recibir_mapa(map_message())
    spin_until(executor, lambda: coverage._failed_start is not None)
    for _ in range(5):
        coverage._tick()
    assert len(calls) == 1
    clock[0] = 102.1
    spin_until(executor, lambda: coverage._planned_generation == coverage._generation)
    assert len(calls) == 2


def test_obsolete_worker_failure_does_not_poison_current_generation(ros_pair):
    coverage, _, _ = ros_pair
    coverage.al_recibir_mapa(map_message())
    failed = Future()
    failed.set_exception(ValueError('old start'))
    coverage._future = failed
    coverage._future_generation = coverage._generation - 1
    coverage._tick()
    assert coverage._failed_start is None


def test_small_tf_correction_cannot_publish_recovery_as_an_in_place_goal(ros_pair, monkeypatch):
    coverage, _, _ = ros_pair
    coverage.al_recibir_mapa(map_message())
    put_tf(coverage, x=.31, y=.55)
    coverage._planning_pose = Pose(.26, .55, 0)
    coverage._planning_camera = Camera(.1, 0, 0)
    future = Future()
    future.set_result(Plan((Pose(.55, .55, 0),), 1, 1, 1, 1,
                           start_anchor=Pose(.35, .55, 0)))
    coverage._future = future
    coverage._future_generation = coverage._generation
    published = []
    monkeypatch.setattr(coverage, '_publish', published.append)
    coverage._tick()
    assert not published
    assert coverage._planned_generation == -1


@pytest.mark.parametrize('name,value', [('retry_interval_s', 0.0), ('retry_interval_s', -1.0),
                                       ('retry_interval_s', math.nan), ('start_max_shift_m', -.01)])
def test_invalid_recovery_parameters_are_rejected_atomically(ros_pair, name, value):
    coverage, _, _ = ros_pair
    assert not coverage.set_parameters_atomically([Parameter(name, value=value)]).successful


def test_stale_localization_does_not_start_planning(ros_pair):
    coverage, _, _ = ros_pair
    put_tf(coverage, static=False)
    coverage.al_recibir_mapa(map_message())
    coverage._tick()
    assert coverage._future is None


def test_camera_uses_optical_z_axis():
    transform = TransformStamped().transform
    transform.translation.x = 0.1
    q = transform.rotation
    q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    camera = camera_from_transform(transform)
    assert camera.x == pytest.approx(0.1)
    assert camera.yaw == pytest.approx(0.0)
    q.x = q.y = q.z = 0.0
    q.w = 1.0
    with pytest.raises(ValueError, match='horizontal projection'):
        camera_from_transform(transform)
