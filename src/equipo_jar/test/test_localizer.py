"""El localizer sin lo que el simulador siempre da: IMU sana y un solo reloj (el software puede correr en otra compu que el robot)."""
from pathlib import Path
from types import SimpleNamespace

import yaml

from equipo_jar.localizer import turn_source

PARAMS = Path(__file__).parents[1] / 'params' / 'params.yaml'


def test_turn_falls_back_to_odom_without_imu_or_when_reversed():
    check = yaml.safe_load(PARAMS.read_text())['localizer']['ros__parameters']['imu_check']
    assert turn_source(0.20, 0.16, check) == 'imu'
    assert turn_source(0.20, None, check) == 'odom'     # no llega la IMU o falta su TF
    assert turn_source(0.20, -0.16, check) == 'odom'    # montada distinto de lo que dice su TF
    assert turn_source(0.01, -0.01, check) == 'imu'     # derecho: los dos son ruido


def test_tf_uses_the_sensor_clock():
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.duration import Duration
    from rclpy.parameter import Parameter
    from rclpy.time import Time

    from equipo_jar.localizer import Localizer

    rclpy.init(args=['--ros-args', '--params-file', str(PARAMS)])
    try:
        node = Localizer()
        sent = []
        node.tf_pub = SimpleNamespace(sendTransform=sent.append)
        robot = node.get_clock().now() - Duration(seconds=3600) # el robot con el reloj una hora atrasado
        odom = Odometry()
        odom.header.stamp = robot.to_msg()
        odom.pose.pose.orientation.w = 1.0

        node.publish_tf()
        assert not sent # antes del primer /odom no sabe la hora del robot
        node.set_parameters([Parameter('use_sim_time', value=True)]) # en el simulador, la hora es 0 hasta el primer /clock
        node.on_odom(odom)
        node.publish_tf()
        assert not sent
        node.set_parameters([Parameter('use_sim_time', value=False)])

        node.on_odom(odom)
        node.publish_tf()
        lead = (Time.from_msg(sent[-1].header.stamp) - robot).nanoseconds * 1e-9
        assert abs(lead - node.tf_lead) < 0.1
    finally:
        rclpy.shutdown()
