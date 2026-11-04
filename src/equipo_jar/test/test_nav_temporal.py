"""Odometría continua y LiDAR compensado entre captura y control."""
import math
import time

import pytest
import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan

from equipo_jar.localizacion_activa import Observacion
from equipo_jar.nodo_equipo import NodoEquipo


def transform(parent, child, stamp=0, x=0., y=0., yaw=0.):
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.header.stamp.sec = stamp
    t.transform.translation.x, t.transform.translation.y = float(x), float(y)
    t.transform.rotation.z = math.sin(yaw/2)
    t.transform.rotation.w = math.cos(yaw/2)
    return t


def scan(stamp):
    s = LaserScan()
    s.header.frame_id, s.header.stamp.sec = 'laser_link', stamp
    s.range_min, s.range_max, s.ranges = .05, 10., [1.] * 60
    return s


@pytest.fixture
def node():
    rclpy.init()
    n = NodoEquipo()
    n.tf_buffer.set_transform_static(transform('map','odom',x=10.,y=20.,yaw=math.pi/2),'test')
    n.tf_buffer.set_transform_static(transform('base_footprint','laser_link'),'test')
    n.tf_buffer.set_transform(transform('odom','base_footprint',8,x=1.,y=1.),'test')
    n.tf_buffer.set_transform(transform('odom','base_footprint',10,x=2.,y=1.),'test')
    odom=Odometry();odom.header.stamp.sec=10
    n.al_odom(odom)
    yield n
    n.destroy_node()
    rclpy.shutdown()


def observation():
    # La pose AMCL retenida es intencionalmente distinta a la pose TF actual.
    return Observacion(0.,(1.,1.,0.),1,.01,.01,1.,1.,True)


def test_control_usa_odom_actual_y_compensa_scan_antiguo(node):
    node.al_scan(scan(8))
    pose, puntos, mapa = node.datos_navegacion(time.monotonic(), observation())
    assert (pose.x,pose.y,pose.yaw) == pytest.approx((9.,22.,math.pi/2))
    assert puntos[0] == pytest.approx((0.,0.))
    assert mapa[0] == pytest.approx((9.,22.))
    assert node.diagnostico_observacion['nav_control_stamp_s'] == 10.
    assert node.diagnostico_observacion['nav_scan_stamp_s'] == 8.


def test_scan_futuro_sin_tf_prefiere_el_anterior_fresco(node):
    node.al_scan(scan(8));node.al_scan(scan(11))
    assert node.datos_navegacion(time.monotonic(), observation()) is not None
    assert node.diagnostico_observacion['nav_scan_stamp_s'] == 8.


def test_scan_vencido_o_sin_tf_no_habilita_movimiento(node):
    node.al_scan(scan(8))
    message,_=node.historial_scan[0]
    node.historial_scan[0]=(message,time.monotonic()-.7)
    assert node.datos_navegacion(time.monotonic(), observation()) is None
    node.historial_scan.clear();node.al_scan(scan(11))
    assert node.datos_navegacion(time.monotonic(), observation()) is None


def test_sin_observacion_m1_no_se_usa_pose_tf(node):
    node.al_scan(scan(8))
    assert node.datos_navegacion(time.monotonic(), None) is None


def test_tf_retenido_no_se_rejuvenece_con_odom_nuevo(node):
    node.al_scan(scan(8))
    nuevo=Odometry();nuevo.header.stamp.sec=12
    node.al_odom(nuevo)
    assert node.datos_navegacion(time.monotonic(), observation()) is None
    assert 'vigencia' in node.diagnostico_observacion['nav_tf_error']
