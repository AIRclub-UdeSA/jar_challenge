"""El diagnóstico distingue sensores, pose y TF sin relajar el rechazo."""
import time

import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException

from equipo_jar.nodo_equipo import NodoEquipo


@pytest.fixture
def node():
    rclpy.init()
    n = NodoEquipo()
    scan = LaserScan()
    scan.range_max = 10.
    scan.ranges = [1.]*60
    n.al_scan(scan)
    n.al_odom(Odometry())
    pose = PoseWithCovarianceStamped()
    pose.header.frame_id = 'map'
    pose.pose.pose.orientation.w = 1.
    n.al_pose(pose)
    n.mapa = object()
    yield n
    n.destroy_node()
    rclpy.shutdown()


def test_sensor_vencido_informa_edades_y_no_consulta_tf(node, monkeypatch):
    now = time.monotonic()
    node.t_scan = now-.7
    monkeypatch.setattr(node.tf_buffer, 'lookup_transform',
                        lambda *a, **k: pytest.fail('No consultar TF con scan vencido'))
    assert node.observacion(now) is None
    assert node.diagnostico_observacion['codigo'] == 'SENSORES_VENCIDOS'
    assert node.diagnostico_observacion['edad_scan_s'] == pytest.approx(.7)


def test_pose_vencida_es_distinta_de_sensor_vencido(node):
    node.t_pose = time.monotonic()-4.
    assert node.observacion(time.monotonic()) is None
    assert node.diagnostico_observacion['codigo'] == 'POSE_VENCIDA'


def test_tf_ausente_con_datos_frescos_conserva_detalle(node, monkeypatch):
    def missing(*a, **k):
        raise TransformException('extrapolacion de prueba')
    monkeypatch.setattr(node.tf_buffer, 'lookup_transform', missing)
    assert node.observacion(time.monotonic()) is None
    assert node.diagnostico_observacion['codigo'] == 'TF_NO_DISPONIBLE'
    assert 'extrapolacion' in node.diagnostico_observacion['detalle']


def test_dato_ausente_reemplaza_diagnostico_previo(node):
    node.diagnostico_observacion = {'codigo': 'TF_NO_DISPONIBLE', 'detalle': 'viejo'}
    node.scan = None
    assert node.observacion(time.monotonic()) is None
    assert node.diagnostico_observacion == {'codigo': 'DATOS_AUSENTES', 'faltantes': ['scan']}
