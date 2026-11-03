"""Redondeo de varianzas casi nulas no invalida una pose utilizable."""
import json
import math
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import equipo_jar.nodo_equipo as modulo
from equipo_jar.nodo_equipo import NodoEquipo


@pytest.fixture
def node(monkeypatch):
    rclpy.init()
    n=NodoEquipo()
    scan=LaserScan()
    scan.header.frame_id='laser_link'
    scan.range_max=10.
    scan.ranges=[1.]*60
    n.al_scan(scan)
    odom=Odometry()
    odom.pose.pose.orientation.w=1.
    n.al_odom(odom)
    pose=PoseWithCovarianceStamped()
    pose.header.frame_id='map'
    pose.pose.pose.orientation.w=1.
    pose.pose.covariance[0]=.001
    pose.pose.covariance[7]=.001
    pose.pose.covariance[35]=.001
    n.al_pose(pose)
    n.mapa=object()
    transform=TransformStamped()
    transform.transform.rotation.w=1.
    monkeypatch.setattr(n.tf_buffer,'lookup_transform',lambda *a,**k:transform)
    monkeypatch.setattr(modulo,'concordancia_mapa',lambda *a,**k:(1.,60))
    monkeypatch.setattr(modulo,'concordancia_mapa_simple',lambda *a,**k:(1.,60))
    yield n
    n.destroy_node()
    rclpy.shutdown()


def observe(n):
    import time
    return n.observacion(time.monotonic())


def test_covarianzas_capturadas_de_amcl_no_rechazan_pose_valida(node):
    # Solo valores de varianza del testigo real; no posiciones del mundo.
    node.pose.pose.covariance[0]=-4.824699667560495e-17
    node.pose.pose.covariance[7]=-5.993469609499869e-16
    node.pose.pose.covariance[35]=4.085620730620618e-14
    observation=observe(node)
    assert observation is not None
    assert observation.std_xy==0.
    assert observation.std_yaw==pytest.approx(math.sqrt(4.085620730620618e-14))
    assert node.control.calidad(observation)


@pytest.mark.parametrize('index',[0,7,35])
def test_negativo_minimo_equivale_a_cero_sin_mutar_el_mensaje(node,index):
    node.pose.pose.covariance[index]=-1e-13
    rounded=observe(node)
    assert rounded is not None
    assert node.pose.pose.covariance[index]==-1e-13
    node.pose.pose.covariance[index]=0.
    exact=observe(node)
    assert rounded.std_xy==exact.std_xy
    assert rounded.std_yaw==exact.std_yaw


@pytest.mark.parametrize('value',[-1e-8,-1e-4])
def test_covarianza_realmente_negativa_sigue_rechazada(node,value):
    node.pose.pose.covariance[35]=value
    assert observe(node) is None
    assert node.diagnostico_observacion['codigo']=='POSE_INVALIDA'


@pytest.mark.parametrize('value',[math.nan,math.inf,-math.inf])
def test_no_finito_sigue_rechazado_y_diagnostico_serializable(node,value):
    node.pose.pose.covariance[0]=value
    assert observe(node) is None
    json.dumps(node.diagnostico_observacion,allow_nan=False)
