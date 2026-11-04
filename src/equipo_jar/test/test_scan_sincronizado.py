"""Scans y TF asíncronos sin extrapolar ni prolongar la frescura."""
import time
import pytest
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
import equipo_jar.nodo_equipo as modulo
from equipo_jar.nodo_equipo import NodoEquipo


def scan(stamp, distance=1.):
    message=LaserScan()
    message.header.frame_id='laser_link'
    message.header.stamp.sec=stamp
    message.range_max=10.
    message.ranges=[distance]*60
    return message


@pytest.fixture
def node(monkeypatch):
    rclpy.init()
    n=NodoEquipo()
    n.al_odom(Odometry())
    pose=PoseWithCovarianceStamped()
    pose.header.frame_id='map'
    pose.pose.pose.orientation.w=1.
    pose.pose.covariance[0]=pose.pose.covariance[7]=pose.pose.covariance[35]=.001
    n.al_pose(pose)
    n.mapa=object()
    monkeypatch.setattr(modulo,'concordancia_mapa',lambda *a,**k:(1.,60))
    monkeypatch.setattr(modulo,'concordancia_mapa_simple',lambda *a,**k:(1.,60))
    yield n
    n.destroy_node()
    rclpy.shutdown()


def buffer(node,monkeypatch,ready):
    calls=[]
    def lookup(target,source,stamp):
        if target=='base_footprint':
            result=TransformStamped()
            result.transform.rotation.w=1.
            return result
        sec=stamp.nanoseconds/1e9
        calls.append(sec)
        if sec not in ready:
            raise TransformException('TF aun no llega para '+str(sec))
        result=TransformStamped()
        result.transform.rotation.w=1.
        return result
    monkeypatch.setattr(node.tf_buffer,'lookup_transform',lookup)
    return calls


def test_scan_actual_adelantado_usa_anterior_a_su_estampa_y_guardia_actual(node,monkeypatch):
    node.al_scan(scan(10))
    node.al_scan(scan(11,.2))
    calls=buffer(node,monkeypatch,{10})
    observation=node.observacion(time.monotonic())
    assert observation is not None
    assert calls==[11,10]
    assert observation.distancia_min==pytest.approx(.2)
    assert node.diagnostico_observacion['scan_tf_stamp_s']==10


def test_si_tf_actual_llego_prefiere_scan_mas_reciente(node,monkeypatch):
    node.al_scan(scan(10))
    node.al_scan(scan(11))
    calls=buffer(node,monkeypatch,{10,11})
    assert node.observacion(time.monotonic()) is not None
    assert calls==[11]


def test_no_reutiliza_scan_anterior_vencido_aunque_tenga_tf(node,monkeypatch):
    node.al_scan(scan(10))
    node.al_scan(scan(11))
    previous,receipt=node.historial_scan[0]
    node.historial_scan[0]=(previous,time.monotonic()-.7)
    calls=buffer(node,monkeypatch,{10})
    assert node.observacion(time.monotonic()) is None
    assert calls==[11]
    assert node.diagnostico_observacion['codigo']=='TF_NO_DISPONIBLE'


def test_sin_tf_en_historial_rechaza_y_conserva_detalle(node,monkeypatch):
    node.al_scan(scan(10))
    node.al_scan(scan(11))
    calls=buffer(node,monkeypatch,set())
    assert node.observacion(time.monotonic()) is None
    assert calls==[11,10]
    assert node.diagnostico_observacion['codigo']=='TF_NO_DISPONIBLE'


def test_replay_y_scan_retrasado_no_renuevan_recibo_ni_historial(node):
    node.al_scan(scan(10))
    receipt=node.t_scan
    node.al_scan(scan(10,.2))
    node.al_scan(scan(9,.2))
    assert node.scan.header.stamp.sec==10
    assert node.t_scan==receipt
    assert len(node.historial_scan)==1


def test_historial_es_acotado(node):
    for stamp in range(20):
        node.al_scan(scan(stamp))
    assert len(node.historial_scan)==8
    assert node.historial_scan[0][0].header.stamp.sec==12


def test_scan_actual_invalido_no_se_oculta_con_historial_valido(node,monkeypatch):
    node.al_scan(scan(10))
    node.al_scan(scan(11,float('nan')))
    buffer(node,monkeypatch,{10})
    assert node.observacion(time.monotonic()) is None
    assert node.diagnostico_observacion['codigo']=='SCAN_INSUFICIENTE'


@pytest.mark.parametrize('expired',['scan','pose','tf'])
def test_calidad_reutilizable_no_sustituye_vigencia_de_datos(node,monkeypatch,expired):
    from types import SimpleNamespace
    calls=[]
    def cached(*args):
        calls.append(1)
        return (1.,60),{'concordancia_simple':1.}
    node.evaluador_calidad=SimpleNamespace(calcular=cached)
    node.al_scan(scan(10))
    buffer(node,monkeypatch,{10})
    now=time.monotonic()
    assert node.observacion(now) is not None
    assert len(calls)==1
    if expired=='scan':
        node.t_scan=now-node.valor('frescura_sensores_s')-1.
    elif expired=='pose':
        node.t_pose=now-node.valor('frescura_pose_s')-1.
    else:buffer(node,monkeypatch,set())
    assert node.observacion(now) is None
    assert len(calls)==1
