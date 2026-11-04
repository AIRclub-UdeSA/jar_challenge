"""La guardia usa el scan actual transformado al centro del robot."""
import math,time
import pytest,rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
import equipo_jar.nodo_equipo as modulo
from equipo_jar.nodo_equipo import NodoEquipo


@pytest.fixture
def node(monkeypatch):
    rclpy.init()
    n=NodoEquipo()
    scan=LaserScan();scan.header.frame_id='laser_link'
    scan.header.stamp.sec=10
    scan.angle_min=-math.pi/2
    scan.range_max=10.
    scan.ranges=[.32]*60
    n.al_scan(scan);n.al_odom(Odometry())
    pose=PoseWithCovarianceStamped()
    pose.header.frame_id='map';pose.pose.pose.orientation.w=1.
    n.al_pose(pose);n.mapa=object()
    def lookup(target,*a,**k):
        tf=TransformStamped();tf.transform.rotation.w=1.
        if target=='base_footprint':tf.transform.translation.x=.043
        return tf
    monkeypatch.setattr(n.tf_buffer,'lookup_transform',lookup)
    monkeypatch.setattr(modulo,'concordancia_mapa',lambda *a,**k:(1.,60))
    monkeypatch.setattr(modulo,'concordancia_mapa_simple',lambda *a,**k:(1.,60))
    yield n
    n.destroy_node();rclpy.shutdown()


def test_pared_lateral_se_evalua_en_base_y_permite_giro(node):
    o=node.observacion(time.monotonic())
    assert o is not None and o.giro_seguro
    assert o.distancia_min<.35
    node.control.paso(0.,True,o)
    assert node.control.paso(.1,True,o)>0


def test_sin_tf_base_laser_no_habilita_aunque_mapa_tenga_tf(node,monkeypatch):
    def lookup(target,*a,**k):
        if target=='base_footprint':raise TransformException('base-laser ausente')
        result=TransformStamped();result.transform.rotation.w=1.
        return result
    monkeypatch.setattr(node.tf_buffer,'lookup_transform',lookup)
    assert node.observacion(time.monotonic()) is None
    assert node.diagnostico_observacion['etapa']=='guardia_giro'


def test_m1_lista_delega_barrido_a_m2_sin_omitir_frescura(node,monkeypatch):
    node.navegacion=object()
    node.control.estado='LISTA'
    def inesperado(*args):
        raise AssertionError('La inicialización no solicita otro giro después de entregar a M2')
    monkeypatch.setattr(modulo,'comprobar_giro',inesperado)
    obs=node.observacion(time.monotonic())
    assert obs is not None and obs.concordancia==1. and obs.giro_seguro is None
    node.t_scan=time.monotonic()-.7
    assert node.observacion(time.monotonic()) is None
    assert node.diagnostico_observacion['codigo']=='SENSORES_VENCIDOS'

def test_giro_de_inicializacion_conserva_guardia_aunque_exista_navegacion(node,monkeypatch):
    node.navegacion=object()
    node.control.estado='GIRANDO';node.control.yaw_inicial=0.
    calls=[]
    def comprobar(*args):
        calls.append(args)
        return {'seguro':False}
    monkeypatch.setattr(modulo,'comprobar_giro',comprobar)
    obs=node.observacion(time.monotonic())
    assert obs is not None and obs.giro_seguro is False and len(calls)==1


def test_filtro_vectorizado_conserva_indices_rangos_y_muestreo(node,monkeypatch):
    values=[float('inf'),float('nan'),.1,10.]+[.4+i*.001 for i in range(200)]
    node.scan.range_min=.2;node.scan.ranges=values
    expected=[(i,r) for i,r in enumerate(node.scan.ranges)
              if math.isfinite(r) and .2<=r<10.]
    captured=[]
    def concordancia(mapa,muestra,*args,**kwargs):
        captured.extend(muestra)
        return 1.,len(muestra)
    monkeypatch.setattr(modulo,'concordancia_mapa',concordancia)
    obs=node.observacion(time.monotonic())
    assert captured==expected[::max(1,len(expected)//node.valor('muestra_rayos'))]
    assert obs is not None and obs.distancia_min==min(r for _,r in expected)


def test_recuperacion_horaria_refleja_barrido_y_conserva_geometria_base(node,monkeypatch):
    node.control.recuperacion_activa=True
    node.control.secuencia_lista=1
    node.control.reinicios_locales=1  # próximo intento gira en sentido horario
    node.control.estado='RECUPERANDO'
    calls=[]
    original=modulo.comprobar_giro
    def comprobar(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(modulo,'comprobar_giro',comprobar)
    o=node.observacion(time.monotonic())
    assert o is not None and calls
    assert calls[0][0][0][0]==pytest.approx(.043)
    assert calls[0][0][0][1]==pytest.approx(.32)
    assert calls[0][4]==pytest.approx(math.radians(35)+.12)
    assert node.diagnostico_observacion['guardia_giro']['direccion']==-1


def test_guardia_recuperacion_descuenta_progreso_en_sentido_horario(node,monkeypatch):
    node.control.recuperacion_activa=True
    node.control.secuencia_lista=1
    node.control.estado='GIRANDO';node.control.giro_activo=True
    node.control.direccion_activa=-1;node.control.yaw_inicial=0.
    node.odom.pose.pose.orientation.z=math.sin(-.2/2)
    node.odom.pose.pose.orientation.w=math.cos(-.2/2)
    calls=[];original=modulo.comprobar_giro
    monkeypatch.setattr(modulo,'comprobar_giro',lambda *a:(calls.append(a) or original(*a)))
    assert node.observacion(time.monotonic()) is not None
    assert calls[0][4]==pytest.approx(math.radians(35)-.2+.12)
