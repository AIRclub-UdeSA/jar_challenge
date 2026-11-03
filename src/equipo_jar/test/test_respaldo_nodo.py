import json,time
from dataclasses import replace
import pytest,rclpy
from rclpy.parameter import Parameter
from std_msgs.msg import String,Empty
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from equipo_jar.nodo_percepcion import NodoPercepcion
from equipo_jar.nodo_equipo import NodoEquipo
from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.percepcion import Deteccion
from test_percepcion_nodo import Captura,preparar,nube
from test_nav_nodo import M1Lista
from test_mision import nueva
from equipo_jar.localizacion_activa import Observacion
from equipo_jar.navegacion.tipos import Pose2D

@pytest.fixture
def node():
    rclpy.init()
    n=NodoPercepcion(parameter_overrides=[Parameter('cantidad_victimas',value=3),
        Parameter('error_reporte_max_m',value=.5)])
    n.pub_reportes=Captura();n.pub_estado=Captura();preparar(n)
    yield n
    n.destroy_node();rclpy.shutdown()


def pending(node):
    assert not node.registro.observar([Deteccion(1.,.2,.217,(.2,.2),80,(.9,.2))],1.)


def test_antes_del_plazo_se_guarda_y_al_plazo_se_publica_mejor(node):
    pending(node);node.cerrar_rafaga();assert not node.pub_reportes.mensajes
    node.inicio_mapa_s=node.get_clock().now().nanoseconds/1e9-271.
    node.cerrar_rafaga()
    assert len(node.pub_reportes.mensajes)==1
    status=json.loads(node.pub_estado.mensajes[-1].data)
    assert status['reportes_respaldo']==1 and status['pendientes_reportes']==0
    assert status['causa_respaldo']=='PLAZO_BONO'
    node.cerrar_rafaga();assert len(node.pub_reportes.mensajes)==1


def test_regreso_vacia_sin_esperar_plazo_y_done_no_duplica(node):
    pending(node);node.al_mision(String(data=json.dumps(dict(estado='REGRESANDO'))))
    assert len(node.pub_reportes.mensajes)==1
    node.al_done(Empty());node.al_mision(String(data=json.dumps(dict(estado='FINALIZADA'))))
    assert len(node.pub_reportes.mensajes)==1


def test_solo_observar_no_publica_respaldo(node):
    pending(node);node.set_parameters([Parameter('solo_observar',value=True)])
    node.al_mision(String(data=json.dumps(dict(estado='REGRESANDO'))))
    assert not node.pub_reportes.mensajes and node.registro.reportes==0


def test_mapa_republicado_no_reinicia_plazo(node):
    node.inicio_mapa_s-=100.;old=node.inicio_mapa_s
    preparar(node);assert node.inicio_mapa_s==old


def test_log_de_deteccion_incluye_superficie_y_calidad(node):
    node.al_nube(nube(node))
    state=json.loads(node.pub_estado.mensajes[-1].data)
    assert state['detecciones'][0]['superficie'] is not None
    assert state['detecciones'][0]['calidad_preferida']


def test_done_espera_acuse_del_reporte_de_respaldo(node):
    eq=NodoEquipo(parameter_overrides=[Parameter('localizacion_activa',value=False)])
    try:
        _,nav=nueva();nav.largada=(.7,.7,0.);nav.ultima_pose=Pose2D(.7,.7,0.)
        eq.navegacion=nav;eq.mision=CoordinadorMision(nav,0.);eq.mision.estado='FINALIZADA'
        eq.mision.finalizar=True;eq.mision.pendientes_reportes=1
        eq.mapa=object();eq.control=M1Lista();eq.odom=Odometry()
        eq.observacion=lambda now:Observacion(0.,(.7,.7,0.),1,.01,.01,1.,1.,True)
        eq.paso_navegacion=lambda *a:Twist()
        eq.pub_done=Captura();eq.pub_cmd_vel=Captura();eq.pub_mision=Captura()
        eq.t_fin_quieto=time.monotonic()-.8
        eq.ciclo_de_control();assert not eq.pub_done.mensajes
        pending(node);node.al_mision(String(data=json.dumps(dict(estado='FINALIZADA'))))
        eq.al_percepcion_mision(node.pub_estado.mensajes[-1])
        eq.ciclo_de_control()
        assert len(node.pub_reportes.mensajes)==1 and len(eq.pub_done.mensajes)==1
    finally:eq.destroy_node()
