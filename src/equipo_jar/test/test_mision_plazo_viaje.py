from concurrent.futures import Future
from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from test_mision import nueva


class Pendiente:
    def submit(self,*args):return Future()


def mision(largo):
    _,nav=nueva();nav.largada=(.7,.7,0.);nav.estado='YENDO'
    m=CoordinadorMision(nav,0.,executor=Pendiente())
    m.estado='YENDO_VISTA';m.extrinseca=(0.,0.,0.);m.t_vista=0.
    m.vista=dict(x=1.5,y=.7,yaw=0.,camino_m=largo,regreso_m=1.,sugerida=False)
    return m,nav,EntradaNavegacion(100.,Pose2D(.7,.7,0.),True,[],[])


def test_viaje_largo_no_se_cancela_al_minimo_de_90_segundos():
    m,nav,e=mision(13.7);m.ultima_visual=100.
    m.paso(100.,100.,e,2)
    assert m.estado=='YENDO_VISTA' and nav.estado=='YENDO'
    m.ultima_visual=200.;m.paso(200.,200.,e,2)
    assert nav.estado=='FALLO' and m.estado=='PLANIFICANDO'


def test_viaje_corto_conserva_watchdog_minimo():
    m,nav,e=mision(.5);m.ultima_visual=100.
    m.paso(100.,100.,e,2)
    assert nav.estado=='FALLO' and m.estado=='PLANIFICANDO'
