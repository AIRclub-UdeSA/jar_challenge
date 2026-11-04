from concurrent.futures import Future
from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from test_mision import nueva

class Pendiente:
    def submit(self,*args):return Future()


def test_meta_elegida_sin_movimiento_no_reinicia_watchdog():
    _,nav=nueva();nav.largada=(.7,.7,0.)
    m=CoordinadorMision(nav,0.,executor=Pendiente())
    m.estado='PLANIFICANDO';m.extrinseca=(0.,0.,0.)
    m.pose_progreso=(.7,.7);m.t_busqueda_total=(0.,0.);m.t_plan=0.
    m.ultima_visual=1.;m.futuro=Future()
    m.futuro.set_result(dict(vista=dict(x=1.5,y=.7,yaw=0.,sugerida=False,
            camino_m=1.,regreso_m=1.),diagnostico={}))
    e=EntradaNavegacion(1.,Pose2D(.7,.7,0.),True,[],[])
    m.paso(1.,1.,e,2)
    assert m.estado=='YENDO_VISTA' and m.t_busqueda_total==(0.,0.)
    nav.estado='FALLO';m.ultima_visual=5.;m.paso(5.,5.,e,2)
    assert m.estado=='PLANIFICANDO'
    calls=[];m.retroceder=lambda *a:calls.append(a)
    m.ultima_visual=31.;m.paso(10.,31.,e,2)
    assert calls and 'sin avance' in calls[0][1]


def test_movimiento_real_reinicia_watchdog_de_busqueda():
    _,nav=nueva();nav.largada=(.7,.7,0.)
    m=CoordinadorMision(nav,0.,executor=Pendiente())
    m.estado='YENDO_VISTA';m.extrinseca=(0.,0.,0.)
    m.pose_progreso=(.7,.7);m.t_busqueda_total=(0.,0.)
    m.vista=dict(x=1.5,y=.7,yaw=0.,regreso_m=1.)
    m.t_vista=0.;m.ultima_visual=10.;nav.estado='YENDO'
    m.paso(10.,10.,EntradaNavegacion(10.,Pose2D(.9,.7,0.),True,[],[]),2)
    assert m.t_busqueda_total is None


def test_reporte_confirmado_es_progreso_tras_pausa_quieta():
    from equipo_jar.coordinador_mision import CoordinadorMision
    from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
    from equipo_jar.navegacion.tipos import Pose2D
    from test_mision import nueva,ExecutorInmediato
    _,nav=nueva();nav.largada=(.7,.7,0.);nav.estado='ESPERANDO_META'
    m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.estado='CONFIRMANDO';m.confirmacion_incierta=True;m.t_confirmacion=0.
    m.reportes_al_confirmar=0;m.t_busqueda_total=(0.,0.)
    m.extrinseca=(0.,0.,0.);m.ultima_visual=35.
    m.percepcion(dict(reportes=1,reportadas=[dict(x=1.5,y=1.5)]))
    assert m.t_busqueda_total is None
    e=EntradaNavegacion(35.,Pose2D(.7,.7,0.),True,[],[])
    m.paso(15.,35.,e,3)
    assert m.estado=='PLANIFICANDO' and not nav.solicitud_regreso


def test_mismo_conteo_repetido_no_reinicia_watchdog_sin_progreso():
    from equipo_jar.coordinador_mision import CoordinadorMision
    from test_mision import nueva,ExecutorInmediato
    _,nav=nueva();m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.reportes=1;m.t_busqueda_total=(10.,10.)
    m.percepcion(dict(reportes=1));assert m.t_busqueda_total==(10.,10.)


def test_confirmacion_acotada_no_consum_cuenta_de_busqueda():
    from test_mision import ExecutorInmediato
    _,nav=nueva();nav.largada=(.7,.7,0.)
    m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.estado='BUSCANDO_VISTA';m.ultima_pose=(.7,.7,0.);m.pose_progreso=(.7,.7)
    m.ultimo_reloj=10.;m.ultimo_monotono_s=20.;m.t_busqueda_total=(10.,5.)
    m.extrinseca=(0.,0.,0.);m.ultima_visual=50.
    m.percepcion(dict(reportes=0,diagnostico=dict(rechazos=[dict(
        motivo='INCERTIDUMBRE',centro=[1.5,1.5])])))
    assert m.estado=='CONFIRMANDO'
    e=EntradaNavegacion(50.,Pose2D(.7,.7,0.),True,[],[])
    m.paso(25.,50.,e,3)
    assert m.estado=='PLANIFICANDO' and not nav.solicitud_regreso
    # Antes de confirmar ya se habían consumido 10 s wall/5 sim; se conservan.
    assert m.t_busqueda_total==(40.,20.)
    m.futuro=Future();m.t_plan=50.;m.ultima_visual=51.
    calls=[];m.retroceder=lambda *a:calls.append(a)
    m.paso(26.,51.,e,3)
    assert not calls  # No se regresa un tick después de la pausa.
    m.ultima_visual=71.;m.paso(30.,71.,e,3)
    assert calls  # Una búsqueda realmente estancada sí conserva su watchdog.


def test_reporte_durante_confirmacion_no_reinstala_reloj_viejo():
    from test_mision import ExecutorInmediato
    _,nav=nueva();nav.largada=(.7,.7,0.)
    m=CoordinadorMision(nav,0.,executor=ExecutorInmediato())
    m.estado='CONFIRMANDO';m.reportes_al_confirmar=0;m.t_confirmacion=10.
    m.pausa_busqueda_confirmacion=(20.,10.);m.t_busqueda_total=(10.,5.)
    m.extrinseca=(0.,0.,0.);m.ultima_visual=50.
    m.percepcion(dict(reportes=1))
    m.paso(25.,50.,EntradaNavegacion(50.,Pose2D(.7,.7,0.),True,[],[]),3)
    assert not nav.solicitud_regreso and m.pausa_busqueda_confirmacion is None
    assert m.t_busqueda_total==(50.,25.)  # Nuevo ciclo, no reloj previo restaurado.
