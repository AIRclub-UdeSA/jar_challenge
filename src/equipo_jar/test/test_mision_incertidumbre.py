from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from test_mision import nueva,ExecutorInmediato


def test_candidato_geometrico_incierto_pausa_sin_reportar_y_tiene_limite():
    _,nav=nueva();nav.largada=(.7,.7,0.);nav.estado='ESPERANDO_META'
    m=CoordinadorMision(nav,0.,executor=ExecutorInmediato());m.ultima_pose=(.7,.7,0.)
    m.estado='YENDO_VISTA';m.vista=dict(x=1.3,y=.7,yaw=0.,sugerida=False)
    m.percepcion(dict(reportes=0,detecciones=[],diagnostico=dict(rechazos=[
        dict(motivo='INCERTIDUMBRE',centro=[1.4,.7],error_m=.3)])))
    assert m.estado=='CONFIRMANDO' and m.reportes==0 and nav.meta is None
    m.extrinseca=(0.,0.,0.);m.ultima_visual=10.
    e=EntradaNavegacion(10.,Pose2D(.7,.7,0.),True,[],[])
    m.paso(10.,10.,e,2);assert m.estado=='CONFIRMANDO'
    m.ultima_visual=16.;m.paso(16.,16.,e,2)
    assert m.estado!='CONFIRMANDO' and m.reportes==0
