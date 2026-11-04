from concurrent.futures import Future
from equipo_jar.coordinador_mision import CoordinadorMision
from equipo_jar.mision import elegir_vista,ParametrosMision
from equipo_jar.navegacion.ida_vuelta import EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from test_mision import nueva

class Pendiente:
    def submit(self,*args):return Future()

def preparada():
    _,nav=nueva();nav.largada=(.7,.7,0.);nav.estado='ESPERANDO_META'
    m=CoordinadorMision(nav,0.,executor=Pendiente());m.extrinseca=(0.,0.,0.);m.ultima_visual=1.
    v=dict(x=1.3,y=.7,yaw=0.,sugerida=True,regreso_m=.6,mirar_x=1.6,mirar_y=.7)
    m.estado='PLANIFICANDO';m.futuro=Future();m.futuro.set_result(dict(vista=v,diagnostico={}))
    e=EntradaNavegacion(1.,Pose2D(.7,.7,0.),True,[],[])
    m.paso(1.,1.,e,2)
    return m,nav,e,v

def test_fallo_navegando_no_gasta_inspeccion_de_candidato():
    m,nav,e,v=preparada()
    assert m.intentos_objetivo(v)==0
    nav.estado='FALLO';m.paso(2.,2.,e,2)
    assert m.intentos_objetivo(v)==0
    assert not m.finalizar

def test_meta_alcanzada_cuenta_una_inspeccion_y_no_cada_callback():
    m,nav,e,v=preparada();assert m.intentos_objetivo(v)==0
    nav.estado='EN_META';m.paso(2.,2.,e,2)
    assert m.estado=='OBSERVANDO' and m.intentos_objetivo(v)==1
    m.paso(2.1,2.1,e,2)
    assert m.intentos_objetivo(v)==1

def test_fallo_descarta_vecindad_con_otro_rumbo_y_conserva_otras_vistas():
    _,nav=nueva();m=CoordinadorMision(nav,0.,executor=Pendiente())
    m.vista=dict(x=1.281,y=1.281,yaw=2.,sugerida=True);m._descartar(fallo=True)
    near=dict(x=1.289,y=1.298,yaw=2.1)
    other=dict(x=1.8,y=1.28,yaw=2.)
    class SinGanancia:
        def ganancia(self,pose):return 0
    v=elegir_vista(nav.grilla,nav.costos,(.7,.7,0.),(.7,.7,0.),SinGanancia(),m.descartadas,600.,ParametrosMision(),[near,other])
    assert v is not None and v['x']==other['x']
