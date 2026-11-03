"""La selección usa accesibilidad necesaria; M2 autoriza el movimiento."""
import numpy as np
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.costos import ParametrosCosto
from equipo_jar.navegacion.guardia import huella_convexa_x3
from equipo_jar.coordinador_mision import resolver_busqueda
from equipo_jar.mision import ParametrosMision

class OtraSala:
    def ganancia(self, pose):
        return 50 if pose[0]>3.2 and abs(pose[2])<.01 else 0

def caso(cerrado=False):
    occ=np.ones((60,100),bool)
    occ[8:52,8:30]=False;occ[8:52,60:92]=False;occ[26:34,30:60]=False
    if cerrado:occ[:,45:47]=True
    g=Grilla(.05,0.,0.,0.,100,60,occ)
    fp=huella_convexa_x3(-.19,.13,.07,.08,.0845,.033,.04,.018)
    fp['heading_uncertainty_rad']=.1
    pc=ParametrosCosto(.12,fp['radius_m'],.04,.25,5.)
    return (g,pc,np.zeros_like(occ),(.9,1.5,0.),(.9,1.5,0.),OtraSala(),set(),600.,ParametrosMision(),[],(0.,0.,0.),fp)

def test_corredor_real_admisible_no_se_excluye_por_disco_circunscrito(monkeypatch):
    def prohibido(*a,**k):raise AssertionError('No duplicar planificación M2 en selector')
    monkeypatch.setattr('equipo_jar.coordinador_mision.planificar_orientado',prohibido)
    args=caso();r=resolver_busqueda(*args)
    assert r['vista'] and r['vista']['x']>3.2
    assert r['vista']['camino_m']>2.
    assert not r['vista'].get('validacion_se2',False)
    assert r['diagnostico']['sondeos']==[]
    assert args[-1]['margin_m']==.04

def test_barrera_no_conectada_no_se_propone():
    r=resolver_busqueda(*caso(True))
    assert r['vista'] is None

def test_reserva_de_regreso_se_conserva():
    a=list(caso());a[7]=20.
    assert resolver_busqueda(*a)['vista'] is None


def test_m2_valida_la_ruta_real_propuesta_por_el_selector():
    from equipo_jar.navegacion.orientado import planificar_orientado
    from equipo_jar.navegacion.tipos import Pose2D
    a=caso();g,pc,extra,pose,start=a[:5];fp=a[-1]
    v=resolver_busqueda(*a)['vista'];xy=(v['x'],v['y'])
    ida=planificar_orientado(g,g.ocupada,Pose2D(*pose),xy,fp,rumbo_meta=v['yaw'],parametros_costos=pc)
    assert ida is not None
    vuelta=planificar_orientado(g,g.ocupada,Pose2D(*xy,v['yaw']),start[:2],fp,parametros_costos=pc)
    assert vuelta is not None
    assert fp['margin_m']==.04
