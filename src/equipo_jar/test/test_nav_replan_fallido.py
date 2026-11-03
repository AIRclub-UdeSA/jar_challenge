"""Un plan de regreso fallido mantiene cero y reintenta sin usar ruta anterior."""
from concurrent.futures import Future
import numpy as np
from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D

def test_regreso_asincrono_sin_ruta_no_usa_seguidor_sin_inicializar():
    world=Mundo(np.zeros((60,100),dtype=bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    pose=Pose2D(1.,1.5,0.);base,points=world.puntos(pose)
    nav.largada=(.5,1.5,0.);nav.meta=(1.,1.5)
    nav.estado='VOLVIENDO';nav.replanificaciones=1
    nav.t_movimiento=0.
    future=Future();future.set_result((None,None))
    nav.plan_pendiente=(future,'VOLVIENDO',True)
    assert nav.paso(EntradaNavegacion(0.,pose,True,base,points)).es_cero()
    assert nav.estado=='BLOQUEADO'
    for k in range(1,10):
        assert nav.paso(EntradaNavegacion(k*.1,pose,True,base,points)).es_cero()
        assert nav.estado=='BLOQUEADO'
    nav._planificar=lambda *args:False
    for k in range(10,350):
        assert nav.paso(EntradaNavegacion(k*.1,pose,True,base,points)).es_cero()
        if nav.estado=='FALLO':break
    assert nav.estado=='FALLO'

def test_guardia_conserva_obstaculo_cercano_a_pared_que_capa_normal_omite():
    world=Mundo(np.zeros((60,100),dtype=bool))
    world.grilla.ocupada[30,30]=True
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    point=(1.45,1.52);cell=world.grilla.a_celda(*point)
    nav.capa.actualizar(0.,[point])
    assert not nav.capa.ocupadas(0.)[cell]
    nav.capa.registrar_bloqueo(1.,[point])
    assert nav.capa.ocupadas(1.)[cell]
    assert not world.grilla.ocupada[cell]

def test_seguidor_nuevo_se_puede_detener_sin_ruta_ni_llegada_falsa():
    from equipo_jar.navegacion.seguimiento import Seguidor
    follower=Seguidor(P_SEG)
    follower.detener()
    assert follower.paso(0.,Pose2D(1.,1.,0.),.5).es_cero()
    assert not follower.llego

def test_perder_localizacion_durante_primer_plan_de_vuelta_descarta_todo():
    from equipo_jar.navegacion.planificador import Ruta
    world=Mundo(np.zeros((60,100),dtype=bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.largada=(.5,1.5,0.);nav.meta=(1.,1.5)
    nav.ruta=Ruta([(.5,1.5),(1.,1.5)],.5,.5)
    nav.estado='VOLVIENDO'
    future=Future()
    nav.plan_pendiente=(future,'VOLVIENDO',False)
    nav.invalidar_plan_por_localizacion()
    assert nav.ruta is None and nav.plan_pendiente is None
    assert future.cancelled()
    assert not nav.seguidor_vuelta.llego
    assert nav.paso(EntradaNavegacion(0.,None,False,None,None)).es_cero()
    assert nav.estado=='DETENIDO'
