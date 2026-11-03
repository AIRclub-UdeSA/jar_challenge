"""Memoria duradera, liberación con rayos y frescura de observación."""
import math
import numpy as np
from test_nav_ida_vuelta import Mundo,P_COSTO,P_SEG,HUELLA,P
from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
from equipo_jar.navegacion.costos import CapaViva,mapa_costos
from equipo_jar.navegacion.tipos import Pose2D

def capa():
    world=Mundo(np.zeros((80,100),dtype=bool))
    _,distance=mapa_costos(world.grilla,P_COSTO)
    layer=CapaViva(world.grilla,distance,600.,.1)
    point=(1.5,1.)
    layer.actualizar(0.,[point])
    return world,layer,world.grilla.a_celda(*point)

def fan(distance,missing=False):
    angles=np.arange(-.15,.151,.005)
    if missing:angles=angles[np.abs(angles)>.04]
    return [(.5+distance*math.cos(a),1.+distance*math.sin(a)) for a in angles]

def test_obstaculo_recordado_al_volver_despues_de_diez_segundos():
    world=Mundo(np.zeros((80,100),dtype=bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.capa.actualizar(0.,[(1.5,1.)])
    assert nav.capa.ocupadas(30.)[world.grilla.a_celda(1.5,1.)]

def test_rayos_que_muestran_espacio_libre_liberan_la_celda():
    world,layer,cell=capa()
    layer.actualizar(1.,fan(2.),origen=(.5,1.),incremento_angular=.005)
    assert not layer.ocupadas(1.)[cell]

def test_no_se_borra_lo_que_esta_oculto_por_otro_obstaculo():
    world,layer,cell=capa()
    layer.actualizar(1.,fan(.5),origen=(.5,1.),incremento_angular=.005)
    assert layer.ocupadas(1.)[cell]

def test_hueco_angular_no_es_evidencia_de_espacio_libre():
    world,layer,cell=capa()
    layer.actualizar(1.,fan(2.,missing=True),origen=(.5,1.),incremento_angular=.005)
    assert layer.ocupadas(1.)[cell]

def test_scan_retenido_y_datos_invalidos_no_renuevan_la_marca():
    world=Mundo(np.zeros((80,100),dtype=bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    pose=Pose2D(.5,1.,0.);points=[(1.5,1.)]
    for now,stamp,valid in [(0.,100.,True),(2.,100.,True),(3.,101.,False)]:
        nav.paso(EntradaNavegacion(now,pose,valid,[(1.,0.)],points,None,(.5,1.),.005,stamp))
    assert nav.capa.marca[world.grilla.a_celda(1.5,1.)]==0.
    nav.paso(EntradaNavegacion(4.,pose,True,[(1.,0.)],points,None,(.5,1.),.005,101.))
    assert nav.capa.marca[world.grilla.a_celda(1.5,1.)]==4.

def test_celda_entera_libre_no_conserva_halo_de_una_celda():
    world,layer,cell=capa()
    # Centro antiguo a ~1,025 m: toda su circunferencia de 3,54 cm queda
    # antes del retorno de 1,08 m; el retorno actual cae en la celda siguiente.
    layer.actualizar(1.,fan(1.08),origen=(.5,1.),incremento_angular=.005)
    assert not layer.ocupadas(1.)[cell]

def test_scan_anterior_al_ultimo_aceptado_tampoco_renueva_la_marca():
    world=Mundo(np.zeros((80,100),dtype=bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    pose=Pose2D(.5,1.,0.);points=[(1.5,1.)]
    for now,stamp in [(0.,101.),(2.,100.)]:
        nav.paso(EntradaNavegacion(now,pose,True,[(1.,0.)],points,None,(.5,1.),.005,stamp))
    assert nav.capa.marca[world.grilla.a_celda(1.5,1.)]==0.
