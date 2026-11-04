"""El retorno omite vueltas sólo con enlaces cortos de huella admisible."""
import math
from dataclasses import replace

import numpy as np

from equipo_jar.navegacion.ida_vuelta import IdaVuelta
from equipo_jar.navegacion.tipos import Pose2D
from test_nav_ida_vuelta import Mundo, P_COSTO, P_SEG, HUELLA, P


def navegar(traza, ocupada=None):
    world=Mundo(np.zeros((100,120),bool) if ocupada is None else ocupada)
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.largada=traza[0]
    nav.traza_ejecutada=list(traza)
    nav.ultima_pose=Pose2D(*traza[-1])
    return nav


def test_retorno_elige_visita_temprana_en_lugar_de_repetir_vuelta():
    traza=[(1.,1.,0.),(1.5,1.,.1),(2.,1.,0.),(2.,2.,0.),
           (1.5,2.,0.),(1.5,1.1,.2)]
    nav=navegar(traza)
    ruta=nav._regreso_por_historial()
    assert nav.diagnostico_historial['indice']==1
    assert ruta.puntos==[(1.5,1.1),(1.5,1.),(1.,1.)]
    assert ruta.orientaciones==[.2,.1,0.]
    assert math.isclose(ruta.largo_m,.6)
    assert nav.traza_ejecutada==traza


def test_retorno_elimina_bucle_interno_y_conserva_tramo_posterior():
    traza=[(1.,1.,0.),(1.5,1.,0.),(2.,1.,0.),(2.,2.,0.),
           (1.5,2.,0.),(1.5,1.1,0.),(1.5,.7,0.)]
    nav=navegar(traza)
    ruta=nav._regreso_por_historial()
    assert ruta.puntos==[(1.5,.7),(1.5,.7),(1.5,1.1),(1.5,1.),(1.,1.)]
    assert nav.diagnostico_historial['saltos_sin_bucles']==1
    assert math.isclose(ruta.largo_m,1.)


def test_traza_lineal_sin_visita_cercana_no_cambia():
    traza=[(1.,1.,0.),(1.3,1.,0.),(1.6,1.,0.),(1.9,1.,0.)]
    nav=navegar(traza)
    ruta=nav._regreso_por_historial()
    assert ruta.puntos==[traza[-1][:2]]+[p[:2] for p in reversed(traza)]
    assert math.isclose(ruta.largo_m,.9)


def test_cercania_no_autoriza_atajo_a_traves_de_obstaculo():
    traza=[(1.,1.,0.),(1.5,1.,0.),(2.,1.,0.),(2.,2.,0.),
           (1.5,2.,0.),(1.5,1.19,0.)]
    nav=navegar(traza)
    # El punto obstruye el enlace corto; no permitir cercanía como prueba.
    celda=nav.grilla.a_celda(1.5,1.1)
    nav.grilla.ocupada[celda]=True
    ruta=nav._regreso_por_historial()
    assert nav.diagnostico_historial['indice']==len(traza)-1
    assert (2.,2.) in ruta.puntos


def test_capa_viva_impide_atajo_sin_borrar_memoria():
    traza=[(1.,1.,0.),(1.5,1.,0.),(2.,1.,0.),(2.,2.,0.),
           (1.5,2.,0.),(1.5,1.19,0.)]
    nav=navegar(traza)
    nav.capa.registrar_bloqueo(0.,[(1.5,1.1)])
    previa=nav.capa.ocupadas(0.).copy()
    ruta=nav._regreso_por_historial()
    assert nav.diagnostico_historial['indice']==len(traza)-1
    assert (2.,2.) in ruta.puntos
    assert np.array_equal(nav.capa.ocupadas(0.),previa)


def test_enlace_comprueba_giro_completo_no_solo_centros():
    nav=navegar([(1.,1.,0.)])
    nav.huella_operativa=dict(HUELLA,margin_m=0.,heading_uncertainty_rad=0.)
    a=(1.,1.,0.);b=(1.,1.,math.pi)
    # Fuera de la huella en yaw0 y pi, dentro del arco de la esquina trasera.
    obstaculos=np.array([[1.,1.18]])
    assert nav._enlace_historial_libre(a,a,obstaculos)
    assert not nav._enlace_historial_libre(a,b,obstaculos)


def test_retorno_funciona_con_origen_rotado():
    nav=navegar([(1.,1.,0.),(1.4,1.,0.),(1.8,1.,0.),(1.4,1.1,0.)])
    nav.grilla=replace(nav.grilla,origen_x=6.,origen_y=0.,origen_yaw=math.pi/2)
    ruta=nav._regreso_por_historial()
    assert nav.diagnostico_historial['indice']==1
    assert math.isclose(ruta.largo_m,.5)


def test_historial_discontinuo_o_pose_ajena_no_crea_ruta():
    nav=navegar([(1.,1.,0.),(1.4,1.,0.)])
    nav.traza_integra=False
    assert nav._regreso_por_historial() is None
    nav.traza_integra=True
    nav.ultima_pose=Pose2D(4.,4.,0.)
    assert nav._regreso_por_historial() is None
