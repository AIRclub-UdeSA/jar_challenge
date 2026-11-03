from dataclasses import replace
import math
import numpy as np
from equipo_jar.navegacion.ida_vuelta import IdaVuelta,EntradaNavegacion
from equipo_jar.navegacion.tipos import Pose2D
from test_nav_ida_vuelta import Mundo,correr,P_COSTO,P_SEG,HUELLA,P

def test_regresa_por_recorrido_real_si_planificador_de_regreso_falla(monkeypatch):
    world=Mundo(np.zeros((60,100),bool));nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    original=nav._calcular_plan
    def plan(*a):return (None,None) if a[1]=='VOLVIENDO' else original(*a)
    monkeypatch.setattr(nav,'_calcular_plan',plan)
    nav.fijar_meta((2.,1.5));rows=correr(nav,world,(.7,1.5,0.),1500)
    assert nav.estado=='EN_LARGADA',nav.resumen()
    assert nav.retorno_con_historial
    assert math.dist(rows[-1][1:3],(.7,1.5))<=P.tolerancia_largada_m

def test_fallo_nuevo_no_reinicia_camino_de_historial(monkeypatch):
    world=Mundo(np.zeros((60,100),bool));nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.largada=(.7,1.5,0.);nav.ultima_pose=Pose2D(1.5,1.5,0.)
    nav.traza_ejecutada=[nav.largada,(1.,1.5,0.),(1.5,1.5,0.)]
    assert nav._aplicar_plan((None,None),'VOLVIENDO')
    ruta=nav.ruta
    assert nav._aplicar_plan((None,None),'VOLVIENDO')
    assert nav.ruta is ruta

def test_historial_no_inventa_ruta_de_ida():
    world=Mundo(np.zeros((60,100),bool));nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.largada=(.7,1.5,0.);nav.ultima_pose=Pose2D(1.5,1.5,0.)
    nav.traza_ejecutada=[nav.largada,(1.5,1.5,0.)]
    assert not nav._aplicar_plan((None,None),'YENDO')

def test_presupuesto_de_bloqueo_no_es_el_total_del_recorrido(monkeypatch):
    world=Mundo(np.zeros((60,100),bool));nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.estado='BLOQUEADO';nav.previo='YENDO';nav.t_bloqueo=nav.t_inicio_bloqueo=0.
    nav.replanificaciones=P.max_replanificaciones
    monkeypatch.setattr(nav,'_planificar',lambda *a:True)
    nav._bloqueado(EntradaNavegacion(1.1,Pose2D(.7,1.5,0.),True,[],[]),'YENDO')
    assert nav.estado=='YENDO' and nav.replanificaciones==P.max_replanificaciones+1
    assert nav.reintentos_bloqueo==1

def test_historial_demasiado_lejos_no_autoriza_salto():
    world=Mundo(np.zeros((60,100),bool));nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,P)
    nav.largada=(.7,1.5,0.);nav.traza_ejecutada=[nav.largada]
    nav.ultima_pose=Pose2D(3.,1.5,0.)
    assert not nav._aplicar_plan((None,None),'VOLVIENDO')


def test_nueva_vista_no_hereda_reintentos_agotados():
    world=Mundo(np.zeros((60,100),bool));nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,replace(P,regreso_automatico=False))
    nav.estado='FALLO';nav.reintentos_bloqueo=P.max_replanificaciones
    nav.continuar_meta((1.4,1.5),0.)
    assert nav.reintentos_bloqueo==0


def test_regreso_bloqueado_prueba_historial_una_vez_antes_de_fallar():
    world=Mundo(np.zeros((60,100),bool))
    nav=IdaVuelta(world.grilla,P_COSTO,P_SEG,HUELLA,replace(P,regreso_automatico=False))
    nav.largada=(.7,1.5,0.);nav.ultima_pose=Pose2D(1.5,1.5,0.)
    nav.traza_ejecutada=[nav.largada,(1.,1.5,0.),(1.5,1.5,0.)]
    nav.estado='BLOQUEADO';nav.previo='VOLVIENDO'
    nav.reintentos_bloqueo=P.max_replanificaciones
    nav.t_bloqueo=nav.t_inicio_bloqueo=0.
    e=EntradaNavegacion(1.1,nav.ultima_pose,True,[],[])
    assert nav._bloqueado(e,'VOLVIENDO').es_cero()
    assert nav.estado=='VOLVIENDO' and nav.retorno_con_historial
    assert nav.ruta.puntos[-1]==nav.largada[:2]
    nav.estado='BLOQUEADO';nav.previo='VOLVIENDO'
    nav.reintentos_bloqueo=P.max_replanificaciones
    nav.t_bloqueo=nav.t_inicio_bloqueo=0.
    nav._bloqueado(e,'VOLVIENDO')
    assert nav.estado=='FALLO'
