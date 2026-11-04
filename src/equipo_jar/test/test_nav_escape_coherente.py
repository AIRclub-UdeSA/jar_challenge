"""Selección de metas y escape comparten conectividad libre de paredes."""
import math
import numpy as np
from equipo_jar.navegacion.costos import ParametrosCosto,mapa_costos
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.metas import elegir_meta_prueba
from equipo_jar.navegacion.planificador import planificar

def test_meta_se_puede_elegir_desde_zona_inflada_escapable():
    g=Grilla(.05,0.,0.,0.,60,30,np.zeros((30,60),dtype=bool))
    costs,d=mapa_costos(g,ParametrosCosto(.12,math.hypot(.19,.12),.1,.25,5.))
    start=(.3,.75)
    assert math.isinf(costs[g.a_celda(*start)])
    goal=elegir_meta_prueba(g,costs,d,start,.5,1.2,.35)
    assert goal is not None
    assert planificar(g,costs,start,goal) is not None

def test_escape_no_cruza_pared_para_llegar_a_celda_finita():
    occupied=np.zeros((9,9),dtype=bool);occupied[:,5]=True
    g=Grilla(.05,0.,0.,0.,9,9,occupied)
    costs=np.full((9,9),np.inf);costs[4,6]=0.
    assert planificar(g,costs,g.a_mundo(4,4),g.a_mundo(4,6)) is None


def test_escape_conserva_meta_exacta_en_primera_celda_finita():
    g=Grilla(.05,0.,0.,0.,9,9,np.zeros((9,9),dtype=bool))
    costs=np.full((9,9),np.inf);costs[4,5]=0.
    goal=(.276,.224)
    route=planificar(g,costs,g.a_mundo(4,4),goal)
    assert route is not None
    assert route.puntos[-1]==goal
