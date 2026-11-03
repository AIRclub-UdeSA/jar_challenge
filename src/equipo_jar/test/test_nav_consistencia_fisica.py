"""El plan contempla todo el cuerpo y la mira no salta una esquina."""
import math
import numpy as np
import pytest
from equipo_jar.navegacion.costos import ParametrosCosto,mapa_costos
from equipo_jar.navegacion.grilla import grilla_desde_occupancy
from equipo_jar.navegacion.planificador import Ruta
from equipo_jar.navegacion.seguimiento import ParametrosSeguimiento,Seguidor
from equipo_jar.navegacion.tipos import Pose2D

def test_inflado_prohibe_centro_donde_no_cabe_cuerpo_completo():
    occ=np.zeros((60,80),dtype=int)
    occ[:,0]=occ[:,-1]=100;occ[0,:]=occ[-1,:]=100
    g=grilla_desde_occupancy(occ.ravel(),80,60,.05,0.,0.,0.)
    p=ParametrosCosto(.12,.225,.10,.25,5.)
    costs,d=mapa_costos(g,p)
    cell=g.a_celda(.325,1.5)
    assert d[cell]<p.radio_circunscripto_m+p.margen_m+.025
    assert math.isinf(costs[cell])
    assert math.isfinite(costs[g.a_celda(.475,1.5)])

def test_mira_no_corta_esquina_antes_de_alcanzar_su_vertice():
    p=ParametrosSeguimiento(.3,.5,.2,.03,1.5,.5,.25,.45,.1,.2)
    s=Seguidor(p);s.reiniciar(Ruta([(0.,0.),(1.,0.),(1.,1.)],0.,2.),0.)
    cmd=s.paso(0.,Pose2D(.9,0.,0.),1.)
    assert cmd.vx>0.
    assert abs(cmd.vy)<1e-12


def test_mantiene_limite_de_aceleracion_vectorial_al_cambiar_direccion():
    p=ParametrosSeguimiento(.3,.5,.2,.03,1.5,.5,.25,.45,.1,.2)
    s=Seguidor(p);s.reiniciar(Ruta([(0.,0.),(1.,0.),(1.,1.)],0.,2.),0.)
    previous=s.paso(0.,Pose2D(.9,0.,0.),1.)
    command=s.paso(.1,Pose2D(.98,0.,0.),1.)
    assert math.hypot(command.vx-previous.vx,command.vy-previous.vy)<=.5*.1+1e-12
