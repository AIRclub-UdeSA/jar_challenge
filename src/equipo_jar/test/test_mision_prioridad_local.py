"""Preferencia local sin ocultar destinos lejanos ni saltar reserva/huella."""
from dataclasses import replace
import math

import numpy as np
import pytest

from equipo_jar.mision import ParametrosMision,elegir_vista
from equipo_jar.navegacion.grilla import Grilla


def escena(exponente,ganancia_lejana=70,local_observada=False,**kwargs):
    g=Grilla(.1,0.,0.,0.,60,25,np.zeros((25,60),bool))
    permitidas=np.zeros_like(g.ocupada)
    permitidas[14,14]=permitidas[14,44]=True
    ida=np.full_like(g.ocupada,np.inf,dtype=float)
    ida[14,14]=.5;ida[14,44]=3.5
    class Cobertura:
        def ganancia(self,pose):
            return ganancia_lejana if pose[0]>3. else (0 if local_observada else 20)
        def cota_ganancia(self,pose):return self.ganancia(pose)
    p=replace(ParametrosMision(),exponente_distancia=exponente)
    return elegir_vista(g,np.zeros_like(ida),(.95,1.45,0.),(.95,1.45,0.),
        Cobertura(),set(),600.,p,metas_permitidas=permitidas,
        distancias=(ida,ida),**kwargs)


def test_ganancia_moderada_local_se_completa_antes_de_viaje_largo():
    assert escena(1.)['x']>3.
    assert escena(2.)['x']<2.


def test_destino_lejano_sigue_disponible_si_local_ya_se_observo():
    assert escena(2.,local_observada=True)['x']>3.


def test_ganancia_lejana_suficiente_puede_superar_preferencia_local():
    assert escena(2.,ganancia_lejana=1000)['x']>3.


def test_preferencia_local_no_autoriza_pose_sin_huella_libre():
    assert escena(2.,filtro_pose=lambda x,y,yaw:x>3.)['x']>3.


@pytest.mark.parametrize('exponente',[0.,-1.,math.inf,math.nan])
def test_exponente_invalido_no_se_acepta(exponente):
    with pytest.raises(ValueError):replace(ParametrosMision(),exponente_distancia=exponente)
