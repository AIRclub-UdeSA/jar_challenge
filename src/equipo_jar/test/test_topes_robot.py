"""Los límites de velocidad de params.yaml y nav2.yaml respetan los topes del robot real.

El robot recorta a 0,5 m/s en x, 0,5 m/s en y y 1 rad/s (mail de la organización del 31/10).
El simulador acepta hasta 1 m/s y 5 rad/s, así que sin este test una config puede andar bien
en simulación y distinta en el robot.
"""
from pathlib import Path

import yaml

PARAMS = Path(__file__).resolve().parents[1] / 'params'
TOPE_LINEAL = 0.5     # m/s, en x y en y
TOPE_ANGULAR = 1.0    # rad/s


def cargar(nombre):
    return yaml.safe_load((PARAMS / nombre).read_text())


def test_supervisor_respeta_los_topes():
    s = cargar('params.yaml')['supervisor']['ros__parameters']
    assert s['vel_lineal_max'] <= TOPE_LINEAL
    assert s['vel_lateral_max'] <= TOPE_LINEAL
    assert s['vel_angular_max'] <= TOPE_ANGULAR


def test_giros_de_la_mision_respetan_el_tope():
    m = cargar('params.yaml')['mision']['ros__parameters']
    for k in ('vel_giro_localizacion', 'vel_giro_max', 'vel_giro_desatasco', 'vel_giro_escape'):
        assert m[k] <= TOPE_ANGULAR, k


def test_nav2_respeta_los_topes():
    c = cargar('nav2.yaml')['controller_server']['ros__parameters']['FollowPath']
    assert c['desired_linear_vel'] <= TOPE_LINEAL
    assert c['rotate_to_heading_angular_vel'] <= TOPE_ANGULAR
