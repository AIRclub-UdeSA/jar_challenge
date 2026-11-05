"""El mismo objeto conserva su centro al cambiar de cara y de ángulo de observación."""
import math
from types import SimpleNamespace

import numpy as np
import pytest

from equipo_jar.observacion_objetos import centro_superficie_lidar, objetos_del_scan


def barrido_caja(sensor, centro, giro, lado=0.25):
    rumbo = math.atan2(centro[1] - sensor[1], centro[0] - sensor[0])
    angulos = rumbo + np.linspace(-0.4, 0.4, 321)
    rot = np.array([[math.cos(giro), -math.sin(giro)], [math.sin(giro), math.cos(giro)]])
    origen = (np.asarray(sensor[:2]) - centro) @ rot
    rayos = np.column_stack((np.cos(angulos), np.sin(angulos))) @ rot
    with np.errstate(divide='ignore', invalid='ignore'):
        a = (-lado / 2 - origen) / rayos
        b = (lado / 2 - origen) / rayos
    entrada = np.minimum(a, b).max(axis=1)
    salida = np.maximum(a, b).min(axis=1)
    rangos = np.where((salida >= entrada) & (entrada > 0), entrada, np.inf)
    return SimpleNamespace(ranges=rangos, angle_min=angulos[0], angle_increment=0.0025,
                           range_min=0.1, range_max=10.0)


@pytest.mark.parametrize('giro', [0.0, 0.4, 1.1])
@pytest.mark.parametrize('sensor', [(0.5, 0.5, 0.0), (3.5, 0.5, 0.0),
                                    (0.5, 3.5, 0.0), (3.5, 3.5, 0.0), (0.5, 2.1, 0.0)])
def test_centro_consistente_desde_caras_y_esquinas(giro, sensor):
    centro = np.array([2.0, 2.0])
    scan = barrido_caja(sensor, centro, giro)
    libre = np.ones((250, 250), dtype=bool)
    grilla = SimpleNamespace(ox=0.0, oy=0.0, res=0.02, libre=libre)
    objetos = objetos_del_scan(scan, sensor, grilla, np.full(libre.shape, 1.0),
                               alcance=3.0, profundidad=0.125)
    assert len(objetos) == 1
    assert objetos[0] == pytest.approx(centro, abs=0.015)


def test_una_cara_oblicua_se_desplaza_por_su_normal():
    # Una cara vertical, vista desde abajo a la izquierda: el centro no baja con el rayo.
    xs = np.full(30, 0.875)
    ys = np.linspace(0.875, 1.125, 30)
    assert centro_superficie_lidar(xs, ys, (0.0, 0.0), 0.125) == pytest.approx((1.0, 1.0))


def test_forma_curva_se_conserva_como_objeto():
    angulos = np.linspace(-0.115, 0.115, 41)
    rangos = np.cos(angulos) - np.sqrt(0.12**2 - np.sin(angulos)**2)
    centro = centro_superficie_lidar(rangos * np.cos(angulos), rangos * np.sin(angulos),
                                     (0.0, 0.0), 0.12)
    assert centro == pytest.approx((1.0, 0.0), abs=0.05)
