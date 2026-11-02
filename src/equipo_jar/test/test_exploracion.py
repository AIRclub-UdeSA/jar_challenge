import math

import numpy as np
import pytest

from equipo_jar.exploracion import (Cobertura, angulos_relativos, distancia_en_sector,
                                    elegir_direccion, factor_velocidad, lado_mas_libre,
                                    velocidad_lateral)


def scan(funcion_rango, n=360):
    """Scan sintético con el frente en el ángulo 0: rangos[i] = funcion_rango(angulo_rel)."""
    rel = angulos_relativos(n, -math.pi, 2 * math.pi / n, 0.0)
    return np.array([funcion_rango(a) for a in rel]), rel


def test_angulos_relativos_con_lidar_girado():
    # Lidar montado a 180°: el rayo de ángulo pi es el frente (relativo 0).
    rel = angulos_relativos(4, 0.0, math.pi / 2, math.pi)
    assert abs(rel[0]) == pytest.approx(math.pi)          # atrás (±pi es el mismo ángulo)
    assert rel[1:] == pytest.approx([-math.pi / 2, 0.0, math.pi / 2])


def test_distancia_en_sector_ignora_lo_de_afuera_y_lo_invalido():
    rangos, rel = scan(lambda a: 0.4 if abs(a) < 0.2 else 2.0)
    assert distancia_en_sector(rangos, rel, 0.0, math.radians(60), 0.05) == pytest.approx(0.4)
    assert distancia_en_sector(rangos, rel, math.pi / 2, math.radians(30), 0.05) == pytest.approx(2.0)
    rangos[:] = math.inf
    assert distancia_en_sector(rangos, rel, 0.0, 1.0, 0.05) == math.inf


def test_lado_mas_libre():
    rangos, rel = scan(lambda a: 0.3 if a > 0 else 2.5)    # pared a la izquierda
    assert lado_mas_libre(rangos, rel, 0.05) == -1
    rangos, rel = scan(lambda a: 2.5 if a > 0 else 0.3)
    assert lado_mas_libre(rangos, rel, 0.05) == 1


def test_velocidad_lateral_se_aleja_de_la_pared_cercana():
    assert velocidad_lateral(0.1, 1.0, 0.2, 0.1) == -0.1    # pared a la izquierda: ir a la derecha
    assert velocidad_lateral(1.0, 0.1, 0.2, 0.1) == 0.1
    assert velocidad_lateral(0.1, 0.1, 0.2, 0.1) == 0.0     # pasillo angosto: no hacer nada
    assert velocidad_lateral(1.0, 1.0, 0.2, 0.1) == 0.0


def test_factor_velocidad():
    assert factor_velocidad(3.0, 0.5, 1.0) == 1.0
    assert factor_velocidad(0.75, 0.5, 1.0) == pytest.approx(0.5)
    assert factor_velocidad(0.5, 0.5, 1.0) == 0.3


def test_elegir_direccion_prefiere_lo_no_visto_entre_lo_libre():
    libres = [2.0, 2.0, 0.3]
    no_vistos = [0.0, 0.8, 1.0]          # la tercera tiene todo sin ver pero está bloqueada
    assert elegir_direccion(libres, no_vistos, 0.7, 1.0, 2.0) == 1
    assert elegir_direccion([0.2, 0.3], [1.0, 1.0], 0.7, 1.0, 2.0) is None


def cuarto():
    """Cuarto de 4 x 4 m con origen en (0, 0) y una pared en x = 2 que llega hasta y = 3."""
    g = np.zeros((80, 80), dtype=int)
    g[0, :] = g[-1, :] = g[:, 0] = g[:, -1] = 100
    g[0:60, 40] = 100
    return Cobertura(g, 0.05, 0.0, 0.0)


def test_cobertura_marca_el_cono_y_no_atraviesa_paredes():
    c = cuarto()
    c.marcar(1.0, 1.0, 0.0, math.radians(60), 3.0)       # mirando a +x, hacia la pared
    assert c.visto[c.celda(1.5, 1.0)]                      # delante, antes de la pared
    assert not c.visto[c.celda(2.5, 1.0)]                  # detrás de la pared
    assert not c.visto[c.celda(0.5, 1.0)]                  # a la espalda


def test_no_visto_hacia_y_alrededor():
    c = cuarto()
    assert c.no_visto_hacia(1.0, 1.0, 0.0, math.radians(30), 2.0) == 1.0
    c.marcar(1.0, 1.0, 0.0, math.radians(60), 3.0)
    assert c.no_visto_hacia(1.0, 1.0, 0.0, math.radians(30), 2.0) < 0.1
    # Hacia atrás no se vio nada (salvo la celda donde está parado el robot).
    assert c.no_visto_hacia(1.0, 1.0, math.pi, math.radians(30), 2.0) > 0.95
    assert 0.3 < c.no_visto_alrededor(1.0, 1.0, 0.8) < 0.95
    assert 0.0 < c.porcentaje_visto() < 100.0
