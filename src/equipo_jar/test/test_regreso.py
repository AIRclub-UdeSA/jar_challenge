import math

import numpy as np
import pytest

from equipo_jar.regreso import distancia_a_paredes, linea_libre, velocidad_hacia

RES = 0.05


def cuarto_con_pared():
    """Cuarto de 2 x 2 m (origen en (0, 0)) con una pared vertical en x = 1 m, de y = 0 a 1,5."""
    grilla = np.zeros((40, 40), dtype=int)
    grilla[0, :] = grilla[-1, :] = grilla[:, 0] = grilla[:, -1] = 100
    grilla[0:30, 20] = 100
    return grilla


def test_distancia_a_paredes_en_metros():
    d = distancia_a_paredes(cuarto_con_pared(), RES)
    assert d[10, 10] == pytest.approx(0.5)     # a 10 celdas de la pared de abajo y de x=1
    assert d[0, 5] == 0.0


def test_desconocido_cuenta_como_pared():
    grilla = np.zeros((10, 10), dtype=int)
    grilla[5, 5] = -1
    assert distancia_a_paredes(grilla, RES)[5, 5] == 0.0


def test_linea_libre_y_bloqueada():
    d = distancia_a_paredes(cuarto_con_pared(), RES)
    # Del mismo lado de la pared: libre.
    assert linea_libre(d, RES, 0.0, 0.0, (0.4, 0.4), (0.6, 1.2), margen=0.15)
    # Cruzando la pared: bloqueada.
    assert not linea_libre(d, RES, 0.0, 0.0, (0.5, 0.5), (1.5, 0.5), margen=0.15)
    # Por arriba de la pared (y = 1,75) se puede cruzar.
    assert linea_libre(d, RES, 0.0, 0.0, (0.5, 1.75), (1.5, 1.75), margen=0.15)


def test_linea_que_sale_del_mapa_no_es_libre():
    d = distancia_a_paredes(cuarto_con_pared(), RES)
    assert not linea_libre(d, RES, 0.0, 0.0, (0.5, 0.5), (5.0, 0.5), margen=0.0)


def test_velocidad_hacia_en_el_frame_del_robot():
    # Robot mirando a +y (yaw 90°); el objetivo está a su derecha en el mapa (+x): en su
    # propio frame eso es -y (moverse de costado a la derecha).
    vx, vy = velocidad_hacia(0.0, 0.0, math.pi / 2, 1.0, 0.0, velocidad_max=0.2, ganancia=1.0)
    assert vx == pytest.approx(0.0, abs=1e-9)
    assert vy == pytest.approx(-0.2)


def test_velocidad_hacia_frena_al_acercarse():
    vx, vy = velocidad_hacia(0.0, 0.0, 0.0, 0.05, 0.0, velocidad_max=0.2, ganancia=1.0)
    assert (vx, vy) == pytest.approx((0.05, 0.0))
    assert velocidad_hacia(1.0, 1.0, 0.3, 1.0, 1.0, 0.2, 1.0) == (0.0, 0.0)
