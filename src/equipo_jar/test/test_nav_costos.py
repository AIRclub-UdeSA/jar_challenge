"""Inflado por huella, costo blando y capa de obstáculos no mapeados."""
import math
import time

import numpy as np
import pytest

from equipo_jar.navegacion.costos import (CapaViva, ParametrosCosto, distancia_a_ocupado,
                                          mapa_costos)
from equipo_jar.navegacion.grilla import Grilla

RES = 0.05
P = ParametrosCosto(radio_inscripto_m=0.12, radio_circunscripto_m=math.hypot(0.19,0.12), margen_m=0.10,
                    alcance_blando_m=0.25, peso=5.0)


def grilla(ocupada):
    alto, ancho = ocupada.shape
    return Grilla(RES, 0.0, 0.0, 0.0, ancho, alto, ocupada)


def pasillo():
    """Paredes horizontales con caras internas en y=0,10 y y=1,00 (0,90 m)."""
    ocupada = np.zeros((22, 40), dtype=bool)
    ocupada[0:2, :] = True
    ocupada[20:22, :] = True
    return grilla(ocupada)


def test_distancia_exacta_a_una_celda_ocupada():
    ocupada = np.zeros((41, 41), dtype=bool)
    ocupada[20, 20] = True
    d = distancia_a_ocupado(ocupada, RES)
    assert d[20, 20] == 0.0
    assert d[27, 27] == pytest.approx(math.hypot(7, 7) * RES)
    assert d[20, 30] == pytest.approx(10 * RES)


def test_fuera_del_mapa_cuenta_como_ocupado():
    d = distancia_a_ocupado(np.zeros((41, 41), dtype=bool), RES)
    assert d[0, 20] == pytest.approx(RES)


def test_distancia_se_satura_en_el_maximo():
    d = distancia_a_ocupado(np.zeros((81, 81), dtype=bool), RES, maximo_m=0.5)
    assert d[40, 40] == pytest.approx(0.5)


def test_pasillo_de_0_90_m_deja_centro_libre_y_bordes_prohibidos():
    costos, _ = mapa_costos(pasillo(), P)
    assert math.isfinite(costos[10, 20]) and math.isfinite(costos[11, 20])
    # El círculo que contiene el cuerpo completo sustituye al radio inscripto.
    for fila in (2, 3, 4, 5, 16, 17, 18, 19):
        assert math.isinf(costos[fila, 20])
    assert math.isinf(costos[7,20])
    assert math.isfinite(costos[8,20])


def test_costo_blando_decrece_hacia_el_centro():
    costos, _ = mapa_costos(pasillo(), P)
    assert costos[8, 20] > costos[9, 20] > costos[10, 20] >= 0.0


def test_ocupadas_extra_se_inflan_igual_que_el_mapa():
    extra = np.zeros((22, 40), dtype=bool)
    extra[10, 20] = True
    costos, _ = mapa_costos(pasillo(), P, extra_ocupadas=extra)
    assert math.isinf(costos[10, 22]) and math.isinf(costos[12, 20])


def test_parametros_invalidos_fallan():
    with pytest.raises(ValueError):
        ParametrosCosto(0.3, 0.2, 0.1, 0.25, 5.0)
    with pytest.raises(ValueError):
        ParametrosCosto(0.12, 0.225, float('nan'), 0.25, 5.0)


def test_capa_viva_ignora_paredes_del_mapa_y_vence():
    g = pasillo()
    _, distancia = mapa_costos(g, P)
    capa = CapaViva(g, distancia, vida_s=5.0, tolerancia_mapa_m=0.10)
    capa.actualizar(0.0, [(1.0, 0.08), (1.0, 0.55)])
    marcadas = capa.ocupadas(1.0)
    assert marcadas[g.a_celda(1.0, 0.55)]
    assert not marcadas[g.a_celda(1.0, 0.08)]
    assert marcadas.sum() == 1
    assert not capa.ocupadas(5.1).any()


def test_capa_viva_ignora_puntos_fuera_y_no_finitos():
    g = pasillo()
    _, distancia = mapa_costos(g, P)
    capa = CapaViva(g, distancia, vida_s=5.0, tolerancia_mapa_m=0.10)
    capa.actualizar(0.0, [(-5.0, 0.5), (float('nan'), 0.5), (1.0, float('inf'))])
    assert not capa.ocupadas(0.0).any()


def test_rendimiento_200x200():
    ocupada = np.zeros((200, 200), dtype=bool)
    ocupada[::20, :] = True
    ocupada[:, ::25] = True
    inicio = time.perf_counter()
    mapa_costos(grilla(ocupada), P)
    assert time.perf_counter() - inicio < 0.5
