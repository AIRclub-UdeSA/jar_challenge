"""Pruebas del agrupamiento de víctimas sin ejecutar ROS."""

import math

from jar_mision.victimas import AgrupadorVictimas, centro_desde_cara


def test_centro_se_aleja_del_robot():
    x, y = centro_desde_cara((0.0, 0.0), (1.0, 0.0), 0.09)
    assert math.isclose(x, 1.09) and math.isclose(y, 0.0)


def test_centro_sobre_el_robot():
    assert centro_desde_cara((1.0, 1.0), (1.0, 1.0), 0.09) is None


def test_publica_una_vez_al_juntar_minimas():
    agrupador = AgrupadorVictimas(victimas=2, detecciones_minimas=3, radio_grupo=0.3)
    assert agrupador.agregar(1.0, 1.0) is None
    assert agrupador.agregar(1.1, 1.0) is None
    assert agrupador.agregar(1.0, 1.1) == (1.0, 1.0)
    assert agrupador.agregar(1.0, 1.0) is None
    assert agrupador.publicados == 1


def test_mediana_ignora_detecciones_lejanas():
    agrupador = AgrupadorVictimas(victimas=1, detecciones_minimas=4, radio_grupo=0.5)
    for punto in [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]:
        agrupador.agregar(*punto)
    assert agrupador.agregar(0.4, 0.0) == (0.0, 0.0)


def test_no_supera_la_cantidad_de_victimas():
    agrupador = AgrupadorVictimas(victimas=1, detecciones_minimas=1, radio_grupo=0.3)
    assert agrupador.agregar(0.0, 0.0) == (0.0, 0.0)
    assert agrupador.agregar(2.0, 2.0) is None
    assert len(agrupador.grupos) == 2


def sembrar(agrupador, x, y, n):
    for _ in range(n):
        agrupador.agregar(x, y)


def test_al_final_no_reporta_hasta_cerrar():
    agrupador = AgrupadorVictimas(victimas=2, detecciones_minimas=3, inmediato=False)
    sembrar(agrupador, 0.0, 0.0, 5)
    assert agrupador.publicados == 0
    assert agrupador.cerrar(1, 0.5) == [(0.0, 0.0)]


def test_al_final_elige_los_grupos_con_mas_detecciones():
    """Un falso positivo que se confirma primero no gasta un reporte."""
    agrupador = AgrupadorVictimas(victimas=2, detecciones_minimas=3, inmediato=False)
    sembrar(agrupador, 5.0, 5.0, 4)     # falso positivo, temprano y con pocas detecciones
    sembrar(agrupador, 0.0, 0.0, 40)
    sembrar(agrupador, 2.0, 0.0, 30)
    assert agrupador.cerrar(1, 0.5) == [(0.0, 0.0), (2.0, 0.0)]


def test_al_final_completa_con_candidatos_debiles():
    """Si sobran reportes, usa grupos con menos detecciones: errar no resta puntos."""
    agrupador = AgrupadorVictimas(victimas=2, detecciones_minimas=30, inmediato=False)
    sembrar(agrupador, 0.0, 0.0, 40)
    sembrar(agrupador, 2.0, 0.0, 12)
    sembrar(agrupador, 4.0, 0.0, 3)
    assert agrupador.cerrar(10, 0.5) == [(0.0, 0.0), (2.0, 0.0)]


def test_al_final_no_reporta_dos_veces_la_misma_caja():
    agrupador = AgrupadorVictimas(victimas=2, detecciones_minimas=3, radio_grupo=0.3,
                                  inmediato=False)
    sembrar(agrupador, 0.0, 0.0, 40)
    sembrar(agrupador, 0.35, 0.0, 30)   # la misma caja partida en dos grupos
    sembrar(agrupador, 3.0, 0.0, 10)
    assert agrupador.cerrar(3, 0.5) == [(0.0, 0.0), (3.0, 0.0)]


def test_cerrar_respeta_lo_ya_reportado():
    agrupador = AgrupadorVictimas(victimas=2, detecciones_minimas=3)
    sembrar(agrupador, 0.0, 0.0, 3)
    assert agrupador.publicados == 1
    sembrar(agrupador, 0.1, 0.0, 30)    # mismo grupo, ya reportado
    sembrar(agrupador, 2.0, 0.0, 2)     # no llega al mínimo inmediato
    assert agrupador.publicados == 1
    assert agrupador.cerrar(1, 0.5) == [(2.0, 0.0)]
    assert agrupador.cerrar(1, 0.5) == []
