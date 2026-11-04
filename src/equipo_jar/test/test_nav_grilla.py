"""Grilla del mapa recibido: celdas, origen rotado y celdas desconocidas."""
import math

import pytest

from equipo_jar.navegacion.grilla import grilla_desde_occupancy


def grilla(datos=None, ancho=10, alto=8, resolucion=0.05, ox=1.0, oy=2.0, oyaw=0.0):
    datos = [0] * (ancho * alto) if datos is None else datos
    return grilla_desde_occupancy(datos, ancho, alto, resolucion, ox, oy, oyaw)


def test_ida_y_vuelta_celda_mundo_con_origen_rotado():
    g = grilla(oyaw=math.pi / 2)
    for x, y in ((0.9, 2.1), (0.67, 2.33), (0.62, 2.48)):
        celda = g.a_celda(x, y)
        assert celda is not None
        cx, cy = g.a_mundo(*celda)
        assert math.hypot(cx - x, cy - y) <= g.resolucion / 2 * math.sqrt(2) + 1e-12


def test_celda_0_0_cae_en_el_origen_rotado():
    yaw = 0.7
    g = grilla(oyaw=yaw)
    x, y = g.a_mundo(0, 0)
    h = g.resolucion / 2
    assert x == pytest.approx(1.0 + math.cos(yaw) * h - math.sin(yaw) * h)
    assert y == pytest.approx(2.0 + math.sin(yaw) * h + math.cos(yaw) * h)
    assert g.a_celda(x, y) == (0, 0)


def test_fila_es_y_local_y_columna_es_x_local():
    g = grilla()
    assert g.a_celda(1.0 + 0.05 * 3.5, 2.0 + 0.05 * 1.5) == (1, 3)


def test_desconocidas_y_ocupadas_son_ocupadas():
    g = grilla_desde_occupancy([-1, 0, 49, 50, 100], 5, 1, 0.05, 0.0, 0.0, 0.0)
    assert g.ocupada.tolist() == [[True, False, False, True, True]]


def test_fuera_de_la_grilla_es_none():
    g = grilla()
    assert g.a_celda(0.99, 2.01) is None
    assert g.a_celda(1.0 + 0.05 * 10 + 0.001, 2.01) is None
    assert g.a_celda(1.01, 2.0 + 0.05 * 8 + 0.001) is None


def test_datos_de_largo_incorrecto_fallan():
    with pytest.raises(ValueError):
        grilla([0] * 7)


def test_resolucion_invalida_falla():
    with pytest.raises(ValueError):
        grilla(resolucion=0.0)


def test_refinado_conserva_paredes_desconocidos_y_origen_rotado():
    import numpy as np
    from equipo_jar.navegacion.grilla import refinar_para_obstaculos
    g = grilla_desde_occupancy([-1, 0, 100, 0], 2, 2, .05, 1., -2., .7)
    fine = refinar_para_obstaculos(g)
    assert fine.resolucion == pytest.approx(.025)
    assert np.array_equal(fine.ocupada, np.repeat(np.repeat(g.ocupada, 2, 0), 2, 1))
    for row in range(fine.alto):
        for col in range(fine.ancho):
            assert g.a_celda(*fine.a_mundo(row, col)) == (row // 2, col // 2)


def test_retornos_lidar_se_registran_finos_antes_de_planificar():
    import numpy as np
    from equipo_jar.navegacion.grilla import refinar_para_obstaculos
    from equipo_jar.navegacion.costos import CapaViva
    g = refinar_para_obstaculos(grilla(ancho=40, alto=40, ox=0., oy=0.))
    layer = CapaViva(g, np.ones(g.ocupada.shape), 600., .1)
    layer.actualizar(0., [(1.012, 1.012)])
    occupied = layer.ocupadas(1.)
    assert occupied.sum() == 1
    assert occupied[g.a_celda(1.012, 1.012)]
    assert not occupied[g.a_celda(1.037, 1.012)]


def test_refinado_no_engorda_ni_repite_grillas_ya_finas():
    from equipo_jar.navegacion.grilla import refinar_para_obstaculos
    g = grilla(resolucion=.025)
    assert refinar_para_obstaculos(g) is g
    with pytest.raises(ValueError):
        refinar_para_obstaculos(g, 0.)
