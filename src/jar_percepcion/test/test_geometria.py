"""Pruebas de geometría de percepción sin ejecutar ROS."""

import math

import numpy as np

from jar_percepcion.geometria import (agrupar_en_rachas, pixeles_en_mascara,
                                      probabilidad_en_campo, puntos_laser_a_pixeles,
                                      scan_a_map)


def listas(rachas):
    return [r.tolist() for r in rachas]


def test_rachas_simples():
    b = [False, False, True, True, True, False, True, True, False]
    assert listas(agrupar_en_rachas(b)) == [[2, 3, 4], [6, 7]]


def test_rachas_vacias():
    assert agrupar_en_rachas([False, False]) == []


def test_racha_circular():
    """El primer y el último rayo son contiguos."""
    b = [True, True, False, True, False, True]
    assert listas(agrupar_en_rachas(b)) == [[5, 0, 1], [3]]


def test_racha_completa_no_se_duplica():
    assert listas(agrupar_en_rachas([True] * 4)) == [[0, 1, 2, 3]]


def test_proyeccion_pinhole():
    """Con la cámara mirando hacia +x del lidar, el rayo frontal cae en el centro."""
    # Frame óptico: z adelante, x a la derecha, y hacia abajo.
    rotacion = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    rangos = np.array([2.0, 2.0])
    angulos = np.array([0.0, math.pi])
    u, v, adelante = puntos_laser_a_pixeles(
        rangos, angulos, rotacion, np.zeros(3), 100.0, 100.0, 160.0, 120.0)
    assert adelante.tolist() == [True, False]
    assert math.isclose(u[0], 160.0) and math.isclose(v[0], 120.0)


def test_pixeles_en_mascara():
    mascara = np.zeros((4, 6), dtype=np.uint8)
    mascara[1, 2] = 255
    u = np.array([2.5, 2.5, 10.0, np.nan])
    v = np.array([1.2, 3.0, 1.0, 1.0])
    adelante = np.array([True, True, True, False])
    assert pixeles_en_mascara(u, v, adelante, mascara).tolist() == [True, False, False, False]


def test_scan_a_map_y_campo():
    """Un punto a 1 m adelante de un robot girado 90° cae en +y."""
    xs, ys = scan_a_map(np.array([1.0]), np.array([0.0]), 0.5, 0.5, math.pi / 2)
    assert math.isclose(xs[0], 0.5, abs_tol=1e-9) and math.isclose(ys[0], 1.5)
    campo = np.zeros((20, 10))
    campo[15, 5] = 80.0
    prob, dentro = probabilidad_en_campo(
        np.array([xs[0], -1.0]), np.array([ys[0], 0.0]), campo, 0.1, 0.0, 0.0)
    assert dentro.tolist() == [True, False]
    assert prob.tolist() == [80.0, 0.0]
