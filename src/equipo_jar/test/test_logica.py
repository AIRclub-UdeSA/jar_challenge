import math

import numpy as np

from equipo_jar import logica


def test_normalizar():
    assert abs(logica.normalizar(3 * math.pi) - math.pi) < 1e-9


def test_lecturas_con_offset():
    # Una lectura en angle_min=0 con offset pi queda atrás del robot.
    lect = logica.lecturas_validas([1.0, float('inf'), 0.01], 0.0, 0.1, math.pi, 0.05, 10.0)
    assert len(lect) == 1
    assert abs(abs(lect[0][0]) - math.pi) < 1e-9


def test_proyectar():
    x, y = logica.proyectar(1.0, 0.0, math.pi / 2, 2.0, 0.0)
    assert abs(x - 1.0) < 1e-9 and abs(y - 2.0) < 1e-9


def test_rumbo_de_pixel():
    assert logica.rumbo_de_pixel(320, 320, 500) == 0.0
    assert logica.rumbo_de_pixel(0, 320, 500) > 0  # izquierda de la imagen = positivo


def test_mascara_y_blobs():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 70:90] = (255, 0, 0)   # rojo en RGB
    img[10:30, 10:30] = (0, 0, 255)   # azul: no tiene que aparecer
    m = logica.mascara_roja(img, 100, 80, 10, 170)
    b = logica.blobs(m, 50)
    assert len(b) == 1
    assert abs(b[0][0] - 79.5) < 1.0


def test_agrupador():
    a = logica.Agrupador(0.3, 3)
    for _ in range(2):
        a.agregar(1.0, 1.0)
    a.agregar(3.0, 3.0)
    assert a.listos() == []
    a.agregar(1.1, 1.0)
    listos = a.listos()
    assert len(listos) == 1
    listos[0][0]['reportado'] = True
    assert a.cantidad_reportados() == 1
    assert a.listos() == []
