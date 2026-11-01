import math
import types

import numpy as np
import pytest

from equipo_jar.exploracion import Exploracion, ordenar_recorrido

RES = 0.05


class Logger:

    def info(self, _msg):
        pass

    error = warn = info


def mapa_sintetico():
    """Caja de 4 x 3 m con una pared interna que deja un paso de 0.8 m."""
    alto, ancho = int(3.0 / RES), int(4.0 / RES)
    datos = np.zeros((alto, ancho), dtype=np.int8)
    datos[:2, :] = datos[-2:, :] = datos[:, :2] = datos[:, -2:] = 100
    # Pared vertical en x = 2 m, con un hueco arriba.
    datos[:int(2.2 / RES), int(2.0 / RES):int(2.1 / RES)] = 100
    info = types.SimpleNamespace(
        resolution=RES, width=ancho, height=alto,
        origin=types.SimpleNamespace(position=types.SimpleNamespace(x=0.0, y=0.0)))
    return types.SimpleNamespace(info=info, data=datos.ravel().tolist())


@pytest.fixture
def exploracion():
    e = Exploracion(mapa_sintetico(), Logger())
    e.planificar((0.5, 0.5, 0.0))
    return e


def test_hay_puntos_de_vista_en_las_dos_mitades(exploracion):
    xs = [x for x, _, _ in exploracion.pendientes]
    assert exploracion.pendientes
    assert any(x > 2.1 for x in xs), 'tiene que cruzar la pared para ver la otra mitad'


def test_los_puntos_de_vista_estan_lejos_de_las_paredes(exploracion):
    for x, y, _ in exploracion.pendientes:
        fila, col = exploracion.celda(x, y)
        assert exploracion.dist_pared[fila, col] >= 0.25 - 1e-9


def test_distancia_por_el_laberinto(exploracion):
    # Del otro lado de la pared, a 1 m en línea recta de la largada: por el
    # laberinto hay que dar la vuelta por el hueco.
    d = exploracion.distancia_camino((3.0, 0.5), None)
    assert d is not None and d > 2.5


def test_reintento_al_final(exploracion):
    primero = exploracion.pendientes[0][:2]
    exploracion.objetivo_terminado(False)
    assert exploracion.pendientes[-1][:2] == primero
    while exploracion.pendientes[0][:2] != primero:
        exploracion.objetivo_terminado(True)
    exploracion.objetivo_terminado(False)   # segundo fallo: se descarta
    assert primero not in [p[:2] for p in exploracion.pendientes]


def test_siguiente_devuelve_none_al_terminar(exploracion):
    while exploracion.pendientes:
        exploracion.objetivo_terminado(True)
    assert exploracion.siguiente((0.5, 0.5, 0.0)) is None


def test_ordenar_recorrido_visita_cuadrado_en_orden():
    puntos = [(0, 0), (0, 1), (5, 1), (5, 0), (0.1, 0.5)]
    d = np.array([[math.dist(a, b) for b in puntos] for a in puntos])
    orden = ordenar_recorrido(d)
    assert orden[0] == 0 and sorted(orden) == list(range(len(puntos)))
    largo = sum(d[a, b] for a, b in zip(orden, orden[1:] + orden[:1]))
    assert largo == pytest.approx(12.0, abs=0.3)
