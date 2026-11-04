"""Meta de prueba calculada desde el mapa recibido, sin coordenadas fijas."""
import numpy as np

from equipo_jar.navegacion.costos import ParametrosCosto, mapa_costos
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.metas import elegir_meta_prueba
from equipo_jar.navegacion.planificador import costo_desde, planificar

RES = 0.05
P = ParametrosCosto(0.12, 0.225, 0.10, 0.25, 5.0)


def laberinto(cerrar_habitacion=True):
    """Pasillo en U de 1 m y una habitación amplia cerrada a la derecha."""
    ocupada = np.zeros((60, 100), dtype=bool)
    ocupada[18:22, 0:45] = True          # tabique de la U
    ocupada[:, 60:62] = True             # pared de la habitación
    if not cerrar_habitacion:
        ocupada[25:35, 60:62] = False
    g = Grilla(RES, 0.0, 0.0, 0.0, 100, 60, ocupada)
    costos, distancia = mapa_costos(g, P)
    return g, costos, distancia


def elegir(g, costos, distancia, inicio=(0.5, 0.45), dmin=1.5, dmax=3.0, espacio=0.35):
    return elegir_meta_prueba(g, costos, distancia, inicio, dmin, dmax, espacio)


def test_meta_en_rango_y_alcanzable():
    g, costos, distancia = laberinto()
    meta = elegir(g, costos, distancia)
    assert meta is not None
    _, largo = costo_desde(g, costos, (0.5, 0.45))
    assert 1.5 <= largo[g.a_celda(*meta)] <= 3.0
    assert distancia[g.a_celda(*meta)] >= 0.35
    assert planificar(g, costos, (0.5, 0.45), meta) is not None


def test_ninguna_candidata_devuelve_none():
    g, costos, distancia = laberinto()
    assert elegir(g, costos, distancia, dmin=50.0, dmax=60.0) is None
    assert elegir(g, costos, distancia, espacio=2.0) is None


def test_determinista():
    g, costos, distancia = laberinto()
    assert elegir(g, costos, distancia) == elegir(g, costos, distancia)


def test_no_elige_celdas_aisladas_inalcanzables():
    g, costos, distancia = laberinto()
    meta = elegir(g, costos, distancia, dmin=0.5, dmax=20.0)
    assert meta[0] < 3.0  # la habitación cerrada (x > 3,1 m) es más amplia pero inalcanzable


def test_inicio_prohibido_devuelve_none():
    g, costos, distancia = laberinto()
    assert elegir(g, costos, distancia, inicio=(0.5, 0.95)) is None
