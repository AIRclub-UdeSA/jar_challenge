"""A* sobre costos inflados: rutas seguras, poda, escape de zona inflada."""
import math
import time

import numpy as np

from equipo_jar.navegacion.costos import ParametrosCosto, mapa_costos
from equipo_jar.navegacion.grilla import Grilla
from equipo_jar.navegacion.planificador import costo_desde, planificar

RES = 0.05
P = ParametrosCosto(0.12, math.hypot(0.19,0.12), 0.10, 0.25, 5.0)


def grilla(ocupada):
    alto, ancho = ocupada.shape
    return Grilla(RES, 0.0, 0.0, 0.0, ancho, alto, ocupada)


def con_pared_en_l(hasta_fila=26):
    ocupada = np.zeros((40, 40), dtype=bool)
    ocupada[0:hasta_fila, 18:22] = True
    g = grilla(ocupada)
    return g, mapa_costos(g, P)[0]


def pasillo():
    ocupada = np.zeros((22, 40), dtype=bool)
    ocupada[0:2, :] = True
    ocupada[20:22, :] = True
    g = grilla(ocupada)
    return g, mapa_costos(g, P)[0]


def recorre_solo_finitas(g, costos, puntos, desde=0):
    for (x0, y0), (x1, y1) in list(zip(puntos, puntos[1:]))[desde:]:
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.01))
        for i in range(n + 1):
            celda = g.a_celda(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            if celda is None or not math.isfinite(costos[celda]):
                return False
    return True


def test_ruta_en_l_rodea_la_pared_y_no_toca_celdas_inf():
    g, costos = con_pared_en_l()
    ruta = planificar(g, costos, (0.4, 0.4), (1.6, 0.4))
    assert ruta is not None
    assert ruta.puntos[0] == (0.4, 0.4) and ruta.puntos[-1] == (1.6, 0.4)
    assert recorre_solo_finitas(g, costos, ruta.puntos)
    assert max(y for _, y in ruta.puntos) > 1.3
    assert ruta.largo_m == sum(math.dist(a, b) for a, b in zip(ruta.puntos, ruta.puntos[1:]))


def test_sin_ruta_devuelve_none():
    g, costos = con_pared_en_l(hasta_fila=40)
    assert planificar(g, costos, (0.4, 0.4), (1.6, 0.4)) is None


def test_no_atraviesa_esquina_entre_dos_celdas_prohibidas():
    g = grilla(np.array([[False, True], [True, False]]))
    costos = np.array([[0., np.inf], [np.inf, 0.]])
    assert planificar(g, costos, (.025, .025), (.075, .075)) is None


def test_meta_prohibida_devuelve_none():
    g, costos = pasillo()
    assert planificar(g, costos, (0.3, 0.55), (1.0, 0.15)) is None


def test_poda_reduce_vertices_en_pasillo_recto():
    g, costos = pasillo()
    ruta = planificar(g, costos, (0.4, 0.55), (1.6, 0.55))
    assert ruta.puntos == [(0.4, 0.55), (1.6, 0.55)]


def test_inicio_en_zona_inflada_escapa():
    g, costos = pasillo()
    assert math.isinf(costos[g.a_celda(1.0, 0.30)])
    ruta = planificar(g, costos, (1.0, 0.30), (1.6, 0.55))
    assert ruta is not None
    assert ruta.puntos[0] == (1.0, 0.30)
    salida=next(i for i,p in enumerate(ruta.puntos) if math.isfinite(costos[g.a_celda(*p)]))
    assert recorre_solo_finitas(g,np.where(g.ocupada,np.inf,0.),ruta.puntos[:salida+1])
    assert recorre_solo_finitas(g,costos,ruta.puntos,desde=salida)


def test_inicio_demasiado_hundido_no_escapa():
    g, costos = pasillo()
    assert planificar(g, costos, (1.0, 0.12), (1.7, 0.55), escape_m=0.10) is None


def test_costo_desde_coincide_con_largo_de_ruta():
    g, costos = con_pared_en_l()
    costo, largo = costo_desde(g, costos, (0.4, 0.4))
    ruta = planificar(g, costos, (0.4, 0.4), (1.6, 0.4))
    meta = g.a_celda(1.6, 0.4)
    assert math.isfinite(costo[meta])
    assert ruta.largo_m <= largo[meta] + 1e-6
    assert largo[meta] <= 1.10 * ruta.largo_m
    assert math.isinf(largo[g.a_celda(0.97, 0.4)])  # dentro de la pared


def test_rendimiento_200x200():
    ocupada = np.zeros((200, 200), dtype=bool)
    for fila in range(20, 200, 40):
        ocupada[fila:fila + 2, 0:170] = True
    for fila in range(40, 200, 40):
        ocupada[fila:fila + 2, 30:200] = True
    g = grilla(ocupada)
    costos = mapa_costos(g, P)[0]
    inicio = time.perf_counter()
    ruta = planificar(g, costos, (0.5, 0.5), (9.5, 9.5))
    assert ruta is not None
    assert time.perf_counter() - inicio < 0.2


def test_meta_antes_admitida_por_radio_inscripto_no_cabe_en_envolvente_completa():
    g,costos=pasillo()
    assert planificar(g,costos,(1.,.55),(1.7,.55)) is None
