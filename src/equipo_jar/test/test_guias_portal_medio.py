"""Guías por zona: portal, medio de la zona y a lo sumo un punto de unión; nunca rozando objetos."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from equipo_jar.exploracion import Exploracion, Grilla
from equipo_jar.planificador import segmento_transitable
from equipo_jar.planificador import ruta_por_portales


def preparar():
    # Tres zonas en fila: [A | B | C], unidas por puertas; en B hay una caja.
    libre = np.ones((40, 90), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[:, 30] = False; libre[16:24, 30] = True       # puerta A-B
    libre[:, 60] = False; libre[16:24, 60] = True       # puerta B-C
    msg = SimpleNamespace(info=SimpleNamespace(resolution=0.05, width=90, height=40,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))),
                          header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'holgura_guia_m': 0.30}
    n.grilla = g = Grilla(msg, 0.13, 0.17)
    regiones = np.zeros((40, 90), dtype=np.int32)
    regiones[:, 31:61] = 1; regiones[:, 61:] = 2
    n.sectorizador = SimpleNamespace(regiones=regiones)
    g.registrar_bloque_objeto(2.25, 1.0, 0.26)           # caja en el medio de B, sobre la recta
    return n, g, regiones


def test_cada_zona_intermedia_tiene_portal_medio_y_portal_sin_rozar_la_caja():
    n, g, regiones = preparar()
    from equipo_jar.exploracion import dilatar
    mascara = g.transitable & ~dilatar(g.obstaculos_extra, 0.22 / g.res)
    # Guías "originales" en zigzag por la recta del medio (y = 1.0 m)
    guias = [((10, 20), False), ((29, 20), True), ((31, 20), True), ((36, 14), False),
             ((40, 26), False), ((52, 14), False), ((59, 20), True), ((61, 20), True), ((75, 20), False)]
    nuevas = Exploracion.guias_portal_medio(n, guias, mascara)
    zona_b = [c for c, _ in nuevas if regiones[c[1], c[0]] == 1]
    assert zona_b[0] == (31, 20) and zona_b[-1] == (59, 20)      # portales conservados
    assert len(zona_b) <= 5                                    # entrada, [unión], medio, [unión], salida
    celdas = [c for c, _ in nuevas]
    for a, b in zip(celdas, celdas[1:]):
        assert segmento_transitable(mascara, a, b)             # ninguna recta toca la caja


def test_zona_con_muchas_curvas_no_vuelve_a_enviar_toda_la_ruta():
    libre = np.zeros((72, 72), dtype=bool)
    for i, fila in enumerate(range(4, 65, 10)):
        libre[fila:fila + 5, 4:67] = True
        if i < 6:
            col = 62 if i % 2 == 0 else 4
            libre[fila:fila + 15, col:col + 5] = True
    msg = SimpleNamespace(info=SimpleNamespace(resolution=0.05, width=72, height=72,
        origin=SimpleNamespace(position=SimpleNamespace(x=0., y=0.))),
        header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'holgura_guia_m': 0.30, 'radio_alejar_guia_m': 0.25}
    n.grilla = g = Grilla(msg, 0., 0.)
    regiones = np.zeros_like(libre, dtype=np.int32)
    n.sectorizador = SimpleNamespace(regiones=regiones)
    inicio, destino = (6, 6), (64, 66)
    ruta = ruta_por_portales(libre, regiones, {}, inicio, destino, g.res, g.res)
    assert len(ruta) > 300
    pose = (*g.a_mundo(inicio[1], inicio[0]), 0.)
    meta = (*g.a_mundo(destino[1], destino[0]), 0.)
    pasos, distancia = n.guias_de_ruta(ruta, pose, meta, seguro=libre)
    assert 3 <= len(pasos) <= 5
    assert pasos[0][:2] == pose[:2] and pasos[-1] == meta
    assert distancia > 15.  # Se sigue calculando con el camino real, no con las rectas entre metas.
    from equipo_jar.exploracion import bfs
    for x, y, _ in pasos:
        f, c = g.a_celda(x, y)
        assert libre[f, c] and bfs(libre, inicio[1], inicio[0])[f, c] >= 0
