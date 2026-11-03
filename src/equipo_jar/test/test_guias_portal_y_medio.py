"""Modo `guias_portal_y_medio`: por cada zona nueva sólo el portal y el medio de la zona
(con holgura a paredes y objetos); Nav2 traza el resto. Cada acción lleva esas dos poses."""
from types import SimpleNamespace

import numpy as np

from equipo_jar.exploracion import Exploracion, Grilla, dilatar


def preparar():
    # Tres zonas en fila: [A | B | C], unidas por puertas; en el medio de B hay una caja.
    libre = np.ones((40, 90), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[:, 30] = False; libre[16:24, 30] = True       # puerta A-B
    libre[:, 60] = False; libre[16:24, 60] = True       # puerta B-C
    msg = SimpleNamespace(info=SimpleNamespace(resolution=0.05, width=90, height=40,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))),
                          header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'holgura_guia_m': 0.30, 'guias_portal_y_medio': True, 'radio_portal_guia_m': 0.35}
    n.grilla = g = Grilla(msg, 0.13, 0.17)
    regiones = np.zeros((40, 90), dtype=np.int32)
    regiones[:, 31:61] = 1; regiones[:, 61:] = 2
    n.sectorizador = SimpleNamespace(regiones=regiones)
    g.registrar_bloque_objeto(2.25, 1.0, 0.26)           # caja en el medio de B, sobre la recta
    mascara = g.transitable & ~dilatar(g.obstaculos_extra, 0.22 / g.res)
    return n, g, regiones, mascara


GUIAS = [((10, 20), False), ((29, 20), True), ((31, 20), True), ((45, 20), False),
         ((59, 20), True), ((61, 20), True), ((75, 20), False)]


def test_portal_y_medio_por_zona_lejos_de_la_caja():
    n, g, regiones, mascara = preparar()
    nuevas = Exploracion.guias_portal_y_centro(n, GUIAS, mascara)
    celdas = [c for c, _ in nuevas]
    # Zona A (donde está el robot) no recibe poses; B y C: portal + medio.
    assert [regiones[c[1], c[0]] for c in celdas] == [1, 1, 2, 2]
    assert [p for _, p in nuevas] == [True, False, True, False]
    holgura = Exploracion.holgura_m(n)
    for c in celdas:
        assert mascara[c[1], c[0]]
        assert holgura[c[1], c[0]] >= 0.18                  # ni pegado a la pared ni a la caja
    caja = g.a_celda(2.25, 1.0)[::-1]
    medio_b = celdas[1]
    assert np.hypot(medio_b[0] - caja[0], medio_b[1] - caja[1]) * g.res >= 0.30   # corrido de la caja


def test_destino_fijo_es_la_ultima_pose_y_acciones_de_dos_poses():
    n, g, regiones, mascara = preparar()
    nuevas = Exploracion.guias_portal_y_centro(n, GUIAS, mascara, destino_fijo=True)
    assert nuevas[-1][0] == (75, 20)
    pasos = [(*g.a_mundo(c[1], c[0]), 0.0) for c, _ in nuevas]
    accion, resto = n.dividir_tramo(pasos)
    assert len(accion) == 2 and len(resto) == 2
    accion, resto = n.dividir_tramo(resto)
    assert len(accion) == 2 and not resto


def test_destino_en_la_misma_zona_es_una_sola_pose():
    n, g, regiones, mascara = preparar()
    nuevas = Exploracion.guias_portal_y_centro(n, [((10, 20), False), ((20, 10), False)], mascara,
                                               destino_fijo=True)
    assert nuevas == [((20, 10), False)]
