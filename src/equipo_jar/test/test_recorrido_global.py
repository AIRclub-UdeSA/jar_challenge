"""Orden de sectores: recorrido global robot -> pendientes -> largada (no el más cercano)."""
import itertools
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from equipo_jar.exploracion import Exploracion, Grilla
from equipo_jar.sectores_nbv import SectoresNBV, _bfs_celdas


def laberinto():
    libre = np.ones((40, 60), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[13, 1:45] = False        # tres pasillos horizontales unidos por la derecha
    libre[26, 15:59] = False
    libre[1:13, 30] = False        # y algunos tabiques que generan varios sectores
    libre[27:39, 30] = False
    return libre


def costo(s, seguro, inicio, orden):
    reps, campos = s._distancias_sectores(seguro)
    actual = _bfs_celdas(seguro, inicio)
    total = 0.0
    for r in orden:
        c = reps[r]
        total += actual[c[1], c[0]] * s.resolucion
        actual = campos[r]
    total += actual[s.largada[1], s.largada[0]] * s.resolucion
    return total


def test_recorrido_global_es_optimo_en_mapas_chicos():
    libre = laberinto()
    s = SectoresNBV(libre, libre, 0.1, (5, 5), 0.85, area_minima_sector_m2=0, ancho_minimo_sector_m=0)
    pendientes = list(s.orden[1:])[:7]
    orden, metros = s.recorrido_global(libre, (5, 5), pendientes)
    assert sorted(orden) == sorted(pendientes)
    optimo = min(costo(s, libre, (5, 5), list(p)) for p in itertools.permutations(pendientes))
    assert metros <= optimo + 1e-6
    assert abs(costo(s, libre, (5, 5), orden) - metros) < 1e-6


def test_nunca_peor_que_ir_siempre_al_mas_cercano():
    libre = laberinto()
    s = SectoresNBV(libre, libre, 0.1, (5, 5), 0.85, area_minima_sector_m2=0, ancho_minimo_sector_m=0)
    voraz = list(s.orden)               # orden inicial: voraz por cercanía
    s.indice = 0
    elegido = s.elegir_proximo(libre, (5, 5))
    assert elegido == s.recorrido_planificado[0]
    assert costo(s, libre, (5, 5), s.recorrido_planificado) <= costo(s, libre, (5, 5), voraz) + 1e-6


def test_atasco_en_paso_angosto_lo_saca_de_las_rutas_salvo_junto_al_robot():
    libre = np.ones((10, 60), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    msg = SimpleNamespace(info=SimpleNamespace(resolution=0.05, width=60, height=10,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))),
                          header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'holgura_paso_estrecho_m': 0.25, 'radio_paso_fallido_m': 0.15}
    n.grilla = Grilla(msg, 0.13, 0.17)
    n.portales_fallidos, n.ruta_sector = set(), []
    n.get_logger = lambda: Mock()
    n.marcar_paso_estrecho((1.5, 0.25, 0.0))        # pasillo de 40 cm: holgura 20 cm en el centro
    assert n.portales_fallidos
    seguro = n.grilla.transitable.copy()
    f, c = n.grilla.a_celda(1.5, 0.25)
    n.bloquear_pasos_fallidos(seguro, (5, 5))
    assert not seguro[f, c]
    seguro = n.grilla.transitable.copy()
    n.bloquear_pasos_fallidos(seguro, (c, f))         # el robot está ahí: no se bloquea su lugar
    assert seguro[f, c]
