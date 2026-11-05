"""Huecos angostos entre una caja y una pared: se cierran del más angosto al más ancho,
salvo el último que mantiene conectado el mapa."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from equipo_jar.exploracion import Exploracion, Grilla


def nodo(libre, res=0.05):
    alto, ancho = libre.shape
    msg = SimpleNamespace(info=SimpleNamespace(resolution=res, width=ancho, height=alto,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))),
                          header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'lado_bloque_objeto_m': 0.26, 'celdas_max_cortadas_por_bloque': 20, 'ancho_paso_minimo_m': 0.40,
             'cerrar_huecos_objetos': True}   # la función sigue disponible (apagada por defecto)
    n.grilla = Grilla(msg, 0.13, 0.17)
    n.bloques_objetos, n.sectorizador = [], None
    n.publicar_mapa_objetos = lambda: None
    n.get_logger = lambda: Mock()
    n.pose_robot = lambda: (0.3, 0.3, 0.0)
    n.dist_pared = Exploracion.campo_distancia_paredes(n.grilla)
    return n


def sala():
    libre = np.ones((60, 60), dtype=bool)          # 3 x 3 m
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    return libre


def test_hueco_angosto_contra_la_pared_se_cierra_si_hay_otro_camino():
    libre = sala()
    libre[30, 1:20] = False                          # pared horizontal hasta x = 1.0 m
    n = nodo(libre)
    n.registrar_objeto_local((1.22, 1.62))          # caja junto a la punta: hueco diagonal ~0.30 m
    g = n.grilla
    assert n.bloques_objetos
    f, c = g.a_celda(1.08, 1.50)                     # entre la punta de la pared y la caja
    assert g.bloques_conocidos[f, c]
    f, c = g.a_celda(2.5, 2.5)                       # lo demás sigue accesible
    assert n.alcanzable_desde_robot(g.transitable)[f, c]


def test_si_la_caja_esta_en_un_cuello_queda_abierto_el_hueco_mas_ancho():
    libre = sala()
    libre[30, :] = False                             # pared que separa dos mitades...
    libre[30, 24:36] = True                          # ...con una puerta de 0.60 m (x 1.2 a 1.8)
    n = nodo(libre)
    n.registrar_objeto_local((1.42, 1.52))          # caja en la puerta, corrida a la izquierda
    g = n.grilla
    f, c = g.a_celda(1.5, 2.6)                       # la otra mitad sigue alcanzable
    assert n.alcanzable_desde_robot(g.transitable)[f, c]
    f, c = g.a_celda(1.24, 1.52)                     # el hueco angosto (izquierda) quedó cerrado
    assert g.bloques_conocidos[f, c]
