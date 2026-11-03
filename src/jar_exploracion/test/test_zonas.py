import math

import numpy as np
import pytest

from jar_exploracion import zonas
from jar_exploracion.cobertura import Grilla

RES = 0.05


def test_distancias_por_camino_rodean_la_pared():
    m = np.ones((5, 5), dtype=bool)
    m[0:4, 2] = False                                   # pared con paso abajo
    matriz, indice = zonas.grafo(m, 1.0)
    d = zonas.distancias(matriz, indice, np.array([[0, 0]]))
    assert d[0, 4] == pytest.approx(4 + 4 * math.sqrt(2), abs=1e-6)  # baja, cruza, sube
    assert math.isinf(d[0, 2])                          # la pared


def test_rincon_de_paso_primero_y_sala_grande_despues():
    # Rincón de 600 celdas al lado (3 s); sala de 2300 a 6 s; de la sala (promedio) al rincón,
    # 60 s. Faltando 5 víctimas hay que ver casi todo: mejor no dejar el rincón atrás.
    orden, _ = zonas.ordenar([600, 2300], [40.0, 150.0], viaje_robot=[3.0, 6.0],
                             viaje_entre=[[0.0, 8.0], [60.0, 0.0]], victimas=5)
    assert orden == [0, 1]


def test_rincon_lejos_se_deja_para_despues():
    # El mismo rincón, pero ir y volver cuesta un minuto: retrasa demasiado a la sala.
    orden, _ = zonas.ordenar([200, 2300], [13.0, 150.0], viaje_robot=[60.0, 6.0],
                             viaje_entre=[[0.0, 66.0], [66.0, 0.0]], victimas=2)
    assert orden == [1, 0]


def test_con_mas_victimas_pesa_mas_el_recorrido_total():
    # Dos zonas iguales en camino, una grande lejos y una chica cerca que queda atrás si se
    # va primero a la grande: con 1 víctima conviene la grande; con 5, la chica (cerrar el
    # recorrido sin volver).
    args = ([100, 1000], [10.0, 100.0], [5.0, 10.0], [[0.0, 20.0], [60.0, 0.0]])
    assert zonas.ordenar(*args, victimas=1)[0] == [1, 0]
    assert zonas.ordenar(*args, victimas=5)[0] == [0, 1]


def test_histeresis_mantiene_la_zona_actual_si_casi_empata():
    pesos, tiempos = [100, 100], [10.0, 10.0]
    viaje = [[0.0, 5.0], [5.0, 0.0]]
    assert zonas.ordenar(pesos, tiempos, [5.0, 5.2], viaje)[0][0] == 0
    assert zonas.ordenar(pesos, tiempos, [5.0, 5.2], viaje, actual=1, histeresis=0.1)[0][0] == 1
    assert zonas.ordenar(pesos, tiempos, [5.0, 50.0], viaje, actual=1, histeresis=0.1)[0][0] == 0


def pasillo_con_callejon():
    """Pasillo de 6 × 1 m con un callejón de 1 × 1.5 m hacia arriba en x ∈ [0.5, 1.5]."""
    w, h = int(6.0 / RES), int(3.0 / RES)
    m = np.full((h, w), 100, dtype=np.int8)
    m[2:int(1.0 / RES), 2:-2] = 0                       # pasillo, y de 0 a 1 m
    m[int(1.0 / RES) - 1:int(2.5 / RES), int(0.5 / RES):int(1.5 / RES)] = 0   # callejón
    return Grilla(m.ravel().tolist(), w, h, RES, 0.0, 0.0)


def test_grilla_elige_el_callejon_que_tiene_al_lado():
    g = pasillo_con_callejon()
    g.visto[:, :] = False
    zona = g.zona_objetivo(1.0, 0.5, despeje_min=0.2)
    # Todo es una sola zona sin ver: la elige.
    assert zona is not None and zona[g.celda(1.0, 2.0)]
    # Visto el pasillo cerca del robot: quedan el callejón y el resto del pasillo.
    g.visto[:, : g.celda(2.0, 0.0)[1]] = True
    g.visto[g.celda(0, 1.05)[0]:, :] = False
    g.zona_previa = None                                # sin histéresis por la zona de antes
    zona = g.zona_objetivo(1.0, 0.5, despeje_min=0.2)
    assert zona[g.celda(1.0, 2.0)] and not zona[g.celda(4.0, 0.5)]
    assert len(g.plan) == 2


def test_un_posible_suma_peso_a_su_zona():
    g = pasillo_con_callejon()
    g.visto[:, : g.celda(2.0, 0.0)[1]] = True
    g.visto[g.celda(0, 1.05)[0]:, :] = False
    sin = {p['x']: p['peso'] for p in (g.zona_objetivo(1.0, 0.5, despeje_min=0.2) is not None
                                       and g.plan)}
    con = {p['x']: p['peso'] for p in (g.zona_objetivo(1.0, 0.5, despeje_min=0.2,
                                                       posibles=[(4.0, 0.5)]) is not None
                                       and g.plan)}
    assert sum(con.values()) == sum(sin.values()) + 100


def test_mejor_goal_con_zona_solo_cuenta_esa_zona():
    g = pasillo_con_callejon()
    zona = np.zeros_like(g.visto)
    i, j = g.celda(1.0, 2.0)
    zona[i - 3:i + 4, j - 3:j + 4] = True
    goal = g.mejor_goal(1.0, 0.5, fov=1.0, alcance=1.5, despeje_min=0.2, zona=zona,
                        ganancia_minima=1.0)
    assert goal is not None and goal[1] > 0.6           # mira hacia el callejón, hacia arriba


def test_histeresis_en_la_grilla_sigue_la_zona_que_venia_cubriendo():
    g = pasillo_con_callejon()
    g.visto[:, : g.celda(2.0, 0.0)[1]] = True
    g.visto[g.celda(0, 1.05)[0]:, :] = False
    resto = np.zeros_like(g.visto)
    resto[:, g.celda(2.0, 0.0)[1]:] = True
    g.zona_previa = resto                               # venía cubriendo el resto del pasillo
    zona = g.zona_objetivo(1.0, 0.5, despeje_min=0.2, histeresis=10.0)
    assert zona[g.celda(4.0, 0.5)]
