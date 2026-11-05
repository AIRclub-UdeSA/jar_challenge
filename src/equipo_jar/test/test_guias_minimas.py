"""Entrada, intermedio y salida: pocas referencias y un portal por acción."""
from collections import deque
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from builtin_interfaces.msg import Time

from equipo_jar.exploracion import Exploracion, Grilla
from equipo_jar.planificador import ruta_por_portales


def preparar():
    libre = np.ones((40, 90), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[:, 30] = False; libre[16:24, 30] = True       # puerta A-B
    msg = SimpleNamespace(info=SimpleNamespace(resolution=0.05, width=90, height=40,
                                               origin=SimpleNamespace(position=SimpleNamespace(x=0.0, y=0.0))),
                          header=SimpleNamespace(frame_id='map'), data=np.where(libre, 0, 100).ravel().tolist())
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'holgura_guia_m': 0.30, 'radio_alejar_guia_m': 0.25}
    n.grilla = Grilla(msg, 0.13, 0.17)
    regiones = np.zeros((40, 90), dtype=np.int32)
    regiones[:, 31:] = 1
    n.sectorizador = SimpleNamespace(regiones=regiones)
    return n


def calcular(n, inicio, destino):
    g = n.grilla
    ruta = ruta_por_portales(g.transitable, n.sectorizador.regiones, {}, inicio, destino, g.res, g.res)
    assert len(ruta) > 20  # Muestreo denso: no debe llegar así al servidor de Nav2.
    pose = (*g.a_mundo(inicio[1], inicio[0]), 0.)
    meta = (*g.a_mundo(destino[1], destino[0]), 0.)
    return n.guias_de_ruta(ruta, pose, meta, seguro=g.transitable)[0]


def test_cada_zona_conserva_entrada_intermedio_y_salida():
    n = preparar()
    g = n.grilla
    pasos = calcular(n, (10, 20), (70, 20))
    for region in (0, 1):
        zona = [p for p in pasos if n.sectorizador.regiones[g.a_celda(*p[:2])] == region]
        assert len(zona) == 3  # También la primera y la última zona llevan intermedio.
        assert zona[0][0] < zona[1][0] < zona[2][0]
    assert pasos[-1][:2] == g.a_mundo(20, 70)


def test_progreso_con_distancia_restante_de_nav2():
    n = Exploracion.__new__(Exploracion)
    n.cfg = {'tiempo_atascado': 10.0, 'distancia_atascado': 0.12}
    reloj = {'t': 0.0}
    n.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int(reloj['t'] * 1e9)))
    n.hist = deque()
    n.goal_actual = dict(pasos=[(3.0, 0.0, 0.0)], pasos_restantes=1, distancias_guias=[3.0],
                         mejor_avance=None, t_progreso=0.0)
    n.cb_feedback_ruta = lambda m, gid: Exploracion.cb_feedback_ruta(n, m, gid)
    n.goal_id = 1
    # Rodea una pared: la recta al destino no mejora, pero a Nav2 le falta cada vez menos.
    for t, restante in [(0, 5.0), (4, 4.5), (8, 4.0), (12, 3.5), (16, 3.0)]:
        reloj['t'] = t
        n.cb_feedback_ruta(SimpleNamespace(feedback=SimpleNamespace(number_of_poses_remaining=1,
                                                                    distance_remaining=restante)), 1)
        assert not Exploracion.detectar_atasco(n, (0.0, float(t) * 0.1, 0.0))


def test_destino_lejano_va_por_partes_de_portal_en_portal():
    n = preparar()
    n.sectorizador.regiones[:, 61:] = 2                  # tres zonas en fila: A | B | C
    n.grilla.libre[:, 60] = True
    pasos = calcular(n, (10, 20), (75, 20))
    assert len(pasos) == 9  # Tres referencias por zona, guardadas internamente.
    pendientes = pasos
    tramos = []
    while pendientes:
        tramo, resto = n.dividir_tramo(pendientes)
        assert len(tramo) <= 6 and len(resto) < len(pendientes)
        tramos.append(tramo)
        pendientes = resto
    assert len(tramos) == 3 and tramos[-1][-1] == pasos[-1]
    assert pasos[-1] not in tramos[0]  # No se manda el destino lejano en la primera acción.


def test_el_servidor_recibe_un_tramo_acotado_aunque_el_llamador_pase_una_ruta_densa():
    n = preparar()
    g = n.grilla
    n.cfg['velocidad_estimada'] = 0.2
    n.nav_ruta = Mock(server_is_ready=lambda: True)
    n.heredado_vivo = lambda: False
    n.goal_id, n.hist, n.bt_portales = 0, deque(), ''
    n.pose_robot = lambda: (*g.a_mundo(20, 10), 0.)
    n.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1,
        to_msg=lambda: Time()))
    n.get_logger = lambda: Mock()
    n.cobertura, n.tiempo_transcurrido = lambda: 0., lambda: 1.
    pasos = [(*g.a_mundo(20, c), 0.) for c in range(10, 76)]
    assert n.enviar_goal(*pasos[-1], 10., 4., pasos=pasos)
    goal = n.nav_ruta.send_goal_async.call_args.args[0]
    assert len(goal.poses) == 4  # Entrada, intermedio, salida y cruce al portal siguiente.
    assert n.goal_actual['continuacion'] and n.goal_actual['meta_tramo'] != pasos[-1]
    assert n.goal_actual['pasos'][0] == pasos[0]
    assert n.goal_actual['pasos'][-2] == pasos[20]  # Último punto de la zona A (columna 30).


def test_nbv_global_tambien_acota_una_ruta_densa_sin_sectores():
    n = preparar()
    n.sectorizador = None
    pasos = [(i * 0.05, 1., 0.) for i in range(100)]
    tramo, resto = n.dividir_tramo(pasos)
    assert len(tramo) == 3 and not resto
    assert tramo[0] == pasos[0] and tramo[-1] == pasos[-1]


def test_preparar_goal_conserva_el_intermedio_y_la_salida_en_cualquier_destino():
    n = preparar()
    g = n.grilla
    pasos = calcular(n, (10, 20), (70, 20))
    n.pose_robot = lambda: (*g.a_mundo(20, 10), 0.)
    n.ruta_a_destino = Mock(return_value=(pasos, 4.))
    n.direccion_de_ruta = lambda *a: 0.
    n.spin_cli = Mock(server_is_ready=lambda: False)
    n.enviar_goal = Mock(return_value=True)
    from equipo_jar.exploracion import NAVEGANDO, VOLVIENDO
    for etiqueta, estado in (('vista', NAVEGANDO), ('regreso', VOLVIENDO)):
        assert n.encarar_y_enviar(*pasos[-1], 10., 4., etiqueta, estado)
        assert n.enviar_goal.call_args.kwargs['pasos'] == pasos
