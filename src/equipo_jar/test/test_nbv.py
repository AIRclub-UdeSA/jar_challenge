"""Regresiones del explorador importado, sin robot ni servidores Nav2."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from geometry_msgs.msg import Twist
from action_msgs.msg import GoalStatus
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan

from equipo_jar.exploracion import (
    APROXIMANDO, COMPORTAMIENTO, DESENCAJANDO, ELIGIENDO, NAVEGANDO,
    REGRESAR, VOLVIENDO, Exploracion, Grilla, bfs,
)
from equipo_jar.sectores_nbv import SectoresNBV


def test_camino_no_corta_esquinas_ni_alcanza_islas():
    libre = np.array([[True, False, True], [False, True, True]], dtype=bool)
    dist = bfs(libre, 0, 0)
    assert dist[0, 0] == 0
    assert np.all(dist[libre][1:] == -1)
    assert bfs(np.zeros((3, 3), dtype=bool), 1, 1) is None


def test_distancia_respeta_pared_y_puerta():
    libre = np.ones((7, 7), dtype=bool)
    libre[:6, 3] = False
    dist = bfs(libre, 1, 1)
    assert dist[1, 5] > 10
    assert np.all(dist[~libre] == -1)


def test_grilla_incorpora_y_retira_obstaculos_sin_cambiar_mapa():
    msg = OccupancyGrid()
    msg.info.width = msg.info.height = 20
    msg.info.resolution = 0.1
    msg.info.origin.position.x = -1.0
    msg.info.origin.position.y = -2.0
    msg.data = [0] * 400
    g = Grilla(msg, 0.1, 0.2)
    assert g.a_celda(*g.a_mundo(8, 9)) == (8, 9)
    extra = np.zeros((20, 20), dtype=bool)
    extra[10, 10] = True
    g.actualizar_obstaculos(extra)
    assert g.libre[10, 10] and not g.transitable[10, 10]
    assert not g.candidatable[10, 10]
    g.actualizar_obstaculos(np.zeros_like(extra))
    assert g.transitable[10, 10] and g.candidatable[10, 10]


def test_nbv_descarta_destino_sin_tiempo_para_volver():
    g = SimpleNamespace(transitable=np.ones((3, 3), dtype=bool),
        candidatable=np.ones((3, 3), dtype=bool), alto=3, ancho=3, res=1.0,
        a_celda=lambda x, y: (int(y), int(x)), a_mundo=lambda f, c: (float(c), float(f)))
    n = SimpleNamespace(grilla=g, lista_negra=[], campo_casa=lambda: np.zeros((3, 3)),
        velocidad_media=lambda: 0.2, tiempo_restante=lambda: 5.0, pose_camara=lambda: None,
        ganancia_en=Mock(return_value=(100., 0.)), cfg={
            'margen_regreso_s': 3., 'paso_candidatos': 1., 'radio_lista_negra': .3,
            'ganancia_minima': 1, 'lambda_distancia': .7, 'peso_giro_nbv': .35})
    assert Exploracion.elegir_vista(n, (0., 0., 0.)) == (None, [])
    n.ganancia_en.assert_not_called()
    n.tiempo_restante = lambda: 100.
    n.lista_negra = [(0., 0.)]
    mejor, candidatos = Exploracion.elegir_vista(n, (0., 0., 0.))
    assert mejor is not None and candidatos
    assert all((x, y) != (0., 0.) for _, x, y, *_ in candidatos)


def test_traslado_a_otro_sector_envia_una_ruta_continua():
    libre = np.ones((20, 30), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[8:12, 10:20] = False
    msg = OccupancyGrid()
    msg.info.width, msg.info.height = 30, 20
    msg.info.resolution = 0.1
    msg.data = np.where(libre, 0, 100).ravel().tolist()
    grilla = Grilla(msg, 0, 0)
    sectores = SectoresNBV(libre, grilla.transitable_base, 0.1, (5, 5), 0.85)
    sectores.indice = 1
    n = SimpleNamespace(sectorizador=sectores, grilla=grilla, ruta_sector=[],
        ruta_region=None, portales_fallidos=set(), lista_negra=[],
        cfg={'paso_maximo_portal_m': 0.30, 'radio_alejar_guia_m': 0.25, 'holgura_guia_m': 0.30,
             'radio_entrada_sector_m': 0.30}, encarar_y_enviar=Mock(return_value=True),
        publicar_debug=Mock(), get_logger=lambda: Mock())
    n.guias_de_ruta = lambda ruta, pose, destino=None, seguro=None: Exploracion.guias_de_ruta(
        n, ruta, pose, destino, seguro=seguro)
    n.mascaras_portales = lambda: Exploracion.mascaras_portales(n)
    n.seleccionar_ruta_portales = lambda *a: Exploracion.seleccionar_ruta_portales(n, *a)
    n.alejar_guias = lambda *a, **k: Exploracion.alejar_guias(n, *a, **k)
    n.en_sector = lambda pose, radio=None: Exploracion.en_sector(n, pose, radio)
    n.holgura_m = lambda: Exploracion.holgura_m(n)
    pose = (*grilla.a_mundo(5, 5), 0.0)
    assert Exploracion.paso_hacia_sector(n, pose)
    x, y, _, _, distancia, etiqueta, estado, guias = n.encarar_y_enviar.call_args.args
    assert (x, y) == guias[-1][:2]
    assert distancia >= math.hypot(x - pose[0], y - pose[1])
    assert len(guias) > 1 and etiqueta.startswith('portal sector') and estado == NAVEGANDO
    n.pose_robot = lambda: (x, y, 0.0)          # llegó a la última guía, dentro del sector
    n.n_atascos, n.fallos_localizacion = 0, []
    Exploracion.al_llegar(n, dict(x=x, y=y, yaw=0.0, etiqueta=etiqueta))
    assert not n.ruta_sector
    assert not n.lista_negra


def test_regreso_por_portales_envia_una_ruta_y_luego_aproxima():
    libre = np.ones((20, 30), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, -1]] = False
    libre[8:12, 10:20] = False
    msg = OccupancyGrid()
    msg.info.width, msg.info.height = 30, 20
    msg.info.resolution = 0.1
    msg.data = np.where(libre, 0, 100).ravel().tolist()
    grilla = Grilla(msg, 0, 0)
    sectores = SectoresNBV(libre, grilla.transitable_base, 0.1, (5, 5), 0.85)
    sectores.indice = 1
    celda_inicio = sectores.ruta_entrada(libre, (5, 5), paso_maximo_m=0.30)[-1][0]
    inicio = (*grilla.a_mundo(celda_inicio[1], celda_inicio[0]), 0.0)
    casa = (*grilla.a_mundo(5, 5), 0.0)
    n = SimpleNamespace(
        cfg={'modo_exploracion': 'sectores_nbv', 'volver_al_inicio': True, 'tolerancia_llegada': 0.2,
             'paso_maximo_portal_m': 0.30, 'radio_alejar_guia_m': 0.25, 'holgura_guia_m': 0.30},
        sectorizador=sectores, grilla=grilla, ruta_regreso=[], largada=casa,
        estado=REGRESAR, goal_handle=None, goal_actual=None,
        fijar_largada=Mock(), pose_robot=lambda: inicio,
        encarar_y_enviar=Mock(return_value=True), publicar_debug=Mock(),
        iniciar_aproximacion=Mock(), reintentar_regreso=Mock(),
        get_logger=lambda: Mock(), cancelacion_cobertura=None, regreso_directo=False)
    n.guias_de_ruta = lambda ruta, pose, destino=None, seguro=None: Exploracion.guias_de_ruta(
        n, ruta, pose, destino, seguro=seguro)
    n.mascaras_portales = lambda: Exploracion.mascaras_portales(n)
    n.seleccionar_ruta_portales = lambda *a: Exploracion.seleccionar_ruta_portales(n, *a)
    n.alejar_guias = lambda *a, **k: Exploracion.alejar_guias(n, *a, **k)
    n.en_sector = lambda pose, radio=None: Exploracion.en_sector(n, pose, radio)
    n.holgura_m = lambda: Exploracion.holgura_m(n)
    Exploracion.volver_a_largada(n)
    assert n.estado == VOLVIENDO
    assert len(n.ruta_regreso) > 1
    assert any(portal for _, portal in n.ruta_regreso)
    x, y, _, _, d, etiqueta, _, guias = n.encarar_y_enviar.call_args.args
    assert (x, y) == guias[-1][:2]
    assert d >= math.hypot(x - inicio[0], y - inicio[1])
    assert len(guias) > 1
    assert etiqueta.startswith('regreso')

    n.pose_robot = lambda: casa
    Exploracion.resultado_goal(n, GoalStatus.STATUS_SUCCEEDED)
    assert not n.ruta_regreso
    n.iniciar_aproximacion.assert_called_once()


def test_aceptacion_tardia_cancela_navegacion():
    h = Mock(accepted=True)
    n = SimpleNamespace(goal_id=2)
    Exploracion.cb_goal_aceptado(n, Mock(result=lambda: h), 1)
    h.cancel_goal_async.assert_called_once()
    h.get_result_async.assert_not_called()


def test_aceptacion_tardia_cancela_giro():
    n = SimpleNamespace(goal_id=1, hist=[], hist_global=[])
    cliente = Mock()
    Exploracion.comportamiento(n, cliente, object(), ELIGIENDO)
    assert n.estado == COMPORTAMIENTO
    callback = cliente.send_goal_async.return_value.add_done_callback.call_args.args[0]
    n.goal_id += 1
    h = Mock(accepted=True)
    callback(Mock(result=lambda: h))
    h.cancel_goal_async.assert_called_once()


@pytest.mark.parametrize('edad,tf,rangos', [
    (1., (0., 0., 0.), [1.]),
    (-1., (0., 0., 0.), [1.]),
    (0., None, [1.]),
    (0., (0., 0., 0.), [math.nan, -math.inf, 0.]),
])
def test_no_usa_lidar_viejo_invalido_o_sin_tf(edad, tf, rangos):
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.header.stamp.sec = 10
    scan.range_min, scan.range_max = .1, 10.
    scan.ranges = rangos
    n = SimpleNamespace(scan=scan, cfg={'base_frame': 'base_footprint', 'timeout_scan_s': .5},
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=int((10+edad)*1e9))),
        lookup=lambda *args: tf)
    assert Exploracion.puntos_scan(n) is None


@pytest.mark.parametrize('estado', [APROXIMANDO, DESENCAJANDO])
def test_perdida_de_pose_detiene_maniobras_directas(estado):
    n = SimpleNamespace(estado=estado, pose_robot=lambda: None, pub_cmd=Mock())
    Exploracion.tick_rapido(n)
    n.pub_cmd.publish.assert_called_once_with(Twist())


@pytest.mark.parametrize('puntos,libre', [(None, 1.), ((np.array([.2]), np.array([0.])), .01)])
def test_aproximacion_frena_sin_lidar_o_con_obstaculo(monkeypatch, puntos, libre):
    monkeypatch.setattr('equipo_jar.exploracion.time.monotonic', lambda: 3.)
    n = SimpleNamespace(t_aprox=0., pose_robot=lambda: (0., 0., 0.), largada=(.2, 0., 0.),
        cfg={'tolerancia_final': .06, 'velocidad_aproximacion': .12},
        puntos_scan=lambda: puntos, espacio_libre=lambda *args: libre, pub_cmd=Mock(), terminar=Mock())
    Exploracion.paso_aproximacion(n)
    n.pub_cmd.publish.assert_called_once_with(Twist())
    n.terminar.assert_not_called()
