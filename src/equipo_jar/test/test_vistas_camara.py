"""Puntos NBV: piso único, orientación óptica y margen ante objetos nuevos."""
import math
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo
from pathlib import Path
import yaml

from equipo_jar.exploracion import Exploracion, Grilla
from equipo_jar.observacion_objetos import ObjetosFueraMapa


def _grilla():
    mapa = OccupancyGrid()
    mapa.info.width = mapa.info.height = 60
    mapa.info.resolution = 0.1
    mapa.info.origin.position.x = mapa.info.origin.position.y = -3.0
    mapa.data = [0] * 3600
    return Grilla(mapa, 0.15, 0.25)


def test_ganancia_cuenta_cada_celda_una_vez_por_vista():
    f = np.zeros((4, 3), dtype=int)
    c = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 2], [3, 3, 3]])
    n = SimpleNamespace()
    n.rayos_desde = lambda fila, col: (f, c, np.ones((4, 3), dtype=bool))
    n.grilla = SimpleNamespace(ancho=4, libre=np.ones((1, 4), dtype=bool),
                               bloques_conocidos=np.zeros((1, 4), dtype=bool))
    n.visto = np.zeros((1, 4), dtype=bool)
    n.peso_rayo, n.nrayos, n.ventana = np.ones(3), 4, 2
    ganancia, yaw = Exploracion.ganancia_en(n, 0, 0)
    assert ganancia == 3.0  # Los seis impactos de la ventana cubren tres celdas.
    assert yaw == pytest.approx(-math.pi / 4)


def test_cono_marca_lo_que_mira_el_eje_optico_sin_inventar_piso_alrededor():
    g = _grilla()
    angulos = np.linspace(-math.pi, math.pi, 72, endpoint=False)
    dist = np.arange(1, 21)
    n = Exploracion.__new__(Exploracion)
    n.grilla, n.visto, n.angulos = g, np.zeros_like(g.libre), angulos
    n.ray_dx, n.ray_dy = np.cos(angulos)[:, None] * dist, np.sin(angulos)[:, None] * dist
    n.fov = math.radians(60)
    n.marcar_visto((0.2, 0.0, math.pi / 2))  # Cámara al norte, base hacia el este.
    assert n.visto[g.a_celda(0.2, 1.0)]
    assert not n.visto[g.a_celda(1.2, 0.0)]
    assert not n.visto[g.a_celda(0.2, -1.0)]
    assert not n.visto[g.a_celda(0.4, 0.0)]


def test_obstaculo_nuevo_tiene_margen_de_chasis_y_de_parada():
    g = _grilla()
    extra = np.zeros_like(g.libre)
    extra[30, 30] = True
    g.actualizar_obstaculos(extra)
    assert not g.transitable[30, 31]  # a 10 cm del obstáculo: no pasa el centro del chasis
    assert g.transitable[30, 32] and not g.candidatable[30, 32]
    assert g.candidatable[30, 33]


def test_intrinseca_define_el_fov_y_el_frame_usado_para_apuntar():
    n = SimpleNamespace(frame_camara=None, nrayos=72, fov=math.radians(70))
    info = CameraInfo(width=320, height=240, k=[320., 0., 159.5, 0., 320., 119.5, 0., 0., 1.])
    info.header.frame_id = 'color_optico'
    Exploracion.cb_info_camara(n, info)
    assert n.frame_camara == 'color_optico'
    assert math.degrees(n.fov) == pytest.approx(53.1301, abs=0.001)
    assert n.ventana < 14  # No usar el FOV nominal de 70° cuando la imagen cubre menos.


def _nbv(ganancia, camara):
    g = SimpleNamespace(transitable=np.ones((1, 3), bool),
        candidatable=np.array([[True, False, True]]), alto=1, ancho=3, res=1.0,
        a_celda=lambda x, y: (0, int(x)), a_mundo=lambda f, c: (float(c), 0.0))
    return SimpleNamespace(grilla=g, lista_negra=[], campo_casa=lambda: np.zeros((1, 3)),
        pose_camara=lambda: camara, velocidad_media=lambda: 0.5, tiempo_restante=lambda: 100.0,
        ganancia_en=ganancia, cfg=dict(margen_regreso_s=5.0, paso_candidatos=1.0,
            radio_lista_negra=0.3, ganancia_minima=1, lambda_distancia=0.7, peso_giro_nbv=0.35))


def test_entre_dos_vistas_equivalentes_prefiere_menos_giro():
    n = _nbv(lambda f, c: (100.0, 0.0 if c == 0 else math.pi), (1.0, 0.0, 0.0))
    vista, _ = Exploracion.elegir_vista(n, (1.0, 0.0, 0.0))
    assert vista[1] == 0.0


def test_yaw_del_destino_compensa_el_montaje_de_la_camara():
    n = _nbv(Mock(return_value=(100.0, math.pi / 2)), (1.0, 0.0, math.pi / 2))
    vista, _ = Exploracion.elegir_vista(n, (1.0, 0.0, 0.0))
    assert vista[3] == pytest.approx(0.0)


def test_no_infla_por_segunda_vez_la_franja_de_nav2_en_un_pasillo():
    mapa = OccupancyGrid()
    mapa.info.width, mapa.info.height, mapa.info.resolution = 20, 14, 0.1
    datos = np.full((14, 20), 100, dtype=np.int8)
    datos[5:9, 1:19] = 0  # Pasillo de 40 cm que admite el chasis.
    mapa.data = datos.ravel().tolist()
    g = Grilla(mapa, 0.13, 0.17)
    assert g.transitable[6, 10]
    costo = datos.copy()
    costo[5, 1:19] = costo[8, 1:19] = 99  # Coste de seguridad de Nav2, no un objeto físico.
    costmap = OccupancyGrid()
    costmap.info, costmap.data = mapa.info, costo.ravel().tolist()
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / 'params' / 'exploracion.yaml').read_text())
    n = SimpleNamespace(grilla=g, cfg=cfg['exploracion']['ros__parameters'], estado='ELIGIENDO',
        loc_mala_desde=None, dist_pared=Exploracion.campo_distancia_paredes(g),
        objetos_fuera_mapa=ObjetosFueraMapa(), registrar_objeto_local=Mock())
    Exploracion.cb_costmap(n, costmap)
    assert g.transitable[6, 10] and g.transitable[7, 10]
    assert not g.obstaculos_extra.any()


def test_inicio_bloqueado_no_descarta_todos_los_sectores():
    g = _grilla()
    g.transitable[30, 30] = False
    sectores = SimpleNamespace(actual=2, avanzar_inaccesible=Mock())
    n = SimpleNamespace(grilla=g, sectorizador=sectores, get_logger=lambda: Mock(),
                        cancelar_actual=Mock(), iniciar_escape=Mock(), n_atascos=0)
    pose = (*g.a_mundo(30, 30), 0.0)
    assert Exploracion.paso_hacia_sector(n, pose)
    n.iniciar_escape.assert_called_once()
    sectores.avanzar_inaccesible.assert_not_called()
