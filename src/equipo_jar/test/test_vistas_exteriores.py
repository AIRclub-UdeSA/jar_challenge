"""Zonas visibles tras un paso que no permite entrar al robot."""
import math
from unittest.mock import Mock

import numpy as np
import pytest
from nav_msgs.msg import OccupancyGrid

from equipo_jar.exploracion import ELIGIENDO, NAVEGANDO, Exploracion, Grilla, bfs
from equipo_jar.sectores_nbv import SectoresNBV


def nodo(abertura=True):
    libre = np.ones((24, 44), dtype=bool)
    libre[[0, -1], :] = False
    libre[:, [0, 22, -1]] = False
    libre[12, 22] = abertura  # 10 cm: visible, demasiado angosto para el chasis.
    msg = OccupancyGrid()
    msg.info.width, msg.info.height, msg.info.resolution = 44, 24, 0.1
    msg.data = np.where(libre, 0, 100).ravel().tolist()
    n = Exploracion.__new__(Exploracion)
    n.grilla = g = Grilla(msg, 0.13, 0.17)
    n.sectorizador = s = SectoresNBV(libre, g.transitable_base, g.res, (10, 12), 0.68)
    n.visto = libre & (s.regiones >= 0)  # Los sectores transitables ya fueron revisados.
    s.avanzar_cubiertos(n.visto)
    assert s.actual is None and (s.regiones[1:-1, 23:-1] < 0).all()
    n.nrayos, n.ventana = 72, 14
    n.angulos = np.linspace(-math.pi, math.pi, n.nrayos, endpoint=False)
    distancias = np.arange(1, 21)
    n.ray_dx = np.cos(n.angulos)[:, None] * distancias
    n.ray_dy = np.sin(n.angulos)[:, None] * distancias
    n.peso_rayo, n.fov = np.ones(20), math.radians(70)
    n.cfg = dict(margen_regreso_s=5., paso_candidatos=0.3, radio_lista_negra=0.35,
                 ganancia_minima=25, lambda_distancia=0.7, peso_giro_nbv=0.35)
    n.lista_negra = []
    n.campo_casa = lambda: bfs(g.transitable, 12, 10)
    n.velocidad_media, n.tiempo_restante = lambda: 0.5, lambda: 500.
    n.pose_robot = lambda: (*g.a_mundo(12, 10), 0.)
    n.pose_camara = n.pose_robot
    return n


def test_elige_vista_parcial_desde_el_lado_alcanzable_sin_inventar_cobertura():
    n = nodo()
    g = n.grilla
    antes = n.visto.copy()
    mejor, top = n.elegir_vista(n.pose_robot())
    assert mejor is not None and top
    _, x, y, yaw, ganancia, _ = mejor
    f, c = g.a_celda(x, y)
    assert g.candidatable[f, c] and c < 22
    assert n.campo_casa()[f, c] >= 0 and n.campo_casa()[12, 30] < 0
    assert 0 < ganancia < np.count_nonzero(n.sectorizador.mascara_observacion())
    np.testing.assert_array_equal(n.visto, antes)  # Elegir una vista todavía no la observa.
    n.marcar_visto((x, y, yaw))
    assert (n.visto[:, 23:] & ~antes[:, 23:]).any()
    assert (g.libre[:, 23:] & ~n.visto[:, 23:]).any()


def test_sin_vistas_restantes_termina_aunque_quede_piso_oculto():
    n = nodo()
    # Unión de lo visible desde todas las paradas seguras del lado alcanzable.
    for f, c in zip(*np.nonzero(n.grilla.candidatable & (n.campo_casa() >= 0))):
        ff, cc, visible = n.rayos_desde(f, c)
        n.visto[ff[visible], cc[visible]] = True
    assert (n.sectorizador.mascara_observacion() & ~n.visto).any()
    assert n.elegir_vista(n.pose_robot()) == (None, [])


def test_no_busca_observar_a_traves_de_una_pared():
    n = nodo(abertura=False)
    assert n.elegir_vista(n.pose_robot()) == (None, [])
    assert not n.visto[:, 23:].any()


def test_tambien_permite_mirar_un_sector_activo_desde_afuera():
    n = nodo()
    s = n.sectorizador
    s.regiones[s.libre & (s.regiones < 0)] = 7
    s.orden, s.indice = [7], 0
    mejor, _ = n.elegir_vista(n.pose_robot())
    assert mejor is not None
    f, c = n.grilla.a_celda(*mejor[1:3])
    assert not s.dentro_actual((c, f)) and n.grilla.candidatable[f, c]


@pytest.mark.parametrize('sin_vistas_interior', [False, True])
def test_ciclo_busca_vistas_exteriores_antes_de_regresar(sin_vistas_interior):
    n = nodo()
    if sin_vistas_interior:
        vista_exterior = n.elegir_vista(n.pose_robot())
        n.sectorizador.regiones[12, 10] = 7
        n.visto[12, 10] = False
        n.sectorizador.orden, n.sectorizador.indice = [7], 0
        # Sector pendiente sin ganancia interior, pero quedan las zonas excluidas.
        n.elegir_vista = Mock(side_effect=[(None, []), vista_exterior])
    n.estado, n.goal_actual, n.objeto_observado = ELIGIENDO, None, None
    n.cfg.update(modo_exploracion='sectores_nbv', cantidad_victimas=5,
                 tolerancia_llegada=0.1, tiempo_observacion_objeto_s=0.8)
    n.reportes, n.t_ult_revision_lc, n.espera_mirar_hasta = 0, float('inf'), 0.
    n.cancelacion_cobertura = None
    n.log_estado, n.fijar_largada, n.completar_huecos = Mock(), Mock(), Mock()
    n.nav2_activo = lambda: True
    n.segundos_para_volver = lambda *a: 10.
    n.considerar_objeto_nuevo = lambda *a: False
    n.vigilancia_global = lambda *a: False
    n.elegir_confirmacion = lambda *a: None
    n.preparar_sectores = Mock()
    n.en_sector = lambda *a: True
    n.ruta_sector, n.ruta_region = [], None
    n.get_logger = lambda: Mock()
    n.encarar_y_enviar, n.pedir_regreso, n.publicar_debug = Mock(), Mock(), Mock()
    n.ciclo()
    n.pedir_regreso.assert_not_called()
    assert n.sectorizador.actual is None
    assert n.encarar_y_enviar.call_args.args[-2:] == ('vista exterior', NAVEGANDO)
