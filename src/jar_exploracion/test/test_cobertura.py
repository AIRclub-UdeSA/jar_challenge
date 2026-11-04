"""Pruebas de trazado e integración de cobertura sin comandos al robot."""

import math
from unittest.mock import Mock

import numpy as np
import pytest

from jar_exploracion.trazado import celdas_visibles, trazar_rayo


@pytest.mark.parametrize('destino', [(0, 0), (0, 5), (-5, 0), (5, 3), (-2, -5)])
def test_trazado(destino):
    """El trazado conecta extremos con pasos vecinos y sin repeticiones."""
    celdas = trazar_rayo(0, 0, *destino)
    assert celdas[0] == (0, 0) and celdas[-1] == destino
    assert len(celdas) == len(set(celdas))
    for a, b in zip(celdas, celdas[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


@pytest.fixture
def grilla(monkeypatch):
    """Construye un nodo con TF simulado y publicación interceptada."""
    pytest.importorskip('rclpy')
    from geometry_msgs.msg import TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.clock import Clock
    from jar_exploracion.grilla_cobertura import GrillaCobertura

    monkeypatch.setattr(GrillaCobertura, 'get_clock', lambda self: Clock())
    monkeypatch.setattr(GrillaCobertura, 'get_logger', lambda self: Mock())
    nodo = GrillaCobertura.__new__(GrillaCobertura)
    nodo.info_mapa = None
    nodo.libre_fino = None
    nodo.cubierta = None
    nodo.ultimo_scan = None
    nodo.movimiento_invalido = False
    nodo.map_frame = 'map'
    nodo.timeout_scan_s = 0.5
    nodo.distancia_minima_valida = 0.0
    nodo.factor_escala = 1
    nodo.tiempo_asentamiento_s = 0.0
    nodo.publicar_grilla = Mock()
    mapa = OccupancyGrid()
    mapa.header.frame_id = 'map'
    mapa.info.resolution = 1.0
    mapa.info.width = mapa.info.height = 5
    mapa.info.origin.orientation.w = 1.0
    datos = np.zeros((5, 5), dtype=int)
    datos[:, 2] = 100
    mapa.data = datos.ravel().tolist()
    nodo.recibir_mapa(mapa)
    nodo.recibir_pose_inicial(None)
    tf = TransformStamped()
    tf.transform.translation.x = 0.5
    tf.transform.translation.y = 2.5
    tf.transform.rotation.w = 1.0
    nodo.tf_buffer = Mock()
    nodo.tf_buffer.lookup_transform.return_value = tf
    yield nodo, mapa


def test_pared_scan_y_mapa_repetido(grilla):
    """Una pared fina detiene el rayo y el mismo mapa conserva la cobertura."""
    from sensor_msgs.msg import LaserScan

    nodo, mapa = grilla
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [math.inf]
    nodo.recibir_scan(scan)
    nodo.marcar_cobertura()
    assert nodo.cubierta[2, 0] and nodo.cubierta[2, 1]
    assert not np.any(nodo.cubierta[:, 3:])
    antes = nodo.cubierta.copy()
    nodo.recibir_mapa(mapa)
    assert np.array_equal(antes, nodo.cubierta)
    nodo.marcar_cobertura()
    assert nodo.tf_buffer.lookup_transform.call_count == 1


def test_rechaza_scan_vencido_y_rangos_invalidos(grilla):
    """Datos viejos, NaN e infinito negativo no agregan cobertura."""
    from rclpy.duration import Duration
    from sensor_msgs.msg import LaserScan

    nodo, _ = grilla
    scan = LaserScan()
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.ranges = [math.nan, -math.inf, 0.0]
    nodo.recibir_scan(scan)
    nodo.marcar_cobertura()
    assert not np.any(nodo.cubierta)
    scan.ranges = [math.inf]
    nodo.recibir_scan(scan)
    nodo.scan_procesado = None
    nodo.momento_scan = nodo.get_clock().now() - Duration(seconds=2)
    nodo.marcar_cobertura()
    assert not np.any(nodo.cubierta)


def test_deslizamiento_invalida_cobertura(grilla):
    """No sigue marcando zonas usando una localización que acaba de divergir."""
    from std_msgs.msg import String

    nodo, _ = grilla
    nodo.cubierta[0, 0] = True
    nodo.recibir_estado(String(data='bloqueado_deslizamiento'))
    assert not nodo.cubierta.any()
    assert not nodo.pose_inicial_confirmada
    assert nodo.movimiento_invalido
    nodo.recibir_estado(String(data='inicializando'))
    assert not nodo.movimiento_invalido
    assert not nodo.pose_inicial_confirmada


def test_cobertura_no_atraviesa_esquina_diagonal(grilla):
    """Un rayo diagonal no se cuela por el contacto entre dos obstáculos."""
    from sensor_msgs.msg import LaserScan

    nodo, _ = grilla
    nodo.libre_fino[2, 1] = False
    tf = nodo.tf_buffer.lookup_transform.return_value
    tf.transform.translation.x = 0.5
    tf.transform.translation.y = 1.5
    scan = LaserScan()
    scan.header.frame_id = 'laser'
    scan.range_min = 0.05
    scan.range_max = 10.0
    scan.angle_min = math.pi / 4
    scan.ranges = [math.inf]
    nodo.recibir_scan(scan)
    nodo.marcar_cobertura()
    assert nodo.cubierta[1, 0]
    assert not nodo.cubierta[2, 1]


def test_celdas_visibles_corta_en_paredes_y_esquinas():
    libre = np.ones((11, 11), dtype=bool)
    libre[:, 6] = False                       # pared vertical
    visibles = celdas_visibles(libre, [(5, 2)], 20.0)
    assert visibles[5, 5] and visibles[0, 0]
    assert not visibles[5, 6] and not visibles[5, 8]
    # Diagonal entre dos celdas ocupadas que se tocan en una esquina.
    libre = np.ones((5, 5), dtype=bool)
    libre[1, 2] = libre[2, 1] = False
    assert not celdas_visibles(libre, [(2, 2)], 5.0)[0, 0]
    cerca = celdas_visibles(libre, [(2, 2)], 1.0)
    assert cerca.sum() < celdas_visibles(libre, [(2, 2)], 5.0).sum()
