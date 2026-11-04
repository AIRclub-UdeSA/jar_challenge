"""nodo_percepcion anota qué vio la cámara y lo publica (memoria_camara.py), de punta a punta."""
import math

import numpy as np
import pytest
import rclpy
from nav_msgs.msg import MapMetaData
from sensor_msgs.msg import CameraInfo, LaserScan

from equipo_jar.mapa import GrillaMapa
from equipo_jar.memoria_camara import ROJO
from equipo_jar.nodo_percepcion import NodoPercepcion

RES = 0.05
LADO = 80                      # mapa de 4 x 4 m, origen en (0, 0)
FX, CX, FY, CY = 271.8, 159.5, 271.8, 119.5
ROBOT = (1.0, 2.0, 0.0)        # mirando a +x
PARED_X = 2.5                  # algo que no está en el mapa, a 1,5 m adelante


@pytest.fixture
def nodo():
    rclpy.init()
    n = NodoPercepcion()
    n.mapa = GrillaMapa(LADO, LADO, RES, 0.0, 0.0, 0.0, [0] * (LADO * LADO))
    info = MapMetaData()
    info.width = info.height = LADO
    info.resolution = RES
    n.info_mapa = info
    camara = CameraInfo()
    camara.width, camara.height = 320, 240
    camara.k = [FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0]
    n.info_camara = camara
    scan = LaserScan()
    scan.angle_min = -math.pi
    scan.angle_increment = 2 * math.pi / 1080
    angulos = scan.angle_min + np.arange(1080) * scan.angle_increment
    rangos = np.full(1080, math.inf)
    adelante = np.abs(angulos) < math.radians(45.0)
    rangos[adelante] = (PARED_X - ROBOT[0]) / np.cos(angulos[adelante])
    scan.ranges = rangos.tolist()
    n.scan = scan
    n.lidar_en_robot = (0.0, 0.0)
    n.frente_lidar_deg = 0.0
    n.altura_camara = 0.109
    n.giro_desde_el_scan = lambda angulo_deg, yaw: 0.0  # sin TF: el robot no giró
    yield n
    n.destroy_node()
    rclpy.try_shutdown()


def mascara(columnas_rojas=None):
    m = np.zeros((240, 320), dtype=np.uint8)
    if columnas_rojas is not None:
        m[90:150, columnas_rojas[0]:columnas_rojas[1]] = 255  # a la altura de una caja a 1,5 m
    return m


def grilla(n):
    return n.memoria.datos().reshape(LADO, LADO)


def test_anota_donde_pegan_los_rayos_que_ve_la_camara(nodo):
    nodo.anotar_vista(mascara(), ROBOT, ROBOT[2])
    g = grilla(nodo)
    vistas = np.argwhere(g > 0)
    assert len(vistas) > 0 and (g[g > 0] == 1).all()
    # Todas sobre la pared x = 2,5 (columna 50), entre y = 2 ± 1,5 tan(30°) (lo que abarca la imagen).
    assert set(vistas[:, 1]) <= {49, 50}
    assert vistas[:, 0].min() >= int((2.0 - 1.5 * math.tan(math.radians(31))) / RES)
    assert vistas[:, 0].max() <= int((2.0 + 1.5 * math.tan(math.radians(31))) / RES)


def test_marca_rojo_solo_donde_la_imagen_tiene_rojo(nodo):
    nodo.anotar_vista(mascara(columnas_rojas=(150, 170)), ROBOT, ROBOT[2])
    g = grilla(nodo)
    rojas = np.argwhere(g == ROJO)
    assert len(rojas) > 0
    # Las columnas 150-170 de la imagen están cerca del centro: y ≈ 2 m (fila 40).
    assert abs(rojas[:, 0].mean() - 40) < 3
    assert (g[g > 0] == ROJO).sum() < (g > 0).sum()  # el resto de la pared, sin rojo


def test_publica_la_grilla_con_la_geometria_del_mapa(nodo):
    publicados = []
    nodo.pub_vistas.publish = publicados.append
    nodo.publicar_vistas()
    assert publicados == []                 # todavía no vio nada
    nodo.anotar_vista(mascara(), ROBOT, ROBOT[2])
    nodo.publicar_vistas()
    assert len(publicados) == 1
    msg = publicados[0]
    assert msg.header.frame_id == 'map'
    assert msg.info.width == LADO and len(msg.data) == LADO * LADO
    assert (np.array(msg.data).reshape(LADO, LADO) == grilla(nodo)).all()
    nodo.publicar_vistas()
    assert len(publicados) == 1             # sin cambios, no se vuelve a publicar
