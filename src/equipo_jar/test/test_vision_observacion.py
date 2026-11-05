"""LiDAR + RGB + TF: objetos de cualquier color y paredes excluidas por geometría."""
import json
import math
from collections import deque
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
import cv2
from geometry_msgs.msg import TransformStamped
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan

from equipo_jar.deteccion_victimas import DeteccionVictimas
from equipo_jar.exploracion import Exploracion
from equipo_jar.observacion_objetos import ObjetosFueraMapa


def _detector(monkeypatch):
    reloj = {'t': 1.0}
    monkeypatch.setattr('equipo_jar.deteccion_victimas.time.monotonic', lambda: reloj['t'])
    n = DeteccionVictimas.__new__(DeteccionVictimas)
    params = Path(__file__).resolve().parents[1] / 'params' / 'victimas.yaml'
    n.cfg = yaml.safe_load(params.read_text())['deteccion_victimas']['ros__parameters']
    n.grupos, n.grupos_objetos = [], []
    n.scans, n.imagenes_pendientes = deque(maxlen=30), deque(maxlen=10)
    n.reportes, n.n_max = 0, 2
    n.pub, n.pub_observaciones, n.pub_pendientes = Mock(), Mock(), Mock()
    n.get_logger = lambda: Mock()
    n.publicar_debug = Mock()
    n.get_clock = lambda: SimpleNamespace(now=lambda: Time(nanoseconds=int((100 + reloj['t']) * 1e9)))
    libre = np.ones((100, 100), dtype=bool)
    n.mapa = dict(frame='map', res=0.05, ox=-2.5, oy=-2.5, alto=100, ancho=100,
                  libre=libre, dist=np.ones_like(libre, float), dist_libre=np.zeros_like(libre, float))
    n.loc, n.recorrido, n.t_mapa = None, None, 0.0
    n.K = (271.8, 271.8, 159.5, 119.5)
    n.giro, n.ult_proc, n.ultimo_stamp_imagen = 0.0, 0.0, None
    n.diag = dict.fromkeys(('imagenes', 'girando', 'con_rojo', 'manchas', 'sin_tf',
                           'sin_laser', 'cara_en_pared', 'centro_en_pared', 'aceptadas'), 0)
    identidad = TransformStamped()
    identidad.transform.rotation.w = 1.0
    camara = TransformStamped()
    # Cámara óptica (+Z hacia adelante) -> base (+X hacia adelante).
    q = camara.transform.rotation
    q.x, q.y, q.z, q.w = -0.5, 0.5, -0.5, 0.5
    camara.transform.translation.z = 0.15
    n.tf = lambda padre, hijo, stamp=None: camara if hijo == 'camara_optica' else identidad
    n.tf_buffer = SimpleNamespace(lookup_transform_full=Mock(return_value=identidad),
        can_transform=Mock(return_value=True), can_transform_full=Mock(return_value=True))
    angulos = np.linspace(-0.14, 0.14, 15)
    n.scan = LaserScan(angle_min=-0.14, angle_increment=0.02, range_min=0.1,
                       range_max=5.0, ranges=[0.75 / math.cos(a) for a in angulos])
    n.scan.header.frame_id = 'laser'
    return n, reloj


def _imagen(n, reloj, t, color=(255, 0, 0), actualizar_scan=True, imagen=None):
    reloj['t'] = t
    if imagen is None:
        imagen = np.zeros((240, 320, 3), dtype=np.uint8)
        imagen[92:184, 118:202] = color
    msg = Image(height=240, width=320, encoding='bgr8', step=960, data=imagen.tobytes())
    msg.header.frame_id = 'camara_optica'
    msg.header.stamp = n.get_clock().now().to_msg()
    if actualizar_scan:
        scan = deepcopy(n.scan)
        scan.header.stamp = msg.header.stamp
        n.cb_scan(scan)
    n.cb_imagen(msg)
    reloj['t'] += 0.13
    n.procesar_pendientes()
    return msg


@pytest.mark.parametrize('color,bgr', [('azul', (255, 0, 0)), ('amarillo', (0, 255, 255)),
    ('verde', (0, 255, 0)), ('neutro', (160, 160, 160)), ('neutro', (0, 0, 0)),
    ('neutro', (255, 255, 255)), ('cian', (255, 255, 0)),
    ('naranja', (0, 128, 255)), ('magenta', (255, 0, 255))])
def test_cualquier_color_se_localiza_sin_reportar_victimas(monkeypatch, color, bgr):
    n, reloj = _detector(monkeypatch)
    nav = SimpleNamespace(grilla=SimpleNamespace(frame='map'), ultimo_stamp_vision=0,
                          get_clock=n.get_clock, objetos_fuera_mapa=ObjetosFueraMapa())
    n.pub_observaciones.publish.side_effect = lambda msg: Exploracion.cb_observaciones(nav, msg)
    for t in np.arange(1.0, 2.6, 0.2):
        _imagen(n, reloj, t, bgr)
    obs = nav.observaciones_vision['objetos'][0]
    assert (obs['x'], obs['y']) == pytest.approx((0.85, 0.0), abs=0.015)
    assert obs['color'] == color and obs['tipo'] == 'objeto' and obs['confirmado']
    assert abs(obs['error_angular']) < 0.01
    assert n.grupos == [] and len(n.grupos_objetos) == 1
    assert nav.objetos_fuera_mapa.confirmados(reloj['t']) == []  # RGB no sustituye /scan persistente
    n.revisar_reportes()
    n.pub.publish.assert_not_called()
    primera = json.loads(n.pub_observaciones.publish.call_args_list[0].args[0].data)['objetos'][0]
    assert primera['nuevo'] and primera['tipo'] == 'objeto_nuevo'
    assert primera['id'] == obs['id']


def test_rojo_conserva_reporte_y_azul_sigue_tras_limite_de_victimas(monkeypatch):
    n, reloj = _detector(monkeypatch)
    for t in np.arange(1.0, 2.6, 0.2):
        _imagen(n, reloj, t, (0, 0, 255))
    n.revisar_reportes()
    assert n.reportes == 1 and n.pub.publish.call_count == 1
    n.reportes = n.n_max
    _imagen(n, reloj, 3.0)
    assert json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos'][0]['color'] == 'azul'
    assert n.pub.publish.call_count == 1


def test_no_cuenta_imagen_repetida_laser_viejo_o_tf_ausente(monkeypatch):
    n, reloj = _detector(monkeypatch)
    foto = _imagen(n, reloj, 1.0)
    reloj['t'] = 1.2
    n.cb_imagen(foto)  # mismo stamp: no agrega evidencia
    assert n.grupos_objetos[0].n == 1
    _imagen(n, reloj, 2.0, actualizar_scan=False)
    assert n.pub_observaciones.publish.call_count == 1
    n.tf = lambda *args: None
    _imagen(n, reloj, 2.2)
    assert n.pub_observaciones.publish.call_count == 1


def test_dos_imagenes_con_el_mismo_scan_no_duplican_evidencia_lidar(monkeypatch):
    n, reloj = _detector(monkeypatch)
    _imagen(n, reloj, 1.0)
    _imagen(n, reloj, 1.2, actualizar_scan=False)
    assert n.pub_observaciones.publish.call_count == 2  # puede actualizar el color
    assert n.grupos_objetos[0].n == 1
    obs = json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos'][0]
    assert not obs['confirmado']


def test_imagen_analizada_sin_objetos_tambien_da_respuesta(monkeypatch):
    n, reloj = _detector(monkeypatch)
    n.scan.ranges = [float('inf')] * len(n.scan.ranges)
    _imagen(n, reloj, 1.0, (0, 0, 0))
    salida = json.loads(n.pub_observaciones.publish.call_args.args[0].data)
    assert salida['objetos'] == [] and salida['stamp_ns'] == 101_000_000_000


@pytest.mark.parametrize('color', [(255, 0, 0), (0, 255, 255), (0, 255, 0),
                                    (150, 150, 150), (0, 0, 255)])
def test_pared_de_cualquier_color_no_es_objeto_ni_victima(monkeypatch, color):
    from equipo_jar.deteccion_victimas import campo_distancia_paredes
    n, reloj = _detector(monkeypatch)
    n.mapa['libre'][:, 65:] = False  # pared en x=0.75: coincide con el LiDAR
    n.mapa['dist'] = campo_distancia_paredes(n.mapa['libre'], 0.05)
    for t in np.arange(1.0, 2.6, 0.2):
        _imagen(n, reloj, t, color)
    assert json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos'] == []
    assert not n.grupos and not n.grupos_objetos
    n.revisar_reportes()
    n.pub.publish.assert_not_called()


@pytest.mark.parametrize('forma', ['cilindro', 'triangulo'])
def test_no_exige_forma_de_caja(monkeypatch, forma):
    n, reloj = _detector(monkeypatch)
    imagen = np.zeros((240, 320, 3), np.uint8)
    if forma == 'cilindro':
        cv2.circle(imagen, (160, 140), 60, (0, 255, 0), -1)
        # Intersección de cada rayo con un cilindro: superficie curva.
        a = n.scan.angle_min + np.arange(len(n.scan.ranges)) * n.scan.angle_increment
        discriminante = 0.12**2 - (0.85 * np.sin(a))**2
        n.scan.ranges = np.where(discriminante >= 0,
            0.85 * np.cos(a) - np.sqrt(np.maximum(0, discriminante)), np.inf).tolist()
    else:
        cv2.fillPoly(imagen, [np.array([[160, 80], [105, 190], [215, 190]])], (0, 255, 0))
    _imagen(n, reloj, 1.0, imagen=imagen)
    obs = json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos']
    assert len(obs) == 1 and obs[0]['color'] == 'verde'
    assert obs[0]['x'] == pytest.approx(0.85, abs=0.05)
    assert not n.grupos


def test_objeto_del_mismo_color_que_pared_se_distingue_por_distancia(monkeypatch):
    from equipo_jar.deteccion_victimas import campo_distancia_paredes
    n, reloj = _detector(monkeypatch)
    n.mapa['libre'][:, 76:] = False  # pared en x=1.30; objeto delante a x=0.75
    n.mapa['dist'] = campo_distancia_paredes(n.mapa['libre'], 0.05)
    imagen = np.full((240, 320, 3), 150, np.uint8)  # imagen uniforme, sin bordes de color
    _imagen(n, reloj, 1.0, imagen=imagen)
    obs = json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos']
    assert len(obs) == 1 and obs[0]['color'] == 'neutro'
    assert obs[0]['x'] == pytest.approx(0.85, abs=0.015)


def test_rojo_del_fondo_no_convierte_distractor_verde_en_victima(monkeypatch):
    n, reloj = _detector(monkeypatch)
    imagen = np.zeros((240, 320, 3), np.uint8)
    imagen[40:184, 100:220] = (0, 0, 255)  # pared roja detrás y por encima
    imagen[155:184, 115:205] = (0, 255, 0)  # donde realmente retorna el láser
    for t in np.arange(1.0, 2.6, 0.2):
        _imagen(n, reloj, t, imagen=imagen)
    obs = json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos']
    assert len(obs) == 1 and obs[0]['color'] == 'verde'
    assert not n.grupos
    n.revisar_reportes()
    n.pub.publish.assert_not_called()


@pytest.mark.parametrize('fondo', [(0, 255, 255), (255, 0, 0), (0, 255, 0)])
def test_color_del_fondo_no_se_asocia_al_objeto_neutro_delantero(monkeypatch, fondo):
    n, reloj = _detector(monkeypatch)
    imagen = np.zeros((240, 320, 3), np.uint8)
    imagen[40:184, 100:220] = fondo
    imagen[155:184, 115:205] = (160, 160, 160)
    for t in np.arange(1.0, 2.6, 0.2):
        _imagen(n, reloj, t, imagen=imagen)
    objetos = json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos']
    assert len(objetos) == 1 and objetos[0]['color'] == 'neutro'
    assert len(n.grupos_objetos) == 1


def test_cambio_de_color_no_cambia_id_ni_duplica_objeto(monkeypatch):
    n, reloj = _detector(monkeypatch)
    ids = []
    for t, bgr in ((1.0, (255, 0, 0)), (1.2, (0, 255, 255)), (1.4, (160, 160, 160))):
        _imagen(n, reloj, t, bgr)
        ids.append(json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos'][0]['id'])
    assert len(set(ids)) == 1 and len(n.grupos_objetos) == 1


def test_no_indica_vista_completa_si_la_caja_toca_el_borde_vertical(monkeypatch):
    n, reloj = _detector(monkeypatch)
    foto = _imagen(n, reloj, 1.0)
    imagen = np.zeros((240, 320, 3), dtype=np.uint8)
    imagen[:, 118:202] = (255, 0, 0)
    reloj['t'] = 1.2
    foto.data = imagen.tobytes()
    foto.header.stamp = n.get_clock().now().to_msg()
    n.scan.header.stamp = foto.header.stamp
    n.cb_imagen(foto)
    reloj['t'] += 0.13
    n.procesar_pendientes()
    obs = json.loads(n.pub_observaciones.publish.call_args.args[0].data)['objetos'][0]
    assert not obs['completo']


def test_espera_tf_sin_perder_la_imagen_y_usa_el_scan_historico(monkeypatch):
    n, reloj = _detector(monkeypatch)
    n.tf_buffer.can_transform_full.return_value = False
    _imagen(n, reloj, 1.0)
    assert len(n.imagenes_pendientes) == 1
    n.pub_observaciones.publish.assert_not_called()
    historico = n.scan
    nuevo = deepcopy(historico)
    nuevo.header.stamp = Time(nanoseconds=101_500_000_000).to_msg()
    n.cb_scan(nuevo)
    n.tf_buffer.can_transform_full.return_value = True
    reloj['t'] = 1.6
    n.procesar_pendientes()
    assert len(n.imagenes_pendientes) == 0
    assert n.grupos_objetos[0].n == 1
    assert n.tf_buffer.lookup_transform_full.call_args.args[3].nanoseconds == 101_000_000_000
    n.procesar_pendientes()
    assert n.grupos_objetos[0].n == 1


def test_no_espera_tf_indefinidamente_ni_analiza_fotos_vencidas(monkeypatch):
    n, reloj = _detector(monkeypatch)
    n.tf_buffer.can_transform.return_value = False
    _imagen(n, reloj, 1.0)
    reloj['t'] = 2.0
    n.procesar_pendientes()
    assert not n.imagenes_pendientes
    n.pub_observaciones.publish.assert_not_called()
    assert n.diag['vencidas'] == 1
