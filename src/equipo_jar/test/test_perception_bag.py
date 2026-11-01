"""La percepción completa (localizer.py + nodo_equipo.py) contra el bag real laberinto_victimas, puntuada como el juez.

Le pasa los mensajes del bag a los nodos de la competencia en el orden en que se grabaron (sin ros2 bag play: más rápido que el tiempo real y siempre
igual) con el mapa de la arena vacía, y compara los reportes con las cajas medidas. Mapa, cajas y su tamaño salen de la carpeta de la arena: no están
en el repo.

Sin el bag (el CI no lo tiene) se saltea. Rutas: variables JAR_BAG (el .db3) y JAR_ARENA (carpeta con laberinto_vacio.yaml y cajas.yaml).
python3 test_perception_bag.py imprime los grupos y los reportes.
"""
import math
import os
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

BAG = Path(os.environ.get('JAR_BAG', Path.home() / 'rosmaster_ws/bags/laberinto_victimas/laberinto_victimas_0.db3'))
ARENA = Path(os.environ.get('JAR_ARENA', Path.home() / 'rosmaster_ws/bags/semana10'))
PARAMS = Path(__file__).parents[1] / 'params' / 'params.yaml'
NODO = yaml.safe_load(PARAMS.read_text())['nodo_equipo']['ros__parameters']
TOPICS = ('/tf_static', '/tf', '/odom', '/imu/data', '/scan', '/cam_1/color/camera_info', '/cam_1/color/image_raw')

def load_map(path):
    """yaml + pgm de map_server -> OccupancyGrid (modo trinary)."""
    from nav_msgs.msg import OccupancyGrid

    meta = yaml.safe_load(path.read_text())
    p = (255 - cv2.imread(str(path.parent / meta['image']), cv2.IMREAD_GRAYSCALE)) / 255
    msg = OccupancyGrid()
    msg.header.frame_id = 'map'
    msg.info.resolution = float(meta['resolution'])
    msg.info.height, msg.info.width = p.shape
    msg.info.origin.position.x, msg.info.origin.position.y = map(float, meta['origin'][:2])
    msg.info.origin.orientation.w = 1.0
    msg.data = np.flipud(np.where(p > meta['occupied_thresh'], 100, np.where(p < meta['free_thresh'], 0, -1))).astype(np.int8).ravel().tolist()

    return msg

def messages(path):
    """(topic, mensaje, t de grabación) de TOPICS en el orden en que se grabaron. Generador: las 3600 fotos no entran juntas en memoria."""
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    db = sqlite3.connect(path)
    types = {i: (name, get_message(t)) for i, name, t in db.execute('select id, name, type from topics') if name in TOPICS}
    for tid, data, t in db.execute(f'select topic_id, data, timestamp from messages where topic_id in ({",".join(map(str, types))}) order by timestamp'):
        yield types[tid][0], deserialize_message(data, types[tid][1]), t * 1e-9

def first_motion(path):
    """Instante de grabación en que el robot empieza a moverse según /odom: el mapa llega antes, con el robot quieto, como en la competencia."""
    p0 = None
    for topic, m, t in messages(path):
        if topic != '/odom': continue

        p = m.pose.pose.position
        p0 = p0 or (p.x, p.y)
        if math.hypot(p.x - p0[0], p.y - p0[1]) > 0.01: return t

def replay(imu=lambda m: m):
    """(víctimas, distractores, reportes, clusterer) de una pasada del bag por Localizer y NodoEquipo. imu: cambia cada /imu/data (None: no llega)."""
    import rclpy
    from geometry_msgs.msg import TransformStamped
    from rclpy.duration import Duration
    from rclpy.time import Time

    import equipo_jar.localizer as loc
    from equipo_jar.core import ParticleFilter
    from equipo_jar.nodo_equipo import NodoEquipo

    boxes = yaml.safe_load((ARENA / 'cajas.yaml').read_text())['evaluador']['ros__parameters']
    victims = np.reshape(boxes['victimas'], (-1, 2))
    loc.ParticleFilter = lambda board, n: ParticleFilter(board, n, seed=0) # siempre el mismo resultado
    rclpy.init(args=['--ros-args', '--params-file', str(PARAMS), '-p', f'cantidad_victimas:={len(victims)}']) # N, como victimas:=N en el launch
    try:
        localizer, node = loc.Localizer(), NodoEquipo()
        node.procesador_vision.mitad_caja = boxes['lado_caja'] / 2
        reports = []
        node.pub_victimas = SimpleNamespace(publish=lambda m: reports.append((m.point.x, m.point.y)))
        map_msg, map_time = load_map(ARENA / 'laberinto_vacio.yaml'), first_motion(BAG) - 0.5

        for topic, msg, t in messages(BAG):
            if map_msg is not None and t >= map_time:
                localizer.on_map(map_msg)
                node.al_recibir_mapa(map_msg)
                map_msg = None
            if topic in ('/tf', '/tf_static'):
                for tr in msg.transforms:
                    for b in (localizer.tf, node.tf_buffer): b.set_transform_static(tr, 'bag') if topic == '/tf_static' else b.set_transform(tr, 'bag')
            elif topic == '/odom': localizer.on_odom(msg)
            elif topic == '/imu/data' and imu(msg) is not None: localizer.on_imu(msg)
            elif topic == '/cam_1/color/camera_info': node.al_recibir_info_camara(msg)
            elif topic == '/cam_1/color/image_raw': node.al_recibir_imagen(msg)
            elif topic == '/scan':
                localizer.process_scan(msg)
                # map -> odom en el instante medio del scan (el nodo la publica adelantada; acá no hace falta)
                x, y, th = localizer.map_odom
                tr = TransformStamped()
                tr.header.stamp = (Time.from_msg(msg.header.stamp) + Duration(seconds=msg.time_increment * (len(msg.ranges) - 1) / 2)).to_msg()
                tr.header.frame_id, tr.child_frame_id = 'map', 'odom'
                tr.transform.translation.x, tr.transform.translation.y = float(x), float(y)
                tr.transform.rotation.z, tr.transform.rotation.w = math.sin(th / 2), math.cos(th / 2)
                if localizer.pf is not None: node.tf_buffer.set_transform(tr, 'localizer')
                node.al_recibir_scan(msg)

        return victims, np.reshape(boxes['distractores'], (-1, 2)), reports, node.clusterer
    finally:
        rclpy.shutdown()

def flip(m):
    m.angular_velocity.z *= -1
    return m

# La IMU como en el bag, sin IMU y montada al revés de lo que dice su TF: en los dos últimos el giro sale de /odom (localizer.turn_source)
IMU = {'imu': lambda m: m, 'sin imu': lambda m: None, 'imu al revés': flip}

@pytest.fixture(scope='module', params=IMU)
def run(request):
    if not (BAG.exists() and (ARENA / 'cajas.yaml').exists()): pytest.skip(f'sin el bag {BAG} o la arena {ARENA}')

    return replay(IMU[request.param])

def test_reports_every_victim_within_25_cm(run):
    victims, _, reports, _ = run
    assert len(reports) == len(victims)
    for v in victims:
        assert min(math.dist(v, r) for r in reports) <= 0.25, f'víctima en {v}: reportes {reports}'

def test_no_group_on_the_distractor(run):
    _, distractors, _, clusterer = run
    for g in clusterer.groups:
        assert min(math.dist(d, clusterer.center(g)) for d in distractors) > 0.3, f'grupo en un distractor: {clusterer.center(g)}'

if __name__ == '__main__':
    victims, distractors, reports, clusterer = replay()
    print('víctimas', victims.tolist(), '| distractores', distractors.tolist())
    for g in clusterer.groups:
        c = clusterer.center(g)
        o = np.asarray(g['obs'])
        print(f'grupo ({c[0]:+.3f}, {c[1]:+.3f}): {len(o)} obs, a {min(math.dist(c, v) for v in victims):.3f} m de una víctima, '
              f'listo {clusterer.ready(g, NODO["observaciones_min"], NODO["dispersion_max"], NODO["base_min"])}, reportado {g["reported"]}')
    for r in reports:
        print(f'reporte ({r[0]:+.3f}, {r[1]:+.3f}): a {min(math.dist(r, v) for v in victims):.3f} m de una víctima')
