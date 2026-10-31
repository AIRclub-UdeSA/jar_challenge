"""Umbrales del rojo (ProcesadorVision) contra cambios de luz, en fotos reales del robot: el bag laberinto_victimas.

No se puede probar en el robot antes de la evaluación, pero el bag es su cámara en un laberinto real. Cada foto se altera con
la luz que podría haber en el evento (más oscura, más brillante, gamma, cálida, fría) y se mide como el pipeline: máscara de
rojo -> erosión 5x5. Caja: envolvente convexa de las manchas rojas de la foto original, con un criterio holgado y fijo (la
silueta de una caja es convexa: rellena los huecos de las caras iluminadas). Falso: mancha de >= 20 px después de la
erosión, a más de 7 px de toda caja.

Sin el bag (el CI no lo tiene) se saltea. Ruta: variable JAR_BAG o ~/rosmaster_ws/bags/laberinto_victimas.
python3 test_color_lighting.py imprime la tabla.
"""
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from equipo_jar.detector import ProcesadorVision, mascara_rojo

BAG = Path(os.environ.get('JAR_BAG', Path.home() / 'rosmaster_ws/bags/laberinto_victimas/laberinto_victimas_0.db3'))
PARAMS = Path(__file__).parents[1] / 'params' / 'params.yaml'
THRESHOLDS = ('hue_rojo_bajo_1', 'hue_rojo_alto_1', 'hue_rojo_bajo_2', 'hue_rojo_alto_2', 'saturacion_min', 'valor_min')
K5, K15 = np.ones((5, 5), np.uint8), np.ones((15, 15), np.uint8)

LIGHTS = {
    'original': lambda f: f,
    'oscura x0.5': lambda f: f * 0.5,
    'brillante x1.6': lambda f: f * 1.6,
    'gamma 0.6': lambda f: 255 * (f / 255) ** 0.6, # lavada: sube los medios tonos y baja la saturación
    'gamma 1.6': lambda f: 255 * (f / 255) ** 1.6,
    'cálida': lambda f: f * [0.8, 1.0, 1.2], # BGR
    'fría': lambda f: f * [1.2, 1.0, 0.8],
    'cálida brillante': lambda f: f * [0.8, 1.0, 1.2] * 1.4,
}
# (mínimo de caja detectada, máximo de fotos con falsos) por luz: lo medido con los umbrales de params.yaml, con margen.
# Los falsos con luz cálida o gamma 1.6 son paredes de MDF; el filtro de paredes del mapa los descarta después.
EXPECTED = {
    'original': (0.85, 0.0),
    'oscura x0.5': (0.80, 0.0),
    'brillante x1.6': (0.78, 0.0),
    'gamma 0.6': (0.55, 0.02),
    'gamma 1.6': (0.85, 0.10),
    'cálida': (0.85, 0.08),
    'fría': (0.60, 0.02),
    'cálida brillante': (0.85, 0.10),
}

def frames(path, step=15):
    """Una de cada `step` fotos de /cam_1/color/image_raw del bag, en BGR."""
    import sqlite3
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    db = sqlite3.connect(path)
    tid = db.execute("select id from topics where name='/cam_1/color/image_raw'").fetchone()[0]
    for (data,) in db.execute('select data from messages where topic_id=? order by timestamp', (tid,)).fetchall()[::step]:
        m = deserialize_message(data, Image)
        yield np.ascontiguousarray(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)[:, :, ::-1]) # rgb8 -> BGR

def red(hsv, u):
    return mascara_rojo(hsv, u.hue_rojo_bajo_1, u.hue_rojo_alto_1, u.hue_rojo_bajo_2, u.hue_rojo_alto_2, u.saturacion_min, u.valor_min)

def label(img):
    """(caja, lejos de toda caja): envolvente convexa de cada mancha roja de más de 60 px, con un criterio holgado y fijo."""
    m = cv2.morphologyEx(red(cv2.cvtColor(img, cv2.COLOR_BGR2HSV), ProcesadorVision(0, 12, 150, 180, 90, 40)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    box = np.zeros(m.shape, np.uint8)
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] >= 60: cv2.fillConvexPoly(box, cv2.convexHull(np.argwhere(lab == k)[:, ::-1].astype(np.int32)), 1)

    return box > 0, cv2.dilate(box, K15) == 0

def evaluate(images, u):
    """Por luz: (fracción de la caja detectada, fracción de fotos con alguna mancha falsa)."""
    labels = [label(img) for img in images]
    out = {}
    for name, light in LIGHTS.items():
        found, total, false = 0, 0, 0
        for img, (box, far) in zip(images, labels):
            m = red(cv2.cvtColor(np.clip(light(img.astype(float)), 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV), u)
            found += np.count_nonzero(m[box])
            total += np.count_nonzero(box)
            e = cv2.erode(m, K5) & far.astype(np.uint8) * 255 # como el detector: el borde de la máscara no cuenta
            if np.count_nonzero(e) >= 20:
                _, _, st, _ = cv2.connectedComponentsWithStats(e)
                false += (st[1:, cv2.CC_STAT_AREA] >= 20).any()
        out[name] = (found / total, false / len(images))

    return out

@pytest.fixture(scope='module')
def results():
    if not BAG.exists(): pytest.skip(f'sin el bag {BAG}')

    return evaluate(list(frames(BAG)), ProcesadorVision())

@pytest.mark.parametrize('light', LIGHTS)
def test_red_threshold_under_light(results, light):
    found, false = results[light]
    min_found, max_false = EXPECTED[light]
    assert found >= min_found, f'{light}: se detecta {found:.0%} de la caja (mínimo {min_found:.0%})'
    assert false <= max_false, f'{light}: {false:.1%} de las fotos con falsos (máximo {max_false:.0%})'

def test_params_yaml_matches_defaults():
    """El nodo lee los umbrales de params.yaml y este test mide los de ProcesadorVision(): tienen que ser los mismos."""
    params = yaml.safe_load(PARAMS.read_text())['nodo_equipo']['ros__parameters']
    default = ProcesadorVision()
    assert {k: params[k] for k in THRESHOLDS} == {k: getattr(default, k) for k in THRESHOLDS}

if __name__ == '__main__':
    images = list(frames(BAG))
    print(f'{len(images)} fotos, {sum(label(img)[0].any() for img in images)} con caja')
    for name, (found, false) in evaluate(images, ProcesadorVision()).items():
        print(f'{name:17} caja {found:6.1%} | fotos con falsos {false:5.1%}')
