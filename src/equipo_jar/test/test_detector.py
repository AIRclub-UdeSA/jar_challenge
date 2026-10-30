"""Precisión de la detección de víctimas (detector.py) en escenas sintéticas, sin ROS ni simulador.

Cada escena pone cajas de la competencia (0.25 x 0.25 x 0.30 m, colores de los .world: rojas = víctimas, azules =
señuelos) frente al robot, simula el scan del lidar (1080 rayos, ruido de 1 cm) y la foto de la cámara (320x240,
intrínsecos y montajes del contrato del robot real), y pasa las detecciones por DetectionClusterer como el nodo.
Mide dos cosas, porque las dos cuestan puntos:
    - error: distancia del grupo más cercano a cada caja roja. El juez da 100 puntos a <= 0.25 m y 50 a <= 0.50 m;
      los umbrales de acá son más exigentes porque se le suman el error del localizer y el de la TF.
    - falsos: detecciones en grupos lejos de toda caja roja. Un grupo así puede terminar en un reporte, que gasta
      uno de los N y tira el bono.

Los casos marcados xfail fallan hoy a propósito: son las mejoras pendientes, con el motivo.
Tabla por escenario (200 semillas): python3 src/equipo_jar/test/test_detector.py
"""
import math

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from equipo_jar.core import DetectionClusterer
from equipo_jar.detector import ProcesadorVision, mascara_rojo, puntos_laser_a_pixeles

# Cámara del robot real (contrato del simulador: 320x240, sin distorsión)
W, H, FX, FY, CX, CY = 320, 240, 271.809, 271.809, 158.719, 120.092
# Montajes respecto de base_footprint (base_link está 0.0714 m más arriba)
LASER_T = np.array([0.0435, 0.0000526, 0.0714 + 0.11])
LASER_R = Rotation.from_euler('z', math.pi).as_matrix()
COLOR_T = np.array([0.054937, 0.025094, 0.0714 + 0.037410])
COLOR_R = Rotation.from_euler('xyz', [-0.000336, 0.004697, 0.003574]).as_matrix()
OPTICO_R = Rotation.from_quat([-0.5, 0.5, -0.5, 0.5]).as_matrix()  # color -> óptico: z adelante, x derecha, y abajo
# TF laser_link -> cam_1_color_optical_frame, la que el nodo le pasa al detector
R_OL = (COLOR_R @ OPTICO_R).T @ LASER_R
T_OL = (COLOR_R @ OPTICO_R).T @ (LASER_T - COLOR_T)

LADO, ALTO = 0.25, 0.30
ROJO, AZUL = (0.8, 0.05, 0.05), (0.05, 0.05, 0.8)  # diffuse RGB de cuadrado_rojo y cuadrado_azul
ANGULOS = np.linspace(-math.pi, math.pi, 1080)
RADIO_GRUPO = DetectionClusterer().radius  # más lejos que esto de una caja roja, el grupo es otra "caja"
SEMILLAS = range(10)


def camara(desfase=0.0):
    """Pose del frame óptico en base_footprint. desfase: cuánto giró el robot entre el scan y la foto (rad)."""
    giro = Rotation.from_euler('z', desfase).as_matrix()
    return giro @ COLOR_R @ OPTICO_R, giro @ COLOR_T


def rayos_contra_caja(o, d, c, giro):
    """Distancia por cada rayo (origen o, direcciones d) hasta la caja de centro c; inf si no la toca."""
    cg, sg = math.cos(giro), math.sin(giro)
    a_caja = np.array([[cg, sg], [-sg, cg]])
    oc, dc = a_caja @ (o - c), d @ a_caja.T
    with np.errstate(divide='ignore', invalid='ignore'):
        t1, t2 = (-LADO / 2 - oc) / dc, (LADO / 2 - oc) / dc
    cerca, lejos = np.minimum(t1, t2).max(axis=1), np.maximum(t1, t2).min(axis=1)
    return np.where((cerca <= lejos) & (cerca > 0), cerca, np.inf)


def caja(rng, dist=(0.5, 3.0), rumbo=0.35, giro=None, color=ROJO):
    """Una caja al azar: dist (m) y rumbo (rad) de su centro visto desde el robot; giro None = al azar."""
    r = rng.uniform(-rumbo, rumbo) if np.isscalar(rumbo) else rng.choice([-1, 1]) * rng.uniform(*rumbo)
    return dict(dist=rng.uniform(*dist), rumbo=r, giro=rng.uniform(0, math.pi / 2) if giro is None else r + giro, color=color)


def otra_caja(rng, primera, **kw):
    """Una segunda caja a más de 0.5 m de la primera: no se pisan."""
    while True:
        k = caja(rng, **kw)
        ca, cb = (np.array([c['dist'] * math.cos(c['rumbo']), c['dist'] * math.sin(c['rumbo'])]) for c in (primera, k))
        if np.hypot(*(ca - cb)) > 0.5:
            return k


def escena(rng, cajas, pared=None, desfase=0.0, ruido=0.01):
    """Foto BGR, rangos del lidar y centros reales (base_footprint) de las cajas rojas que el lidar ve.

    pared: distancia de una pared detrás de la primera caja, perpendicular a la línea de vista (o None).
    desfase: cuánto giró el robot entre el scan y la foto (rad).
    """
    centros = [k['dist'] * np.array([math.cos(k['rumbo']), math.sin(k['rumbo'])]) for k in cajas]

    # Lidar: rayos desde laser_link, que está girado 180°
    o = LASER_T[:2]
    d = np.column_stack([np.cos(ANGULOS + math.pi), np.sin(ANGULOS + math.pi)])
    choques = [rayos_contra_caja(o, d, c, k['giro']) for c, k in zip(centros, cajas)]
    rangos = np.min(choques, axis=0)
    if pared is not None:
        n = centros[0] / np.linalg.norm(centros[0])
        with np.errstate(divide='ignore'):
            t = (centros[0] + n * pared - o) @ n / (d @ n)
        rangos = np.minimum(rangos, np.where(t > 0, t, np.inf))
    visible = [np.sum(np.isfinite(h) & (h <= rangos)) >= 3 for h in choques]  # una caja tapada no se le puede exigir
    rangos = rangos + rng.normal(0, ruido, len(rangos))

    # Foto: fondo gris (paredes claras, piso oscuro) y la silueta de cada caja (convexa), de la más lejana a la más cercana
    img = np.empty((H, W, 3), np.float64)
    img[: int(CY)] = rng.uniform(160, 230)
    img[int(CY):] = rng.uniform(80, 140)
    r_bo, t_bo = camara(desfase)
    for c, k in sorted(zip(centros, cajas), key=lambda ck: -ck[1]['dist']):
        cg, sg = math.cos(k['giro']), math.sin(k['giro'])
        base = np.array([[sx * LADO / 2, sy * LADO / 2] for sx in (-1, 1) for sy in (-1, 1)]) @ np.array([[cg, sg], [-sg, cg]]) + c
        p = (np.vstack([np.column_stack([base, np.full(4, z)]) for z in (0.0, ALTO)]) - t_bo) @ r_bo
        uv = np.column_stack([FX * p[:, 0] / p[:, 2] + CX, FY * p[:, 1] / p[:, 2] + CY])
        luz = rng.uniform(0.55, 1.0)
        cv2.fillConvexPoly(img, cv2.convexHull(np.round(uv).astype(np.int32)), [255 * luz * v for v in k['color'][::-1]])
    img = cv2.GaussianBlur(img + rng.normal(0, 4, img.shape), (3, 3), 0)
    rojas = [c for c, k, ve in zip(centros, cajas, visible) if k['color'] == ROJO and ve]
    return np.clip(img, 0, 255).astype(np.uint8), rangos, rojas


def grupos(img, rangos):
    """Lo que hace el nodo con una escena: detecciones -> base_footprint -> DetectionClusterer. [(centro, n)]."""
    det = ProcesadorVision().extraer_coordenadas_cajas(img, rangos, ANGULOS, R_OL, T_OL, FX, FY, CX, CY)
    cl = DetectionClusterer()
    for x, y in np.asarray(det).reshape(-1, 2) @ LASER_R[:2, :2].T + LASER_T[:2]:  # laser_link -> base_footprint
        cl.add(x, y, 0.0, 0.0)
    return [(np.array(cl.center(g)), len(g['obs'])) for g in cl.groups]


def medir(gs, rojas):
    """(error, falsos): peor distancia de una caja roja a su grupo más cercano, y detecciones en grupos lejos de toda caja roja."""
    error = max((min((np.hypot(*(c - r)) for c, _ in gs), default=math.inf) for r in rojas), default=0.0)
    falsos = sum(n for c, n in gs if all(np.hypot(*(c - r)) > RADIO_GRUPO for r in rojas))
    return float(error), falsos


ESCENARIOS = {  # nombre: semilla -> argumentos de escena
    'frontal': lambda rng: dict(cajas=[caja(rng, giro=0.0)]),
    'girada': lambda rng: dict(cajas=[caja(rng)]),
    'pared detrás': lambda rng: dict(cajas=[caja(rng, dist=(0.5, 2.5))], pared=rng.uniform(0.05, 0.6)),
    'borde de la foto': lambda rng: dict(cajas=[caja(rng, dist=(0.7, 2.5), rumbo=(0.45, 0.52))], pared=rng.uniform(0.2, 1.0)),
    'caja y señuelo': lambda rng: dict(cajas=[k := caja(rng, dist=(0.6, 2.5)), otra_caja(rng, k, dist=(0.6, 2.5), color=AZUL)]),
    'dos cajas rojas': lambda rng: dict(cajas=[k := caja(rng, dist=(0.6, 2.5)), otra_caja(rng, k, dist=(0.6, 2.5))]),
    'desfase scan-foto': lambda rng: dict(cajas=[caja(rng, dist=(0.5, 2.5))], pared=rng.uniform(0.2, 1.0),
                                          desfase=rng.choice([-1, 1]) * math.radians(rng.uniform(3, 6))),
}


def evaluar(nombre, semilla):
    rng = np.random.default_rng(semilla)
    img, rangos, rojas = escena(rng, **ESCENARIOS[nombre](rng))
    return medir(grupos(img, rangos), rojas)


# ---------- piezas sueltas ----------

def test_proyeccion_eje_optico_y_detras():
    """Un punto sobre el eje óptico cae en (cx, cy); uno detrás de la cámara queda marcado."""
    r_bo, t_bo = camara()
    optico = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    xyz = (optico @ r_bo.T + t_bo - LASER_T) @ LASER_R  # óptico -> base_footprint -> laser_link
    u, v, adelante = puntos_laser_a_pixeles(xyz, R_OL, T_OL, FX, FY, CX, CY)
    assert u[0] == pytest.approx(CX) and v[0] == pytest.approx(CY)
    assert adelante.tolist() == [True, False]


def _es_rojo(bgr):
    p = ProcesadorVision()
    m = mascara_rojo(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV), p.hue_rojo_bajo_1, p.hue_rojo_alto_1,
                     p.hue_rojo_bajo_2, p.hue_rojo_alto_2, p.saturacion_min, p.valor_min)
    return m[0, 0] == 255


@pytest.mark.parametrize('luz', [0.45, 0.7, 1.0])
def test_mascara_toma_la_caja_roja_con_distinta_luz(luz):
    assert _es_rojo([255 * luz * k for k in ROJO[::-1]])


@pytest.mark.parametrize('bgr', [[204, 13, 13], [180, 180, 180], [40, 40, 40], [30, 120, 230]],
                         ids=['señuelo azul', 'pared gris', 'sombra', 'naranja'])
def test_mascara_no_toma_otros_colores(bgr):
    assert not _es_rojo(bgr)


# ---------- escenas completas ----------

TAPADA = pytest.mark.xfail(
    reason='Con una caja tapada en parte por otra, el lidar ve solo un costado de la cara y sumar 12.5 cm a lo largo '
           'de cada rayo corre el centro de costado (hasta ~20 cm). Ajustar la recta de la cara y correrse por su '
           'normal (detector.py, problema 6)', strict=False)


@pytest.mark.parametrize('nombre, max_error', [
    ('frontal', 0.05), ('girada', 0.06), ('pared detrás', 0.06), ('borde de la foto', 0.15),
    ('caja y señuelo', 0.06), pytest.param('dos cajas rojas', 0.06, marks=TAPADA),
])
@pytest.mark.parametrize('semilla', SEMILLAS)
def test_precision(nombre, max_error, semilla):
    assert evaluar(nombre, semilla)[0] < max_error


@pytest.mark.parametrize('nombre', ['frontal', 'girada', 'pared detrás', 'borde de la foto', 'caja y señuelo',
                                    'dos cajas rojas'])
@pytest.mark.parametrize('semilla', SEMILLAS)
def test_sin_grupos_falsos(nombre, semilla):
    """Sin la erosión y los tramos del detector, los rayos de la pared que rozan la caja daban falsos (58 % con pared detrás)."""
    assert evaluar(nombre, semilla)[1] == 0


@pytest.mark.xfail(reason='La foto y el scan no se sincronizan: con 3-6° de giro entre ambos, rayos de la pared caen '
                          'sobre el rojo de la caja y forman grupos falsos (detector.py, problema 7)', strict=False)
@pytest.mark.parametrize('semilla', SEMILLAS)
def test_desfase_entre_scan_y_foto(semilla):
    error, falsos = evaluar('desfase scan-foto', semilla)
    assert error < 0.06 and falsos == 0


@pytest.mark.parametrize('semilla', SEMILLAS)
def test_senuelo_azul_solo_no_se_detecta(semilla):
    rng = np.random.default_rng(semilla)
    img, rangos, _ = escena(rng, [caja(rng, color=AZUL)])
    assert grupos(img, rangos) == []


if __name__ == '__main__':
    print(f'{"escenario":20}{"mediana":>9}{"p90":>8}{"máx":>8}{"sin ver":>9}{"con falsos":>12}{"falsos/escena":>15}'
          '   (200 semillas; error en cm)')
    for nombre in ESCENARIOS:
        r = np.array([evaluar(nombre, s) for s in range(200)])
        e, f = r[:, 0], r[:, 1]
        ok = e[np.isfinite(e)]
        print(f'{nombre:20}{100 * np.median(ok):9.1f}{100 * np.percentile(ok, 90):8.1f}{100 * ok.max():8.1f}'
              f'{np.mean(~np.isfinite(e)):9.0%}{np.mean(f > 0):12.0%}{f.mean():15.1f}')
