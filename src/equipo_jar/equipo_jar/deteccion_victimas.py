"""Detección y reporte de víctimas


1. CÁMARA (dirección): en la imagen RGB se segmenta el rojo en HSV. 
   Cada mancha roja grande da un intervalo de columnas -> con la intrínseca 
   de la cámara y la TF, un intervalo de ÁNGULOS (rumbos) en el plano del robot.
2. LiDAR (distancia): se toman los puntos del /scan que caen dentro de ese
   intervalo de rumbos. Los más cercanos son la cara de la caja.
3. MAPA (descartar falsos positivos): la víctima NO está en /map. Si la "cara"
   cae sobre una pared del mapa se descarta. También se descarta si el centro
   cae en zona ocupada o fuera del mapa.
4. VARIAS VISTAS: cada detección (en el marco map, con la TF del
   instante de la imagen) se agrupa por cercanía. Un grupo se REPORTA cuando
   tiene suficientes detecciones consistentes (poca dispersión), con la mediana
   como posición para los reportes confirmados durante la exploración.
   Reglas extra para no desperdiciar los reportes que sobran:
   - si un grupo con pocas detecciones dejó de verse hace un rato (el robot
     pasó de largo), se reporta igual;
   - cerca del final de la corrida se reportan los mejores grupos pendientes.
   - antes de /done, se usan las detecciones débiles que quedan y, si sobra
     cupo, posiciones estimadas en zonas poco observadas, sin confirmación visual. 

Los demás objetos se localizan por superficies LiDAR ausentes del mapa,
sin exigir una forma o color. La cámara añade color sólo sobre esas superficies;
las paredes se descartan por geometría. Publica /deteccion/observaciones.

El explorador (exploracion.py) escucha /victimas: con N reportes vuelve solo a
la largada y publica /done. Este nodo no mueve al robot.
"""
import json
import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PointStamped, PoseArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from equipo_jar.reportes_finales import lugares_probables
from equipo_jar.observacion_objetos import grupos_lidar, centro_superficie_lidar

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
QOS_REPORTE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


def imagen_a_bgr(encoding, alto, ancho, paso, datos):
    """sensor_msgs/Image -> arreglo BGR uint8 (sin cv_bridge)."""
    canales = {'rgb8': 3, 'bgr8': 3, 'rgba8': 4, 'bgra8': 4}.get(encoding)
    if canales is None:
        return None
    buf = np.frombuffer(datos, dtype=np.uint8).reshape(alto, paso)[:, :ancho * canales]
    img = buf.reshape(alto, ancho, canales)
    if encoding == 'rgb8':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if encoding == 'rgba8':
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if encoding == 'bgra8':
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return np.ascontiguousarray(img)


def segmentar_rojo(bgr, cfg):
    """Manchas rojas: lista de (u_min, u_max, v_min, v_max, area_px)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_min, v_min = int(cfg['saturacion_minima']), int(cfg['brillo_minimo'])
    m1 = cv2.inRange(hsv, (0, s_min, v_min), (int(cfg['tono_rojo_bajo_max']), 255, 255))
    m2 = cv2.inRange(hsv, (int(cfg['tono_rojo_alto_min']), s_min, v_min), (180, 255, 255))
    return manchas_de_mascara(m1 | m2, cfg['area_minima_px'])


NOMBRES_COLORES = ('naranja', 'amarillo', 'verde', 'cian', 'azul', 'violeta', 'magenta')


def segmentar_colores(bgr, cfg):
    """Manchas de color saturado que NO es rojo: lista de (u0, u1, v0, v1, area, color)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s_, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    vivos = (s_ >= cfg['saturacion_minima']) & (v >= cfg['brillo_minimo'])
    vivos &= ~((h <= cfg['tono_rojo_bajo_max']) | (h >= cfg['tono_rojo_alto_min']))
    limites = [0, *(cfg[f'tono_{c}_max'] for c in NOMBRES_COLORES[:-1]), 180]
    alto, ancho = bgr.shape[:2]
    salida = []
    for color, a, b in zip(NOMBRES_COLORES, limites, limites[1:]):
        mascara = (vivos & (h >= a) & (h < b)).astype(np.uint8) * 255
        if not mascara.any():
            continue
        manchas, _ = manchas_de_mascara(mascara, cfg['area_minima_px'])
        for u0, u1, v0, v1, area in manchas:
            if v1 >= alto - 2 or (u1 - u0 + 1) > cfg.get('ancho_maximo_mancha', 0.6) * ancho:
                continue
            salida.append((u0, u1, v0, v1, area, color))
    return salida


def color_superficie(bgr, xs, ys, altura, t_cam, intrinseca, cfg):
    """Color donde el LiDAR toca el objeto, sin muestrear su fondo o la pared."""
    fx, fy, cx, cy = intrinseca
    tr = t_cam.transform.translation
    base = np.column_stack((xs, ys, np.full(len(xs), altura)))
    opt = (base - [tr.x, tr.y, tr.z]) @ quat_a_matriz(t_cam.transform.rotation)
    opt = opt[opt[:, 2] > 0.05]
    if len(opt) < 3:
        return None
    us = fx * opt[:, 0] / opt[:, 2] + cx
    vs = fy * opt[:, 1] / opt[:, 2] + cy
    # Evita los extremos del grupo, donde se mezclan objeto y fondo.
    lo, hi = np.quantile(us, (0.2, 0.8))
    muestras = []
    alto, ancho = bgr.shape[:2]
    for u, v in zip(us, vs):
        if not lo <= u <= hi or not (2 <= u < ancho - 2 and 2 <= v < alto - 2):
            continue
        u, v = int(round(u)), int(round(v))
        muestras.extend(bgr[v - 1:v + 2, u - 1:u + 2].reshape(-1, 3))
    if not muestras:
        return None
    hsv = cv2.cvtColor(np.asarray(muestras, np.uint8).reshape(1, -1, 3), cv2.COLOR_BGR2HSV)[0]
    h, s, v = hsv.T
    etiquetas = np.full(len(h), 'neutro', dtype=object)
    vivos = (s >= cfg['saturacion_minima']) & (v >= cfg['brillo_minimo'])
    nombres = ('naranja', 'amarillo', 'verde', 'cian', 'azul', 'violeta', 'magenta')
    limites = [0, *(cfg[f'tono_{c}_max'] for c in nombres[:-1]), 180]
    for color, a, b in zip(nombres, limites, limites[1:]):
        etiquetas[vivos & (h >= a) & (h < b)] = color
    rojo = vivos & ((h <= cfg['tono_rojo_bajo_max']) | (h >= cfg['tono_rojo_alto_min']))
    etiquetas[rojo] = 'rojo'
    colores, conteos = np.unique(etiquetas, return_counts=True)
    i = int(np.argmax(conteos))
    return str(colores[i]) if conteos[i] / len(etiquetas) >= cfg['fraccion_color_minima'] else 'desconocido'


def manchas_de_mascara(mascara, area_minima):
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
    manchas = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < area_minima or w < 3 or h < 3:
            continue
        manchas.append((int(x), int(x + w - 1), int(y), int(y + h - 1), int(area)))
    return manchas, mascara


def rumbos_de_mancha(u0, u1, v_centro, fx, fy, cx, cy, rot_cam_a_base, recorte=0.2):
    """Intervalo de rumbos (rad, en el plano XY de base_footprint) que ocupa la mancha."""
    w = u1 - u0
    ua, ub = u0 + recorte * w, u1 - recorte * w           # lejos de los bordes de la mancha
    rumbos = []
    for u in (ua, ub, 0.5 * (u0 + u1)):
        d_opt = np.array([(u - cx) / fx, (v_centro - cy) / fy, 1.0])
        d = rot_cam_a_base @ d_opt
        rumbos.append(math.atan2(d[1], d[0]))
    a, b, centro = rumbos
    # El intervalo puede cruzar ±pi: se lo expresa relativo al centro.
    da, db = ang_norm(a - centro), ang_norm(b - centro)
    return centro, min(da, db), max(da, db)


def centro_de_caja(px, py, cam_xy, rumbo, dmin, dmax, rango_max, medio_lado, tolerancia_cara=0.12):
    """A partir de los puntos del láser (marco base) dentro del intervalo de rumbos vistos
    desde la cámara, estima el centro de la caja. Devuelve (cx, cy, cara_x, cara_y, n) o None."""
    ang = np.arctan2(py - cam_xy[1], px - cam_xy[0])
    rel = np.arctan2(np.sin(ang - rumbo), np.cos(ang - rumbo))
    dist = np.hypot(px - cam_xy[0], py - cam_xy[1])
    sel = (rel >= dmin) & (rel <= dmax) & (dist < rango_max) & (dist > 0.05)
    if sel.sum() < 3:
        return None
    d_sel = dist[sel]
    cerca = d_sel <= d_sel.min() + tolerancia_cara          # la cara más cercana (no lo de atrás)
    if cerca.sum() < 3:
        return None
    cx_, cy_ = px[sel][cerca], py[sel][cerca]
    fx_, fy_ = float(np.median(cx_)), float(np.median(cy_))
    ux, uy = fx_ - cam_xy[0], fy_ - cam_xy[1]
    norma = math.hypot(ux, uy)
    if norma < 1e-6:
        return None
    return (fx_ + medio_lado * ux / norma, fy_ + medio_lado * uy / norma, fx_, fy_, int(cerca.sum()),
            angulo_cara_rayo(cx_, cy_, ux / norma, uy / norma))


def angulo_cara_rayo(xs, ys, rx, ry):
    """Ángulo (grados, 0-90) entre la superficie que forman los puntos de la cara y el
    rayo de visión. Cara de caja de frente: ~90°. Pared vista "de costado": ~0°.
    NaN si los puntos no alcanzan para decirlo."""
    if xs.size < 3:
        return float('nan')
    pts = np.stack([xs - xs.mean(), ys - ys.mean()])
    w, v = np.linalg.eigh(pts @ pts.T)
    if w[1] < 1e-6 or w[1] < 4 * max(w[0], 0.0):     # nube redonda o de un solo punto
        return float('nan')
    dx, dy = v[0, 1], v[1, 1]
    return math.degrees(math.acos(min(1.0, abs(dx * rx + dy * ry))))


def ang_norm(a):
    return math.atan2(math.sin(a), math.cos(a))


def quat_a_matriz(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def campo_distancia_paredes(libre, res):
    """Distancia (m) aproximada de cada celda a la celda ocupada más cercana (2 pasadas, chamfer)."""
    alto, ancho = libre.shape
    d = np.where(libre, np.inf, 0.0)
    diag = math.sqrt(2.0)
    for f in range(alto):                       # pasada hacia adelante
        fila = d[f]
        if f > 0:
            arriba = d[f - 1]
            fila = np.minimum(fila, arriba + 1.0)
            fila[1:] = np.minimum(fila[1:], arriba[:-1] + diag)
            fila[:-1] = np.minimum(fila[:-1], arriba[1:] + diag)
        for c in range(1, ancho):
            if fila[c - 1] + 1.0 < fila[c]:
                fila[c] = fila[c - 1] + 1.0
        d[f] = fila
    for f in range(alto - 1, -1, -1):           # pasada hacia atrás
        fila = d[f]
        if f < alto - 1:
            abajo = d[f + 1]
            fila = np.minimum(fila, abajo + 1.0)
            fila[1:] = np.minimum(fila[1:], abajo[:-1] + diag)
            fila[:-1] = np.minimum(fila[:-1], abajo[1:] + diag)
        for c in range(ancho - 2, -1, -1):
            if fila[c + 1] + 1.0 < fila[c]:
                fila[c] = fila[c + 1] + 1.0
        d[f] = fila
    return d * res


class Grupo:
    """Detecciones de una misma víctima candidata, en el marco map."""

    def __init__(self, x, y, t, distancia=None):
        self.xs, self.ys = [x], [y]
        self.t0 = self.t_ult = t
        self.reportado = False
        self.d_min = distancia if distancia is not None else float('inf')
        self.ausente = 0
        self.descartado = False

    @property
    def n(self):
        return len(self.xs)

    def agregar(self, x, y, t, distancia=None):
        self.xs.append(x)
        self.ys.append(y)
        self.xs, self.ys = self.xs[-60:], self.ys[-60:]
        self.t_ult = t
        self.ausente = 0
        if distancia is not None:
            self.d_min = min(self.d_min, distancia)

    def posicion(self):
        return float(np.median(self.xs)), float(np.median(self.ys))

    def dispersion(self):
        mx, my = self.posicion()
        return float(np.median(np.hypot(np.array(self.xs) - mx, np.array(self.ys) - my)))


class DeteccionVictimas(Node):

    def __init__(self):
        super().__init__('deteccion_victimas')
        defaults = {
            'cantidad_victimas': 3,
            # color (HSV de OpenCV: tono 0-180)
            'tono_rojo_bajo_max': 10, 'tono_rojo_alto_min': 165,
            'tono_naranja_max': 20, 'tono_amarillo_max': 35, 'tono_verde_max': 85,
            'tono_cian_max': 95, 'tono_azul_max': 135, 'tono_violeta_max': 150,
            'saturacion_minima': 120, 'brillo_minimo': 50, 'area_minima_px': 40,
            'fraccion_color_minima': 0.60, 'separacion_pared_objeto_m': 0.16,
            'objetos_por_color': True, 'ancho_maximo_mancha': 0.6,
            # geometría
            'medio_lado_caja': 0.10, 'rango_maximo': 2.0, 'max_dentro_pared': 0.10, 'margen_borde_px': 3,
            'max_laser_detras_suelo': 0.20,
            'margen_frente_pared': 0.07, 'angulo_rasante_deg': 30.0,
            # confirmación y reporte
            'periodo_procesamiento': 0.1, 'radio_agrupamiento': 0.40,
            'detecciones_minimas': 8, 'dispersion_maxima': 0.10, 'segundos_minimos_vista': 0.8,
            'detecciones_minimas_paso': 4, 'separacion_minima_cajas': 0.40, 'segundos_sin_ver_paso': 15.0,
            'segundos_reporte_final': 480.0, 'giro_maximo': 1.0, 'retardo_camara': 0.0,
            'estimar_al_cerrar': True, 'separacion_estimaciones_m': 0.75,
            'distancia_confirmacion': 1.6,
            'max_desfase_scan_imagen_s': 0.25, 'edad_maxima_imagen_s': 0.75,
            'espera_tf_vision_s': 0.12,
            # descartar posibles víctimas que, miradas de cerca, no están
            'descartar_no_vistas': True, 'imagenes_para_descartar': 10, 'margen_fov_deg': 3.0,
            # lugares imposibles y localización mala
            'radio_huella_m': 0.12,           # por donde pasó el centro del robot no puede haber una caja
            'error_loc_max': 0.07,            # con el láser a más de esto del mapa no se toman detecciones
            # frames / debug
            'frame_base': 'base_footprint', 'frame_odom': 'odom', 'publicar_debug': True,
            'diagnostico': False,     # imprime cada 3 s por qué se descartan las manchas rojas
        }
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.cfg = {k: self.get_parameter(k).value for k in defaults}
        limites = [self.cfg[f'tono_{c}_max'] for c in ('naranja', 'amarillo', 'verde', 'cian', 'azul', 'violeta')]
        if (not all(a < b for a, b in zip([0, *limites], [*limites, 180]))
                or not 0.5 < self.cfg['fraccion_color_minima'] <= 1.0
                or self.cfg['separacion_pared_objeto_m'] <= 0
                or min(self.cfg['max_desfase_scan_imagen_s'], self.cfg['edad_maxima_imagen_s']) <= 0):
            raise ValueError('Umbrales de color, pared o sincronización inválidos')
        self.n_max = int(self.cfg['cantidad_victimas'])

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(OccupancyGrid, '/map', self.cb_mapa, QOS_MAPA)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.cb_imagen, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.cb_info, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.cb_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.cb_odom, 10)
        # Calidad de localización que mide el explorador (x = error láser-mapa en m,
        # y = 1 si está relocalizando). Topic interno del equipo.
        self.create_subscription(PointStamped, '/exploracion/estado_loc', self.cb_estado_loc, 10)
        self.create_subscription(OccupancyGrid, '/exploracion/visto', self.cb_visto, QOS_MAPA)
        self.create_subscription(Empty, '/done', self.cb_done, QOS_REPORTE)
        self.create_service(Trigger, '/deteccion/reportar_finales', self.cerrar_reportes)
        self.pub = self.create_publisher(PointStamped, '/victimas', QOS_REPORTE)
        self.pub_pendientes = self.create_publisher(PoseArray, '/deteccion/pendientes', 10)
        self.pub_observaciones = self.create_publisher(String, '/deteccion/observaciones', 10)
        if self.cfg['publicar_debug']:
            self.pub_marc = self.create_publisher(MarkerArray, '/deteccion/marcadores', 10)

        self.mapa = None
        self.t_mapa = None
        self.K = None
        self.scan = None
        self.scans = deque(maxlen=30)
        self.imagenes_pendientes = deque(maxlen=10)
        self.giro = 0.0
        self.ult_proc = 0.0
        self.grupos = []
        self.grupos_objetos = []
        self.ultimo_stamp_imagen = None
        self.reportes = 0
        self.recorrido = None              # celdas por donde pasó el robot (bien localizado)
        self.visto = None
        self.reportes_cerrados = False
        self.loc = None                    # (t, error, relocalizando) del explorador
        self.create_timer(1.0, self.revisar_reportes)
        self.create_timer(0.2, self.marcar_recorrido)
        self.create_timer(0.05, self.procesar_pendientes)
        self.diag = dict.fromkeys(('imagenes', 'girando', 'con_rojo', 'manchas', 'sin_tf', 'sin_laser',
                                   'cara_en_pared', 'centro_en_pared', 'aceptadas'), 0)
        self.diag_ult = ''
        if self.cfg['diagnostico']:
            self.create_timer(3.0, self.imprimir_diagnostico)
        self.get_logger().info(f'Detector de víctimas listo (N={self.n_max}). Esperando /map...')

    # ------------------------------------------------------------- callbacks
    def cb_mapa(self, msg):
        info = msg.info
        previo = self.mapa
        datos = np.array(msg.data, dtype=np.int16).reshape(info.height, info.width)
        libre = (datos >= 0) & (datos < 50)
        self.mapa = dict(res=info.resolution, ox=info.origin.position.x, oy=info.origin.position.y,
                         alto=info.height, ancho=info.width, libre=libre,
                         frame=msg.header.frame_id or 'map',
                         dist=campo_distancia_paredes(libre, info.resolution),
                         dist_libre=campo_distancia_paredes(~libre, info.resolution))
        if self.recorrido is None or self.recorrido.shape != libre.shape:
            self.recorrido = np.zeros_like(libre)
        if previo is None or any(previo[k] != self.mapa[k] for k in ('res', 'ox', 'oy', 'alto', 'ancho', 'frame')):
            self.visto = None
        if self.t_mapa is None:
            self.t_mapa = time.monotonic()
            self.get_logger().info('/map recibido: empiezo a buscar víctimas')

    def cb_visto(self, msg):
        m, info = self.mapa, msg.info
        if (m is None or msg.header.frame_id != m['frame']
                or (info.height, info.width) != m['libre'].shape
                or not all(math.isclose(a, b, abs_tol=1e-6) for a, b in (
                    (info.resolution, m['res']), (info.origin.position.x, m['ox']),
                    (info.origin.position.y, m['oy'])))
                or len(msg.data) != m['alto'] * m['ancho']):
            return
        self.visto = np.asarray(msg.data).reshape(m['libre'].shape) >= 50

    def cb_estado_loc(self, msg):
        self.loc = (time.monotonic(), msg.point.x, msg.point.y > 0.5)

    def cb_done(self, _msg):
        self.reportes_cerrados = True

    def localizacion_ok(self):
        """Sin noticias del explorador se asume buena."""
        loc = getattr(self, 'loc', None)
        if loc is None:
            return True
        t, error, relocalizando = loc
        if time.monotonic() - t > 2.0:
            return True
        return not relocalizando and 0.0 <= error <= self.cfg['error_loc_max']

    def marcar_recorrido(self):
        """Huella del centro del robot en el mapa: ahí no puede haber una caja."""
        if self.mapa is None or self.recorrido is None or not self.localizacion_ok():
            return
        t = self.tf(self.mapa['frame'], self.cfg['frame_base'])
        if t is None:
            return
        m = self.mapa
        f, c = self.a_celda(t.transform.translation.x, t.transform.translation.y)
        k = max(1, int(round(self.cfg['radio_huella_m'] / m['res'])))
        f0, f1 = max(0, f - k), min(m['alto'], f + k + 1)
        c0, c1 = max(0, c - k), min(m['ancho'], c + k + 1)
        if f0 < f1 and c0 < c1:
            ff, cc = np.ogrid[f0:f1, c0:c1]
            self.recorrido[f0:f1, c0:c1] |= (ff - f) ** 2 + (cc - c) ** 2 <= k * k

    def en_recorrido(self, x, y):
        """¿El robot ya pasó por (x, y)? Una víctima ahí es imposible (la habría chocado):
        es la misma caja vista con la localización corrida."""
        if getattr(self, 'recorrido', None) is None or self.mapa is None:
            return False
        f, c = self.a_celda(x, y)
        m = self.mapa
        return 0 <= f < m['alto'] and 0 <= c < m['ancho'] and bool(self.recorrido[f, c])

    def cb_info(self, msg):
        if msg.k[0] > 0:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])   # fx, fy, cx, cy

    def cb_scan(self, msg):
        self.scan = msg
        self.scans.append(msg)

    def cb_odom(self, msg):
        self.giro = abs(msg.twist.twist.angular.z)

    def cb_imagen(self, msg):
        if self.mapa is None or self.scan is None:
            return
        ahora = time.monotonic()
        if ahora - self.ult_proc < self.cfg['periodo_procesamiento']:
            return
        self.ult_proc = ahora
        stamp = Time.from_msg(msg.header.stamp).nanoseconds
        if not stamp or (self.ultimo_stamp_imagen is not None and stamp <= self.ultimo_stamp_imagen):
            return
        edad = (self.get_clock().now().nanoseconds - stamp) / 1e9
        if not 0.0 <= edad <= self.cfg['edad_maxima_imagen_s']:
            return
        self.ultimo_stamp_imagen = stamp
        self.diag['imagenes'] += 1
        if self.giro > self.cfg['giro_maximo']:
            self.diag['girando'] += 1
            return          # girando muy rápido: imagen movida
        self.imagenes_pendientes.append((ahora, msg))

    def tiempo_imagen(self, msg):
        stamp = Time.from_msg(msg.header.stamp)
        return stamp - Duration(nanoseconds=int(self.cfg['retardo_camara'] * 1e9))

    @staticmethod
    def tiempos_scan(scan):
        inicio = Time.from_msg(scan.header.stamp)
        duracion = max(0.0, scan.time_increment * (len(scan.ranges) - 1))
        return inicio, inicio + Duration(nanoseconds=int(duracion * 1e9))

    def procesar_pendientes(self):
        """Espera TF sin bloquear su recepción en el executor de un solo hilo.

        Usa el barrido histórico más cercano a la foto, no el último recibido
        durante la espera. Una foto vencida nunca aporta evidencia nueva.
        """
        ahora = time.monotonic()
        while self.imagenes_pendientes:
            recibido, msg = self.imagenes_pendientes[0]
            t_img = self.tiempo_imagen(msg)
            edad = (self.get_clock().now().nanoseconds - t_img.nanoseconds) / 1e9
            if edad > self.cfg['edad_maxima_imagen_s']:
                self.imagenes_pendientes.popleft()
                self.diag['vencidas'] = self.diag.get('vencidas', 0) + 1
                continue
            if ahora - recibido < self.cfg['espera_tf_vision_s'] or not self.scans:
                return
            def desfase(scan):
                inicio, fin = self.tiempos_scan(scan)
                return max(inicio.nanoseconds - t_img.nanoseconds,
                           t_img.nanoseconds - fin.nanoseconds, 0)
            scan = min(self.scans, key=desfase)
            if desfase(scan) > self.cfg['max_desfase_scan_imagen_s'] * 1e9:
                return
            base = self.cfg['frame_base']
            if not self.tf_buffer.can_transform(self.mapa['frame'], base, t_img):
                return
            if not all(self.tf_buffer.can_transform_full(
                    base, t_img, base, stamp, self.cfg['frame_odom'])
                    for stamp in self.tiempos_scan(scan)):
                return
            self.imagenes_pendientes.popleft()
            try:
                self.procesar(msg, scan)
            except Exception as e:  # noqa: BLE001  (nunca tirar abajo el nodo por un frame)
                self.get_logger().warn(f'frame descartado: {e}', throttle_duration_sec=5.0)
            return

    # ------------------------------------------------------------ geometría
    def tf(self, padre, hijo, stamp=None):
        try:
            if stamp is not None:
                return self.tf_buffer.lookup_transform(padre, hijo, stamp)
            return self.tf_buffer.lookup_transform(padre, hijo, Time())
        except Exception as error:  # noqa: BLE001
            self.diag_ult = f'TF {padre} <- {hijo}: {error}'
            return None

    def puntos_scan_en_base(self, stamp_imagen=None, scan=None):
        """Puntos del último scan en el marco base. Con `stamp_imagen`, los lleva a
        donde estaba el robot en el instante de la imagen (con la odometría): si el
        robot gira, el scan y la imagen no son del mismo momento y sin esto la
        caja "se corre" de lado."""
        s = scan if scan is not None else self.scan
        t = self.tf(self.cfg['frame_base'], s.header.frame_id)
        if t is None:
            return None
        r = np.asarray(s.ranges, dtype=np.float64)
        a = s.angle_min + np.arange(r.size) * s.angle_increment
        ok = np.isfinite(r) & (r > max(s.range_min, 0.02)) & (r < min(s.range_max, 6.0))
        xl, yl = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        R = quat_a_matriz(t.transform.rotation)
        tr = t.transform.translation
        x, y = R[0, 0] * xl + R[0, 1] * yl + tr.x, R[1, 0] * xl + R[1, 1] * yl + tr.y
        if stamp_imagen is not None:
            # El LiDAR barre los 360° en ~0.13 s: cada rayo es de un instante distinto
            # (stamp + i * time_increment). Se calcula el movimiento del robot entre el
            # principio y el fin del barrido y el instante de la imagen, y se interpola
            # para cada rayo. Girando a 1 rad/s son hasta 7° de diferencia.
            base = self.cfg['frame_base']
            dt_rayo = float(getattr(s, 'time_increment', 0.0) or 0.0)
            t_ini = Time.from_msg(s.header.stamp)
            instantes = [t_ini]
            if dt_rayo > 0.0 and r.size > 1:
                instantes.append(t_ini + Duration(nanoseconds=int(dt_rayo * (r.size - 1) * 1e9)))
            deltas = []
            try:
                for ti in instantes:
                    m = self.tf_buffer.lookup_transform_full(
                        base, stamp_imagen, base, ti, self.cfg['frame_odom'])
                    q = m.transform.rotation
                    deltas.append((m.transform.translation.x, m.transform.translation.y,
                                   math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))))
            except Exception as error:  # noqa: BLE001  no ubicar un objeto con poses de otros instantes
                self.diag_ult = f'TF para compensar cámara/LiDAR: {error}'
                return None
            if deltas:
                if len(deltas) == 2:
                    f = (np.arange(r.size, dtype=np.float64) / (r.size - 1))[ok]
                    dx = deltas[0][0] + f * (deltas[1][0] - deltas[0][0])
                    dy = deltas[0][1] + f * (deltas[1][1] - deltas[0][1])
                    da = deltas[0][2] + f * ang_norm(deltas[1][2] - deltas[0][2])
                else:
                    dx, dy, da = deltas[0]
                c, sn = np.cos(da), np.sin(da)
                x, y = c * x - sn * y + dx, sn * x + c * y + dy
        return x, y

    def a_celda(self, x, y):
        m = self.mapa
        return int(math.floor((y - m['oy']) / m['res'])), int(math.floor((x - m['ox']) / m['res']))

    def distancia_a_pared_por_rayo(self, x0, y0, ang, maximo):
        """Distancia (m) desde (x0, y0) hasta la primera celda no libre del mapa en la
        dirección `ang` (marco map). `maximo` si no encuentra pared antes."""
        m = self.mapa
        paso = 0.5 * m['res']
        c, s = math.cos(ang), math.sin(ang)
        d = 0.0
        while d < maximo:
            d += paso
            f, col = self.a_celda(x0 + c * d, y0 + s * d)
            if not (0 <= f < m['alto'] and 0 <= col < m['ancho']) or not m['libre'][f, col]:
                return d
        return maximo

    def hundido_en_pared(self, x, y):
        """0 si (x, y) está en piso libre; si no, distancia (m) a la celda libre más
        cercana. Infinito fuera del mapa."""
        f, c = self.a_celda(x, y)
        m = self.mapa
        if not (0 <= f < m['alto'] and 0 <= c < m['ancho']):
            return float('inf')
        return 0.0 if m['libre'][f, c] else float(m['dist_libre'][f, c])

    def dist_pared(self, x, y):
        f, c = self.a_celda(x, y)
        m = self.mapa
        if not (0 <= f < m['alto'] and 0 <= c < m['ancho']):
            return -1.0
        return float(m['dist'][f, c]) if m['libre'][f, c] else 0.0

    # ------------------------------------------------------------ detección
    def procesar(self, msg, scan=None):
        scan = scan if scan is not None else self.scan
        bgr = imagen_a_bgr(msg.encoding, msg.height, msg.width, msg.step, msg.data)
        if bgr is None:
            self.get_logger().warn(f'encoding no soportado: {msg.encoding}', throttle_duration_sec=10.0)
            return
        if not self.localizacion_ok():
            self.diag['loc_mala'] = self.diag.get('loc_mala', 0) + 1
            return                        # con la pose dudosa, la caja quedaría en otro lugar
        manchas, _ = segmentar_rojo(bgr, self.cfg)
        colores = segmentar_colores(bgr, self.cfg) if self.cfg.get('objetos_por_color', True) else []
        if colores:
            self.diag['con_color'] = self.diag.get('con_color', 0) + 1
        candidatos = ([g for g in self.grupos if not g.reportado and not g.descartado]
                      if self.cfg['descartar_no_vistas'] else [])
        if manchas:
            self.diag['con_rojo'] += 1
            self.diag['manchas'] += len(manchas)
        fx, fy, cx, cy = self.K if self.K else (271.8, 271.8, (msg.width - 1) / 2.0, (msg.height - 1) / 2.0)
        base = self.cfg['frame_base']
        t_cam = self.tf(base, msg.header.frame_id)
        t_img = self.tiempo_imagen(msg)
        t_scan = Time.from_msg(scan.header.stamp).nanoseconds
        duracion_scan = max(0.0, scan.time_increment * (len(scan.ranges) - 1)) * 1e9
        desfase = max(t_scan - t_img.nanoseconds, t_img.nanoseconds - t_scan - duracion_scan, 0.0)
        if not t_scan or desfase > self.cfg['max_desfase_scan_imagen_s'] * 1e9:
            self.diag_ult = f'Desfase imagen/LiDAR: {desfase / 1e9:.3f} s'
            return
        t_map = self.tf(self.mapa['frame'], base, t_img)
        t_laser = self.tf(base, scan.header.frame_id)
        puntos = self.puntos_scan_en_base(t_img, scan)
        if t_cam is None or t_map is None or t_laser is None or puntos is None:
            if manchas or colores:
                self.diag['sin_tf'] += 1
            return
        R_cam = quat_a_matriz(t_cam.transform.rotation)
        cam_xy = (t_cam.transform.translation.x, t_cam.transform.translation.y)
        R_map = quat_a_matriz(t_map.transform.rotation)
        T_map = t_map.transform.translation
        t_ahora = time.monotonic()
        borde = int(self.cfg['margen_borde_px'])
        # Rumbos (marco base, vistos desde la cámara) que ocupa CADA mancha roja,
        # aunque después se la descarte por un filtro: para no borrar una caja
        # real que un filtro rechazó en esta imagen.
        rojos = [rumbos_de_mancha(u0, u1, 0.5 * (v0 + v1), fx, fy, cx, cy, R_cam, recorte=0.0)
                 for u0, u1, v0, v1, _a in manchas]
        observaciones, usados = [], set()
        for color, mancha in ([('rojo', m) for m in manchas] + [(m[5], m[:5]) for m in colores]):
            u0, u1, v0, v1, _area = mancha
            if borde > 0 and (u0 < borde or u1 > msg.width - 1 - borde):
                # Caja cortada por el borde de la imagen: el centro de la mancha no es el
                # de la caja y el láser agarra lo de al lado. Se espera a verla entera.
                self.diag['borde'] = self.diag.get('borde', 0) + 1
                continue
            rumbo, dmin, dmax = rumbos_de_mancha(u0, u1, 0.5 * (v0 + v1), fx, fy, cx, cy, R_cam)
            res = centro_de_caja(puntos[0], puntos[1], cam_xy, rumbo, dmin, dmax,
                                 self.cfg['rango_maximo'], self.cfg['medio_lado_caja'])
            if res is None:
                self.diag['sin_laser'] += 1
                self.diag_ult = f'sin puntos del láser en el rumbo {math.degrees(rumbo):.0f}° (área {_area} px)'
                continue
            bx, by, cara_x, cara_y, _n, ang_cara = res
            cerca = np.hypot(puntos[0] - cara_x, puntos[1] - cara_y) <= 0.12
            if color_superficie(bgr, puntos[0][cerca], puntos[1][cerca],
                                t_laser.transform.translation.z, t_cam, (fx, fy, cx, cy), self.cfg) != color:
                continue  # el color del piso o del fondo no pertenece al retorno delantero
            # Chequeo con el SUELO: la caja apoya en el piso, así que el borde de abajo
            # de la mancha da otra estimación de la distancia (altura de la cámara y
            # rayo hacia el piso). Si el láser llega bastante MÁS LEJOS que eso, en
            # ese rumbo pasó de largo (pegó en otra cosa detrás de la caja): la
            # posición saldría corrida. Con el bag real: detecciones buenas ≤ 0.11 m,
            # malas ≥ 0.27 m.
            if v1 < msg.height - 5:
                d = R_cam @ np.array([(0.5 * (u0 + u1) - cx) / fx, (v1 + 0.5 - cy) / fy, 1.0])
                if d[2] < -1e-6:
                    s_ = -t_cam.transform.translation.z / d[2]
                    d_suelo = math.hypot(s_ * d[0], s_ * d[1])
                    d_laser = math.hypot(cara_x - cam_xy[0], cara_y - cam_xy[1])
                    if d_laser - d_suelo > self.cfg['max_laser_detras_suelo']:
                        self.diag['detras'] = self.diag.get('detras', 0) + 1
                        self.diag_ult = (f'el láser marca {d_laser:.2f} m pero la base de la caja está a '
                                         f'{d_suelo:.2f} m: pegó en otra cosa')
                        continue
            mx = R_map[0, 0] * bx + R_map[0, 1] * by + T_map.x
            my = R_map[1, 0] * bx + R_map[1, 1] * by + T_map.y
            fmx = R_map[0, 0] * cara_x + R_map[0, 1] * cara_y + T_map.x
            fmy = R_map[1, 0] * cara_x + R_map[1, 1] * cara_y + T_map.y
            # Filtros con el mapa: las víctimas NO están en /map.
            # 1) Lo rojo no puede estar justo donde el mapa pone la pared en esa
            #    dirección (sería algo pintado/pegado en la pared). Si el láser llega
            #    MÁS LEJOS que la pared del mapa (mapa corrido unos cm, típico de una
            #    caja contra una pared vista de costado), se acepta salvo que los
            #    puntos formen una superficie casi paralela al rayo (pared rasante).
            cam_mx = R_map[0, 0] * cam_xy[0] + R_map[0, 1] * cam_xy[1] + T_map.x
            cam_my = R_map[1, 0] * cam_xy[0] + R_map[1, 1] * cam_xy[1] + T_map.y
            d_cara = math.hypot(fmx - cam_mx, fmy - cam_my)
            ang = math.atan2(fmy - cam_my, fmx - cam_mx)
            d_mapa = self.distancia_a_pared_por_rayo(cam_mx, cam_my, ang, d_cara + 1.0)
            margen = self.cfg['margen_frente_pared']
            if abs(d_cara - d_mapa) < margen:
                self.diag['cara_en_pared'] += 1
                self.diag_ult = (f'cara en ({fmx:.2f}, {fmy:.2f}) a {d_cara:.2f} m, la pared del mapa '
                                 f'está a {d_mapa:.2f} m: es la pared')
                continue
            if d_cara >= d_mapa + margen and not (ang_cara >= self.cfg['angulo_rasante_deg']):
                self.diag['cara_en_pared'] += 1
                self.diag_ult = (f'cara en ({fmx:.2f}, {fmy:.2f}) detrás de la pared del mapa y casi '
                                 f'paralela al rayo ({ang_cara:.0f}°): pared vista de costado')
                continue
            # 2) Ni la cara ni el centro pueden quedar hundidos en una pared (o fuera
            #    del mapa). Se tolera max_dentro_pared: el mapa tiene celdas de 5 cm
            #    y una caja pegada a la pared "pisa" la primera fila de pared.
            if self.hundido_en_pared(fmx, fmy) > self.cfg['max_dentro_pared']:
                self.diag['cara_en_pared'] += 1
                self.diag_ult = f'cara en ({fmx:.2f}, {fmy:.2f}) adentro de una pared del mapa'
                continue
            if self.hundido_en_pared(mx, my) > self.cfg['max_dentro_pared']:
                self.diag['centro_en_pared'] += 1
                self.diag_ult = f'centro en ({mx:.2f}, {my:.2f}) adentro de una pared o fuera del mapa'
                continue
            self.diag['aceptadas'] += 1
            distancia = math.hypot(cara_x - cam_xy[0], cara_y - cam_xy[1])
            grupos = self.grupos if color == 'rojo' else self.grupos_objetos
            existente = next((g for g in grupos
                             if math.dist((mx, my), g.posicion()) < self.cfg['radio_agrupamiento']), None)
            if existente is not None and id(existente) in usados:
                continue  # una caja cuenta una sola vez por imagen
            if color == 'rojo':
                grupo = self.agregar_deteccion(mx, my, t_ahora, distancia)
            else:
                if color in ('naranja', 'magenta') and any(
                        math.dist((mx, my), g.posicion()) < self.cfg['radio_agrupamiento'] for g in self.grupos):
                    continue  # tono vecino del rojo junto a una víctima: es el borde de la víctima
                grupo = self.agregar_objeto(mx, my, t_ahora, distancia, color, t_scan)
            if grupo is None:
                continue
            usados.add(id(grupo))
            confirmado = bool(grupo.n >= self.cfg['detecciones_minimas']
                          and grupo.dispersion() <= self.cfg['dispersion_maxima']
                          and grupo.t_ult - grupo.t0 >= self.cfg['segundos_minimos_vista'])
            gx, gy = grupo.posicion()
            eje = R_cam[:, 2]
            if color != 'rojo':
                observaciones.append(dict(id=f'objeto-{grupos.index(grupo)}', color=grupo.color,
                    tipo='objeto' if confirmado else 'objeto_nuevo', nuevo=existente is None,
                    x=gx, y=gy, n=grupo.n, confirmado=confirmado, reportado=False,
                    completo=bool(v0 >= max(1, borde) and v1 < msg.height - max(1, borde)),
                    distancia=math.hypot(bx - cam_xy[0], by - cam_xy[1]),
                    error_angular=ang_norm(rumbo - math.atan2(eje[1], eje[0]))))
                continue
            observaciones.append(dict(id=f'rojo-{grupos.index(grupo)}', color='rojo',
                tipo='caja' if confirmado else 'objeto_nuevo', nuevo=existente is None,
                x=gx, y=gy, n=grupo.n, confirmado=confirmado, reportado=grupo.reportado,
                completo=bool(v0 >= max(1, borde) and v1 < msg.height - max(1, borde)),
                distancia=math.hypot(bx - cam_xy[0], by - cam_xy[1]),
                error_angular=ang_norm(rumbo - math.atan2(eje[1], eje[0]))))
        observaciones.extend(self.observar_superficies(bgr, puntos, t_laser, t_cam, R_map,
                                                       T_map, (fx, fy, cx, cy), observaciones, t_scan))
        self.revisar_ausentes(candidatos, rojos, R_cam, cam_xy, R_map, T_map, puntos, fx, msg.width)
        salida = String()
        salida.data = json.dumps(dict(version=1, frame=self.mapa['frame'],
            stamp_ns=t_img.nanoseconds, objetos=observaciones), allow_nan=False)
        self.pub_observaciones.publish(salida)

    def observar_superficies(self, bgr, puntos, t_laser, t_cam, R_map, T_map, K, rojas, stamp_scan):
        """El LiDAR propone superficies libres del mapa; la cámara sólo las describe."""
        px, py = puntos
        mx = R_map[0, 0] * px + R_map[0, 1] * py + T_map.x
        my = R_map[1, 0] * px + R_map[1, 1] * py + T_map.y
        tr = t_cam.transform.translation
        cam = R_map @ np.array([tr.x, tr.y, tr.z]) + [T_map.x, T_map.y, T_map.z]
        dist = np.hypot(px - tr.x, py - tr.y)
        validos = (dist >= 0.25) & (dist <= self.cfg['rango_maximo'])
        m = self.mapa
        grupos = grupos_lidar(mx, my, validos, m['libre'], m['dist'],
                             m['res'], m['ox'], m['oy'], self.cfg['separacion_pared_objeto_m'])
        salida, usados = [], set()
        eje = quat_a_matriz(t_cam.transform.rotation)[:, 2]
        for indices in grupos:
            cara_x, cara_y = float(np.median(px[indices])), float(np.median(py[indices]))
            dx, dy = cara_x - tr.x, cara_y - tr.y
            distancia = math.hypot(dx, dy)
            fmx, fmy = float(np.median(mx[indices])), float(np.median(my[indices]))
            ang = math.atan2(fmy - cam[1], fmx - cam[0])
            d_mapa = self.distancia_a_pared_por_rayo(cam[0], cam[1], ang, distancia + 1.0)
            if d_mapa <= distancia + self.cfg['margen_frente_pared']:
                continue  # ni una pared de otro color ni un eco detrás de ella es un objeto
            color = color_superficie(bgr, px[indices], py[indices], t_laser.transform.translation.z,
                                     t_cam, K, self.cfg)
            if color is None or color == 'rojo':
                continue  # las víctimas conservan su circuito de confirmación y reporte
            bx, by = centro_superficie_lidar(px[indices], py[indices], (tr.x, tr.y),
                                             self.cfg['medio_lado_caja'])
            x = R_map[0, 0] * bx + R_map[0, 1] * by + T_map.x
            y = R_map[1, 0] * bx + R_map[1, 1] * by + T_map.y
            if self.hundido_en_pared(x, y) > 0 or any(
                    math.dist((x, y), (o['x'], o['y'])) < self.cfg['radio_agrupamiento'] for o in rojas):
                continue
            if color not in NOMBRES_COLORES and any(
                    math.dist((x, y), g.posicion()) < self.cfg['radio_agrupamiento'] for g in self.grupos):
                continue  # superficie sin color en una víctima ya vista: es la víctima
            grupo = min(self.grupos_objetos, key=lambda g: math.dist((x, y), g.posicion()), default=None)
            nuevo = grupo is None or math.dist((x, y), grupo.posicion()) >= self.cfg['radio_agrupamiento']
            if nuevo:
                grupo = Grupo(x, y, time.monotonic(), distancia)
                grupo.ultimo_scan = stamp_scan
                grupo.color = color
                self.grupos_objetos.append(grupo)
                self.get_logger().info(f'>>> OBJETO NUEVO ({color}) en ({x:.2f}, {y:.2f}); superficie LiDAR')
            elif id(grupo) not in usados and stamp_scan > getattr(grupo, 'ultimo_scan', -1):
                grupo.agregar(x, y, time.monotonic(), distancia)
                grupo.ultimo_scan = stamp_scan
            if id(grupo) in usados:
                continue
            usados.add(id(grupo))
            confirmado = bool(grupo.n >= self.cfg['detecciones_minimas']
                and grupo.dispersion() <= self.cfg['dispersion_maxima']
                and grupo.t_ult - grupo.t0 >= self.cfg['segundos_minimos_vista'])
            gx, gy = grupo.posicion()
            if getattr(grupo, 'color', None) in NOMBRES_COLORES:
                color = grupo.color          # el color de la mancha vista por la cámara manda
            elif color in NOMBRES_COLORES:
                grupo.color = color
            salida.append(dict(id=f'objeto-{self.grupos_objetos.index(grupo)}', color=color,
                tipo='objeto' if confirmado else 'objeto_nuevo', nuevo=nuevo,
                x=gx, y=gy, n=grupo.n, confirmado=confirmado, reportado=False, completo=False,
                distancia=math.hypot(bx - tr.x, by - tr.y),
                error_angular=ang_norm(math.atan2(dy, dx) - math.atan2(eje[1], eje[0]))))
        return salida

    def revisar_ausentes(self, candidatos, rojos, R_cam, cam_xy, R_map, T_map, puntos, fx, ancho):
        """Evidencia en contra: una posible víctima que debería estar entera en
        cuadro (cerca, dentro del campo de visión y sin nada delante según el
        láser) y en la imagen no hay NADA rojo en esa dirección. Tras
        `imagenes_para_descartar` imágenes así (sin ninguna detección suya en el
        medio), se la borra de las posibles: no se reporta ni se la va a mirar."""
        if not candidatos:
            return
        eje = R_cam[:, 2]                         # eje óptico, en el marco base
        if math.hypot(eje[0], eje[1]) < 0.5:
            return
        yaw_cam = math.atan2(eje[1], eje[0])
        medio_fov = math.atan(0.5 * ancho / fx)
        margen = math.radians(self.cfg['margen_fov_deg'])
        cerca = self.cfg['distancia_confirmacion']
        px, py = puntos
        ang_puntos = np.arctan2(py, px)
        r_puntos = np.hypot(px, py)
        for g in candidatos:
            gx, gy = g.posicion()
            dx, dy = gx - T_map.x, gy - T_map.y   # map -> base
            bx = R_map[0, 0] * dx + R_map[1, 0] * dy
            by = R_map[0, 1] * dx + R_map[1, 1] * dy
            d = math.hypot(bx - cam_xy[0], by - cam_xy[1])
            if not 0.4 <= d <= cerca:
                continue
            rumbo_cam = math.atan2(by - cam_xy[1], bx - cam_xy[0])
            medio_caja = math.atan(0.15 / d)      # media caja (con margen) vista desde la cámara
            if abs(ang_norm(rumbo_cam - yaw_cam)) > medio_fov - medio_caja - margen:
                continue                          # no entraría entera en la imagen
            if any(ang_norm(rumbo_cam - (c + a)) >= -medio_caja - margen
                   and ang_norm(rumbo_cam - (c + b)) <= medio_caja + margen for c, a, b in rojos):
                continue                          # hay rojo en esa dirección: no es evidencia en contra
            # ¿Algo la tapa? El láser en ese rumbo tiene que llegar hasta la caja.
            rumbo = math.atan2(by, bx)
            en_rumbo = np.abs(np.arctan2(np.sin(ang_puntos - rumbo), np.cos(ang_puntos - rumbo))) < math.radians(3)
            if not en_rumbo.any() or float(r_puntos[en_rumbo].min()) < math.hypot(bx, by) - 0.30:
                continue
            g.ausente += 1
            if g.ausente >= self.cfg['imagenes_para_descartar']:
                g.descartado = True
                self.get_logger().info(
                    f'>>> Descarto la posible víctima en ({gx:.2f}, {gy:.2f}): la miré de cerca '
                    f'({d:.1f} m) en {g.ausente} imágenes y no hay nada rojo ahí '
                    f'(tenía {g.n} detecciones, la más cercana a {g.d_min:.1f} m)')

    def agregar_objeto(self, x, y, t, distancia, color, stamp_scan):
        """Objeto no rojo visto por la cámara (mancha) y ubicado con el LiDAR."""
        grupo = min(self.grupos_objetos, key=lambda g: math.dist((x, y), g.posicion()), default=None)
        if grupo is None or math.dist((x, y), grupo.posicion()) >= self.cfg['radio_agrupamiento']:
            grupo = Grupo(x, y, t, distancia)
            grupo.color = color
            grupo.ultimo_scan = stamp_scan
            self.grupos_objetos.append(grupo)
            self.get_logger().info(f'>>> OBJETO NUEVO ({color}) en ({x:.2f}, {y:.2f}); cámara + LiDAR')
        else:
            if stamp_scan > getattr(grupo, 'ultimo_scan', -1):    # un barrido = una evidencia
                grupo.agregar(x, y, t, distancia)
                grupo.ultimo_scan = stamp_scan
            if getattr(grupo, 'color', None) not in NOMBRES_COLORES:
                grupo.color = color
        return grupo

    def agregar_deteccion(self, x, y, t, distancia=None):
        mejor, dmin = None, self.cfg['radio_agrupamiento']
        for g in self.grupos:
            gx, gy = g.posicion()
            d = math.hypot(x - gx, y - gy)
            if d < dmin:
                mejor, dmin = g, d
        if mejor is not None and mejor.descartado:
            if distancia is None or distancia > self.cfg['distancia_confirmacion']:
                return                            # de lejos otra vez el mismo reflejo: ignorar
            mejor.descartado = False              # de cerca sí está: vuelve a ser posible
            self.get_logger().info(f'La posible víctima en ({x:.2f}, {y:.2f}) se ve de cerca: '
                                   'deja de estar descartada')
        if mejor is None:
            mejor = Grupo(x, y, t, distancia)
            self.grupos.append(mejor)
            a = f' a {distancia:.1f} m del robot' if distancia is not None else ''
            self.get_logger().info(
                f'>>> CAJA ROJA VISTA en ({x:.2f}, {y:.2f}){a} -> posible víctima #{len(self.grupos)}, '
                'confirmando...')
        else:
            mejor.agregar(x, y, t, distancia)
            if mejor.n == self.cfg['detecciones_minimas'] and not mejor.reportado:
                gx, gy = mejor.posicion()
                self.get_logger().info(
                    f'>>> Víctima en ({gx:.2f}, {gy:.2f}) confirmada '
                    f'({mejor.n} detecciones, dispersión {100 * mejor.dispersion():.0f} cm)')
        return mejor

    # -------------------------------------------------------------- reporte
    def reportar(self, g, motivo):
        if self.reportes >= self.n_max or g.reportado or getattr(self, 'reportes_cerrados', False):
            return
        x, y = g.posicion()
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        # Un grupo pegado a una víctima ya reportada es la misma caja vista corrida
        # (giro, localización): reportarlo gastaría uno de los N reportes.
        for o in self.grupos:
            if o.reportado and math.hypot(x - o.posicion()[0], y - o.posicion()[1]) < \
                    self.cfg['separacion_minima_cajas']:
                if not getattr(g, 'duplicado', False):
                    g.duplicado = True
                    self.get_logger().info(
                        f'Grupo en ({x:.2f}, {y:.2f}) a menos de '
                        f'{self.cfg["separacion_minima_cajas"]:.2f} m de una víctima ya '
                        'reportada: es la misma caja, no lo reporto')
                return
        m = PointStamped()
        m.header.frame_id = 'map'             # §7.2: frame_id distinto de 'map' = reporte perdido
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x, m.point.y, m.point.z = x, y, 0.0
        self.pub.publish(m)
        g.reportado = True
        self.reportes += 1
        t = time.monotonic() - self.t_mapa if self.t_mapa else 0.0
        evidencia = ('posición estimada sin confirmación visual' if getattr(g, 'estimado', False)
                     else f'{g.n} detecciones, dispersión {100 * g.dispersion():.0f} cm')
        self.get_logger().info(
            f'****** VÍCTIMA {self.reportes}/{self.n_max} REPORTADA en ({x:.2f}, {y:.2f}) '
            f'[{motivo}: {evidencia}] t={t:.0f}s ******')

    def cerrar_reportes(self, _req, respuesta):
        """Consume el cupo restante una sola vez, antes de que exploración emita /done."""
        if self.mapa is None:
            respuesta.success, respuesta.message = False, 'Sin mapa para estimar posiciones'
            return respuesta
        if not self.reportes_cerrados:
            self.revisar_reportes()
            candidatos = [g for g in self.grupos if not g.reportado and not g.descartado
                          and not getattr(g, 'duplicado', False)]
            for g in sorted(candidatos, key=lambda g: (-g.n, g.dispersion(), g.d_min)):
                self.reportar(g, 'detección probable al cerrar')
            if self.cfg['estimar_al_cerrar'] and self.reportes < self.n_max:
                excluidos = [g.posicion() for g in self.grupos + self.grupos_objetos]
                lugares = lugares_probables(
                    self.mapa, self.visto, self.recorrido, excluidos, self.n_max - self.reportes,
                    self.cfg['medio_lado_caja'],
                    max(self.cfg['separacion_estimaciones_m'], self.cfg['separacion_minima_cajas']))
                for x, y in lugares:
                    g = Grupo(x, y, time.monotonic())
                    g.estimado = True
                    self.grupos.append(g)
                    self.reportar(g, 'estimación de zona poco observada al cerrar')
            self.reportes_cerrados = True
            self.publicar_debug()
        respuesta.success, respuesta.message = True, f'{self.reportes}/{self.n_max} reportes publicados'
        return respuesta

    def revisar_reportes(self):
        if self.mapa is None or getattr(self, 'reportes_cerrados', False):
            return
        ahora = time.monotonic()
        c = self.cfg
        for g in self.grupos:
            if not g.reportado and not g.descartado and self.en_recorrido(*g.posicion()):
                g.descartado = True
                x, y = g.posicion()
                self.get_logger().info(f'>>> Descarto la posible víctima en ({x:.2f}, {y:.2f}): el robot ya '
                                       'pasó por ahí (es otra caja vista con la pose corrida)')
        pendientes = [g for g in self.grupos if not g.reportado and not g.descartado]
        cerca = c['distancia_confirmacion']
        # 1) Confirmadas: muchas detecciones, consistentes, vistas un rato y al
        #    menos una vez de cerca.
        for g in sorted(pendientes, key=lambda g: -g.n):
            if (g.n >= c['detecciones_minimas'] and g.dispersion() <= c['dispersion_maxima']
                    and g.t_ult - g.t0 >= c['segundos_minimos_vista'] and g.d_min <= cerca):
                self.reportar(g, 'confirmada')
        # 2) Vistas de pasada: pocas detecciones pero el robot ya no las ve.
        for g in sorted(pendientes, key=lambda g: -g.n):
            if (not g.reportado and g.n >= c['detecciones_minimas_paso']
                    and ahora - g.t_ult > c['segundos_sin_ver_paso']
                    and g.dispersion() <= 2 * c['dispersion_maxima'] and g.d_min <= cerca):
                self.reportar(g, 'vista de pasada')
        # 3) Final de la corrida: los reportes que sobran no valen nada; usarlos
        #    (aunque solo se las haya visto de lejos).
        if self.t_mapa and ahora - self.t_mapa > c['segundos_reporte_final']:
            for g in sorted([g for g in self.grupos if not g.reportado and not g.descartado],
                            key=lambda g: -g.n):
                if g.n >= 3:
                    self.reportar(g, 'reporte final')
        self.publicar_pendientes()
        self.publicar_debug()

    def publicar_pendientes(self):
        """Cajas vistas pero todavía sin reportar (posibles víctimas), para que el
        explorador se acerque a confirmarlas. Topic interno del equipo."""
        if self.reportes >= self.n_max:
            return
        arr = PoseArray()
        arr.header.frame_id = 'map'
        arr.header.stamp = self.get_clock().now().to_msg()
        reportadas = [r.posicion() for r in self.grupos if r.reportado]
        for g in self.grupos:
            if g.reportado or g.descartado or g.n < 3 or getattr(g, 'duplicado', False):   # 1-2 manchas sueltas: no vale el desvío
                continue
            x, y = g.posicion()
            if any(math.hypot(x - rx, y - ry) < self.cfg['separacion_minima_cajas'] for rx, ry in reportadas):
                continue                      # es la misma caja que una ya reportada
            p = Pose()
            p.position.x, p.position.y = x, y
            p.position.z = float(g.n)          # cuántas detecciones lleva
            p.orientation.w = 1.0
            arr.poses.append(p)
        self.pub_pendientes.publish(arr)

    def imprimir_diagnostico(self):
        d = self.diag
        self.get_logger().info(
            f'[diag] imágenes {d["imagenes"]} (girando {d["girando"]}) | con rojo {d["con_rojo"]} '
            f'({d["manchas"]} manchas), con azul {d.get("con_azul", 0)} | descartes: sin TF {d["sin_tf"]}, sin láser {d["sin_laser"]}, '
            f'cara en pared {d["cara_en_pared"]}, centro en pared {d["centro_en_pared"]}, '
            f'láser detrás {d.get("detras", 0)}, borde {d.get("borde", 0)} | '
            f'aceptadas {d["aceptadas"]}' + (f' | último descarte: {self.diag_ult}' if self.diag_ult else ''))
        for k in d:
            d[k] = 0
        self.diag_ult = ''

    def publicar_debug(self):
        if not self.cfg['publicar_debug']:
            return
        arr = MarkerArray()
        borrar = Marker()
        borrar.action = Marker.DELETEALL
        arr.markers.append(borrar)
        for i, g in enumerate(self.grupos):
            x, y = g.posicion()
            for tipo, ns, esc, z in ((Marker.CYLINDER, 'grupos', 0.25, 0.15), (Marker.TEXT_VIEW_FACING, 'texto', 0.18, 0.45)):
                mk = Marker()
                mk.header.frame_id = 'map'
                mk.ns, mk.id, mk.type = ns, i, tipo
                mk.pose.position.x, mk.pose.position.y, mk.pose.position.z = x, y, z
                mk.pose.orientation.w = 1.0
                mk.scale.x = mk.scale.y = esc
                mk.scale.z = 0.3 if tipo == Marker.CYLINDER else esc
                if g.reportado:
                    mk.color.r, mk.color.g, mk.color.b = 0.1, 1.0, 0.2
                elif g.descartado:
                    mk.color.r, mk.color.g, mk.color.b = 0.5, 0.5, 0.5
                else:
                    mk.color.r, mk.color.g, mk.color.b = 1.0, 0.8, 0.0
                mk.color.a = 0.8 if tipo == Marker.CYLINDER else 1.0
                if tipo == Marker.TEXT_VIEW_FACING:
                    mk.text = ('EST sin confirmar' if getattr(g, 'estimado', False)
                               else f"{'OK' if g.reportado else 'X' if g.descartado else '?'} n={g.n}")
                arr.markers.append(mk)
        self.pub_marc.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    nodo = DeteccionVictimas()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
