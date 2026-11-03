"""Explorador de competencia: Next-Best-View sobre el mapa + regreso a la largada.

------------------------------
1. ESPERA /map.
2. LOCALIZA: busca cerca del origen la pose donde el láser mejor calza con las
   paredes del mapa se la da a AMCL y da una vuelta en el lugar para que AMCL 
   converja y la cámara vea alrededor.
   Durante toda la corrida mide si el láser sigue calzando con el mapa: si se
   desfasa, frena, redistribuye las partículas de AMCL y espera convergencia
   con un giro comprobado contra colisiones. Baja la velocidad de Nav2
   cerca de paredes/obstáculos o si la localización empeora.
3. EXPLORA con Next-Best-View (NBV):
   - `visto`: grilla de piso libre que ya pasó por el cono de la cámara
     (raycasting contra las paredes del mapa desde la pose de AMCL).
   - Candidatos: celdas libres con margen a las paredes Y a los obstáculos no
     mapeados que ve el LiDAR (se leen del costmap global de Nav2).
   - Ganancia de un candidato: celdas libres no vistas dentro de la mejor
     "porción" de ancho FOV (eso define también hacia dónde mirar).
   - Utilidad = ganancia * exp(-lambda * distancia por camino) (BFS, no recta).
   - Mientras viaja sigue marcando lo visto; si el goal ya "se agotó" en el
     camino, cambia al siguiente sin frenar. Al llegar, gira solo si hace falta.
   - Anti-atascos: 10 s sin avanzar -> lista negra, limpia costmaps, se
     DESENCAJA solo (se mueve ~15 cm hacia donde el LiDAR ve más espacio,
4. VUELVE A LA LARGADA cuando pasa lo primero de:
   - ya se reportaron N víctimas en /victimas 
   - ya no quedan vistas útiles (cobertura completa);
   - el plazo: tiempo restante <= tiempo estimado de regreso + margen. Además
     nunca elige un punto desde el que no llegaría a volver a tiempo.
   La largada se estima anclada a la odometría (pose de odom al recibir el mapa
   transformada con la corrección de AMCL ya convergida), no con el (0, 0)
   Al final hace una aproximación fina (< 6 cm) y publica /done.

Interfaces: publica /cmd_vel (solo al desencajarse y en la aproximación final), /done y topics
internos (/initialpose para AMCL, /exploracion/* para RViz). Escucha /map,
/victimas, /scan, el encabezado de imagen de /cam_1/color/image_raw y topics
internos de Nav2. Nada de ground truth ni coordenadas fijas.
"""
import math
import os
import time
from collections import deque

import numpy as np
import rclpy
from equipo_jar.sectores_nbv import SectoresNBV
from equipo_jar.inflacion_adaptativa import InflacionAdaptativa, PARAMETROS_INFLACION
from equipo_jar.observacion_objetos import (
    ObjetosFueraMapa, leer_observaciones, objetos_del_scan,
)
from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PointStamped, PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose, Spin
from nav2_msgs.msg import SpeedLimit
from nav2_msgs.srv import ClearEntireCostmap
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.task import Future
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Empty, String
from std_srvs.srv import Empty as EmptySrv, Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray
from ament_index_python.packages import get_package_share_directory
from equipo_jar.visualizacion_cobertura import marcadores_sectores
from equipo_jar.planificador import ruta_por_portales, guias_por_zona, segmento_transitable, _distancia_al_borde

QOS_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
QOS_RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)

# Estados
ESPERANDO_MAPA = 'ESPERANDO_MAPA'
LOCALIZANDO = 'LOCALIZANDO'
ACTIVANDO_NAV2 = 'ACTIVANDO_NAV2'
ELIGIENDO = 'ELIGIENDO'
NAVEGANDO = 'NAVEGANDO'
COMPORTAMIENTO = 'COMPORTAMIENTO'   # Spin de behavior_server
DESENCAJANDO = 'DESENCAJANDO'       # escape propio con /scan + /cmd_vel
RELOCALIZANDO = 'RELOCALIZANDO'     # frenado: corrige AMCL calzando el láser con el mapa
OBSERVANDO_OBJETO = 'OBSERVANDO_OBJETO'
ENVIAR_PENDIENTE = 'ENVIAR_PENDIENTE'  # terminó de encarar la ruta: mandar el goal guardado
REGRESAR = 'REGRESAR'               # pedido de volver (se resuelve en el ciclo)
VOLVIENDO = 'VOLVIENDO'
APROXIMANDO = 'APROXIMANDO'         # ajuste fino final con /cmd_vel
TERMINADO = 'TERMINADO'
FINALIZANDO = 'FINALIZANDO'         # quieto: publica los reportes restantes antes de /done


def yaw_a_quat(yaw):
    return math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def quat_a_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def ang_norm(a):
    return math.atan2(math.sin(a), math.cos(a))


def contar_huecos_pequenos(visto, piso, resolucion, area_caja_m2):
    """Cuenta como visto cada componente sin revisar menor que una caja."""
    if area_caja_m2 <= resolucion * resolucion:
        return 0
    pendiente = np.asarray(piso, dtype=bool) & ~visto
    agregado = 0
    alto, ancho = pendiente.shape
    for fila, col in zip(*np.nonzero(pendiente)):
        if not pendiente[fila, col]:
            continue
        cola = [(int(fila), int(col))]
        pendiente[fila, col] = False
        componente = []
        while cola:
            f, c = cola.pop()
            componente.append((f, c))
            for df in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nf, nc = f + df, c + dc
                    if ((df or dc) and 0 <= nf < alto and 0 <= nc < ancho
                            and pendiente[nf, nc]):
                        pendiente[nf, nc] = False
                        cola.append((nf, nc))
        if len(componente) * resolucion ** 2 < area_caja_m2:
            for f, c in componente:
                visto[f, c] = True
            agregado += len(componente)
    return agregado


def buscar_arbol_portales():
    """Ruta del árbol de comportamiento de NavigateThroughPoses."""
    candidatos = [os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                               'behavior_trees', 'navegar_portales.xml')]
    try:
        candidatos.append(os.path.join(get_package_share_directory('equipo_jar'),
                                       'behavior_trees', 'navegar_portales.xml'))
    except Exception:  # noqa: BLE001
        pass
    return next((c for c in candidatos if os.path.isfile(c)), '')


def punto_para_mirar(candidatable, libre, dist_bfs, res, origen, caja, d_min=0.6, d_max=1.2,
                     d_ideal=0.9, fin_vista=0.12):
    """Celda desde donde mirar una caja de cerca: a d_min..d_max de ella, en piso
    candidato, alcanzable (dist_bfs >= 0) y con línea de vista libre (sin paredes
    del mapa entre medio) hasta `fin_vista` m antes del centro de la caja (su cara
    está a ~0.09 m). Minimiza camino + desvío de d_ideal. Devuelve
    (x, y, yaw_hacia_la_caja, metros_de_camino) o None."""
    ox, oy = origen
    bx, by = caja
    alto, ancho = candidatable.shape
    r = int(math.ceil(d_max / res)) + 1
    fb, cb = int(math.floor((by - oy) / res)), int(math.floor((bx - ox) / res))
    mejor = None
    for f in range(max(0, fb - r), min(alto, fb + r + 1)):
        for c in range(max(0, cb - r), min(ancho, cb + r + 1)):
            if not candidatable[f, c] or dist_bfs[f, c] < 0:
                continue
            x, y = ox + (c + 0.5) * res, oy + (f + 0.5) * res
            d = math.hypot(bx - x, by - y)
            if not d_min <= d <= d_max:
                continue
            # Línea de vista hasta casi la cara de la caja: una pared entre medio
            # (p. ej. la caja apoyada del otro lado) descarta el punto.
            pasos = max(1, int((d - fin_vista) / (0.5 * res)))
            visible = True
            for k in range(1, pasos + 1):
                t = k * (d - fin_vista) / pasos / d
                ff = int(math.floor((y + t * (by - y) - oy) / res))
                cc = int(math.floor((x + t * (bx - x) - ox) / res))
                if not (0 <= ff < alto and 0 <= cc < ancho) or not libre[ff, cc]:
                    visible = False
                    break
            if not visible:
                continue
            costo = dist_bfs[f, c] * res + 0.5 * abs(d - d_ideal)
            if mejor is None or costo < mejor[0]:
                mejor = (costo, x, y, math.atan2(by - y, bx - x), dist_bfs[f, c] * res)
    return None if mejor is None else mejor[1:]


def dilatar(mascara, radio_celdas):
    """Dilata una máscara booleana con un disco (sin scipy)."""
    if radio_celdas <= 0:
        return mascara.copy()
    r = int(math.ceil(radio_celdas))
    alto, ancho = mascara.shape
    pad = np.pad(mascara, r, mode='constant', constant_values=True)
    salida = np.zeros_like(mascara)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dx * dx + dy * dy <= radio_celdas * radio_celdas:
                salida |= pad[r + dy:r + dy + alto, r + dx:r + dx + ancho]
    return salida


def bfs(transitable, fila, col):
    """Distancias en celdas, sin atravesar esquinas ocupadas. -1 = inalcanzable."""
    alto, ancho = transitable.shape
    if not (0 <= fila < alto and 0 <= col < ancho):
        return None
    if not transitable[fila, col]:
        filas, cols = np.nonzero(transitable)
        if filas.size == 0:
            return None
        k = int(np.argmin((filas - fila) ** 2 + (cols - col) ** 2))
        fila, col = int(filas[k]), int(cols[k])
    dist = np.full(transitable.shape, -1.0)
    dist[fila, col] = 0.0
    cola = deque([(fila, col)])
    vecinos = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
               (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414))
    while cola:   # BFS por capas con costo diagonal: buena aproximación de la distancia real
        f, c = cola.popleft()
        d0 = dist[f, c]
        for df, dc, w in vecinos:
            nf, nc = f + df, c + dc
            if 0 <= nf < alto and 0 <= nc < ancho and transitable[nf, nc]:
                if df and dc and not (transitable[f, nc] and transitable[nf, c]):
                    continue
                nd = d0 + w
                if dist[nf, nc] < 0 or nd < dist[nf, nc] - 1e-6:
                    dist[nf, nc] = nd
                    cola.append((nf, nc))
    return dist


class Grilla:
    """Mapa + máscaras precalculadas."""

    def __init__(self, msg, margen_transito, margen_candidato):
        info = msg.info
        self.res = info.resolution
        self.ancho = info.width
        self.alto = info.height
        self.ox = info.origin.position.x
        self.oy = info.origin.position.y
        self.frame = msg.header.frame_id or 'map'
        datos = np.array(msg.data, dtype=np.int16).reshape(self.alto, self.ancho)
        self.libre = (datos >= 0) & (datos < 50)
        self.bloqueo = ~self.libre                    # ocupado o desconocido: no se ve ni se pasa
        self.m_trans = margen_transito / self.res
        self.m_cand = margen_candidato / self.res
        self.transitable_base = ~dilatar(self.bloqueo, self.m_trans)
        self.candidatable_base = ~dilatar(self.bloqueo, self.m_cand)
        self.transitable = self.transitable_base
        self.candidatable = self.candidatable_base
        self.bloques_conocidos = np.zeros_like(self.libre)
        self.obstaculos_costmap = np.zeros_like(self.libre)
        self.obstaculos_extra = np.zeros_like(self.libre)
        self.total_libre = max(1, int(self.libre.sum()))

    def actualizar_obstaculos(self, ocupado_extra):
        """Suma los obstáculos no mapeados (del costmap) a las máscaras."""
        self.obstaculos_costmap = np.asarray(ocupado_extra, dtype=bool).copy()
        self.obstaculos_extra = self.obstaculos_costmap | self.bloques_conocidos
        self.transitable = self.transitable_base & ~dilatar(self.obstaculos_extra, self.m_trans)
        self.candidatable = self.candidatable_base & ~dilatar(self.obstaculos_extra, self.m_cand)

    def registrar_bloque_objeto(self, x, y, lado_m):
        """Ocupa un cuadrado conocido alrededor del objeto confirmado."""
        if lado_m <= 0:
            raise ValueError('lado_m debe ser positivo')
        radio = lado_m / 2.0
        f0, c0 = self.a_celda(x - radio, y - radio)
        f1, c1 = self.a_celda(x + radio, y + radio)
        f0, c0 = max(0, f0), max(0, c0)
        f1, c1 = min(self.alto - 1, f1), min(self.ancho - 1, c1)
        if f0 > f1 or c0 > c1:
            return
        self.bloques_conocidos[f0:f1 + 1, c0:c1 + 1] |= self.libre[f0:f1 + 1, c0:c1 + 1]
        self.actualizar_obstaculos(self.obstaculos_costmap)

    def a_celda(self, x, y):
        return int(math.floor((y - self.oy) / self.res)), int(math.floor((x - self.ox) / self.res))

    def a_mundo(self, fila, col):
        return self.ox + (col + 0.5) * self.res, self.oy + (fila + 0.5) * self.res

    def dentro(self, fila, col):
        return 0 <= fila < self.alto and 0 <= col < self.ancho


class Exploracion(Node):

    def __init__(self):
        super().__init__('exploracion')
        defaults = {
            # misión
            'cantidad_victimas': 3, 'duracion_corrida_s': 600.0, 'margen_regreso_s': 25.0,
            'publicar_done': True, 'volver_al_inicio': True,
            'espera_reportes_finales_s': 0.7,
            # cámara
            'fov_camara_deg': 70.0, 'rango_vision': 2.0, 'rayos_360': 72,
            'peso_lejos': 0.5,              # valor de una celda vista en el borde del alcance (1.0 a <= 1 m)
            'giro_maximo_visto': 1.0,       # rad/s (odometría): girando más rápido no se marca "visto"
            'confirmar_cajas': True, 'intentos_confirmacion': 1, 'tiempo_mirar_caja_s': 0.8,
            # NBV
            'paso_candidatos': 0.30, 'margen_transito': 0.17, 'margen_candidato': 0.22,
            'lambda_distancia': 0.70, 'ganancia_minima': 25, 'fraccion_agotado': 0.30,
            'peso_giro_nbv': 0.35,
            'cobertura_objetivo': 0.97, 'radio_lista_negra': 0.35, 'costo_obstaculo': 100,
            'modo_exploracion': 'sectores_nbv',
            'cobertura_objetivo_sector': 0.68, 'paso_maximo_portal_m': 0.50,
            'margen_parada_portal_m': 0.22, 'anticipacion_portal_m': 0.30,
            'desvio_maximo_ruta_m': 0.30, 'fraccion_desvio_ruta': 0.15,
            'area_minima_sector_m2': 0.35, 'ancho_minimo_sector_m': 0.45,
            'area_residual_ignorable_m2': 0.09, 'area_caja_m2': 0.09,
            'radio_alejar_guia_m': 0.25, 'holgura_guia_m': 0.30,
            'radio_visto_robot_m': 0.15, 'radio_visto_objeto_m': 0.45,
            'radio_entrada_sector_m': 0.30, 'done_forzado_s': 599.0, 'tolerancia_cerca_destino_m': 0.20,
            'tolerancia_giro_llegada_deg': 35.0, 'tolerancia_mirar_caja_deg': 25.0,
            'tolerancia_centrar_objeto_deg': 25.0,
            'lado_bloque_objeto_m': 0.26, 'distancia_pared_bloque_m': 0.20,
            'celdas_max_cortadas_por_bloque': 20, 'peso_holgura_nbv': 0.6,
            'holgura_paso_estrecho_m': 0.25, 'radio_paso_fallido_m': 0.15,
            'margen_objeto_ruta_m': 0.22, 'desvio_maximo_por_objetos_m': 4.0,
            'ancho_paso_minimo_m': 0.40, 'profundidad_objeto_m': 0.12, 'cerrar_huecos_objetos': False,
            'corregir_objetos': True, 'radio_correccion_objeto_m': 0.35, 'distancia_max_correccion_m': 1.8,
            'lecturas_correccion': 40, 'lecturas_por_evaluacion': 6, 'paso_max_correccion_m': 0.05, 'correccion_total_max_m': 0.10, 'guias_portal_y_medio': False, 'radio_portal_guia_m': 0.35, 'holgura_minima_giro_previo_m': 0.03, 'lecturas_minimas_correccion': 6, 'correccion_minima_objeto_m': 0.03,
            'inspeccionar_objetos': True, 'tiempo_observacion_objeto_s': 0.8,
            'distancia_observacion_min_m': 0.45, 'distancia_observacion_max_m': 1.60,
            'distancia_observacion_ideal_m': 0.85, 'timeout_observacion_objeto_s': 10.0,
            'imagenes_observacion_minimas': 2, 'porcentaje_vel_observacion': 25.0,
            'separacion_pared_objeto_m': 0.16,
            # ejecución
            'periodo_decision': 0.5, 'tolerancia_llegada': 0.30, 'girar_al_llegar': True,
            'giro_inicial': True, 'velocidad_estimada': 0.25,
            # anti-atascos
            # localización y velocidad segura
            'error_loc_malo': 0.08, 'error_reloc_ok': 0.05, 'segundos_loc_mala': 1.5, 'espera_entre_relocalizaciones': 12.0,
            'rango_laser_localizacion_m': 8.0,
            'relocalizacion_std_xy_m': 0.18, 'relocalizacion_std_yaw_rad': 0.25,
            'confirmacion_relocalizacion_s': 2.5, 'timeout_relocalizacion_s': 45.0,
            # reubicación alrededor de donde cree estar (no en todo el mapa) y moviéndose, no girando
            'std_reubicar_xy_m': 0.5, 'std_reubicar_yaw_rad': 0.6, 'factor_std_reintento': 1.6,
            'max_std_reubicar_xy_m': 1.2, 'espera_mover_reloc_s': 2.5, 'distancia_mover_reloc_m': 0.35,
            'movimientos_reloc': 3, 'velocidad_mover_reloc': 0.12,
            'fallos_para_relocalizar': 3, 'ventana_fallos_relocalizacion_s': 90.0,
            'distancia_lenta': 0.25, 'distancia_muy_lenta': 0.12,
            'porcentaje_loc_mala': 100.0, 'cambio_sin_frenar': True, 'giro_previo_deg': 135.0,
            'porcentaje_lenta': 70.0, 'porcentaje_muy_lenta': 50.0,
            'tiempo_atascado': 10.0, 'tiempo_atascado_global': 16.0, 'distancia_atascado': 0.12,
            'distancia_escape': 0.15, 'velocidad_escape': 0.10,
            'timeout_scan_s': 0.5,
            # localización / largada
            'x_inicial': 0.0, 'y_inicial': 0.0, 'yaw_inicial': 0.0, 'std_xy': 0.25, 'std_yaw': 0.30,
            'std_convergido': 0.06, 'max_espera_convergencia_s': 30.0,
            'tolerancia_final': 0.06, 'velocidad_aproximacion': 0.12, 'max_reintentos_regreso': 3,
            # frames / debug
            'base_frame': 'base_footprint', 'odom_frame': 'odom', 'publicar_debug': True,
        }
        defaults.update(PARAMETROS_INFLACION)
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        self.cfg = {k: self.get_parameter(k).value for k in defaults}
        c = self.cfg
        if c['modo_exploracion'] not in ('sectores_nbv', 'nbv'):
            raise ValueError('modo_exploracion debe ser sectores_nbv o nbv')
        if (not 0 < c['cobertura_objetivo_sector'] <= 1 or c['paso_maximo_portal_m'] <= 0
                or c['margen_parada_portal_m'] < c['margen_transito']
                or c['anticipacion_portal_m'] <= 0):
            raise ValueError('Umbral de sector o paso máximo de portal inválido')
        if c['tiempo_observacion_objeto_s'] <= 0 or c['separacion_pared_objeto_m'] < 0:
            raise ValueError('Parámetros de observación de objetos inválidos')
        if (not 0.35 <= c['distancia_observacion_min_m'] <= c['distancia_observacion_ideal_m']
                <= c['distancia_observacion_max_m'] <= c['rango_vision']
                or c['timeout_observacion_objeto_s'] <= c['tiempo_observacion_objeto_s']
                or c['imagenes_observacion_minimas'] < 2
                or not 0 < c['porcentaje_vel_observacion'] <= 100
                or not 0 < c['tolerancia_centrar_objeto_deg'] < c['fov_camara_deg'] / 2):
            raise ValueError('Distancias, velocidad o confirmación de observación inválidas')
        if (c['area_residual_ignorable_m2'] < 0 or c['area_caja_m2'] <= 0
                or c['lado_bloque_objeto_m'] <= 0):
            raise ValueError('Área residual, área de caja o bloque de objeto inválidos')
        if (min(c['rango_laser_localizacion_m'], c['relocalizacion_std_xy_m'],
                c['relocalizacion_std_yaw_rad'], c['confirmacion_relocalizacion_s'],
                c['ventana_fallos_relocalizacion_s']) <= 0
                or c['timeout_relocalizacion_s'] <= c['confirmacion_relocalizacion_s']
                or c['fallos_para_relocalizar'] < 2):
            raise ValueError('Parámetros de recuperación de AMCL inválidos')
        self.fov = math.radians(c['fov_camara_deg'])
        self.nrayos = int(c['rayos_360'])
        self.angulos = np.linspace(-math.pi, math.pi, self.nrayos, endpoint=False)
        self.ventana = max(1, int(round(self.fov / (2 * math.pi / self.nrayos))))

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.nav_ruta = ActionClient(self, NavigateThroughPoses, 'navigate_through_poses')
        self.bt_portales = buscar_arbol_portales()
        if self.bt_portales:
            self.get_logger().info(f'Árbol de Nav2 para rutas con portales: {self.bt_portales}')
        else:
            self.get_logger().error(
                'No encuentro behavior_trees/navegar_portales.xml en el paquete instalado: las rutas '
                'por portales usan el árbol por defecto de Nav2. Revisá que no haya otro equipo_jar '
                'viejo instalado en ~/rosmaster_ws/install (ros2 pkg prefix equipo_jar).')
        self.spin_cli = ActionClient(self, Spin, 'spin')
        self.nomotion_cli = self.create_client(EmptySrv, '/request_nomotion_update')
        self.reportes_finales_cli = self.create_client(Trigger, '/deteccion/reportar_finales')
        self.clear_cli = [
            self.create_client(ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap'),
            self.create_client(ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')]

        self.create_subscription(OccupancyGrid, '/map', self.cb_mapa, QOS_LATCHED)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', self.cb_costmap, QOS_LATCHED)
        self.create_subscription(PoseWithCovarianceStamped, '/amcl_pose', self.cb_amcl, QOS_RELIABLE)
        self.create_subscription(PointStamped, '/victimas', self.cb_victima, QOS_RELIABLE)
        # Topic interno del equipo: cajas rojas vistas pero sin reportar todavía.
        self.create_subscription(PoseArray, '/deteccion/pendientes', self.cb_pendientes, 10)
        self.create_subscription(String, '/deteccion/observaciones', self.cb_observaciones, 10)
        # /scan es best effort (§7.1): hay que suscribirse con qos_profile_sensor_data.
        self.create_subscription(LaserScan, '/scan', self.cb_scan, qos_profile_sensor_data)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.cb_imagen,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.cb_info_camara,
                                 qos_profile_sensor_data)
        self.pub_init = self.create_publisher(PoseWithCovarianceStamped, '/initialpose', 10)
        self.pub_mapa_amcl = self.create_publisher(OccupancyGrid, '/mapa_amcl', QOS_LATCHED)
        self.pub_mapa_objetos = self.create_publisher(OccupancyGrid, '/exploracion/mapa_objetos', QOS_LATCHED)
        self.pub_estado_loc = self.create_publisher(PointStamped, '/exploracion/estado_loc', 10)
        self.pub_done = self.create_publisher(Empty, '/done', QOS_RELIABLE)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', QOS_RELIABLE)
        self.pub_vel = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        self.pub_visto = self.create_publisher(OccupancyGrid, '/exploracion/visto', QOS_LATCHED)
        if c['publicar_debug']:
            self.pub_marc = self.create_publisher(MarkerArray, '/exploracion/marcadores', 10)
            self.pub_sectores = self.create_publisher(MarkerArray, '/exploracion/sectores', QOS_LATCHED)

        self.grilla = None
        self.mapa_msg = None
        self.visto = None
        self.sectorizador = None
        self.ruta_sector = []
        self.ruta_regreso = []
        self.seguro_ruta_sector = None
        self.seguro_ruta_regreso = None
        self.ruta_region = None
        self.portales_fallidos = set()
        self.sectores_directo = set()      # sectores a los que ya se intentó entrar directo con Nav2
        self.pendientes_vision = []        # cajas vistas sin reportar (del detector): (x, y, n)
        self.t_pendientes = None
        self.intentos_confirmar = []       # [x, y, intentos]
        self.espera_mirar_hasta = 0.0
        self.giro_pendiente = None         # (ángulo, siguiente) a girar después de separarse de la pared
        self.vel_giro = 0.0                # rad/s, estimada con la pose de AMCL
        self.yaw_prev = None
        self.t_estado_log = 0.0
        self.objetos_fuera_mapa = ObjetosFueraMapa()
        self.bloques_objetos = []
        self.lecturas_bloques = {}      # índice de bloque -> lecturas recientes del LiDAR (x, y)
        self.objeto_observado = None
        self.observaciones_vision = None
        self.ultimo_stamp_vision = 0
        self.t_scan_objetos = -float('inf')
        self.scans_objetos = deque(maxlen=15)
        self.ultimo_stamp_scan_objetos = 0
        self.frame_camara = None
        self.t_ultima_imagen = None
        self.ultimo_stamp_imagen_camara = 0
        self.estado = ESPERANDO_MAPA
        self.t_mapa = None                 # reloj de pared (el juez cronometra en reloj de pared)
        self.t_ultimo_init = 0.0
        self.goal_id = 0
        self.goal_handle = None
        self.solicitud_goal = None
        self.cancelacion_en_curso = None
        # Goal de Nav2 que se deja andar mientras se elige el próximo destino (al salir de
        # una zona): el próximo goal lo reemplaza sin frenar (preempción de Nav2).
        self.goal_heredado = None
        self.vel_ventana = []          # (t, metros) para medir la velocidad real reciente
        self.goal_actual = None
        self.cancelacion_cobertura = None
        self.lista_negra = []
        self.ultimo_debug = 0.0
        self.t_ultimo_huecos = 0.0
        self.hist = deque()
        self.hist_global = deque()
        self.n_atascos = 0
        self.reportes = 0
        self.motivo_regreso = None
        self.reintentos_regreso = 0
        self.regreso_directo = False        # tras fallar por portales: destino directo a Nav2
        self.t_aprox = None
        self.scan = None
        self.escape = None
        self.dist_pared = None          # m de cada celda a la pared más cercana del mapa
        self.error_loc = None           # mediana (m) de la distancia scan->pared (EMA)
        self.loc_mala_desde = None
        self.muestras_loc = []
        self.t_resumen_loc = time.monotonic()
        self.t_ult_reloc = -1e9
        self.pose_inicial_estimada = None
        self.lc = {}                       # clientes de ciclo de vida (get_state, change_state)
        self.estados_lc = {}
        self.consultas_lc = {}
        self.transiciones_lc = {}
        self.reanudar_nav2 = ELIGIENDO
        self.giro_reanudar_nav2 = None
        self.t_ult_revision_lc = 0.0
        self.reloc = None
        self.fallos_localizacion = deque()
        self.secuencia_amcl = 0
        self.t_ultima_pose_amcl = None
        self.pendiente = None           # goal guardado mientras el robot gira para encarar la ruta
        self.dir_ruta = None            # rumbo (marco map) del primer tramo de la ruta actual
        self.ultimo_limite = None
        # largada
        self.p0_odom = None                # pose de odom al recibir el mapa (robot quieto)
        self.largada = (c['x_inicial'], c['y_inicial'], c['yaw_inicial'])
        self.largada_fija = False
        self.t_localizado = None
        self.std_amcl = (1.0, 1.0)
        self.dist_casa = None              # campo de distancias a la largada (celdas)
        self.t_dist_casa = 0.0
        # velocidad real medida (para estimar el regreso)
        self.recorrido = 0.0
        self.t_moviendo = 0.0
        self.ultima_pose = None
        self.ultimo_t_pose = None

        self.inflacion = InflacionAdaptativa(self, c)
        self.create_timer(0.5, self.actualizar_inflacion)
        self.create_timer(c['periodo_decision'], self.ciclo)
        self.create_timer(0.2, self.tick_rapido)
        self.get_logger().info(
            f"Explorador listo (N={c['cantidad_victimas']}). Esperando /map del juez...")

    # =================================================================== callbacks
    def cb_mapa(self, msg):
        nuevo = self.grilla is None or (msg.info.width, msg.info.height) != (self.grilla.ancho, self.grilla.alto)
        self.mapa_msg = msg
        self.grilla = Grilla(msg, self.cfg['margen_transito'], self.cfg['margen_candidato'])
        g = self.grilla
        for x, y in self.bloques_objetos:
            g.registrar_bloque_objeto(x, y, self.cfg['lado_bloque_objeto_m'])
        pasos = int(self.cfg['rango_vision'] / g.res)
        dist = np.arange(1, pasos + 1) * g.res
        self.ray_dx = np.cos(self.angulos)[:, None] * dist[None, :] / g.res
        self.ray_dy = np.sin(self.angulos)[:, None] * dist[None, :] / g.res
        # Una celda vista de cerca vale más que una vista en el borde del alcance:
        # de lejos el detector confirma peor (y la ubica con más error).
        lejos = max(1e-6, self.cfg['rango_vision'] - 1.0)
        self.peso_rayo = np.clip(1.0 - (1.0 - self.cfg['peso_lejos']) * (dist - 1.0) / lejos,
                                 self.cfg['peso_lejos'], 1.0)
        if nuevo or self.visto is None:
            self.visto = np.zeros((g.alto, g.ancho), dtype=bool)
            self.sectorizador = None
            self.ruta_sector = []
            self.ruta_regreso = []
            self.seguro_ruta_sector = None
            self.seguro_ruta_regreso = None
            self.ruta_region = None
            self.portales_fallidos.clear()
            # La memoria usa coordenadas de map: una nueva grilla no convierte
            # objetos conocidos en nuevos ni interrumpe la inspección en curso.
        if self.sectorizador is not None:
            self.sectorizador.actualizar_bloques(g.bloques_conocidos)
        self.dist_casa = None
        self.dist_pared = self.campo_distancia_paredes(g)
        self.publicar_mapa_amcl()
        self.publicar_mapa_objetos()
        if self.estado == ESPERANDO_MAPA:
            self.t_mapa = time.monotonic()
            self.p0_odom = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
            self.estado = LOCALIZANDO
            self.get_logger().info(f'/map recibido ({g.ancho}x{g.alto} @ {g.res:.2f} m). ¡Largada!')

    def cb_costmap(self, msg):
        g = self.grilla
        if (g is None or abs(msg.info.resolution - g.res) > 1e-4
                or self.estado == RELOCALIZANDO or self.loc_mala_desde is not None):
            return
        datos = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        ocup = datos >= int(self.cfg['costo_obstaculo'])
        extra = np.zeros((g.alto, g.ancho), dtype=bool)
        # Alinear las dos grillas por su origen (misma resolución).
        df = int(round((msg.info.origin.position.y - g.oy) / g.res))
        dc = int(round((msg.info.origin.position.x - g.ox) / g.res))
        f0, c0 = max(0, df), max(0, dc)
        f1, c1 = min(g.alto, df + msg.info.height), min(g.ancho, dc + msg.info.width)
        if f1 > f0 and c1 > c0:
            extra[f0:f1, c0:c1] = ocup[f0 - df:f1 - df, c0 - dc:c1 - dc]
        # Lo que ya es pared en el mapa no hace falta; queda lo NO mapeado.
        extra &= g.libre
        g.actualizar_obstaculos(extra)
        # El costmap contiene ecos acumulados y nuestro propio mapa de objetos.
        # Sirve para evitar choques, pero no crea recuerdos ni candidatos nuevos:
        # éstos llegan del scan con TF de su instante o de la cámara sincronizada.

    def alcanzable_desde_robot(self, transitable):
        """Celdas transitables conectadas con el robot (o con la largada si no hay pose)."""
        g = self.grilla
        try:
            pose = self.pose_robot()
        except Exception:  # noqa: BLE001  sin TF todavía
            pose = None
        x, y = pose[:2] if pose is not None else getattr(self, 'largada', (0.0, 0.0))[:2]
        f, c = g.a_celda(x, y)
        fs, cs = np.nonzero(transitable)
        if len(fs) == 0:
            return np.zeros_like(transitable)
        if not (g.dentro(f, c) and transitable[f, c]):
            k = int(np.argmin((fs - f) ** 2 + (cs - c) ** 2))
            f, c = int(fs[k]), int(cs[k])
        return bfs(transitable, f, c) >= 0

    def registrar_objeto_local(self, punto):
        """Obstáculo persistente del equipo, independiente de futuros ecos o imágenes.

        Nunca deja que un bloque cierre un paso: si con el bloque quedan celdas libres
        separadas del robot (además de las que ocupa), no se registra. 
        """
        if any(math.dist(punto, previo) < 0.45 for previo in self.bloques_objetos):
            return
        g = self.grilla
        antes_bloques = g.bloques_conocidos.copy()
        antes = self.alcanzable_desde_robot(g.transitable)
        g.registrar_bloque_objeto(*punto, self.cfg['lado_bloque_objeto_m'])
        despues = self.alcanzable_desde_robot(g.transitable)
        cortadas = int((antes & g.transitable & ~despues).sum())
        if cortadas > int(self.cfg.get('celdas_max_cortadas_por_bloque', 20)):
            g.bloques_conocidos = antes_bloques
            g.actualizar_obstaculos(g.obstaculos_costmap)
            self.get_logger().warn(
                f'Objeto en ({punto[0]:.2f}, {punto[1]:.2f}): no lo agrego como bloque, dejaría '
                f'{cortadas * g.res * g.res:.2f} m² sin camino. Nav2 lo esquiva igual si es real')
            return
        self.bloques_objetos.append(tuple(punto))
        if self.cfg.get('cerrar_huecos_objetos', False):
            self.cerrar_huecos_objeto(punto, despues)
        if self.sectorizador is not None:
            self.sectorizador.actualizar_bloques(self.grilla.bloques_conocidos)
        self.dist_casa = None
        self.publicar_mapa_objetos()
        self.get_logger().info(
            f'Bloque conocido para objeto fuera del mapa en ({punto[0]:.2f}, {punto[1]:.2f})')

    def corregir_bloque_con_lectura(self, centro, sensor):
        """Si la lectura del LiDAR cae sobre un objeto ya guardado, se usa para corregir su
        posición en el mapa local (la primera estimación tiene unos cm de error: se ve una
        sola cara y desde un solo lado). Devuelve True si la lectura era de un objeto
        guardado. Sólo lecturas cercanas (más precisas) y con buena localización."""
        bloques = getattr(self, 'bloques_objetos', [])
        if not bloques or not self.cfg.get('corregir_objetos', True):
            return False
        radio = self.cfg.get('radio_correccion_objeto_m', 0.35)
        i = min(range(len(bloques)), key=lambda k: math.dist(centro, bloques[k]))
        if math.dist(centro, bloques[i]) > radio:
            return False
        if math.dist(centro, sensor) > self.cfg.get('distancia_max_correccion_m', 1.8):
            return True
        # Un bloque fijo no puede moverse bajo una ruta que Nav2 está ejecutando.
        # Se corrige entre rutas o durante la inspección, antes de volver a planificar.
        estado = getattr(self, 'estado', None)
        if (estado is not None and estado not in (ELIGIENDO, OBSERVANDO_OBJETO)) or self.heredado_vivo():
            return True
        lecturas = self.lecturas_bloques.setdefault(i, deque(maxlen=int(self.cfg.get('lecturas_correccion', 60))))
        lecturas.append(tuple(centro))
        if len(lecturas) < int(self.cfg.get('lecturas_minimas_correccion', 6)):
            return True
        # Se evalúa cada `lecturas_por_evaluacion` lecturas nuevas, y el objeto sólo se mueve
        # si dos evaluaciones seguidas coinciden (posición estable) y a lo sumo
        # `paso_max_correccion_m` por vez. Antes saltaba 10-19 cm entre lecturas tomadas
        # desde lados distintos y el hueco junto a la caja se abría y cerraba en el plan.
        cuenta = getattr(self, 'cuenta_correcciones', None)
        if cuenta is None:
            cuenta = self.cuenta_correcciones = {}
        cuenta[i] = cuenta.get(i, 0) + 1
        if cuenta[i] % max(1, int(self.cfg.get('lecturas_por_evaluacion', 8))):
            return True
        xs = sorted(p[0] for p in lecturas); ys = sorted(p[1] for p in lecturas)
        nuevo = (xs[len(xs) // 2], ys[len(ys) // 2])
        previas = getattr(self, 'evaluaciones_bloques', None)
        if previas is None:
            previas = self.evaluaciones_bloques = {}
        anterior, previas[i] = previas.get(i), nuevo
        minimo = self.cfg.get('correccion_minima_objeto_m', 0.03)
        if anterior is None or math.dist(anterior, nuevo) > minimo:
            return True                                   # todavía no es estable
        delta = math.dist(nuevo, bloques[i])
        if delta >= minimo:
            paso = self.cfg.get('paso_max_correccion_m', 0.05)
            if delta > paso:
                nuevo = (bloques[i][0] + (nuevo[0] - bloques[i][0]) * paso / delta,
                         bloques[i][1] + (nuevo[1] - bloques[i][1]) * paso / delta)
            # Tope a la deriva acumulada: con el robot trabado contra la caja, pasos de
            # 5 cm seguidos la corrieron 25 cm hacia él (corrida 18:29) y cerraron el paso.
            origenes = getattr(self, 'origen_bloques', None)
            if origenes is None:
                origenes = self.origen_bloques = {}
            origen = origenes.setdefault(i, tuple(bloques[i]))
            tope = self.cfg.get('correccion_total_max_m', 0.10)
            lejos = math.dist(origen, nuevo)
            if lejos > tope:
                nuevo = (origen[0] + (nuevo[0] - origen[0]) * tope / lejos,
                         origen[1] + (nuevo[1] - origen[1]) * tope / lejos)
            if math.dist(nuevo, bloques[i]) < 0.01:
                return True
            self.mover_bloque(i, nuevo, math.dist(nuevo, bloques[i]))
        return True

    def mover_bloque(self, i, nuevo, delta):
        """Rehace los bloques del mapa local con el objeto `i` en su nueva posición,
        sin cerrar pasos."""
        g = self.grilla
        anterior = self.bloques_objetos[i]
        antes_bloques = g.bloques_conocidos.copy()
        alcanzable = self.alcanzable_desde_robot(g.transitable)
        self.bloques_objetos[i] = (float(nuevo[0]), float(nuevo[1]))
        g.bloques_conocidos = np.zeros_like(g.bloques_conocidos)
        for x, y in self.bloques_objetos:
            g.registrar_bloque_objeto(x, y, self.cfg['lado_bloque_objeto_m'])
        despues = self.alcanzable_desde_robot(g.transitable)
        if int((alcanzable & g.transitable & ~despues).sum()) > int(self.cfg.get('celdas_max_cortadas_por_bloque', 20)):
            self.bloques_objetos[i] = anterior
            g.bloques_conocidos = antes_bloques
            g.actualizar_obstaculos(g.obstaculos_costmap)
            return
        if self.sectorizador is not None:
            self.sectorizador.actualizar_bloques(g.bloques_conocidos)
        self.dist_casa = None
        self.publicar_mapa_objetos()
        self.get_logger().info(
            f'Corrijo el objeto del mapa local: ({anterior[0]:.2f}, {anterior[1]:.2f}) -> '
            f'({nuevo[0]:.2f}, {nuevo[1]:.2f}) ({100 * delta:.0f} cm, '
            f'{len(self.lecturas_bloques.get(i, ()))} lecturas del LiDAR)')

    def huecos_angostos_objeto(self, punto):
        """Huecos libres entre el objeto y una pared del mapa por donde el robot pasa
        con dificultad: d(objeto) + d(pared) < ancho_paso_minimo_m. Devuelve una lista
        de (ancho_minimo_m, máscara), una por hueco (componente conexa)."""
        g = self.grilla
        ancho = self.cfg.get('ancho_paso_minimo_m', 0.40)
        medio = self.cfg['lado_bloque_objeto_m'] / 2.0
        k = int(math.ceil((medio + ancho) / g.res)) + 1
        f, c = g.a_celda(*punto)
        f0, f1 = max(0, f - k), min(g.alto, f + k + 1)
        c0, c1 = max(0, c - k), min(g.ancho, c + k + 1)
        if f0 >= f1 or c0 >= c1:
            return []
        filas, cols = np.mgrid[f0:f1, c0:c1]
        xs = g.ox + (cols + 0.5) * g.res
        ys = g.oy + (filas + 0.5) * g.res
        d_obj = np.hypot(np.maximum(np.abs(xs - punto[0]) - medio, 0.0),
                         np.maximum(np.abs(ys - punto[1]) - medio, 0.0))
        paredes = getattr(self, 'dist_pared', None)
        if paredes is None:
            paredes = self.dist_pared = self.campo_distancia_paredes(g)
        suma = d_obj + paredes[f0:f1, c0:c1]
        local = (g.libre[f0:f1, c0:c1] & ~g.bloques_conocidos[f0:f1, c0:c1]
                 & (d_obj > 0) & (suma < ancho))
        # Sólo las celdas que están ENTRE la caja y la pared (la caja y la pared más
        # cercana quedan para lados opuestos); si no, el anillo alrededor de la caja
        # unía todos los huecos en uno solo y no se podía cerrar uno y dejar otro.
        pf, pc = np.nonzero(~g.libre[f0:f1, c0:c1])
        if pf.size and local.any():
            lf, lc = np.nonzero(local)
            px, py = xs[pf, pc], ys[pf, pc]
            cx, cy = xs[lf, lc], ys[lf, lc]
            cerca = np.argmin((cx[:, None] - px[None, :]) ** 2 + (cy[:, None] - py[None, :]) ** 2, axis=1)
            wx, wy = px[cerca] - cx, py[cerca] - cy
            bx = np.clip(cx, punto[0] - medio, punto[0] + medio) - cx
            by = np.clip(cy, punto[1] - medio, punto[1] + medio) - cy
            coseno = (wx * bx + wy * by) / np.maximum(np.hypot(wx, wy) * np.hypot(bx, by), 1e-9)
            local[lf[coseno > -0.5], lc[coseno > -0.5]] = False
        elif not pf.size:
            local[:] = False
        huecos, pendientes = [], set(zip(*np.nonzero(local)))
        while pendientes:
            inicio = pendientes.pop()
            grupo, cola = [inicio], [inicio]
            while cola:
                a, b = cola.pop()
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        v = (a + da, b + db)
                        if v in pendientes:
                            pendientes.remove(v)
                            grupo.append(v)
                            cola.append(v)
            mascara = np.zeros_like(g.libre)
            fs = np.array([q[0] for q in grupo]); cs = np.array([q[1] for q in grupo])
            mascara[fs + f0, cs + c0] = True
            huecos.append((float(suma[fs, cs].min()), mascara))
        return sorted(huecos, key=lambda h: h[0])

    def cerrar_huecos_objeto(self, punto, alcanzable_con_caja=None):
        """En el mapa propio cierra los huecos angostos entre el objeto y las paredes,
        del más angosto al más ancho, mientras quede otro camino: si la caja está en
        un cuello de botella, Nav2 pasa por el hueco más ancho en vez de meterse por
        la rendija entre la caja y la punta de una pared (~30 cm).
        """
        g = self.grilla
        huecos = self.huecos_angostos_objeto(punto)
        if not huecos:
            return
        referencia = (alcanzable_con_caja if alcanzable_con_caja is not None
                      else self.alcanzable_desde_robot(g.transitable))
        maximo = int(self.cfg.get('celdas_max_cortadas_por_bloque', 20))
        cerrados, abiertos = [], []
        for ancho_hueco, hueco in huecos:
            antes_bloques = g.bloques_conocidos.copy()
            g.bloques_conocidos = g.bloques_conocidos | hueco
            g.actualizar_obstaculos(g.obstaculos_costmap)
            despues = self.alcanzable_desde_robot(g.transitable)
            if int((referencia & g.transitable & ~despues).sum()) > maximo:
                g.bloques_conocidos = antes_bloques      # es el paso que queda: abierto
                g.actualizar_obstaculos(g.obstaculos_costmap)
                abiertos.append(ancho_hueco)
            else:
                cerrados.append(ancho_hueco)
        if cerrados or abiertos:
            self.get_logger().info(
                f'Objeto en ({punto[0]:.2f}, {punto[1]:.2f}): cierro {len(cerrados)} hueco(s) angosto(s) '
                f'contra la pared ({", ".join(f"{100 * a:.0f} cm" for a in cerrados) or "-"}); '
                + (f'dejo abierto el de {", ".join(f"{100 * a:.0f} cm" for a in abiertos)} (sin él no hay camino)'
                   if abiertos else 'no queda ninguno abierto'))

    def publicar_mapa_objetos(self):
        """Mapa propio para el plan global de Nav2; nunca modifica /map del juez."""
        if self.mapa_msg is None:
            return
        m = OccupancyGrid()
        m.header.frame_id = self.grilla.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info = self.mapa_msg.info
        datos = np.array(self.mapa_msg.data, dtype=np.int8).reshape(self.grilla.alto, self.grilla.ancho)
        datos[self.grilla.bloques_conocidos] = 100
        m.data = datos.ravel().tolist()
        self.pub_mapa_objetos.publish(m)

    def cb_imagen(self, msg):
        if msg.header.frame_id and msg.width > 0 and msg.height > 0 and msg.data:
            stamp = Time.from_msg(msg.header.stamp).nanoseconds
            if stamp <= self.ultimo_stamp_imagen_camara:
                return
            self.ultimo_stamp_imagen_camara = stamp
            self.frame_camara = msg.header.frame_id
            self.t_ultima_imagen = time.monotonic()

    def cb_info_camara(self, msg):
        if msg.header.frame_id and msg.width > 0 and msg.k[0] > 0:
            self.frame_camara = msg.header.frame_id
            self.fov = 2 * math.atan(msg.width / (2 * msg.k[0]))
            self.ventana = max(1, int(round(self.fov / (2 * math.pi / self.nrayos))))

    def cb_amcl(self, msg):
        cov = msg.pose.covariance
        self.std_amcl = (math.sqrt(max(cov[0], cov[7], 0.0)), math.sqrt(max(cov[35], 0.0)))
        self.secuencia_amcl += 1
        self.t_ultima_pose_amcl = time.monotonic()

    def cb_pendientes(self, msg):
        self.pendientes_vision = [(p.position.x, p.position.y, p.position.z) for p in msg.poses]
        self.t_pendientes = time.monotonic()

    def cb_observaciones(self, msg):
        if self.grilla is None:
            return
        observaciones = leer_observaciones(msg.data, self.grilla.frame)
        if observaciones is None or observaciones['stamp_ns'] <= self.ultimo_stamp_vision:
            return
        edad = (self.get_clock().now().nanoseconds - observaciones['stamp_ns']) / 1e9
        if not 0.0 <= edad <= 0.75:
            return
        self.ultimo_stamp_vision = observaciones['stamp_ns']
        self.observaciones_vision = dict(observaciones, recibido=time.monotonic())
        self.objetos_fuera_mapa.actualizar_vision(observaciones['objetos'], time.monotonic())

    def cb_victima(self, _msg):
        self.reportes += 1
        n = int(self.cfg['cantidad_victimas'])
        self.get_logger().info(f'Reporte en /victimas: {self.reportes}/{n}')
        if self.reportes >= n and self.estado in (ELIGIENDO, NAVEGANDO, COMPORTAMIENTO):
            self.pedir_regreso(f'{n} víctimas reportadas')

    @staticmethod
    def campo_distancia_paredes(g, ocup=None):
        """Distancia (m) de cada celda a la celda ocupada más cercana del mapa (BFS 8-vecinos).
        `ocup`: máscara de ocupado a usar (por defecto, lo no libre del mapa)."""
        ocup = ~g.libre if ocup is None else ocup
        dist = np.full(ocup.shape, np.inf)
        dist[ocup] = 0.0
        frente = list(zip(*np.nonzero(ocup)))
        alto, ancho = ocup.shape
        cola = deque(frente)
        while cola:
            f, c = cola.popleft()
            d0 = dist[f, c]
            for df, dc, w in ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                              (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)):
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho and d0 + w < dist[nf, nc] - 1e-6:
                    dist[nf, nc] = d0 + w
                    cola.append((nf, nc))
        return dist * g.res

    # ======================================================= calidad de localización
    def mascara_objetos(self, radio=0.10):
        """Celdas ocupadas por los objetos fuera del mapa ya confirmados (disco de
        `radio` alrededor de cada uno: una caja de 20 cm)."""
        g = self.grilla
        m = np.zeros((g.alto, g.ancho), dtype=bool)
        k = max(1, int(round(radio / g.res)))
        for x, y in getattr(self, 'bloques_objetos', []):
            f, c = g.a_celda(x, y)
            f0, f1 = max(0, f - k), min(g.alto, f + k + 1)
            c0, c1 = max(0, c - k), min(g.ancho, c + k + 1)
            if f0 < f1 and c0 < c1:
                ff, cc = np.ogrid[f0:f1, c0:c1]
                m[f0:f1, c0:c1] |= (ff - f) ** 2 + (cc - c) ** 2 <= k * k
        return m

    def campo_loc(self):
        """Campo de distancias para medir/corregir la localización: paredes del mapa
        MÁS los objetos fuera del mapa ya observados. """
        objetos = tuple(getattr(self, 'bloques_objetos', []))
        cache = getattr(self, '_campo_loc', None)
        if not objetos:
            return self.dist_pared
        if cache is None or cache[0] != (id(self.grilla), objetos):
            ocup = ~self.grilla.libre | self.mascara_objetos()
            self._campo_loc = cache = ((id(self.grilla), objetos),
                                       self.campo_distancia_paredes(self.grilla, ocup))
        return cache[1]

    def publicar_mapa_amcl(self, con_objetos=True):
        """AMCL usa /mapa_amcl: el /map del juez con los objetos fuera del mapa ya
        observados agregados como ocupados, para que el filtro de partículas los
        tenga en cuenta al reubicarse."""
        if self.mapa_msg is None or not hasattr(self, 'pub_mapa_amcl'):
            return
        m = OccupancyGrid()
        m.header = self.mapa_msg.header
        m.info = self.mapa_msg.info
        datos = np.array(self.mapa_msg.data, dtype=np.int8).reshape(self.grilla.alto, self.grilla.ancho)
        n = 0
        if con_objetos and getattr(self, 'bloques_objetos', None):
            objetos = self.mascara_objetos() & self.grilla.libre
            n = len(self.bloques_objetos)
            datos = datos.copy()
            datos[objetos] = 100
        m.data = datos.ravel().tolist()
        self.pub_mapa_amcl.publish(m)
        if n:
            self.get_logger().info(f'Mapa de AMCL actualizado con {n} objeto(s) fuera del mapa')

    def medir_localizacion(self, pose):
        """Percentil 70 de la distancia de los puntos del láser a la pared más cercana del mapa.

        Bien localizado: ~1-3 cm. Si AMCL se corrió, los puntos de las paredes
        perpendiculares al corrimiento caen al costado y el p70 sube (la mediana
        no alcanzaba: con un corrimiento en una sola dirección, las paredes
        paralelas siguen calzando). El 30% más alto se descarta: son rayos que
        pegan en cosas que no están en el mapa (señuelos, víctimas).

        La distancia es CONTINUA (interpolación bilineal del campo entre centros
        de celda, menos media celda): con la distancia "por celda" el valor salía
        en escalones de 5 cm (0, 5, 10) y en el robot real (bag de la semana 10)
        un robot bien localizado ya marcaba 5 cm, sin margen para distinguir.
        """
        puntos = self.puntos_scan(alcance=self.cfg.get('rango_laser_localizacion_m', 8.0))
        if puntos is None or self.dist_pared is None:
            return None
        x, y = puntos
        if x.size < 30:
            return None
        paso = max(1, math.ceil(x.size / 120))
        x, y = x[::paso], y[::paso]
        c, s = math.cos(pose[2]), math.sin(pose[2])
        mx, my = pose[0] + c * x - s * y, pose[1] + s * x + c * y
        g = self.grilla
        u = (mx - g.ox) / g.res - 0.5
        v = (my - g.oy) / g.res - 0.5
        # Los rayos fuera del mapa también cuentan: descartarlos puede hacer
        # parecer buena una pose falsa que sólo alinea unas pocas paredes.
        uc, vc = np.clip(u, 0, g.ancho - 1), np.clip(v, 0, g.alto - 1)
        col, f = np.floor(uc).astype(int), np.floor(vc).astype(int)
        col1, f1 = np.minimum(col + 1, g.ancho - 1), np.minimum(f + 1, g.alto - 1)
        a, b = uc - col, vc - f
        D = self.campo_loc()
        d = ((1 - a) * (1 - b) * D[f, col] + a * (1 - b) * D[f, col1]
             + (1 - a) * b * D[f1, col] + a * b * D[f1, col1])
        d += g.res * np.hypot(u - uc, v - vc)
        # Una pared de borde ocupa media celda más allá de su centro: no
        # penalizar el rayo que cae en su cara exterior por el redondeo del mapa.
        d = np.clip(d - 0.5 * g.res, 0.0, 1.0)
        return float(np.percentile(d, 70))

    def actualizar_localizacion(self, pose):
        med = self.medir_localizacion(pose)
        if med is None:
            return
        # Resumen cada 15 s (para calibrar error_loc_malo en cada entorno)
        self.muestras_loc.append(med)
        if time.monotonic() - self.t_resumen_loc > 15.0 and len(self.muestras_loc) >= 10:
            m = np.array(self.muestras_loc)
            self.get_logger().info(f'[loc] láser vs mapa (p70) últimos 15 s: mediana {100 * np.median(m):.1f} cm, '
                                   f'máx {100 * m.max():.1f} cm')
            self.muestras_loc = []
            self.t_resumen_loc = time.monotonic()
        self.error_loc = med if self.error_loc is None else 0.7 * self.error_loc + 0.3 * med
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.error_loc > self.cfg['error_loc_malo']:
            if self.loc_mala_desde is None:
                self.loc_mala_desde = t
                self.get_logger().warn(
                    f'Localización empeorando: el láser queda a {100 * self.error_loc:.1f} cm de las paredes '
                    f'del mapa (bien: < {100 * self.cfg["error_loc_malo"]:.0f} cm). Si sigue así '
                    f'{self.cfg["segundos_loc_mala"]:.1f} s, relocalizo.')
        else:
            if self.loc_mala_desde is not None and t - self.loc_mala_desde > 0.5:
                self.get_logger().info(f'Localización recuperada ({100 * self.error_loc:.1f} cm)')
            self.loc_mala_desde = None

    def publicar_estado_loc(self):
        """Para el detector: x = error láser-mapa (m, -1 sin medir), y = 1 relocalizando."""
        if not hasattr(self, 'pub_estado_loc'):
            return
        m = PointStamped()
        m.header.frame_id = 'map'
        m.header.stamp = self.get_clock().now().to_msg()
        m.point.x = float(self.error_loc) if self.error_loc is not None else -1.0
        m.point.y = 1.0 if self.estado == RELOCALIZANDO or self.loc_mala_desde is not None else 0.0
        self.pub_estado_loc.publish(m)

    def localizacion_mala(self):
        if self.loc_mala_desde is None:
            return False
        t = self.get_clock().now().nanoseconds * 1e-9
        return t - self.loc_mala_desde >= self.cfg['segundos_loc_mala']

    def relocalizar(self, siguiente, motivo=None):
        """Detiene la misión y corrige AMCL: primero calce local láser-mapa; si no
        alcanza, partículas alrededor de donde creía estar (pose al perderse + odometría),
        cada vez más abiertas, moviéndose de a tramos cortos para que AMCL converja."""
        self.t_ult_reloc = time.monotonic()
        self.loc_mala_desde = None
        motivo = motivo or f'láser/mapa fuera de tolerancia ({100 * (self.error_loc or 0):.1f} cm)'
        self.get_logger().warn(f'Localización perdida: {motivo}; freno y relocalizo (primero calce local; '
                               'si no alcanza, partículas alrededor de donde creo estar)')
        pose_creida = self.pose_robot()
        odom_creida = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
        self.publicar_mapa_amcl()           # AMCL se reubica teniendo en cuenta los objetos
        cancelacion = self.cancelar_actual()
        self.pub_cmd.publish(Twist())
        self.goal_actual = None
        self.pendiente = None
        self.escape = None
        if self.objeto_observado is not None:
            self.recordar_inspeccion(self.objeto_observado, False)
            self.limite_observacion(100.0)
        self.objeto_observado = None
        self.ruta_sector.clear()
        self.ruta_regreso.clear()
        self.ruta_region = None
        self.dir_ruta = None
        self.estado = RELOCALIZANDO
        self.reloc = dict(t0=time.monotonic(), siguiente=siguiente, fase='frenando',
                          cancelacion=cancelacion, solicitud=None, ultimo_aviso=0.0,
                          ultimo_nomotion=0.0, nomotion=None, estable_desde=None,
                          ultimo_scan=None, giro_fin=False, intentos=0,
                          pose_creida=pose_creida, odom_creida=odom_creida, mov='esperando',
                          t_mov=time.monotonic(), movs=0, ang_mov=None)
        self.hist.clear()
        self.hist_global.clear()

    def relocalizar_local(self):
        """Busca cerca (±35 cm/±17°; si no, ±70 cm/±41°) la pose donde el láser calza
        con el mapa y se la da a AMCL. True si encontró una pose clara."""
        pose = self.pose_robot()
        if pose is None:
            return False
        res = self.buscar_pose(pose, rango_xy=0.35, rango_yaw=0.30)
        if res is None or res[3] >= self.cfg['error_reloc_ok']:
            amplia = self.buscar_pose(pose, rango_xy=0.70, rango_yaw=0.72, paso_xy=0.05, paso_yaw=0.06)
            if amplia is not None and (res is None or amplia[3] < res[3]):
                res = amplia
        if res is None or res[3] >= self.cfg['error_reloc_ok'] or res[3] >= res[4] - 0.01:
            self.get_logger().warn('Relocalización local: no encontré una pose clara cerca')
            return False
        x, y, yaw, err, err0 = res
        self.publicar_pose(x, y, yaw, 0.06, 0.06)
        self.get_logger().info(
            f'Relocalización local: corrijo AMCL {100 * math.hypot(x - pose[0], y - pose[1]):.0f} cm y '
            f'{math.degrees(abs(ang_norm(yaw - pose[2]))):.0f}° (láser {100 * err0:.1f} -> {100 * err:.1f} cm)')
        return True

    def registrar_fallo_localizacion(self, siguiente):
        """El calce del láser puede dar falsa confianza en pasillos repetidos.
        Solo se relocaliza por fallos de navegación si además el láser ya empezó
        a despegarse del mapa: en un paso angosto Nav2 falla seguido aunque la
        localización esté bien, y frenar a relocalizar ahí solo pierde tiempo."""
        if self.error_loc is None or self.error_loc < 0.5 * self.cfg['error_loc_malo']:
            return False
        ahora = time.monotonic()
        self.fallos_localizacion.append(ahora)
        while (self.fallos_localizacion
               and ahora - self.fallos_localizacion[0] > self.cfg['ventana_fallos_relocalizacion_s']):
            self.fallos_localizacion.popleft()
        if (len(self.fallos_localizacion) >= self.cfg['fallos_para_relocalizar']
                and ahora - self.t_ult_reloc >= self.cfg['espera_entre_relocalizaciones']):
            self.relocalizar(siguiente, f'{len(self.fallos_localizacion)} fallos de navegación sin resolver')
            return True
        return False

    def pose_creida_ahora(self, r):
        """Donde el robot creía estar al perderse, más lo que se movió desde ahí
        según la odometría (que no se pierde: solo acumula error de a poco)."""
        p, o0 = r['pose_creida'], r['odom_creida']
        o1 = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
        if p is None:
            return self.pose_robot()
        if o0 is None or o1 is None:
            return p
        dx, dy = o1[0] - o0[0], o1[1] - o0[1]
        c, s_ = math.cos(-o0[2]), math.sin(-o0[2])
        lx, ly = c * dx - s_ * dy, s_ * dx + c * dy           # avance en el marco del robot al perderse
        c, s_ = math.cos(p[2]), math.sin(p[2])
        return (p[0] + c * lx - s_ * ly, p[1] + s_ * lx + c * ly, ang_norm(p[2] + o1[2] - o0[2]))

    def mover_para_relocalizar(self, r, ahora):
        """Quieto `espera_mover_reloc_s` (AMCL se actualiza igual con /request_nomotion_update);
        si no convergió, avanza un tramo corto hacia donde hay lugar y vuelve a esperar.
        Devuelve True mientras el robot está quieto (se puede validar AMCL)."""
        c = self.cfg
        if r['mov'] == 'moviendo':
            odom = self.lookup(c['odom_frame'], c['base_frame'])
            movido = math.dist(odom[:2], r['p0_mov'][:2]) if odom is not None and r['p0_mov'] else 0.0
            puntos = self.puntos_scan()
            libre = self.espacio_libre(puntos, r['ang_mov']) if puntos is not None else 0.0
            v = c['velocidad_mover_reloc']
            if (movido >= r['dist_mov'] or libre < 0.05
                    or ahora - r['t_mov'] > 2.0 + 3.0 * r['dist_mov'] / v):
                self.pub_cmd.publish(Twist())
                r.update(mov='esperando', t_mov=ahora, movs=r['movs'] + 1, estable_desde=None)
                return False
            cmd = Twist()
            cmd.linear.x, cmd.linear.y = v * math.cos(r['ang_mov']), v * math.sin(r['ang_mov'])
            self.pub_cmd.publish(cmd)
            return False
        self.pub_cmd.publish(Twist())
        if (r['estable_desde'] is None and r['movs'] < c['movimientos_reloc']
                and ahora - r['t_mov'] >= c['espera_mover_reloc_s']):
            puntos = self.puntos_scan()
            mejor = None
            if puntos is not None:
                for ang in np.radians(np.arange(0, 360, 15)):
                    libre = self.espacio_libre(puntos, float(ang))
                    puntaje = min(libre, 0.8)
                    if r['ang_mov'] is not None:       # preferir una dirección nueva
                        puntaje -= 0.2 * math.cos(float(ang) - r['ang_mov'])
                    if libre >= 0.20 and (mejor is None or puntaje > mejor[0]):
                        mejor = (puntaje, float(ang), libre)
            if mejor is None:
                r['movs'] = c['movimientos_reloc']    # sin lugar: queda quieto validando
                return True
            dist = min(c['distancia_mover_reloc_m'], mejor[2] - 0.12)
            self.get_logger().info(f'AMCL sin converger: me muevo {100 * dist:.0f} cm hacia '
                                   f'{math.degrees(mejor[1]):.0f}° para ver el lugar desde otro punto')
            r.update(mov='moviendo', ang_mov=mejor[1], dist_mov=dist, t_mov=ahora,
                     p0_mov=self.lookup(c['odom_frame'], c['base_frame']))
            return False
        return True

    def paso_relocalizacion(self):
        r = self.reloc
        if r is None:
            return
        ahora = time.monotonic()
        if r['fase'] == 'frenando':
            self.pub_cmd.publish(Twist())
            cancelacion = r['cancelacion']
            if ahora - r['t0'] < 1.0 or (cancelacion is not None and not cancelacion.done()):
                if ahora - r['t0'] >= 5.0 and ahora - r['ultimo_aviso'] >= 5.0:
                    self.get_logger().warn('Sigo esperando que termine la navegación cancelada antes de reiniciar AMCL')
                    r['ultimo_aviso'] = ahora
                return
            # 1º) Corrección LOCAL (~1 s): buscar cerca la pose donde el láser vuelve
            #     a calzar con el mapa y dársela a AMCL. Es lo que pasa casi siempre
            #     (AMCL atrasado en un giro). Reiniciar AMCL en todo el mapa tarda
            #     decenas de segundos y en un laberinto con pasillos parecidos puede
            #     converger a otro lugar: queda como último recurso.
            if not r.get('local_hecho'):
                r['local_hecho'] = True
                if self.relocalizar_local():
                    r.update(fase='confirmando_local', t_local=ahora)
                    return
            # 2º) Partículas alrededor de donde cree estar (como en la largada), no
            #     en todo el mapa: el reinicio global lo ubicaba en lugares a los que
            #     el robot no pudo haber llegado. Cada reintento abre más la nube.
            centro = self.pose_creida_ahora(r)
            if centro is None:
                if ahora - r['ultimo_aviso'] >= 5.0:
                    self.get_logger().warn('Sin pose para reubicar AMCL; navegación detenida')
                    r['ultimo_aviso'] = ahora
                return
            k = self.cfg['factor_std_reintento'] ** r['intentos']
            std_xy = min(self.cfg['std_reubicar_xy_m'] * k, self.cfg['max_std_reubicar_xy_m'])
            std_yaw = min(self.cfg['std_reubicar_yaw_rad'] * k, math.pi)
            self.publicar_pose(centro[0], centro[1], centro[2], std_xy, std_yaw)
            r.update(fase='reiniciando', centro=centro, std_reubicar=std_xy)
            self.get_logger().warn(
                f'Reparto las partículas de AMCL alrededor de donde creo estar ({centro[0]:.2f}, '
                f'{centro[1]:.2f}, {math.degrees(centro[2]):.0f}°) con σ={100 * std_xy:.0f} cm, '
                f'{math.degrees(std_yaw):.0f}°')
            return
        if r['fase'] == 'confirmando_local':
            self.pub_cmd.publish(Twist())
            if ahora - r['t_local'] < 1.5:
                return
            pose = self.pose_robot()
            error = self.medir_localizacion(pose) if pose is not None else None
            if error is not None and error <= self.cfg['error_reloc_ok']:
                self.error_loc = None
                self.loc_mala_desde = None
                self.fallos_localizacion.clear()
                self.hist.clear()
                self.hist_global.clear()
                self.get_logger().info(f'Relocalización local confirmada (láser a {100 * error:.1f} cm del mapa)')
                self.reloc = None
                self.estado = r['siguiente']
                return
            self.get_logger().warn('La corrección local no se sostuvo: reparto partículas alrededor de donde creo estar')
            r.update(fase='frenando', t0=ahora - 1.0, cancelacion=None)
            return
        if r['fase'] == 'reiniciando':
            self.pub_cmd.publish(Twist())
            r.update(fase='convergiendo', t0=ahora, secuencia_inicio=self.secuencia_amcl,
                     estable_desde=None, ultimo_scan=None, giro_fin=False,
                     intentos=r['intentos'] + 1, mov='esperando', t_mov=ahora, movs=0)
            self.std_amcl = (math.inf, math.inf)
            self.error_loc = None
            self.loc_mala_desde = None
            self.lista_negra.clear()
            self.portales_fallidos.clear()
            # Los objetos fuera del mapa ya confirmados se CONSERVAN: son referencias
            # para ubicarse (también están en el mapa de AMCL). Solo se limpia lo del
            # costmap, que pudo quedar marcado con la pose equivocada.
            self.grilla.actualizar_obstaculos(np.zeros_like(self.grilla.libre))
            if self.sectorizador is not None:
                self.sectorizador.indice_ordenado = -1
            self.dist_casa = None
            self.get_logger().warn(f'AMCL reubicado (intento {r["intentos"]}); espero convergencia con lecturas nuevas')
            return

        if (self.nomotion_cli.service_is_ready() and ahora - r['ultimo_nomotion'] >= 0.5
                and (r['nomotion'] is None or r['nomotion'].done())):
            try:
                if r['nomotion'] is not None:
                    r['nomotion'].result()
                r['nomotion'] = self.nomotion_cli.call_async(EmptySrv.Request())
            except Exception as error:  # noqa: BLE001
                r['nomotion'] = None
                if ahora - r['ultimo_aviso'] >= 5.0:
                    self.get_logger().warn(f'AMCL no aceptó la actualización sin movimiento: {error}')
                    r['ultimo_aviso'] = ahora
            r['ultimo_nomotion'] = ahora
        r['giro_fin'] = self.mover_para_relocalizar(r, ahora)
        if not r['giro_fin']:
            r['estable_desde'] = None
        else:
            pose = self.pose_robot()
            stamp = Time.from_msg(self.scan.header.stamp).nanoseconds if self.scan is not None else None
            if stamp is not None and stamp != r['ultimo_scan']:
                r['ultimo_scan'] = stamp
                error = self.medir_localizacion(pose) if pose is not None else None
                fresca = (self.secuencia_amcl >= r['secuencia_inicio'] + 3
                          and self.t_ultima_pose_amcl is not None
                          and ahora - self.t_ultima_pose_amcl <= 2.0)
                creida = self.pose_creida_ahora(r) if r.get('centro') is not None else None
                # Una pose a más de 3σ de donde creía estar no es alcanzable: no aceptarla.
                cerca = (creida is None or pose is None
                         or math.dist(pose[:2], creida[:2]) <= 3.0 * r.get('std_reubicar', 1.0))
                buena = (fresca and error is not None and error <= self.cfg['error_reloc_ok'] and cerca
                         and self.std_amcl[0] <= self.cfg['relocalizacion_std_xy_m']
                         and self.std_amcl[1] <= self.cfg['relocalizacion_std_yaw_rad'])
                if buena:
                    if r['estable_desde'] is None:
                        r['estable_desde'] = ahora
                        r['pose_estable'] = pose
                    elif (math.dist(pose[:2], r['pose_estable'][:2]) > 0.10
                          or abs(ang_norm(pose[2] - r['pose_estable'][2])) > 0.15):
                        r['estable_desde'] = ahora
                        r['pose_estable'] = pose
                else:
                    r['estable_desde'] = None
        if (r['estable_desde'] is not None
                and ahora - r['estable_desde'] >= self.cfg['confirmacion_relocalizacion_s']
                and self.t_ultima_pose_amcl is not None and ahora - self.t_ultima_pose_amcl <= 2.0
                and self.std_amcl[0] <= self.cfg['relocalizacion_std_xy_m']
                and self.std_amcl[1] <= self.cfg['relocalizacion_std_yaw_rad']
                and self.puntos_scan() is not None):
            self.error_loc = None
            self.loc_mala_desde = None
            self.fallos_localizacion.clear()
            self.reintentos_regreso = 0
            self.n_atascos = 0
            self.hist.clear()
            self.hist_global.clear()
            self.limpiar_costmaps()
            self.get_logger().info(f'AMCL convergió: σxy={self.std_amcl[0]:.2f} m, '
                                   f'σyaw={math.degrees(self.std_amcl[1]):.1f}°; recalculo la ruta')
            self.reloc = None
            self.estado = r['siguiente']
            return
        if ahora - r['t0'] >= self.cfg['timeout_relocalizacion_s']:
            r['cancelacion'] = self.cancelar_actual()
            r.update(fase='frenando', t0=ahora, estable_desde=None)
            self.pub_cmd.publish(Twist())
            self.pub_cmd.publish(Twist())
            self.get_logger().warn('AMCL todavía no converge: vuelvo a repartir las partículas, un poco más abiertas')

    # ====================================================== arranque de Nav2/AMCL
    NODOS_NAV2 = ('controller_server', 'smoother_server', 'planner_server', 'behavior_server',
                  'bt_navigator', 'waypoint_follower', 'velocity_smoother')

    def revisar_ciclo_de_vida(self, nodos, activar=True):
        """Con la PC cargada, a veces se pierde la respuesta de un nodo al
        lifecycle_manager ("failed to send response to /amcl/change_state") y
        el manager queda esperando para siempre: el nodo queda configurado pero
        inactivo. Acá se pregunta el estado y se lo termina de activar."""
        ahora = time.monotonic()
        for n in nodos:
            if n not in self.lc:
                self.lc[n] = (self.create_client(GetState, f'/{n}/get_state'),
                              self.create_client(ChangeState, f'/{n}/change_state'))
            get_cli, _ = self.lc[n]
            pendiente = self.consultas_lc.get(n)
            if pendiente is not None and not pendiente[0].done() and ahora - pendiente[1] < 4.0:
                continue
            if get_cli.service_is_ready():
                try:
                    futuro = get_cli.call_async(GetState.Request())
                except Exception as error:  # noqa: BLE001
                    self.estados_lc.pop(n, None)
                    self.get_logger().warn(f'No pude consultar {n}: {error}', throttle_duration_sec=5.0)
                    continue
                self.consultas_lc[n] = (futuro, ahora)
                futuro.add_done_callback(lambda f, n=n: self._estado_ciclo_de_vida(n, f, activar))

    def _estado_ciclo_de_vida(self, nodo, fut, activar=True):
        consulta = self.consultas_lc.get(nodo)
        if consulta is not None and consulta[0] is not fut:
            return
        try:
            estado = fut.result().current_state.id
        except Exception:  # noqa: BLE001
            self.estados_lc.pop(nodo, None)
            return
        ahora = time.monotonic()
        self.estados_lc[nodo] = (estado, ahora)
        if not activar:
            return
        # 1 = unconfigured, 2 = inactive (3 = active, el resto son transitorios).
        transicion = {1: Transition.TRANSITION_CONFIGURE, 2: Transition.TRANSITION_ACTIVATE}.get(estado)
        if transicion is None:
            return
        pendiente = self.transiciones_lc.get(nodo)
        if pendiente is not None and not pendiente[0].done() and ahora - pendiente[1] < 8.0:
            return
        self.get_logger().warn(
            f"{nodo} quedó {'sin configurar' if estado == 1 else 'inactivo'} (se perdió una "
            f"respuesta del lifecycle_manager): lo {'configuro' if estado == 1 else 'activo'} yo")
        req = ChangeState.Request()
        req.transition.id = transicion
        try:
            self.transiciones_lc[nodo] = (self.lc[nodo][1].call_async(req), ahora)
        except Exception as error:  # noqa: BLE001
            self.get_logger().warn(f'No pude activar {nodo}: {error}', throttle_duration_sec=5.0)

    def nav2_activo(self):
        """Un action server puede existir antes de estar activo: no alcanza con descubrirlo."""
        ahora = time.monotonic()
        return all(n in self.estados_lc and self.estados_lc[n][0] == State.PRIMARY_STATE_ACTIVE
                   and ahora - self.estados_lc[n][1] <= 10.0 for n in self.NODOS_NAV2)

    def recuperar_nav2(self, siguiente=None, giro=None):
        """Recuperación de infraestructura: esperar quieto, sin marcar metas como obstáculos."""
        previo = self.estado
        cancelacion = self.cancelar_actual()
        self.goal_actual = None
        self.pub_cmd.publish(Twist())
        self.escape = None
        self.giro_pendiente = None
        if self.objeto_observado is not None:
            self.reanudar_nav2 = OBSERVANDO_OBJETO
            self.objeto_observado.update(cancelacion=cancelacion, fase='frenando', desde=None,
                                         quieto_desde=None, giros=0)
            giro = None  # Recalcular hacia el objeto al recuperar los servidores.
        elif self.motivo_regreso is not None or previo in (VOLVIENDO, REGRESAR):
            self.reanudar_nav2 = REGRESAR
        else:
            self.reanudar_nav2 = siguiente or ELIGIENDO
        self.giro_reanudar_nav2 = giro
        self.estado = ACTIVANDO_NAV2
        self.estados_lc.clear()
        self.t_ult_revision_lc = time.monotonic()
        self.revisar_ciclo_de_vida(self.NODOS_NAV2)
        self.get_logger().warn('Nav2 no está operativo: espero su activación; no es un atasco físico')

    def paso_activacion_nav2(self, ahora):
        self.pub_cmd.publish(Twist())
        if ahora - self.t_ult_revision_lc >= 1.0:
            self.t_ult_revision_lc = ahora
            self.revisar_ciclo_de_vida(self.NODOS_NAV2)
        if (not self.nav2_activo() or not self.nav.server_is_ready()
                or not self.nav_ruta.server_is_ready() or not self.spin_cli.server_is_ready()):
            return
        o = self.objeto_observado
        if o is not None:
            o.update(inicio=ahora, quieto_desde=None, desde=None)
        self.limite_observacion(100.0)
        siguiente = self.reanudar_nav2
        giro = self.giro_reanudar_nav2
        self.giro_reanudar_nav2 = None
        self.estado = siguiente
        self.get_logger().info('Nav2 activo: reanudo la misión sin maniobras de escape')
        if giro is not None:
            self.girar(giro, siguiente)

    # ============================================================ velocidad segura
    def distancia_minima_chasis(self):
        puntos = self.puntos_scan()
        if puntos is None or puntos[0].size == 0:
            return None
        x, y = puntos
        dx = np.maximum(np.maximum(-self.ATRAS - x, x - self.ADELANTE), 0.0)
        dy = np.maximum(np.abs(y) - self.MEDIO_ANCHO, 0.0)
        return float(np.min(np.hypot(dx, dy)))

    def libre_hacia_donde_va(self):
        """Metros libres que tiene el chasis en la dirección en que se está moviendo
        (None si está casi quieto o sin láser). Una pared al costado no cuenta."""
        vel = getattr(self, 'vel_base', None)
        if vel is None or math.hypot(*vel) < 0.05:
            return None
        puntos = self.puntos_scan()
        if puntos is None or puntos[0].size == 0:
            return None
        return self.espacio_libre(puntos, math.atan2(vel[1], vel[0]))

    def publicar_limite_velocidad(self):
        """Velocidad máxima salvo que vaya HACIA un obstáculo: frena según los metros
        libres en la dirección de avance. Pasar al lado de una pared no frena."""
        limite = 100.0
        libre = self.libre_hacia_donde_va()
        if libre is not None:
            if libre < self.cfg['distancia_muy_lenta']:
                limite = float(self.cfg['porcentaje_muy_lenta'])
            elif libre < self.cfg['distancia_lenta']:
                limite = float(self.cfg['porcentaje_lenta'])
        if self.error_loc is not None and self.error_loc > self.cfg['error_loc_malo']:
            limite = min(limite, float(self.cfg['porcentaje_loc_mala']))
        if limite != self.ultimo_limite or int(time.monotonic()) % 2 == 0:
            m = SpeedLimit()
            m.header.stamp = self.get_clock().now().to_msg()
            m.percentage = True
            m.speed_limit = limite
            self.pub_vel.publish(m)
            if limite != self.ultimo_limite and limite < 100.0:
                self.get_logger().debug(f'Límite de velocidad {limite:.0f}%')
            self.ultimo_limite = limite

    # ======================================================================= pose
    def lookup(self, padre, hijo):
        try:
            t = self.tf_buffer.lookup_transform(padre, hijo, Time())
        except Exception:  # noqa: BLE001
            return None
        tr = t.transform.translation
        return tr.x, tr.y, quat_a_yaw(t.transform.rotation)

    def pose_robot(self):
        if self.grilla is None:
            return None
        return self.lookup(self.grilla.frame, self.cfg['base_frame'])

    def buscar_pose(self, centro, rango_xy=0.30, rango_yaw=0.40, paso_xy=0.03, paso_yaw=0.04,
                    prior=0.004):
        """Pose cercana a `centro` (x, y, yaw) donde el láser mejor calza con las paredes
        del mapa. Devuelve (x, y, yaw, error_medio_m, error_en_centro_m) o None."""
        puntos = self.puntos_scan(alcance=self.cfg.get('rango_laser_localizacion_m', 8.0))
        if puntos is None or self.dist_pared is None or puntos[0].size < 30:
            return None
        px, py = puntos
        paso = max(1, math.ceil(px.size / 120))
        px, py = px[::paso], py[::paso]
        g = self.grilla
        x0, y0, yaw0 = centro
        off = np.arange(-rango_xy, rango_xy + 1e-4, paso_xy)
        ox, oy = np.meshgrid(off, off)
        ox, oy = (x0 + ox).ravel(), (y0 + oy).ravel()
        def error_scan(mx, my):
            # Interpolar igual que en medir_localizacion, incluyendo todos los
            # rayos fuera del mapa y la media celda de las paredes del borde.
            u, v = (mx - g.ox) / g.res - 0.5, (my - g.oy) / g.res - 0.5
            uc, vc = np.clip(u, 0, g.ancho - 1), np.clip(v, 0, g.alto - 1)
            col, f = np.floor(uc).astype(int), np.floor(vc).astype(int)
            col1, f1 = np.minimum(col + 1, g.ancho - 1), np.minimum(f + 1, g.alto - 1)
            a, b = uc - col, vc - f
            d = ((1 - a) * (1 - b) * D[f, col] + a * (1 - b) * D[f, col1]
                 + (1 - a) * b * D[f1, col] + a * b * D[f1, col1])
            d += g.res * np.hypot(u - uc, v - vc)
            return np.clip(d - 0.5 * g.res, 0.0, 0.30).mean(axis=-1)

        D = self.campo_loc()
        c0, s0 = math.cos(yaw0), math.sin(yaw0)
        en_centro = float(error_scan(x0 + c0 * px - s0 * py, y0 + s0 * px + c0 * py))
        mejor = None
        for dyaw in np.arange(-rango_yaw, rango_yaw + 1e-4, paso_yaw):
            yaw = yaw0 + dyaw
            c, s = math.cos(yaw), math.sin(yaw)
            rx, ry = c * px - s * py, s * px + c * py
            mx = ox[:, None] + rx[None, :]
            my = oy[:, None] + ry[None, :]
            crudo = error_scan(mx, my)
            # Desempate hacia el centro: en un pasillo largo varias poses "calzan"
            # igual; preferimos la más cercana a la estimación actual.
            err = crudo + prior * (np.hypot(ox - x0, oy - y0) / rango_xy + abs(dyaw) / rango_yaw)
            i = int(np.argmin(err))
            if mejor is None or err[i] < mejor[3]:
                mejor = (float(ox[i]), float(oy[i]), float(ang_norm(yaw)), float(err[i]), float(crudo[i]))
        return mejor[0], mejor[1], mejor[2], mejor[4], en_centro

    def estimar_pose_inicial(self):
        """Pose de largada: la que mejor calza el láser con el mapa cerca de la nominal."""
        r = self.buscar_pose((self.cfg['x_inicial'], self.cfg['y_inicial'], self.cfg['yaw_inicial']))
        return None if r is None else r[:4]

    def publicar_pose(self, x, y, yaw, std_xy, std_yaw):
        m = PoseWithCovarianceStamped()
        m.header.frame_id = self.grilla.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.pose.pose.position.x, m.pose.pose.position.y = x, y
        m.pose.pose.orientation.z, m.pose.pose.orientation.w = yaw_a_quat(yaw)
        cov = [0.0] * 36
        cov[0] = cov[7] = std_xy ** 2
        cov[35] = std_yaw ** 2
        m.pose.covariance = cov
        self.pub_init.publish(m)

    def publicar_pose_inicial(self):
        x, y, yaw = self.cfg['x_inicial'], self.cfg['y_inicial'], self.cfg['yaw_inicial']
        std_xy, std_yaw = self.cfg['std_xy'], self.cfg['std_yaw']
        if self.pose_inicial_estimada is None:
            est = self.estimar_pose_inicial()
            if est is not None:
                self.pose_inicial_estimada = est
                self.get_logger().info(
                    f'Pose de largada por láser vs. mapa: ({est[0]:.2f}, {est[1]:.2f}, '
                    f'{math.degrees(est[2]):.0f}°), error medio {100 * est[3]:.1f} cm')
        est = self.pose_inicial_estimada
        if est is not None and est[3] < 0.08:
            x, y, yaw = est[0], est[1], est[2]
            std_xy, std_yaw = 0.12, 0.12   # confiamos más: el láser ya calzó
        self.publicar_pose(x, y, yaw, std_xy, std_yaw)

    def fijar_largada(self, forzar=False):
        """Largada en el marco map = (map->odom actual) ∘ (pose de odom al largar)."""
        if self.largada_fija:
            return
        convergido = (max(self.std_amcl) < self.cfg['std_convergido']
                      and self.error_loc is not None and self.error_loc < 0.6 * self.cfg['error_loc_malo'])
        vencido = time.monotonic() - (self.t_localizado or time.monotonic()) > self.cfg['max_espera_convergencia_s']
        if not (convergido or vencido or forzar):
            return
        m_o = self.lookup(self.grilla.frame, self.cfg['odom_frame'])
        if m_o is not None and self.p0_odom is not None:
            tx, ty, th = m_o
            px, py, pth = self.p0_odom
            self.largada = (tx + math.cos(th) * px - math.sin(th) * py,
                            ty + math.sin(th) * px + math.cos(th) * py,
                            ang_norm(th + pth))
        self.largada_fija = True
        self.dist_casa = None
        self.get_logger().info(
            f'Largada estimada en ({self.largada[0]:.2f}, {self.largada[1]:.2f}) '
            f'[AMCL std={self.std_amcl[0]:.3f} m]')

    # ================================================================= raycasting
    def rayos_desde(self, fila, col):
        g = self.grilla
        f = np.floor(fila + 0.5 + self.ray_dy).astype(np.int32)
        c = np.floor(col + 0.5 + self.ray_dx).astype(np.int32)
        fuera = (f < 0) | (f >= g.alto) | (c < 0) | (c >= g.ancho)
        f = np.clip(f, 0, g.alto - 1)
        c = np.clip(c, 0, g.ancho - 1)
        visible = ~np.maximum.accumulate(g.bloqueo[f, c] | g.obstaculos_extra[f, c] | fuera, axis=1)
        return f, c, visible

    def marcar_visto(self, pose):
        """Cono desde el centro óptico y rumbo real de cámara, no desde la base."""
        g = self.grilla
        fila, col = g.a_celda(pose[0], pose[1])
        if not g.dentro(fila, col):
            return
        f, c, vis = self.rayos_desde(fila, col)
        dif = np.abs(np.arctan2(np.sin(self.angulos - pose[2]), np.cos(self.angulos - pose[2])))
        sel = vis & (dif <= self.fov / 2.0)[:, None]
        self.visto[f[sel], c[sel]] = True

    def marcar_alrededor(self, x, y, radio):
        """Da por visto el piso alrededor de un objeto que no está en el mapa: la
        sombra que deja detrás es chica y no vale ir a rodearlo para mirarla."""
        g = self.grilla
        if g is None or self.visto is None:
            return
        f, c = g.a_celda(x, y)
        k = max(1, int(round(radio / g.res)))
        f0, f1 = max(0, f - k), min(g.alto, f + k + 1)
        c0, c1 = max(0, c - k), min(g.ancho, c + k + 1)
        ff, cc = np.ogrid[f0:f1, c0:c1]
        disco = (ff - f) ** 2 + (cc - c) ** 2 <= k * k
        self.visto[f0:f1, c0:c1] |= disco & g.libre[f0:f1, c0:c1]

    def cobertura(self):
        piso = self.grilla.libre & ~self.grilla.bloques_conocidos
        return float((self.visto & piso).sum()) / max(1, int(piso.sum()))

    def completar_huecos(self):
        if self.grilla is None or self.visto is None:
            return
        ahora = time.monotonic()
        if ahora - self.t_ultimo_huecos < 1.0:
            return
        self.t_ultimo_huecos = ahora
        g = self.grilla
        contar_huecos_pequenos(self.visto, g.libre & ~g.bloques_conocidos,
                               g.res, self.cfg['area_caja_m2'])

    def preparar_sectores(self, pose):
        if self.cfg['modo_exploracion'] != 'sectores_nbv' or self.sectorizador is not None:
            return
        fila, col = self.grilla.a_celda(*pose[:2])
        self.sectorizador = SectoresNBV(
            self.grilla.libre, self.grilla.transitable_base, self.grilla.res,
            (col, fila), self.cfg['cobertura_objetivo_sector'],
            self.cfg['area_minima_sector_m2'], self.cfg['ancho_minimo_sector_m'],
            self.cfg['area_residual_ignorable_m2'])
        self.sectorizador.actualizar_bloques(self.grilla.bloques_conocidos)
        if self.cfg['publicar_debug']:
            self.pub_sectores.publish(marcadores_sectores(self.mapa_msg, self.sectorizador.regiones))
        self.get_logger().info(
            f'BCD: {len(self.sectorizador.orden)} sectores alcanzables; '
            f'depuración {self.sectorizador.depuracion}')

    def mascaras_portales(self):
        """
        A las cajas/objetos (bloques conocidos y obstáculos fuera del mapa) siempre se
        les deja `margen_objeto_ruta_m`, aunque a las paredes se les deje menos.
        """
        g = self.grilla
        margen = self.cfg.get('margen_parada_portal_m', 0.22)
        estricto = g.transitable & ~dilatar(
            g.bloqueo | g.obstaculos_extra, margen / g.res)
        margen_obj = self.cfg.get('margen_objeto_ruta_m', 0.22)
        lejos_objetos = ~dilatar(g.obstaculos_extra, margen_obj / g.res) if g.obstaculos_extra.any() \
            else np.ones_like(g.transitable)
        return ((estricto, margen),
                (g.candidatable & g.transitable & lejos_objetos, g.m_cand * g.res),
                (g.transitable & lejos_objetos, g.m_trans * g.res),
                (g.transitable, g.m_trans * g.res - 1e-3))     # último recurso: rozando objetos

    def seleccionar_ruta_portales(self, pose, planificar, destino=None):
        """Compara recorridos completos antes de preferir un margen mayor.

        La holgura no justifica cambiar a un rodeo: sólo compiten las rutas
        que se mantienen cerca de la longitud mínima realmente enviada a Nav2.
        Devuelve también la máscara elegida para conservar el mismo recorrido.
        """
        candidatas = []
        g = self.grilla
        regiones = self.sectorizador.regiones if self.sectorizador is not None else None
        mascaras = list(self.mascaras_portales())
        for indice_mascara, (mascara, margen) in enumerate(mascaras):
            seguro = mascara.copy()
            ruta = planificar(seguro)
            if not ruta:
                continue
            pasos, distancia = self.guias_de_ruta(ruta, pose, destino, seguro=seguro)
            if not pasos:
                continue
            puntos = [pose[:2]]
            for punto in pasos:
                if math.dist(puntos[-1], punto[:2]) > 1e-6:
                    puntos.append(punto[:2])
            rumbos = [math.atan2(b[1] - a[1], b[0] - a[0])
                      for a, b in zip(puntos, puntos[1:]) if math.dist(a, b) >= 0.08]
            cambios = [abs(ang_norm(b - a)) for a, b in zip(rumbos, rumbos[1:])]
            giros = sum(cambio > math.radians(25) for cambio in cambios)
            giro_total = sum(cambios)
            cruces = 0
            if regiones is not None:
                ids = []
                for punto in puntos:
                    f, c = g.a_celda(*punto)
                    ids.append(int(regiones[f, c]) if g.dentro(f, c) else -1)
                cruces = sum(a != b for a, b in zip(ids, ids[1:]))
            candidatas.append((ruta, pasos, distancia, seguro, margen,
                               cruces, giros, giro_total, indice_mascara == len(mascaras) - 1))
        if not candidatas:
            return [], [], 0.0, None, 0.0
        # Una ruta que respeta el margen a las cajas gana aunque sea bastante más larga:
        # pasar rozando una caja costó varios minutos de atascos.
        con_margen = [c for c in candidatas if not c[8]]
        if con_margen:
            extra = self.cfg.get('desvio_maximo_por_objetos_m', 4.0)
            minima_con_margen = min(c[2] for c in con_margen)
            if minima_con_margen <= min(c[2] for c in candidatas) + extra:
                candidatas = con_margen
        minima = min(candidata[2] for candidata in candidatas)
        desvio = min(max(0.0, self.cfg.get('desvio_maximo_ruta_m', 0.30)),
                     minima * max(0.0, self.cfg.get('fraccion_desvio_ruta', 0.15)))
        cercanas = [candidata for candidata in candidatas
                    if candidata[2] <= minima + desvio + 1e-6]
        # Entre rutas de largo parecido manda la distancia a las paredes: antes ganaba
        # la de menos giros y, en 7 de 15 tramos, las paradas quedaban a 13 cm.
        elegida = min(cercanas, key=lambda c: (
            c[5], -c[4], c[6], round(c[7], 1), c[2]))
        return elegida[:5]

    def seleccionar_sector_pendiente(self, pose):
        """Reordena sólo los sectores pendientes desde la pose actual del robot."""
        s = self.sectorizador
        fila, col = self.grilla.a_celda(*pose[:2])
        while s.actual is not None:
            elegido = s.elegir_proximo(self.grilla.transitable, (col, fila))
            if elegido is not None:
                self.get_logger().info(
                    f'Siguiente sector {elegido}: recorrido global desde acá '
                    f'{" -> ".join(str(r) for r in s.recorrido_planificado)} -> largada '
                    f'(≈{s.metros_planificados:.1f} m)')
            anterior = s.actual
            s.avanzar_cubiertos(self.visto)
            if s.actual == anterior:
                break

    def en_sector(self, pose, radio=None):
        """¿El robot ya está en el sector actual, o a menos de `radio_entrada_sector_m`
        de él? Exigir el último par de centímetros del portal dejaba a Nav2 oscilando
        ~20 s en cada cambio de zona: desde ahí, las vistas NBV (que están dentro del
        sector) lo meten adentro con un goal normal."""
        s, g = self.sectorizador, self.grilla
        if s is None or s.actual is None or pose is None:
            return False
        fila, col = g.a_celda(*pose[:2])
        if s.dentro_actual((col, fila)):
            return True
        k = int(round((self.cfg['radio_entrada_sector_m'] if radio is None else radio) / g.res))
        if k <= 0:
            return False
        f0, f1 = max(0, fila - k), min(g.alto, fila + k + 1)
        c0, c1 = max(0, col - k), min(g.ancho, col + k + 1)
        if f0 >= f1 or c0 >= c1:
            return False
        ff, cc = np.ogrid[f0:f1, c0:c1]
        disco = (ff - fila) ** 2 + (cc - col) ** 2 <= k * k
        return bool((disco & (s.regiones[f0:f1, c0:c1] == s.actual) & g.transitable[f0:f1, c0:c1]).any())

    def paso_hacia_sector(self, pose):
        """Envía una sola navegación continua por los portales hasta el sector."""
        s, g = self.sectorizador, self.grilla
        fila, col = g.a_celda(*pose[:2])
        if not g.dentro(fila, col) or not g.transitable[fila, col]:
            self.get_logger().warn('Inicio dentro del margen: me separo sin descartar sectores pendientes')
            self.cancelar_actual()
            self.goal_actual = None
            self.n_atascos += 1
            self.iniciar_escape(ELIGIENDO)
            return True
        if self.en_sector(pose):
            self.ruta_sector.clear()
            self.ruta_region = None
            return False
        if self.ruta_region != s.actual or not self.ruta_sector:
            def planificar(seguro):
                Exploracion.bloquear_pasos_fallidos(self, seguro, (col, fila))
                return s.ruta_entrada(
                    seguro, (col, fila), self.cfg['paso_maximo_portal_m'])
            (self.ruta_sector, pasos, distancia, self.seguro_ruta_sector,
             margen_usado) = self.seleccionar_ruta_portales(pose, planificar)
            self.ruta_region = s.actual
            if not self.ruta_sector:
                destino = None if s.actual in self.sectores_directo else self.punto_interior_sector(pose)
                if destino is not None:
                    self.sectores_directo.add(s.actual)
                    x, y, rumbo, d = destino
                    self.get_logger().warn(f'Sector {s.actual} sin ruta por portales: entro directo con Nav2 '
                                           f'hasta ({x:.2f}, {y:.2f})')
                    if self.encarar_y_enviar(x, y, rumbo, 1, d, f'portal sector {s.actual}', NAVEGANDO):
                        return True
                self.get_logger().warn(f'Sector {s.actual} sin ruta por portales')
                s.avanzar_inaccesible()
                self.ruta_region = None
                return False
            self.get_logger().info(
                f'Sector {s.actual}: trayectoria por {len(self.ruta_sector)} puntos calculados, '
                f'{sum(portal for _, portal in self.ruta_sector)} portal(es), '
                f'margen de paradas {margen_usado:.2f} m')
        else:
            pasos, distancia = self.guias_de_ruta(
                self.ruta_sector, pose, seguro=self.seguro_ruta_sector)
        if not pasos:
            self.get_logger().warn('No hay conexión libre desde el robot a la ruta del sector; recalculo tras desencajar')
            self.ruta_sector.clear()
            self.n_atascos += 1
            self.cancelar_actual()
            self.goal_actual = None
            self.iniciar_escape(ELIGIENDO)
            return True
        x, y, rumbo = pasos[-1]
        etiqueta = f'portal sector {s.actual}'
        self.encarar_y_enviar(x, y, rumbo, 1, distancia, etiqueta, NAVEGANDO, pasos)
        self.publicar_debug([])
        return True

    def punto_interior_sector(self, pose):
        """Entrada cercana y alcanzable; mejora la holgura sin cruzar todo el sector."""
        s, g = self.sectorizador, self.grilla
        fila, col = g.a_celda(*pose[:2])
        dist = bfs(g.transitable, fila, col)
        if dist is None or s.actual is None:
            return None
        cand = (s.regiones == s.actual) & g.transitable & (dist >= 0)
        if not cand.any():
            return None
        interior = cand & ~dilatar(s.regiones != s.actual, 0.15 / g.res)
        if interior.any():
            cand = interior
        minima = float(dist[cand].min())
        cercanas = cand & (dist <= minima + 0.15 / g.res)
        filas, columnas = np.nonzero(cercanas)
        i = min(range(len(filas)), key=lambda i: (
            -self.dist_pared[filas[i], columnas[i]], dist[filas[i], columnas[i]]))
        f, c = int(filas[i]), int(columnas[i])
        x, y = g.a_mundo(f, c)
        return x, y, math.atan2(y - pose[1], x - pose[0]), float(dist[f, c] * g.res)

    def guias_de_ruta(self, ruta, pose, destino=None, seguro=None):
        """Entrada, intermedio y salida de cada zona; como máximo dos uniones extra."""
        g = self.grilla
        inicio = g.a_celda(*pose[:2])[1::-1]
        regiones = (self.sectorizador.regiones if self.sectorizador is not None
                    else np.zeros_like(g.transitable, dtype=np.int32))
        guias_celdas = []
        mascaras = ([seguro & g.transitable] if seguro is not None
                    else [mascara for mascara, _margen in self.mascaras_portales()])
        for mascara in mascaras:
            guias_celdas = guias_por_zona(mascara, regiones, inicio, ruta, g.transitable)
            if guias_celdas:
                break
        if not guias_celdas:
            return [], 0.0
        # La longitud se calcula con la ruta completa, aunque Nav2 reciba pocas
        # referencias y trace por su cuenta el camino entre ellas.
        puntos_ruta = [pose[:2]] + [g.a_mundo(celda[1], celda[0]) for celda, _ in ruta]
        distancia_ruta = sum(math.dist(a, b) for a, b in zip(puntos_ruta, puntos_ruta[1:]))
        mascara_ruta = (seguro & g.transitable) if seguro is not None else g.transitable
        if self.cfg.get('guias_portal_y_medio', False):
            guias_celdas = Exploracion.guias_portal_y_centro(
                self, guias_celdas, mascara_ruta, destino_fijo=destino is not None)
        else:
            guias_celdas = Exploracion.guias_portal_medio(self, guias_celdas, mascara_ruta)
            guias_celdas = self.alejar_guias(guias_celdas, inicio, fijar_ultima=destino is not None,
                                             mascara=(seguro & g.transitable) if seguro is not None else None)
        if not guias_celdas:
            return [], 0.0
        guias = [(*g.a_mundo(celda[1], celda[0]), portal) for celda, portal in guias_celdas]
        distancia = math.dist(pose[:2], guias[0][:2])
        distancia += sum(math.dist(a[:2], b[:2]) for a, b in zip(guias, guias[1:]))
        pasos = []
        for i, (x, y, _portal) in enumerate(guias):
            if i + 1 < len(guias):
                siguiente = guias[i + 1]
                yaw = math.atan2(siguiente[1] - y, siguiente[0] - x)
            elif destino is not None:
                yaw = destino[2]
            else:
                anterior = guias[i - 1] if i else (pose[0], pose[1])
                yaw = math.atan2(y - anterior[1], x - anterior[0])
            pasos.append((x, y, yaw))
        return pasos, max(distancia, distancia_ruta)

    def holgura_m(self):
        """Metros de cada celda a la pared, zona desconocida u obstáculo (del costmap
        o bloque conocido) más cercano. Se recalcula si cambian los obstáculos."""
        g = self.grilla
        bloqueado = ~g.libre | g.obstaculos_extra
        clave = (id(g), hash(np.packbits(bloqueado).tobytes()))
        cache = getattr(self, '_holgura', None)
        if cache is None or cache[0] != clave:
            cache = (clave, _distancia_al_borde(~bloqueado) * g.res)
            self._holgura = cache
        return cache[1]

    @staticmethod
    def _bfs_en_zona(zona, origen):
        """Distancias (celdas, 4-vecinos) dentro de `zona` desde `origen` (x, y); -1 si no llega."""
        dist = np.full(zona.shape, -1, dtype=np.int32)
        if not zona[origen[1], origen[0]]:
            return dist
        dist[origen[1], origen[0]] = 0
        cola = deque([origen])
        while cola:
            x, y = cola.popleft()
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= ny < zona.shape[0] and 0 <= nx < zona.shape[1] and zona[ny, nx] and dist[ny, nx] < 0:
                    dist[ny, nx] = dist[y, x] + 1
                    cola.append((nx, ny))
        return dist

    def guias_portal_y_centro(self, guias_celdas, mascara, destino_fijo=False):
        """Por cada zona nueva de la ruta, sólo DOS poses: el portal de entrada y el medio
        de la zona (o el destino, si es la última zona y hay un destino fijo). Nav2 traza
        por su cuenta el camino entre ellas con el costmap (paredes, inflación y objetos
        del mapa local); antes seguía una cadena de 4-6 puntos que pasaba rozando cajas.
        Las dos poses se corren hacia donde hay más lugar a paredes y objetos
        (`holgura_guia_m`) sin salir de la zona: si el medio cae sobre o junto a un
        objeto, se mueve al punto libre con holgura más cercano."""
        g = self.grilla
        if not guias_celdas:
            return guias_celdas
        regiones = (self.sectorizador.regiones if getattr(self, 'sectorizador', None) is not None
                    else np.zeros_like(g.transitable, dtype=np.int32))
        holgura = Exploracion.holgura_m(self) / g.res          # celdas a pared/objeto
        tope = self.cfg.get('holgura_guia_m', 0.30) / g.res
        radio_portal = max(1, int(round(self.cfg.get('radio_portal_guia_m', 0.35) / g.res)))
        bfs_zona = Exploracion._bfs_en_zona

        corridas = []                                        # [(region, [celda, ...])]
        for celda, _portal in guias_celdas:
            celda = (int(celda[0]), int(celda[1]))
            reg = int(regiones[celda[1], celda[0]])
            if corridas and corridas[-1][0] == reg:
                corridas[-1][1].append(celda)
            else:
                corridas.append((reg, [celda]))

        def entrar_a_zona(zona, celda):
            """La celda si está en la zona; si no, la celda de la zona más cercana."""
            if zona[celda[1], celda[0]]:
                return celda
            ys, xs = np.nonzero(zona)
            if not len(xs):
                return None
            k = int(np.argmin(np.hypot(xs - celda[0], ys - celda[1])))
            if math.hypot(xs[k] - celda[0], ys[k] - celda[1]) > 2 * radio_portal:
                return None
            return int(xs[k]), int(ys[k])

        def holgado_cerca(zona, celda, radio):
            """Punto de la zona a menos de `radio` celdas de camino con más holgura."""
            celda = entrar_a_zona(zona, celda)
            if celda is None:
                return None
            dist = bfs_zona(zona, celda)
            ys, xs = np.nonzero((dist >= 0) & (dist <= radio))
            puntaje = np.minimum(holgura[ys, xs], tope) * 4.0 - dist[ys, xs] * 0.5
            k = int(np.argmax(puntaje))
            return int(xs[k]), int(ys[k])

        def medio_entre(zona, a, b, respaldo):
            """Mitad del camino a->b dentro de la zona, corrida hacia donde hay lugar."""
            if b is not None and zona[a[1], a[0]] and zona[b[1], b[0]]:
                da, db = bfs_zona(zona, a), bfs_zona(zona, b)
                minimo = da[b[1], b[0]]
                if minimo > 1:
                    ok = ((da > 0) & (db > 0) & (da + db <= minimo + max(4, 0.15 * minimo))
                          & (np.abs(da - db) <= max(2, minimo / 3)))
                    ys, xs = np.nonzero(ok)
                    if len(xs):
                        puntaje = (np.minimum(holgura[ys, xs], tope) * 4.0
                                   - (da[ys, xs] + db[ys, xs] - minimo) * 0.15
                                   - np.abs(da[ys, xs] - db[ys, xs]) * 0.05)
                        k = int(np.argmax(puntaje))
                        return int(xs[k]), int(ys[k])
            return holgado_cerca(zona, respaldo, radio_portal)

        def centro_de_zona(zona, desde):
            """Medio de la parte de la zona alcanzable desde el portal, con holgura."""
            if desde is None:
                return None
            dist = bfs_zona(zona, desde)
            ys, xs = np.nonzero(dist >= 0)
            if not len(xs):
                return None
            cx, cy = float(xs.mean()), float(ys.mean())
            puntaje = np.minimum(holgura[ys, xs], tope) * 4.0 - 0.3 * np.hypot(xs - cx, ys - cy)
            k = int(np.argmax(puntaje))
            return int(xs[k]), int(ys[k])

        nuevas = []
        ultima = len(corridas) - 1
        for k, (reg, celdas) in enumerate(corridas):
            entrada, salida = celdas[0], celdas[-1]
            zona = mascara & (regiones == reg)
            final = k == ultima
            if k == 0 and not final:
                continue                    # el robot ya está en esta zona: Nav2 va al próximo portal
            if k == 0:
                # Destino en la misma zona que el robot: una sola pose.
                punto = salida if destino_fijo else (centro_de_zona(zona, entrar_a_zona(zona, entrada))
                                                     or salida)
                nuevas.append((punto, False))
                continue
            portal = holgado_cerca(zona, entrada, radio_portal) or entrada
            nuevas.append((portal, True))
            if final and destino_fijo:
                segundo = salida
            elif final:
                segundo = centro_de_zona(zona, portal)
            else:
                segundo = medio_entre(zona, portal, entrar_a_zona(zona, salida), celdas[len(celdas) // 2])
            if segundo is not None and segundo != portal:
                nuevas.append((segundo, False))
        return nuevas

    def guias_portal_medio(self, guias_celdas, mascara):
        """Rehace las guías de cada zona como: portal de entrada -> medio de la zona ->
        portal de salida, con a lo sumo UN punto de unión cuando la recta entre dos de
        ellos no es transitable. `mascara` es la de la ruta elegida (respeta el margen a
        objetos del mapa propio). Si hacen falta más giros, se conservan sólo las tres
        referencias transitables y Nav2 calcula el camino entre ellas. Nunca se
        recupera la lista completa de guías como alternativa."""
        g = self.grilla
        if len(guias_celdas) < 2:
            return guias_celdas
        regiones = (self.sectorizador.regiones if getattr(self, 'sectorizador', None) is not None
                    else np.zeros_like(g.transitable, dtype=np.int32))
        holgura = Exploracion.holgura_m(self) / g.res          # celdas a pared/objeto
        tope = self.cfg.get('holgura_guia_m', 0.30) / g.res

        def region(celda):
            return int(regiones[celda[1], celda[0]])

        corridas = []                                        # [(region, [(celda, portal), ...])]
        for celda, portal in guias_celdas:
            celda = tuple(celda)
            if corridas and corridas[-1][0] == region(celda):
                corridas[-1][1].append((celda, portal))
            else:
                corridas.append((region(celda), [(celda, portal)]))

        def bfs_zona(zona, origen):
            dist = np.full(zona.shape, -1, dtype=np.int32)
            if not zona[origen[1], origen[0]]:
                return dist
            dist[origen[1], origen[0]] = 0
            cola = deque([origen])
            while cola:
                x, y = cola.popleft()
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= ny < zona.shape[0] and 0 <= nx < zona.shape[1] and zona[ny, nx] and dist[ny, nx] < 0:
                        dist[ny, nx] = dist[y, x] + 1
                        cola.append((nx, ny))
            return dist

        def union(zona, u, v):
            """Un punto P de la zona con recta libre u->P y P->v (o None)."""
            x0, x1 = sorted((u[0], v[0])); y0, y1 = sorted((u[1], v[1]))
            r = int(round(0.6 / g.res))
            ys, xs = np.nonzero(zona[max(0, y0 - r):y1 + r + 1, max(0, x0 - r):x1 + r + 1])
            xs = xs + max(0, x0 - r); ys = ys + max(0, y0 - r)
            if not len(xs):
                return None
            largo = np.hypot(xs - u[0], ys - u[1]) + np.hypot(v[0] - xs, v[1] - ys)
            castigo = 2.0 * (tope - np.minimum(holgura[ys, xs], tope))
            for k in np.argsort(largo + castigo)[:400]:
                p = (int(xs[k]), int(ys[k]))
                if segmento_transitable(mascara, u, p) and segmento_transitable(mascara, p, v):
                    return p
            return None

        def unir(zona, puntos):
            """[a, b, c...] -> guías con a lo sumo un punto de unión entre consecutivos."""
            salida = [puntos[0]]
            for u, v in zip(puntos, puntos[1:]):
                if u == v:
                    continue
                if not segmento_transitable(mascara, u, v):
                    p = union(zona, u, v)
                    if p is None:
                        return None
                    salida.append(p)
                salida.append(v)
            return salida

        nuevas = []
        for k, (reg, corrida) in enumerate(corridas):
            celdas = [c for c, _ in corrida]
            entrada, salida = celdas[0], celdas[-1]
            zona = mascara & (regiones == reg)
            puntos = [entrada]
            if zona[entrada[1], entrada[0]] and zona[salida[1], salida[0]]:
                da, db = bfs_zona(zona, entrada), bfs_zona(zona, salida)
                minimo = da[salida[1], salida[0]]
                if minimo > 1:
                    ok = ((da > 0) & (db > 0) & (da + db <= minimo + max(2, 0.1 * minimo))
                          & (np.abs(da - db) <= max(2, minimo / 3)))
                    ys, xs = np.nonzero(ok)
                    if len(xs):
                        puntaje = (np.minimum(holgura[ys, xs], tope) * 4.0
                                   - (da[ys, xs] + db[ys, xs] - minimo) * 0.15
                                   - np.abs(da[ys, xs] - db[ys, xs]) * 0.05)
                        i = int(np.argmax(puntaje))
                        puntos.append((int(xs[i]), int(ys[i])))
            if len(puntos) == 1 and len(celdas) > 2:
                largos = np.cumsum([0.] + [math.dist(a, b) for a, b in zip(celdas, celdas[1:])])
                medio = 1 + int(np.argmin(np.abs(largos[1:-1] - largos[-1] / 2)))
                puntos.append(celdas[medio])
            puntos.append(salida)
            unidas = unir(mascara if k == 0 else zona, puntos)
            if unidas is None:
                unidas = list(dict.fromkeys(puntos))  # Nav2 resuelve los giros entre las anclas.
            portal_de = {c: p for c, p in corrida}
            for c in unidas:
                es_portal = portal_de.get(c, False) or (c == salida and k < len(corridas) - 1) \
                    or (c == entrada and k > 0)
                if nuevas and nuevas[-1][0] == c:
                    nuevas[-1] = (c, nuevas[-1][1] or es_portal)
                else:
                    nuevas.append((c, es_portal))
        return nuevas

    def alejar_guias(self, guias_celdas, inicio, fijar_ultima=False, mascara=None):
        """Corre cada guía (portales y puntos intermedios) hacia donde hay más lugar
        a las paredes y obstáculos, dentro de `radio_alejar_guia_m`, sin cambiar de
        sector y con línea recta transitable desde la guía anterior y hasta la
        siguiente. Así Nav2 no pasa rozando esquinas ni marcos de puertas."""
        g = self.grilla
        trans = mascara if mascara is not None else g.transitable
        holgura = self.holgura_m()
        objetivo = self.cfg['holgura_guia_m']
        radio = max(1, int(round(self.cfg['radio_alejar_guia_m'] / g.res)))
        regiones = self.sectorizador.regiones if self.sectorizador is not None else None
        disco = [(dx, dy) for dx in range(-radio, radio + 1) for dy in range(-radio, radio + 1)
                 if 0 < dx * dx + dy * dy <= radio * radio]
        salida = []
        anterior = tuple(inicio)
        for i, (celda, portal) in enumerate(guias_celdas):
            celda = tuple(celda)
            ultima = i == len(guias_celdas) - 1
            if celda == tuple(inicio) or (ultima and fijar_ultima):
                salida.append((celda, portal))
                anterior = celda
                continue
            siguiente = tuple(guias_celdas[i + 1][0]) if not ultima else None
            x, y = celda

            def largo(c):
                d = math.hypot(c[0] - anterior[0], c[1] - anterior[1])
                if siguiente is not None:
                    d += math.hypot(siguiente[0] - c[0], siguiente[1] - c[1])
                return d * g.res
            base = largo(celda)
            mejor, puntaje_mejor = celda, min(holgura[y, x], objetivo)
            for dx, dy in disco:
                c = (x + dx, y + dy)
                if not g.dentro(c[1], c[0]) or not trans[c[1], c[0]]:
                    continue
                if regiones is not None and regiones[c[1], c[0]] != regiones[y, x]:
                    continue
                puntaje = min(holgura[c[1], c[0]], objetivo) - 0.3 * (largo(c) - base)
                if puntaje <= puntaje_mejor + 0.02:
                    continue
                if not segmento_transitable(trans, anterior, c):
                    continue
                if siguiente is not None and not segmento_transitable(trans, c, siguiente):
                    continue
                mejor, puntaje_mejor = c, puntaje
            salida.append((mejor, portal))
            anterior = mejor
        return salida

    def ruta_a_destino(self, pose, x, y, yaw):
        """También guía las vistas NBV y el regreso global alrededor de paredes."""
        g = self.grilla
        inicio = g.a_celda(*pose[:2])[1::-1]
        meta = g.a_celda(x, y)[1::-1]
        if not g.dentro(meta[1], meta[0]):
            return [], 0.0
        def planificar(seguro):
            Exploracion.bloquear_pasos_fallidos(self, seguro, inicio, meta)
            if not seguro[meta[1], meta[0]]:
                return []
            if self.sectorizador is not None:
                return self.sectorizador.ruta_hacia(
                    seguro, inicio, meta, self.cfg['paso_maximo_portal_m'])
            if not g.dentro(inicio[1], inicio[0]) or not seguro[inicio[1], inicio[0]]:
                return []
            return ruta_por_portales(seguro, np.zeros_like(seguro, dtype=np.int32), {},
                                     inicio, meta, g.res, self.cfg['paso_maximo_portal_m'])
        _ruta, pasos, distancia, _seguro, _margen = self.seleccionar_ruta_portales(
            pose, planificar, (x, y, yaw))
        return pasos, distancia

    def ganancia_en(self, fila, col):
        f, c, vis = self.rayos_desde(fila, col)
        nuevo = (vis & self.grilla.libre[f, c] & ~self.grilla.bloques_conocidos[f, c]
                 & ~self.visto[f, c])
        sectorizador = getattr(self, 'sectorizador', None)
        if sectorizador is not None:
            nuevo &= sectorizador.mascara_observacion()[f, c]
        if not nuevo.any():
            return 0.0, 0.0
        # Varios rayos atraviesan la misma celda, sobre todo cerca del robot.
        # Contar celdas únicas evita sobrevalorar rincones y vistas muy próximas.
        _, inversa = np.unique(f[nuevo] * self.grilla.ancho + c[nuevo], return_inverse=True)
        rayos = np.broadcast_to(np.arange(self.nrayos)[:, None], nuevo.shape)[nuevo]
        pesos = np.zeros(int(inversa.max()) + 1)
        np.maximum.at(pesos, inversa, np.broadcast_to(self.peso_rayo, nuevo.shape)[nuevo])
        presencia = np.zeros((self.nrayos, len(pesos)), dtype=np.int16)
        presencia[rayos, inversa] = 1
        w = self.ventana
        extendida = np.concatenate((presencia, presencia[:w - 1]), axis=0)
        acum = np.vstack((np.zeros((1, len(pesos)), dtype=np.int16),
                          np.cumsum(extendida, axis=0, dtype=np.int16)))
        visibles = acum[w:w + self.nrayos] - acum[:self.nrayos] > 0
        porciones = (visibles * pesos[None, :]).sum(axis=1)
        i = int(np.argmax(porciones))
        yaw = -math.pi + (i + (w - 1) / 2.0) * (2.0 * math.pi / self.nrayos)
        return float(porciones[i]), ang_norm(yaw)

    # ======================================================================= plazo
    def tiempo_transcurrido(self):
        return time.monotonic() - self.t_mapa if self.t_mapa else 0.0

    def velocidad_media(self):
        v = self.recorrido / self.t_moviendo if self.t_moviendo > 20.0 else self.cfg['velocidad_estimada']
        return min(0.50, max(0.12, v))

    def campo_casa(self):
        """Distancias por camino a la largada (se recalcula cada 5 s por obstáculos nuevos)."""
        ahora = time.monotonic()
        if self.dist_casa is None or ahora - self.t_dist_casa > 5.0:
            f, c = self.grilla.a_celda(self.largada[0], self.largada[1])
            self.dist_casa = bfs(self.grilla.transitable, f, c)
            self.t_dist_casa = ahora
        return self.dist_casa

    def segundos_para_volver(self, x, y):
        d = self.campo_casa()
        g = self.grilla
        f, c = g.a_celda(x, y)
        if d is None or not g.dentro(f, c) or d[f, c] < 0:
            metros = 1.4 * math.hypot(x - self.largada[0], y - self.largada[1])
        else:
            metros = d[f, c] * g.res
        return 1.3 * metros / self.velocidad_media() + 5.0

    def tiempo_restante(self):
        return self.cfg['duracion_corrida_s'] - self.tiempo_transcurrido()

    # ========================================================================= NBV
    def elegir_vista(self, pose):
        g = self.grilla
        sectorizador = getattr(self, 'sectorizador', None)
        if (sectorizador is not None
                and not (sectorizador.mascara_observacion() & ~self.visto).any()):
            return None, []
        fila_r, col_r = g.a_celda(pose[0], pose[1])
        dist = bfs(g.transitable, fila_r, col_r)
        if dist is None:
            return None, []
        casa = self.campo_casa()
        v = self.velocidad_media()
        cam = self.pose_camara()
        desfase_camara = ang_norm(cam[2] - pose[2]) if cam is not None else 0.0
        presupuesto = self.tiempo_restante() - self.cfg['margen_regreso_s']
        paso_base = max(1, int(round(self.cfg['paso_candidatos'] / g.res)))
        intentos = [(paso_base, self.cfg['ganancia_minima'])]
        if sectorizador is not None:
            # Para terminar un sector pequeño, no abandonar las últimas vistas
            # sólo porque ninguna suma 25 celdas nuevas en la grilla gruesa.
            intentos.append((max(1, paso_base // 2), 1))
            intentos.append((1, 1))
        preferida = g.candidatable
        if hasattr(g, 'bloqueo'):
            preferida = g.candidatable & ~dilatar(
                g.bloqueo | g.obstaculos_extra,
                self.cfg.get('margen_parada_portal_m', 0.22) / g.res)
        holgura = self.holgura_m() if hasattr(self, 'cfg') and 'holgura_guia_m' in self.cfg else None
        objetivo_holgura = self.cfg.get('holgura_guia_m', 0.30)
        peso_holgura = self.cfg.get('peso_holgura_nbv', 0.6)
        mejores = []
        for paso, ganancia_minima in intentos:
            for mascara_metas in (preferida, g.candidatable):
                for fila in range(paso // 2, g.alto, paso):
                    for col in range(paso // 2, g.ancho, paso):
                        # La meta debe ser segura y alcanzable; el piso observado
                        # puede estar del otro lado de un paso que no admite el robot.
                        if not mascara_metas[fila, col] or dist[fila, col] < 0:
                            continue
                        x, y = g.a_mundo(fila, col)
                        if any(math.hypot(x - bx, y - by) < self.cfg['radio_lista_negra']
                               for bx, by in self.lista_negra):
                            continue
                        d_m = dist[fila, col] * g.res
                        # ¿Llego, miro y vuelvo a tiempo?
                        d_casa = casa[fila, col] * g.res if casa is not None and casa[fila, col] >= 0 else d_m
                        if 1.3 * (d_m + d_casa) / v + 8.0 > presupuesto:
                            continue
                        gan, yaw = self.ganancia_en(fila, col)
                        if gan < ganancia_minima:
                            continue
                        yaw = ang_norm(yaw - desfase_camara)
                        giro = abs(ang_norm(yaw - pose[2]))
                        util = (gan * math.exp(-self.cfg['lambda_distancia'] * d_m)
                                / (1.0 + self.cfg['peso_giro_nbv'] * giro / math.pi))
                        if holgura is not None:
                            # Mismo punto de vista, pero lejos de la pared: Nav2 llega
                            # sin rozar y el robot gira sin "Collision Ahead".
                            util *= (1.0 - peso_holgura) + peso_holgura * min(
                                1.0, float(holgura[fila, col]) / objetivo_holgura)
                        mejores.append((util, x, y, yaw, gan, d_m))
                if mejores:
                    break
            if mejores:
                break
        if not mejores:
            return None, []
        mejores.sort(reverse=True)
        return mejores[0], mejores[:40]

    def elegir_confirmacion(self, pose):
        """Caja roja vista pero sin reportar: punto a 0.6-1.2 m con vista libre para
        que el detector la confirme de cerca. None si no hay o no da el tiempo."""
        if (not self.cfg['confirmar_cajas'] or not self.pendientes_vision or self.t_pendientes is None
                or time.monotonic() - self.t_pendientes > 3.0):
            return None
        g = self.grilla
        fila, col = g.a_celda(pose[0], pose[1])
        dist = bfs(g.transitable, fila, col)
        if dist is None:
            return None
        for bx, by, _n in sorted(self.pendientes_vision, key=lambda p: math.hypot(p[0] - pose[0], p[1] - pose[1])):
            if not self.objetos_fuera_mapa.disponible((bx, by), time.monotonic()):
                continue
            registro = next((r for r in self.intentos_confirmar if math.hypot(r[0] - bx, r[1] - by) < 0.35), None)
            if registro is not None and registro[2] >= self.cfg['intentos_confirmacion']:
                continue
            # La caja aparece en el costmap: ese obstáculo no tapa la vista. Las
            # paredes del mapa sí (nunca se "limpian").
            vista = g.libre & ~(g.obstaculos_costmap & ~self._cerca_de(bx, by))
            punto = punto_para_mirar(g.candidatable, vista, dist, g.res, (g.ox, g.oy), (bx, by))
            if punto is None:
                continue
            x, y, yaw, d_m = punto
            if (self.tiempo_restante() - self.cfg['margen_regreso_s'] - self.cfg['tiempo_mirar_caja_s']
                    < 1.3 * d_m / self.velocidad_media() + self.segundos_para_volver(x, y)):
                continue
            if registro is None:
                registro = [bx, by, 0]
                self.intentos_confirmar.append(registro)
            registro[2] += 1
            return x, y, yaw, d_m, (bx, by)
        return None

    def _cerca_de(self, bx, by, radio=0.30):
        """Máscara de las celdas a menos de `radio` de la caja (la caja misma no es pared)."""
        g = self.grilla
        filas = (np.arange(g.alto) + 0.5) * g.res + g.oy
        cols = (np.arange(g.ancho) + 0.5) * g.res + g.ox
        return ((filas[:, None] - by) ** 2 + (cols[None, :] - bx) ** 2) < radio ** 2

    # ==================================================================== acciones
    def dividir_tramo(self, pasos):
        """Un sector por acción: hasta cinco guías y la entrada del próximo portal."""
        if not pasos:
            return pasos, []

        def acotar(tramo):
            if len(tramo) <= 5:
                return tramo
            # Última barrera: aunque un llamador entregue el trazado denso,
            # sólo enviar entrada, mitad por distancia recorrida y salida.
            largos = np.cumsum([0.] + [math.dist(a[:2], b[:2]) for a, b in zip(tramo, tramo[1:])])
            medio = 1 + int(np.argmin(np.abs(largos[1:-1] - largos[-1] / 2)))
            return [tramo[0], tramo[medio], tramo[-1]]

        if self.sectorizador is None:
            return acotar(pasos), []
        regiones = self.sectorizador.regiones

        def region(punto):
            f, c = self.grilla.a_celda(*punto[:2])
            return int(regiones[f, c]) if self.grilla.dentro(f, c) else -1

        actual = region(pasos[0])
        for i in range(1, len(pasos)):
            if region(pasos[i]) != actual:
                # El cruce pertenece también al comienzo del tramo siguiente.
                # Mantenerlo en la cola permite continuar desde esa entrada,
                # sin recuperar puntos de sectores ya recorridos.
                if self.cfg.get('guias_portal_y_medio', False):
                    # Portal + medio de esta zona; el portal siguiente abre la próxima acción.
                    return acotar(pasos[:i]), pasos[i:]
                resto = pasos[i:] if i < len(pasos) - 1 else []
                return acotar(pasos[:i]) + [pasos[i]], resto
        return acotar(pasos), []

    def enviar_goal(self, x, y, yaw, ganancia, d_m, etiqueta='vista', pasos=None):
        heredado = self.heredado_vivo()
        if heredado and not pasos:
            pasos = [(x, y, yaw)]          # mismo servidor que el goal en marcha: lo reemplaza sin frenar
        pasos, continuacion = self.dividir_tramo(pasos)
        ruta_continua = bool(pasos)
        cliente = self.nav_ruta if ruta_continua else self.nav
        if not cliente.server_is_ready():
            accion = 'navigate_through_poses' if ruta_continua else 'navigate_to_pose'
            self.get_logger().warn(f'Nav2 ({accion}) todavía no está listo')
            return False
        self.goal_id += 1
        gid = self.goal_id
        pose = self.pose_robot()
        distancias_guias = []
        d_tramo = 0.0
        anterior = pose[:2] if pose is not None else ((pasos[0][:2]) if pasos else (x, y))
        for punto in pasos or []:
            d_tramo += math.dist(anterior, punto[:2])
            distancias_guias.append(d_tramo)
            anterior = punto[:2]
        if not pasos:
            d_tramo = d_m
        if ruta_continua:
            goal = NavigateThroughPoses.Goal()
            goal.behavior_tree = self.bt_portales
            goal.poses = []
            for px, py, rumbo in pasos:
                punto = PoseStamped()
                punto.header.frame_id = self.grilla.frame
                punto.header.stamp = self.get_clock().now().to_msg()
                punto.pose.position.x, punto.pose.position.y = px, py
                punto.pose.orientation.z, punto.pose.orientation.w = yaw_a_quat(rumbo)
                goal.poses.append(punto)
        else:
            goal = NavigateToPose.Goal()
            goal.pose = PoseStamped()
            goal.pose.header.frame_id = self.grilla.frame
            goal.pose.header.stamp = self.get_clock().now().to_msg()
            goal.pose.pose.position.x, goal.pose.pose.position.y = x, y
            goal.pose.pose.orientation.z, goal.pose.pose.orientation.w = yaw_a_quat(yaw)
        self.goal_actual = dict(x=x, y=y, yaw=yaw, ganancia0=max(ganancia, 1.0), t0=time.monotonic(),
                                p_inicio=pose, continuacion=continuacion,
                                meta_tramo=pasos[-1] if pasos else (x, y, yaw),
                                distancias_guias=distancias_guias, mejor_avance=None,
                                t_progreso=self.get_clock().now().nanoseconds * 1e-9,
                                t_max=20.0 + 2.5 * d_tramo / max(0.05, self.cfg['velocidad_estimada']),
                                etiqueta=etiqueta, pasos=pasos,
                                pasos_restantes=len(pasos) if pasos else 0)
        self.hist.clear()
        if heredado:
            self.goal_heredado = None      # Nav2 lo reemplaza por este (preempción, sin frenar)
        if ruta_continua:
            futuro = cliente.send_goal_async(
                goal, feedback_callback=lambda mensaje, gid=gid: self.cb_feedback_ruta(mensaje, gid))
        else:
            futuro = cliente.send_goal_async(goal)
        self.solicitud_goal = futuro
        futuro.add_done_callback(lambda f, gid=gid: self.cb_goal_aceptado(f, gid))
        local = self.goal_actual['meta_tramo']
        self.get_logger().info(
            f'[{etiqueta}] goal #{gid}: tramo hasta ({local[0]:.2f}, {local[1]:.2f}), '
            f'destino final ({x:.2f}, {y:.2f}); '
            f'pasos={len(pasos) if pasos else 1} yaw={math.degrees(yaw):.0f}° gan={ganancia:.0f} '
            f'dist tramo≈{d_tramo:.1f} m; guías pendientes={len(continuacion)} | cobertura {100 * self.cobertura():.1f}% | '
            f't={self.tiempo_transcurrido():.0f}s')
        if pasos:
            self.get_logger().info('Guías Nav2 entrada/medio/salida: ' +
                                   ' -> '.join(f'({px:.2f}, {py:.2f})' for px, py, _ in pasos))
        return True

    def continuar_tramo(self, ga):
        pasos = ga.get('continuacion', [])
        if not pasos:
            return False
        pose = self.pose_robot()
        distancia = math.dist(pose[:2], pasos[0][:2]) if pose is not None else 0.0
        distancia += sum(math.dist(a[:2], b[:2]) for a, b in zip(pasos, pasos[1:]))
        if pose is not None:
            proxima = next((p for p in pasos if math.dist(p[:2], pose[:2]) >= 0.25), pasos[-1])
            self.dir_ruta = self.direccion_de_ruta(pose, proxima[0], proxima[1])
        return self.enviar_goal(ga['x'], ga['y'], ga['yaw'], ga['ganancia0'], distancia,
                                etiqueta=ga['etiqueta'], pasos=pasos)

    def anticipar_siguiente_tramo(self, pose):
        """Actualiza la misma acción cerca de la salida para atravesarla sin pausa."""
        ga = self.goal_actual
        if (ga is None or not ga.get('continuacion') or self.goal_handle is None
                or self.solicitud_goal is not None or ga['pasos_restantes'] > 2):
            return False
        meta = ga['meta_tramo']
        if math.dist(pose[:2], meta[:2]) > self.cfg['anticipacion_portal_m']:
            return False
        inicio = self.grilla.a_celda(*pose[:2])[1::-1]
        destino = self.grilla.a_celda(*meta[:2])[1::-1]
        if not segmento_transitable(self.grilla.transitable, inicio, destino):
            return False
        return self.continuar_tramo(ga)

    def cb_feedback_ruta(self, mensaje, gid):
        if gid == self.goal_id and self.goal_actual is not None:
            self.goal_actual['pasos_restantes'] = min(
                self.goal_actual['pasos_restantes'], mensaje.feedback.number_of_poses_remaining)
            restante = getattr(mensaje.feedback, 'distance_remaining', None)
            if restante is not None and math.isfinite(restante) and restante > 0:
                self.goal_actual['restante_nav2'] = float(restante)

    def siguiente_guia(self, ga):
        pasos = ga.get('pasos') if ga else None
        if not pasos:
            return ga['x'], ga['y']
        indice = max(0, min(len(pasos) - 1, len(pasos) - ga['pasos_restantes']))
        return pasos[indice][:2]

    def direccion_de_ruta(self, pose, gx, gy, pasos=10):
        """Rumbo (marco map) del primer tramo (~0.5 m) del camino más corto hacia (gx, gy)."""
        g = self.grilla
        dist = bfs(g.transitable, *g.a_celda(gx, gy))
        if dist is None:
            return None
        f, c = g.a_celda(pose[0], pose[1])
        if not g.dentro(f, c) or dist[f, c] < 0:
            return math.atan2(gy - pose[1], gx - pose[0])
        for _ in range(pasos):
            mejor = None
            for df in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nf, nc = f + df, c + dc
                    if (df and dc and g.dentro(nf, nc)
                            and not (g.transitable[f, nc] and g.transitable[nf, c])):
                        continue
                    if (df or dc) and g.dentro(nf, nc) and 0 <= dist[nf, nc] < dist[f, c] and \
                            (mejor is None or dist[nf, nc] < dist[mejor]):
                        mejor = (nf, nc)
            if mejor is None:
                break
            f, c = mejor
        wx, wy = g.a_mundo(f, c)
        if math.hypot(wx - pose[0], wy - pose[1]) < 0.05:
            return math.atan2(gy - pose[1], gx - pose[0])
        return math.atan2(wy - pose[1], wx - pose[0])

    def encarar_y_enviar(self, x, y, yaw, ganancia, d_m, etiqueta, estado_destino, pasos=None):
        """Si el camino arranca "para atrás" (>100° del rumbo del robot), primero gira en el
        lugar para encararlo y recién después manda el goal."""
        pose = self.pose_robot()
        if pose is None:
            return False
        if pasos is None and pose is not None:
            fila, col = self.grilla.a_celda(*pose[:2])
            if not self.grilla.dentro(fila, col) or not self.grilla.transitable[fila, col]:
                self.get_logger().warn('El robot está dentro del margen de un obstáculo: me separo antes de calcular otra ruta')
                self.cancelar_actual()
                self.goal_actual = None
                self.n_atascos += 1
                self.dir_ruta = None
                self.iniciar_escape(REGRESAR if estado_destino == VOLVIENDO else ELIGIENDO)
                return True
            pasos, d_guias = self.ruta_a_destino(pose, x, y, yaw)
            if pasos:
                d_m = d_guias
            else:
                # El trazador propio (portales con margen) no encontró guías, pero el
                # destino sí es alcanzable (lo eligió el BFS): que planifique Nav2.
                # Antes se lo descartaba y, en zonas angostas, el robot se quedaba
                # quieto descartando un destino tras otro.
                self.get_logger().info(f'[{etiqueta}] sin guías por portales hasta ({x:.2f}, {y:.2f}): '
                                       'le paso el destino directo a Nav2')
                pasos = None
        primero = next((p for p in pasos or [] if math.dist(p[:2], pose[:2]) >= 0.25),
                       pasos[-1] if pasos else (x, y))
        self.dir_ruta = self.direccion_de_ruta(pose, primero[0], primero[1]) if pose else None
        if pose is not None and self.dir_ruta is not None and self.spin_cli.server_is_ready():
            giro = ang_norm(self.dir_ruta - pose[2])
            sin_lugar = False
            if abs(giro) > math.radians(getattr(self, 'cfg', {}).get('giro_previo_deg', 135.0)):
                try:
                    holgura = self.holgura_giro()
                except Exception:  # noqa: BLE001
                    holgura = None
                # Sin lugar para girar (pasillo alrededor de una caja), el giro previo
                # terminaba en "Muy cerca de una pared para girar" + desencaje: el X3 es
                # omnidireccional, mejor dejar que Nav2 arranque sin girar.
                sin_lugar = holgura is not None and holgura < getattr(self, 'cfg', {}).get('holgura_minima_giro_previo_m', 0.03)
                if sin_lugar:
                    self.get_logger().info(
                        f'[{etiqueta}] la ruta arranca a {math.degrees(giro):.0f}° pero no hay lugar para girar '
                        f'(holgura {100 * holgura:.0f} cm): no giro, Nav2 arranca como está')
            if not sin_lugar and abs(giro) > math.radians(getattr(self, 'cfg', {}).get('giro_previo_deg', 135.0)):
                self.get_logger().info(
                    f'[{etiqueta}] la ruta arranca a {math.degrees(giro):.0f}° de donde miro: giro primero')
                self.pendiente = (x, y, yaw, ganancia, d_m, etiqueta, estado_destino, pasos)
                self.girar(giro, ENVIAR_PENDIENTE)
                return True
        if self.enviar_goal(x, y, yaw, ganancia, d_m, etiqueta=etiqueta, pasos=pasos):
            self.estado = estado_destino
            return True
        return False

    def cb_goal_aceptado(self, fut, gid):
        handle = fut.result()
        if gid != self.goal_id:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        self.solicitud_goal = None
        if not handle.accepted:
            if self.goal_handle is not None:
                self.goal_handle.cancel_goal_async()
            self.get_logger().warn(f'goal #{gid} rechazado por Nav2')
            self.recuperar_nav2()
            return
        self.goal_handle = handle
        handle.get_result_async().add_done_callback(
            lambda f, gid=gid: gid == self.goal_id and self.resultado_goal(f.result().status))

    def resultado_goal(self, status):
        self.goal_handle = None
        ok = status == GoalStatus.STATUS_SUCCEEDED
        ga = self.goal_actual
        o = getattr(self, 'objeto_observado', None)
        if (self.estado == OBSERVANDO_OBJETO and o is not None
                and o['fase'] == 'acercando'):
            self.goal_actual = None
            if ok:
                o.update(fase='frenando', quieto_desde=None, desde=None, imagenes=0)
                self.pub_cmd.publish(Twist())
            else:
                self.terminar_observacion_objeto(False, 'Nav2 no alcanzó la posición de observación')
            return
        if (ok and self.estado in (NAVEGANDO, VOLVIENDO)
                and ga is not None and ga.get('continuacion')):
            if not self.continuar_tramo(ga):
                self.goal_actual = None
                self.estado = REGRESAR if self.estado == VOLVIENDO else ELIGIENDO
            return
        if self.estado == VOLVIENDO:
            self.goal_actual = None
            if ok:
                self.ruta_regreso.clear()
                if self.sectorizador is None:
                    self.iniciar_aproximacion()
                else:
                    pose = self.pose_robot()
                    if pose is not None and math.hypot(
                            pose[0] - self.largada[0], pose[1] - self.largada[1]) <= 0.6:
                        self.iniciar_aproximacion()
                    else:
                        self.reintentar_regreso()
            else:
                self.reintentar_regreso()
            return
        if self.estado != NAVEGANDO:
            return
        if ok:
            self.al_llegar()
        else:
            self.fallo_goal()

    def se_movio_desde(self, ga, minimo=0.10):
        pose = self.pose_robot()
        p0 = ga.get('p_inicio') if ga else None
        return pose is None or p0 is None or math.hypot(pose[0] - p0[0], pose[1] - p0[1]) >= minimo

    def fallo_goal(self):
        ga = self.goal_actual
        if ga is not None:
            if ga.get('etiqueta', '').startswith('portal sector'):
                self.portales_fallidos.add(self.grilla.a_celda(*self.siguiente_guia(ga))[1::-1])
                self.ruta_sector.clear()
                self.get_logger().warn('Ruta por portales falló: recalculo desde la posición actual')
            else:
                self.lista_negra.append((ga['x'], ga['y']))
                self.get_logger().warn(f"goal ({ga['x']:.2f}, {ga['y']:.2f}) falló: lista negra")
        self.goal_actual = None
        self.estado = ELIGIENDO
        if ga is not None and self.registrar_fallo_localizacion(ELIGIENDO):
            return
        if ga is not None and not self.se_movio_desde(ga):
            # Nav2 rechazó el goal sin mover al robot: casi seguro está trabado
            # contra algo (DWB: "No valid trajectories"). Desencajar ya.
            inflacion = getattr(self, 'inflacion', None)
            if inflacion is not None:
                inflacion.reducir(self.lookup(self.cfg['odom_frame'], self.cfg['base_frame']),
                                  'Nav2 falló sin desplazamiento')
            self.n_atascos += 1
            self.get_logger().warn('Nav2 falló sin mover al robot -> me desencajo')
            self.limpiar_costmaps()
            self.iniciar_escape(ELIGIENDO)

    def al_llegar(self, ga=None):
        ga = ga or self.goal_actual
        self.goal_actual = None
        self.estado = ELIGIENDO
        if ga is None:
            return
        if ga.get('etiqueta', '').startswith('portal sector'):
            self.ruta_sector.clear()
            pose = self.pose_robot()
            if (pose is not None and self.sectorizador is not None
                    and not self.en_sector(pose)):
                self.portales_fallidos.add(self.grilla.a_celda(ga['x'], ga['y'])[1::-1])
                self.get_logger().warn('Nav2 terminó fuera del sector: descarto esa entrada y busco otra más interior')
                self.registrar_fallo_localizacion(ELIGIENDO)
            else:
                self.n_atascos = 0
                self.fallos_localizacion.clear()
            return
        self.n_atascos = 0
        self.fallos_localizacion.clear()
        self.lista_negra.append((ga['x'], ga['y']))   # visitado: no volver a elegirlo
        pose = self.pose_robot()
        giro = ang_norm(ga['yaw'] - pose[2]) if pose is not None else 0.0
        if ga.get('etiqueta', '').startswith('confirmar'):
            # Quedarse mirando la caja: el detector necesita varias imágenes de cerca.
            tolerancia = math.radians(self.cfg['tolerancia_mirar_caja_deg'])
            self.espera_mirar_hasta = (time.monotonic() + self.cfg['tiempo_mirar_caja_s']
                                       + (abs(giro) / 0.8 if abs(giro) > tolerancia else 0.0))
            if pose is not None and abs(giro) > tolerancia:   # dentro de ±25° la caja ya está en cuadro
                self.girar(giro)
            return
        if self.cfg['girar_al_llegar'] and pose is not None:
            if abs(giro) > math.radians(self.cfg['tolerancia_giro_llegada_deg']):
                self.girar(giro)

    def holgura_giro(self):
        """Metros libres alrededor del círculo que barre el chasis al girar en el lugar
        (esquinas a 0.21 m del centro). Negativo: alguna esquina tocaría algo."""
        puntos = self.puntos_scan()
        if puntos is None or puntos[0].size == 0:
            return None
        radio = math.hypot(self.ATRAS, self.MEDIO_ANCHO) + 0.01
        return float(np.min(np.hypot(*puntos))) - radio

    def girar(self, angulo, siguiente=ELIGIENDO):
        if not self.spin_cli.server_is_ready():
            self.recuperar_nav2(siguiente=siguiente, giro=angulo)
            return
        # Pegado a una pared, Spin de Nav2 aborta ("Collision Ahead") y el robot no
        # gira: primero separarse ~10 cm hacia donde hay lugar y después girar.
        holgura = self.holgura_giro() if self.estado != LOCALIZANDO else None
        if holgura is not None and holgura < 0.02 and self.giro_pendiente is None:
            self.giro_pendiente = (angulo, siguiente)
            self.get_logger().info(f'Muy cerca de una pared para girar (holgura {100 * holgura:.0f} cm): '
                                   'me separo antes')
            self.iniciar_escape(siguiente)
            self.escape['dist'] = 0.10
            return
        g = Spin.Goal()
        g.target_yaw = float(angulo)
        g.time_allowance = Duration(sec=10)
        self.comportamiento(self.spin_cli, g, siguiente)

    def comportamiento(self, cliente, goal, siguiente):
        """Corre una acción de behavior_server (Spin) y al terminar (bien o mal) pasa a `siguiente`."""
        if getattr(self, 'goal_heredado', None) is not None:
            self.cancelar_actual()
        self.estado = COMPORTAMIENTO
        self.goal_id += 1
        gid = self.goal_id
        self.hist.clear()
        self.hist_global.clear()

        def fin(_f=None, gid=gid):
            if gid == self.goal_id and self.estado == COMPORTAMIENTO:
                self.goal_handle = None
                if siguiente == OBSERVANDO_OBJETO and self.objeto_observado is not None:
                    self.objeto_observado['giro_resultado'] = _f.result().status
                self.estado = siguiente

        def aceptado(f, gid=gid):
            h = f.result()
            if gid != self.goal_id:
                if h.accepted:
                    h.cancel_goal_async()
                return
            self.solicitud_goal = None
            if not h.accepted:
                self.get_logger().warn('Acción de giro rechazada: verifico la activación de Nav2')
                self.recuperar_nav2(siguiente=siguiente, giro=goal.target_yaw)
                return
            self.goal_handle = h
            h.get_result_async().add_done_callback(fin)

        self.solicitud_goal = cliente.send_goal_async(goal)
        self.solicitud_goal.add_done_callback(aceptado)

    def soltar_sin_frenar(self):
        """Al salir de una zona: en vez de cancelar (Nav2 frena en seco y recién después
        arranca hacia el destino nuevo), deja andar el goal actual de NavigateThroughPoses
        hasta que el próximo goal lo reemplace (preempción: el controlador sigue con la
        ruta nueva sin detenerse). Devuelve False si no se puede (otro tipo de acción)."""
        ga = self.goal_actual
        if (not self.cfg.get('cambio_sin_frenar', True) or ga is None or not ga.get('pasos')
                or (self.goal_handle is None and self.solicitud_goal is None)):
            return False
        self.goal_id += 1     # su resultado ya no cuenta
        self.goal_heredado = dict(handle=self.goal_handle, solicitud=self.solicitud_goal,
                                  t=time.monotonic())
        self.goal_handle = None
        self.solicitud_goal = None
        return True

    def heredado_vivo(self):
        h = getattr(self, 'goal_heredado', None)
        if h is None:
            return False
        handle = h['handle']
        if handle is not None and handle.status in (GoalStatus.STATUS_SUCCEEDED, GoalStatus.STATUS_CANCELED,
                                                    GoalStatus.STATUS_ABORTED):
            self.goal_heredado = None
            return False
        return True

    def cancelar_actual(self):
        """Espera el resultado final de todas las acciones que aún puedan mover el robot."""
        self.goal_id += 1     # cualquier resultado pendiente queda obsoleto
        anterior = self.cancelacion_en_curso
        handle, solicitud = self.goal_handle, self.solicitud_goal
        if getattr(self, 'goal_heredado', None) is not None:
            if handle is None and solicitud is None:
                handle, solicitud = self.goal_heredado['handle'], self.goal_heredado['solicitud']
            self.goal_heredado = None
        self.solicitud_goal = None
        self.goal_handle = None
        if handle is None and solicitud is None:
            return anterior if anterior is not None and not anterior.done() else None

        barrera = Future()
        self.cancelacion_en_curso = barrera
        # El centinela impide completar la barrera si un callback se ejecuta
        # inmediatamente, antes de registrar las otras acciones pendientes.
        pendientes = 1

        def completar():
            nonlocal pendientes
            pendientes -= 1
            if pendientes == 0 and not barrera.done():
                barrera.set_result(None)

        def fallo(error):
            # Sin resultado definitivo no sabemos si dejó de publicar velocidad.
            # Mantener pendiente la barrera conserva detenida la recuperación.
            self.get_logger().warn(f'No pude confirmar el fin de una acción cancelada: {error}; mantengo la espera')

        def resultado_final(futuro):
            try:
                resultado = futuro.result()
                if resultado.status not in (GoalStatus.STATUS_SUCCEEDED,
                                             GoalStatus.STATUS_CANCELED,
                                             GoalStatus.STATUS_ABORTED):
                    fallo(f'estado final desconocido {resultado.status}')
                    return
            except Exception as error:  # noqa: BLE001
                fallo(error)
                return
            completar()

        def cancelar_handle(h):
            nonlocal pendientes
            pendientes += 1
            try:
                h.get_result_async().add_done_callback(resultado_final)
            except Exception as error:  # noqa: BLE001
                fallo(error)
            try:
                h.cancel_goal_async()
            except Exception as error:  # noqa: BLE001
                fallo(error)

        def cancelar_al_aceptar(futuro):
            try:
                h = futuro.result()
                if h.accepted:
                    cancelar_handle(h)
            except Exception as error:  # noqa: BLE001
                fallo(error)
                return
            completar()

        if anterior is not None and not anterior.done():
            pendientes += 1

            def fin_anterior(futuro):
                try:
                    futuro.result()
                except Exception as error:  # noqa: BLE001
                    fallo(error)
                    return
                completar()

            try:
                anterior.add_done_callback(fin_anterior)
            except Exception as error:  # noqa: BLE001
                fallo(error)
        if handle is not None:
            cancelar_handle(handle)
        if solicitud is not None:
            pendientes += 1
            try:
                solicitud.add_done_callback(cancelar_al_aceptar)
            except Exception as error:  # noqa: BLE001
                fallo(error)
        completar()  # liberar el centinela después de registrar todas las esperas
        return barrera

    def limpiar_costmaps(self):
        for cli in self.clear_cli:
            if cli.service_is_ready():
                cli.call_async(ClearEntireCostmap.Request())

    # ================================================================ anti-atascos
    def actualizar_inflacion(self):
        """Aviso temprano; el anti-atascos completo sigue disponible si no alcanza."""
        forzar_normal = self.estado in (ESPERANDO_MAPA, LOCALIZANDO, ACTIVANDO_NAV2,
            RELOCALIZANDO, APROXIMANDO, FINALIZANDO, TERMINADO)
        navegando = self.estado in (NAVEGANDO, VOLVIENDO) and self.goal_handle is not None
        pose = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame']) if navegando else None
        ga = self.goal_actual or {}
        restante = ga.get('restante_nav2')
        self.inflacion.paso(pose, navegando, progreso=-restante if restante is not None else None,
            meta=self.goal_id, restaurar=forzar_normal or self.loc_mala_desde is not None)

    def detectar_atasco(self, pose):
        t = self.get_clock().now().nanoseconds * 1e-9      # sim time en Gazebo
        T = self.cfg['tiempo_atascado']
        ga = self.goal_actual
        if ga is not None and ga.get('pasos'):
            pasos = ga['pasos']
            indice = max(0, min(len(pasos) - 1, len(pasos) - ga['pasos_restantes']))
            avance = ga['distancias_guias'][indice] - math.dist(pose[:2], pasos[indice][:2])
            if ga.get('restante_nav2') is not None:
                # Con pocas guías el camino de Nav2 puede alejarse de la recta (rodear
                # una pared): el progreso se mide con lo que le falta según Nav2.
                avance = -ga['restante_nav2']
            mejor = ga['mejor_avance']
            if mejor is None or avance >= mejor + self.cfg['distancia_atascado']:
                ga['mejor_avance'], ga['t_progreso'] = avance, t
            elif t - ga['t_progreso'] >= T:
                self.get_logger().warn('Oscilación: hay movimiento pero no progreso hacia la próxima guía del tramo')
                return True
        self.hist.append((t, pose[0], pose[1], pose[2]))
        while self.hist and t - self.hist[0][0] > T + 2.0:
            self.hist.popleft()
        if not self.hist or t - self.hist[0][0] < T:
            return False
        ventana = [h for h in self.hist if t - h[0] <= T]
        if len(ventana) < 3:
            return False
        # Solo cuenta la traslación: girar en el lugar 6 s navegando tampoco es avanzar.
        desp = max(math.hypot(h[1] - pose[0], h[2] - pose[1]) for h in ventana)
        return desp < self.cfg['distancia_atascado']

    def vigilancia_global(self, pose):
        """Atasco aunque se cambie de goal seguido: `tiempo_atascado_global` s sin trasladarse 12 cm."""
        t = self.get_clock().now().nanoseconds * 1e-9
        self.hist_global.append((t, pose[0], pose[1]))
        T = self.cfg['tiempo_atascado_global']
        while self.hist_global and t - self.hist_global[0][0] > T + 2.0:
            self.hist_global.popleft()
        if t - self.hist_global[0][0] < T:
            return False
        ventana = [h for h in self.hist_global if t - h[0] <= T]
        return max(math.hypot(h[1] - pose[0], h[2] - pose[1]) for h in ventana) < self.cfg['distancia_atascado']

    def bloquear_pasos_fallidos(self, seguro, *libres, radio_m=0.25):
        """Quita de `seguro` los pasos donde ya se atascó, salvo cerca del robot o del destino."""
        g = self.grilla
        k2 = (radio_m / g.res) ** 2
        for x, y in getattr(self, 'portales_fallidos', ()):
            if not g.dentro(y, x):
                continue
            if any((x - lx) ** 2 + (y - ly) ** 2 <= k2 for lx, ly in libres):
                continue
            seguro[y, x] = False

    def marcar_paso_estrecho(self, pose):
        """Si se atascó en un lugar angosto, las próximas rutas no vuelven a pasar por ahí
        (mientras exista otro camino). En la corrida de las 14:59 intentó 3 veces el
        mismo paso de ~30 cm junto a la caja roja y perdió 150 s."""
        g = self.grilla
        f, c = g.a_celda(*pose[:2])
        if not g.dentro(f, c):
            return
        holgura = float(self.holgura_m()[f, c])
        if holgura >= self.cfg.get('holgura_paso_estrecho_m', 0.25):
            return
        k = max(1, int(round(self.cfg.get('radio_paso_fallido_m', 0.15) / g.res)))
        for df in range(-k, k + 1):
            for dc in range(-k, k + 1):
                if df * df + dc * dc <= k * k and g.dentro(f + df, c + dc):
                    self.portales_fallidos.add((c + dc, f + df))
        self.ruta_sector.clear()
        self.get_logger().warn(
            f'Paso angosto en ({pose[0]:.2f}, {pose[1]:.2f}) (holgura {100 * holgura:.0f} cm): '
            'las próximas rutas lo evitan si hay otro camino')

    def resolver_atasco(self, pose):
        self.n_atascos += 1
        volviendo = self.estado == VOLVIENDO
        ga = self.goal_actual
        if self.registrar_fallo_localizacion(REGRESAR if volviendo else ELIGIENDO):
            return
        inflacion = getattr(self, 'inflacion', None)
        if inflacion is not None:
            inflacion.reducir(self.lookup(self.cfg['odom_frame'], self.cfg['base_frame']), 'atasco confirmado')
        self.get_logger().warn(
            f"ATASCADO (sin avanzar) en ({pose[0]:.2f}, {pose[1]:.2f}) "
            f'(#{self.n_atascos} seguido) -> limpio costmaps y '
            + ('recalculo el regreso' if volviendo else 'cambio de punto de vista'))
        self.cancelar_actual()
        self.marcar_paso_estrecho(pose)
        if volviendo:
            self.ruta_regreso.clear()
        if ga is not None and not volviendo:
            if ga.get('etiqueta', '').startswith('portal sector'):
                self.portales_fallidos.add(self.grilla.a_celda(*self.siguiente_guia(ga))[1::-1])
                self.ruta_sector.clear()
            else:
                self.lista_negra.append((ga['x'], ga['y']))
        self.goal_actual = None
        self.limpiar_costmaps()
        # Desencajar con el láser: si el robot quedó pegado a algo, Nav2 (DWB,
        # Spin, BackUp) se niega a moverlo ("Collision Ahead"); acá se lo mueve
        # despacio hacia donde el LiDAR ve más espacio libre.
        self.iniciar_escape(REGRESAR if volviendo else ELIGIENDO)

    # ------------------------------------------------------------ desencajar
    # Chasis del X3 respecto de base_footprint (mismo footprint que Nav2).
    ADELANTE, ATRAS, MEDIO_ANCHO = 0.119, 0.181, 0.100
    ESQUINAS = np.array([[ADELANTE, MEDIO_ANCHO], [ADELANTE, -MEDIO_ANCHO],
                         [-ATRAS, -MEDIO_ANCHO], [-ATRAS, MEDIO_ANCHO]])

    def cb_scan(self, msg):
        self.scan = msg
        self.scans_objetos.append((msg, self.vel_giro))
        self.actualizar_objetos_lidar()

    def actualizar_objetos_lidar(self):
        """Procesa el barrido más reciente cuya TF ya llegó, sin esperas bloqueantes."""
        ahora = time.monotonic()
        if (self.grilla is None or not self.cfg['inspeccionar_objetos']
                or self.estado in (ESPERANDO_MAPA, LOCALIZANDO, ACTIVANDO_NAV2, RELOCALIZANDO, TERMINADO)
                or self.loc_mala_desde is not None or self.vel_giro > 0.5
                or ahora - self.t_scan_objetos < 0.2):
            return
        elegido = None
        for msg, giro in reversed(self.scans_objetos):
            stamp = Time.from_msg(msg.header.stamp).nanoseconds
            edad = (self.get_clock().now().nanoseconds - stamp) / 1e9
            if (not msg.header.frame_id or not stamp or stamp <= self.ultimo_stamp_scan_objetos
                    or not 0 <= edad <= self.cfg['timeout_scan_s'] or giro > 0.5):
                continue
            try:
                tf = self.tf_buffer.lookup_transform(self.grilla.frame, msg.header.frame_id,
                                                     Time.from_msg(msg.header.stamp)).transform
            except Exception:  # noqa: BLE001  conservar el scan hasta que llegue su TF
                continue
            elegido = msg, stamp, tf
            break
        if elegido is None:
            return
        msg, stamp, tf = elegido
        q, p = tf.rotation, tf.translation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.t_scan_objetos = ahora
        self.ultimo_stamp_scan_objetos = stamp
        centros = objetos_del_scan(msg, (p.x, p.y, yaw), self.grilla, self.dist_pared,
            self.cfg['separacion_pared_objeto_m'], self.cfg['rango_vision'],
            profundidad=self.cfg.get('profundidad_objeto_m', 0.12))
        centros = [c for c in centros if self.linea_libre_al_objeto((p.x, p.y), c)]
        centros = [c for c in centros if not self.corregir_bloque_con_lectura(c, (p.x, p.y))]
        self.objetos_fuera_mapa.actualizar(centros, ahora)

    def puntos_scan(self, alcance=3.0):
        """Puntos del último /scan en el marco base_footprint (sin los que caen sobre el robot)."""
        s = self.scan
        if s is None or not s.header.frame_id:
            return None
        edad = (self.get_clock().now().nanoseconds - Time.from_msg(s.header.stamp).nanoseconds) / 1e9
        if edad < 0.0 or edad > self.cfg['timeout_scan_s']:
            return None
        tf = self.lookup(self.cfg['base_frame'], s.header.frame_id)
        if tf is None:
            return None
        r = np.asarray(s.ranges, dtype=np.float64)
        validos = np.isfinite(r) & (r >= s.range_min) & (r <= s.range_max)
        if not np.any(validos | np.isposinf(r)):
            return None
        a = s.angle_min + np.arange(r.size) * s.angle_increment
        ok = validos & (r > max(s.range_min, 0.02)) & (r < alcance)
        xl, yl = r[ok] * np.cos(a[ok]), r[ok] * np.sin(a[ok])
        c, sn = math.cos(tf[2]), math.sin(tf[2])
        x, y = tf[0] + c * xl - sn * yl, tf[1] + sn * xl + c * yl
        fuera = ~((x > -0.19) & (x < 0.13) & (np.abs(y) < 0.11))
        return x[fuera], y[fuera]

    def espacio_libre(self, puntos, ang):
        """Metros que el chasis puede avanzar en la dirección `ang` (marco del robot)."""
        x, y = puntos
        d0, d1 = math.cos(ang), math.sin(ang)
        frente = float(np.max(self.ESQUINAS @ np.array([d0, d1])))
        # El chasis no está centrado en base_footprint (sobresale 0.18 m atrás y
        # 0.12 m adelante): el "tubo" que barre al moverse es asimétrico.
        lat_esq = -self.ESQUINAS[:, 0] * d1 + self.ESQUINAS[:, 1] * d0
        lat_min, lat_max = float(lat_esq.min()) - 0.01, float(lat_esq.max()) + 0.01
        proy = x * d0 + y * d1
        lat = -x * d1 + y * d0
        en = (proy > 0.0) & (lat > lat_min) & (lat < lat_max)
        return min(0.6, (float(proy[en].min()) if en.any() else 1.0) - frente)

    def mejor_escape(self):
        puntos = self.puntos_scan()
        if puntos is None:
            return None
        # Preferir, entre las direcciones libres, la que va hacia la ruta deseada
        # (antes elegía "la más libre" y a veces se metía más adentro de un rincón).
        pose = self.pose_robot()
        pref = ang_norm(self.dir_ruta - pose[2]) if (self.dir_ruta is not None and pose) else None
        mejor, mejor_puntaje = None, -1e9
        for ang in np.radians(np.arange(0, 360, 15)):
            libre = self.espacio_libre(puntos, float(ang))
            puntaje = min(libre, 0.6)
            if pref is not None and libre > 0.10:
                puntaje += 0.25 * math.cos(float(ang) - pref)
            if puntaje > mejor_puntaje + 1e-3:
                mejor, mejor_puntaje = (libre, float(ang)), puntaje
        return mejor

    def iniciar_escape(self, siguiente):
        if getattr(self, 'goal_heredado', None) is not None:
            self.cancelar_actual()
        self.estado = DESENCAJANDO
        # Si se traba varias veces seguidas en el mismo lugar, se aleja más.
        dist = self.cfg['distancia_escape'] * min(3, max(1, self.n_atascos))
        self.escape = dict(t0=time.monotonic(), siguiente=siguiente, ang=None, p0=self.pose_robot(),
                           dist=dist)
        self.hist.clear()
        self.hist_global.clear()

    def paso_escape(self):
        e = self.escape
        t = time.monotonic() - e['t0']
        if t < 1.0:          # que el velocity_smoother de Nav2 termine de frenar
            return
        if e['ang'] is None:
            mejor = self.mejor_escape()
            if mejor is None or mejor[0] < 0.03:
                self.fin_escape('sin espacio libre alrededor')
                return
            e['ang'] = mejor[1]
            self.get_logger().info(
                f"Desencajando hacia {math.degrees(mejor[1]):.0f}° (libre {mejor[0]:.2f} m)")
        pose = self.pose_robot()
        movido = math.hypot(pose[0] - e['p0'][0], pose[1] - e['p0'][1]) if pose and e['p0'] else 0.0
        puntos = self.puntos_scan()
        libre = self.espacio_libre(puntos, e['ang']) if puntos is not None else 0.0
        v = self.cfg['velocidad_escape']
        if movido >= e['dist'] or libre < 0.03 or t > 1.0 + 3.0 * e['dist'] / v:
            self.fin_escape(f'movido {movido:.2f} m')
            return
        cmd = Twist()
        cmd.linear.x = v * math.cos(e['ang'])
        cmd.linear.y = v * math.sin(e['ang'])
        self.pub_cmd.publish(cmd)

    def fin_escape(self, detalle):
        self.pub_cmd.publish(Twist())
        self.get_logger().info(f'Fin del desencaje: {detalle}')
        self.estado = self.escape['siguiente']
        self.escape = None
        self.hist.clear()
        if self.giro_pendiente is not None:
            angulo, siguiente = self.giro_pendiente
            self.girar(angulo, siguiente)       # una sola vez: giro_pendiente sigue puesto
            self.giro_pendiente = None

    # =================================================== objetos fuera del mapa
    def pose_camara(self):
        """Posición y rumbo del eje óptico +Z en el marco del mapa."""
        if self.grilla is None or self.frame_camara is None:
            return None
        try:
            t = self.tf_buffer.lookup_transform(self.grilla.frame, self.frame_camara, Time())
        except Exception:  # noqa: BLE001
            return None
        p, q = t.transform.translation, t.transform.rotation
        eje_x = 2 * (q.x * q.z + q.w * q.y)
        eje_y = 2 * (q.y * q.z - q.w * q.x)
        if math.hypot(eje_x, eje_y) < 0.2:
            return None
        return p.x, p.y, math.atan2(eje_y, eje_x)

    def linea_libre_al_objeto(self, origen, objeto):
        """Descarta un obstáculo que está detrás de una pared del mapa."""
        g = self.grilla
        dx, dy = objeto[0] - origen[0], objeto[1] - origen[1]
        d = math.hypot(dx, dy)
        recorrido = max(0.0, d - 0.12)
        pasos = max(1, int(math.ceil(recorrido / (g.res / 2))))
        for i in range(pasos + 1):
            fraccion = i * recorrido / pasos / max(d, 1e-6)
            f, c = g.a_celda(origen[0] + fraccion * dx, origen[1] + fraccion * dy)
            if not g.dentro(f, c) or not g.libre[f, c]:
                return False
        return True

    def considerar_objeto_nuevo(self, pose):
        if not self.cfg['inspeccionar_objetos'] or self.objeto_observado is not None:
            return False
        cam = self.pose_camara() or pose
        if cam is None:
            return False
        ahora = time.monotonic()
        candidatos = sorted(self.objetos_fuera_mapa.confirmados(ahora),
                            key=lambda p: math.dist(p, cam[:2]))
        for objeto in candidatos:
            distancia = math.dist(objeto, cam[:2])
            if not (0.25 <= distancia <= self.cfg['rango_vision']):
                continue
            if not self.linea_libre_al_objeto(cam[:2], objeto):
                continue
            cancelacion = self.cancelar_actual()
            self.goal_actual = None
            self.pendiente = None
            visual = self.observacion_visual(objeto)
            self.objeto_observado = dict(punto=objeto, inicio=ahora,
                puntos=[objeto], id_visual=visual['id'] if visual else None,
                desde=None, quieto_desde=None, quieto_stamp_ns=None, ultimo_frame=None,
                imagenes=0, positivas=0, color=visual['color'] if visual else None,
                fase='frenando', reubicaciones=0, giros=0, cancelacion=cancelacion)
            self.estado = OBSERVANDO_OBJETO
            self.pub_cmd.publish(Twist())
            self.get_logger().info(
                f'Objeto fuera del mapa en ({objeto[0]:.2f}, {objeto[1]:.2f}): '
                f'mirada breve de {self.cfg["tiempo_observacion_objeto_s"]:.1f} s; '
                'usaré LiDAR y la cámara si está disponible')
            return True
        return False

    def observacion_visual(self, punto):
        mensaje = self.observaciones_vision
        if mensaje is None or time.monotonic() - mensaje['recibido'] > 0.5:
            return None
        objetos = [o for o in mensaje['objetos'] if math.dist(punto, (o['x'], o['y'])) < 0.40]
        return min(objetos, key=lambda o: math.dist(punto, (o['x'], o['y'])), default=None)

    def limite_observacion(self, porcentaje):
        m = SpeedLimit()
        m.header.stamp = self.get_clock().now().to_msg()
        m.percentage = True
        m.speed_limit = float(porcentaje)
        self.pub_vel.publish(m)
        self.ultimo_limite = porcentaje

    def posicion_observacion(self, pose, objeto):
        """Piso alcanzable, distancia útil y vista libre de paredes y otros objetos."""
        g = self.grilla
        # El objeto tampoco puede atravesarse si todavía no apareció en el costmap.
        objetivo = self._cerca_de(*objeto, radio=self.cfg['lado_bloque_objeto_m'] / 2)
        seguro = g.transitable_base & ~dilatar(g.obstaculos_extra | objetivo, g.m_trans)
        dist = bfs(seguro, *g.a_celda(*pose[:2]))
        if dist is None:
            return None
        vista = g.libre & ~(g.obstaculos_extra & ~self._cerca_de(*objeto))
        return punto_para_mirar(g.candidatable & seguro, vista, dist, g.res, (g.ox, g.oy), objeto,
            d_min=self.cfg['distancia_observacion_min_m'], d_max=self.cfg['distancia_observacion_max_m'],
            d_ideal=self.cfg['distancia_observacion_ideal_m'], fin_vista=0.25)

    def recordar_inspeccion(self, o, observado):
        self.objetos_fuera_mapa.finalizar(o['punto'], time.monotonic(), observado,
            aliases=o['puntos'], id_visual=o['id_visual'])
        punto = o['punto']
        # El color de una pared nunca habilita un bloque que la geometría rechaza.
        f, c = self.grilla.a_celda(*punto)
        campo = getattr(self, 'dist_pared', None)
        if not self.grilla.dentro(f, c):
            return
        d = float(campo[f, c]) if campo is not None else float('inf')
        if d < self.cfg.get('distancia_pared_bloque_m', 0.20):
            self.get_logger().info(
                f'Objeto en ({punto[0]:.2f}, {punto[1]:.2f}) a '
                f'{100 * d:.0f} cm de una pared del mapa: no lo bloqueo (probable eco de la pared)')
            return
        self.registrar_objeto_local(punto)

    def terminar_observacion_objeto(self, observado, motivo):
        o = self.objeto_observado
        if o is None:
            return
        cancelacion = self.cancelar_actual()
        self.goal_actual = None
        # No reanudar una ruta hasta el resultado final de la acción anterior.
        self.cancelacion_cobertura = (cancelacion, float('inf')) if cancelacion is not None else None
        self.pub_cmd.publish(Twist())
        self.limite_observacion(100.0)
        self.recordar_inspeccion(o, observado)
        self.limpiar_costmaps()
        self.ruta_sector.clear()
        self.ruta_region = None
        if observado:
            self.marcar_alrededor(*o['punto'], self.cfg['radio_visto_objeto_m'])
        self.objeto_observado = None
        self.pendiente = None
        self.escape = None
        self.giro_pendiente = None
        self.estado = ELIGIENDO
        self.get_logger().info(f'Observación de objeto: {motivo}; guardado en el mapa local, no lo vuelvo a analizar')

    def paso_observacion_objeto(self):
        """Una mirada corta; la ausencia de confirmación visual nunca repite el objeto."""
        o = self.objeto_observado
        if o is None:
            return
        ahora = time.monotonic()
        if ahora - o['inicio'] > self.cfg['timeout_observacion_objeto_s']:
            self.terminar_observacion_objeto(False, 'conservo la posición del LiDAR sin insistir con la cámara')
            return
        if o['fase'] == 'acercando':
            self.limite_observacion(self.cfg['porcentaje_vel_observacion'])
            return  # Nav2 sigue usando LiDAR aunque dejen de llegar imágenes.
        self.pub_cmd.publish(Twist())
        cancelacion = o['cancelacion']
        if ahora - o['inicio'] < 0.3 or (cancelacion is not None and not cancelacion.done()):
            return
        if math.hypot(*self.vel_base) > 0.05 or self.vel_giro > 0.15:
            o.update(desde=None, quieto_desde=None, imagenes=0, positivas=0)
            return
        if o['quieto_desde'] is None:
            o['quieto_desde'] = ahora
            o['quieto_stamp_ns'] = self.get_clock().now().nanoseconds
        if ahora - o['quieto_desde'] < 0.2:
            return
        cam = self.pose_camara()
        camara_activa = (cam is not None and self.t_ultima_imagen is not None
                         and ahora - self.t_ultima_imagen <= 0.75)
        punto_lidar = self.objetos_fuera_mapa.mas_cercano(o['punto'], ahora)
        if punto_lidar is not None:
            o['punto'] = punto_lidar
            if all(math.dist(punto_lidar, p) >= 0.10 for p in o['puntos']):
                o['puntos'].append(punto_lidar)
        visual = self.observacion_visual(o['punto']) if camara_activa else None
        if visual is not None:
            # La cámara añade atributos sin desplazar la posición confirmada por /scan.
            o['color'], o['id_visual'] = visual['color'], visual['id']
        punto = o['punto']
        referencia = cam or self.pose_robot()
        error = 0.0
        en_cuadro = False
        if referencia is not None:
            error = ang_norm(math.atan2(punto[1] - referencia[1], punto[0] - referencia[0]) - referencia[2])
            mensaje = self.observaciones_vision
            # La mancha corrige el rumbo solo si la foto se tomó después de frenar.
            if visual is not None and mensaje['stamp_ns'] > o['quieto_stamp_ns']:
                error = visual['error_angular']
            tolerancia = min(math.radians(self.cfg['tolerancia_centrar_objeto_deg']), self.fov / 2 - 0.1)
            max_giros = 2 if o.get('giro_resultado') == GoalStatus.STATUS_SUCCEEDED else 1
            if abs(error) > tolerancia and o['giros'] < max_giros:
                if not self.spin_cli.server_is_ready():
                    self.recuperar_nav2(siguiente=OBSERVANDO_OBJETO)
                    return
                o['giros'] += 1
                o.update(desde=None, quieto_desde=None, imagenes=0, positivas=0)
                self.get_logger().info(f'[objeto] Apunto al objeto del LiDAR: giro {math.degrees(error):.0f}° '
                                       f'con el eje {self.frame_camara or "frontal del robot"}')
                self.girar(error, OBSERVANDO_OBJETO)
                return
            en_cuadro = cam is not None and abs(error) <= tolerancia
        if camara_activa:
            distancia = visual['distancia'] if visual else math.dist(punto, cam[:2])
            if (o['reubicaciones'] == 0 and not self.cfg['distancia_observacion_min_m'] <= distancia
                    <= self.cfg['distancia_observacion_max_m']):
                o['reubicaciones'] = 1
                pose = self.pose_robot()
                posicion = self.posicion_observacion(pose, punto) if pose is not None else None
                if posicion is not None:
                    x, y, yaw, d_m = posicion
                    if (d_m <= 0.60 and self.tiempo_restante() > self.segundos_para_volver(x, y)
                            + self.cfg['margen_regreso_s'] + self.cfg['tiempo_observacion_objeto_s'] + d_m / 0.10):
                        yaw = ang_norm(yaw - ang_norm(cam[2] - pose[2]))
                        o.update(fase='acercando', desde=None, quieto_desde=None, imagenes=0, positivas=0)
                        self.limite_observacion(self.cfg['porcentaje_vel_observacion'])
                        if not self.enviar_goal(x, y, yaw, 1.0, d_m, etiqueta='observar objeto'):
                            self.terminar_observacion_objeto(False, 'Nav2 no disponible; conservo el obstáculo del LiDAR')
                        return
        if o['desde'] is None:
            o['desde'] = ahora
        mensaje = self.observaciones_vision
        if (mensaje is not None and ahora - mensaje['recibido'] <= 0.5
                and mensaje['stamp_ns'] > o['quieto_stamp_ns'] + 200_000_000
                and mensaje['stamp_ns'] != o['ultimo_frame']):
            o['ultimo_frame'] = mensaje['stamp_ns']
            o['imagenes'] += 1
            if visual is not None:
                o['positivas'] += 1
        if ahora - o['desde'] >= self.cfg['tiempo_observacion_objeto_s']:
            con_color = (visual is not None and visual['color'] != 'desconocido'
                         and o['positivas'] >= self.cfg['imagenes_observacion_minimas']
                         and abs(error) <= math.radians(self.cfg['tolerancia_centrar_objeto_deg']))
            mirada = (en_cuadro and camara_activa
                      and self.t_ultima_imagen > o['quieto_desde'] + 0.2)
            o['color_confirmado'] = bool(mirada and con_color)
            detalle = (f'objeto {o["color"]} visto con cámara y LiDAR' if mirada and con_color
                       else 'mirada completada hacia el objeto; sin color confirmado' if mirada
                       else 'posición del LiDAR guardada; no hubo una mirada de cámara válida')
            self.terminar_observacion_objeto(mirada, detalle)

    # ==================================================================== regreso
    def pedir_regreso(self, motivo):
        if self.motivo_regreso is not None or self.estado in (
                VOLVIENDO, APROXIMANDO, TERMINADO, REGRESAR, DESENCAJANDO, RELOCALIZANDO):
            return
        self.motivo_regreso = motivo
        self.get_logger().info(f'Vuelvo a la largada: {motivo}')
        cancelacion = self.cancelar_actual()
        self.cancelacion_cobertura = (
            cancelacion, time.monotonic() + 2.0) if cancelacion is not None else None
        self.goal_actual = None
        if self.objeto_observado is not None:
            self.recordar_inspeccion(self.objeto_observado, False)
            self.limite_observacion(100.0)
        self.objeto_observado = None
        self.ruta_sector.clear()
        self.ruta_regreso.clear()
        self.estado = REGRESAR

    def volver_a_largada(self):
        if self.cancelacion_cobertura is not None:
            cancelacion, limite = self.cancelacion_cobertura
            if not cancelacion.done() and time.monotonic() < limite:
                return
            self.cancelacion_cobertura = None
        if not self.cfg['volver_al_inicio']:
            self.terminar('sin regreso (volver_al_inicio=false)')
            return
        self.fijar_largada(forzar=True)
        pose = self.pose_robot()
        if (self.cfg['modo_exploracion'] == 'sectores_nbv'
                and self.sectorizador is None and pose is not None):
            self.preparar_sectores(pose)
        x, y, yaw = self.largada
        d = math.hypot(x - pose[0], y - pose[1]) if pose else 3.0
        self.estado = VOLVIENDO
        if d < self.cfg['tolerancia_llegada']:
            self.iniciar_aproximacion()
            return
        if self.regreso_directo:
            if pose is None or not self.enviar_goal(x, y, yaw, 1.0, d, etiqueta='regreso directo'):
                self.estado = REGRESAR
            else:
                self.estado = VOLVIENDO
            self.publicar_debug([])
            return
        if self.sectorizador is not None:
            if pose is None:
                self.estado = REGRESAR
                return
            g = self.grilla
            if not self.ruta_regreso:
                fila, col = g.a_celda(pose[0], pose[1])
                fila_casa, col_casa = g.a_celda(x, y)
                def planificar(seguro):
                    ruta = self.sectorizador.ruta_hacia(
                        seguro, (col, fila), (col_casa, fila_casa),
                        self.cfg['paso_maximo_portal_m'])
                    if ruta:
                        ultima, _ = ruta[-1]
                        wx, wy = g.a_mundo(ultima[1], ultima[0])
                        if math.hypot(wx - x, wy - y) <= 0.35:
                            return ruta
                    return []
                (self.ruta_regreso, pasos, d, self.seguro_ruta_regreso,
                 margen_usado) = self.seleccionar_ruta_portales(pose, planificar, self.largada)
                if not self.ruta_regreso:
                    self.get_logger().warn('Sin ruta transitable por portales hasta la largada')
                    self.reintentar_regreso()
                    return
                self.get_logger().info(
                    f'Regreso por {len(self.ruta_regreso)} puntos calculados, '
                    f'{sum(portal for _, portal in self.ruta_regreso)} portal(es), '
                    f'margen de paradas {margen_usado:.2f} m')
            else:
                pasos, d = self.guias_de_ruta(
                    self.ruta_regreso, pose, self.largada, seguro=self.seguro_ruta_regreso)
            if not pasos:
                self.reintentar_regreso()
                return
            x, y, yaw = pasos[-1]
            etiqueta = 'regreso portales'
        else:
            etiqueta = 'regreso'
            pasos = None
        if not self.encarar_y_enviar(x, y, yaw, 1.0, d, etiqueta, VOLVIENDO, pasos):
            self.estado = REGRESAR   # Nav2 no está: se reintenta en el próximo ciclo
        self.publicar_debug([])

    def reintentar_regreso(self):
        self.ruta_regreso.clear()
        if self.registrar_fallo_localizacion(REGRESAR):
            return
        self.reintentos_regreso += 1
        if self.reintentos_regreso > self.cfg['max_reintentos_regreso']:
            if not self.regreso_directo:
                # Por portales no hubo caso: probar con el destino directo a Nav2.
                self.regreso_directo = True
                self.reintentos_regreso = 0
                self.get_logger().warn('El regreso por portales falló varias veces: pruebo con la '
                                       'largada como destino directo de Nav2')
            elif self.tiempo_restante() > 10.0:
                # Mientras quede tiempo se sigue intentando: /done antes de llegar
                # pierde el bono de regreso sin ganar nada.
                self.regreso_directo = False
                self.reintentos_regreso = 0
                self.get_logger().warn(f'Regreso sigue fallando; quedan {self.tiempo_restante():.0f} s: '
                                       'vuelvo a intentar')
            else:
                self.get_logger().error('No pude volver a la largada y se acaba el tiempo: termino donde estoy')
                self.terminar('regreso fallido')
                return
        self.get_logger().warn(f'Regreso falló, reintento {self.reintentos_regreso}')
        self.limpiar_costmaps()
        self.n_atascos += 1
        self.iniciar_escape(REGRESAR)   # despegarse antes de reintentar

    def iniciar_aproximacion(self):
        self.estado = APROXIMANDO
        self.t_aprox = time.monotonic()

    def paso_aproximacion(self):
        """Ajuste fino holonómico hasta < tolerancia_final de la largada (bono ≤ 0.25 m)."""
        t = time.monotonic() - self.t_aprox
        if t < 1.2:          # dejar que el velocity_smoother de Nav2 se calle
            return
        pose = self.pose_robot()
        if pose is None:
            return
        dx, dy = self.largada[0] - pose[0], self.largada[1] - pose[1]
        d = math.hypot(dx, dy)
        if d < self.cfg['tolerancia_final']:
            self.pub_cmd.publish(Twist())
            self.terminar(f'en la largada (error {d:.2f} m)')
            return
        if t > 9.0 or d > 0.6:
            self.pub_cmd.publish(Twist())
            self.terminar(f'aproximación incompleta (error {d:.2f} m)')
            return
        puntos = self.puntos_scan()
        direccion = ang_norm(math.atan2(dy, dx) - pose[2])
        if puntos is None or self.espacio_libre(puntos, direccion) < 0.05:
            self.pub_cmd.publish(Twist())
            return
        v = min(self.cfg['velocidad_aproximacion'], 0.8 * d + 0.03)
        c, s = math.cos(-pose[2]), math.sin(-pose[2])
        cmd = Twist()
        cmd.linear.x = v * (c * dx - s * dy) / d
        cmd.linear.y = v * (s * dx + c * dy) / d
        self.pub_cmd.publish(cmd)

    def terminar(self, detalle):
        if self.estado in (TERMINADO, FINALIZANDO):
            return
        self.cancelar_actual()
        self.pub_cmd.publish(Twist())
        inflacion = getattr(self, 'inflacion', None)
        if inflacion is not None:
            inflacion.restaurar('fin de la corrida')
        if self.cfg['publicar_done'] and self.reportes < int(self.cfg['cantidad_victimas']):
            self.estado = FINALIZANDO
            self.publicar_visto()
            espera = min(self.cfg['espera_reportes_finales_s'], max(
                0.0, self.cfg['duracion_corrida_s'] - self.tiempo_transcurrido() - 0.3))
            self.cierre_reportes = dict(detalle=detalle, limite=time.monotonic() + espera,
                                       futuro=None, listo=None)
            self.get_logger().info(
                f'Antes de /done quedan {int(self.cfg["cantidad_victimas"]) - self.reportes} '
                'reportes: solicito las mejores detecciones y estimaciones disponibles')
            self.paso_reportes_finales()
            return
        self.finalizar_done(detalle)

    def paso_reportes_finales(self):
        """Espera asíncrona y acotada; el robot permanece detenido."""
        cierre, ahora = self.cierre_reportes, time.monotonic()
        self.pub_cmd.publish(Twist())
        if cierre['futuro'] is None and ahora < cierre['limite'] and self.reportes_finales_cli.service_is_ready():
            cierre['futuro'] = self.reportes_finales_cli.call_async(Trigger.Request())
        futuro = cierre['futuro']
        if futuro is not None and futuro.done() and cierre['listo'] is None:
            try:
                respuesta = futuro.result()
                self.get_logger().info(f'Cierre de reportes: {respuesta.message}')
            except Exception as exc:
                self.get_logger().warn(f'No se completó el cierre de reportes: {exc}')
            # Deja que /victimas llegue a los suscriptores antes de emitir /done.
            cierre['listo'] = ahora + 0.15
        if ahora >= cierre['limite'] or (cierre['listo'] is not None and ahora >= cierre['listo']):
            if cierre['listo'] is None:
                self.get_logger().warn('Detector sin respuesta al cierre: continúo con /done por el plazo')
            self.finalizar_done(cierre['detalle'])

    def finalizar_done(self, detalle):
        self.estado = TERMINADO
        self.get_logger().info(
            f'FIN: {self.motivo_regreso or "-"} | {detalle} | cobertura {100 * self.cobertura():.1f}% '
            f'| reportes {self.reportes} | t={self.tiempo_transcurrido():.0f}s')
        if self.cfg['publicar_done']:
            self.pub_done.publish(Empty())
            self.get_logger().info('/done publicado')

    # ====================================================================== ciclos
    def tick_rapido(self):
        """5 Hz: marcar lo visto, medir velocidad, aproximación final."""
        if self.estado == TERMINADO:
            return
        if self.estado == FINALIZANDO:
            self.paso_reportes_finales()
            return
        # A los 9:59 inicia el cierre, esté donde esté, con margen para los reportes finales.
        if getattr(self, 't_mapa', None) and self.tiempo_transcurrido() >= self.cfg['done_forzado_s']:
            self.terminar(f'tiempo: {self.cfg["done_forzado_s"]:.0f} s desde /map')
            return
        if self.estado in (ESPERANDO_MAPA, LOCALIZANDO, ACTIVANDO_NAV2):
            return
        o = getattr(self, 'objeto_observado', None)
        if o is not None and time.monotonic() - o['inicio'] > self.cfg['timeout_observacion_objeto_s']:
            self.terminar_observacion_objeto(False, 'agoté el tiempo para orientar y analizar el objeto')
            return
        if hasattr(self, 'pub_estado_loc'):
            self.publicar_estado_loc()
        if self.estado == RELOCALIZANDO:
            pose = self.pose_robot()
            if pose is not None:
                self.actualizar_localizacion(pose)
            self.paso_relocalizacion()
            self.publicar_debug([])
            return
        pose = self.pose_robot()
        if pose is None:
            if self.estado in (APROXIMANDO, DESENCAJANDO, OBSERVANDO_OBJETO):
                self.pub_cmd.publish(Twist())
            return
        ahora_t = time.monotonic()
        # Velocidad de giro con la ODOMETRÍA: el yaw de AMCL salta al corregirse y
        # daba "giros" falsos que impedían marcar lo visto.
        odom = self.lookup(self.cfg['odom_frame'], self.cfg['base_frame'])
        yaw_giro = odom[2] if odom is not None else pose[2]
        if self.yaw_prev is not None and ahora_t > self.yaw_prev[1]:
            dt = ahora_t - self.yaw_prev[1]
            self.vel_giro = abs(ang_norm(yaw_giro - self.yaw_prev[0])) / dt
            # Velocidad de traslación en el marco del robot (hacia dónde va).
            prev = getattr(self, 'odom_prev', None)
            if odom is not None and prev is not None:
                vx, vy = (odom[0] - prev[0]) / dt, (odom[1] - prev[1]) / dt
                c, s_ = math.cos(-odom[2]), math.sin(-odom[2])
                self.vel_base = (c * vx - s_ * vy, s_ * vx + c * vy)
        self.odom_prev = odom
        self.yaw_prev = (yaw_giro, ahora_t)
        self.actualizar_localizacion(pose)
        # Girando rápido el láser sale "corrido" y el error sube solo: no relocalizar
        # en medio de un giro, ni repetir antes de `espera_entre_relocalizaciones`.
        if (self.localizacion_mala() and self.vel_giro < 0.5
                and ahora_t - self.t_ult_reloc > self.cfg['espera_entre_relocalizaciones']):
            volviendo = self.motivo_regreso is not None or self.estado in (REGRESAR, VOLVIENDO, APROXIMANDO)
            self.relocalizar(REGRESAR if volviendo else ELIGIENDO)
            return
        # "Visto" solo si la cámara realmente sirvió: girando rápido el detector descarta
        # las imágenes, y si dejaron de llegar imágenes no se vio nada (si nunca
        # llegó ninguna, no se bloquea la exploración).
        imagen_ok = self.t_ultima_imagen is not None and ahora_t - self.t_ultima_imagen < 1.0
        if self.error_loc is not None and self.loc_mala_desde is None:
            self.marcar_alrededor(*pose[:2], self.cfg['radio_visto_robot_m'])
        if (self.error_loc is not None and self.loc_mala_desde is None and imagen_ok
                and self.vel_giro <= self.cfg['giro_maximo_visto']):
            cam = self.pose_camara()
            if cam is not None:
                self.marcar_visto(cam)
        if ahora_t - getattr(self, 't_publicacion_visto', 0.0) >= 1.0:
            self.publicar_visto()
            self.t_publicacion_visto = ahora_t
        if self.estado in (NAVEGANDO, VOLVIENDO):
            self.publicar_limite_velocidad()
        ahora = time.monotonic()
        if self.ultima_pose is not None and self.estado in (NAVEGANDO, VOLVIENDO):
            d = math.hypot(pose[0] - self.ultima_pose[0], pose[1] - self.ultima_pose[1])
            if d > 0.01:
                self.recorrido += d
                self.t_moviendo += ahora - self.ultimo_t_pose
        if self.ultima_pose is not None and hasattr(self, 'vel_ventana'):
            self.vel_ventana.append((ahora, math.hypot(pose[0] - self.ultima_pose[0], pose[1] - self.ultima_pose[1])))
            while self.vel_ventana and ahora - self.vel_ventana[0][0] > 15.0:
                self.vel_ventana.pop(0)
        self.ultima_pose, self.ultimo_t_pose = pose, ahora
        if self.estado == APROXIMANDO:
            self.paso_aproximacion()
        elif self.estado == DESENCAJANDO:
            self.paso_escape()
        elif self.estado == OBSERVANDO_OBJETO:
            self.paso_observacion_objeto()

    def log_estado(self):
        ahora = time.monotonic()
        if ahora - self.t_estado_log < 15.0 or self.grilla is None or self.t_mapa is None:
            return
        self.t_estado_log = ahora
        ga = self.goal_actual
        sector = self.sectorizador.actual if self.sectorizador is not None else '-'
        self.get_logger().info(
            f'[estado] t={self.tiempo_transcurrido():.0f}s {self.estado} | sector {sector} | '
            f'cobertura {100 * self.cobertura():.0f}% | goal '
            f'{(ga["etiqueta"] + " ({:.2f}, {:.2f})".format(ga["x"], ga["y"])) if ga else "-"} | '
            f'atascos {self.n_atascos} | lista negra {len(self.lista_negra)} | '
            f'láser-mapa {"-" if self.error_loc is None else f"{100 * self.error_loc:.1f} cm"} | '
            f'cajas sin confirmar {len(self.pendientes_vision)} | reportes {self.reportes} | '
            f'vel 15 s {self.velocidad_reciente():.2f} m/s (media en marcha {self.velocidad_media():.2f})')

    def velocidad_reciente(self):
        v = getattr(self, 'vel_ventana', [])
        if len(v) < 2:
            return 0.0
        return sum(d for _, d in v[1:]) / max(1e-3, v[-1][0] - v[0][0])

    def ciclo(self):
        self.log_estado()
        e = self.estado
        if e in (ESPERANDO_MAPA, TERMINADO, FINALIZANDO, APROXIMANDO, DESENCAJANDO, RELOCALIZANDO):
            return
        ahora = time.monotonic()

        if e == LOCALIZANDO:
            # Rescate del arranque: si AMCL o Nav2 no terminan de activarse, los activo yo.
            espera = ahora - self.t_mapa
            if espera > 1.0 and ahora - self.t_ult_revision_lc > 1.0:
                self.t_ult_revision_lc = ahora
                self.revisar_ciclo_de_vida(('amcl', *self.NODOS_NAV2), activar=espera > 8.0)
            pose = self.pose_robot()
            if (pose is not None and self.nav2_activo() and self.nav.server_is_ready()
                    and self.nav_ruta.server_is_ready() and self.spin_cli.server_is_ready()):
                self.t_localizado = ahora
                self.get_logger().info('AMCL localizado y Nav2 listo: ¡a explorar!')
                if self.cfg['giro_inicial']:
                    self.girar(2 * math.pi - 0.2)
                else:
                    self.estado = ELIGIENDO
            elif pose is None and ahora - self.t_ultimo_init > 2.0:
                self.publicar_pose_inicial()
                self.t_ultimo_init = ahora
            return

        if e == ACTIVANDO_NAV2:
            self.paso_activacion_nav2(ahora)
            return

        if ahora - self.t_ult_revision_lc >= 2.0:
            self.t_ult_revision_lc = ahora
            self.revisar_ciclo_de_vida(self.NODOS_NAV2)
        if not self.nav2_activo():
            self.recuperar_nav2()
            return

        self.fijar_largada()
        pose = self.pose_robot()
        self.completar_huecos()
        if e == ELIGIENDO and pose is not None:
            f, c = self.grilla.a_celda(*pose[:2])
            if not self.grilla.dentro(f, c) or not self.grilla.transitable[f, c]:
                self.get_logger().warn('Inicio dentro del margen: recupero espacio antes de elegir otra vista')
                self.cancelar_actual()
                self.goal_actual = None
                self.n_atascos += 1
                self.iniciar_escape(ELIGIENDO)
                return

        # Plazo: si no vuelvo ahora, no llego antes de los 10 minutos.
        if e in (ELIGIENDO, NAVEGANDO, COMPORTAMIENTO, OBSERVANDO_OBJETO) and pose is not None:
            if self.tiempo_restante() <= self.segundos_para_volver(pose[0], pose[1]) + self.cfg['margen_regreso_s']:
                self.pedir_regreso('se acaba el tiempo')
                return

        if self.objeto_observado is not None and self.reportes >= int(self.cfg['cantidad_victimas']):
            self.pedir_regreso(f'{self.reportes} víctimas reportadas')
            return

        if e == OBSERVANDO_OBJETO:
            self.publicar_debug([])
            return

        if e == REGRESAR:
            self.volver_a_largada()
            return

        if (e == NAVEGANDO and self.sectorizador is not None and self.goal_actual is not None
                and self.goal_actual.get('etiqueta', '').startswith('portal sector')
                and self.en_sector(pose)):
            if self.soltar_sin_frenar():
                self.cancelacion_cobertura = None
            else:
                cancelacion = self.cancelar_actual()
                self.cancelacion_cobertura = (cancelacion, ahora + 2.0) if cancelacion is not None else None
            self.goal_actual = None
            self.pendiente = None
            self.ruta_sector.clear()
            self.ruta_region = None
            self.n_atascos = 0
            self.estado = ELIGIENDO
            self.get_logger().info(f'Ya estoy en el sector {self.sectorizador.actual}: no espero el último '
                                   'tramo del portal, sigo con las vistas')
            return

        if e == NAVEGANDO and self.sectorizador is not None:
            sector = self.sectorizador.actual
            if sector is not None:
                cobertura = self.sectorizador.cobertura(self.visto)
                self.sectorizador.avanzar_cubiertos(self.visto)
                if self.sectorizador.actual != sector:
                    if self.soltar_sin_frenar():
                        self.cancelacion_cobertura = None
                    else:
                        cancelacion = self.cancelar_actual()
                        self.cancelacion_cobertura = (
                            cancelacion, ahora + 2.0) if cancelacion is not None else None
                    self.goal_actual = None
                    self.pendiente = None
                    self.ruta_sector.clear()
                    self.ruta_region = None
                    self.estado = ELIGIENDO
                    self.get_logger().info(
                        f'Sector {sector} cerrado durante la navegación '
                        f'({100 * cobertura:.1f}%, {self.sectorizador.resultados[sector]}): '
                        'cancelo la pose pendiente y elijo el próximo destino')
                    return

        if e == ELIGIENDO and self.cancelacion_cobertura is not None:
            cancelacion, limite = self.cancelacion_cobertura
            if not cancelacion.done() and ahora < limite:
                return
            self.cancelacion_cobertura = None

        if e == ENVIAR_PENDIENTE:
            p = self.pendiente
            self.pendiente = None
            if p is None or not self.enviar_goal(*p[:5], etiqueta=p[5], pasos=p[7]):
                self.estado = REGRESAR if (p and p[6] == VOLVIENDO) else ELIGIENDO
            else:
                self.estado = p[6]
            return

        if (e in (NAVEGANDO, ELIGIENDO) and pose is not None
                and self.tiempo_restante() > self.segundos_para_volver(pose[0], pose[1])
                + self.cfg['margen_regreso_s'] + self.cfg['tiempo_observacion_objeto_s'] + 3.0
                and self.considerar_objeto_nuevo(pose)):
            return

        # A menos de `tolerancia_cerca_destino_m` del destino final ya está: no esperar a
        # que Nav2 clave los últimos centímetros (oscilaba y terminaba "atascado").
        ga = self.goal_actual
        if (e == NAVEGANDO and pose is not None and ga is not None and not ga.get('continuacion')
                and math.hypot(pose[0] - ga['x'], pose[1] - ga['y']) <= self.cfg['tolerancia_cerca_destino_m']):
            self.cancelar_actual()
            self.get_logger().info(f'[{ga.get("etiqueta", "")}] a '
                                   f'{100 * math.hypot(pose[0] - ga["x"], pose[1] - ga["y"]):.0f} cm del destino: llegué')
            self.al_llegar(ga)
            return
        if e in (NAVEGANDO, VOLVIENDO) and pose is not None and self.detectar_atasco(pose):
            self.resolver_atasco(pose)
            return
        if e in (NAVEGANDO, VOLVIENDO, ELIGIENDO) and pose is not None and self.vigilancia_global(pose):
            self.get_logger().warn(f"Vigilancia: {self.cfg['tiempo_atascado_global']:.0f} s sin avanzar aunque cambié de goal")
            self.resolver_atasco(pose)
            return

        if e in (NAVEGANDO, VOLVIENDO) and pose is not None and self.anticipar_siguiente_tramo(pose):
            return

        if (e == VOLVIENDO and self.goal_actual is not None
                and ahora - self.goal_actual['t0'] > self.goal_actual['t_max']):
            self.get_logger().warn('Tramo de regreso demorado: cancelo y recalculo desde la pose actual')
            self.cancelar_actual()
            self.goal_actual = None
            self.reintentar_regreso()
            return

        if e in (COMPORTAMIENTO, VOLVIENDO) or pose is None:
            self.publicar_debug([])
            return

        if e == NAVEGANDO:
            ga = self.goal_actual
            if ga is None:
                self.estado = ELIGIENDO
            else:
                if ahora - ga['t0'] > ga['t_max']:
                    self.get_logger().warn('goal demorado: lo cancelo')
                    self.cancelar_actual()
                    self.fallo_goal()
                elif ga.get('etiqueta', '').startswith(('portal sector', 'confirmar')):
                    # Un portal (o ir a mirar una caja) no tiene ganancia visual; esperar el resultado
                    # de Nav2 antes de enviar el siguiente tramo.
                    self.publicar_debug([])
                    return
                else:
                    gan, _ = self.ganancia_en(*self.grilla.a_celda(ga['x'], ga['y']))
                    umbral = (max(1, self.cfg['fraccion_agotado'] * ga['ganancia0'])
                              if ga['ganancia0'] < self.cfg['ganancia_minima']
                              else max(self.cfg['ganancia_minima'],
                                       self.cfg['fraccion_agotado'] * ga['ganancia0']))
                    if gan < umbral:
                        # Cancelar antes de cambiar de tipo de acción evita dos
                        # navegadores compitiendo por el mismo FollowPath.
                        self.lista_negra.append((ga['x'], ga['y']))
                        if self.soltar_sin_frenar():
                            self.cancelacion_cobertura = None
                        else:
                            cancelacion = self.cancelar_actual()
                            self.cancelacion_cobertura = (
                                cancelacion, ahora + 2.0) if cancelacion is not None else None
                        self.goal_actual = None
                        self.estado = ELIGIENDO
                        return
                    else:
                        self.publicar_debug([])
                        return

        if self.estado == ELIGIENDO:
            if self.reportes >= int(self.cfg['cantidad_victimas']):
                self.pedir_regreso(f"{self.reportes} víctimas reportadas")
                return
            if ahora < self.espera_mirar_hasta:
                self.publicar_debug([])
                return                          # mirando una caja para que el detector la confirme
            conf = self.elegir_confirmacion(pose)
            if conf is not None:
                x, y, yaw, d_m, caja = conf
                self.get_logger().info(f'Caja roja sin confirmar en ({caja[0]:.2f}, {caja[1]:.2f}): voy a '
                                       f'mirarla de cerca desde ({x:.2f}, {y:.2f})')
                if math.hypot(x - pose[0], y - pose[1]) < self.cfg['tolerancia_llegada']:
                    self.cancelar_actual()
                    self.al_llegar(dict(x=x, y=y, yaw=yaw, etiqueta='confirmar caja'))
                else:
                    self.encarar_y_enviar(x, y, yaw, 50, d_m, 'confirmar caja', NAVEGANDO)
                return
            cob = self.cobertura()
            if self.cfg['modo_exploracion'] == 'nbv' and cob >= self.cfg['cobertura_objetivo']:
                self.pedir_regreso(f'cobertura {100 * cob:.1f}%')
                return
            if self.cfg['modo_exploracion'] == 'sectores_nbv':
                self.preparar_sectores(pose)
                self.sectorizador.avanzar_cubiertos(self.visto)
                self.seleccionar_sector_pendiente(pose)
                while self.sectorizador.actual is not None:
                    if self.en_sector(pose):
                        self.ruta_sector.clear()
                        self.ruta_region = None
                        break
                    anterior = self.sectorizador.actual
                    if self.paso_hacia_sector(pose):
                        return
                    if self.sectorizador.actual == anterior:
                        break
                    self.seleccionar_sector_pendiente(pose)
            while True:
                mejor, top = self.elegir_vista(pose)
                if (mejor is not None or self.sectorizador is None
                        or self.sectorizador.actual is None):
                    break
                region = self.sectorizador.actual
                self.get_logger().warn(
                    f'Sector {region} sin vistas NBV alcanzables; '
                    f'cobertura {100 * self.sectorizador.cobertura(self.visto):.1f}%')
                self.sectorizador.avanzar_sin_vistas()
                self.sectorizador.avanzar_cubiertos(self.visto)
                self.seleccionar_sector_pendiente(pose)
                if self.sectorizador.actual is None:
                    continue  # probar vistas exteriores de las zonas que quedaron sin acceso
                if not self.en_sector(pose):
                    self.paso_hacia_sector(pose)
                    return
            if mejor is None:
                self.pedir_regreso(f'no quedan vistas útiles alcanzables (cobertura {100 * cob:.1f}%; '
                                  f'sectores {self.sectorizador.resultados if self.sectorizador else "NBV"})')
                return
            _, x, y, yaw, gan, d_m = mejor
            etiqueta = 'vista'
            if self.sectorizador is not None:
                region = self.sectorizador.actual
                etiqueta = f'sector {region}' if region is not None else 'vista exterior'
            if math.hypot(x - pose[0], y - pose[1]) < self.cfg['tolerancia_llegada']:
                self.cancelar_actual()
                self.al_llegar(dict(x=x, y=y, yaw=yaw))   # ya estoy: solo apuntar la cámara
            else:
                self.encarar_y_enviar(x, y, yaw, gan, d_m, etiqueta, NAVEGANDO)
            self.publicar_debug(top)

    def publicar_visto(self):
        """Cobertura operativa para las estimaciones finales, también sin RViz."""
        if self.grilla is None or self.visto is None:
            return
        g = self.grilla
        m = OccupancyGrid()
        m.header.frame_id = g.frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.info.resolution = g.res
        m.info.width, m.info.height = g.ancho, g.alto
        m.info.origin.position.x, m.info.origin.position.y = g.ox, g.oy
        m.info.origin.orientation.w = 1.0
        capa = np.zeros((g.alto, g.ancho), dtype=np.int8)
        capa[self.visto & g.libre] = 60
        capa[g.bloques_conocidos] = 100
        m.data = capa.ravel().tolist()
        self.pub_visto.publish(m)

    # ======================================================================= debug
    def publicar_debug(self, top):
        if not self.cfg['publicar_debug'] or self.grilla is None:
            return
        ahora = time.monotonic()
        if not top and ahora - self.ultimo_debug < 1.0:
            return
        self.ultimo_debug = ahora
        g = self.grilla
        arr = MarkerArray()
        borrar = Marker()
        borrar.action = Marker.DELETEALL
        arr.markers.append(borrar)

        def marcador(ns, i, tipo, x, y, escala, rgba, yaw=None):
            mk = Marker()
            mk.header.frame_id = g.frame
            mk.ns, mk.id, mk.type = ns, i, tipo
            mk.pose.position.x, mk.pose.position.y = x, y
            if yaw is None:
                mk.pose.orientation.w = 1.0
            else:
                mk.pose.orientation.z, mk.pose.orientation.w = yaw_a_quat(yaw)
            mk.scale.x, mk.scale.y, mk.scale.z = escala
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = rgba
            arr.markers.append(mk)
            return mk

        if top:
            umax = max(t[0] for t in top) or 1.0
            for i, (u, x, y, *_r) in enumerate(top):
                s = 0.06 + 0.10 * u / umax
                marcador('candidatos', i, Marker.SPHERE, x, y, (s, s, s), (1.0 - u / umax, u / umax, 0.2, 0.9))
        for i, (celda, portal) in enumerate(self.ruta_sector):
            wx, wy = g.a_mundo(celda[1], celda[0])
            marcador('ruta_portales', i, Marker.SPHERE, wx, wy,
                     (0.15, 0.15, 0.15), (1.0, 0.45 if portal else 0.8, 0.1, 0.9))
        for i, (celda, portal) in enumerate(self.ruta_regreso):
            wx, wy = g.a_mundo(celda[1], celda[0])
            marcador('ruta_regreso', i, Marker.SPHERE, wx, wy,
                     (0.15, 0.15, 0.15), (0.2, 0.9 if portal else 0.5, 1.0, 0.9))
        for i, (x, y) in enumerate(self.bloques_objetos):
            lado = self.cfg['lado_bloque_objeto_m']
            marcador('bloques_objetos', i, Marker.CUBE, x, y,
                     (lado, lado, 0.08), (1.0, 0.25, 0.1, 0.8))
        if self.goal_actual is not None:
            ga = self.goal_actual
            meta = ga['meta_tramo']
            marcador('goal', 0, Marker.ARROW, meta[0], meta[1],
                     (0.4, 0.07, 0.07), (1.0, 0.1, 0.9, 1.0), meta[2])
            for i, (px, py, rumbo) in enumerate(ga.get('pasos') or []):
                marcador('guias_nav2', i, Marker.ARROW, px, py,
                         (0.20, 0.035, 0.035), (1.0, 0.65, 0.0, 1.0), rumbo)
        if self.objeto_observado is not None:
            ox, oy = self.objeto_observado['punto']
            marcador('objeto_nuevo', 0, Marker.SPHERE, ox, oy,
                     (0.22, 0.22, 0.22), (1.0, 0.2, 0.2, 0.95))
        lx, ly, _ = self.largada
        marcador('largada', 0, Marker.CYLINDER, lx, ly, (0.5, 0.5, 0.02),
                 (0.1, 0.6, 1.0, 0.5 if self.largada_fija else 0.2))
        txt = marcador('estado', 0, Marker.TEXT_VIEW_FACING, g.ox + 0.3, g.oy + g.alto * g.res + 0.3,
                       (0.25, 0.25, 0.25), (1.0, 1.0, 1.0, 1.0))
        txt.text = (f'{self.estado}  cob {100 * self.cobertura():.0f}%  t={self.tiempo_transcurrido():.0f}s  '
                    f'rep {self.reportes}/{self.cfg["cantidad_victimas"]}'
                    + (f'  sector {self.sectorizador.actual} '
                       f'{100 * self.sectorizador.cobertura(self.visto):.0f}%'
                       if self.sectorizador is not None and self.sectorizador.actual is not None else ''))
        self.pub_marc.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    nodo = Exploracion()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    except RuntimeError:
        # Humble puede fallar al convertir un mensaje si SIGINT acaba de cerrar el contexto.
        if rclpy.ok():
            raise
    finally:
        try:
            nodo.cancelar_actual()
            nodo.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
