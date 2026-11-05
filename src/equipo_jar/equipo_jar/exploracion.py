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
import numpy as np
import rclpy
from collections import deque
from equipo_jar.inflacion_adaptativa import InflacionAdaptativa, PARAMETROS_INFLACION
from equipo_jar.observacion_objetos import ObjetosFueraMapa
from geometry_msgs.msg import PointStamped, PoseArray, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateThroughPoses, NavigateToPose, Spin
from nav2_msgs.msg import SpeedLimit
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Empty, String
from std_srvs.srv import Empty as EmptySrv, Trigger
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray
from ament_index_python.packages import get_package_share_directory
from equipo_jar.exploracion_estados import (
    ESPERANDO_MAPA,
    LOCALIZANDO,
    ACTIVANDO_NAV2,
    ELIGIENDO,
    NAVEGANDO,
    COMPORTAMIENTO,
    DESENCAJANDO,
    RELOCALIZANDO,
    OBSERVANDO_OBJETO,
    ENVIAR_PENDIENTE,
    REGRESAR,
    VOLVIENDO,
    APROXIMANDO,
    TERMINADO,
    FINALIZANDO,
)
from equipo_jar.exploracion_geometria import (
    yaw_a_quat,
    quat_a_yaw,
    ang_norm,
    contar_huecos_pequenos,
    punto_para_mirar,
    dilatar,
    bfs,
    Grilla,
)
from equipo_jar.exploracion_mapa import MapaExploracion
from equipo_jar.exploracion_localizacion import LocalizacionExploracion
from equipo_jar.exploracion_navegacion import NavegacionExploracion
from equipo_jar.exploracion_rutas import RutasExploracion
from equipo_jar.exploracion_cobertura import CoberturaExploracion
from equipo_jar.exploracion_movimiento import MovimientoExploracion
from equipo_jar.exploracion_objetos import ObjetosExploracion
from equipo_jar.exploracion_regreso import RegresoExploracion


QOS_LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
QOS_RELIABLE = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)


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


class Exploracion(
    MapaExploracion,
    LocalizacionExploracion,
    NavegacionExploracion,
    RutasExploracion,
    CoberturaExploracion,
    MovimientoExploracion,
    ObjetosExploracion,
    RegresoExploracion,
    Node,
):

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

    def cb_victima(self, _msg):
        self.reportes += 1
        n = int(self.cfg['cantidad_victimas'])
        self.get_logger().info(f'Reporte en /victimas: {self.reportes}/{n}')
        if self.reportes >= n and self.estado in (ELIGIENDO, NAVEGANDO, COMPORTAMIENTO):
            self.pedir_regreso(f'{n} víctimas reportadas')

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
