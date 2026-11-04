"""Misión del equipo: explorar, reportar las N víctimas y volver a la largada.

Máquina de estados:

    ESPERANDO_MAPA -> LOCALIZANDO -> EXPLORANDO -> VOLVIENDO -> TERMINADO

- ESPERANDO_MAPA: no mueve el robot ni reporta nada hasta recibir /map (§5).
- LOCALIZANDO: reinicializa amcl alrededor del origen con dispersión (los
  jueces ponen el robot *cerca* del origen, §1) y le pide actualizaciones sin
  moverse hasta que la nube converge. Recién ahí guarda la largada (pose del
  robot en `map`), avisa por equipo/localizado (juntar_victimas no reporta
  antes) y espera a que Nav2 acepte objetivos.
- EXPLORANDO: le pide a `Exploracion` el siguiente objetivo y lo manda a Nav2
  (NavigateToPose), de a uno, esperando el resultado. Pasa a VOLVIENDO cuando
  ya se publicaron N reportes en /victimas, cuando no quedan objetivos o
  cuando se acaba el tiempo (dejando margen para volver).
- VOLVIENDO: navega a la largada. Al llegar (o si no puede) publica /done: el
  bono de +100 exige estar a <= 0.25 m del inicio, y /done además corta el
  reloj, que es el desempate (§8).
- TERMINADO: el robot queda quieto.

Los reportes los publica juntar_victimas; acá solo se cuentan.
"""
import math
import time

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from geometry_msgs.msg import (PointStamped, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped, Twist)
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Empty
from std_srvs.srv import Empty as ServicioVacio
from tf2_ros import Buffer, TransformListener

from equipo_jar.exploracion import PARAMETROS_POR_DEFECTO, Exploracion

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

ESPERANDO_MAPA = 'ESPERANDO_MAPA'
LOCALIZANDO = 'LOCALIZANDO'
EXPLORANDO = 'EXPLORANDO'
VOLVIENDO = 'VOLVIENDO'
FINALIZANDO = 'FINALIZANDO'
TERMINADO = 'TERMINADO'


def yaw_de_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NodoEquipo(Node):

    def __init__(self):
        super().__init__('nodo_equipo')
        self.declare_parameter('frecuencia_control_hz', 5.0)
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.declare_parameter('cantidad_victimas', 5)
        # Duración de la corrida (§5) y margen para volver antes de que termine.
        self.declare_parameter('tiempo_corrida_s', 600.0)
        self.declare_parameter('margen_vuelta_s', 30.0)
        # Velocidad media con la que se estima cuánto tarda la vuelta.
        self.declare_parameter('velocidad_vuelta_estimada', 0.2)
        # Distancia a la largada (m) a la que se considera que volvió.
        self.declare_parameter('tolerancia_largada', 0.15)
        self.declare_parameter('intentos_vuelta', 3)
        # Si el robot no se mueve en este tiempo con un objetivo activo, se
        # abandona ese objetivo (el juez corta la corrida a los 60 s quieto).
        self.declare_parameter('tiempo_sin_avance_s', 20.0)
        self.declare_parameter('distancia_encadenar', 0.3)
        self.declare_parameter('inspecciones_maximas', 15)
        # Puntas de pared: si navegando el robot no avanza `avance_minimo_atasco` m en
        # `tiempo_atasco_s`, se le da un subobjetivo sobre el camino global a
        # `distancia_subobjetivo` m (del otro lado de la punta). Con el objetivo local
        # de DWB lejos (forward_prune_distance) el robot va rápido en lo abierto, pero
        # en una vuelta en U el objetivo queda del otro lado de la pared y no se anima.
        self.declare_parameter('tiempo_atasco_s', 4.0)
        self.declare_parameter('avance_minimo_atasco', 0.08)
        self.declare_parameter('distancia_subobjetivo', 0.5)
        self.declare_parameter('tiempo_subobjetivo_s', 8.0)
        self.declare_parameter('tiempo_quieto_vuelta_s', 35.0)   # s de reloj ROS
        self.declare_parameter('distancia_minima_inspeccion', 0.5)   # m
        # Candidatos que se miran apenas aparecen (aunque se esté navegando): a
        # esta distancia (m) o más de una pared del mapa.
        self.declare_parameter('despeje_candidato_claro', 0.2)
        self.declare_parameter('recoveries_para_cortar', 2)   # 0 = nunca se corta
        # Con 1 detección roja se sigue mirando hasta este tiempo (s).
        self.declare_parameter('tiempo_mirar_dudoso_s', 2.0)
        # Tope por objetivo: base + por_metro * distancia en línea recta (s).
        self.declare_parameter('tiempo_objetivo_base_s', 20.0)
        self.declare_parameter('tiempo_objetivo_por_metro_s', 6.0)
        # Segunda pasada: solo si faltan víctimas y sobra este tiempo (s).
        self.declare_parameter('margen_segunda_pasada_s', 120.0)
        self.declare_parameter('alcance_segunda_pasada', 1.8)
        # Con un objetivo activo, si la pose en el mapa (amcl) no se traslada 0.10 m
        # en este tiempo (girar no cuenta), se abandona: patinando contra una
        # pared /odom marca movimiento y el watchdog de arriba no salta.
        self.declare_parameter('tiempo_sin_desplazamiento_s', 15.0)
        # Giro en el lugar para apuntar la cámara a un candidato.
        self.declare_parameter('despeje_giro', 0.22)        # m libres a la pared para girar
        # m libres según el lidar (cajas incluidas) para empezar a girar: la esquina
        # del chasis está a 0.21 m del centro.
        self.declare_parameter('despeje_giro_scan', 0.26)
        self.declare_parameter('giro_minimo_inspeccion', 0.0)   # rad/s (zona muerta del real)
        self.declare_parameter('giro_maximo_inspeccion', 0.8)   # rad/s
        self.declare_parameter('tiempo_giro_inspeccion', 6.0)   # s
        # Pose inicial: dispersión (desvío estándar) de la nube de amcl alrededor
        # del origen, y cuándo se la considera convergida.
        self.declare_parameter('dispersion_inicial_xy', 0.3)
        self.declare_parameter('dispersion_inicial_yaw', 0.6)
        self.declare_parameter('sigma_localizado_xy', 0.05)
        self.declare_parameter('sigma_localizado_yaw', 0.1)
        self.declare_parameter('tiempo_localizacion_s', 5.0)
        # Destrabe: si un objetivo falla (típicamente porque el footprint quedó
        # tocando una pared y Nav2 ya no encuentra trayectorias válidas), se
        # aleja de lo que tiene cerca a esta velocidad durante este tiempo.
        self.declare_parameter('radio_destrabe', 0.4)
        self.declare_parameter('velocidad_destrabe', 0.15)
        self.declare_parameter('tiempo_destrabe_s', 1.0)
        # Objetivos sobre obstáculos que no están en /map (cajas): si en el
        # costmap global en vivo el objetivo tiene más costo que esto (0-100;
        # 99 = el robot ya toca), se lo corre a la celda libre más cercana.
        self.declare_parameter('costo_maximo_objetivo', 50)
        self.declare_parameter('costo_libre_objetivo', 20)
        self.declare_parameter('radio_ajuste_objetivo', 0.8)
        # Si Nav2 no acepta objetivos este tiempo después del mapa (bringup
        # abortado), se le pide al lifecycle_manager que lo reinicie.
        self.declare_parameter('espera_nav2_s', 30.0)
        self.declare_parameter('reintentos_nav2', 2)
        # Nodos de Nav2, en el orden del lifecycle_manager. Si el manager se
        # cuelga (en Humble espera sin timeout la respuesta de cada nodo), la
        # misión los configura y activa uno por uno.
        self.declare_parameter(
            'nodos_nav2',
            ['amcl', 'controller_server', 'planner_server', 'behavior_server', 'bt_navigator'])
        # Candidatos: objetos que el lidar ve fuera del mapa (detector_no_mapeados).
        # Se gira la cámara hacia los que estén a menos de distancia_inspeccion
        # y con vista directa, y se mira tiempo_mirar_s para ver si son rojos.
        self.declare_parameter('radio_candidato', 0.35)
        self.declare_parameter('detecciones_candidato', 3)
        self.declare_parameter('distancia_inspeccion', 3.0)
        self.declare_parameter('tiempo_mirar_s', 2.0)
        self.declare_parameter('espera_reporte_s', 5.0)
        # Parámetros del planificador de exploración (ver exploracion.py).
        for nombre in ('alcance_cobertura', 'despeje_punto_vista', 'despeje_transito',
                       'despeje_caja', 'fraccion_cobertura', 'reintentos_punto_vista'):
            self.declare_parameter(nombre, PARAMETROS_POR_DEFECTO[nombre])

        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value
        self.tiempo_corrida = self.get_parameter('tiempo_corrida_s').value
        self.margen_vuelta = self.get_parameter('margen_vuelta_s').value
        self.velocidad_vuelta = self.get_parameter('velocidad_vuelta_estimada').value
        self.tolerancia_largada = self.get_parameter('tolerancia_largada').value
        self.intentos_vuelta = self.get_parameter('intentos_vuelta').value
        self.tiempo_sin_avance = self.get_parameter('tiempo_sin_avance_s').value
        self.distancia_encadenar = self.get_parameter('distancia_encadenar').value
        self.inspecciones_maximas = self.get_parameter('inspecciones_maximas').value
        self.inspecciones_hechas = 0
        self.tiempo_atasco = self.get_parameter('tiempo_atasco_s').value
        self.avance_minimo_atasco = self.get_parameter('avance_minimo_atasco').value
        self.distancia_subobjetivo = self.get_parameter('distancia_subobjetivo').value
        self.tiempo_subobjetivo = self.get_parameter('tiempo_subobjetivo_s').value
        self.plan_global = None
        self.subobjetivo = None    # {'original': (x, y, yaw), 't': inicio}
        self.ancla_atasco = None   # (x, y, t)
        self.tiempo_quieto_vuelta = self.get_parameter('tiempo_quieto_vuelta_s').value
        self.distancia_minima_inspeccion = self.get_parameter('distancia_minima_inspeccion').value
        self.despeje_candidato_claro = self.get_parameter('despeje_candidato_claro').value
        self.recoveries_para_cortar = self.get_parameter('recoveries_para_cortar').value
        self.tiempo_mirar_dudoso = self.get_parameter('tiempo_mirar_dudoso_s').value
        self.tiempo_objetivo_base = self.get_parameter('tiempo_objetivo_base_s').value
        self.tiempo_objetivo_por_metro = self.get_parameter('tiempo_objetivo_por_metro_s').value
        self.t_inicio_objetivo = None
        self.tope_objetivo = math.inf
        self.id_objetivo = 0
        self.cortar_por_recovery = False
        self.margen_segunda_pasada = self.get_parameter('margen_segunda_pasada_s').value
        self.alcance_segunda_pasada = self.get_parameter('alcance_segunda_pasada').value
        self.segunda_pasada_hecha = False
        self.tiempo_sin_desplazamiento = self.get_parameter('tiempo_sin_desplazamiento_s').value
        self.ancla = None   # (x, y, yaw, t): última pose desde la que hubo desplazamiento
        self.recien_llegado = False   # se acaba de llegar a un punto de inspección
        self.despeje_giro = self.get_parameter('despeje_giro').value
        self.despeje_giro_scan = self.get_parameter('despeje_giro_scan').value
        self.giro_minimo = self.get_parameter('giro_minimo_inspeccion').value
        self.giro_maximo = self.get_parameter('giro_maximo_inspeccion').value
        self.tiempo_giro = self.get_parameter('tiempo_giro_inspeccion').value
        self.dispersion_xy = self.get_parameter('dispersion_inicial_xy').value
        self.dispersion_yaw = self.get_parameter('dispersion_inicial_yaw').value
        self.sigma_localizado_xy = self.get_parameter('sigma_localizado_xy').value
        self.sigma_localizado_yaw = self.get_parameter('sigma_localizado_yaw').value
        self.tiempo_localizacion = self.get_parameter('tiempo_localizacion_s').value
        self.radio_destrabe = self.get_parameter('radio_destrabe').value
        self.velocidad_destrabe = self.get_parameter('velocidad_destrabe').value
        self.tiempo_destrabe = self.get_parameter('tiempo_destrabe_s').value
        self.costo_maximo_objetivo = self.get_parameter('costo_maximo_objetivo').value
        self.costo_libre_objetivo = self.get_parameter('costo_libre_objetivo').value
        self.radio_ajuste_objetivo = self.get_parameter('radio_ajuste_objetivo').value
        self.espera_nav2 = self.get_parameter('espera_nav2_s').value
        self.reintentos_nav2 = self.get_parameter('reintentos_nav2').value
        self.reinicios_nav2 = 0
        self.radio_candidato = self.get_parameter('radio_candidato').value
        self.detecciones_candidato = self.get_parameter('detecciones_candidato').value
        self.distancia_inspeccion = self.get_parameter('distancia_inspeccion').value
        self.tiempo_mirar = self.get_parameter('tiempo_mirar_s').value
        self.espera_reporte = self.get_parameter('espera_reporte_s').value
        self.candidatos = []        # dicts: x, y, n, rojo, estado, inspecciones, visitas
        self.inspeccion = None      # candidato que se está mirando
        self.objetivo_es_inspeccion = False
        self.reportadas = []        # posiciones publicadas en /victimas
        self.t_espera_nav2 = None
        self.gestor_nav2 = self.create_client(
            ManageLifecycleNodes, 'lifecycle_manager/manage_nodes')
        self.nodos_nav2 = list(self.get_parameter('nodos_nav2').value)
        self.clientes_estado = {
            n: (self.create_client(GetState, f'{n}/get_state'),
                self.create_client(ChangeState, f'{n}/change_state'))
            for n in self.nodos_nav2}
        self.arranque_directo = None    # índice del nodo que se está levantando
        self.t_nodo_directo = None      # desde cuándo se intenta levantar ese nodo
        self.bt_activo = False
        self.t_consulta_bt = -math.inf
        self.rechazos = 0
        self.reintentar_objetivo = False
        self.goal_previo = None
        self.t_nav_caido = None
        self.t_ultimo_giro_espera = -math.inf
        self.t_llamada_directa = None   # instante de la llamada pendiente

        self.estado = ESPERANDO_MAPA
        self.mapa = None
        self.t_mapa = None
        self.largada = None
        self.exploracion = None
        self.reportes = 0
        # Localización inicial.
        self.t_pose_inicial = None
        self.pose_inicial_aceptada = False
        self.localizado = False
        self.cov_amcl = None

        # Navegación: un solo objetivo de Nav2 a la vez.
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.goal_handle = None
        self.navegando = False
        self.objetivo = None
        self.vueltas_fallidas = 0
        self.yaw_objetivo = 0.0
        self.ultimo_scan = None
        self.destrabe = None        # (vx, vy, hasta) mientras se destraba
        self.costmap = None         # (costos, info) del costmap global en vivo
        self.t_revision_objetivo = 0.0
        # Para detectar que el robot quedó quieto con un objetivo activo.
        self.ultimo_avance = None
        self.pose_avance = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        # Solo para contar los reportes que publica juntar_victimas.
        self.create_subscription(PointStamped, '/victimas', self.al_reporte, 10)
        self.create_subscription(Odometry, '/odom', self.al_odom, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan', lambda m: setattr(self, 'ultimo_scan', m), qos_profile_sensor_data)
        self.create_subscription(
            OccupancyGrid, 'global_costmap/costmap', self.al_costmap, QOS_MAPA)
        self.create_subscription(
            PoseArray, 'objetos_no_mapeados', self.al_objetos_no_mapeados, 10)
        self.create_subscription(PoseArray, 'objetos_rojos', self.al_objetos_rojos, 10)
        # Camino global de Nav2: para los subobjetivos en las puntas de pared.
        self.create_subscription(Path, 'plan', lambda m: setattr(self, 'plan_global', m), 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        self.pub_vaciar = self.create_publisher(Empty, 'equipo/vaciar_reportes', 10)
        self.t_finalizando = None
        self.pub_pose_inicial = self.create_publisher(
            PoseWithCovarianceStamped, 'equipo/pose_inicial', 10)
        self.pub_localizado = self.create_publisher(Empty, 'equipo/localizado', QOS_MAPA)
        self.create_subscription(
            PoseWithCovarianceStamped, 'amcl_pose', self.al_amcl_pose, QOS_MAPA)
        self.nomotion = self.create_client(ServicioVacio, 'request_nomotion_update')

        periodo = 1.0 / self.get_parameter('frecuencia_control_hz').value
        self.create_timer(periodo, self.ciclo_de_control)
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    # ---------- callbacks ----------

    def al_recibir_mapa(self, mapa):
        if self.mapa is not None:
            return
        self.mapa = mapa
        self.t_mapa = self.ahora()
        self.t_mapa_pared = time.monotonic()
        self.get_logger().info(
            f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
            f'resolución {mapa.info.resolution:.3f} m')
        self.cambiar_estado(LOCALIZANDO)

    def al_reporte(self, msg):
        self.reportes += 1
        self.reportadas.append((msg.point.x, msg.point.y))
        for c in self.candidatos:
            if math.hypot(c['x'] - msg.point.x, c['y'] - msg.point.y) < 0.5:
                c['estado'] = 'REPORTADO'
        self.get_logger().info(f'Reportes publicados: {self.reportes}/{self.cantidad_victimas}')

    def al_amcl_pose(self, msg):
        c = msg.pose.covariance
        self.cov_amcl = (c[0], c[7], c[35])
        if self.estado in (EXPLORANDO, VOLVIENDO):
            self.get_logger().info(
                f'amcl: sigma xy {math.sqrt(max(c[0], c[7])):.3f} m, '
                f'sigma yaw {math.sqrt(c[35]):.3f} rad', throttle_duration_sec=10.0)
        # La primera pose con la dispersión que se le mandó confirma que amcl
        # tomó la pose inicial (antes tenía covarianza cero).
        if self.t_pose_inicial is not None and c[0] > 0.25 * self.dispersion_xy ** 2:
            self.pose_inicial_aceptada = True

    def al_objetos_no_mapeados(self, msg):
        if self.estado not in (EXPLORANDO, VOLVIENDO):
            return
        for p in msg.poses:
            x, y = p.position.x, p.position.y
            for c in self.candidatos:
                if math.hypot(c['x'] - x, c['y'] - y) < self.radio_candidato:
                    c['n'] += 1
                    k = min(c['n'], 20)
                    c['x'] += (x - c['x']) / k
                    c['y'] += (y - c['y']) / k
                    break
            else:
                reportado = any(math.hypot(r[0] - x, r[1] - y) < 0.5 for r in self.reportadas)
                self.candidatos.append({
                    'x': x, 'y': y, 'n': 1, 'rojo': 0, 'inspecciones': 0, 'visitas': 0,
                    'estado': 'REPORTADO' if reportado else 'PENDIENTE'})

    def al_objetos_rojos(self, msg):
        for p in msg.poses:
            x, y = p.position.x, p.position.y
            for c in self.candidatos:
                if math.hypot(c['x'] - x, c['y'] - y) < 0.4:
                    c['rojo'] += 1
                    break
            else:
                # Rojo visto de pasada (o por una rendija) donde el lidar todavía
                # no armó un candidato: también cuenta como candidato, así la
                # misión va a mirarlo de cerca en vez de dejarlo con pocas
                # detecciones.
                if self.estado not in (EXPLORANDO, VOLVIENDO):
                    continue
                reportado = any(math.hypot(r[0] - x, r[1] - y) < 0.5 for r in self.reportadas)
                self.candidatos.append({
                    'x': x, 'y': y, 'n': 1, 'rojo': 1, 'inspecciones': 0, 'visitas': 0,
                    'estado': 'REPORTADO' if reportado else 'PENDIENTE'})
                continue
            if c['n'] < self.detecciones_candidato:
                c['n'] += 1

    def al_costmap(self, msg):
        costos = np.array(msg.data, dtype=np.int16).reshape(msg.info.height, msg.info.width)
        self.costmap = (costos, msg.info)

    def al_odom(self, msg):
        # Cualquier movimiento (traslación o giro) cuenta como avance.
        t = msg.twist.twist
        if abs(t.linear.x) + abs(t.linear.y) > 0.02 or abs(t.angular.z) > 0.05:
            self.ultimo_avance = self.ahora()

    # ---------- utilidades ----------

    def ahora(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def segundos_de_corrida(self):
        # Reloj de pared, como el juez (el límite de 10 min, el bono de 5 y los
        # 60 s quieto). En el sim el tiempo simulado corre más lento.
        return 0.0 if self.t_mapa is None else time.monotonic() - self.t_mapa_pared

    def pose_robot(self):
        """(x, y, yaw) del robot en `map`, o None si todavía no hay tf."""
        try:
            tf = self.tf_buffer.lookup_transform('map', 'base_footprint', Time())
        except Exception:
            return None
        t = tf.transform
        return t.translation.x, t.translation.y, yaw_de_quaternion(t.rotation)

    def cambiar_estado(self, nuevo):
        self.get_logger().info(
            f'[{self.segundos_de_corrida():5.1f} s] {self.estado} -> {nuevo}')
        self.estado = nuevo

    # ---------- navegación ----------

    def costo_en(self, x, y):
        """Costo (0-100, -1 desconocido) del costmap global en (x, y), o None."""
        if self.costmap is None:
            return None
        costos, info = self.costmap
        fila = int((y - info.origin.position.y) / info.resolution)
        col = int((x - info.origin.position.x) / info.resolution)
        if not (0 <= fila < info.height and 0 <= col < info.width):
            return None
        return int(costos[fila, col])

    def ajustar_objetivo(self, x, y, yaw):
        """Corre el objetivo fuera de obstáculos que no están en /map.

        Los puntos de vista se calculan sobre el mapa estático, que no tiene
        las cajas: uno puede caer encima de una. Devuelve el objetivo tal cual,
        corrido a la celda libre más cercana, o None si no hay ninguna cerca.
        """
        costo = self.costo_en(x, y)
        if costo is None or 0 <= costo <= self.costo_maximo_objetivo:
            return x, y, yaw
        costos, info = self.costmap
        r = info.resolution
        fila = int((y - info.origin.position.y) / r)
        col = int((x - info.origin.position.x) / r)
        radio = int(self.radio_ajuste_objetivo / r)
        f0, f1 = max(0, fila - radio), min(info.height, fila + radio + 1)
        c0, c1 = max(0, col - radio), min(info.width, col + radio + 1)
        ventana = costos[f0:f1, c0:c1]
        libres = np.argwhere((ventana >= 0) & (ventana <= self.costo_libre_objetivo))
        if len(libres) == 0:
            return None
        df = libres[:, 0] + f0 - fila
        dc = libres[:, 1] + c0 - col
        k = int(np.argmin(df * df + dc * dc))
        if df[k] * df[k] + dc[k] * dc[k] > radio * radio:
            return None
        nx = info.origin.position.x + (col + dc[k] + 0.5) * r
        ny = info.origin.position.y + (fila + df[k] + 0.5) * r
        self.get_logger().info(
            f'Objetivo ({x:.2f}, {y:.2f}) sobre un obstáculo (costo {costo}): '
            f'se corre a ({nx:.2f}, {ny:.2f})')
        return nx, ny, yaw

    def ir_a(self, x, y, yaw, nuevo=True):
        """Manda un objetivo a Nav2. Devuelve False si Nav2 todavía no está listo.

        nuevo=False cuando se reenvía el mismo objetivo (corrido o reintentado):
        no reinicia el tope de tiempo del objetivo.
        """
        if not self.nav.server_is_ready():
            return False
        # Si había otro objetivo activo, Nav2 lo reemplaza; su resultado se ignora.
        self.goal_handle = None
        self.id_objetivo += 1
        if nuevo or self.t_inicio_objetivo is None:
            pose = self.pose_robot()
            distancia = 0.0 if pose is None else math.hypot(x - pose[0], y - pose[1])
            self.t_inicio_objetivo = self.ahora()
            self.tope_objetivo = (self.tiempo_objetivo_base
                                  + self.tiempo_objetivo_por_metro * distancia)
        meta = NavigateToPose.Goal()
        meta.pose = PoseStamped()
        meta.pose.header.frame_id = 'map'
        meta.pose.header.stamp = self.get_clock().now().to_msg()
        meta.pose.pose.position.x = float(x)
        meta.pose.pose.position.y = float(y)
        meta.pose.pose.orientation.z = math.sin(yaw / 2.0)
        meta.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.objetivo = (x, y)
        self.yaw_objetivo = yaw
        self.navegando = True
        if nuevo:
            # Un reenvío (subobjetivo, objetivo corrido, reintento) no reinicia los
            # watchdogs: si no, el ciclo subobjetivo/retomar cada 4-8 s los anulaba.
            self.ultimo_avance = self.ahora()
            self.ancla = None
        ident = self.id_objetivo
        self.cortar_por_recovery = False
        self.nav.send_goal_async(
            meta, feedback_callback=lambda fb: self._feedback(ident, fb)).add_done_callback(
            lambda f: self._objetivo_aceptado(ident, f))
        return True

    def _feedback(self, ident, fb):
        # Los recoveries del BT (borrar costmaps, Spin, Wait 5 s, BackUp) cuestan
        # 20-35 s y casi nunca destraban. number_of_recoveries cuenta también los
        # borrados de costmap; con 2 se probó mejor que con 1 (maze_2 199 -> 130 s)
        # sin perder puntos. En la vuelta no se corta.
        # En la vuelta no: ahí los recoveries de Nav2 sí sirven y no hay otro objetivo.
        if (ident == self.id_objetivo and self.estado == EXPLORANDO
                and fb.feedback.number_of_recoveries >= self.recoveries_para_cortar):
            self.cortar_por_recovery = self.recoveries_para_cortar > 0

    def _objetivo_aceptado(self, ident, futuro):
        gh = futuro.result()
        if ident != self.id_objetivo:
            # Respuesta de un objetivo que ya se reemplazó o se canceló: si Nav2
            # lo aceptó y no hay otro en curso, que no siga manejando.
            if gh.accepted and not self.navegando:
                gh.cancel_goal_async()
            return
        if not gh.accepted:
            # bt_navigator rechaza mientras no está activo: se reintenta el
            # mismo objetivo en el próximo ciclo antes de darlo por fallido.
            self.rechazos += 1
            self.get_logger().warn(f'Nav2 rechazó el objetivo ({self.rechazos}/3)')
            if self.rechazos < 3:
                self.reintentar_objetivo = True
                return
            self.rechazos = 0
            self.navegando = False
            self.al_terminar_objetivo(False)
            return
        self.rechazos = 0
        self.goal_handle = gh
        self.goal_previo = None   # Nav2 ya reemplazó el objetivo anterior
        gh.get_result_async().add_done_callback(lambda r: self._objetivo_terminado(gh, r))

    def _objetivo_terminado(self, gh, futuro):
        if gh is not self.goal_handle:
            return  # resultado de un objetivo que ya se reemplazó o canceló
        self.goal_handle = None
        self.navegando = False
        self.al_terminar_objetivo(futuro.result().status == GoalStatus.STATUS_SUCCEEDED)

    def cancelar(self):
        self.id_objetivo += 1   # las respuestas pendientes quedan viejas
        self.subobjetivo = None
        self.ancla_atasco = None
        self.reintentar_objetivo = False
        for gh in (self.goal_handle, self.goal_previo):
            if gh is not None:
                gh.cancel_goal_async()
        self.goal_handle = None
        self.goal_previo = None
        self.navegando = False

    def al_terminar_objetivo(self, exito):
        if self.subobjetivo is not None:
            self.retomar_objetivo('llegó' if exito else 'falló')
            return
        if not exito:
            self.empezar_destrabe()
        if self.estado == EXPLORANDO and self.objetivo_es_inspeccion:
            self.objetivo_es_inspeccion = False
            # Se llegó a mirar un candidato: si su posición estimada se corrió y
            # quedó un poco más cerca que el mínimo, igual se lo mira desde acá.
            self.recien_llegado = exito
        elif self.estado == EXPLORANDO:
            if not exito:
                self.get_logger().info(
                    f'No se llegó a ({self.objetivo[0]:.2f}, {self.objetivo[1]:.2f}) '
                    f'desde {self.texto_pose()}: se sigue con el próximo')
            self.exploracion.objetivo_terminado(exito)
        elif self.estado == VOLVIENDO:
            distancia = self.distancia_a_largada()
            # Nav2 puede dar "llegué" con la pose de un instante antes de que amcl
            # se corrija: el bono de vuelta pide <= 0.25 m, así que se mira de nuevo.
            if distancia <= self.tolerancia_largada or (
                    distancia <= 0.2 and (exito or self.vueltas_fallidas >= 1)):
                self.terminar()
            else:
                if exito:
                    self.get_logger().info(
                        f'Nav2 dice que llegó, pero amcl da {distancia:.2f} m: se reintenta')
                # Se sigue intentando mientras haya tiempo: rendirse lejos de la
                # largada pierde el +100 (con poca CPU el controlador falla seguido
                # y los intentos se gastaban en segundos).
                self.vueltas_fallidas += 1
                if (self.vueltas_fallidas >= self.intentos_vuelta
                        and self.segundos_de_corrida() > self.tiempo_corrida - 20.0):
                    self.get_logger().warn('No se pudo volver a la largada; se termina igual')
                    self.terminar()

    def despeje_scan(self):
        """Distancia (m) del centro del robot al punto más cercano del scan (cajas incluidas).

        El mapa no tiene las cajas: para girar en el lugar al lado de una hay que
        mirar el lidar. Los puntos a menos de 0.12 m son rebotes del propio robot.
        """
        scan = self.ultimo_scan
        if scan is None:
            return math.inf
        try:
            tf = self.tf_buffer.lookup_transform('base_footprint', scan.header.frame_id, Time())
        except Exception:
            return math.inf
        yaw_laser = yaw_de_quaternion(tf.transform.rotation)
        tx, ty = tf.transform.translation.x, tf.transform.translation.y
        r = np.asarray(scan.ranges)
        a = scan.angle_min + np.arange(len(r)) * scan.angle_increment + yaw_laser
        ok = np.isfinite(r) & (r > scan.range_min)
        d = np.hypot(tx + r[ok] * np.cos(a[ok]), ty + r[ok] * np.sin(a[ok]))
        d = d[d > 0.12]
        return float(d.min()) if len(d) else math.inf

    def empezar_destrabe(self):
        """Calcula hacia dónde alejarse de lo que el lidar ve cerca del robot.

        Suma, para cada punto a menos de radio_destrabe, un empuje en sentido
        contrario (más fuerte cuanto más cerca). El robot es mecanum: se mueve
        en esa dirección sin girar, así que sirve aunque la pared esté al costado
        (el BackUp de Nav2 solo retrocede y con el footprint ya tocando falla).
        """
        scan = self.ultimo_scan
        if scan is None:
            return
        try:
            tf = self.tf_buffer.lookup_transform('base_footprint', scan.header.frame_id, Time())
        except Exception:
            return
        yaw_laser = yaw_de_quaternion(tf.transform.rotation)
        tx, ty = tf.transform.translation.x, tf.transform.translation.y
        ex = ey = 0.0
        puntos = []
        for i, r in enumerate(scan.ranges):
            if not (scan.range_min < r < scan.range_max):
                continue
            a = scan.angle_min + i * scan.angle_increment + yaw_laser
            px, py = tx + r * math.cos(a), ty + r * math.sin(a)
            d = math.hypot(px, py)
            if 0.12 < d < 1.0:
                puntos.append((px, py))
            if 0.05 < d < self.radio_destrabe:
                peso = (self.radio_destrabe - d) / d
                ex -= px * peso
                ey -= py * peso
        norma = math.hypot(ex, ey)
        if norma < 1e-6:
            return
        ex, ey = ex / norma, ey / norma
        # El empuje promedio puede mandar al robot contra una punta de pared fina
        # (una caja al lado aporta muchos más puntos que la punta). Se elige la
        # dirección, entre 16, con más espacio libre para el ancho del robot; ante
        # empates, la más parecida al empuje.
        mejor, libre_mejor = (ex, ey), -1.0
        for k in range(16):
            ux, uy = math.cos(k * math.pi / 8), math.sin(k * math.pi / 8)
            libre = 0.6
            for px, py in puntos:
                adelante = px * ux + py * uy
                if adelante > 0 and abs(py * ux - px * uy) < 0.17:
                    libre = min(libre, adelante - 0.18)
            puntaje = libre + 0.03 * (ux * ex + uy * ey)
            if puntaje > libre_mejor:
                mejor, libre_mejor = (ux, uy), puntaje
        if libre_mejor > 0.05:
            ex, ey = mejor
        v = self.velocidad_destrabe
        self.destrabe = (v * ex, v * ey, self.ahora() + self.tiempo_destrabe)
        self.get_logger().info(
            f'Destrabe en {self.texto_pose()}: hacia ({ex:.2f}, {ey:.2f}), el lado más libre')

    def texto_pose(self):
        pose = self.pose_robot()
        if pose is None:
            return '(sin pose)'
        return f'({pose[0]:.2f}, {pose[1]:.2f}, {math.degrees(pose[2]):.0f}°)'

    def distancia_a_largada(self):
        pose = self.pose_robot()
        if pose is None or self.largada is None:
            return math.inf
        return math.hypot(pose[0] - self.largada[0], pose[1] - self.largada[1])

    # ---------- máquina de estados ----------

    def ciclo_de_control(self):
        # Un error en un ciclo no puede matar la misión: sin el nodo no hay /done.
        try:
            self._ciclo_de_control()
        except Exception as error:   # noqa: B902
            self.get_logger().error(f'Error en el ciclo de control: {error!r}',
                                    throttle_duration_sec=2.0)

    def _ciclo_de_control(self):
        if self.estado in (ESPERANDO_MAPA, TERMINADO):
            return  # sin mapa no se mueve ni reporta (§5); al terminar, quieto
        if self.estado == FINALIZANDO:
            # Se le da un momento a juntar_victimas para publicar los reportes
            # que sobran antes del /done.
            self.pub_cmd_vel.publish(Twist())
            if self.ahora() - self.t_finalizando >= 1.5:
                self.publicar_done()
            return
        if self.destrabe is not None:
            vx, vy, hasta = self.destrabe
            cmd = Twist()
            if self.ahora() < hasta:
                cmd.linear.x, cmd.linear.y = vx, vy
            else:
                self.destrabe = None
            self.pub_cmd_vel.publish(cmd)
            if self.destrabe is not None:
                return
        pose = self.pose_robot()
        if self.estado == LOCALIZANDO and self.segundos_de_corrida() > 45.0:
            # Nav2 todavía no está: el juez corta a los 60 s sin moverse. Un giro
            # corto cada 30 s (en el lugar, ya con el mapa publicado) lo evita.
            if self.segundos_de_corrida() - self.t_ultimo_giro_espera > 30.0:
                self.t_ultimo_giro_espera = self.segundos_de_corrida()
                self.get_logger().warn('Nav2 sigue sin arrancar: giro corto para no quedar quieto')
            desde_giro = self.segundos_de_corrida() - self.t_ultimo_giro_espera
            if desde_giro < 1.5:
                cmd = Twist()   # 1 s girando y después cero: el driver real puede no frenar solo
                cmd.angular.z = 0.5 if desde_giro < 1.0 else 0.0
                self.pub_cmd_vel.publish(cmd)
        if self.estado == LOCALIZANDO:
            self.consultar_bt_navigator()
            if not self.nav2_listo() or self.arranque_directo is not None:
                # También mientras se espera a amcl: si el bringup abortó,
                # amcl tampoco está activo y la localización no avanza.
                self.quizas_reiniciar_nav2()
            if not self.localizado:
                self.localizar()
                return
            if pose is None or not self.nav2_listo() or self.arranque_directo is not None:
                self.get_logger().info(
                    'Esperando ' + ('la tf map -> base_footprint' if pose is None else 'a Nav2'),
                    throttle_duration_sec=5.0)
                return
            self.largada = pose
            self.exploracion = Exploracion(
                self.mapa, self.get_logger(), self.parametros_exploracion())
            self.exploracion.planificar(pose)
            self.cambiar_estado(EXPLORANDO)

        if pose is None:
            self.get_logger().warn('Sin tf map -> base_footprint', throttle_duration_sec=5.0)
            return

        if self.reintentar_objetivo:
            self.reintentar_objetivo = False
            self.ir_a(self.objetivo[0], self.objetivo[1], self.yaw_objetivo, nuevo=False)

        if self.navegando and self.estado in (EXPLORANDO, VOLVIENDO):
            self.revisar_atasco(pose)

        if self.navegando and self.cortar_por_recovery:
            self.get_logger().warn(f'Nav2 entró en recovery en {self.texto_pose()}: se corta')
            self.cancelar()
            self.al_terminar_objetivo(False)
            return

        if self.navegando and self.ahora() - self.t_inicio_objetivo > self.tope_objetivo:
            self.get_logger().warn(
                f'Objetivo sin alcanzar en {self.tope_objetivo:.0f} s, en '
                f'{self.texto_pose()}: se abandona')
            self.cancelar()
            self.al_terminar_objetivo(False)
            return   # el destrabe usa /cmd_vel en este ciclo

        if self.navegando and self.sin_desplazamiento(pose):
            self.get_logger().warn(
                f'{self.tiempo_sin_desplazamiento:.0f} s sin desplazarse en el mapa en '
                f'{self.texto_pose()}: se abandona el objetivo')
            self.cancelar()
            self.al_terminar_objetivo(False)
            return

        if self.navegando and self.ahora() - self.ultimo_avance > self.tiempo_sin_avance:
            self.get_logger().warn(
                f'{self.tiempo_sin_avance:.0f} s sin moverse en {self.texto_pose()}: '
                'se abandona el objetivo')
            self.cancelar()
            self.al_terminar_objetivo(False)
            return

        if self.estado == VOLVIENDO and self.vuelta_sin_salida(pose):
            self.terminar()
            return

        if self.estado == EXPLORANDO:
            if self.reportes >= self.cantidad_victimas:
                self.get_logger().info('Se reportaron todas las víctimas')
                self.volver()
            elif self.segundos_de_corrida() > self.limite_exploracion(pose):
                self.get_logger().info('Se acaba el tiempo')
                self.volver()
            elif self.inspeccion is not None:
                self.paso_inspeccion(pose)
            elif self.empezar_inspeccion(pose, solo_claros=self.navegando):
                # (también los pegados a una pared si la cámara ya vio rojo ahí: las
                # víctimas pueden estar contra la pared)
                # Apenas hay un candidato claro a la vista se lo mira, aunque se esté
                # yendo a un punto de vista: ese objetivo se cancela sin darlo por
                # visitado y se retoma después de mirar. Los pegados a una pared
                # (casi siempre pared corrida por el error de amcl) esperan a que
                # el robot esté detenido.
                pass
            elif not self.navegando:
                siguiente = self.exploracion.siguiente(pose)
                if siguiente is None and self.ir_a_candidato_pendiente(pose):
                    pass
                elif siguiente is None and self.quizas_segunda_pasada(pose):
                    pass
                elif siguiente is None:
                    self.get_logger().info('No quedan objetivos de exploración')
                    self.volver()
                else:
                    ajustado = self.ajustar_objetivo(*siguiente)
                    if ajustado is None:
                        self.get_logger().info(
                            f'Objetivo ({siguiente[0]:.2f}, {siguiente[1]:.2f}) tapado por un '
                            'obstáculo: se saltea')
                        self.exploracion.objetivo_terminado(False)
                    else:
                        self.ir_a(*ajustado)
            elif self.cerca_del_punto_de_vista(pose):
                # Se da por visitado sin frenar: el próximo objetivo reemplaza
                # a este en Nav2 y el robot sigue de largo. Se guarda por si lo
                # que sigue no es otro objetivo (inspección, vuelta): ahí se cancela.
                self.goal_previo = self.goal_handle
                self.goal_handle = None
                self.navegando = False
                self.exploracion.objetivo_terminado(True)
            else:
                self.revisar_objetivo_activo()
        elif self.estado == VOLVIENDO and not self.navegando:
            if self.distancia_a_largada() <= self.tolerancia_largada:
                self.terminar()
            elif self.ir_a(self.largada[0], self.largada[1], self.largada[2]):
                self.t_nav_caido = None
            else:
                # Nav2 no responde: no se espera para siempre sin /done.
                if self.t_nav_caido is None:
                    self.t_nav_caido = self.ahora()
                elif self.ahora() - self.t_nav_caido > 10.0:
                    self.get_logger().warn('Nav2 no responde en la vuelta: se termina igual')
                    self.terminar()

    def vuelta_sin_salida(self, pose):
        """En la vuelta, /done igual si se acaba la corrida o el robot no se mueve."""
        ahora = self.segundos_de_corrida()   # reloj de pared, como el juez
        if ahora > self.tiempo_corrida - 10.0:
            self.get_logger().warn('Se acaba la corrida: /done donde esté')
            return True
        # Quieto como lo mide el juez (velocidad de /odom, girar también cuenta): el
        # juez corta a los 60 s y sin /done se pierde el +100. Mientras Nav2 gire en
        # sus recoveries no se corta (antes se cortaba sin traslación y se perdía).
        if (self.ultimo_avance is not None
                and self.ahora() - self.ultimo_avance > self.tiempo_quieto_vuelta):
            self.get_logger().warn(
                f'{self.tiempo_quieto_vuelta:.0f} s quieto en la vuelta: /done donde esté')
            return True
        return False

    def sin_desplazamiento(self, pose):
        ahora = self.ahora()
        if self.ancla is None:
            self.ancla = (*pose, ahora)
            return False
        x, y, _, t = self.ancla
        # Solo traslación: los Spin de los recoveries de Nav2 no son avance.
        if math.hypot(pose[0] - x, pose[1] - y) > 0.10:
            self.ancla = (*pose, ahora)
            return False
        # En la vuelta se deja más margen a los recoveries de Nav2 (Spin + Wait).
        tope = self.tiempo_sin_desplazamiento * (2.0 if self.estado == VOLVIENDO else 1.0)
        return ahora - t > tope

    def parametros_exploracion(self):
        return {nombre: self.get_parameter(nombre).value
                for nombre in ('alcance_cobertura', 'despeje_punto_vista',
                               'despeje_transito', 'despeje_caja',
                               'fraccion_cobertura', 'reintentos_punto_vista')}

    def quizas_segunda_pasada(self, pose):
        """Si faltan víctimas y sobra tiempo, otra ruta con alcance corto (más cerca de todo)."""
        if self.segunda_pasada_hecha or self.reportes >= self.cantidad_victimas:
            return False
        sobra = self.limite_exploracion(pose) - self.segundos_de_corrida()
        if sobra < self.margen_segunda_pasada:
            return False
        self.segunda_pasada_hecha = True
        parametros = self.parametros_exploracion()
        parametros['alcance_cobertura'] = self.alcance_segunda_pasada
        nueva = Exploracion(self.mapa, self.get_logger(), parametros)
        nueva.planificar(pose)
        # La distancia de vuelta se sigue midiendo a la largada, no a donde se replanificó.
        nueva.distancias_largada = self.exploracion.distancias_largada
        nueva.celda_largada = self.exploracion.celda_largada
        if not nueva.pendientes:
            return False
        self.get_logger().info(
            f'Faltan víctimas y sobran {sobra:.0f} s: segunda pasada con alcance '
            f'{self.alcance_segunda_pasada:.1f} m')
        self.exploracion = nueva
        return True

    def revisar_atasco(self, pose):
        """Subobjetivo cercano sobre el camino global si el robot no avanza (punta de pared)."""
        ahora = self.ahora()
        if self.subobjetivo is not None:
            if ahora - self.subobjetivo['t'] > self.tiempo_subobjetivo:
                self.retomar_objetivo('se acabó el tiempo')
            return
        ancla = self.ancla_atasco
        if ancla is None or math.hypot(pose[0] - ancla[0], pose[1] - ancla[1]) > \
                self.avance_minimo_atasco:
            self.ancla_atasco = (pose[0], pose[1], ahora)
            return
        if ahora - self.ancla_atasco[2] < self.tiempo_atasco:
            return
        self.ancla_atasco = None
        punto = self.punto_adelante_en_plan(pose)
        if punto is None:
            return
        original = (self.objetivo[0], self.objetivo[1], self.yaw_objetivo)
        self.get_logger().info(
            f'Atascado en {self.texto_pose()}: subobjetivo ({punto[0]:.2f}, {punto[1]:.2f}) '
            'sobre el camino global')
        es_inspeccion = self.objetivo_es_inspeccion
        if self.ir_a(punto[0], punto[1], punto[2], nuevo=False):
            self.subobjetivo = {'original': original, 't': ahora, 'inspeccion': es_inspeccion}

    def punto_adelante_en_plan(self, pose):
        """Punto del camino global a distancia_subobjetivo (por el camino) del robot."""
        plan = self.plan_global
        if plan is None or len(plan.poses) < 3:
            return None
        xy = [(p.pose.position.x, p.pose.position.y) for p in plan.poses]
        i0 = min(range(len(xy)), key=lambda i: math.hypot(xy[i][0] - pose[0], xy[i][1] - pose[1]))
        if math.hypot(xy[i0][0] - pose[0], xy[i0][1] - pose[1]) > 0.3:
            return None   # el plan no es de donde está el robot
        recorrido = 0.0
        for i in range(i0 + 1, len(xy)):
            recorrido += math.dist(xy[i - 1], xy[i])
            if recorrido >= self.distancia_subobjetivo:
                yaw = math.atan2(xy[i][1] - xy[i - 1][1], xy[i][0] - xy[i - 1][0])
                return xy[i][0], xy[i][1], yaw
        return None   # el objetivo ya está más cerca que eso

    def retomar_objetivo(self, motivo):
        sub = self.subobjetivo
        self.subobjetivo = None
        self.ancla_atasco = None
        self.get_logger().info(f'Subobjetivo: {motivo}; se retoma el objetivo')
        x, y, yaw = sub['original']
        self.objetivo_es_inspeccion = sub['inspeccion']
        if not self.ir_a(x, y, yaw, nuevo=False):
            self.navegando = False

    def cerca_del_punto_de_vista(self, pose):
        return (pose is not None and not self.objetivo_es_inspeccion
                and self.subobjetivo is None
                and math.dist(pose[:2], self.objetivo) < self.distancia_encadenar
                and self.exploracion.linea_de_vista(pose[:2], self.objetivo))

    def localizar(self):
        """Pose inicial con dispersión y actualizaciones de amcl sin moverse."""
        ahora = self.ahora()
        if self.t_pose_inicial is None or (
                not self.pose_inicial_aceptada and ahora - self.t_pose_inicial > 1.0):
            if self.pub_pose_inicial.get_subscription_count() == 0:
                self.get_logger().info('Esperando a amcl', throttle_duration_sec=5.0)
                return
            self.publicar_pose_inicial()
            if self.t_pose_inicial is None:
                self.t_pose_inicial = ahora
            return
        if self.nomotion.service_is_ready():
            self.nomotion.call_async(ServicioVacio.Request())
        convergio = (self.pose_inicial_aceptada and self.cov_amcl is not None
                     and max(self.cov_amcl[0], self.cov_amcl[1]) < self.sigma_localizado_xy ** 2
                     and self.cov_amcl[2] < self.sigma_localizado_yaw ** 2)
        vencido = ahora - self.t_pose_inicial > self.tiempo_localizacion
        if convergio or vencido:
            cov = self.cov_amcl or (math.nan,) * 3
            self.get_logger().info(
                f'Localizado ({"convergió" if convergio else "se venció el tiempo"}): '
                f'sigma xy {math.sqrt(max(cov[0], cov[1])):.3f} m, '
                f'sigma yaw {math.sqrt(cov[2]):.3f} rad')
            self.localizado = True
            self.pub_localizado.publish(Empty())

    def quizas_reiniciar_nav2(self):
        """Si Nav2 no arrancó, levanta los nodos uno por uno y, si no alcanza, RESET y STARTUP."""
        ahora = self.ahora()
        if self.t_espera_nav2 is None:
            self.t_espera_nav2 = ahora
        if self.arranque_directo is not None:
            self.paso_arranque_directo()
            return
        if ahora - self.t_espera_nav2 >= self.espera_nav2 and self.reinicios_nav2 == 0:
            self.get_logger().warn('Nav2 no arrancó: se levantan sus nodos desde la misión')
            self.arranque_directo = 0
            self.t_llamada_directa = None
            self.t_nodo_directo = None
            self.reinicios_nav2 = 1
            self.t_espera_nav2 = ahora
            return
        if (ahora - self.t_espera_nav2 < self.espera_nav2
                or self.reinicios_nav2 >= self.reintentos_nav2
                or not self.gestor_nav2.service_is_ready()):
            return
        self.reinicios_nav2 += 1
        self.t_espera_nav2 = ahora
        self.get_logger().warn(
            f'Nav2 no arrancó: se reinicia (intento {self.reinicios_nav2}/{self.reintentos_nav2})')
        reset = ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.RESET)
        self.gestor_nav2.call_async(reset).add_done_callback(self._arrancar_nav2)

    def consultar_bt_navigator(self):
        """Una vez por segundo, el estado de bt_navigator (rechaza objetivos si no está activo)."""
        if 'bt_navigator' not in self.clientes_estado:
            return
        if self.ahora() - self.t_consulta_bt < 1.0:
            return
        cliente, _ = self.clientes_estado['bt_navigator']
        if not cliente.service_is_ready():
            return
        self.t_consulta_bt = self.ahora()
        cliente.call_async(GetState.Request()).add_done_callback(self._estado_bt)

    def _estado_bt(self, futuro):
        if futuro.result() is not None:
            self.bt_activo = futuro.result().current_state.id == State.PRIMARY_STATE_ACTIVE

    def nav2_listo(self):
        if 'bt_navigator' not in self.clientes_estado:
            return self.nav.server_is_ready()
        return self.nav.server_is_ready() and self.bt_activo

    def paso_arranque_directo(self):
        """Lleva el nodo actual a 'active' (configure + activate) y pasa al siguiente."""
        if self.bt_activo and self.arranque_directo < len(self.nodos_nav2) - 1:
            # El lifecycle_manager terminó por su cuenta.
            self.arranque_directo = len(self.nodos_nav2)
        if self.arranque_directo >= len(self.nodos_nav2):
            self.arranque_directo = None
            self.get_logger().info('Nodos de Nav2 levantados desde la misión')
            return
        if self.t_nodo_directo is None:
            self.t_nodo_directo = self.ahora()
        elif self.ahora() - self.t_nodo_directo > 15.0:
            self.get_logger().warn(
                f'{self.nodos_nav2[self.arranque_directo]} no se pudo levantar: se sigue')
            self.arranque_directo += 1
            self.t_nodo_directo = None
            self.t_llamada_directa = None
            return
        if self.t_llamada_directa is not None:
            if self.ahora() - self.t_llamada_directa < 5.0:
                return   # esperando respuesta
            self.get_logger().warn(
                f'{self.nodos_nav2[self.arranque_directo]} no responde: se reintenta')
        cliente, _ = self.clientes_estado[self.nodos_nav2[self.arranque_directo]]
        if not cliente.service_is_ready():
            return
        self.t_llamada_directa = self.ahora()
        indice = self.arranque_directo
        cliente.call_async(GetState.Request()).add_done_callback(
            lambda f: self._estado_nodo(indice, f))

    def _estado_nodo(self, indice, futuro):
        if indice != self.arranque_directo or futuro.result() is None:
            return
        nodo = self.nodos_nav2[indice]
        estado = futuro.result().current_state.id
        if estado >= 10:
            # En medio de una transición (configuring, activating, ...): se espera.
            self.t_llamada_directa = self.ahora() - 4.0
            return
        transicion = {State.PRIMARY_STATE_UNCONFIGURED: Transition.TRANSITION_CONFIGURE,
                      State.PRIMARY_STATE_INACTIVE: Transition.TRANSITION_ACTIVATE}.get(estado)
        if transicion is None:
            # Activo (o en un estado del que no se sale con configure/activate).
            self.arranque_directo += 1
            self.t_llamada_directa = None
            self.t_nodo_directo = None
            return
        self.get_logger().info(f'{nodo}: transición {transicion}')
        _, cambio = self.clientes_estado[nodo]
        pedido = ChangeState.Request()
        pedido.transition.id = transicion
        cambio.call_async(pedido).add_done_callback(lambda f: self._transicion_hecha(indice))

    def _transicion_hecha(self, indice):
        if indice == self.arranque_directo:
            self.t_llamada_directa = None   # se vuelve a mirar el estado del mismo nodo

    def _arrancar_nav2(self, _futuro):
        inicio = ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.STARTUP)
        self.gestor_nav2.call_async(inicio).add_done_callback(self._nav2_reiniciado)

    def _nav2_reiniciado(self, futuro):
        ok = futuro.result() is not None and futuro.result().success
        self.get_logger().info(f'Reinicio de Nav2: {"ok" if ok else "falló"}')
        # amcl arranca otra vez con su pose por defecto: localizar de nuevo.
        self.localizado = False
        self.t_pose_inicial = None
        self.pose_inicial_aceptada = False

    def publicar_pose_inicial(self):
        msg = PoseWithCovarianceStamped()
        msg.header.frame_id = 'map'
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance[0] = self.dispersion_xy ** 2
        msg.pose.covariance[7] = self.dispersion_xy ** 2
        msg.pose.covariance[35] = self.dispersion_yaw ** 2
        self.pub_pose_inicial.publish(msg)
        self.get_logger().info(
            f'Pose inicial (0, 0, 0) con sigma {self.dispersion_xy} m / {self.dispersion_yaw} rad')

    def revisar_objetivo_activo(self):
        """Una vez por segundo: si el lidar descubrió algo sobre el objetivo, correrlo."""
        if self.ahora() - self.t_revision_objetivo < 1.0 or self.objetivo is None:
            return
        self.t_revision_objetivo = self.ahora()
        x, y = self.objetivo
        costo = self.costo_en(x, y)
        if costo is None or 0 <= costo <= self.costo_maximo_objetivo:
            return
        ajustado = self.ajustar_objetivo(x, y, self.yaw_objetivo)
        if ajustado is None:
            self.get_logger().info('El objetivo activo quedó tapado: se abandona')
            self.cancelar()
            self.al_terminar_objetivo(False)
        elif (ajustado[0], ajustado[1]) != (x, y):
            self.ir_a(*ajustado, nuevo=False)

    # ---------- candidatos ----------

    def pendientes(self):
        if self.inspecciones_hechas >= self.inspecciones_maximas:
            return []   # con un mapa corrido puede haber muchos candidatos falsos
        return [c for c in self.candidatos
                if c['estado'] == 'PENDIENTE' and c['n'] >= self.detecciones_candidato
                and c['inspecciones'] < 1]   # una sola vez por candidato

    def empezar_inspeccion(self, pose, solo_claros=False):
        """Si hay un candidato pendiente a la vista, empieza a girar hacia él."""
        if pose is None or self.exploracion.despeje(pose[0], pose[1]) < self.despeje_giro:
            return False   # sin lugar para girar en el lugar
        if self.despeje_scan() < self.despeje_giro_scan:
            return False   # una caja (que no está en el mapa) demasiado cerca para girar
        # De muy cerca la caja tapa toda la imagen y los puntos del lidar sobre ella
        # caen fuera del cuadro: daba "no es rojo" con una víctima a 0.33 m.
        minima = 0.35 if self.recien_llegado else self.distancia_minima_inspeccion
        self.recien_llegado = False
        visibles = [c for c in self.pendientes()
                    if minima <= math.hypot(c['x'] - pose[0], c['y'] - pose[1])
                    < self.distancia_inspeccion
                    and self.exploracion.linea_de_vista(pose[:2], (c['x'], c['y']))
                    and ('pose_corte' not in c or math.dist(pose[:2], c['pose_corte']) > 0.5)
                    and (not solo_claros or c['rojo'] > 0
                         or self.exploracion.despeje(c['x'], c['y'])
                         >= self.despeje_candidato_claro)]
        if not visibles:
            return False
        # El que pide menos giro (evita el zig-zag entre candidatos).

        def giro(c):
            err = math.atan2(c['y'] - pose[1], c['x'] - pose[0]) - pose[2]
            return abs(math.atan2(math.sin(err), math.cos(err)))
        c = min(visibles, key=giro)
        self.cancelar()   # el objetivo en curso (o el encadenado) deja de manejar
        self.objetivo_es_inspeccion = False
        self.inspecciones_hechas += 1
        self.inspeccion = {'c': c, 'fase': 'girar', 't': self.ahora()}
        self.get_logger().info(
            f'Inspeccionando candidato en ({c["x"]:.2f}, {c["y"]:.2f}) desde {self.texto_pose()}')
        return True

    def paso_inspeccion(self, pose):
        """Gira la cámara hacia el candidato y mira si es rojo."""
        insp = self.inspeccion
        c = insp['c']
        cmd = Twist()
        ahora = self.ahora()
        if pose is not None and insp['fase'] == 'girar':
            err = math.atan2(c['y'] - pose[1], c['x'] - pose[0]) - pose[2]
            err = math.atan2(math.sin(err), math.cos(err))
            cerca = self.despeje_scan()
            corte = self.despeje_giro_scan - 0.03   # 0 en params apaga las dos condiciones
            sin_tiempo = ahora - insp['t'] >= self.tiempo_giro and abs(err) > 0.5
            if abs(err) > 0.25 and (cerca < corte or sin_tiempo):
                # Sin lugar para seguir girando: mirar desde acá sería mirar para otro
                # lado y descartar una víctima. No cuenta como inspección; se reintenta
                # cuando el robot esté en otro lugar.
                self.get_logger().info(
                    f'Giro cortado hacia ({c["x"]:.2f}, {c["y"]:.2f}) en {self.texto_pose()}: '
                    + (f'algo a {cerca:.2f} m' if cerca < corte else 'se acabó el tiempo')
                    + '; se mira desde otro lugar')
                c['pose_corte'] = pose[:2]
                self.inspecciones_hechas -= 1
                self.inspeccion = None
                self.pub_cmd_vel.publish(Twist())
                return
            if abs(err) > 0.25 and ahora - insp['t'] < self.tiempo_giro:
                w = min(self.giro_maximo, max(self.giro_minimo, 1.5 * abs(err)))
                cmd.angular.z = math.copysign(w, err)
            else:
                self.get_logger().info(
                    f'Mirando el candidato ({c["x"]:.2f}, {c["y"]:.2f}) desde '
                    f'{self.texto_pose()}: error de rumbo {math.degrees(err):.0f}°'
                    + (' (se acabó el tiempo de giro)' if abs(err) > 0.25 else ''))
                insp['fase'] = 'mirar'
                insp['t'] = ahora
        elif insp['fase'] == 'mirar':
            mirando = ahora - insp['t']
            if c['estado'] == 'REPORTADO':
                self.terminar_inspeccion('reportado')
            elif c['rojo'] >= 2 and mirando < self.tiempo_mirar + self.espera_reporte:
                c['estado'] = 'ROJO'   # esperar a que juntar_victimas reporte
            elif (mirando > self.tiempo_mirar and c['rojo'] == 0) or (
                    mirando > self.tiempo_mirar_dudoso and c['rojo'] < 2):
                c['estado'] = 'DESCARTADO'
                self.terminar_inspeccion('no es rojo')
            elif mirando > self.tiempo_mirar + self.espera_reporte:
                self.terminar_inspeccion('rojo, sin reporte todavía')
        self.pub_cmd_vel.publish(cmd)

    def terminar_inspeccion(self, resultado):
        c = self.inspeccion['c']
        c['inspecciones'] += 1
        if resultado == 'no es rojo':
            self.descartar_en_cuadro(c)
        if c['estado'] == 'ROJO':
            c['estado'] = 'PENDIENTE'   # se vuelve a mirar si no se llegó a reportar
        self.get_logger().info(
            f'Candidato ({c["x"]:.2f}, {c["y"]:.2f}): {resultado} '
            f'({c["rojo"]} detecciones rojas)')
        self.inspeccion = None

    def descartar_en_cuadro(self, mirado):
        """Descarta los otros candidatos que entraron en el mismo cuadro sin nada rojo."""
        pose = self.pose_robot()
        if pose is None:
            return
        otros = []
        for c in self.candidatos:
            if c is mirado or c['estado'] != 'PENDIENTE' or c['rojo'] > 0:
                continue
            err = math.atan2(c['y'] - pose[1], c['x'] - pose[0]) - pose[2]
            d = math.hypot(c['x'] - pose[0], c['y'] - pose[1])
            if (abs(math.atan2(math.sin(err), math.cos(err))) < 0.4
                    and self.distancia_minima_inspeccion <= d < self.distancia_inspeccion
                    and self.exploracion.linea_de_vista(pose[:2], (c['x'], c['y']))):
                c['estado'] = 'DESCARTADO'
                c['inspecciones'] += 1
                otros.append(f'({c["x"]:.2f}, {c["y"]:.2f})')
        if otros:
            self.get_logger().info(f'En el mismo cuadro, tampoco rojos: {", ".join(otros)}')

    def ir_a_candidato_pendiente(self, pose):
        """Terminada la ruta, ir a mirar candidatos que no se vieron de cerca."""
        if pose is None:
            return False
        for c in sorted(self.pendientes(),
                        key=lambda c: math.hypot(c['x'] - pose[0], c['y'] - pose[1])):
            if c['visitas'] >= 1:
                continue
            c['visitas'] += 1
            p = self.exploracion.punto_de_inspeccion((c['x'], c['y']), pose[:2])
            if p is None:
                continue
            self.inspecciones_hechas += 1
            self.get_logger().info(
                f'Yendo a mirar el candidato ({c["x"]:.2f}, {c["y"]:.2f}) desde '
                f'({p[0]:.2f}, {p[1]:.2f})')
            self.objetivo_es_inspeccion = True
            yaw = math.atan2(c['y'] - p[1], c['x'] - p[0])
            if self.ir_a(p[0], p[1], yaw):
                return True
            self.objetivo_es_inspeccion = False
            return False
        return False

    def limite_exploracion(self, pose):
        """Segundo de corrida en el que hay que empezar a volver."""
        distancia = 0.0
        if pose is not None and self.largada is not None:
            distancia = math.hypot(pose[0] - self.largada[0], pose[1] - self.largada[1])
            if self.exploracion is not None:
                distancia = self.exploracion.distancia_camino(pose, self.largada) or distancia
        return self.tiempo_corrida - self.margen_vuelta - distancia / self.velocidad_vuelta

    def volver(self):
        self.inspeccion = None
        self.cancelar()
        self.cambiar_estado(VOLVIENDO)

    def terminar(self):
        """Pide gastar los reportes que sobran y, un momento después, publica /done."""
        self.cancelar()
        self.inspeccion = None
        self.pub_cmd_vel.publish(Twist())
        if self.reportes >= self.cantidad_victimas:
            self.publicar_done()   # no hay reportes que gastar: no se espera
            return
        self.pub_vaciar.publish(Empty())
        self.t_finalizando = self.ahora()
        self.cambiar_estado(FINALIZANDO)

    def publicar_done(self):
        """Publica en /done: termina la corrida (§5) y deja el robot quieto."""
        self.pub_done.publish(Empty())
        self.pub_cmd_vel.publish(Twist())
        self.get_logger().info(
            f'/done publicado a {self.distancia_a_largada():.2f} m de la largada, '
            f'con {self.reportes} reportes')
        self.cambiar_estado(TERMINADO)


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoEquipo()
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
