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

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import (PointStamped, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped, Twist)
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ManageLifecycleNodes
from nav_msgs.msg import OccupancyGrid, Odometry
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
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
            for c in self.candidatos:
                if math.hypot(c['x'] - p.position.x, c['y'] - p.position.y) < 0.4:
                    c['rojo'] += 1

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
        return 0.0 if self.t_mapa is None else self.ahora() - self.t_mapa

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

    def ir_a(self, x, y, yaw):
        """Manda un objetivo a Nav2. Devuelve False si Nav2 todavía no está listo."""
        if not self.nav.server_is_ready():
            return False
        # Si había otro objetivo activo, Nav2 lo reemplaza; su resultado se ignora.
        self.goal_handle = None
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
        self.ultimo_avance = self.ahora()
        self.nav.send_goal_async(meta).add_done_callback(self._objetivo_aceptado)
        return True

    def _objetivo_aceptado(self, futuro):
        gh = futuro.result()
        if not gh.accepted:
            self.get_logger().warn('Nav2 rechazó el objetivo')
            self.navegando = False
            self.al_terminar_objetivo(False)
            return
        self.goal_handle = gh
        gh.get_result_async().add_done_callback(lambda r: self._objetivo_terminado(gh, r))

    def _objetivo_terminado(self, gh, futuro):
        if gh is not self.goal_handle:
            return  # resultado de un objetivo que ya se reemplazó o canceló
        self.goal_handle = None
        self.navegando = False
        self.al_terminar_objetivo(futuro.result().status == GoalStatus.STATUS_SUCCEEDED)

    def cancelar(self):
        if self.goal_handle is not None:
            self.goal_handle.cancel_goal_async()
        self.goal_handle = None
        self.navegando = False

    def al_terminar_objetivo(self, exito):
        if not exito:
            self.empezar_destrabe()
        if self.estado == EXPLORANDO and self.objetivo_es_inspeccion:
            self.objetivo_es_inspeccion = False
        elif self.estado == EXPLORANDO:
            if not exito:
                self.get_logger().info(
                    f'No se llegó a ({self.objetivo[0]:.2f}, {self.objetivo[1]:.2f}) '
                    f'desde {self.texto_pose()}: se sigue con el próximo')
            self.exploracion.objetivo_terminado(exito)
        elif self.estado == VOLVIENDO:
            if exito or self.distancia_a_largada() <= self.tolerancia_largada:
                self.terminar()
            else:
                self.vueltas_fallidas += 1
                if self.vueltas_fallidas >= self.intentos_vuelta:
                    self.get_logger().warn('No se pudo volver a la largada; se termina igual')
                    self.terminar()

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
        for i, r in enumerate(scan.ranges):
            if not (scan.range_min < r < scan.range_max):
                continue
            a = scan.angle_min + i * scan.angle_increment + yaw_laser
            px, py = tx + r * math.cos(a), ty + r * math.sin(a)
            d = math.hypot(px, py)
            if 0.05 < d < self.radio_destrabe:
                peso = (self.radio_destrabe - d) / d
                ex -= px * peso
                ey -= py * peso
        norma = math.hypot(ex, ey)
        if norma < 1e-6:
            return
        v = self.velocidad_destrabe
        self.destrabe = (v * ex / norma, v * ey / norma, self.ahora() + self.tiempo_destrabe)
        self.get_logger().info(
            f'Destrabe en {self.texto_pose()}: alejándose de la pared hacia '
            f'({ex / norma:.2f}, {ey / norma:.2f})')

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
        if self.estado == LOCALIZANDO:
            if not self.nav.server_is_ready():
                # También mientras se espera a amcl: si el bringup abortó,
                # amcl tampoco está activo y la localización no avanza.
                self.quizas_reiniciar_nav2()
            if not self.localizado:
                self.localizar()
                return
            if pose is None or not self.nav.server_is_ready():
                self.get_logger().info(
                    'Esperando ' + ('la tf map -> base_footprint' if pose is None else 'a Nav2'),
                    throttle_duration_sec=5.0)
                return
            self.largada = pose
            parametros = {nombre: self.get_parameter(nombre).value
                          for nombre in ('alcance_cobertura', 'despeje_punto_vista',
                                         'despeje_transito', 'despeje_caja',
                                         'fraccion_cobertura', 'reintentos_punto_vista')}
            self.exploracion = Exploracion(self.mapa, self.get_logger(), parametros)
            self.exploracion.planificar(pose)
            self.cambiar_estado(EXPLORANDO)

        if self.navegando and self.ahora() - self.ultimo_avance > self.tiempo_sin_avance:
            self.get_logger().warn(
                f'{self.tiempo_sin_avance:.0f} s sin moverse en {self.texto_pose()}: '
                'se abandona el objetivo')
            self.cancelar()
            self.al_terminar_objetivo(False)

        if self.estado == EXPLORANDO:
            if self.reportes >= self.cantidad_victimas:
                self.get_logger().info('Se reportaron todas las víctimas')
                self.volver()
            elif self.segundos_de_corrida() > self.limite_exploracion(pose):
                self.get_logger().info('Se acaba el tiempo')
                self.volver()
            elif self.inspeccion is not None:
                self.paso_inspeccion(pose)
            elif not self.navegando and self.empezar_inspeccion(pose):
                pass
            elif not self.navegando:
                siguiente = self.exploracion.siguiente(pose)
                if siguiente is None and self.ir_a_candidato_pendiente(pose):
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
            else:
                self.revisar_objetivo_activo()
        elif self.estado == VOLVIENDO and not self.navegando:
            if self.distancia_a_largada() <= self.tolerancia_largada:
                self.terminar()
            else:
                self.ir_a(self.largada[0], self.largada[1], self.largada[2])

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
        """Si Nav2 no arrancó (bringup abortado), pide RESET y STARTUP."""
        ahora = self.ahora()
        if self.t_espera_nav2 is None:
            self.t_espera_nav2 = ahora
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
            self.ir_a(*ajustado)

    # ---------- candidatos ----------

    def pendientes(self):
        return [c for c in self.candidatos
                if c['estado'] == 'PENDIENTE' and c['n'] >= self.detecciones_candidato
                and c['inspecciones'] < 2]

    def empezar_inspeccion(self, pose):
        """Si hay un candidato pendiente a la vista, empieza a girar hacia él."""
        if pose is None or self.exploracion.despeje(pose[0], pose[1]) < 0.22:
            return False   # sin lugar para girar en el lugar
        visibles = [c for c in self.pendientes()
                    if math.hypot(c['x'] - pose[0], c['y'] - pose[1]) < self.distancia_inspeccion
                    and self.exploracion.linea_de_vista(pose[:2], (c['x'], c['y']))]
        if not visibles:
            return False
        c = min(visibles, key=lambda c: math.hypot(c['x'] - pose[0], c['y'] - pose[1]))
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
            if abs(err) > 0.15 and ahora - insp['t'] < 6.0:
                cmd.angular.z = max(-0.8, min(0.8, 1.5 * err))
            else:
                insp['fase'] = 'mirar'
                insp['t'] = ahora
        elif insp['fase'] == 'mirar':
            mirando = ahora - insp['t']
            if c['estado'] == 'REPORTADO':
                self.terminar_inspeccion('reportado')
            elif c['rojo'] >= 2 and mirando < self.tiempo_mirar + self.espera_reporte:
                c['estado'] = 'ROJO'   # esperar a que juntar_victimas reporte
            elif mirando > self.tiempo_mirar and c['rojo'] < 2:
                c['estado'] = 'DESCARTADO'
                self.terminar_inspeccion('no es rojo')
            elif mirando > self.tiempo_mirar + self.espera_reporte:
                self.terminar_inspeccion('rojo, sin reporte todavía')
        self.pub_cmd_vel.publish(cmd)

    def terminar_inspeccion(self, resultado):
        c = self.inspeccion['c']
        c['inspecciones'] += 1
        if c['estado'] == 'ROJO':
            c['estado'] = 'PENDIENTE'   # se vuelve a mirar si no se llegó a reportar
        self.get_logger().info(
            f'Candidato ({c["x"]:.2f}, {c["y"]:.2f}): {resultado} '
            f'({c["rojo"]} detecciones rojas)')
        self.inspeccion = None

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
            self.get_logger().info(
                f'Yendo a mirar el candidato ({c["x"]:.2f}, {c["y"]:.2f}) desde '
                f'({p[0]:.2f}, {p[1]:.2f})')
            self.objetivo_es_inspeccion = True
            yaw = math.atan2(c['y'] - p[1], c['x'] - p[0])
            return self.ir_a(p[0], p[1], yaw)
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
        if self.reportes < self.cantidad_victimas:
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
