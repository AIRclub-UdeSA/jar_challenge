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

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import Empty
from std_srvs.srv import Empty as ServicioVacio
from tf2_ros import Buffer, TransformListener

from equipo_jar.exploracion import Exploracion

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

ESPERANDO_MAPA = 'ESPERANDO_MAPA'
LOCALIZANDO = 'LOCALIZANDO'
EXPLORANDO = 'EXPLORANDO'
VOLVIENDO = 'VOLVIENDO'
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
        # Para detectar que el robot quedó quieto con un objetivo activo.
        self.ultimo_avance = None
        self.pose_avance = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        # Solo para contar los reportes que publica juntar_victimas.
        self.create_subscription(PointStamped, '/victimas', self.al_reporte, 10)
        self.create_subscription(Odometry, '/odom', self.al_odom, qos_profile_sensor_data)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
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

    def al_reporte(self, _msg):
        self.reportes += 1
        self.get_logger().info(f'Reportes publicados: {self.reportes}/{self.cantidad_victimas}')

    def al_amcl_pose(self, msg):
        c = msg.pose.covariance
        self.cov_amcl = (c[0], c[7], c[35])
        # La primera pose con la dispersión que se le mandó confirma que amcl
        # tomó la pose inicial (antes tenía covarianza cero).
        if self.t_pose_inicial is not None and c[0] > 0.25 * self.dispersion_xy ** 2:
            self.pose_inicial_aceptada = True

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

    def ir_a(self, x, y, yaw):
        """Manda un objetivo a Nav2. Devuelve False si Nav2 todavía no está listo."""
        if not self.nav.server_is_ready():
            return False
        meta = NavigateToPose.Goal()
        meta.pose = PoseStamped()
        meta.pose.header.frame_id = 'map'
        meta.pose.header.stamp = self.get_clock().now().to_msg()
        meta.pose.pose.position.x = float(x)
        meta.pose.pose.position.y = float(y)
        meta.pose.pose.orientation.z = math.sin(yaw / 2.0)
        meta.pose.pose.orientation.w = math.cos(yaw / 2.0)
        self.objetivo = (x, y)
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
        if self.estado == EXPLORANDO:
            if not exito:
                self.get_logger().info(f'No se llegó a {self.objetivo}: se sigue con el próximo')
            self.exploracion.objetivo_terminado(exito)
        elif self.estado == VOLVIENDO:
            if exito or self.distancia_a_largada() <= self.tolerancia_largada:
                self.terminar()
            else:
                self.vueltas_fallidas += 1
                if self.vueltas_fallidas >= self.intentos_vuelta:
                    self.get_logger().warn('No se pudo volver a la largada; se termina igual')
                    self.terminar()

    def distancia_a_largada(self):
        pose = self.pose_robot()
        if pose is None or self.largada is None:
            return math.inf
        return math.hypot(pose[0] - self.largada[0], pose[1] - self.largada[1])

    # ---------- máquina de estados ----------

    def ciclo_de_control(self):
        if self.estado in (ESPERANDO_MAPA, TERMINADO):
            return  # sin mapa no se mueve ni reporta (§5); al terminar, quieto
        pose = self.pose_robot()
        if self.estado == LOCALIZANDO:
            if not self.localizado:
                self.localizar()
                return
            if pose is None or not self.nav.server_is_ready():
                self.get_logger().info(
                    'Esperando ' + ('la tf map -> base_footprint' if pose is None else 'a Nav2'),
                    throttle_duration_sec=5.0)
                return
            self.largada = pose
            self.exploracion = Exploracion(self.mapa, self.get_logger())
            self.exploracion.planificar(pose)
            self.cambiar_estado(EXPLORANDO)

        if self.navegando and self.ahora() - self.ultimo_avance > self.tiempo_sin_avance:
            self.get_logger().warn(
                f'{self.tiempo_sin_avance:.0f} s sin moverse: se abandona el objetivo')
            self.cancelar()
            self.al_terminar_objetivo(False)

        if self.estado == EXPLORANDO:
            if self.reportes >= self.cantidad_victimas:
                self.get_logger().info('Se reportaron todas las víctimas')
                self.volver()
            elif self.segundos_de_corrida() > self.limite_exploracion(pose):
                self.get_logger().info('Se acaba el tiempo')
                self.volver()
            elif not self.navegando:
                siguiente = self.exploracion.siguiente(pose)
                if siguiente is None:
                    self.get_logger().info('No quedan objetivos de exploración')
                    self.volver()
                else:
                    self.ir_a(*siguiente)
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

    def limite_exploracion(self, pose):
        """Segundo de corrida en el que hay que empezar a volver."""
        distancia = 0.0
        if pose is not None and self.largada is not None:
            distancia = math.hypot(pose[0] - self.largada[0], pose[1] - self.largada[1])
            if self.exploracion is not None:
                distancia = self.exploracion.distancia_camino(pose, self.largada) or distancia
        return self.tiempo_corrida - self.margen_vuelta - distancia / self.velocidad_vuelta

    def volver(self):
        self.cancelar()
        self.cambiar_estado(VOLVIENDO)

    def terminar(self):
        """Publica en /done: termina la corrida (§5) y deja el robot quieto."""
        self.cancelar()
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
