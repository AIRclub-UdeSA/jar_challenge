"""Configura, activa e inicializa AMCL con plazos y reconsulta de estado."""
import math
import time

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav_msgs.msg import OccupancyGrid
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

QOS_RETENIDO = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL)


class InicializarLocalizacion(Node):
    """Un único propietario de las transiciones lifecycle de AMCL."""

    def __init__(self, **kwargs):
        super().__init__('inicializar_localizacion', **kwargs)
        for nombre, valor in [('initial_std_xy', 0.30), ('initial_std_yaw', 0.35),
                              ('rpc_timeout_s', 2.0), ('startup_timeout_s', 15.0)]:
            self.declare_parameter(nombre, valor)
        self.std_xy = float(self.get_parameter('initial_std_xy').value)
        self.std_yaw = float(self.get_parameter('initial_std_yaw').value)
        if not all(math.isfinite(v) and v > 0 for v in (
                self.std_xy, self.std_yaw,
                self.get_parameter('rpc_timeout_s').value,
                self.get_parameter('startup_timeout_s').value)):
            raise ValueError('Los desvíos y plazos deben ser positivos y finitos')
        self.mapa_recibido = False
        self.inicio_mapa = None
        self.enviado = False
        self.fallo = False
        self.peticion = None
        self.tipo_peticion = None
        self.cliente = self.create_client(GetState, '/amcl/get_state')
        self.transiciones = self.create_client(ChangeState, '/amcl/change_state')
        self.publicador = self.create_publisher(
            PoseWithCovarianceStamped, '/equipo/initialpose', 10)
        self.estado_pub = self.create_publisher(
            String, '/equipo/inicializacion_estado', QOS_RETENIDO)
        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_mapa, QOS_RETENIDO)
        self.timer = self.create_timer(
            0.2, self.comprobar, clock=Clock(clock_type=ClockType.STEADY_TIME))

    def al_mapa(self, _):
        """La primera publicación inicia el plazo; las siguientes no lo reinician."""
        if not self.mapa_recibido:
            self.inicio_mapa = time.monotonic()
            # Una respuesta previa a la largada no confirma el estado actual.
            if self.peticion is not None and self.tipo_peticion == 'estado':
                self.peticion.cancel()
                self.peticion = None
        self.mapa_recibido = True

    def estado(self, texto):
        self.estado_pub.publish(String(data=texto))

    def consultar(self, cliente, peticion, tipo):
        self.peticion = cliente.call_async(peticion)
        self.tipo_peticion = tipo
        self.inicio_peticion = time.monotonic()

    def comprobar(self):
        """Reconsulta tras respuestas perdidas sin repetir transiciones a ciegas."""
        if self.enviado or self.fallo:
            return
        ahora = time.monotonic()
        if (self.inicio_mapa is not None and
                ahora - self.inicio_mapa > self.get_parameter('startup_timeout_s').value):
            self.fallo = True
            if self.peticion is not None:
                self.peticion.cancel()
            self.estado('FALLO: AMCL no pudo inicializarse dentro del plazo')
            self.get_logger().error('AMCL no pudo inicializarse dentro del plazo')
            return
        if self.peticion is not None:
            if not self.peticion.done():
                if ahora - self.inicio_peticion > self.get_parameter('rpc_timeout_s').value:
                    self.peticion.cancel()
                    self.peticion = None
                    self.get_logger().warn('Respuesta lifecycle vencida; reconsultando estado')
                return
            tipo = self.tipo_peticion
            try:
                respuesta = self.peticion.result()
            except Exception as error:
                self.get_logger().warn(f'Error lifecycle: {error}')
                respuesta = None
            self.peticion = None
            if respuesta is None or tipo != 'estado':
                return
            actual = respuesta.current_state.id
            if actual in (State.PRIMARY_STATE_UNCONFIGURED, State.PRIMARY_STATE_INACTIVE):
                if not self.transiciones.service_is_ready():
                    return
                request = ChangeState.Request()
                request.transition.id = (
                    Transition.TRANSITION_CONFIGURE
                    if actual == State.PRIMARY_STATE_UNCONFIGURED
                    else Transition.TRANSITION_ACTIVATE)
                self.estado('CONFIGURANDO' if actual == State.PRIMARY_STATE_UNCONFIGURED
                            else 'ACTIVANDO')
                self.consultar(self.transiciones, request, 'transicion')
                return
            if actual == State.PRIMARY_STATE_ACTIVE:
                if not self.mapa_recibido:
                    self.estado('ACTIVO_ESPERANDO_MAPA')
                    return
                if self.publicador.get_subscription_count() == 0:
                    return
                mensaje = PoseWithCovarianceStamped()
                mensaje.header.frame_id = 'map'
                mensaje.header.stamp = self.get_clock().now().to_msg()
                mensaje.pose.pose.orientation.w = 1.0
                mensaje.pose.covariance[0] = self.std_xy ** 2
                mensaje.pose.covariance[7] = self.std_xy ** 2
                mensaje.pose.covariance[35] = self.std_yaw ** 2
                self.publicador.publish(mensaje)
                self.enviado = True
                self.estado('INICIALIZADO')
                self.get_logger().info('Pose inicial enviada a AMCL con incertidumbre explícita')
                return
        if self.cliente.service_is_ready():
            self.consultar(self.cliente, GetState.Request(), 'estado')


def main(args=None):
    rclpy.init(args=args)
    nodo = InicializarLocalizacion()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        import signal
        # El primer SIGINT ya detuvo spin; no interrumpir otra vez la limpieza.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            nodo.destroy_node()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        try:
            rclpy.try_shutdown()
        except (KeyboardInterrupt, ExternalShutdownException):
            pass
        finally:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
