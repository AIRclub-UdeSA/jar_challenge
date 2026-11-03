"""Nodo esqueleto del equipo.

No hace nada útil todavía: sirve para que el template compile, arranque, espere
el mapa y pase el CI. Reemplazalo (o agregá nodos) con tu estrategia.

Lo que ya resuelve:
- Se suscribe a /map con el QoS que usa el juez (reliable + transient local).
- No publica nada en /cmd_vel ni en /victimas hasta recibir el mapa (§5).
- Deja armado el canal de /victimas, /done y /cmd_vel.

Lo que tenés que hacer vos, en este orden:
1. Localizarte en el mapa. Arrancás cerca del origen (0, 0), pero no exactamente ahí (§1).
2. Explorar y esquivar obstáculos.
3. Detectar cajas rojas y reportarlas con `armar_reporte`.
4. Opcional: volver a la largada y publicar en /done.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

from equipo_jar.regreso import distancia_a_paredes, linea_libre, velocidad_hacia
from equipo_jar.reportes import armar_reporte

ESPERANDO_MAPA = 'esperando_mapa'
EXPLORANDO = 'explorando'     # el evasor maneja; se reportan las víctimas confirmadas
VOLVIENDO = 'volviendo'       # sigue el evasor, pero si la largada queda cerca y a la vista, va
APROXIMANDO = 'aproximando'   # derecho hacia la largada
TERMINADO = 'terminado'       # /done publicado: el robot no se vuelve a mover


def yaw_de_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class NodoEquipo(Node):
    def __init__(self):
        super().__init__('nodo_equipo')
        self.declare_parameter('frecuencia_control_hz', 10.0)
        self.declare_parameter('velocidad_lineal_max', 0.30)
        self.declare_parameter('velocidad_angular_max', 1.00)
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value
        self.velocidad_lineal_max = self.get_parameter('velocidad_lineal_max').value
        self.velocidad_angular_max = self.get_parameter('velocidad_angular_max').value
        # Fin de corrida y regreso a la largada. Se leen en cada ciclo, así que se pueden
        # cambiar en vivo con `ros2 param set /nodo_equipo <nombre> <valor>`.
        # Pasado este tiempo desde el mapa (o con los N reportes hechos) se intenta volver.
        self.declare_parameter('tiempo_regreso', 420.0)          # s
        # Pasado este tiempo se frena y se publica /done donde esté (la corrida corta a 600 s).
        self.declare_parameter('tiempo_limite', 570.0)           # s
        # Volviendo: si la largada está a menos de esto y en línea recta libre, se va derecho.
        self.declare_parameter('distancia_aproximacion', 0.8)    # m
        # Distancia mínima a las paredes del segmento robot -> largada para ir derecho.
        self.declare_parameter('margen_paredes', 0.2)            # m
        self.declare_parameter('velocidad_regreso', 0.15)        # m/s
        self.declare_parameter('ganancia_regreso', 1.0)          # 1/s
        # A menos de esto de la largada se frena y se publica /done.
        self.declare_parameter('tolerancia_llegada', 0.05)       # m
        # La largada en `map` se fija cuando el robot ya se movió esto (localización convergida).
        self.declare_parameter('distancia_fijar_largada', 0.5)   # m
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')

        self.mapa = None
        self.distancias = None    # distancia a la pared más cercana, por celda del mapa
        self.reportes_hechos = 0  # cada mensaje en /victimas gasta uno de los N, puntúe o no
        self.estado = ESPERANDO_MAPA
        self.t_mapa = None
        # La largada: la pose del robot en `odom` al llegar el mapa (la cinta está donde lo
        # pusieron, no necesariamente en (0, 0) del mapa). Se pasa a `map` una sola vez, cuando
        # la localización ya convergió; después `odom` deriva y ya no sirve.
        self.largada_odom = None
        self.largada_map = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.sub_cmd_vel_evasor = self.create_subscription(
            Twist, '/cmd_vel_evasor', self.al_recibir_cmd_vel_evasor, 10)
        self.sub_confirmadas = self.create_subscription(
            PointStamped, 'victima_confirmada', self.al_confirmar_victima, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        # Mientras vuelve, la largada (en `map`) para que el evasor oriente su exploración hacia ella.
        self.pub_objetivo_regreso = self.create_publisher(PointStamped, 'objetivo_regreso', 10)

        periodo = 1.0 / self.get_parameter('frecuencia_control_hz').value
        self.create_timer(periodo, self.ciclo_de_control)
        self.log(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    def log(self, msg):
        self.get_logger().info(msg)

    def al_recibir_mapa(self, mapa):
        if self.mapa is None:
            self.log(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m')
        self.mapa = mapa
        if self.distancias is None:
            grilla = np.array(mapa.data, dtype=np.int16).reshape((mapa.info.height, mapa.info.width))
            self.distancias = distancia_a_paredes(grilla, mapa.info.resolution)

    def al_recibir_cmd_vel_evasor(self, msg):
        """Recibe la velocidad de evasión y la re-publica en /cmd_vel.

        Esto es solo para que el template compile y pase el CI. Reemplazalo con tu
        estrategia de control.
        """
        # Solo el evasor maneja mientras se explora (o se vuelve sin ver la largada).
        if self.estado in (EXPLORANDO, VOLVIENDO):
            self.pub_cmd_vel.publish(self.recortar(msg))

    def recortar(self, msg):
        """Todo lo que va a /cmd_vel pasa por acá: ningún nodo puede superar los límites
        (la organización limita a 0,5 m/s). El avance se recorta como vector (x, y), para no
        cambiar la dirección de un movimiento en diagonal."""
        salida = Twist()
        rapidez = math.hypot(msg.linear.x, msg.linear.y)
        escala = 1.0 if rapidez <= self.velocidad_lineal_max else self.velocidad_lineal_max / rapidez
        salida.linear.x = msg.linear.x * escala
        salida.linear.y = msg.linear.y * escala
        salida.angular.z = max(-self.velocidad_angular_max,
                               min(self.velocidad_angular_max, msg.angular.z))
        return salida

    def al_confirmar_victima(self, msg):
        """gestor_victimas confirmó una caja: se reporta, mientras queden reportes."""
        if self.mapa is None or self.estado == TERMINADO:
            return  # no reportar antes del mapa (§5) ni después del /done
        if self.reportes_hechos >= self.cantidad_victimas:
            self.get_logger().warn(
                f'Confirmada otra víctima en ({msg.point.x:.2f}, {msg.point.y:.2f}), pero ya '
                f'se usaron los {self.cantidad_victimas} reportes: no se reporta.')
            return
        self.reportar_victima(msg.point.x, msg.point.y)
        self.reportes_hechos += 1
        self.log(f'Reporte {self.reportes_hechos}/{self.cantidad_victimas}: '
                 f'({msg.point.x:.2f}, {msg.point.y:.2f})')

    def p(self, nombre):
        return self.get_parameter(nombre).value

    def ahora(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose(self, frame):
        """(x, y, yaw) de la base del robot en `frame`, o None si no hay tf todavía."""
        try:
            t = self.tf_buffer.lookup_transform(frame, self.p('base_frame'), Time()).transform
        except Exception:
            return None
        return t.translation.x, t.translation.y, yaw_de_quaternion(t.rotation)

    def fijar_largada(self):
        if self.largada_odom is None:
            self.largada_odom = self.pose(self.p('odom_frame'))
            return
        if self.largada_map is not None:
            return
        actual = self.pose(self.p('odom_frame'))
        if actual is None or math.hypot(actual[0] - self.largada_odom[0], actual[1] -
                                        self.largada_odom[1]) < self.p('distancia_fijar_largada'):
            return
        try:
            t = self.tf_buffer.lookup_transform(
                self.p('map_frame'), self.p('odom_frame'), Time()).transform
        except Exception:
            return
        yaw = yaw_de_quaternion(t.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        x0, y0 = self.largada_odom[0], self.largada_odom[1]
        self.largada_map = (t.translation.x + c * x0 - s * y0, t.translation.y + s * x0 + c * y0)
        self.log(f'Largada en el mapa: ({self.largada_map[0]:.2f}, {self.largada_map[1]:.2f})')

    def cambiar_estado(self, nuevo, motivo):
        self.log(f'{self.estado} -> {nuevo}: {motivo}')
        self.estado = nuevo

    def ciclo_de_control(self):
        if self.estado == ESPERANDO_MAPA:
            if self.mapa is None:
                return  # esperar el mapa: no mover el robot ni reportar antes
            self.t_mapa = self.ahora()
            self.cambiar_estado(EXPLORANDO, 'llegó el mapa')
        if self.estado == TERMINADO:
            self.pub_cmd_vel.publish(Twist())   # quieto (el robot se frena solo igual)
            return

        self.fijar_largada()
        t = self.ahora() - self.t_mapa
        if t >= self.p('tiempo_limite'):
            self.terminar(f'se acabó el tiempo ({t:.0f} s)')
            return
        if self.estado == EXPLORANDO:
            if self.reportes_hechos >= self.cantidad_victimas:
                self.cambiar_estado(VOLVIENDO, f'{self.reportes_hechos} reportes hechos')
            elif t >= self.p('tiempo_regreso'):
                self.cambiar_estado(VOLVIENDO, f'pasaron {t:.0f} s')

        if self.estado in (VOLVIENDO, APROXIMANDO):
            if self.largada_map is not None:
                objetivo = PointStamped()
                objetivo.header.frame_id = self.p('map_frame')
                objetivo.header.stamp = self.get_clock().now().to_msg()
                objetivo.point.x, objetivo.point.y = self.largada_map
                self.pub_objetivo_regreso.publish(objetivo)
            self.intentar_regreso()

    def intentar_regreso(self):
        robot = self.pose(self.p('map_frame'))
        if robot is None or self.largada_map is None or self.distancias is None:
            return
        gx, gy = self.largada_map
        distancia = math.hypot(gx - robot[0], gy - robot[1])
        if distancia <= self.p('tolerancia_llegada'):
            self.terminar(f'en la largada (a {distancia:.2f} m)')
            return
        info = self.mapa.info
        a_la_vista = (distancia <= self.p('distancia_aproximacion') and linea_libre(
            self.distancias, info.resolution, info.origin.position.x, info.origin.position.y,
            robot[:2], self.largada_map, self.p('margen_paredes')))
        if a_la_vista and self.estado == VOLVIENDO:
            self.cambiar_estado(APROXIMANDO, f'largada a la vista, a {distancia:.2f} m')
        elif not a_la_vista and self.estado == APROXIMANDO:
            self.cambiar_estado(VOLVIENDO, 'se perdió de vista la largada')
        if self.estado == APROXIMANDO:
            vx, vy = velocidad_hacia(robot[0], robot[1], robot[2], gx, gy,
                                     self.p('velocidad_regreso'), self.p('ganancia_regreso'))
            msg = Twist()
            msg.linear.x, msg.linear.y = vx, vy
            self.pub_cmd_vel.publish(self.recortar(msg))

    def reportar_victima(self, x, y):
        """Publica un reporte. Cada llamada gasta un reporte."""
        self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))

    def terminar(self, motivo):
        """Frena y publica en /done: termina la corrida (§5). Después no se mueve más."""
        self.pub_cmd_vel.publish(Twist())
        self.pub_done.publish(Empty())
        self.cambiar_estado(TERMINADO, motivo)


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
