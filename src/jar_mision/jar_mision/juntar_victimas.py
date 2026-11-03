"""Reporta en /victimas una posición por cada caja roja detectada."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PointStamped, PoseArray
from std_msgs.msg import Empty, String

from tf2_ros import Buffer, TransformListener

from jar_mision.victimas import AgrupadorVictimas, centro_desde_cara


class JuntarVictimas(Node):
    """
    Reporta una posición por caja roja en /victimas.

    Escucha /objetos_rojos (detector_obstaculos sobre /scan_rojo): un punto
    en map por cada racha roja, ubicado sobre la cara visible de la caja.
    Lleva cada punto al centro, agrupa por caja y publica la mediana de
    cada grupo una sola vez, con un máximo de `victimas` reportes.

    Con modo_reporte 'al_final' no reporta mientras explora: elige los mejores
    grupos cuando /estado pasa a regresando_al_origen o terminado, cuando se
    cumple tiempo_limite_s o cuando llega un mensaje en reportar_victimas.
    Así un falso positivo temprano no gasta un reporte.

    Con 'inmediato' reporta cada grupo apenas junta `detecciones_minimas`
    (para el bono de rescate veloz) y, al cerrar, completa los reportes que
    sobren con los mejores grupos restantes. Cuando ya reportó las
    `victimas`, publica en victimas_completas para que el robot vuelva.
    """

    ESTADOS_FINALES = ('regresando_al_origen', 'terminado')

    def __init__(self):
        super().__init__('juntar_victimas')
        self.declare_parameter('victimas', 2)
        self.declare_parameter('detecciones_minimas', 30)
        self.declare_parameter('radio_grupo', 0.30)
        self.declare_parameter('mitad_caja', 0.09)
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('modo_reporte', 'al_final')
        # Segundos desde la primera detección; 0 lo desactiva.
        self.declare_parameter('tiempo_limite_s', 0.0)
        self.declare_parameter('detecciones_minimas_final', 10)
        self.declare_parameter('separacion_reportes', 0.5)
        modo = self.get_parameter('modo_reporte').value
        if modo not in ('al_final', 'inmediato'):
            raise ValueError("modo_reporte debe ser 'al_final' o 'inmediato'")
        self.agrupador = AgrupadorVictimas(
            self.get_parameter('victimas').value,
            self.get_parameter('detecciones_minimas').value,
            self.get_parameter('radio_grupo').value,
            inmediato=modo == 'inmediato')
        self.tiempo_limite_s = self.get_parameter('tiempo_limite_s').value
        self.minimas_final = self.get_parameter('detecciones_minimas_final').value
        self.separacion = self.get_parameter('separacion_reportes').value
        self.inicio = None
        self.cerrado = False
        self.mitad = self.get_parameter('mitad_caja').value
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseArray, 'objetos_rojos', self.recibir_objetos, 10)
        self.create_subscription(String, 'estado', self.recibir_estado, 10)
        self.create_subscription(Empty, 'reportar_victimas', self.recibir_pedido, 10)
        self.pub = self.create_publisher(PointStamped, 'victimas', 10)
        self.pub_completas = self.create_publisher(Empty, 'victimas_completas', 10)
        self.create_timer(1.0, self.revisar_tiempo)

    def posicion_robot(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def recibir_objetos(self, msg: PoseArray):
        if self.inicio is None and msg.poses:
            self.inicio = self.get_clock().now()
        robot = self.posicion_robot()
        if robot is None:
            return
        for pose in msg.poses:
            centro = centro_desde_cara(robot, (pose.position.x, pose.position.y), self.mitad)
            if centro is None:
                continue
            reporte = self.agrupador.agregar(*centro)
            if reporte is not None:
                self.publicar(*reporte, self.agrupador.publicados)

    def recibir_estado(self, msg: String):
        if msg.data in self.ESTADOS_FINALES:
            self.cerrar(f'estado {msg.data}')

    def recibir_pedido(self, _msg: Empty):
        self.cerrar('pedido en reportar_victimas')

    def revisar_tiempo(self):
        if self.tiempo_limite_s <= 0 or self.inicio is None:
            return
        transcurrido = (self.get_clock().now() - self.inicio).nanoseconds / 1e9
        if transcurrido >= self.tiempo_limite_s:
            self.cerrar(f'{transcurrido:.0f} s desde la primera detección')

    def cerrar(self, motivo):
        """Reporta una sola vez los mejores grupos acumulados."""
        if self.cerrado:
            return
        self.cerrado = True
        grupos = sorted(len(g['puntos']) for g in self.agrupador.grupos)[::-1]
        self.get_logger().info(
            f'Reporte final ({motivo}): {len(grupos)} grupos, detecciones {grupos[:5]}.')
        previos = self.agrupador.publicados
        for i, (x, y) in enumerate(self.agrupador.cerrar(self.minimas_final, self.separacion)):
            self.publicar(x, y, previos + i + 1)

    def publicar(self, x, y, numero):
        if not (math.isfinite(x) and math.isfinite(y)):
            # Un reporte con NaN o infinito se gasta y no suma (§7.2).
            self.get_logger().error(f'Reporte descartado por no ser finito: ({x}, {y}).')
            return
        punto = PointStamped()
        punto.header.frame_id = self.map_frame
        punto.header.stamp = self.get_clock().now().to_msg()
        punto.point.x = x
        punto.point.y = y
        self.pub.publish(punto)
        self.get_logger().info(
            f'Víctima {numero}/{self.agrupador.victimas} '
            f'reportada en ({x:.2f}, {y:.2f}).')
        if self.agrupador.publicados >= self.agrupador.victimas:
            self.pub_completas.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    nodo = JuntarVictimas()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
