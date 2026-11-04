"""Convierte las detecciones de detector_obstaculos en víctimas confirmadas.

Entrada: /objetos_no_mapeados (PoseArray en `map`), los objetos rojos que no están en el
mapa vistos en cada scan. Salida: /victima_confirmada (PointStamped en `map`), una única vez
por caja. No reporta al juez: eso lo hace nodo_equipo, que lleva la cuenta de los N reportes.

La lógica (asociar, promediar, confirmar, corregir cara -> centro) vive en seguimiento.py.
"""
import math

import rclpy
from geometry_msgs.msg import PointStamped, PoseArray
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from equipo_jar.seguimiento import Seguidor, cara_a_centro


class GestorVictimas(Node):

    def __init__(self):
        super().__init__('gestor_victimas')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        # Dos detecciones a menos de esto son la misma caja.
        self.declare_parameter('radio_asociacion', 0.35)
        # Cuántas veces hay que ver un candidato para reportarlo.
        self.declare_parameter('observaciones_minimas', 8)
        # Segundos sin ver un candidato sin confirmar antes de descartarlo.
        self.declare_parameter('tiempo_olvido', 5.0)
        # Más lejos que esto, la detección es poco precisa y se ignora.
        self.declare_parameter('distancia_maxima_observacion', 2.0)
        # Mitad del lado de la caja: cuánto hay que correr la cara para llegar al centro.
        self.declare_parameter('medio_lado_caja', 0.125)
        # Con la localización por debajo de esto (ver localizador.calidad) la pose del robot
        # es dudosa: una caja ya vista caería en otro lugar y se reportaría dos veces.
        self.declare_parameter('calidad_minima', 0.6)
        # Si la última calidad tiene más de esto, se la considera vencida.
        self.declare_parameter('calidad_vencimiento', 1.0)   # s

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.distancia_maxima = self.get_parameter('distancia_maxima_observacion').value
        self.medio_lado = self.get_parameter('medio_lado_caja').value
        self.calidad_minima = self.get_parameter('calidad_minima').value
        self.calidad_vencimiento = self.get_parameter('calidad_vencimiento').value
        self.calidad = None            # (valor, instante) de la última calidad recibida
        self.seguidor = Seguidor(
            self.get_parameter('radio_asociacion').value,
            self.get_parameter('observaciones_minimas').value,
            self.get_parameter('tiempo_olvido').value,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(PoseArray, 'objetos_no_mapeados', self.recibir_objetos, 10)
        self.create_subscription(Float32, 'calidad_localizacion', self.recibir_calidad, 10)
        self.pub_confirmada = self.create_publisher(PointStamped, 'victima_confirmada', 10)
        self.pub_marcadores = self.create_publisher(MarkerArray, 'candidatos_markers', 10)
        self.create_timer(1.0, self.mantenimiento)

    def ahora(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_robot(self):
        """(x, y) del robot en `map`, o None si todavía no hay tf (sin mapa, sin localizador)."""
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def recibir_calidad(self, msg: Float32):
        self.calidad = (msg.data, self.ahora())

    def localizacion_confiable(self) -> bool:
        if self.calidad is None:
            return False
        valor, instante = self.calidad
        if self.ahora() - instante > self.calidad_vencimiento:
            return False
        if valor < self.calidad_minima:
            self.get_logger().warn(
                f'Localización dudosa (calidad {valor:.2f} < {self.calidad_minima}): '
                'ignoro las detecciones.', throttle_duration_sec=2.0)
            return False
        return True

    def recibir_objetos(self, msg: PoseArray):
        if not msg.poses or not self.localizacion_confiable():
            return
        robot = self.pose_robot()
        if robot is None:
            return
        t = self.ahora()
        for pose in msg.poses:
            x, y = pose.position.x, pose.position.y
            if math.hypot(x - robot[0], y - robot[1]) > self.distancia_maxima:
                continue
            x, y = cara_a_centro(x, y, robot[0], robot[1], self.medio_lado)
            confirmado = self.seguidor.observar(x, y, t)
            if confirmado is not None:
                self.publicar_confirmada(confirmado)

    def publicar_confirmada(self, track):
        msg = PointStamped()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x = track.x
        msg.point.y = track.y
        self.pub_confirmada.publish(msg)
        self.get_logger().info(
            f'Víctima confirmada en ({track.x:.2f}, {track.y:.2f}) '
            f'tras {track.observaciones} observaciones')

    def mantenimiento(self):
        self.seguidor.olvidar(self.ahora())
        self.publicar_marcadores()

    def publicar_marcadores(self):
        """Candidatos en RViz: amarillo = acumulando, verde = confirmado."""
        marcadores = MarkerArray()
        borrar = Marker()
        borrar.action = Marker.DELETEALL
        marcadores.markers.append(borrar)
        stamp = self.get_clock().now().to_msg()
        minimas = self.seguidor.observaciones_minimas
        for i, track in enumerate(self.seguidor.tracks):
            cilindro = Marker()
            cilindro.header.frame_id = self.map_frame
            cilindro.header.stamp = stamp
            cilindro.ns = 'candidatos'
            cilindro.id = 2 * i
            cilindro.type = Marker.CYLINDER
            cilindro.pose.position.x = track.x
            cilindro.pose.position.y = track.y
            cilindro.pose.orientation.w = 1.0
            # Diámetro 0,5 m: el círculo de 50 puntos alrededor del centro estimado.
            cilindro.scale.x = cilindro.scale.y = 0.5
            cilindro.scale.z = 0.02
            if track.confirmado:
                cilindro.color.g, cilindro.color.a = 1.0, 0.5
            else:
                cilindro.color.r = cilindro.color.g = 1.0
                cilindro.color.a = 0.15 + 0.35 * min(1.0, track.observaciones / minimas)
            marcadores.markers.append(cilindro)

            texto = Marker()
            texto.header = cilindro.header
            texto.ns = 'candidatos'
            texto.id = 2 * i + 1
            texto.type = Marker.TEXT_VIEW_FACING
            texto.pose.position.x = track.x
            texto.pose.position.y = track.y
            texto.pose.position.z = 0.3
            texto.pose.orientation.w = 1.0
            texto.scale.z = 0.15
            texto.color.r = texto.color.g = texto.color.b = texto.color.a = 1.0
            texto.text = f'{track.observaciones}/{minimas}'
            marcadores.markers.append(texto)
        self.pub_marcadores.publish(marcadores)


def main(args=None):
    rclpy.init(args=args)
    nodo = GestorVictimas()
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
