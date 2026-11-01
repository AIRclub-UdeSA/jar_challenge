"""Junta las detecciones de cajas rojas en una posición por caja (semana 10).

Escucha /objetos_rojos, la salida de detector_obstaculos corriendo sobre
/scan_rojo: un PoseArray en `map` con un punto por cada racha roja del lidar.
Cada punto es el promedio de **una cara** de la caja, no su centro. Con eso:

1. corre cada detección media caja hacia adentro, para acercarla al centro;
2. agrupa las detecciones que están cerca (una caja = un grupo);
3. cuando un grupo junta `detecciones_minimas`, publica la mediana del grupo
   en /victimas, una sola vez por grupo y como mucho `cantidad_victimas`
   veces en total: cada mensaje en /victimas gasta un reporte (§7.2).
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseArray
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from equipo_jar.reportes import armar_reporte


class JuntarVictimas(Node):
    """Agrupa /objetos_rojos y publica un reporte por caja en /victimas."""

    def __init__(self):
        super().__init__('juntar_victimas')
        self.declare_parameter('cantidad_victimas', 5)
        self.declare_parameter('detecciones_minimas', 30)
        self.declare_parameter('radio_grupo', 0.30)
        self.declare_parameter('mitad_caja', 0.09)
        self.declare_parameter('base_frame', 'base_footprint')
        self.victimas = self.get_parameter('cantidad_victimas').value
        self.minimas = self.get_parameter('detecciones_minimas').value
        self.radio = self.get_parameter('radio_grupo').value
        self.mitad = self.get_parameter('mitad_caja').value
        self.base_frame = self.get_parameter('base_frame').value

        # Cada grupo es un dict: {'puntos': [(x, y), ...], 'publicado': bool}
        self.grupos = []
        self.publicados = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseArray, 'objetos_rojos', self.recibir_objetos, 10)
        self.pub = self.create_publisher(PointStamped, '/victimas', 10)

    def posicion_robot(self):
        """Devuelve (x, y) del robot en `map`, o None si todavía no hay tf."""
        try:
            tf = self.tf_buffer.lookup_transform('map', self.base_frame, Time())
        except Exception:
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def recibir_objetos(self, msg: PoseArray):
        """Lleva cada detección de la cara al centro de la caja y la agrupa.

        El lidar ve la cara que mira al robot: el centro está media caja más
        lejos, en la misma dirección en que el robot ve la caja.
        """
        pos_robot = self.posicion_robot()
        if pos_robot is None:
            return
        robot_x, robot_y = pos_robot
        for pose in msg.poses:
            x, y = pose.position.x, pose.position.y
            dx = x - robot_x
            dy = y - robot_y
            d = math.hypot(dx, dy)
            if d < 1e-6:
                continue
            x += self.mitad * dx / d
            y += self.mitad * dy / d
            self.agregar(x, y)

    def agregar(self, x, y):
        """Suma (x, y) al grupo cuya mediana está a menos de `radio_grupo`.

        Si no entra en ninguno, arma un grupo nuevo. Mediana y no promedio:
        unos pocos puntos que caen lejos de la caja no la mueven.
        """
        for g in self.grupos:
            mediana = np.median(np.array(g['puntos']), axis=0)
            distancia = math.hypot(x - mediana[0], y - mediana[1])
            if distancia < self.radio:
                g['puntos'].append((x, y))
                self.quizas_publicar(g)
                return
        self.grupos.append({'puntos': [(x, y)], 'publicado': False})
        self.quizas_publicar(self.grupos[-1])

    def quizas_publicar(self, g):
        """Publica la mediana del grupo en /victimas si ya está listo."""
        if (g['publicado'] or self.publicados >= self.victimas
                or len(g['puntos']) < self.minimas):
            return
        mediana = np.median(np.array(g['puntos']), axis=0)
        try:
            reporte = armar_reporte(
                float(mediana[0]), float(mediana[1]), self.get_clock().now().to_msg())
        except ValueError as error:
            self.get_logger().warn(f'No se publica víctima: {error}')
            return
        self.pub.publish(reporte)
        g['publicado'] = True
        self.publicados += 1
        self.get_logger().info(
            f'Publicada víctima {self.publicados}/{self.victimas} en '
            f'({reporte.point.x:.2f}, {reporte.point.y:.2f})')


def main(args=None):
    rclpy.init(args=args)
    nodo = JuntarVictimas()
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
