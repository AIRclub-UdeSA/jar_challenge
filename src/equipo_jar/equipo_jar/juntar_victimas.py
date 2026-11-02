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
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.duration import Duration
from rclpy.time import Time
from std_msgs.msg import Empty
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
        # Un grupo cuya mediana queda a menos de esto (m) de una víctima ya
        # reportada es la misma caja vista desde otro lado (§6.1: las víctimas
        # están separadas para que un reporte no pueda servir para dos).
        self.declare_parameter('distancia_minima_reportes', 0.7)
        # Mediana (m) de la distancia de los puntos del grupo a su mediana: un
        # grupo más disperso que esto (p. ej. puntos corridos por una pared
        # durante un giro) no se reporta todavía.
        self.declare_parameter('dispersion_maxima', 0.08)
        # Al final de la corrida (equipo/vaciar_reportes) se publican los
        # mejores grupos que no llegaron a detecciones_minimas: un reporte
        # errado da 0 y no resta, así que los lugares que sobran se usan.
        self.declare_parameter('detecciones_minimas_final', 3)
        self.victimas = self.get_parameter('cantidad_victimas').value
        self.minimas = self.get_parameter('detecciones_minimas').value
        self.radio = self.get_parameter('radio_grupo').value
        self.mitad = self.get_parameter('mitad_caja').value
        self.base_frame = self.get_parameter('base_frame').value
        self.distancia_minima_reportes = self.get_parameter('distancia_minima_reportes').value
        self.dispersion_maxima = self.get_parameter('dispersion_maxima').value
        self.minimas_final = self.get_parameter('detecciones_minimas_final').value
        self.reportadas = []        # medianas ya publicadas

        # Cada grupo es un dict: {'puntos': [(x, y), ...], 'publicado': bool}
        self.grupos = []
        self.publicados = 0

        # Hasta que nodo_equipo avisa que amcl convergió, la tf map -> robot
        # puede estar muy corrida y una detección caería en cualquier lado:
        # se ignora (cada reporte gastado es irrecuperable).
        self.localizado = False
        qos_latch = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Empty, 'equipo/localizado', self.al_localizar, qos_latch)
        self.create_subscription(Empty, 'equipo/vaciar_reportes', self.vaciar_reportes, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(PoseArray, 'objetos_rojos', self.recibir_objetos, 10)
        self.pub = self.create_publisher(PointStamped, '/victimas', 10)

    def al_localizar(self, _msg):
        self.localizado = True
        self.get_logger().info('Robot localizado: se empiezan a juntar detecciones')

    def posicion_robot(self, instante):
        """(x, y) del robot en `map` en el instante de la detección, o None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'map', self.base_frame, Time.from_msg(instante),
                timeout=Duration(seconds=0.1))
        except Exception:
            return None
        return tf.transform.translation.x, tf.transform.translation.y

    def recibir_objetos(self, msg: PoseArray):
        """Lleva cada detección de la cara al centro de la caja y la agrupa.

        El lidar ve la cara que mira al robot: el centro está media caja más
        lejos, en la misma dirección en que el robot ve la caja.
        """
        if not self.localizado:
            return
        pos_robot = self.posicion_robot(msg.header.stamp)
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
        puntos = np.array(g['puntos'])
        mediana = np.median(puntos, axis=0)
        for r in self.reportadas:
            if math.hypot(mediana[0] - r[0], mediana[1] - r[1]) < self.distancia_minima_reportes:
                g['publicado'] = True   # misma caja que una ya reportada
                self.get_logger().info(
                    f'Grupo en ({mediana[0]:.2f}, {mediana[1]:.2f}) es la víctima ya '
                    f'reportada en ({r[0]:.2f}, {r[1]:.2f}): no se reporta')
                return
        dispersion = float(np.median(np.hypot(*(puntos - mediana).T)))
        if dispersion > self.dispersion_maxima:
            return
        self.publicar(g, mediana)

    def cerca_de_reportada(self, x, y):
        return any(math.hypot(x - r[0], y - r[1]) < self.distancia_minima_reportes
                   for r in self.reportadas)

    def vaciar_reportes(self, _msg):
        """Gasta los reportes que sobran en los grupos con más detecciones."""
        restantes = self.victimas - self.publicados
        if restantes <= 0:
            return
        grupos = [g for g in self.grupos
                  if not g['publicado'] and len(g['puntos']) >= self.minimas_final]
        for g in sorted(grupos, key=lambda g: -len(g['puntos'])):
            if self.publicados >= self.victimas:
                break
            mediana = np.median(np.array(g['puntos']), axis=0)
            if self.cerca_de_reportada(mediana[0], mediana[1]):
                continue
            self.get_logger().info(
                f'Fin de corrida: se reporta un grupo de {len(g["puntos"])} detecciones')
            self.publicar(g, mediana)

    def publicar(self, g, mediana):
        try:
            reporte = armar_reporte(
                float(mediana[0]), float(mediana[1]), self.get_clock().now().to_msg())
        except ValueError as error:
            self.get_logger().warn(f'No se publica víctima: {error}')
            return
        self.pub.publish(reporte)
        g['publicado'] = True
        self.reportadas.append((float(mediana[0]), float(mediana[1])))
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
