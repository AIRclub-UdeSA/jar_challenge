"""Publica el camino de /odom y el mismo camino con las escalas corregidas."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

from jar_localizacion.localizador import yaw_a_quaternion, yaw_de_quaternion
from jar_localizacion.odometria import aplicar_paso, paso_en_frame_robot


class OdomCorregida(Node):
    """
    Publica el camino de /odom con las escalas aplicadas.

    Solo para mirar la corrección en RViz: no republica /odom ni toca la
    transformada odom -> base_footprint. No correr junto al localizador,
    que publica en los mismos tópicos.
    """

    def __init__(self):
        super().__init__('odom_corregida')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('escala_avance', 1.0)
        self.declare_parameter('escala_giro', 1.0)
        self.map_frame = self.get_parameter('map_frame').value
        self.escala_avance = self.get_parameter('escala_avance').value
        self.escala_giro = self.get_parameter('escala_giro').value

        self.anterior = None
        self.corregida = (0.0, 0.0, 0.0)
        self.camino_odom = Path()
        self.camino_corregido = Path()
        for camino in (self.camino_odom, self.camino_corregido):
            camino.header.frame_id = self.map_frame

        self.create_subscription(Odometry, 'odom', self.recibir_odom, 10)
        self.pub_odom = self.create_publisher(Path, 'camino_odom', 10)
        self.pub_corregido = self.create_publisher(Path, 'camino_corregido', 10)

    def recibir_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        actual = (p.x, p.y, yaw_de_quaternion(msg.pose.pose.orientation))
        if self.anterior is not None:
            self.corregida = aplicar_paso(
                self.corregida, *paso_en_frame_robot(self.anterior, actual),
                self.escala_avance, self.escala_giro)
        self.anterior = actual
        self.acumular(self.camino_odom, self.pub_odom, actual, msg.header.stamp)
        self.acumular(self.camino_corregido, self.pub_corregido, self.corregida,
                      msg.header.stamp)

    def acumular(self, camino, publisher, pose, stamp):
        x, y, theta = pose
        punto = PoseStamped()
        punto.header.frame_id = self.map_frame
        punto.header.stamp = stamp
        punto.pose.position.x = float(x)
        punto.pose.position.y = float(y)
        punto.pose.orientation = yaw_a_quaternion(float(theta))
        camino.poses.append(punto)
        camino.header.stamp = stamp
        publisher.publish(camino)


def main(args=None):
    rclpy.init(args=args)
    nodo = OdomCorregida()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
