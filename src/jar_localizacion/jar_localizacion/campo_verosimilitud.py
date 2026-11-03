"""Publica el campo de verosimilitud del mapa en /likelihood_map."""

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid

from jar_localizacion.filtro import campo_de_probabilidad

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class CampoVerosimilitud(Node):
    """Convierte cada /map recibido en una grilla de probabilidad 0-100."""

    def __init__(self):
        super().__init__('campo_verosimilitud')
        # Valores chicos exigen que el rayo termine muy cerca de la pared.
        self.declare_parameter('sigma_sensor', 0.2)
        self.sigma_sensor = self.get_parameter('sigma_sensor').value

        # Transient local: los nodos que arrancan después igual lo reciben.
        self.publisher_ = self.create_publisher(OccupancyGrid, 'likelihood_map', QOS_MAPA)
        self.create_subscription(OccupancyGrid, 'map', self.recibir_mapa, QOS_MAPA)

    def recibir_mapa(self, msg: OccupancyGrid):
        grid = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        campo = campo_de_probabilidad(grid, msg.info.resolution, self.sigma_sensor)

        salida = OccupancyGrid()
        salida.header = msg.header
        salida.info = msg.info
        salida.data = np.rint(campo).astype(np.int8).flatten().tolist()
        self.publisher_.publish(salida)
        self.get_logger().info(
            f'Campo de verosimilitud publicado (sigma_sensor={self.sigma_sensor}).')


def main(args=None):
    rclpy.init(args=args)
    nodo = CampoVerosimilitud()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
