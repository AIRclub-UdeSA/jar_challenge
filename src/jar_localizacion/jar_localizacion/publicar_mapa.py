"""Publica un mapa de map_server en /map, retenido, sin depender de Nav2."""

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from nav_msgs.msg import OccupancyGrid

from jar_localizacion.mapa import cargar_mapa

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class PublicarMapa(Node):
    """
    Reemplazo de nav2_map_server para el localizador propio.

    Publica una sola vez con transient local, así que los nodos que se
    suscriben después igual lo reciben. No publica transformadas.
    """

    def __init__(self):
        super().__init__('publicar_mapa')
        self.declare_parameter('yaml_filename', '')
        self.declare_parameter('frame_id', 'map')
        ruta = self.get_parameter('yaml_filename').value
        if not ruta:
            raise ValueError('falta el parámetro yaml_filename (ruta al .yaml)')

        grilla, resolucion, origen = cargar_mapa(ruta)
        mapa = OccupancyGrid()
        mapa.header.frame_id = self.get_parameter('frame_id').value
        mapa.header.stamp = self.get_clock().now().to_msg()
        mapa.info.resolution = resolucion
        mapa.info.height, mapa.info.width = grilla.shape
        mapa.info.origin.position.x, mapa.info.origin.position.y = origen
        mapa.info.origin.orientation.w = 1.0
        mapa.data = grilla.flatten().tolist()

        self.pub = self.create_publisher(OccupancyGrid, 'map', QOS_MAPA)
        self.pub.publish(mapa)
        self.get_logger().info(
            f'Mapa publicado en /map: {mapa.info.width}x{mapa.info.height} celdas, '
            f'{resolucion:.3f} m/celda, origen ({origen[0]:.3f}, {origen[1]:.3f})')


def main(args=None):
    rclpy.init(args=args)
    nodo = PublicarMapa()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
