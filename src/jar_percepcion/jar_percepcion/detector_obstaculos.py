"""Ubica en el frame map los objetos del scan que el mapa no tiene."""

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time

from geometry_msgs.msg import Pose, PoseArray
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray

from tf2_ros import Buffer, TransformListener

from jar_percepcion.geometria import (agrupar_en_rachas, probabilidad_en_campo, scan_a_map,
                                      yaw_de_quaternion)
from jar_percepcion.scan import copiar_scan

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


class DetectorObstaculos(Node):
    """
    Publica en map los objetos del scan que el mapa no tiene.

    Compara cada punto del /scan contra /likelihood_map, agrupa en rachas
    contiguas los que el mapa no esperaba y publica el centro de cada una.
    Para ubicar cajas rojas se remapea scan a scan_rojo y se usa
    umbral_probabilidad 101, porque esos puntos ya están filtrados por color.
    """

    def __init__(self):
        super().__init__('detector_obstaculos')
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('umbral_probabilidad', 25.0)
        self.declare_parameter('racha_minima', 3)
        # Descarta rebotes del lidar contra la propia estructura del robot.
        self.declare_parameter('distancia_minima_valida', 0.25)

        self.map_frame = self.get_parameter('map_frame').value
        self.umbral_probabilidad = self.get_parameter('umbral_probabilidad').value
        self.racha_minima = self.get_parameter('racha_minima').value
        self.distancia_minima_valida = self.get_parameter('distancia_minima_valida').value

        self.campo = None
        self.info_mapa = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.aviso_tf_mostrado = False

        self.create_subscription(OccupancyGrid, 'likelihood_map', self.recibir_campo, QOS_MAPA)
        self.create_subscription(LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data)
        self.pub_scan_no_mapeado = self.create_publisher(LaserScan, 'scan_no_mapeado', 10)
        self.pub_objetos = self.create_publisher(PoseArray, 'objetos_no_mapeados', 10)
        self.pub_marcadores = self.create_publisher(
            MarkerArray, 'objetos_no_mapeados_markers', 10)

    def recibir_campo(self, msg: OccupancyGrid):
        self.campo = np.array(msg.data, dtype=np.float64).reshape(
            (msg.info.height, msg.info.width))
        self.info_mapa = msg.info
        self.get_logger().info('Campo de verosimilitud recibido.')

    def obtener_transform_a_map(self, frame_id_scan):
        try:
            return self.tf_buffer.lookup_transform(self.map_frame, frame_id_scan, Time())
        except Exception as e:
            if not self.aviso_tf_mostrado:
                self.get_logger().warn(
                    f'Todavía no hay tf {frame_id_scan}->{self.map_frame}: {e}')
                self.aviso_tf_mostrado = True
            return None

    def recibir_scan(self, msg: LaserScan):
        if self.campo is None:
            return
        transform = self.obtener_transform_a_map(msg.header.frame_id)
        if transform is None:
            return

        rangos = np.array(msg.ranges)
        angulos = msg.angle_min + np.arange(len(rangos)) * msg.angle_increment
        minimo = max(msg.range_min, self.distancia_minima_valida)
        validos = np.isfinite(rangos) & (rangos >= minimo) & (rangos <= msg.range_max)

        t = transform.transform
        xs_map, ys_map = scan_a_map(np.where(validos, rangos, 0.0), angulos,
                                    t.translation.x, t.translation.y,
                                    yaw_de_quaternion(t.rotation))
        origen = self.info_mapa.origin.position
        probabilidad, dentro = probabilidad_en_campo(
            xs_map, ys_map, self.campo, self.info_mapa.resolution, origen.x, origen.y)
        # Fuera del mapa no hay nada que comparar: no se reporta como objeto.
        no_mapeado = validos & dentro & (probabilidad < self.umbral_probabilidad)

        self.pub_scan_no_mapeado.publish(
            copiar_scan(msg, np.where(no_mapeado, rangos, np.inf)))
        self.publicar_objetos(no_mapeado, xs_map, ys_map)

    def publicar_objetos(self, no_mapeado, xs_map, ys_map):
        ahora = self.get_clock().now().to_msg()
        objetos = PoseArray()
        objetos.header.frame_id = self.map_frame
        objetos.header.stamp = ahora
        marcadores = MarkerArray()
        marcadores.markers.append(Marker(action=Marker.DELETEALL))

        for i, indices in enumerate(agrupar_en_rachas(no_mapeado)):
            if len(indices) < self.racha_minima:
                continue
            pose = Pose()
            pose.position.x = float(np.mean(xs_map[indices]))
            pose.position.y = float(np.mean(ys_map[indices]))
            pose.orientation.w = 1.0
            objetos.poses.append(pose)

            marcador = Marker()
            marcador.header = objetos.header
            marcador.ns = 'objetos_no_mapeados'
            marcador.id = i
            marcador.type = Marker.SPHERE
            marcador.action = Marker.ADD
            marcador.pose = pose
            marcador.scale.x = marcador.scale.y = marcador.scale.z = 0.2
            marcador.color.r = 1.0
            marcador.color.g = 0.3
            marcador.color.a = 1.0
            marcadores.markers.append(marcador)

        self.pub_objetos.publish(objetos)
        self.pub_marcadores.publish(marcadores)


def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorObstaculos()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
