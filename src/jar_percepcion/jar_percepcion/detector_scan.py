"""Publica en /scan_rojo los rayos del lidar que caen sobre píxeles rojos."""

import math

import cv2
import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from sensor_msgs.msg import CameraInfo, Image, LaserScan

import tf2_ros

from jar_percepcion.geometria import (cuaternion_a_rotacion, pixeles_en_mascara,
                                      puntos_laser_a_pixeles)
from jar_percepcion.scan import copiar_scan
from jar_percepcion.vision import PARAMETROS_ROJO, mascara_rojo


class DetectorScanColor(Node):
    """
    Publica en /scan_rojo los rayos del lidar que caen sobre rojo.

    Proyecta cada punto del /scan sobre la imagen (tf lidar -> cámara e
    intrínsecos de camera_info) y conserva solo los que caen sobre rojo.
    Mientras el robot gira más rápido que `giro_maximo`, imagen y scan no
    corresponden al mismo instante y no se marca ningún punto.
    """

    def __init__(self):
        super().__init__('detector_scan_color')
        for nombre, valor in PARAMETROS_ROJO.items():
            self.declare_parameter(nombre, valor)
        # rad/s; 0 desactiva el filtro.
        self.declare_parameter('giro_maximo', 0.0)
        self.rojo = {n: self.get_parameter(n).value for n in PARAMETROS_ROJO}
        self.giro_maximo = self.get_parameter('giro_maximo').value

        self.puente = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.ultima_imagen = None
        self.info_camara = None
        self.giro = 0.0
        self.aviso_tf_mostrado = False

        self.create_subscription(Image, 'cam_1/color/image_raw', self.recibir_imagen,
                                 qos_profile_sensor_data)
        self.create_subscription(CameraInfo, 'cam_1/color/camera_info',
                                 self.recibir_info_camara, qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.recibir_odom, 10)
        self.publisher_ = self.create_publisher(LaserScan, 'scan_rojo', 10)

    def recibir_imagen(self, msg: Image):
        self.ultima_imagen = self.puente.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def recibir_info_camara(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.info_camara = {
            'fx': msg.k[0], 'fy': msg.k[4], 'cx': msg.k[2], 'cy': msg.k[5],
            'frame_id': msg.header.frame_id,
        }

    def recibir_odom(self, msg: Odometry):
        self.giro = msg.twist.twist.angular.z

    def obtener_transform_laser_a_camara(self, frame_id_scan):
        try:
            return self.tf_buffer.lookup_transform(
                self.info_camara['frame_id'], frame_id_scan, Time())
        except tf2_ros.TransformException as e:
            if not self.aviso_tf_mostrado:
                self.get_logger().warn(f'Todavía no hay tf laser->cámara: {e}')
                self.aviso_tf_mostrado = True
            return None

    def recibir_scan(self, msg: LaserScan):
        if self.ultima_imagen is None or self.info_camara is None:
            return
        rangos = np.array(msg.ranges)
        if self.giro_maximo > 0.0 and abs(self.giro) > self.giro_maximo:
            self.publisher_.publish(copiar_scan(msg, np.full(len(rangos), math.inf)))
            return
        transform = self.obtener_transform_laser_a_camara(msg.header.frame_id)
        if transform is None:
            return

        t = transform.transform.translation
        q = transform.transform.rotation
        angulos = msg.angle_min + np.arange(len(rangos)) * msg.angle_increment
        validos = np.isfinite(rangos) & (rangos >= msg.range_min) & (rangos <= msg.range_max)

        info = self.info_camara
        u, v, adelante = puntos_laser_a_pixeles(
            np.where(validos, rangos, 0.0), angulos,
            cuaternion_a_rotacion(q.x, q.y, q.z, q.w), np.array([t.x, t.y, t.z]),
            info['fx'], info['fy'], info['cx'], info['cy'])

        imagen_hsv = cv2.cvtColor(self.ultima_imagen, cv2.COLOR_BGR2HSV)
        es_rojo = pixeles_en_mascara(u, v, adelante, mascara_rojo(imagen_hsv, **self.rojo))
        self.publisher_.publish(
            copiar_scan(msg, np.where(validos & es_rojo, rangos, math.inf)))


def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorScanColor()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
