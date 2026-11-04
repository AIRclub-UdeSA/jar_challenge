"""Avisa en /rojo_detectado cuando la cámara ve un cuadrado rojo."""

import cv2

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import Bool

from jar_percepcion.vision import PARAMETROS_ROJO, area_mayor_contorno, mascara_rojo

FRECUENCIA_HZ = 10.0


class DetectorColor(Node):
    """Publica True (nunca False) mientras haya una mancha roja grande."""

    def __init__(self):
        super().__init__('detector_color')
        for nombre, valor in PARAMETROS_ROJO.items():
            self.declare_parameter(nombre, valor)
        # Área mínima del contorno rojo para no confundirlo con ruido.
        self.declare_parameter('area_minima_px', 800.0)
        self.rojo = {n: self.get_parameter(n).value for n in PARAMETROS_ROJO}
        self.area_minima_px = self.get_parameter('area_minima_px').value

        self.puente = CvBridge()
        self.ultima_imagen = None
        self.ultimo_valor_publicado = None

        self.publisher_ = self.create_publisher(Bool, 'rojo_detectado', 10)
        # La cámara publica best effort: con reliable no llegaría ningún frame.
        self.create_subscription(Image, 'cam_1/color/image_raw', self.recibir_imagen,
                                 qos_profile_sensor_data)
        self.create_timer(1.0 / FRECUENCIA_HZ, self.procesar_imagen)

    def recibir_imagen(self, msg: Image):
        self.ultima_imagen = self.puente.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def hay_cuadrado_rojo(self) -> bool:
        if self.ultima_imagen is None:
            return False
        imagen_hsv = cv2.cvtColor(self.ultima_imagen, cv2.COLOR_BGR2HSV)
        return area_mayor_contorno(mascara_rojo(imagen_hsv, **self.rojo)) > self.area_minima_px

    def procesar_imagen(self):
        detectado = self.hay_cuadrado_rojo()
        if detectado:
            self.publisher_.publish(Bool(data=True))
        if detectado != self.ultimo_valor_publicado:
            self.get_logger().info('Veo un cuadrado rojo' if detectado else 'Ya no veo rojo')
            self.ultimo_valor_publicado = detectado


def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorColor()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
