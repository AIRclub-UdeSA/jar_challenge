import math

import cv2
import numpy as np
import rclpy
import tf2_ros
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Bool
from cv_bridge import CvBridge

from equipo_jar.imu import GiroImu


def mascara_rojo(imagen_hsv, hue_bajo_1, hue_alto_1, hue_bajo_2, hue_alto_2,
                  saturacion_min, valor_min):
    """Máscara binaria (255 = rojo). Es la misma función que ya escribiste
    en detector.py — la repetimos acá tal cual para que detector_scan sea
    un nodo independiente (no depende de importar detector.py)."""
    bajo1 = np.array([hue_bajo_1, saturacion_min, valor_min])
    alto1 = np.array([hue_alto_1, 255, 255])
    bajo2 = np.array([hue_bajo_2, saturacion_min, valor_min])
    alto2 = np.array([hue_alto_2, 255, 255])
    mascara1 = cv2.inRange(imagen_hsv, bajo1, alto1)
    mascara2 = cv2.inRange(imagen_hsv, bajo2, alto2)
    return cv2.bitwise_or(mascara1, mascara2)


def filtrar_por_profundidad(rangos, es_rojo, tolerancia):
    """Saca los rayos "rojos" que en realidad pasaron rozando el borde de una caja y pegaron
    en algo de atrás: como cámara y lidar no están en el mismo lugar, esos rayos caen sobre
    píxeles rojos aunque midan lo que hay detrás.

    Los rayos rojos contiguos forman grupos (una cara de caja: distancias parecidas). En cada
    grupo se descartan los que están más de `tolerancia` metros detrás del más cercano. El
    scan es circular: el primer y el último rayo son vecinos (con el lidar montado a 180°,
    quedan justo adelante del robot), así que un grupo puede cruzar ese corte.
    """
    resultado = es_rojo.copy()
    if tolerancia <= 0:
        return resultado
    indices = np.flatnonzero(es_rojo)
    if indices.size == 0:
        return resultado
    grupos = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    if len(grupos) > 1 and grupos[0][0] == 0 and grupos[-1][-1] == len(es_rojo) - 1:
        grupos[0] = np.concatenate([grupos.pop(), grupos[0]])
    for grupo in grupos:
        r = rangos[grupo]
        resultado[grupo[r > r.min() + tolerancia]] = False
    return resultado


def cuaternion_a_rotacion(x, y, z, w):
    """Matriz de rotación 3x3 equivalente a un quaternion (x, y, z, w).
    Álgebra de siempre, no es el objetivo de este workshop."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def puntos_laser_a_pixeles(rangos, angulos, rotacion, traslacion, fx, fy, cx, cy):
    """
    Convertir cada punto del lidar (dado como range/angulo) en el
    pixel (u, v) donde caería si la cámara lo estuviera mirando.

    Parámetros: rangos y angulos son arrays de igual largo (uno por rayo
    del lidar); rotacion es una matriz 3x3 y traslacion un vector de 3
    (la tf real que ubica la cámara respecto del lidar); fx, fy, cx, cy son
    los intrínsecos de la cámara (de camera_info).
    """
    x = rangos * np.cos(angulos)
    y = rangos * np.sin(angulos)
    z = np.zeros(len(rangos))
    puntos_laser = np.stack((x,y,z), axis=1)
    punto_camara = (puntos_laser @ rotacion.T) + traslacion
    x_cam, y_cam, z_cam = punto_camara.T
    adelante = z_cam > 0.01
    with np.errstate(divide='ignore', invalid='ignore'):
        u = fx * x_cam / z_cam + cx
        v = fy * y_cam / z_cam + cy
    return u, v, adelante


class DetectorScanColor(Node):
    """
    Proyecta cada punto del /scan sobre la imagen de la cámara (usando la
    tf entre el lidar y la cámara, y los intrínsecos de camera_info) y
    publica en /scan_rojo una copia del LaserScan donde solo sobreviven
    los rangos cuyo punto cae, en la imagen, sobre un pixel rojo. El resto
    se pone en infinito (sin retorno), como cualquier LaserScan filtrado.
    """

    def __init__(self):
        super().__init__('detector_scan')

        self.declare_parameter('hue_rojo_bajo_1', 0.0)
        self.declare_parameter('hue_rojo_alto_1', 10.0)
        self.declare_parameter('hue_rojo_bajo_2', 170.0)
        self.declare_parameter('hue_rojo_alto_2', 180.0)
        self.declare_parameter('saturacion_min', 120.0)
        self.declare_parameter('valor_min', 80.0)
        # Píxeles que se achica cada mancha roja antes de cruzarla con el lidar. Cámara y lidar no
        # están en el mismo lugar ni miden en el mismo instante: sin esto, rayos que pasan rozando
        # el borde de una caja y pegan en lo que hay detrás caen sobre píxeles rojos. 0 = apagado.
        self.declare_parameter('erosion_mascara', 2)
        # m: un rayo rojo más de esto detrás del más cercano de su grupo pasó por el borde de
        # la caja y pegó en lo de atrás; se descarta. 0 = filtro apagado.
        self.declare_parameter('tolerancia_profundidad', 0.3)
        # Girando rápido, la imagen y el scan son de instantes distintos (el scan tarda ~0,13 s
        # en dar la vuelta) y los puntos rojos se corren sobre la pared (semana 10). Por encima
        # de esta velocidad de giro (rad/s, medida con el IMU) no se marca nada como rojo.
        # 0 = filtro apagado.
        self.declare_parameter('giro_maximo', 0.3)

        self.hue_rojo_bajo_1 = self.get_parameter('hue_rojo_bajo_1').value
        self.hue_rojo_alto_1 = self.get_parameter('hue_rojo_alto_1').value
        self.hue_rojo_bajo_2 = self.get_parameter('hue_rojo_bajo_2').value
        self.hue_rojo_alto_2 = self.get_parameter('hue_rojo_alto_2').value
        self.saturacion_min = self.get_parameter('saturacion_min').value
        self.valor_min = self.get_parameter('valor_min').value
        self.erosion_mascara = self.get_parameter('erosion_mascara').value
        self.tolerancia_profundidad = self.get_parameter('tolerancia_profundidad').value
        # Píxeles rojos (después de la erosión) para considerar que hay rojo a la vista.
        self.declare_parameter('pixeles_rojo_minimos', 150)
        self.pixeles_rojo_minimos = self.get_parameter('pixeles_rojo_minimos').value
        self.giro_maximo = self.get_parameter('giro_maximo').value

        self.puente = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.giro = GiroImu(self, self.tf_buffer)

        self.ultima_imagen = None
        self.info_camara = None
        self.aviso_tf_mostrado = False

        # Las tres suscripciones son a sensores, así que las tres van con
        # qos_profile_sensor_data (best effort) — ver semana 03.
        self.create_subscription(
            Image, 'cam_1/color/image_raw', self.recibir_imagen,
            qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, 'cam_1/color/camera_info', self.recibir_info_camara,
            qos_profile_sensor_data
        )
        self.create_subscription(
            LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data
        )
        
        self.publisher_ = self.create_publisher(
            LaserScan, 'scan_rojo', 10
        )
        # Máscara de rojo como imagen, para calibrar el HSV en RViz. Solo se arma si alguien
        # está suscripto, así que en la corrida no cuesta nada.
        self.pub_mascara = self.create_publisher(Image, 'mascara_rojo', 10)
        # Si la cámara ve rojo, girando o no. El evasor lo usa para girar más despacio y darle
        # tiempo a este nodo de marcar los puntos del lidar (girando rápido no los marca).
        self.pub_rojo_a_la_vista = self.create_publisher(Bool, 'rojo_a_la_vista', 10)
        # Los umbrales de color y giro_maximo se pueden cambiar en vivo para calibrar, p. ej.:
        #   ros2 param set /detector_scan saturacion_min 90.0
        self.add_on_set_parameters_callback(self.cambiar_parametros)

    def recibir_imagen(self, msg: Image):
        # Se guarda el mensaje crudo y se convierte recién en recibir_scan: la cámara publica a
        # 30 Hz y el scan llega a ~10 Hz, así que convertir cada imagen era trabajo tirado.
        self.ultima_imagen = msg

    def recibir_info_camara(self, msg: CameraInfo):
        # K es la matriz intrínseca 3x3 en fila: [fx 0 cx; 0 fy cy; 0 0 1]
        self.info_camara = {
            'fx': msg.k[0], 'fy': msg.k[4],
            'cx': msg.k[2], 'cy': msg.k[5],
            'frame_id': msg.header.frame_id,
            'width': msg.width, 'height': msg.height,
        }

    def obtener_transform_laser_a_camara(self, frame_id_scan):
        """Busca en tf2 la transformada del frame del lidar al frame
        óptico de la cámara. None si todavía no está disponible (por
        ejemplo, en el primer instante después de lanzar el nodo)."""
        try:
            return self.tf_buffer.lookup_transform(
                self.info_camara['frame_id'], frame_id_scan, Time()
            )
        except tf2_ros.TransformException as e:
            if not self.aviso_tf_mostrado:
                self.get_logger().warn(f'Todavía no hay tf laser->cámara: {e}')
                self.aviso_tf_mostrado = True
            return None

    def recibir_scan(self, msg: LaserScan):
        """
        Callback del /scan — corre una vez por cada scan que llega.
        """
        # 1)
        if self.ultima_imagen is None or self.info_camara is None:
            return

        # La máscara se calcula siempre: aunque gire rápido, saber que hay rojo a la vista le
        # sirve al evasor para frenar el giro.
        mask = self.calcular_mascara()
        self.pub_rojo_a_la_vista.publish(
            Bool(data=bool(np.count_nonzero(mask) >= self.pixeles_rojo_minimos)))

        # Girando rápido no se confía en qué rayos son rojos (la imagen y el scan son de
        # instantes distintos): se publica el scan sin puntos rojos.
        if self.girando_rapido():
            self.publicar_rojo(msg, np.full(len(msg.ranges), math.inf))
            return

        # 2)
        transformada = self.obtener_transform_laser_a_camara(msg.header.frame_id)
        if transformada is None:
            return

        # 3)
        rot = transformada.transform.rotation
        t = transformada.transform.translation
        traslacion = np.array([t.x, t.y, t.z])
        rotacion = cuaternion_a_rotacion(rot.x, rot.y, rot.z, rot.w)

        # 4)
        rangos = np.array(msg.ranges)
        n = len(rangos)
        angulos = msg.angle_min + np.arange(n) * msg.angle_increment

        # 5)
        validos = np.isfinite(rangos) & (rangos >= msg.range_min) & (rangos <= msg.range_max)

        # 6)
        cam = self.info_camara
        u, v, adelante = puntos_laser_a_pixeles(rangos, angulos, rotacion, traslacion, cam['fx'], cam['fy'], cam['cx'], cam['cy'])

        # 7)
        dentro_de_imagen = (
            adelante
            & (u >= 0) & (u < cam['width'])
            & (v >= 0) & (v < cam['height'])
        )

        # 8) la máscara ya se calculó al principio (calcular_mascara)

        # 9)
        idx_dentro_imagen = np.where(dentro_de_imagen)[0]
        es_rojo = np.zeros(len(rangos), dtype=bool) # False por defecto
        filas = v[idx_dentro_imagen].astype(int)
        columnas = u[idx_dentro_imagen].astype(int)
        es_rojo[idx_dentro_imagen] = mask[filas, columnas] > 0

        # 10) sin los rayos que se escaparon por el borde de la caja hacia lo de atrás
        es_rojo = filtrar_por_profundidad(rangos, validos & es_rojo, self.tolerancia_profundidad)
        rangos_filtrados = np.where(es_rojo, rangos, math.inf)

        # 11)
        self.publicar_rojo(msg, rangos_filtrados)

    def cambiar_parametros(self, parametros):
        calibrables = ('hue_rojo_bajo_1', 'hue_rojo_alto_1', 'hue_rojo_bajo_2',
                       'hue_rojo_alto_2', 'saturacion_min', 'valor_min', 'giro_maximo',
                       'erosion_mascara', 'tolerancia_profundidad')
        for p in parametros:
            if p.name in calibrables:
                setattr(self, p.name, int(p.value) if p.name == 'erosion_mascara' else float(p.value))
                self.get_logger().info(f'{p.name} = {p.value}')
        return SetParametersResult(successful=True)

    def calcular_mascara(self):
        """Máscara de rojo de la última imagen (255 = rojo), erosionada. Si alguien la mira,
        la publica en /mascara_rojo."""
        imagen = self.puente.imgmsg_to_cv2(self.ultima_imagen, desired_encoding='bgr8')
        img_hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        mask = mascara_rojo(img_hsv, self.hue_rojo_bajo_1, self.hue_rojo_alto_1,
                            self.hue_rojo_bajo_2, self.hue_rojo_alto_2,
                            self.saturacion_min, self.valor_min)
        if self.erosion_mascara > 0:
            mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=int(self.erosion_mascara))
        if self.pub_mascara.get_subscription_count() > 0:
            imagen_mascara = self.puente.cv2_to_imgmsg(mask, encoding='mono8')
            imagen_mascara.header = self.ultima_imagen.header
            self.pub_mascara.publish(imagen_mascara)
        return mask

    def girando_rapido(self) -> bool:
        return (self.giro_maximo > 0 and self.giro.listo
                and abs(self.giro.velocidad) > self.giro_maximo)

    def publicar_rojo(self, msg: LaserScan, rangos_filtrados):
        """Copia de `msg` con los rangos filtrados (inf = no es rojo), en /scan_rojo."""
        new_msg = LaserScan()
        new_msg.header = msg.header
        new_msg.angle_min = msg.angle_min
        new_msg.angle_max = msg.angle_max
        new_msg.angle_increment = msg.angle_increment
        new_msg.time_increment = msg.time_increment
        new_msg.scan_time = msg.scan_time
        new_msg.range_min = msg.range_min
        new_msg.range_max = msg.range_max
        new_msg.ranges = rangos_filtrados.tolist()

        self.publisher_.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorScanColor()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
