"""Nodo esqueleto del equipo.

No hace nada útil todavía: sirve para que el template compile, arranque, espere
el mapa y pase el CI. Reemplazalo (o agregá nodos) con tu estrategia.

Lo que ya resuelve:
- Se suscribe a /map con el QoS que usa el juez (reliable + transient local).
- No publica nada en /cmd_vel ni en /victimas hasta recibir el mapa (§5).
- Deja armado el canal de /victimas, /done y /cmd_vel.

Lo que tenés que hacer vos, en este orden:
1. Localizarte en el mapa. Arrancás cerca del origen (0, 0), pero no exactamente ahí (§1).
2. Explorar y esquivar obstáculos.
3. Detectar cajas rojas y reportarlas con `armar_reporte`.
4. Opcional: volver a la largada y publicar en /done.
"""
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from std_msgs.msg import Empty
from sensor_msgs.msg import CameraInfo, Image, LaserScan


import tf2_ros
from cv_bridge import CvBridge


from equipo_jar.core import Board, DetectionClusterer, ParticleFilter
from equipo_jar.detector import ProcesadorVision
from equipo_jar.reportes import armar_reporte


QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class NodoEquipo(Node):
    def __init__(self):
        super().__init__('nodo_equipo')
        self.declare_parameter('frecuencia_control_hz', 10.0)
        self.declare_parameter('velocidad_lineal_max', 0.30)
        self.declare_parameter('velocidad_angular_max', 1.00)
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value

        self.mapa = None

        # --- INSTANCIAS DE LÓGICA PURA ---
        self.procesador_vision = ProcesadorVision()
        self.clusterer = DetectionClusterer()
        self.filtro = None  # Se instancia al recibir el mapa

        # --- HERRAMIENTAS DE ROS Y TF2 ---
        self.puente_cv = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # --- MEMORIA A CORTO PLAZO DE SENSORES ---
        self.ultima_imagen = None
        self.info_camara = None
        self.ultimo_scan = None

        # --- SUSCRIPCIONES ---
        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.sub_cam_info = self.create_subscription(
            CameraInfo, '/cam_1/color/camera_info', self.al_recibir_info_camara, qos_profile_sensor_data)
        self.sub_cam_img = self.create_subscription(
            Image, '/cam_1/color/image_raw', self.al_recibir_imagen, qos_profile_sensor_data)
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)

        # --- PUBLICADORES ---
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)

        periodo = 1.0 / self.get_parameter('frecuencia_control_hz').value
        self.create_timer(periodo, self.ciclo_de_control)
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    def al_recibir_mapa(self, mapa):
        if self.mapa is None:
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m')
        self.mapa = Board.from_msg(mapa_msg)
        self.filtro = ParticleFilter(self.mapa)
        self.filtro.initialize(pose=[0.0, 0.0, 0.0])

    def al_recibir_info_camara(self, msg):
        self.info_camara = {
            'fx': msg.k[0], 'fy': msg.k[4],
            'cx': msg.k[2], 'cy': msg.k[5],
            'frame_id': msg.header.frame_id,
            'width': msg.width, 'height': msg.height,
        }

    def al_recibir_imagen(self, msg):
        self.ultima_imagen = self.puente_cv.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def al_recibir_scan(self, msg):
        self.ultimo_scan = msg

    def ciclo_de_control(self):
        if self.mapa is None:
            return  # esperar el mapa: no mover el robot ni reportar antes

        # TODO: acá va tu estrategia. Por ahora el robot se queda quieto.
        # TODO 1: Actualizar pose del filtro con odometría y láser (predict / correct)
        # TODO 2: Procesamiento de visión y actualización del clusterer
        # TODO 3: Control de navegación (publicación en /cmd_vel)
        self.pub_cmd_vel.publish(Twist())

    def reportar_victima(self, x, y):
        """Publica un reporte. Cada llamada gasta un reporte (§7.2)."""
        self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))

    def terminar(self):
        """Publica en /done: termina la corrida (§5)."""
        self.pub_done.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoEquipo()
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
