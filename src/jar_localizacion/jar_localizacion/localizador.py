"""Localización en el mapa con un filtro de partículas propio."""

import math

import numpy as np

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time

from geometry_msgs.msg import (Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped,
                               Quaternion, TransformStamped)
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan

from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from jar_localizacion.filtro import (calidad_scan, componer_poses, dispersar, dispersion_xy,
                                     estimar_pose, invertir_pose, mover_particulas,
                                     normalizar_angulo, pesar_particulas, remuestrear,
                                     sembrar_particulas, usar_filtro)

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def yaw_de_quaternion(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_a_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


class Localizador(Node):
    """
    Filtro de partículas que publica la transformada map -> odom.

    Predice con cada /odom y corrige con cada /scan contra /likelihood_map.
    `initialpose` (2D Pose Estimate de RViz) vuelve a sembrar las partículas
    alrededor de la pose marcada.

    Con `respaldo_odometria`, si la pose del filtro no es confiable se deja
    fija map -> odom: la pose sigue a la odometría desde la última corrección
    buena. Si eso dura `scans_para_resembrar` scans y la odometría calza
    claramente mejor, las partículas se vuelven a sembrar alrededor de ella.
    """

    def __init__(self):
        super().__init__('localizador')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('num_particulas', 300)
        self.declare_parameter('pose_inicial_x', 0.0)
        self.declare_parameter('pose_inicial_y', 0.0)
        self.declare_parameter('pose_inicial_theta', 0.0)
        self.declare_parameter('dispersion_inicial_xy', 0.3)
        self.declare_parameter('dispersion_inicial_theta', 0.3)
        # Varianzas del modelo de movimiento mecanum (no desvíos).
        for i in range(1, 7):
            self.declare_parameter(f'gamma{i}', 0.0025)
        # Corrigen errores de escala de la odometría (semana 10).
        self.declare_parameter('escala_avance', 1.0)
        self.declare_parameter('escala_giro', 1.0)
        self.declare_parameter('submuestreo_scan', 15)
        # Ruido después de remuestrear: sin él, con el robot quieto el filtro
        # colapsa a una sola partícula (0 para desactivarlo).
        self.declare_parameter('ruido_remuestreo_xy', 0.01)
        self.declare_parameter('ruido_remuestreo_theta', 0.01)
        self.declare_parameter('respaldo_odometria', True)
        self.declare_parameter('dispersion_maxima', 0.15)
        # Puntos de probabilidad (0-100) que el filtro puede calzar peor que la odometría.
        self.declare_parameter('tolerancia_calidad', 5.0)
        self.declare_parameter('scans_para_resembrar', 10)
        self.declare_parameter('dispersion_resembrado', 0.05)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.num_particulas = self.get_parameter('num_particulas').value
        self.dispersion_xy = self.get_parameter('dispersion_inicial_xy').value
        self.dispersion_theta = self.get_parameter('dispersion_inicial_theta').value
        self.gammas = [self.get_parameter(f'gamma{i}').value for i in range(1, 7)]
        self.escala_avance = self.get_parameter('escala_avance').value
        self.escala_giro = self.get_parameter('escala_giro').value
        self.submuestreo_scan = max(1, self.get_parameter('submuestreo_scan').value)
        self.ruido_xy = self.get_parameter('ruido_remuestreo_xy').value
        self.ruido_theta = self.get_parameter('ruido_remuestreo_theta').value
        self.respaldo_odometria = self.get_parameter('respaldo_odometria').value
        self.dispersion_maxima = self.get_parameter('dispersion_maxima').value
        self.tolerancia_calidad = self.get_parameter('tolerancia_calidad').value
        self.scans_para_resembrar = self.get_parameter('scans_para_resembrar').value
        self.dispersion_resembrado = self.get_parameter('dispersion_resembrado').value

        self.pose_inicial = (self.get_parameter('pose_inicial_x').value,
                             self.get_parameter('pose_inicial_y').value,
                             self.get_parameter('pose_inicial_theta').value)
        self.particulas = sembrar_particulas(
            *self.pose_inicial, self.dispersion_xy, self.dispersion_theta, self.num_particulas)

        self.campo = None
        self.info_mapa = None
        self.ultimo_odom = None
        # Última map -> odom aceptada; None hasta la primera odometría.
        self.map_odom = None
        self.scans_con_odometria = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(OccupancyGrid, 'likelihood_map', self.recibir_campo, QOS_MAPA)
        self.create_subscription(LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.recibir_odom, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, 'initialpose', self.recibir_pose_inicial, 10)

        self.pub_particulas = self.create_publisher(PoseArray, 'particlecloud', 10)
        self.pub_camino_odom = self.create_publisher(Path, 'camino_odom', 10)
        self.pub_camino_corregido = self.create_publisher(Path, 'camino_corregido', 10)

        # camino_odom carga poses de odom en map: solo coincide si map y odom
        # arrancan alineados (pose inicial en el origen).
        self.camino_odom_msg = Path()
        self.camino_corregido_msg = Path()
        for camino in (self.camino_odom_msg, self.camino_corregido_msg):
            camino.header.frame_id = self.map_frame

    # ---------- callbacks ----------

    def recibir_campo(self, msg: OccupancyGrid):
        self.campo = np.array(msg.data, dtype=np.float64).reshape(
            (msg.info.height, msg.info.width))
        self.info_mapa = msg.info
        self.get_logger().info('Campo de verosimilitud recibido.')

    def recibir_pose_inicial(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_de_quaternion(msg.pose.pose.orientation)
        self.particulas = sembrar_particulas(
            x, y, theta, self.dispersion_xy, self.dispersion_theta, self.num_particulas)
        self.pose_inicial = (x, y, theta)
        if self.ultimo_odom is not None:
            self.map_odom = componer_poses(self.pose_inicial, invertir_pose(self.ultimo_odom))
        self.scans_con_odometria = 0
        self.publicar_particulas()
        self.get_logger().info(f'Partículas re-sembradas en ({x:.2f}, {y:.2f}, {theta:.2f}).')

    def recibir_odom(self, msg: Odometry):
        """Predicción con el desplazamiento desde la odometría anterior."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = yaw_de_quaternion(msg.pose.pose.orientation)

        if self.ultimo_odom is not None:
            x0, y0, theta0 = self.ultimo_odom
            self.particulas = mover_particulas(
                self.particulas, x - x0, y - y0, normalizar_angulo(theta - theta0), theta0,
                self.gammas, self.escala_avance, self.escala_giro)
            self.publicar_particulas()
            self.acumular_camino(self.camino_odom_msg, self.pub_camino_odom, x, y, theta)

        self.ultimo_odom = (x, y, theta)
        if self.map_odom is None:
            self.map_odom = componer_poses(self.pose_inicial, invertir_pose(self.ultimo_odom))

    def recibir_scan(self, msg: LaserScan):
        """Corrección: pesa, remuestrea y publica la pose estimada."""
        if self.campo is None or self.ultimo_odom is None:
            return
        puntos = self.scan_a_puntos(msg)
        if puntos is None or len(puntos) == 0:
            return

        origen = self.info_mapa.origin.position
        pesos = pesar_particulas(self.particulas, puntos, self.campo,
                                 self.info_mapa.resolution, origen.x, origen.y)
        self.particulas = dispersar(remuestrear(self.particulas, pesos),
                                    self.ruido_xy, self.ruido_theta)
        self.publicar_particulas()

        pose = estimar_pose(self.particulas)
        if self.respaldo_odometria:
            pose = self.elegir_pose(pose, puntos)
        self.acumular_camino(self.camino_corregido_msg, self.pub_camino_corregido, *pose)
        self.map_odom = componer_poses(pose, invertir_pose(self.ultimo_odom))
        self.publicar_transformada()

    # ---------- auxiliares ----------

    def elegir_pose(self, pose_filtro, puntos):
        """Pose del filtro si es confiable; si no, la de odometría (map -> odom fija)."""
        pose_odometria = componer_poses(self.map_odom, self.ultimo_odom)
        origen = self.info_mapa.origin.position
        argumentos = (puntos, self.campo, self.info_mapa.resolution, origen.x, origen.y)
        calidad_filtro = calidad_scan(pose_filtro, *argumentos)
        calidad_odometria = calidad_scan(pose_odometria, *argumentos)
        dispersion = dispersion_xy(self.particulas)

        if usar_filtro(calidad_filtro, calidad_odometria, dispersion,
                       self.dispersion_maxima, self.tolerancia_calidad):
            if self.scans_con_odometria:
                self.get_logger().info(
                    f'Vuelvo al filtro después de {self.scans_con_odometria} scans '
                    f'con odometría (calidad {calidad_filtro:.0f}).')
            self.scans_con_odometria = 0
            return pose_filtro

        if not self.scans_con_odometria:
            self.get_logger().warn(
                f'Filtro poco confiable (calidad {calidad_filtro:.0f} contra '
                f'{calidad_odometria:.0f} de odometría, dispersión {dispersion:.2f} m): '
                'sigo con odometría.')
        self.scans_con_odometria += 1
        if (self.scans_con_odometria >= self.scans_para_resembrar
                and calidad_odometria > calidad_filtro + self.tolerancia_calidad):
            self.particulas = sembrar_particulas(
                *pose_odometria, self.dispersion_resembrado, self.dispersion_resembrado,
                self.num_particulas)
            self.get_logger().warn('Partículas re-sembradas alrededor de la odometría.')
            self.scans_con_odometria = 0
        return pose_odometria

    def scan_a_puntos(self, msg: LaserScan):
        """Puntos Nx2 del scan en base_frame, submuestreados."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except Exception:
            return None

        rangos = np.array(msg.ranges)[::self.submuestreo_scan]
        angulos = (msg.angle_min + np.arange(len(msg.ranges)) * msg.angle_increment)[
            ::self.submuestreo_scan]
        validos = np.isfinite(rangos) & (rangos >= msg.range_min) & (rangos <= msg.range_max)
        rangos, angulos = rangos[validos], angulos[validos]

        xs_laser = rangos * np.cos(angulos)
        ys_laser = rangos * np.sin(angulos)
        tx, ty = tf.transform.translation.x, tf.transform.translation.y
        yaw = yaw_de_quaternion(tf.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.stack([tx + c * xs_laser - s * ys_laser,
                         ty + s * xs_laser + c * ys_laser], axis=1)

    def publicar_particulas(self):
        msg = PoseArray()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, theta, _ in self.particulas:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.orientation = yaw_a_quaternion(float(theta))
            msg.poses.append(pose)
        self.pub_particulas.publish(msg)

    def acumular_camino(self, camino_msg: Path, publisher, x, y, theta):
        pose = PoseStamped()
        pose.header.frame_id = self.map_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation = yaw_a_quaternion(float(theta))
        camino_msg.poses.append(pose)
        camino_msg.header.stamp = pose.header.stamp
        publisher.publish(camino_msg)

    def publicar_transformada(self):
        """Publica map -> odom = (map -> base) * inversa(odom -> base)."""
        x, y, theta = self.map_odom
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.rotation = yaw_a_quaternion(theta)
        self.tf_broadcaster.sendTransform(t)


def main(args=None):
    rclpy.init(args=args)
    nodo = Localizador()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
