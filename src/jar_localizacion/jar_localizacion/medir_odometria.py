"""Compara /odom contra el IMU (giro) y el lidar (avance)."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, LaserScan

from tf2_ros import Buffer, TransformListener

from jar_localizacion.filtro import normalizar_angulo
from jar_localizacion.localizador import yaw_de_quaternion
from jar_localizacion.odometria import indice_rayo_hacia_atras

PERIODO_REPORTE_S = 2.0


def segundos(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class MedirOdometria(Node):
    """
    Compara /odom contra el IMU (giro) y el lidar trasero (avance).

    Cada PERIODO_REPORTE_S segundos (del bag) imprime el giro acumulado
    según /odom y según el IMU, su cociente, cuánto avanzó el robot según
    /odom y la distancia que mide el lidar hacia atrás del robot.
    """

    def __init__(self):
        super().__init__('medir_odometria')
        self.declare_parameter('base_frame', 'base_footprint')
        self.base_frame = self.get_parameter('base_frame').value

        self.giro_odom = 0.0
        self.giro_imu = 0.0
        self.yaw_anterior = None
        self.t_imu_anterior = None
        self.inicio = None
        self.avance_odom = 0.0
        self.rango_atras = float('nan')
        self.t0 = None
        self.t_ultimo_reporte = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Odometry, 'odom', self.recibir_odom, 10)
        self.create_subscription(Imu, 'imu/data', self.recibir_imu, 10)
        self.create_subscription(LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data)

    def recibir_odom(self, msg: Odometry):
        p = msg.pose.pose.position
        if self.inicio is None:
            self.inicio = (p.x, p.y)
        self.avance_odom = math.hypot(p.x - self.inicio[0], p.y - self.inicio[1])

        # Con signo y normalizado: de 179° a -179° son +2°, no -358°.
        yaw = yaw_de_quaternion(msg.pose.pose.orientation)
        if self.yaw_anterior is not None:
            self.giro_odom += normalizar_angulo(yaw - self.yaw_anterior)
        self.yaw_anterior = yaw

        t = segundos(msg.header.stamp)
        if self.t0 is None:
            self.t0 = t
            self.t_ultimo_reporte = t
        elif t - self.t_ultimo_reporte >= PERIODO_REPORTE_S:
            self.t_ultimo_reporte = t
            self.reportar(t - self.t0)

    def recibir_imu(self, msg: Imu):
        t = segundos(msg.header.stamp)
        if self.t_imu_anterior is not None:
            self.giro_imu += msg.angular_velocity.z * (t - self.t_imu_anterior)
        self.t_imu_anterior = t

    def recibir_scan(self, msg: LaserScan):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except Exception:
            return
        i = indice_rayo_hacia_atras(msg.angle_min, msg.angle_increment, len(msg.ranges),
                                    yaw_de_quaternion(tf.transform.rotation))
        r = msg.ranges[i]
        self.rango_atras = r if math.isfinite(r) else float('nan')

    def reportar(self, t_bag):
        odom, imu = math.degrees(self.giro_odom), math.degrees(self.giro_imu)
        cociente = imu / odom if abs(odom) > 1.0 else float('nan')
        self.get_logger().info(
            f'bag {t_bag:5.1f} s | giro odom {odom:7.1f}°  IMU {imu:7.1f}°  '
            f'IMU/odom {cociente:6.3f} | avance odom {self.avance_odom:5.3f} m  '
            f'lidar atrás {self.rango_atras:5.3f} m')


def main(args=None):
    rclpy.init(args=args)
    nodo = MedirOdometria()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
