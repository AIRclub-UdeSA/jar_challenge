"""Odometry with the IMU's heading for AMCL: publishes TF odom_fused -> odom.

On the real robot /odom counts turns ~25 % too much, and AMCL keeps that heading error
after each turn. This node keeps the wheels' steps but turns them with the IMU heading
(logic and params in rumbo_imu.py). AMCL uses `odom_frame_id: odom_fused`, so the tree is
map -> odom_fused -> odom -> base_footprint. The robot's odom -> base_footprint and /odom
are untouched: the local costmap, the behaviors and mission still use `odom`.

Each transform has the stamp of its /odom message, so AMCL finds odom_fused -> base at
the scan stamp just as it found odom -> base before. Launched only with AMCL
(navegacion.launch.py): in pruebas.launch.py map -> odom is fixed.
"""
import math

from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

from equipo_jar.comun import correr, parametros_en_vivo
from equipo_jar.rumbo_imu import RumboImu


def yaw_de(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def segundos(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class OdometriaImu(Node):
    def __init__(self):
        super().__init__('odometria_imu')
        self.p = parametros_en_vivo(self, {
            'fuente_rumbo': 'giroscopo',     # giroscopo, orientacion or ruedas (off)
            'signo_giroscopo': 0,            # 0: learned from the wheels; 1 or -1: fixed
            'imu_vieja_s': 0.5,              # s without IMU before a step uses the wheels
            'signo_wz_minimo_rads': 0.2,     # only samples where the wheels turn this fast
            'signo_giro_minimo_rad': 0.5,    # wheel rotation needed to decide the sign
            'signo_correlacion_minima': 0.7,
            'escala_avance': 0.0,            # 0: learned (IMU turn / wheels turn); > 0 fixed
            'escala_giro_minimo_rad': 1.0,   # wheel turning needed to use the learned scale
            'escala_minima': 0.7,
            'escala_maxima': 1.1,
        })
        self.frame_fusionado = self.declare_parameter('frame_fusionado', 'odom_fused').value
        self.rumbo = RumboImu(self.p)
        self.signo_avisado = 0
        self.pasos_avisados = 0
        self.escala_avisada = 1.0
        self.tf = TransformBroadcaster(self)
        self.frame_odom = 'odom'
        # /odom and /imu/data are reliable (§7.1).
        self.create_subscription(Odometry, '/odom', self.al_recibir_odom, 50)
        self.create_subscription(Imu, '/imu/data', self.al_recibir_imu, 50)
        self.create_timer(5.0, self.informar)
        self.descartados = 0

    def descartar(self, valores):
        # One NaN would stay in the integrated heading for the rest of the run.
        if all(math.isfinite(v) for v in valores):
            return False
        self.descartados += 1
        if self.descartados == 1 or self.descartados % 100 == 0:
            self.get_logger().warning(
                f'{self.descartados} /odom or /imu/data messages with NaN/inf dropped')
        return True

    def al_recibir_imu(self, msg):
        t, gz, yaw = segundos(msg.header.stamp), msg.angular_velocity.z, yaw_de(msg.orientation)
        if self.descartar((t, gz, yaw)):
            return
        self.publicar(self.rumbo.con_imu(t, gz, yaw))

    def al_recibir_odom(self, msg):
        pose = msg.pose.pose
        valores = (segundos(msg.header.stamp), pose.position.x, pose.position.y,
                   yaw_de(pose.orientation))
        if self.descartar(valores):
            return
        # The turn rate only feeds the gyro sign: a bad one must not drop a good pose.
        # 0 is below signo_wz_minimo_rads, so that sample is left out of the sign.
        wz = msg.twist.twist.angular.z
        if not math.isfinite(wz):
            wz = 0.0
        self.frame_odom = msg.header.frame_id or 'odom'
        self.publicar(self.rumbo.con_odom(*valores, wz))

    def publicar(self, transformadas):
        if not transformadas:
            return
        mensajes = []
        for t, x, y, yaw in transformadas:
            tr = TransformStamped()
            tr.header.stamp = Time(nanoseconds=round(t * 1e9)).to_msg()
            tr.header.frame_id = self.frame_fusionado
            tr.child_frame_id = self.frame_odom
            tr.transform.translation.x = x
            tr.transform.translation.y = y
            tr.transform.rotation.z = math.sin(yaw / 2.0)
            tr.transform.rotation.w = math.cos(yaw / 2.0)
            mensajes.append(tr)
        self.tf.sendTransform(mensajes)

    def informar(self):
        signo = self.rumbo.signo
        if signo != self.signo_avisado and self.p['fuente_rumbo'] == 'giroscopo':
            self.get_logger().info(f'gyro sign {signo:+d} (wheels vs gyro)')
            self.signo_avisado = signo
        escala = self.rumbo.escala
        if abs(escala - self.escala_avisada) >= 0.02:
            self.get_logger().info(
                f'odom distance scale {escala:.3f} (IMU turn / wheels turn over '
                f'{self.rumbo.giro_ruedas:.1f} rad)')
            self.escala_avisada = escala
        if self.rumbo.pasos_ruedas != self.pasos_avisados:
            self.get_logger().info(
                f'{self.rumbo.pasos_ruedas} odom steps used the wheels heading '
                f'(no IMU or gyro sign not known yet)')
            self.pasos_avisados = self.rumbo.pasos_ruedas


def main(args=None):
    correr(OdometriaImu, args)
