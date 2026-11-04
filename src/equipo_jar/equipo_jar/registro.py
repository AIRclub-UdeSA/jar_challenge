"""Registro de la corrida para ajustar params/ entre las dos corridas de la clasificación.

Solo lee tópicos y escribe archivos chicos en `carpeta/<fecha-hora>/` (ver
plan_de_trabajo/LOGS_CLASIFICACION.md). No publica nada ni toca la misión: si
algo falla, se avisa una vez y se sigue.

- pose.csv: pose y dispersión de amcl.
- movimiento.csv: /cmd_vel pedido contra /odom medido.
- roces.csv: distancia mínima entre el scan y el rectángulo del robot, y rayos rojos.
- rojo.csv: percentiles de H, S y V de los pixeles rojizos de la cámara.
- cpu.csv: carga de la computadora y frecuencia de /cmd_vel.
- img/: una imagen achicada cada `periodo_imagen_s`.
"""
import math
import os
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, LaserScan
from tf2_ros import Buffer, TransformListener


class Registro(Node):

    def __init__(self):
        super().__init__('registro')
        self.declare_parameter('registrar', True)
        self.declare_parameter('carpeta', '~/.ros/polibots')
        self.declare_parameter('periodo_imagen_s', 5.0)
        self.declare_parameter('periodo_color_s', 1.0)
        # Chasis del URDF en base_footprint (x de -0.18 a 0.12, y de -0.10 a 0.10).
        self.declare_parameter('chasis_atras', 0.18)
        self.declare_parameter('chasis_adelante', 0.12)
        self.declare_parameter('chasis_medio_ancho', 0.10)
        if not self.get_parameter('registrar').value:
            self.get_logger().info('Registro apagado (registrar: false)')
            self.activo = False
            return
        self.activo = True
        base = os.path.expanduser(self.get_parameter('carpeta').value)
        self.dir = os.path.join(base, time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(os.path.join(self.dir, 'img'), exist_ok=True)
        self.periodo_imagen = self.get_parameter('periodo_imagen_s').value
        self.periodo_color = self.get_parameter('periodo_color_s').value
        atras = self.get_parameter('chasis_atras').value
        adelante = self.get_parameter('chasis_adelante').value
        medio = self.get_parameter('chasis_medio_ancho').value
        self.chasis = (-atras, adelante, -medio, medio)
        self.archivos = {}
        self.cabeceras = {
            'pose': 't_pared,t_ros,x,y,yaw,sigma_xy,sigma_yaw',
            'movimiento': 't_pared,cmd_vx,cmd_vy,cmd_wz,odom_vx,odom_vy,odom_wz',
            'roces': 't_pared,distancia_min,rayos_rojos',
            'rojo': 't_pared,pixeles_rojizos,h5,h50,h95,s5,s50,s95,v5,v50,v95',
            'cpu': 't_pared,carga_1min,cmd_vel_hz',
        }
        self.avisos = set()
        self.puente = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        self.laser = None   # (x, y, yaw) del lidar en base_footprint
        self.ultimo_cmd = (0.0, 0.0, 0.0)
        self.cmds = 0
        self.t_imagen = 0.0
        self.t_color = 0.0
        self.t_odom = 0.0
        self.t0 = time.monotonic()
        self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.protegido(self.al_pose), 10)
        self.create_subscription(Twist, '/cmd_vel', self.protegido(self.al_cmd), 10)
        self.create_subscription(
            Odometry, '/odom', self.protegido(self.al_odom), qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, '/scan', self.protegido(self.al_scan), qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, 'scan_rojo', self.protegido(self.al_scan_rojo), qos_profile_sensor_data)
        self.rayos_rojos = 0
        self.create_subscription(
            Image, 'cam_1/color/image_raw', self.protegido(self.al_imagen),
            qos_profile_sensor_data)
        self.create_timer(5.0, self.protegido(self.al_cpu))
        self.get_logger().info(f'Registrando la corrida en {self.dir}')

    def protegido(self, funcion):
        def envuelta(*args):
            try:
                funcion(*args)
            except Exception as error:   # noqa: B902
                nombre = funcion.__name__
                if nombre not in self.avisos:
                    self.avisos.add(nombre)
                    self.get_logger().warn(f'Registro: error en {nombre}: {error!r}')
        return envuelta

    def escribir(self, nombre, valores):
        if nombre not in self.archivos:
            f = open(os.path.join(self.dir, nombre + '.csv'), 'w', buffering=1)
            f.write(self.cabeceras[nombre] + '\n')
            self.archivos[nombre] = f
        texto = ','.join(f'{v:.4f}' if isinstance(v, float) else str(v) for v in valores)
        self.archivos[nombre].write(f'{time.monotonic() - self.t0:.2f},{texto}\n')

    def al_pose(self, msg):
        p = msg.pose.pose
        c = msg.pose.covariance
        yaw = 2.0 * math.atan2(p.orientation.z, p.orientation.w)
        t_ros = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        self.escribir('pose', [t_ros, p.position.x, p.position.y, yaw,
                               math.sqrt(max(c[0], c[7], 0.0)), math.sqrt(max(c[35], 0.0))])

    def al_cmd(self, msg):
        self.ultimo_cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmds += 1

    def al_odom(self, msg):
        ahora = time.monotonic()
        if ahora - self.t_odom < 0.2:
            return
        self.t_odom = ahora
        t = msg.twist.twist
        self.escribir('movimiento', [*self.ultimo_cmd, t.linear.x, t.linear.y, t.angular.z])

    def al_scan_rojo(self, msg):
        r = np.asarray(msg.ranges)
        self.rayos_rojos = int(np.isfinite(r).sum())

    def al_scan(self, msg):
        if self.laser is None:
            tf = self.tf_buffer.lookup_transform('base_footprint', msg.header.frame_id, Time())
            q = tf.transform.rotation
            self.laser = (tf.transform.translation.x, tf.transform.translation.y,
                          2.0 * math.atan2(q.z, q.w))
        lx, ly, lyaw = self.laser
        r = np.asarray(msg.ranges)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment + lyaw
        ok = np.isfinite(r) & (r > msg.range_min)
        x = lx + r[ok] * np.cos(a[ok])
        y = ly + r[ok] * np.sin(a[ok])
        x0, x1, y0, y1 = self.chasis
        afuera = ~((x > x0) & (x < x1) & (y > y0) & (y < y1))   # rebotes en el propio robot
        if not afuera.any():
            return
        dx = np.maximum(np.maximum(x0 - x[afuera], 0.0), x[afuera] - x1)
        dy = np.maximum(np.maximum(y0 - y[afuera], 0.0), y[afuera] - y1)
        self.escribir('roces', [float(np.hypot(dx, dy).min()), self.rayos_rojos])

    def al_imagen(self, msg):
        ahora = time.monotonic()
        guardar = ahora - self.t_imagen >= self.periodo_imagen
        medir = ahora - self.t_color >= self.periodo_color
        if not (guardar or medir):
            return
        imagen = self.puente.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        chica = cv2.resize(imagen, (320, 240))
        if guardar:
            self.t_imagen = ahora
            cv2.imwrite(os.path.join(self.dir, 'img', f'{ahora - self.t0:07.1f}.jpg'), chica,
                        [cv2.IMWRITE_JPEG_QUALITY, 70])
        if medir:
            self.t_color = ahora
            hsv = cv2.cvtColor(chica, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
            h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
            # Rojizos con un criterio más amplio que el del detector (hue 0-10 / 170-180,
            # S >= 120, V >= 80), para ver dónde caen las cajas con la luz de la arena.
            # Hue 15 ya no: el piso de madera cae ahí.
            rojizos = ((h <= 12) | (h >= 165)) & (s >= 60) & (v >= 40)
            n = int(rojizos.sum())
            if n < 30:
                self.escribir('rojo', [n] + [''] * 9)
                return
            hh = np.where(h[rojizos] >= 160, h[rojizos] - 180, h[rojizos])   # rojo continuo
            fila = [n]
            for canal in (hh, s[rojizos], v[rojizos]):
                fila += [float(np.percentile(canal, q)) for q in (5, 50, 95)]
            self.escribir('rojo', fila)

    def al_cpu(self):
        hz = self.cmds / 5.0
        self.cmds = 0
        self.escribir('cpu', [os.getloadavg()[0], hz])

    def cerrar(self):
        for f in getattr(self, 'archivos', {}).values():
            f.close()


def main(args=None):
    rclpy.init(args=args)
    nodo = Registro()
    try:
        if nodo.activo:
            rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.cerrar()
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
