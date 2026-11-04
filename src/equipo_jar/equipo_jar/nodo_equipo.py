"""Nodo del equipo Donatello para el Challenge JAR 2026.

Integra en un solo nodo lo que el equipo desarrolló en donatello-jar:
- Navegación reactiva con el LiDAR (control_obstaculos.py / elegir_direccion.py),
  convertida en un seguidor de pared derecha para recorrer el laberinto.
- Detección de rojo en la cámara (detector_rojo.py), ahora con estimación de
  posición: el rumbo sale de la cámara y la distancia del LiDAR en ese rumbo.
- Reporte en /victimas con `armar_reporte`, confirmando cada víctima en varios
  frames y sin repetir reportes.

Reglas que respeta:
- No mueve el robot ni reporta nada hasta recibir /map (§5).
- Solo usa topics oficiales: /map, /scan, /odom, /cam_1/color/*, /cmd_vel,
  /victimas, /done. Sin ground truth.
- Localización: el robot larga cerca del origen mirando a +x, así que se usa
  /odom como aproximación del marco `map`. Nada de coordenadas del mapa.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty

from equipo_jar import logica
from equipo_jar.reportes import armar_reporte

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class NodoEquipo(Node):
    def __init__(self):
        super().__init__('nodo_equipo')
        p = self.declare_parameter
        p('frecuencia_control_hz', 10.0)
        p('velocidad_lineal_max', 0.20)
        p('velocidad_angular_max', 0.80)
        p('cantidad_victimas', 5)
        # LiDAR
        p('offset_angulo_lidar', math.pi)   # montaje del LiDAR en el Rosmaster
        p('distancia_seguridad', 0.30)
        p('distancia_liberacion', 0.45)
        p('distancia_pared', 0.30)
        p('ganancia_pared', 2.0)
        # Atascos
        p('tiempo_atasco_s', 8.0)
        p('distancia_atasco', 0.05)
        p('tiempo_escape_s', 2.0)
        # Visión
        p('hz_vision', 5.0)
        p('fov_horizontal_deg', 69.4)  # solo si no llega camera_info
        p('rojo_s_min', 100)
        p('rojo_v_min', 80)
        p('rojo_h_bajo_max', 10)
        p('rojo_h_alto_min', 170)
        p('area_min_px', 400.0)
        p('distancia_max_deteccion', 2.0)
        p('ventana_rumbo_rad', 0.05)
        p('profundidad_caja', 0.05)
        p('giro_max_para_detectar', 0.4)
        p('radio_agrupado', 0.35)
        p('observaciones_min', 5)
        # Fin de corrida
        p('tiempo_max_s', 540.0)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.cantidad_victimas = int(g('cantidad_victimas'))
        self.agrupador = logica.Agrupador(g('radio_agrupado'), int(g('observaciones_min')))

        self.mapa = None
        self.t_inicio = None
        self.terminado = False
        self.lecturas = None
        self.pose = None            # (x, y, yaw)
        self.fx = self.cx = None
        self.estado = 'SEGUIR'
        self.ultimo_cmd_w = 0.0
        self.ref_atasco = None      # (t, x, y)
        self.fin_escape = None
        self.ultima_vision = None

        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_MAPA)
        self.create_subscription(LaserScan, '/scan', self.al_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.al_odom, 10)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.al_info,
                                 qos_profile_sensor_data)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.al_imagen,
                                 qos_profile_sensor_data)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)

        self.create_timer(1.0 / g('frecuencia_control_hz'), self.ciclo_de_control)
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    # ------------------------------------------------------------------ entradas
    def ahora(self):
        return self.get_clock().now().nanoseconds / 1e9

    def al_mapa(self, mapa):
        if self.mapa is None:
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas. Arranco.')
            self.t_inicio = self.ahora()
        self.mapa = mapa

    def al_scan(self, msg):
        self.lecturas = logica.lecturas_validas(
            msg.ranges, msg.angle_min, msg.angle_increment,
            self.g('offset_angulo_lidar'), msg.range_min, msg.range_max)

    def al_odom(self, msg):
        p = msg.pose.pose
        self.pose = (p.position.x, p.position.y, logica.yaw_de_cuaternion(p.orientation))

    def al_info(self, msg):
        if msg.k[0] > 0:
            self.fx, self.cx = msg.k[0], msg.k[2]

    def al_imagen(self, msg):
        if self.mapa is None or self.terminado:
            return
        if self.lecturas is None or self.pose is None:
            return
        t = self.ahora()
        if self.ultima_vision is not None and t - self.ultima_vision < 1.0 / self.g('hz_vision'):
            return
        self.ultima_vision = t
        if abs(self.ultimo_cmd_w) > self.g('giro_max_para_detectar'):
            return  # girando rápido: imagen y scan no coinciden bien

        imagen = self.a_rgb(msg)
        if imagen is None:
            return
        fov = math.radians(self.g('fov_horizontal_deg'))
        fx = self.fx or msg.width / (2.0 * math.tan(fov / 2.0))
        cx = self.cx if self.cx is not None else msg.width / 2.0

        mascara = logica.mascara_roja(
            imagen, int(self.g('rojo_s_min')), int(self.g('rojo_v_min')),
            int(self.g('rojo_h_bajo_max')), int(self.g('rojo_h_alto_min')))
        x, y, yaw = self.pose
        for u, _area in logica.blobs(mascara, self.g('area_min_px')):
            rumbo = logica.rumbo_de_pixel(u, cx, fx)
            d = logica.distancia_en_rumbo(self.lecturas, rumbo, self.g('ventana_rumbo_rad'))
            if d is None or d > self.g('distancia_max_deteccion'):
                continue
            px, py = logica.proyectar(x, y, yaw, d + self.g('profundidad_caja'), rumbo)
            self.agrupador.agregar(px, py)
        self.reportar_confirmadas()

    @staticmethod
    def a_rgb(msg):
        if msg.encoding not in ('rgb8', 'bgr8'):
            return None
        img = np.frombuffer(msg.data, dtype=np.uint8)
        if img.size < msg.height * msg.step:
            return None
        img = img[:msg.height * msg.step].reshape((msg.height, msg.step))
        img = img[:, :msg.width * 3].reshape((msg.height, msg.width, 3))
        return img if msg.encoding == 'rgb8' else img[:, :, ::-1].copy()

    # ----------------------------------------------------------------- reportes
    def reportar_confirmadas(self):
        for grupo, x, y in self.agrupador.listos():
            if self.agrupador.cantidad_reportados() >= self.cantidad_victimas:
                return
            try:
                self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))
            except ValueError as e:
                self.get_logger().warn(f'Reporte descartado: {e}')
                grupo['reportado'] = True
                continue
            grupo['reportado'] = True
            self.get_logger().info(
                f'Víctima reportada en ({x:.2f}, {y:.2f}) '
                f'[{self.agrupador.cantidad_reportados()}/{self.cantidad_victimas}]')

    # ------------------------------------------------------------------ control
    def publicar(self, v, w):
        vmax, wmax = self.g('velocidad_lineal_max'), self.g('velocidad_angular_max')
        cmd = Twist()
        cmd.linear.x = float(max(-vmax, min(vmax, v)))
        cmd.angular.z = float(max(-wmax, min(wmax, w)))
        self.ultimo_cmd_w = cmd.angular.z
        self.pub_cmd_vel.publish(cmd)

    def terminar(self, motivo):
        self.get_logger().info(f'Fin de corrida: {motivo}. Publico /done.')
        self.terminado = True
        self.publicar(0.0, 0.0)
        self.pub_done.publish(Empty())

    def ciclo_de_control(self):
        if self.mapa is None:
            return  # esperar el mapa: no mover el robot ni reportar antes
        if self.terminado:
            self.publicar(0.0, 0.0)
            return
        t = self.ahora()
        if self.agrupador.cantidad_reportados() >= self.cantidad_victimas:
            self.terminar('todas las víctimas reportadas')
            return
        if t - self.t_inicio > self.g('tiempo_max_s'):
            self.terminar('tiempo máximo')
            return
        if self.lecturas is None:
            self.publicar(0.0, 0.0)
            return

        v, w = self.navegar(t)
        self.publicar(v, w)

    def navegar(self, t):
        """Seguidor de pared derecha con frenado y escape de atascos."""
        L = self.lecturas
        rad = math.radians
        frente = logica.minimo_en_sector(L, rad(-20), rad(20))
        diag_der = logica.minimo_en_sector(L, rad(-60), rad(-25))
        derecha = logica.minimo_en_sector(L, rad(-100), rad(-60))
        izquierda = logica.minimo_en_sector(L, rad(60), rad(100))
        vmax, wmax = self.g('velocidad_lineal_max'), self.g('velocidad_angular_max')
        d_seg, d_lib = self.g('distancia_seguridad'), self.g('distancia_liberacion')
        d_pared = self.g('distancia_pared')

        # Escape de atasco en curso: retroceder girando hacia el lado libre.
        if self.fin_escape is not None:
            if t < self.fin_escape:
                lado = 1.0 if izquierda >= derecha else -1.0
                return -0.5 * vmax, 0.8 * wmax * lado
            self.fin_escape = None
            self.ref_atasco = None

        # Detección de atasco: casi sin moverse durante un rato.
        if self.pose is not None:
            x, y, _ = self.pose
            if self.ref_atasco is None:
                self.ref_atasco = (t, x, y)
            elif math.hypot(x - self.ref_atasco[1], y - self.ref_atasco[2]) > \
                    self.g('distancia_atasco'):
                self.ref_atasco = (t, x, y)
            elif t - self.ref_atasco[0] > self.g('tiempo_atasco_s'):
                self.get_logger().info('Atascado: maniobra de escape')
                self.fin_escape = t + self.g('tiempo_escape_s')
                return 0.0, 0.0

        # Pared adelante: girar a la izquierda en el lugar hasta liberar.
        if frente < d_seg or (self.estado == 'GIRAR' and frente < d_lib):
            self.estado = 'GIRAR'
            return 0.0, 0.7 * wmax
        self.estado = 'SEGUIR'

        if diag_der < 0.8 * d_seg:
            return 0.4 * vmax, 0.5 * wmax          # muy cerca en diagonal: abrirse
        if derecha > 2.5 * d_pared and diag_der > 2.5 * d_pared:
            return 0.5 * vmax, -0.6 * wmax         # abertura a la derecha: doblar
        error = derecha - d_pared                  # >0: lejos de la pared
        w = -self.g('ganancia_pared') * error
        v = vmax * (0.6 if frente < 2 * d_lib else 1.0)
        return v, w


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
