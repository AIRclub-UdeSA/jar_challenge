"""Nodo supervisor de movimiento (ADR-0007): el único que publica /cmd_vel.

- Publica a ritmo fijo desde un hilo propio (ritmo con reloj monotónico), siempre: cero
  antes de /map, después de /done y al vencer la duración de la corrida.
- Entradas internas, de mayor a menor prioridad: /jar/cmd_vel_mision (maniobras
  de la misión) y /jar/cmd_vel_nav (Nav2).
- Frena la traslación si el LiDAR ve algo cerca en la dirección de avance.
"""
import json
import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import Point
from std_msgs.msg import Empty, String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from jar_supervisor.logica import (Config, Estado, Supervisor, matriz_de_cuaternion,
                                   scan_a_puntos)

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)
FUENTES = ('mision', 'nav')


class NodoSupervisor(Node):
    def __init__(self):
        super().__init__('supervisor')
        config = Config()
        for nombre, valor in vars(config).items():
            setattr(config, nombre, self.declare_parameter(nombre, valor).value)
        self.frecuencia_hz = self.declare_parameter('frecuencia_hz', 20.0).value
        self.duracion_corrida_s = self.declare_parameter('duracion_corrida_s', 600.0).value
        self.marco_robot = self.declare_parameter('marco_robot', 'base_footprint').value

        self.logica = Supervisor(config, FUENTES)
        self.lock = threading.Lock()
        self.t_mapa = None
        self.tf_laser = None
        self.patinando = False
        self.r_imu = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_estado = self.create_publisher(String, '/jar/supervisor/estado', 10)
        self.pub_marcas = self.create_publisher(MarkerArray, '/jar/supervisor/marcadores', 10)
        self.create_subscription(OccupancyGrid, '/map', self.al_mapa, QOS_MAPA)
        self.create_subscription(Empty, '/done', self.al_done, 10)
        self.create_subscription(LaserScan, '/scan', self.al_scan, qos_profile_sensor_data)
        # Giro de las ruedas contra el de la IMU: para detectar patinamiento (§7.1, ambos).
        self.create_subscription(Odometry, '/odom', self.al_odom, 10)
        self.create_subscription(Imu, '/imu/data', self.al_imu, 10)
        self.create_subscription(Twist, '/jar/cmd_vel_mision',
                                 lambda m: self.al_pedido('mision', m), 10)
        self.create_subscription(Twist, '/jar/cmd_vel_nav', lambda m: self.al_pedido('nav', m), 10)

        self.corriendo = True
        self.hilo = threading.Thread(target=self.bucle, daemon=True)
        self.hilo.start()
        self.get_logger().info('Supervisor listo: publica cero en /cmd_vel hasta recibir /map')

    def al_mapa(self, _msg):
        with self.lock:
            if self.t_mapa is None:
                self.t_mapa = self.get_clock().now()
                self.logica.mapa_recibido()
                self.get_logger().info('Mapa recibido: supervisor activo')

    def al_done(self, _msg):
        with self.lock:
            if self.logica.estado != Estado.DETENIDO:
                self.logica.detener()
                self.get_logger().info('/done: robot detenido para siempre')

    def al_pedido(self, fuente, msg):
        with self.lock:
            self.logica.pedir(fuente, msg.linear.x, msg.linear.y, msg.angular.z, self.ahora())

    def al_odom(self, msg):
        with self.lock:
            self.logica.actualizar_giros(giro_odom=msg.twist.twist.angular.z)

    def al_imu(self, msg):
        # La IMU puede estar montada girada (en el simulador, dada vuelta: rpy = 0, pi, pi/2):
        # su velocidad angular se pasa al marco del robot con la TF del montaje.
        if self.r_imu is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.marco_robot, msg.header.frame_id,
                                                     rclpy.time.Time())
            except Exception:  # noqa: BLE001 - TF todavía no disponible
                return
            q = tf.transform.rotation
            self.r_imu = matriz_de_cuaternion(q.x, q.y, q.z, q.w)
        w = msg.angular_velocity
        giro = float((self.r_imu @ np.array([w.x, w.y, w.z]))[2])
        with self.lock:
            self.logica.actualizar_giros(giro_imu=giro)

    def al_scan(self, msg):
        if self.tf_laser is None:
            try:
                tf = self.tf_buffer.lookup_transform(self.marco_robot, msg.header.frame_id,
                                                     rclpy.time.Time())
            except Exception:  # noqa: BLE001 - TF todavía no disponible
                return
            q = tf.transform.rotation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            self.tf_laser = (tf.transform.translation.x, tf.transform.translation.y, yaw)
        puntos = scan_a_puntos(msg.ranges, msg.angle_min, msg.angle_increment,
                               msg.range_min, msg.range_max, *self.tf_laser)
        with self.lock:
            self.logica.actualizar_obstaculos(puntos)

    def ahora(self):
        """Segundos del reloj de ROS: simulado en el simulador, de pared en el robot.

        El vencimiento de los pedidos se mide con este reloj porque Nav2 publica en
        tiempo de ROS: con el simulador lento (RTF < 1), medirlo en reloj de pared
        vencía comandos válidos y el robot avanzaba a los tirones.
        """
        return self.get_clock().now().nanoseconds * 1e-9

    def bucle(self):
        periodo = 1.0 / self.frecuencia_hz
        proximo = time.monotonic()
        contador = 0
        while self.corriendo and rclpy.ok():
            with self.lock:
                if (self.t_mapa is not None and self.logica.estado == Estado.ACTIVO
                        and (self.get_clock().now() - self.t_mapa).nanoseconds * 1e-9
                        >= self.duracion_corrida_s):
                    self.logica.detener()
                    self.get_logger().info('Duración de la corrida cumplida: robot detenido')
                t = self.ahora()
                vx, vy, wz = self.logica.comando(t)
                fuente = self.logica.fuente_activa(t)
                estado = self.logica.estado.value
                diag = dict(self.logica.diag)
            msg = Twist()
            msg.linear.x, msg.linear.y, msg.angular.z = float(vx), float(vy), float(wz)
            self.pub_cmd_vel.publish(msg)
            contador += 1
            if contador % max(1, int(self.frecuencia_hz / 5)) == 0:   # diagnóstico a 5 Hz
                self.publicar_diagnostico(estado, fuente, diag, (vx, vy, wz))
            proximo += periodo
            time.sleep(max(0.0, proximo - time.monotonic()))

    def publicar_diagnostico(self, estado, fuente, diag, salida):
        hueco = diag['hueco']
        self.pub_estado.publish(String(data=json.dumps({
            'estado': estado, 'fuente': fuente,
            'pedido': [round(v, 3) for v in diag['pedido']],
            'salida': [round(v, 3) for v in salida],
            'hueco_m': None if math.isinf(hueco) else round(hueco, 3),
            'escala_freno': round(diag['escala'], 2),
            'giro_frenado': bool(diag.get('giro_frenado')), 'patina': bool(diag.get('patina')),
        })))
        if diag.get('patina') and not self.patinando:
            self.get_logger().warning('patinamiento: las ruedas giran y la IMU no; giros cortados')
        self.patinando = bool(diag.get('patina'))
        # En RViz: flecha verde = lo que pide la fuente activa, roja = lo que sale a /cmd_vel.
        marcas = MarkerArray()
        for i, (v, color) in enumerate(((diag['pedido'], (0.0, 1.0, 0.0)), (salida, (1.0, 0.0, 0.0)))):
            m = Marker()
            m.header.frame_id = self.marco_robot
            m.ns, m.id, m.type, m.action = 'supervisor', i, Marker.ARROW, Marker.ADD
            m.points = [Point(z=0.25), Point(x=float(v[0]) * 3.0, y=float(v[1]) * 3.0, z=0.25)]
            m.scale.x, m.scale.y, m.scale.z = 0.03, 0.06, 0.08
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.9
            marcas.markers.append(m)
        texto = Marker()
        texto.header.frame_id = self.marco_robot
        texto.ns, texto.id, texto.type, texto.action = 'supervisor', 2, Marker.TEXT_VIEW_FACING, Marker.ADD
        texto.pose.position.z = 0.6
        texto.pose.orientation.w = 1.0
        texto.scale.z = 0.12
        texto.color.r = texto.color.g = texto.color.b = texto.color.a = 1.0
        hueco_txt = '-' if math.isinf(hueco) else f'{hueco:.2f} m'
        texto.text = f'{fuente or "nadie"} | freno x{diag["escala"]:.2f} | hueco {hueco_txt}'
        marcas.markers.append(texto)
        self.pub_marcas.publish(marcas)

    def destroy_node(self):
        self.corriendo = False
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoSupervisor()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
