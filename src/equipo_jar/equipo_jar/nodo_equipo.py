"""Ejemplo 1: recorrer el laberinto siguiendo la pared y esquivando lo que aparece de frente.

Ejemplo de qué se puede entregar: es lo más simple que anda. El robot espera el mapa y después
avanza con la pared de su derecha a una distancia fija. Si aparece algo adelante, gira en el lugar
hasta tener el frente libre (a la derecha si por ahí hay lugar, a la izquierda si hay pared); si la
pared de la derecha se termina, curva hacia ella. No busca víctimas ni reporta nada (puntaje 0 a
propósito: el siguiente ejemplo suma la detección).

Por qué seguir una pared: un robot que solo esquiva lo de adelante se queda dando vueltas en la
zona de largada. Seguir siempre la misma pared obliga a recorrer el laberinto.

Límites que conviene conocer: no tiene idea de dónde está ni de qué ya recorrió. En un lugar abierto
da vueltas, y en un recoveco angosto o con algo en la salida (una caja, por ejemplo) puede quedar
yendo y viniendo. Mejorar eso es lo que enseñan los workshops de localización (semana 06) y de
exploración del mapa (semana 08).

Qué cambia respecto del template:
- Este nodo: dos estados (AVANZAR y GIRAR) en vez de quedarse quieto.
- `evasion.py`: las cuentas con el lidar (qué ve en un sector, cuánto girar para seguir la pared),
  sin ROS, con sus tests en test/test_evasion.py.
- `params/params.yaml`: las distancias y las velocidades.

Lo que NO cambia y no hay que tocar: la espera del mapa, el canal de /victimas, /done y /cmd_vel
(que ya vienen armados) y el launch.

De dónde sale cada idea (workshops del Challenge, https://airclub-udesa.github.io/jar_site/workshops/):
- semana 01, timers y callbacks: el timer que corre `ciclo_de_control`;
- semana 02, `Twist` y /cmd_vel: cómo se manda una velocidad al robot;
- semana 03, evasión de obstáculos: la máquina de estados y el sector frontal del lidar.
"""
import math

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty

from equipo_jar.evasion import distancia_en_sector, giro_para_seguir_pared, sentido_de_giro
from equipo_jar.reportes import armar_reporte

# /map lo publica el juez con esta QoS (docs/QOS.md).
QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Los dos estados de la máquina. En cada instante el robot está en uno solo.
AVANZAR = 'avanzar'  # sigue la pared de la derecha
GIRAR = 'girar'      # frente bloqueado: gira en el lugar hasta despejarlo


class NodoEquipo(Node):
    def __init__(self):
        super().__init__('nodo_equipo')
        # Los números no van escritos en el código: viven en params/params.yaml, que es lo único
        # que se puede ajustar después del cierre (§9.1).
        self.declare_parameter('frecuencia_control_hz', 10.0)
        self.declare_parameter('velocidad_lineal_max', 0.30)
        self.declare_parameter('velocidad_angular_max', 1.00)
        self.declare_parameter('angulo_frente_deg', 180.0)
        self.declare_parameter('ancho_frontal_deg', 40.0)
        self.declare_parameter('ancho_lateral_deg', 20.0)
        self.declare_parameter('distancia_choque_m', 0.45)
        self.declare_parameter('despeje_extra_m', 0.15)
        self.declare_parameter('distancia_pared_m', 0.45)
        self.declare_parameter('pared_perdida_m', 1.20)
        self.declare_parameter('memoria_pared_s', 2.0)
        self.declare_parameter('ganancia_pared', 2.0)
        self.declare_parameter('amortiguacion_pared', 4.0)
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        # Este ejemplo no busca, pero el parámetro tiene que seguir declarado.
        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value

        self.mapa = None
        self.scan = None
        self.estado = AVANZAR
        self.sentido_giro = 1  # +1 izquierda, -1 derecha
        self.derecha_anterior = None  # distancia a la pared en el ciclo anterior
        self.t_pared_perdida = None   # última vez (s) que la pared de la derecha se terminó

        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        # /scan es de un sensor: best effort. Con la QoS por defecto no llegaría ni un mensaje.
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)

        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)

        periodo = 1.0 / self.get_parameter('frecuencia_control_hz').value
        self.create_timer(periodo, self.ciclo_de_control)
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    # --- Sensores: los callbacks solo guardan el último dato; decide el timer ---------------

    def al_recibir_mapa(self, mapa):
        if self.mapa is None:
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m')
        self.mapa = mapa

    def al_recibir_scan(self, scan):
        self.scan = scan

    # --- Control: se ejecuta a frecuencia_control_hz ----------------------------------------

    def ciclo_de_control(self):
        if self.mapa is None:
            return  # esperar el mapa: no mover el robot ni reportar antes (§5)
        if self.scan is None:
            self.pub_cmd_vel.publish(Twist())  # todavía sin lidar: quieto
            return

        frente = self.distancia(0.0, self.get_parameter('ancho_frontal_deg').value)
        derecha = self.distancia(-90.0, self.get_parameter('ancho_lateral_deg').value)
        if derecha > self.get_parameter('pared_perdida_m').value:
            self.t_pared_perdida = self.get_clock().now().nanoseconds * 1e-9
        self.actualizar_estado(frente, derecha)
        self.pub_cmd_vel.publish(self.comando(derecha))
        self.derecha_anterior = derecha

    def distancia(self, centro_deg, ancho_deg):
        """Distancia más corta que ve el lidar en un sector (0 = frente, positivo = izquierda)."""
        return distancia_en_sector(
            self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
            frente_deg=self.get_parameter('angulo_frente_deg').value,
            centro_deg=centro_deg, ancho_deg=ancho_deg,
            rango_min=self.scan.range_min, rango_max=self.scan.range_max)

    def actualizar_estado(self, frente, derecha):
        """Transiciones: ¿hay que cambiar de estado?"""
        choque = self.get_parameter('distancia_choque_m').value
        if self.estado == AVANZAR and frente < choque:
            self.estado = GIRAR
            self.sentido_giro = sentido_de_giro(self.hay_lugar_a_la_derecha())
            lado = 'izquierda' if self.sentido_giro > 0 else 'derecha'
            self.get_logger().info(f'Obstáculo a {frente:.2f} m: giro a la {lado}')
        elif self.estado == GIRAR and frente >= choque + self.get_parameter(
                'despeje_extra_m').value:
            # Se sigue girando hasta tener un poco más de lugar que el mínimo: así no se
            # alterna entre avanzar y girar justo en el borde de la distancia de choque.
            self.estado = AVANZAR
            self.get_logger().info('Frente libre: avanzo')

    def hay_lugar_a_la_derecha(self):
        """¿La pared de la derecha se terminó hace poco? Al curvar hacia ella el sector derecho
        ya apunta a otro lado cuando aparece la pared de adelante: por eso se recuerda."""
        if self.t_pared_perdida is None:
            return False
        ahora = self.get_clock().now().nanoseconds * 1e-9
        return ahora - self.t_pared_perdida < self.get_parameter('memoria_pared_s').value

    def comando(self, derecha):
        """Acción: qué velocidad mandar según el estado."""
        cmd = Twist()
        giro_max = self.get_parameter('velocidad_angular_max').value
        if self.estado == GIRAR:
            cmd.angular.z = self.sentido_giro * giro_max
            return cmd
        cmd.linear.x = self.get_parameter('velocidad_lineal_max').value
        cmd.angular.z = giro_para_seguir_pared(
            derecha, self.velocidad_distancia(derecha),
            self.get_parameter('distancia_pared_m').value,
            self.get_parameter('ganancia_pared').value,
            self.get_parameter('amortiguacion_pared').value, giro_max,
            self.get_parameter('pared_perdida_m').value)
        return cmd

    def velocidad_distancia(self, derecha):
        """Cuánto cambió la distancia a la pared por segundo (m/s) desde el ciclo anterior."""
        if self.derecha_anterior is None or not (
                math.isfinite(derecha) and math.isfinite(self.derecha_anterior)):
            return 0.0
        return (derecha - self.derecha_anterior) * self.get_parameter('frecuencia_control_hz').value

    # --- Listos para usar en los próximos pasos (el ejemplo 2 los usa) -----------------------

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
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
