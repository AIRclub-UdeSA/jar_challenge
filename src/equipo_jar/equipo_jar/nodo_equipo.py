"""Nodo principal: recorre el laberinto siguiendo la pared y reporta las víctimas.

- Navegación: sigue la pared de la derecha con el lidar (ejemplo 1, workshop de la semana 03).
- Víctimas: NO mira la cámara. Las cajas las encuentra nodo_percepcion y las publica, ya
  confirmadas y en el marco `map`, en el topic interno /equipo/cajas_confirmadas. Este nodo es el
  único que publica en /victimas (§7.2): reporta cada caja nueva y, al llegar a N, publica /done y
  deja el robot quieto para siempre (§5).
- Pose: TF map -> base_footprint (AMCL). Si AMCL no se localiza en `espera_localizacion_s`, usa
  /odom como respaldo para no quedarse quieto toda la corrida.

Límite conocido: sigue una pared, no recorre el laberinto de forma sistemática y puede dar vueltas
en lugares abiertos (workshop de la semana 08).
"""
import math

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformException, TransformListener

from equipo_jar.deteccion import yaw_de_cuaternion
from equipo_jar.evasion import distancia_en_sector, giro_para_seguir_pared, sentido_de_giro
from equipo_jar.reportes import armar_reporte

# /map lo publica el juez con esta QoS (docs/QOS.md).
QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

# Los estados de la máquina. En cada instante el robot está en uno solo.
AVANZAR = 'avanzar'      # sigue la pared de la derecha (y busca cajas rojas)
GIRAR = 'girar'          # frente bloqueado: gira en el lugar hasta despejarlo
TERMINADO = 'terminado'  # ya reportó las N víctimas y publicó /done: quieto


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
        self.declare_parameter('frame_mapa', 'map')
        self.declare_parameter('frame_robot', 'base_footprint')
        self.declare_parameter('espera_localizacion_s', 15.0)
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value

        self.mapa = None
        self.scan = None
        self.pose = None       # (x, y, yaw) en el mapa; la actualiza el ciclo de control
        self.pose_odom = None  # (x, y, yaw) según la odometría: respaldo si AMCL no anda
        self.t_mapa = None     # cuándo (s) llegó el mapa: desde ahí corre la espera de AMCL
        self.usando_odom = False
        self.estado = AVANZAR
        self.sentido_giro = 1         # +1 izquierda, -1 derecha
        self.derecha_anterior = None  # distancia a la pared en el ciclo anterior
        self.t_pared_perdida = None   # última vez (s) que la pared de la derecha se terminó
        self.cajas_nuevas = []        # cajas que confirmó nodo_percepcion y falta reportar
        self.reportadas = []          # posiciones de las víctimas ya reportadas

        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        # Sensores (lidar y cámara): best effort. Con la QoS por defecto no llegaría nada.
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)
        # Cajas confirmadas por nodo_percepcion (topic interno, ver nodo_percepcion.py).
        self.sub_cajas = self.create_subscription(
            PointStamped, '/equipo/cajas_confirmadas', self.al_recibir_caja, 10)
        # /odom es reliable: sirve la QoS por defecto.
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.al_recibir_odom, 10)
        # La pose en el mapa se lee de TF: AMCL publica map -> odom y el robot odom -> base.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

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
        if self.t_mapa is None:
            self.t_mapa = self.get_clock().now().nanoseconds * 1e-9
        self.mapa = mapa

    def al_recibir_scan(self, scan):
        self.scan = scan

    def al_recibir_caja(self, msg):
        self.cajas_nuevas.append((msg.point.x, msg.point.y))

    def al_recibir_odom(self, odom):
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        self.pose_odom = (p.x, p.y, yaw_de_cuaternion(q.x, q.y, q.z, q.w))

    def pose_en_el_mapa(self):
        """(x, y, yaw) del robot en el mapa según AMCL, o None si todavía no se localizó."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.get_parameter('frame_mapa').value,
                self.get_parameter('frame_robot').value, Time())
        except TransformException:
            return None
        t, q = tf.transform.translation, tf.transform.rotation
        return t.x, t.y, yaw_de_cuaternion(q.x, q.y, q.z, q.w)

    def actualizar_pose(self):
        """La pose de AMCL; si no llega en `espera_localizacion_s` desde el mapa, la odometría."""
        pose = self.pose_en_el_mapa()
        if pose is not None:
            if self.usando_odom:
                self.get_logger().info('AMCL localizado: dejo de usar la odometría')
                self.usando_odom = False
            self.pose = pose
            return
        ahora = self.get_clock().now().nanoseconds * 1e-9
        if self.usando_odom or ahora - self.t_mapa > self.get_parameter(
                'espera_localizacion_s').value:
            if not self.usando_odom:
                self.get_logger().warn('AMCL no se localizó a tiempo: uso la odometría como mapa')
                self.usando_odom = True
            self.pose = self.pose_odom

    # --- Control: se ejecuta a frecuencia_control_hz ----------------------------------------

    def ciclo_de_control(self):
        if self.mapa is None:
            return  # esperar el mapa: no mover el robot ni reportar antes (§5)
        self.actualizar_pose()
        self.reportar_cajas_nuevas()
        if self.estado == TERMINADO or self.scan is None or self.pose is None:
            self.pub_cmd_vel.publish(Twist())  # después de /done el robot no se vuelve a mover
            return

        frente = self.distancia(0.0, self.get_parameter('ancho_frontal_deg').value)
        derecha = self.distancia(-90.0, self.get_parameter('ancho_lateral_deg').value)
        if derecha > self.get_parameter('pared_perdida_m').value:
            self.t_pared_perdida = self.get_clock().now().nanoseconds * 1e-9
        self.actualizar_estado(frente, derecha)
        self.pub_cmd_vel.publish(
            Twist() if self.estado == TERMINADO else self.comando(derecha))
        self.derecha_anterior = derecha

    def distancia(self, centro_deg, ancho_deg):
        """Distancia más corta que ve el lidar en un sector (0 = frente, positivo = izquierda)."""
        return distancia_en_sector(
            self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
            frente_deg=self.get_parameter('angulo_frente_deg').value,
            centro_deg=centro_deg, ancho_deg=ancho_deg,
            rango_min=self.scan.range_min, rango_max=self.scan.range_max)

    def actualizar_estado(self, frente, derecha):
        """Transiciones entre AVANZAR y GIRAR."""
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

    # --- Víctimas ---------------------------------------------------------------------------

    def reportar_cajas_nuevas(self):
        """Reporta las cajas que confirmó nodo_percepcion, hasta llegar a N."""
        while self.cajas_nuevas and self.estado != TERMINADO:
            x, y = self.cajas_nuevas.pop(0)
            self.reportadas.append((x, y))
            self.reportar_victima(x, y)
            self.get_logger().info(
                f'Víctima {len(self.reportadas)}/{self.cantidad_victimas} reportada en '
                f'({x:.2f}, {y:.2f})')
            if len(self.reportadas) >= self.cantidad_victimas:
                self.estado = TERMINADO
                self.terminar()

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
