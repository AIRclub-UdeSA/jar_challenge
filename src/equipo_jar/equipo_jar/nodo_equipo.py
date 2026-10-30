"""Ejemplo 2: recorrer el laberinto, detectar cajas rojas y reportarlas.

Parte del ejemplo 1 (el robot sigue la pared con el lidar) y le suma lo que da puntos: mirar con
la cámara, ubicar cada caja roja en el plano y reportarla una sola vez.

Cómo se reporta una víctima, paso a paso:
1. La cámara ve una mancha roja: su columna en la imagen da un ángulo respecto del frente.
2. El lidar, en esa misma dirección, da la distancia a la cara de la caja.
3. Con la pose del robot (/odom) se pasa a coordenadas del plano y se suma medio lado de caja para
   llegar al centro, que es lo que puntúa (§6.1).
4. Se espera a ver lo mismo varias veces seguidas antes de reportar: cada reporte gasta uno (§7.2),
   puntúe o no. Una caja ya reportada no se vuelve a reportar.
5. Al reportar las N víctimas se publica /done y el robot se detiene para siempre (§5).

Qué cambia respecto del ejemplo 1:
- `deteccion.py`: filtro de rojo, ángulo de la caja y posición, sin ROS, con sus tests en
  test/test_deteccion.py.
- Este nodo: `buscar_victima` (detecta, confirma y reporta) y el estado TERMINADO.
- `params/params.yaml`: los umbrales de color, de tamaño y de confirmación.
- `package.xml`: dependencias nuevas (cv_bridge y OpenCV). La imagen Docker las instala sola con
  rosdep: no hace falta tocar el Dockerfile.

Límites que conviene conocer (es lo que mejora el equipo que sigue con los workshops):
- Usa la odometría como si fuera el marco `map`. Anda cuando el robot arranca en el origen, pero la
  odometría deriva y en la competencia el robot arranca cerca del origen, no exactamente en él:
  localizarse en el mapa (workshop de la semana 06) es lo que corrige eso.
- Sigue una pared: no recorre el laberinto de forma sistemática y puede dar vueltas en lugares
  abiertos o quedar yendo y viniendo en un recoveco (workshop de la semana 08).
- Solo busca mientras avanza: al girar, la imagen y la odometría dejan de ser del mismo instante.

De dónde sale cada idea (workshops del Challenge, https://airclub-udesa.github.io/jar_site/workshops/):
semanas 01 a 03, el timer, /cmd_vel y la máquina de estados; semana 04, detección de color con
cámara y lidar.
"""
import math

import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image, LaserScan
from std_msgs.msg import Empty

from equipo_jar.deteccion import (
    angulo_de_columna, buscar_caja_roja, cerca_de_alguno, coinciden, distancia_a_la_caja,
    distancia_por_altura, promedio, punto_en_el_plano, yaw_de_cuaternion)
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
        self.declare_parameter('saturacion_min', 120)
        self.declare_parameter('valor_min', 80)
        self.declare_parameter('area_minima_px', 400.0)
        self.declare_parameter('margen_borde_px', 5)
        self.declare_parameter('fraccion_central', 0.6)
        self.declare_parameter('distancia_min_deteccion_m', 0.35)
        self.declare_parameter('distancia_max_deteccion_m', 3.0)
        self.declare_parameter('semilado_caja_m', 0.125)
        self.declare_parameter('alto_caja_m', 0.30)
        self.declare_parameter('tolerancia_distancia', 0.25)
        self.declare_parameter('confirmaciones', 5)
        self.declare_parameter('dispersion_max_m', 0.25)
        self.declare_parameter('radio_duplicado_m', 0.60)
        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value

        self.puente = CvBridge()
        self.mapa = None
        self.scan = None
        self.imagen = None
        self.info_camara = None
        self.pose = None  # (x, y, yaw) según la odometría
        self.estado = AVANZAR
        self.sentido_giro = 1         # +1 izquierda, -1 derecha
        self.derecha_anterior = None  # distancia a la pared en el ciclo anterior
        self.t_pared_perdida = None   # última vez (s) que la pared de la derecha se terminó
        self.observaciones = []       # posiciones seguidas de la caja que se está viendo
        self.reportadas = []          # posiciones de las víctimas ya reportadas

        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        # Sensores (lidar y cámara): best effort. Con la QoS por defecto no llegaría nada.
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)
        self.sub_imagen = self.create_subscription(
            Image, '/cam_1/color/image_raw', self.al_recibir_imagen, qos_profile_sensor_data)
        self.sub_info = self.create_subscription(
            CameraInfo, '/cam_1/color/camera_info', self.al_recibir_info,
            qos_profile_sensor_data)
        # /odom es reliable: sirve la QoS por defecto.
        self.sub_odom = self.create_subscription(Odometry, '/odom', self.al_recibir_odom, 10)

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

    def al_recibir_imagen(self, msg):
        self.imagen = self.puente.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def al_recibir_info(self, info):
        self.info_camara = info

    def al_recibir_odom(self, odom):
        p, q = odom.pose.pose.position, odom.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_de_cuaternion(q.x, q.y, q.z, q.w))

    # --- Control: se ejecuta a frecuencia_control_hz ----------------------------------------

    def ciclo_de_control(self):
        if self.mapa is None:
            return  # esperar el mapa: no mover el robot ni reportar antes (§5)
        if self.estado == TERMINADO or self.scan is None or self.pose is None:
            self.pub_cmd_vel.publish(Twist())  # después de /done el robot no se vuelve a mover
            return

        frente = self.distancia(0.0, self.get_parameter('ancho_frontal_deg').value)
        derecha = self.distancia(-90.0, self.get_parameter('ancho_lateral_deg').value)
        if derecha > self.get_parameter('pared_perdida_m').value:
            self.t_pared_perdida = self.get_clock().now().nanoseconds * 1e-9
        self.actualizar_estado(frente, derecha)
        if self.estado == AVANZAR:
            self.buscar_victima()
        else:
            self.observaciones.clear()  # al girar la imagen y la odometría no coinciden
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

    def buscar_victima(self):
        """Mira la cámara; si hay una caja roja nueva y se la ve estable, la reporta."""
        punto = self.ubicar_caja_roja()
        if punto is None or cerca_de_alguno(
                punto, self.reportadas, self.get_parameter('radio_duplicado_m').value):
            self.observaciones.clear()
            return

        # Una sola vista puede estar equivocada (ruido, una pared detrás de la caja): se
        # confirma con varias seguidas que coincidan.
        if self.observaciones and not cerca_de_alguno(
                punto, self.observaciones, self.get_parameter('dispersion_max_m').value):
            self.observaciones.clear()
        self.observaciones.append(punto)
        if len(self.observaciones) < self.get_parameter('confirmaciones').value:
            return

        x, y = promedio(self.observaciones)
        self.observaciones.clear()
        self.reportadas.append((x, y))
        self.reportar_victima(x, y)
        self.get_logger().info(
            f'Víctima {len(self.reportadas)}/{self.cantidad_victimas} reportada en '
            f'({x:.2f}, {y:.2f})')
        if len(self.reportadas) >= self.cantidad_victimas:
            self.estado = TERMINADO
            self.terminar()

    def ubicar_caja_roja(self):
        """Posición (x, y) de la caja roja que se ve ahora, o None si no hay una confiable."""
        if self.imagen is None or self.info_camara is None:
            return None
        caja = buscar_caja_roja(
            self.imagen, self.get_parameter('saturacion_min').value,
            self.get_parameter('valor_min').value, self.get_parameter('area_minima_px').value,
            self.get_parameter('margen_borde_px').value)
        if caja is None:
            return None
        columna, ancho_px, alto_px = caja

        # 1. Ángulo de la caja respecto del frente (la cámara mira al frente del robot).
        fx, cx = self.info_camara.k[0], self.info_camara.k[2]
        angulo = angulo_de_columna(columna, fx, cx)

        # 2. Distancia que ve el lidar en esa dirección: la mediana de los rayos del tramo
        #    central de la caja (el tamaño del tramo sale de cuántos píxeles ocupa en la imagen).
        ancho_deg = max(3.0, math.degrees(ancho_px / fx)
                        * self.get_parameter('fraccion_central').value)
        distancia = distancia_a_la_caja(
            self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
            self.get_parameter('angulo_frente_deg').value, math.degrees(angulo), ancho_deg,
            self.scan.range_min, self.scan.range_max)
        if distancia is None or not (
                self.get_parameter('distancia_min_deteccion_m').value <= distancia
                <= self.get_parameter('distancia_max_deteccion_m').value):
            return None

        # 3. Contraste: el tamaño de la caja en la imagen da otra estimación de la distancia. Si el
        #    lidar mide otra cosa (por ejemplo, el borde de una pared que tapa la caja), se descarta.
        por_altura = distancia_por_altura(
            alto_px, self.info_camara.k[4], self.get_parameter('alto_caja_m').value)
        if not coinciden(distancia, por_altura, self.get_parameter('tolerancia_distancia').value):
            return None

        # 4. El lidar ve la cara de la caja; el punto que puntúa es su centro (§6.1).
        distancia += self.get_parameter('semilado_caja_m').value
        x, y, yaw = self.pose
        return punto_en_el_plano(x, y, yaw, angulo, distancia)

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
