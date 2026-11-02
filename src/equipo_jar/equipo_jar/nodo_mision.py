"""La misión: recorre el laberinto con Nav2, reporta las cajas y vuelve a la largada.

Es el único dueño de /cmd_vel y de /victimas. Nav2 publica en /equipo/cmd_vel_nav y este nodo lo
deja pasar a /cmd_vel, salvo cuando manda él: mirando un objeto, escapando de una traba o al
terminar. Así dos nodos nunca se pisan el comando.

Cuando el robot se traba, cancela la meta y sale despacio (equipo_jar/escape.py). Reintenta la meta
una vez; si se vuelve a trabar, la abandona: se pierde esa meta, no la corrida.

Lo que el lidar ve y no está en el mapa se va a mirar de cerca (inspección), salvo que la cámara
ya lo haya visto sin rojo (percepción lo publica en /equipo/vistas_camara, ver memoria_camara.py):
una caja de otro color, un señuelo o una pared que falta en el mapa no son víctimas. Si se entera
en camino, corta el viaje; si se entera mirándolo, deja de mirar.

También es el único que decide cuándo AMCL convergió al arrancar: cuando el robot se alejó
`distancia_convergencia_m` de la largada y pasaron `espera_convergencia_s`. Quieto, AMCL no fija
bien el ángulo (en simulación quedaba 5,5° corrido aunque su covarianza ya era chica, y una caja
a 2,7 m se ubicaba 0,3 m mal); moviéndose unos decímetros baja a menos de 1°. Lo avisa una vez en
/equipo/amcl_convergido y percepción no ubica cajas hasta recibirlo: un solo criterio para los dos.
"""
import math
import traceback

from action_msgs.msg import GoalStatus
from equipo_jar.cobertura import Cobertura
from equipo_jar.escape import Cuerpo, DetectorTraba, Escape, puntos_en_robot
from equipo_jar.evasion import normalizar_angulo
from equipo_jar.memoria_camara import ROJO_VISTO, SIN_ROJO, SIN_VER
from equipo_jar.reportes import armar_reporte
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy, QoSProfile, qos_profile_sensor_data, ReliabilityPolicy)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

COBERTURA, INSPECCION, REGRESO = 'COBERTURA', 'INSPECCION', 'REGRESO'
TOPIC_CONVERGENCIA = '/equipo/amcl_convergido'  # interno: lo escucha nodo_percepcion
TOPIC_VISTAS = '/equipo/vistas_camara'          # interno: qué miró la cámara (nodo_percepcion)


def yaw_de(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class NodoMision(Node):
    def __init__(self):
        super().__init__('nodo_mision')
        self.declare_parameter('frecuencia_hz', 20.0)
        self.declare_parameter('cantidad_victimas', 5)
        self.declare_parameter('frame_mapa', 'map')
        self.declare_parameter('frame_odom', 'odom')
        self.declare_parameter('frame_robot', 'base_footprint')
        self.declare_parameter('radio_paso_m', 0.17)
        self.declare_parameter('despeje_meta_m', 0.30)
        self.declare_parameter('tiempo_regreso_s', 480.0)
        self.declare_parameter('tiempo_max_meta_s', 45.0)
        self.declare_parameter('espera_nav2_s', 25.0)
        self.declare_parameter('distancia_convergencia_m', 0.6)
        self.declare_parameter('espera_convergencia_s', 8.0)
        self.declare_parameter('radio_llegada_m', 0.15)
        self.declare_parameter('intentos_regreso', 6)
        self.declare_parameter('tiempo_inspeccion_s', 4.0)
        self.declare_parameter('despeje_giro_m', 0.25)
        self.declare_parameter('radio_duplicado_m', 0.40)
        self.declare_parameter('radio_objeto_conocido_m', 0.50)
        self.declare_parameter('margen_pared_objetos_m', 0.20)
        self.declare_parameter('descartar_con_camara', True)
        self.declare_parameter('veces_sin_rojo', 3)
        self.declare_parameter('celdas_sin_rojo', 3)
        self.declare_parameter('rango_min_lidar_m', 0.05)
        self.declare_parameter('rango_max_lidar_m', 3.5)
        self.declare_parameter('frente_robot_m', 0.13)
        self.declare_parameter('atras_robot_m', 0.195)
        self.declare_parameter('semiancho_robot_m', 0.125)
        self.declare_parameter('traba_radio_m', 0.08)
        self.declare_parameter('traba_giro_rad', 0.30)
        self.declare_parameter('traba_tiempo_s', 6.0)
        self.declare_parameter('traba_radio_avance_m', 0.15)
        self.declare_parameter('traba_tiempo_avance_s', 15.0)
        self.declare_parameter('escape_velocidad_mps', 0.10)
        self.declare_parameter('escape_despeje_objetivo_m', 0.15)
        self.declare_parameter('escape_margen_m', 0.02)
        self.declare_parameter('escape_alcance_m', 0.25)
        self.declare_parameter('escape_duracion_max_s', 6.0)
        self.declare_parameter('trabas_por_meta', 2)
        p = self.get_parameter
        self.cantidad_victimas = p('cantidad_victimas').value
        self.frame_mapa = p('frame_mapa').value
        self.frame_odom = p('frame_odom').value
        self.frame_robot = p('frame_robot').value
        self.cuerpo = Cuerpo(
            p('frente_robot_m').value, p('atras_robot_m').value, p('semiancho_robot_m').value)

        self.cobertura = None
        self.scan = None
        self.lidar = None
        self.pose = None
        self.t_inicio = None
        self.estado = 'EXPLORAR'        # EXPLORAR -> REGRESO -> FIN
        self.yaw_anterior = None
        self.t_anterior = None
        self.pose_arranque = None
        self.largada_odom = None
        self.largada = None
        self.t_convergencia = None
        self.ultimo_visto = 0.0
        self.ultimo_objetos = 0.0
        self.confirmadas = []
        self.cajas_en_espera = []       # llegaron antes de que AMCL convergiera
        self.reportes = 0
        self.inspeccionados = []
        self.pendientes = []
        self.descartados = []           # la cámara los vio sin rojo: no se van a mirar
        self.vistas_camara = None       # grilla de /equipo/vistas_camara (memoria_camara.py)
        self.objeto_meta = None         # el objeto que se va a mirar con la meta de inspección
        self.objeto_mirando = None      # el que se está mirando ahora
        self.intentos_regreso = 0

        self.meta = None                # (x, y, yaw, tipo)
        self.k_meta = None              # candidato de cobertura de la meta actual
        self.t_meta = 0.0
        self.trabas_meta = 0
        self.meta_a_retomar = None
        self.num_meta = 0               # distingue la respuesta de Nav2 de una meta ya cancelada
        self.handle_meta = None
        self.resultado_meta = None
        self.meta_rechazada = False
        self.t_reintentar = 0.0
        self.t_mirar_hasta = 0.0
        self.yaw_mirar = 0.0
        self.escape = None
        self.t_cmd_nav = -math.inf
        self.traba = DetectorTraba(
            p('traba_radio_m').value, p('traba_giro_rad').value, p('traba_tiempo_s').value,
            p('traba_radio_avance_m').value, p('traba_tiempo_avance_s').value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos_mapa = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, qos_mapa)
        self.sub_scan = self.create_subscription(
            LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)
        self.sub_cajas = self.create_subscription(
            PointStamped, '/equipo/cajas_confirmadas', self.al_recibir_caja, 10)
        self.sub_cmd_nav = self.create_subscription(
            Twist, '/equipo/cmd_vel_nav', self.al_recibir_cmd_nav, 10)
        self.sub_vistas = self.create_subscription(
            OccupancyGrid, TOPIC_VISTAS, self.al_recibir_vistas, qos_mapa)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        # Transient local, como /map: si percepción arranca (o se reinicia) después del aviso,
        # igual lo recibe.
        self.pub_convergencia = self.create_publisher(Empty, TOPIC_CONVERGENCIA, qos_mapa)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.create_timer(1.0 / p('frecuencia_hz').value, self.ciclo)

    def ahora(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def valor(self, nombre):
        return self.get_parameter(nombre).value

    # Callbacks: solo guardan el dato. La lógica va en ciclo().

    def al_recibir_mapa(self, mapa):
        if self.cobertura is not None:
            return
        grilla = np.array(mapa.data, dtype=np.int16).reshape(mapa.info.height, mapa.info.width)
        origen = mapa.info.origin.position
        pose = self.leer_tf(self.frame_mapa, self.frame_robot)
        self.cobertura = Cobertura(
            grilla, mapa.info.resolution, origen.x, origen.y, pose[:2] if pose else (0.0, 0.0),
            self.valor('radio_paso_m'), self.valor('despeje_meta_m'),
            margen_pared_m=self.valor('margen_pared_objetos_m'))
        self.t_inicio = self.ahora()
        self.get_logger().info(
            f'mapa {mapa.info.width}x{mapa.info.height}, '
            f'{len(self.cobertura.candidatos)} puntos de observación')

    def al_recibir_scan(self, scan):
        self.scan = scan

    def al_recibir_vistas(self, grilla):
        self.vistas_camara = np.array(grilla.data, dtype=np.int16).reshape(
            grilla.info.height, grilla.info.width)

    def al_recibir_cmd_nav(self, cmd):
        self.t_cmd_nav = self.ahora()
        if self.escape is None and self.t_mirar_hasta == 0.0 and self.estado != 'FIN':
            self.pub_cmd.publish(cmd)

    def al_recibir_caja(self, caja):
        x, y = caja.point.x, caja.point.y
        if self.t_convergencia is None or self.ahora() < self.t_convergencia:
            # Percepción solo publica con su supervisor conforme, y no la vuelve a mandar: si se
            # tirara, se perdería. Se reporta cuando acá también se considera convergido.
            self.cajas_en_espera.append((x, y))
            self.get_logger().info(f'caja en espera: AMCL sin converger ({x:.2f}, {y:.2f})')
            return
        self.registrar_caja(x, y)

    def registrar_caja(self, x, y):
        if self.estado == 'FIN':
            return  # tras /done no se reporta más
        if not (math.isfinite(x) and math.isfinite(y)):
            self.get_logger().error(f'caja con coordenadas no finitas ({x}, {y}): no gasto un reporte')
            return
        radio = self.valor('radio_duplicado_m')
        if any(math.hypot(x - a, y - b) < radio for a, b in self.confirmadas):
            return
        self.confirmadas.append((x, y))
        if self.reportes < self.cantidad_victimas:
            self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))
            self.reportes += 1
            self.get_logger().info(
                f'REPORTE {self.reportes}/{self.cantidad_victimas} ({x:.2f}, {y:.2f}) '
                f't={self.ahora() - self.t_inicio:.1f}')

    # Lectura de TF y del lidar

    def leer_tf(self, padre, hijo):
        try:
            t = self.tf_buffer.lookup_transform(padre, hijo, Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_de(t.transform.rotation)

    def en_mapa(self, punto_odom):
        x, y, yaw = self.leer_tf(self.frame_mapa, self.frame_odom)
        px, py = punto_odom
        return (x + math.cos(yaw) * px - math.sin(yaw) * py,
                y + math.sin(yaw) * px + math.cos(yaw) * py)

    def puntos_lidar(self):
        """Puntos del último scan en el marco del robot, o None si todavía no se pueden calcular."""
        if self.scan is None:
            return None
        if self.lidar is None:
            try:
                t = self.tf_buffer.lookup_transform(
                    self.frame_robot, self.scan.header.frame_id, Time())
            except Exception:
                return None
            q, a = t.transform.rotation, t.transform.translation
            self.lidar = (
                a.x, a.y,
                1.0 - 2.0 * (q.y * q.y + q.z * q.z), 2.0 * (q.x * q.y - q.w * q.z),
                2.0 * (q.x * q.y + q.w * q.z), 1.0 - 2.0 * (q.x * q.x + q.z * q.z))
        return puntos_en_robot(
            self.scan.ranges, self.scan.angle_min, self.scan.angle_increment,
            self.valor('rango_min_lidar_m'), self.valor('rango_max_lidar_m'), self.lidar)

    def actualizar_mundo(self, ahora):
        x, y, yaw = self.pose
        giro = 0.0
        if self.yaw_anterior is not None:
            dt = max(ahora - self.t_anterior, 1e-3)
            giro = abs(normalizar_angulo(yaw - self.yaw_anterior)) / dt
        self.yaw_anterior, self.t_anterior = yaw, ahora

        # Lo que se ve antes de que AMCL converja cae en cualquier lado: no se usa.
        recorrido = math.hypot(x - self.pose_arranque[0], y - self.pose_arranque[1])
        if self.t_convergencia is None and recorrido > self.valor('distancia_convergencia_m'):
            self.t_convergencia = ahora + self.valor('espera_convergencia_s')
        if ahora - self.ultimo_visto > 0.5:
            self.ultimo_visto = ahora
            self.cobertura.marcar_visto(x, y)
        convergio = self.t_convergencia is not None and ahora >= self.t_convergencia
        if convergio and self.largada is None:
            self.largada = self.en_mapa(self.largada_odom)
            self.get_logger().info(f'largada estimada {self.largada}')
            self.pub_convergencia.publish(Empty())  # una sola vez: percepción ya puede ubicar
            self.get_logger().info(f'AMCL convergido: aviso en {TOPIC_CONVERGENCIA}')
        if convergio and self.cajas_en_espera:
            for cx, cy in self.cajas_en_espera:
                self.registrar_caja(cx, cy)
            self.cajas_en_espera = []
        puntos = self.puntos_lidar()
        if not convergio or puntos is None or giro > 0.6:
            return
        px, py = puntos
        self.cobertura.acumular_lidar(
            self.pose,
            x + math.cos(yaw) * px - math.sin(yaw) * py,
            y + math.sin(yaw) * px + math.cos(yaw) * py)
        if ahora - self.ultimo_objetos > 1.0:
            self.ultimo_objetos = ahora
            self.anotar_objetos(self.cobertura.objetos_nuevos(), ahora)
            self.revisar_objetos(ahora)

    # Objetos fuera del mapa: cuáles hace falta ir a mirar

    def veredicto_camara(self, x, y):
        """Qué vio la cámara del objeto en (x, y): ROJO_VISTO, SIN_ROJO o SIN_VER."""
        camara = self.vistas_camara
        if (not self.valor('descartar_con_camara') or camara is None
                or camara.shape != (self.cobertura.alto, self.cobertura.ancho)):
            return SIN_VER
        return self.cobertura.veredicto_camara(
            x, y, camara, self.valor('veces_sin_rojo'), self.valor('celdas_sin_rojo'))

    def motivo_para_no_mirar(self, objeto):
        """Por qué ya no hace falta ir a mirar este objeto, o None si hace falta."""
        if objeto is None:
            return None
        radio = self.valor('radio_objeto_conocido_m')
        if any(math.hypot(objeto[0] - a, objeto[1] - b) < radio for a, b in self.confirmadas):
            return 'ya es una caja confirmada'
        if self.veredicto_camara(*objeto) == SIN_ROJO:
            return 'la cámara lo vio sin rojo'
        return None

    def anotar_objetos(self, objetos, ahora):
        """Los objetos nuevos que el lidar ve fuera del mapa quedan para inspeccionar, salvo los
        que la cámara ya vio sin rojo: una caja de otro color, un señuelo, una pared que falta."""
        radio = self.valor('radio_objeto_conocido_m')
        for cx, cy in objetos:
            conocidos = self.confirmadas + self.inspeccionados + self.pendientes + self.descartados
            if any(math.hypot(cx - a, cy - b) < radio for a, b in conocidos):
                continue
            texto = f'objeto no mapeado ({cx:.2f}, {cy:.2f}) t={ahora - self.t_inicio:.1f}'
            if self.veredicto_camara(cx, cy) == SIN_ROJO:
                self.descartados.append((cx, cy))
                self.get_logger().info(texto + ': la cámara ya lo vio sin rojo, no lo inspecciono')
            else:
                self.pendientes.append((cx, cy))
                self.get_logger().info(texto)

    def revisar_objetos(self, ahora):
        """Con lo que la cámara vio después: un pendiente sin rojo se descarta, y un descartado
        en el que apareció rojo vuelve a quedar para inspeccionar."""
        t = ahora - self.t_inicio
        for objeto in list(self.pendientes):
            if self.veredicto_camara(*objeto) == SIN_ROJO:
                self.pendientes.remove(objeto)
                self.descartados.append(objeto)
                self.get_logger().info(
                    f'descarto el objeto ({objeto[0]:.2f}, {objeto[1]:.2f}) t={t:.1f}: '
                    'la cámara lo vio sin rojo')
        for objeto in list(self.descartados):
            if self.veredicto_camara(*objeto) == ROJO_VISTO:
                self.descartados.remove(objeto)
                self.pendientes.append(objeto)
                self.get_logger().info(
                    f'el objeto ({objeto[0]:.2f}, {objeto[1]:.2f}) tiene rojo t={t:.1f}: '
                    'vuelve a quedar para inspeccionar')

    # Metas de Nav2

    def ir(self, x, y, yaw, tipo, k=None, objeto=None):
        ahora = self.ahora()
        self.meta = (x, y, yaw, tipo)
        self.k_meta = k
        self.objeto_meta = objeto
        self.t_meta = ahora
        self.resultado_meta = None
        self.handle_meta = None
        self.num_meta += 1
        self.get_logger().info(
            f'{tipo} -> ({x:.2f}, {y:.2f}) t={ahora - self.t_inicio:.1f} '
            f'visto {self.cobertura.porcentaje_visto():.0f}% cajas {len(self.confirmadas)}')
        destino = PoseStamped()
        destino.header.frame_id = self.frame_mapa
        destino.header.stamp = self.get_clock().now().to_msg()
        destino.pose.position.x, destino.pose.position.y = x, y
        a = yaw or 0.0
        destino.pose.orientation.z, destino.pose.orientation.w = math.sin(a / 2), math.cos(a / 2)
        objetivo = NavigateToPose.Goal()
        objetivo.pose = destino
        num = self.num_meta
        self.nav.send_goal_async(objetivo).add_done_callback(lambda f: self.al_aceptar(f, num))

    def al_aceptar(self, futuro, num):
        handle = futuro.result()
        if num != self.num_meta:
            if handle.accepted:
                handle.cancel_goal_async()
            return
        if not handle.accepted:
            self.meta_rechazada = True
            return
        self.handle_meta = handle
        handle.get_result_async().add_done_callback(lambda f: self.al_terminar(f, num))

    def al_terminar(self, futuro, num):
        if num == self.num_meta:
            self.resultado_meta = futuro.result().status

    def soltar_meta(self):
        """Cancela la meta actual; lo que Nav2 responda sobre ella ya no cuenta."""
        if self.handle_meta is not None and self.resultado_meta is None:
            self.handle_meta.cancel_goal_async()
        self.handle_meta = None
        self.num_meta += 1
        self.meta = None

    def abandonar_meta(self):
        tipo = self.meta[3]
        self.get_logger().warn(f'falla meta {tipo}')
        if tipo == COBERTURA:
            self.cobertura.lista_negra.add(self.k_meta)
        self.soltar_meta()
        self.trabas_meta = 0

    def llegar(self):
        if self.meta[3] == INSPECCION:
            self.t_mirar_hasta = self.ahora() + self.valor('tiempo_inspeccion_s')
            self.yaw_mirar = self.meta[2]
            self.objeto_mirando = self.objeto_meta
        self.soltar_meta()
        self.trabas_meta = 0

    # Trabas

    def atender_traba(self, ahora):
        tiempo = ahora - self.t_inicio
        self.trabas_meta += 1
        self.get_logger().warn(f'traba t={tiempo:.1f}, intento {self.trabas_meta} de esta meta')
        if self.trabas_meta >= self.valor('trabas_por_meta'):
            self.abandonar_meta()
        else:
            self.meta_a_retomar = (*self.meta, self.k_meta, self.objeto_meta)
            self.soltar_meta()
        self.traba.reiniciar()
        self.escape = Escape(
            self.cuerpo, self.valor('escape_velocidad_mps'),
            self.valor('escape_despeje_objetivo_m'), self.valor('escape_margen_m'),
            self.valor('escape_alcance_m'), self.valor('escape_duracion_max_s'), ahora)

    def seguir_escape(self, ahora):
        puntos = self.puntos_lidar()
        velocidad = self.escape.comando(*puntos, ahora) if puntos is not None else None
        cmd = Twist()
        if velocidad is not None:
            cmd.linear.x, cmd.linear.y = velocidad
            self.pub_cmd.publish(cmd)
            return
        self.pub_cmd.publish(cmd)
        self.escape = None
        self.traba.reiniciar()
        self.get_logger().info(f'fin del escape t={ahora - self.t_inicio:.1f}')
        if self.meta_a_retomar is not None:
            x, y, yaw, tipo, k, objeto = self.meta_a_retomar
            self.meta_a_retomar = None
            if tipo == INSPECCION and self.motivo_para_no_mirar(objeto):
                return  # mientras escapaba se supo que no hace falta ir a mirarlo
            self.ir(x, y, yaw, tipo, k, objeto)

    # Máquina de estados

    def elegir_meta(self):
        distancias = self.cobertura.distancias_desde(*self.pose[:2])
        while self.pendientes:
            self.pendientes.sort(key=lambda q: math.hypot(q[0] - self.pose[0], q[1] - self.pose[1]))
            objeto = self.pendientes.pop(0)
            if self.motivo_para_no_mirar(objeto):
                self.descartados.append(objeto)
                continue
            self.inspeccionados.append(objeto)
            meta = self.cobertura.meta_inspeccion(*objeto, distancias)
            if meta is not None:
                self.ir(*meta, INSPECCION, objeto=objeto)
                return True
        meta = self.cobertura.meta_cobertura(distancias)
        if meta is None:
            return False
        k, x, y = meta
        self.ir(x, y, None, COBERTURA, k)
        return True

    def planificar(self, ahora):
        """Sin meta activa: decide a dónde ir."""
        if self.estado == REGRESO:
            x, y = self.largada or self.en_mapa(self.largada_odom)
            distancia = math.hypot(x - self.pose[0], y - self.pose[1])
            self.intentos_regreso += 1
            if distancia < self.valor('radio_llegada_m') \
                    or self.intentos_regreso > self.valor('intentos_regreso'):
                self.pub_done.publish(Empty())
                self.estado = 'FIN'
                self.get_logger().info(
                    f'/done t={ahora - self.t_inicio:.1f} a {distancia:.2f} de la largada')
                return
            self.ir(x, y, None, REGRESO)
            return
        if self.elegir_meta():
            return
        cobertura = self.cobertura
        if cobertura.nuevos.any() and cobertura.porcentaje_visto() < 95.0:
            self.get_logger().warn('sin metas alcanzables: olvido los objetos vistos')
            cobertura.olvidar_objetos()
            if self.elegir_meta():
                return
        if cobertura.fase == 1 and len(self.confirmadas) < self.cantidad_victimas:
            self.get_logger().info(f'pasada fina t={ahora - self.t_inicio:.1f}')
            cobertura.pasar_a_fase_fina()
            self.ultimo_visto = 0.0
            return
        self.get_logger().info(
            f'cobertura terminada t={ahora - self.t_inicio:.1f}, '
            f'{cobertura.porcentaje_visto():.0f}% visto')
        self.estado = REGRESO

    def supervisar_meta(self, ahora):
        """Con meta activa: la corta si dejó de hacer falta, y atiende lo que responda Nav2."""
        if self.meta_rechazada:
            # Nav2 todavía no está activo: la meta no tiene nada de malo, se reintenta después.
            self.meta_rechazada = False
            self.soltar_meta()
            self.t_reintentar = ahora + 1.0
            return
        x, y, _, tipo = self.meta
        if tipo == INSPECCION:
            motivo = self.motivo_para_no_mirar(self.objeto_meta)
            if motivo:
                ox, oy = self.objeto_meta
                self.get_logger().info(
                    f'dejo de ir a mirar ({ox:.2f}, {oy:.2f}) t={ahora - self.t_inicio:.1f}: '
                    f'{motivo}')
                self.soltar_meta()
                self.trabas_meta = 0
                return
        if tipo == COBERTURA:
            ya_no_sirve = self.pendientes \
                or self.cobertura.ganancia(self.k_meta) < self.cobertura.ganancia_minima
            cerca = math.hypot(x - self.pose[0], y - self.pose[1]) < 0.3
            if cerca:
                self.cobertura.lista_negra.add(self.k_meta)
            if ya_no_sirve or cerca:
                self.soltar_meta()
                self.trabas_meta = 0
                return
        if ahora - self.t_meta > self.valor('tiempo_max_meta_s'):
            self.abandonar_meta()
        elif self.resultado_meta == GoalStatus.STATUS_SUCCEEDED:
            self.llegar()
        elif self.resultado_meta is not None:
            self.abandonar_meta()

    def mirar(self, ahora):
        cmd = Twist()
        motivo = self.motivo_para_no_mirar(self.objeto_mirando)
        if ahora > self.t_mirar_hasta or motivo:
            if motivo:
                ox, oy = self.objeto_mirando
                self.get_logger().info(
                    f'termino de mirar ({ox:.2f}, {oy:.2f}) antes t={ahora - self.t_inicio:.1f}: '
                    f'{motivo}')
            self.t_mirar_hasta = 0.0
            self.objeto_mirando = None
        else:
            error = normalizar_angulo(self.yaw_mirar - self.pose[2])
            despeje = self.cobertura.despeje_en(*self.pose[:2])
            if abs(error) > 0.08 and despeje >= self.valor('despeje_giro_m'):
                cmd.angular.z = float(np.clip(1.5 * error, -0.8, 0.8))
        self.pub_cmd.publish(cmd)

    def ciclo(self):
        """Un error en un ciclo no puede matar el nodo: sin misión, el robot queda parado y se pierde
        el resto de la corrida. Se frena, se loguea (con el traceback, para arreglarlo) y se
        reintenta en el ciclo siguiente."""
        try:
            self.paso()
        except Exception:  # a propósito: cualquier error
            self.pub_cmd.publish(Twist())
            self.get_logger().error(
                f'Error en el ciclo de la misión, sigo con el próximo:\n{traceback.format_exc()}',
                throttle_duration_sec=5.0)

    def paso(self):
        if self.cobertura is None:
            return
        self.pose = self.leer_tf(self.frame_mapa, self.frame_robot)
        if self.pose is None:
            return
        if self.largada_odom is None:
            odom = self.leer_tf(self.frame_odom, self.frame_robot)
            if odom is None:
                return
            self.largada_odom = odom[:2]
            self.pose_arranque = self.pose[:2]
        ahora = self.ahora()
        self.actualizar_mundo(ahora)
        if self.estado == 'FIN':
            self.pub_cmd.publish(Twist())
            return
        if self.escape is not None:
            self.seguir_escape(ahora)
            return
        if self.t_mirar_hasta > 0.0:
            self.traba.reiniciar()
            self.mirar(ahora)
            return
        if self.estado == 'EXPLORAR' and (
                len(self.confirmadas) >= self.cantidad_victimas
                or ahora - self.t_inicio > self.valor('tiempo_regreso_s')):
            self.estado = REGRESO
            if self.meta is not None:
                self.soltar_meta()
            self.get_logger().info(f'regreso t={ahora - self.t_inicio:.1f}')
        if self.meta is None:
            self.traba.reiniciar()
            if self.nav.server_is_ready() and ahora >= self.t_reintentar:
                self.planificar(ahora)
        else:
            if self.traba.actualizar(*self.pose, ahora) \
                    and ahora - self.t_inicio > self.valor('espera_nav2_s') \
                    and self.puntos_lidar() is not None:
                self.atender_traba(ahora)
                return
            self.supervisar_meta(ahora)
        # El watchdog de /cmd_vel corta a los 0,5 s: si Nav2 no está mandando, se manda frenar.
        if ahora - self.t_cmd_nav > 0.3:
            self.pub_cmd.publish(Twist())


def main():
    rclpy.init()
    nodo = NodoMision()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        # Si la señal de apagado corta el spin a mitad de camino, rclpy tira RCLError: no es un
        # error de la corrida. Con ROS funcionando sí lo es: que se vea.
        if rclpy.ok():
            raise
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
