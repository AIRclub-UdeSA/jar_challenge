import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Imu, LaserScan
from std_msgs.msg import Float32
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from equipo_jar.core import Board, ParticleFilter, initial_search, match_fraction, normalize

MAP_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)

def node_params(node):
    """
    Parámetros del nodo como diccionario (nombre: valor): los que ROS le cargó de params.yaml y del launch.

    Los nodos se crean con automatically_declare_parameters_from_overrides=True: ROS declara solo lo que trae el yaml, así que los valores están en un
    único lugar y el código no tiene copias. Si falta uno, el nodo falla al arrancar (KeyError con su nombre).
    """
    return {name: p.value for name, p in node.get_parameters_by_prefix('').items()}

def yaw_of(q):
    """
    Yaw (rad, en (-π, π]) de un cuaternión de ROS: hacia dónde mira el robot visto desde arriba (tipo brújula).

    Un objeto en 3D gira de tres formas, como la cabeza: 
        - yaw es decir "no" (girar a los costados)
        - pitch es decir "sí" (arriba y abajo)
        - roll es inclinarla hacia un hombro. 
        
    El robot anda sobre un piso plano: pitch y roll quedan en 0 y el único giro que cambia es el yaw. ROS guarda las orientaciones como
    cuaterniones (x, y, z, w), que describen los tres giros juntos. Esta función separa el yaw.
    """
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

def z_row(q):
    """
    Traduce el giro que mide la IMU al giro del robot. q es cómo está montada la IMU (la TF base_footprint -> imu_link).

    La IMU y el robot giran juntos y a la misma velocidad, pero cada uno lo cuenta con sus propios ejes. Un giro es positivo
    si, mirando desde la punta del eje z, se ve antihorario:
        - el robot tiene el z hacia el techo: doblar a la izquierda lo ve antihorario, +0.5 rad/s
        - la IMU de Donatello está boca abajo, con el z hacia el piso: el mismo giro lo ve horario, -0.5 rad/s

    Devuelve cuánto aporta cada eje de la IMU al giro del robot. En Donatello da (0, 0, -1): solo cuenta el z, con el signo cambiado.
    
    ACLARACIÓN: alcanzaría con multiplicar por -1 el z que mide la IMU, pero se calcula desde la TF por robustez: si la IMU del robot de la
    competencia llegara a estar montada distinto y su TF lo refleja, se ajustaría solo.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)])

def seconds(stamp):
    """
    Stamp de ROS (sec y nanosec, dos enteros). Lo convertimos en un solo float de segundos para poder restar e interpolar instantes.

    Se aplica siempre a header.stamp. Es decir, en el instante en que midió el sensor no la hora en que llegó el mensaje (un scan llega después de su stamp).
    Todos los stamps salen del mismo reloj (el robot, el simulador o el bag), así que se pueden comparar entre sí.
    """
    return Time.from_msg(stamp).nanoseconds * 1e-9

class Localizer(Node):
    def __init__(self):
        super().__init__('localizer', automatically_declare_parameters_from_overrides=True)  # Registra el nodo en ROS con los parámetros de params.yaml
        p = node_params(self)  # Los valores están solo en params.yaml (sección localizer), comentados

        # LIDAR
        self.base_frame = p['base_frame']
        self.rays = int(p['rays'])
        self.min_range = p['min_range']
        self.max_range = p['max_range']
        self.deskew = p['deskew']  # Orden temporal de los rayos para enderezar el scan; 0 lo desactiva

        # Corrección
        self.sigma = p['sigma']
        self.z_rand = p['z_rand']
        self.temperature = p['temperature']
        self.min_forward = p['min_forward']
        self.min_turn = p['min_turn']

        # Predicción
        self.trans_noise = p['trans_noise']
        self.rot_noise = p['rot_noise']
        self.forward_scale = p['forward_scale']    # se calibra en pista
        self.turn_scale = p['turn_scale']
        self.imu_turn = p['imu_turn']              # False: el giro sale de /odom

        # Relocalización
        self.rescue_confidence = p['rescue_confidence']
        self.rescue_scans = int(p['rescue_scans'])
        self.rescue_margin = p['rescue_margin']

        # Largada
        self.search_radius = p['search_radius']

        # Salida
        self.tf_lead = p['tf_lead']

        # Tamaño
        self.particles = int(p['particles'])

        self.tf = Buffer()
        self.tf_listener = TransformListener(self.tf, self)  # Guardamos los TF que van llegando
        self.tf_pub = TransformBroadcaster(self)

        self.pose_pub = self.create_publisher(PoseStamped, '/jar/pose', 10)
        self.confidence_pub = self.create_publisher(Float32, '/jar/confidence', 10)

        # Escuchamos
        self.create_subscription(OccupancyGrid, '/map', self.on_map, MAP_QOS)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 50)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 50)
        # Refrescamos 20 veces por segundo para asegurar que Nav2 tenga map->odom. Grupo propio: sigue publicando mientras
        # process_scan relocaliza (~0.35 s bloqueado, más que tf_lead)
        self.create_timer(0.05, self.publish_tf, callback_group=MutuallyExclusiveCallbackGroup())

        # Estado inicial
        self.board = self.pf = self.last = None
        self.map_odom = (0.0, 0.0, 0.0)  # Prior: Origen. Se cambia al localizarse
        self.odoms = deque(maxlen=200)  # (t, x, y, yaw) de /odom: los últimos 20 s, para interpolar
        self.turns = deque(maxlen=400)  # (t, yaw integrado de la IMU)
        self.imu_yaw, self.imu_t, self.imu_row = 0.0, None, None
        self.imu_rate = 0.0  # Último giro de la IMU sin sesgo (rad/s), para el deskew
        self.still = deque(maxlen=500)  # Lo que mide el giróscopio mientras el robot espera quieto (mide sesgo)
        self.bias = 0.0  # Sesgo del giróscopio, se calcula cuando llega el /map
        self.bad = 0  # Contador de scans mal calzados (relocalización)

    # ---------- entradas ----------
    def on_map(self, msg):
        """
        Largada del localizer: llega el mapa que publica el juez una sola vez. Antes del mapa, el localizer solo escucha /odom y la IMU. Después,
        el primer scan busca la largada y arranca el filtro.

        Hace dos cosas, una sola vez por corrida:
            - Convierte el OccupancyGrid en un Board, con la tabla de distancias a la pared más cercana que el filtro consulta en cada scan.
            - Estima el sesgo del giróscopo. Mientras el robot esperaba quieto el mapa, el giróscopo debería haber medido 0: lo que midió es el sesgo.
              Se toma la mediana de las últimas 100 muestras: son las más cercanas a la largada y la mediana no se mueve si alguien toca el robot.
        """
        if self.board is not None: return

        self.board = Board.from_msg(msg)
        if self.still: self.bias = float(np.median(list(self.still)[-100:]))
        self.get_logger().info(f'¡Mapa recibido! Sesgo del giróscopo: {self.bias:+.4f} rad/s')

    def on_odom(self, msg): # Cuánto avanzó el robot
        """
        Guarda la pose de /odom (t, x, y, yaw) en self.odoms. /odom es lo que cuentan las ruedas desde que se encendió el robot (Suave y sin saltos, pero acumula error por patinar).
        
        El filtro no usa la pose en sí, sino cuánto se movió el robot entre dos scans: es la entrada de la predicción. Solo guarda porque /odom (10 Hz) y /scan (7,6 Hz) casi nunca
        coinciden: process_scan interpola el historial en el instante de cada scan.

        atan2 devuelve (-π, π], así que al cruzar ±π la lectura salta de forma discreta aunque el robot gire apenas. Se guarda anterior + normalize(Δ). Puede exceder π: quien lo usa
        toma diferencias o normaliza.
        """
        yaw = yaw_of(msg.pose.pose.orientation)

        if self.odoms: yaw = self.odoms[-1][3] + normalize(yaw - self.odoms[-1][3])
        self.odoms.append((seconds(msg.header.stamp), msg.pose.pose.position.x, msg.pose.pose.position.y, yaw))

    def on_imu(self, msg): # Cuánto giró el robot
        """
        Integra el giróscopo y guarda (t, yaw) en self.turns. Se usa para el giro en vez del yaw de /odom porque las ruedas patinan y el giróscopo no.

        Igual que on_odom, solo guarda: process_scan interpola el historial en el instante de cada scan y toma la diferencia entre scans (dth).
        Solo usa angular_velocity, llevada al eje vertical del robot con imu_row. Antes del mapa junta las muestras crudas en self.still para el sesgo (on_map).
        
        Integramos la velocidad angular que nos entrega la IMU para obtener el ángulo. Se descartan dt <= 0 (repetidos o desordenados) y
        dt >= 0.5 s (huecos: ese giro se pierde y lo corrige el lidar). Al ser una suma, el yaw es continuo y no hace falta desenrollarlo.
        """
        if self.imu_row is None:
            try: tf = self.tf.lookup_transform(self.base_frame, msg.header.frame_id, Time())
            except Exception: return

            self.imu_row = z_row(tf.transform.rotation)

        w = msg.angular_velocity
        wz = float(self.imu_row @ [w.x, w.y, w.z])
        t = seconds(msg.header.stamp)
        
        if self.board is None: self.still.append(wz) # Antes del mapa esperamos quietitos

        # Cuadratura
        if self.imu_t is not None and 0.0 < t - self.imu_t < 0.5: self.imu_yaw += (wz - self.bias) * (t - self.imu_t)

        self.imu_rate = wz - self.bias  # para el deskew de points
        
        self.imu_t = t
        self.turns.append((t, self.imu_yaw))

    @staticmethod
    def interpolate(queue, t):
        """
        /odom y la IMU reportan cada ~0.1 s, nunca justo en t: se traza una recta entre el dato de antes y el de después y se lee el punto en t. 
        Ej.: x = 1.00 a los 10.00 s y x = 1.05 a los 10.10 s -> a los 10.06 s, x = 1.03.

        process_scan la llama con odoms -> [x, y, yaw] y con turns -> [yaw] (devuelve cada columna menos el tiempo).

        Casos borde:
            - None (el scan se descarta): no hay datos, t es más viejo que todo el historial, o el último dato tiene más de 0.25 s (sensor caído).
            - t después del último dato (el siguiente todavía no llegó): se prolonga la recta de los dos últimos datos, suponiendo que el robot no realiza cambios bruscos de movimiento.
        """
        if not queue or t > queue[-1][0] + 0.25 or t < queue[0][0]: return None
        data = np.asarray(queue)

        if t > data[-1, 0] and len(data) > 1 and data[-1, 0] > data[-2, 0]:  # el siguiente dato no llegó
            a, b = data[-2], data[-1]
            return [float(v) for v in b[1:] + (b[1:] - a[1:]) * (t - b[0]) / (b[0] - a[0])]

        return [float(np.interp(t, data[:, 0], data[:, k])) for k in range(1, data.shape[1])]

    def points(self, msg, tf):
        """
        Recibe el scan, que es una lista de 1080 distancias, y devuelve hasta self.rays (90) puntos (x, y) medidos desde el centro del robot.
        tf: dónde está montado el lidar en el robot (la TF base_footprint -> laser_link).

        Es la forma en que el filtro "ve" las paredes: initial_search, pf.correct y match_fraction apoyan estos puntos sobre el mapa con la pose
        de cada partícula y miran si caen sobre paredes. Por eso van en el frame del robot, iguales para todas las partículas.

        Pasos:
            - Ángulo de cada rayo según su posición en la lista.
            - Descarta rayos sin eco, más cerca que min_range o más lejos que max_range.
            - Se queda con self.rays repartidos parejo (los vecinos dicen casi lo mismo).
            - Polares -> cartesianas y del lidar al robot con la TF (el lidar está girado 180°).
            - Deskew: gira cada punto lo que giró el robot durante la vuelta del lidar (en el simulador no hace nada).
        """
        rays = np.asarray(msg.ranges)
        theta = msg.angle_min + np.arange(len(rays)) * msg.angle_increment # El ángulo de cada rayo se deduce de su posición en la lista
        ok = np.flatnonzero(np.isfinite(rays) & (rays > max(msg.range_min, self.min_range)) & (rays < min(msg.range_max, self.max_range)))

        if len(ok) > self.rays: ok = ok[np.linspace(0, len(ok) - 1, self.rays).astype(int)]
        x, y = rays[ok] * np.cos(theta[ok]), rays[ok] * np.sin(theta[ok]) # Polares
        tx, ty = tf.transform.translation.x, tf.transform.translation.y
        yaw = yaw_of(tf.transform.rotation)
        c, s = math.cos(yaw), math.sin(yaw)
        px, py = tx + c * x - s * y, ty + s * x + c * y

        # Deskew: cada rayo se midió en otro momento de la vuelta; se gira lo que giró el robot entre ese momento y el instante medio
        d = self.deskew * self.imu_rate * self.turn_scale * (ok - (len(rays) - 1) / 2) * msg.time_increment
        c, s = np.cos(d), np.sin(d)

        return np.column_stack([c * px - s * py, s * px + c * py])

    # ---------- filtro ----------
    def on_scan(self, msg):
        try: self.process_scan(msg)
        except Exception as e: self.get_logger().error(f'on_scan: {e!r}', throttle_duration_sec=1.0)

    def process_scan(self, msg):
        if self.board is None: return
        
        try: laser = self.tf.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except Exception: return

        # Instante medio de los rayos: en el real se reparten a lo largo de la vuelta desde el stamp
        t = seconds(msg.header.stamp) + msg.time_increment * (len(msg.ranges) - 1) / 2
        odom = self.interpolate(self.odoms, t)
        turn = self.interpolate(self.turns, t)
        
        if odom is None or (self.imu_turn and turn is None): return
        ox, oy, oyaw = odom
        yaw = turn[0] if self.imu_turn else oyaw
        
        pts = self.points(msg, laser)
        if len(pts) < 20: return

        if self.pf is None: # Primer scan con mapa: la largada
            pose = initial_search(self.board, pts, radius=self.search_radius)
            self.pf = ParticleFilter(self.board, self.particles)
            self.pf.initialize(pose)
            self.last = (ox, oy, oyaw, yaw)

            self.get_logger().info(f'largada estimada: x={pose[0]:.3f} y={pose[1]:.3f} '
                                   f'yaw={math.degrees(pose[2]):.1f}° '
                                   f'(calce {match_fraction(self.board, pose, pts):.0%})')
        else:
            lx, ly, lyaw, lturn = self.last
            c, s = math.cos(lyaw), math.sin(lyaw)

            dx = (c * (ox - lx) + s * (oy - ly)) * self.forward_scale
            dy = (-s * (ox - lx) + c * (oy - ly)) * self.forward_scale
            dth = (yaw - lturn) * self.turn_scale

            # Con el robot quieto no se corrige: el mismo scan una y otra vez sobreconfía al filtro.
            if math.hypot(dx, dy) > self.min_forward or abs(dth) > self.min_turn:
                self.pf.predict(dx, dy, dth, self.trans_noise, self.rot_noise)
                self.pf.correct(pts, self.sigma, self.z_rand, self.temperature)
                self.last = (ox, oy, oyaw, yaw)
        est, _ = self.pf.estimate()

        # map->odom: cuánto girar y correr el frame odom para que la pose de la odometría coincida con la del filtro (ambas en t)
        th = normalize(est[2] - oyaw)
        c, s = math.cos(th), math.sin(th)
        self.map_odom = (est[0] - (c * ox - s * oy), est[1] - (s * ox + c * oy), th)

        confidence = match_fraction(self.board, est, pts)
        self.confidence_pub.publish(Float32(data=confidence))

        # Contamos los scans seguidos con confianza menor a rescue_confidence, un scan bueno lo devuelve a 0
        self.bad = self.bad + 1 if confidence < self.rescue_confidence else 0

        # Después de 2s sin calzar, busca a fuerza bruta la mejor pose a menos de 0.5m de donde creía estar
        if self.bad > self.rescue_scans:
            candidate = initial_search(self.board, pts, center=tuple(est), radius=0.5)
            
            # Si no la hay, el mundo difiere del mapa (objeto, ventana) y el filtro no estaba perdido.
            if match_fraction(self.board, candidate, pts) > confidence + self.rescue_margin:
                self.get_logger().warn('filtro perdido: relocalizando')
                self.pf.initialize(candidate)
                self.last = (ox, oy, oyaw, yaw)  # candidate es la pose en t: el próximo movimiento se mide desde acá
            self.bad = 0

        # Armamos un mensaje de ROS con la pose en el mapa
        pose = PoseStamped()
        pose.header.frame_id = 'map'
        pose.header.stamp = Time(nanoseconds=round(t * 1e9)).to_msg()  # instante medio del scan, al que corresponde est
        pose.pose.position.x, pose.pose.position.y = float(est[0]), float(est[1])
        pose.pose.orientation.z, pose.pose.orientation.w = math.sin(est[2] / 2), math.cos(est[2] / 2)
        self.pose_pub.publish(pose)

    def publish_tf(self):
        """
        Publica la última corrección map->odom (self.map_odom, la calcula process_scan) en /tf. Es la salida principal del localizer.

        Corre con un timer a 20 Hz, no con cada scan: Nav2 necesita el frame map desde el arranque y un flujo parejo aunque se descarten scans.

        Stamp adelantado tf_lead (0.3 s), como transform_tolerance de AMCL: TF no extrapola hacia adelante y sin esto las consultas en el instante actual pueden fallar.
        Es válido porque map->odom es el desfase de la odometría y cambia despacio. El movimiento rápido va en odom->base_footprint.
        """
        x, y, th = self.map_odom
        t = TransformStamped()
        t.header.stamp = (self.get_clock().now() + Duration(seconds=self.tf_lead)).to_msg()
        t.header.frame_id, t.child_frame_id = 'map', 'odom'

        t.transform.translation.x, t.transform.translation.y = float(x), float(y)
        t.transform.rotation.z, t.transform.rotation.w = math.sin(th / 2), math.cos(th / 2)
        self.tf_pub.sendTransform(t)

def main(args=None):
    """
    Punto de entrada del ejecutable localizer (setup.py). Lo arranca competencia.launch.py con params.yaml.

    Executor multihilo: publish_tf (grupo propio) sigue publicando map->odom mientras un callback de sensores tarda, como una relocalización (~0.35 s).
    Los callbacks de sensores comparten el grupo por defecto y corren de a uno, sin candados.
    """
    rclpy.init(args=args)  # Nos comunicamos con ROS
    node = Localizer()
    executor = MultiThreadedExecutor()  # publish_tf en su propio hilo; los callbacks de sensores siguen de a uno
    executor.add_node(node)

    try: executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException): pass
    except Exception:
        if rclpy.ok(): raise # con ROS ya apagado (Ctrl+C, el launch), spin puede tirar RCLError al armar la espera: no es un error
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok(): rclpy.shutdown()
