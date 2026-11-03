"""Evasor: explora el laberinto esquivando obstáculos y buscando terreno que la cámara no vio.

Publica en cmd_vel_evasor; nodo_equipo decide si lo reenvía a /cmd_vel (no antes del mapa,
no después del /done). La lógica pura está en exploracion.py.

Estados:
  AVANZAR    hacia adelante, más lento cuanto más cerca está lo de enfrente, y alejándose de
             costado de las paredes laterales (base mecanum).
  GIRAR      en el lugar, hacia la dirección elegida (con memoria: la más libre y con más
             terreno sin ver; sin memoria: el lado más libre hasta despejar el frente).
  BARRER     una vuelta completa en el lugar al llegar a una zona con mucho sin ver: la
             cámara ve solo ~61°.
  ACERCARSE  hacia un objeto rojo no mapeado que todavía no se confirmó, para verlo de cerca.

Cada mejora se apaga desde params.yaml (girar_hacia_libre, acercarse_rojo,
explorar_con_memoria, barridos): con todo en false queda el evasor original.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseArray, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

from equipo_jar.exploracion import (Cobertura, angulos_relativos, distancia_al_frente,
                                    distancia_en_sector, distancias_laterales,
                                    elegir_direccion, factor_velocidad, lado_mas_libre,
                                    puntos_xy, velocidad_avance, velocidad_lateral)
from equipo_jar.imu import GiroImu
from equipo_jar.regreso import campo_hacia, distancia_a_paredes, valor_cercano

FRECUENCIA_HZ = 10.0

AVANZAR = 'avanzar'
GIRAR = 'girar'
BARRER = 'barrer'
ACERCARSE = 'acercarse'

QOS_MAPA = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                      durability=DurabilityPolicy.TRANSIENT_LOCAL)


def normalizar(angulo: float) -> float:
    return math.atan2(math.sin(angulo), math.cos(angulo))


def yaw_de_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Evasor(Node):

    def __init__(self):
        super().__init__('evasor')
        parametros = {
            # Cono frontal (grados) y distancia (m) a la que un obstáculo obliga a girar.
            'angulo_vision_deg': 60.0,
            'distancia_choque_m': 0.6,
            # Ángulo del /scan que es el frente del robot: el lidar está montado a 180°.
            'angulo_frente_deg': 180.0,
            'velocidad_adelante': 0.3,
            # En zonas abiertas (lejos del frente y de los dos costados) acelera hasta esta.
            'velocidad_rapida': 0.45,
            'abierto_lateral_desde': 0.4,     # m a la pared lateral más cercana: empieza a acelerar
            'abierto_lateral_hasta': 0.8,     # m: a velocidad_rapida
            'abierto_frente_desde': 1.0,      # m libres adelante: empieza a acelerar
            'abierto_frente_hasta': 1.5,      # m: a velocidad_rapida
            # Giro normal (rápido) y giro con rojo a la vista: tiene que quedar por debajo de
            # giro_maximo de detector_scan para que pueda marcar los puntos rojos del lidar.
            'velocidad_angular': 0.8,
            'velocidad_angular_rojo': 0.4,
            'tiempo_giro_lento': 1.5,         # s que sigue lento después de dejar de ver rojo
            # Solo sin girar_hacia_libre ni memoria (evasor original): giro fijo, siempre a la izq.
            'angulo_giro_deg': 110.0,
            # Nivel 1: girar hacia el lado más libre hasta despejar el frente.
            'girar_hacia_libre': True,
            # Nivel 1: con una pared a menos de esto al costado, alejarse de costado.
            'margen_lateral': 0.22,
            # Forma del robot (m, desde el centro del lidar) para ver adelante y a los costados
            # con un rectángulo en vez de un cono: así no hay punto ciego en las diagonales.
            'medio_ancho_robot': 0.13,
            'medio_largo_robot': 0.16,
            'margen_frontal': 0.05,           # m de más a cada lado del corredor de adelante
            # m delante de la trompa que cuentan como "costado": poco, para no correrse por la
            # punta de una pared que todavía no alcanzó (las diagonales las cubre el corredor).
            'anticipacion_lateral': 0.05,
            'velocidad_lateral': 0.1,
            # Nivel 1: con paredes a los dos lados a menos de esto, centrarse en el pasillo.
            'ancho_pasillo': 0.5,
            'ganancia_centrado': 0.5,
            # Nivel 1: a partir de esta distancia al frente se empieza a frenar.
            'distancia_lenta': 1.0,
            # Acercarse a lo rojo no confirmado.
            'acercarse_rojo': True,
            'distancia_acercamiento': 0.7,    # m: a esta distancia del objeto se queda mirando
            'tiempo_max_acercamiento': 10.0,  # s: después se abandona (y se ignora un rato)
            'distancia_max_rojo': 2.5,        # m: objetos más lejanos no se persiguen
            'radio_confirmada': 0.5,          # m: un objeto tan cerca de una confirmada ya está
            # s que se recuerda un objeto rojo visto: girando, la caja sale de la cámara enseguida
            # y sin memoria el robot se olvida de ella antes de poder ir.
            'memoria_rojo': 8.0,
            # Nivel 2: memoria de lo que vio la cámara.
            'explorar_con_memoria': True,
            'fov_camara_deg': 61.0,           # campo horizontal de la cámara (camera_info)
            'alcance_camara': 2.5,            # m: hasta dónde se cuenta como "visto"
            'peso_libre': 1.0,
            'peso_no_visto': 2.0,
            # Preferencia por seguir derecho (0 = ninguna): evita ir y volver por el mismo pasillo.
            'peso_seguir': 0.8,
            # "Nada nuevo adelante" solo cambia de rumbo si hay otra dirección con al menos esta
            # fracción más de terreno sin ver; si todo alrededor está visto, sigue de largo.
            'mejora_para_cambiar': 0.2,
            'periodo_reevaluar': 4.0,         # s: cada cuánto se mira si queda algo nuevo adelante
            'no_visto_minimo': 0.05,          # fracción: menos que esto adelante = "ya visto"
            # Nivel 2: vueltas completas en zonas con mucho sin ver.
            'barridos': True,
            'radio_barrido': 1.2,             # m
            'fraccion_para_barrer': 0.5,      # fracción sin ver alrededor para hacer una vuelta
            'distancia_entre_barridos': 1.0,  # m
            'espacio_para_barrer': 0.3,       # m libres alrededor para girar sin rozar
            # Volviendo a la largada (nodo_equipo publica objetivo_regreso): en vez de explorar,
            # elegir el rumbo que más acerca CAMINANDO por el laberinto (mapa de distancias).
            'regreso_por_mapa': True,
            'margen_regreso': 0.15,           # m: el camino calculado no pasa más cerca de paredes
            'paso_regreso': 0.5,              # m: cuánto adelante se evalúa cada dirección
            'periodo_regreso': 1.0,           # s entre revisiones del rumbo
            'tolerancia_rumbo_regreso_deg': 25.0,  # ° de desvío que se toleran sin girar
        }
        for nombre, valor in parametros.items():
            self.declare_parameter(nombre, valor)
            setattr(self, nombre, self.get_parameter(nombre).value)
        self.angulo_frente = math.radians(self.angulo_frente_deg)
        self.angulo_vision = math.radians(self.angulo_vision_deg)
        self.angulo_giro = math.radians(self.angulo_giro_deg)
        self.fov_camara = math.radians(self.fov_camara_deg)

        self.publisher_ = self.create_publisher(Twist, 'cmd_vel_evasor', 10)
        # Copia de /scan con solo los rayos que frenan al robot, para verla en RViz.
        self.publisher_scan_cono = self.create_publisher(LaserScan, 'scan_cono', 10)
        self.create_subscription(LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.recibir_odom, 10)
        self.create_subscription(OccupancyGrid, '/map', self.recibir_mapa, QOS_MAPA)
        self.create_subscription(PoseArray, 'objetos_no_mapeados', self.recibir_objetos, 10)
        self.create_subscription(PointStamped, 'victima_confirmada', self.recibir_confirmada, 10)
        self.create_subscription(Bool, 'rojo_a_la_vista', self.recibir_rojo_a_la_vista, 10)
        self.create_subscription(PointStamped, 'objetivo_regreso', self.recibir_objetivo_regreso, 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # Giro medido con el IMU: en el robot real las ruedas patinan al girar.
        self.giro = GiroImu(self, self.tf_buffer)
        self.create_timer(1.0 / FRECUENCIA_HZ, self.maquina_de_estados)

        self.ultimo_scan = None
        self.yaw_actual = None
        self.cobertura = None
        self.estado = AVANZAR
        self.ticks = 0
        self.yaw_inicial_giro = ('odom', 0.0)
        self.sentido_giro = 1
        self.yaw_objetivo = None          # GIRAR con memoria: rumbo en `map` al que ir
        self.ultima_reevaluacion = 0.0
        self.ultimo_barrido = None        # (x, y) en `map`
        self.confirmadas = []             # (x, y) en `map`
        self.descartados = []             # (x, y, hasta_cuando) objetos rojos abandonados
        self.objetos = []                 # objetos rojos no mapeados vistos: (x, y, instante)
        self.objetivo_rojo = None
        self.inicio_acercamiento = 0.0
        self.ultimo_rojo_visto = None     # s (reloj de ROS)
        self.mapa_info = None
        self.distancias_pared = None
        self.objetivo_regreso = None      # (x, y) de la largada, solo mientras vuelve
        self.campo_regreso = None         # m a recorrer hasta la largada, por celda
        self.ultima_revision_regreso = 0.0

    # ---------- entradas ----------

    def recibir_scan(self, msg: LaserScan):
        self.ultimo_scan = msg

    def recibir_odom(self, msg: Odometry):
        self.yaw_actual = yaw_de_quaternion(msg.pose.pose.orientation)

    def recibir_mapa(self, msg: OccupancyGrid):
        if self.cobertura is None:
            grilla = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
            self.cobertura = Cobertura(grilla, msg.info.resolution,
                                       msg.info.origin.position.x, msg.info.origin.position.y)
            self.mapa_info = msg.info
            self.distancias_pared = distancia_a_paredes(grilla, msg.info.resolution)

    def recibir_objetivo_regreso(self, msg: PointStamped):
        if not self.regreso_por_mapa or self.distancias_pared is None:
            return
        x, y = msg.point.x, msg.point.y
        if self.objetivo_regreso is not None and math.hypot(
                x - self.objetivo_regreso[0], y - self.objetivo_regreso[1]) < 0.05:
            return
        i = self.mapa_info
        self.objetivo_regreso = (x, y)
        self.campo_regreso = campo_hacia(self.distancias_pared, i.resolution, i.origin.position.x,
                                         i.origin.position.y, (x, y), self.margen_regreso)
        self.get_logger().info(f'Volviendo: rumbo por el mapa hacia la largada ({x:.2f}, {y:.2f})')

    @property
    def volviendo(self) -> bool:
        return self.campo_regreso is not None

    def rumbo_regreso(self, rangos, rel, rmin, pose):
        """Dirección relativa (rad) libre que más reduce la distancia a recorrer hasta la
        largada, o None si ninguna la reduce."""
        i = self.mapa_info
        ox, oy, res = i.origin.position.x, i.origin.position.y, i.resolution
        actual = valor_cercano(self.campo_regreso, res, ox, oy, pose[0], pose[1], 0.15)
        mejor, progreso_mejor = None, 0.0
        for grados in range(-150, 181, 30):
            c = math.radians(grados)
            if distancia_en_sector(rangos, rel, c, math.radians(40), rmin) < self.distancia_choque_m + 0.15:
                continue
            a = pose[2] + c
            destino = valor_cercano(self.campo_regreso, res, ox, oy,
                                    pose[0] + self.paso_regreso * math.cos(a),
                                    pose[1] + self.paso_regreso * math.sin(a), 0.1)
            progreso = actual - destino
            if math.isfinite(progreso) and progreso > progreso_mejor:
                mejor, progreso_mejor = c, progreso
        return mejor

    def recibir_objetos(self, msg: PoseArray):
        ahora = self.ahora()
        vigentes = [o for o in self.objetos if ahora - o[2] <= self.memoria_rojo]
        for p in msg.poses:
            x, y = p.position.x, p.position.y
            # Uno nuevo cerca de uno recordado es el mismo objeto: se actualiza.
            vigentes = [o for o in vigentes if math.hypot(o[0] - x, o[1] - y) >= self.radio_confirmada]
            vigentes.append((x, y, ahora))
        self.objetos = vigentes

    def recibir_confirmada(self, msg: PointStamped):
        self.confirmadas.append((msg.point.x, msg.point.y))

    def recibir_rojo_a_la_vista(self, msg: Bool):
        if msg.data:
            self.ultimo_rojo_visto = self.ahora()

    def velocidad_de_giro(self) -> float:
        """Rápido en general; lento si la cámara vio rojo hace poco, para que se pueda
        detectar (girando rápido detector_scan no marca puntos rojos)."""
        if (self.ultimo_rojo_visto is not None
                and self.ahora() - self.ultimo_rojo_visto <= self.tiempo_giro_lento):
            return self.velocidad_angular_rojo
        return self.velocidad_angular

    def ahora(self) -> float:
        """Segundos del reloj de ROS (el simulado en Gazebo y en el bag)."""
        return self.get_clock().now().nanoseconds * 1e-9

    def pose_map(self):
        """(x, y, yaw) del robot en `map`, o None sin localización."""
        try:
            t = self.tf_buffer.lookup_transform('map', 'base_footprint', Time()).transform
        except Exception:
            return None
        return t.translation.x, t.translation.y, yaw_de_quaternion(t.rotation)

    # ---------- lidar ----------

    def lidar(self):
        scan = self.ultimo_scan
        rangos = np.asarray(scan.ranges, dtype=float)
        rel = angulos_relativos(len(rangos), scan.angle_min, scan.angle_increment,
                                self.angulo_frente)
        return rangos, rel, scan.range_min

    def distancia_frente(self, rangos, rel, rmin):
        x, y = puntos_xy(rangos, rel, rmin)
        return min(distancia_en_sector(rangos, rel, 0.0, self.angulo_vision, rmin),
                   distancia_al_frente(x, y, self.medio_ancho_robot + self.margen_frontal))

    def hay_obstaculo(self, rangos, rel, rmin) -> bool:
        """Algo a menos de distancia_choque_m en el cono frontal o en el corredor que barre
        el robot (lo que está en diagonal contra las esquinas delanteras)."""
        cono = (np.abs(rel) < self.angulo_vision / 2)
        x = rangos * np.cos(rel)
        y = rangos * np.sin(rel)
        corredor = (x > 0) & (np.abs(y) < self.medio_ancho_robot + self.margen_frontal)
        mascara = (cono | corredor) & (rangos > rmin) & (x < self.distancia_choque_m) & np.isfinite(rangos)
        self.publicar_scan_filtrado(mascara)
        return bool(mascara.any())

    def publicar_scan_filtrado(self, mascara: np.ndarray):
        scan = self.ultimo_scan
        msg = LaserScan()
        msg.header = scan.header
        msg.angle_min, msg.angle_max = scan.angle_min, scan.angle_max
        msg.angle_increment, msg.time_increment = scan.angle_increment, scan.time_increment
        msg.scan_time, msg.range_min, msg.range_max = scan.scan_time, scan.range_min, scan.range_max
        msg.ranges = np.where(mascara, np.asarray(scan.ranges), math.inf).tolist()
        self.publisher_scan_cono.publish(msg)

    # ---------- giros (medidos con el IMU, o con /odom si no hay IMU) ----------

    def yaw_para_giros(self):
        if self.giro.listo:
            return ('imu', self.giro.yaw)
        return ('odom', self.yaw_actual)

    def angulo_girado(self) -> float:
        fuente, yaw = self.yaw_para_giros()
        fuente_inicial, yaw_inicial = self.yaw_inicial_giro
        if fuente != fuente_inicial:
            self.yaw_inicial_giro = (fuente, yaw)
            return 0.0
        if fuente == 'imu':
            return abs(yaw - yaw_inicial)
        return abs(normalizar(yaw - yaw_inicial))

    # ---------- decisiones ----------

    def cambiar(self, estado, motivo):
        if estado != self.estado:
            self.get_logger().info(f'{self.estado} -> {estado}: {motivo}')
        self.estado = estado
        self.yaw_inicial_giro = self.yaw_para_giros()

    def elegir_giro(self, rangos, rel, rmin, motivo):
        """Pasa a GIRAR hacia la mejor dirección (con memoria) o hacia el lado más libre."""
        pose = self.pose_map()
        if self.volviendo and pose is not None:
            rumbo = self.rumbo_regreso(rangos, rel, rmin, pose)
            if rumbo is not None:
                self.yaw_objetivo = normalizar(pose[2] + rumbo)
                self.sentido_giro = 1 if rumbo >= 0 else -1
                self.cambiar(GIRAR, f'{motivo}; volviendo, rumbo {math.degrees(rumbo):+.0f}°')
                return
        if self.explorar_con_memoria and self.cobertura is not None and pose is not None:
            candidatos = [math.radians(a) for a in range(-150, 181, 30)]
            libres = [distancia_en_sector(rangos, rel, c, math.radians(40), rmin)
                      for c in candidatos]
            no_vistos = [self.cobertura.no_visto_hacia(pose[0], pose[1], pose[2] + c,
                                                       math.radians(30), self.alcance_camara)
                         for c in candidatos]
            i = elegir_direccion(libres, no_vistos, self.distancia_choque_m + 0.15,
                                 self.peso_libre, self.peso_no_visto, angulos=candidatos,
                                 peso_seguir=self.peso_seguir)
            if i is not None and candidatos[i] == 0.0 and not motivo.startswith('obstáculo'):
                self.yaw_objetivo = None
                self.cambiar(AVANZAR, f'{motivo}; lo mejor es seguir derecho')
                return
            if i is not None:
                self.yaw_objetivo = normalizar(pose[2] + candidatos[i])
                self.sentido_giro = 1 if candidatos[i] >= 0 else -1
                self.cambiar(GIRAR, f'{motivo}; rumbo {math.degrees(candidatos[i]):+.0f}° '
                                    f'(sin ver {no_vistos[i]:.0%})')
                return
        self.yaw_objetivo = None
        self.sentido_giro = lado_mas_libre(rangos, rel, rmin) if self.girar_hacia_libre else 1
        self.cambiar(GIRAR, motivo)

    def objetivo_rojo_valido(self, pose):
        """El objeto rojo no confirmado más cercano a perseguir, o None."""
        ahora = self.ahora()
        self.descartados = [d for d in self.descartados if d[2] > ahora]
        mejor, d_mejor = None, self.distancia_max_rojo
        for x, y, visto in self.objetos:
            if ahora - visto > self.memoria_rojo:
                continue
            if any(math.hypot(x - cx, y - cy) < self.radio_confirmada for cx, cy in self.confirmadas):
                continue
            if any(math.hypot(x - dx, y - dy) < self.radio_confirmada for dx, dy, _ in self.descartados):
                continue
            d = math.hypot(x - pose[0], y - pose[1])
            if d < d_mejor:
                mejor, d_mejor = (x, y), d
        return mejor

    def maquina_de_estados(self):
        if self.yaw_actual is None or self.ultimo_scan is None:
            return
        self.ticks += 1
        rangos, rel, rmin = self.lidar()
        pose = self.pose_map()
        if pose is not None and self.cobertura is not None and self.ticks % 5 == 0:
            self.cobertura.marcar(pose[0], pose[1], pose[2], self.fov_camara, self.alcance_camara)

        d_frente = self.distancia_frente(rangos, rel, rmin)
        obstaculo = self.hay_obstaculo(rangos, rel, rmin)
        msg = Twist()

        # ----- transiciones -----
        # Un objeto rojo sin confirmar interrumpe cualquier cosa salvo un acercamiento en curso
        # (antes solo desde AVANZAR: si lo veía girando, lo ignoraba y seguía de largo).
        if (self.estado in (AVANZAR, GIRAR, BARRER) and self.acercarse_rojo and pose is not None
                and not self.volviendo):
            objetivo = self.objetivo_rojo_valido(pose)
            if objetivo is not None:
                self.objetivo_rojo = objetivo
                self.inicio_acercamiento = self.ahora()
                self.cambiar(ACERCARSE, f'objeto rojo en ({objetivo[0]:.2f}, {objetivo[1]:.2f})')

        if self.estado == AVANZAR:
            if obstaculo:
                self.elegir_giro(rangos, rel, rmin, f'obstáculo a {d_frente:.2f} m')
            elif self.volviendo:
                # Volviendo: ni barridos ni "nada nuevo"; cada tanto, corregir el rumbo hacia la largada.
                if pose is not None and self.ahora() - self.ultima_revision_regreso >= self.periodo_regreso:
                    self.ultima_revision_regreso = self.ahora()
                    rumbo = self.rumbo_regreso(rangos, rel, rmin, pose)
                    if rumbo is not None and abs(rumbo) > math.radians(self.tolerancia_rumbo_regreso_deg):
                        self.yaw_objetivo = normalizar(pose[2] + rumbo)
                        self.sentido_giro = 1 if rumbo >= 0 else -1
                        self.cambiar(GIRAR, f'volviendo, rumbo {math.degrees(rumbo):+.0f}°')
            elif self.debe_barrer(pose, rangos, rel, rmin):
                self.ultimo_barrido = pose[:2]
                self.cambiar(BARRER, 'zona con mucho sin ver')
            elif self.nada_nuevo_adelante(pose):
                self.elegir_giro(rangos, rel, rmin, 'nada nuevo adelante')

        elif self.estado == GIRAR:
            if self.yaw_objetivo is not None and pose is not None:
                listo = abs(normalizar(self.yaw_objetivo - pose[2])) < 0.1
            elif self.girar_hacia_libre:
                listo = (d_frente > self.distancia_choque_m + 0.2 and self.angulo_girado() > 0.25)
            else:
                listo = self.angulo_girado() > self.angulo_giro
            if listo or self.angulo_girado() > 2 * math.pi:
                self.ultima_reevaluacion = self.ahora()
                self.cambiar(AVANZAR, 'giro terminado')

        elif self.estado == BARRER:
            if self.angulo_girado() >= 2 * math.pi:
                self.ultima_reevaluacion = self.ahora()
                self.elegir_giro(rangos, rel, rmin, 'barrido terminado')

        elif self.estado == ACERCARSE:
            gx, gy = self.objetivo_rojo
            confirmada = any(math.hypot(gx - cx, gy - cy) < self.radio_confirmada
                             for cx, cy in self.confirmadas)
            vencido = self.ahora() - self.inicio_acercamiento > self.tiempo_max_acercamiento
            if confirmada or vencido or pose is None or self.volviendo:
                if not confirmada:
                    self.descartados.append((gx, gy, self.ahora() + 30.0))
                self.cambiar(AVANZAR, 'confirmada' if confirmada else 'acercamiento abandonado')

        # ----- acción -----
        if self.estado == AVANZAR:
            # Lo que está a la altura del robot, de la cola a un poco más allá de la trompa:
            # incluye las diagonales contra las esquinas delanteras.
            x, y = puntos_xy(rangos, rel, rmin)
            d_izq, d_der = distancias_laterales(
                x, y, -self.medio_largo_robot, self.medio_largo_robot + self.anticipacion_lateral)
            msg.linear.x = velocidad_avance(
                d_frente, d_izq, d_der, self.distancia_choque_m, self.distancia_lenta,
                self.velocidad_adelante, self.velocidad_rapida,
                self.abierto_lateral_desde, self.abierto_lateral_hasta,
                self.abierto_frente_desde, self.abierto_frente_hasta)
            if self.girar_hacia_libre:
                msg.linear.y = velocidad_lateral(d_izq, d_der, self.margen_lateral,
                                                 self.velocidad_lateral, self.ancho_pasillo,
                                                 self.ganancia_centrado)
        elif self.estado in (GIRAR, BARRER):
            sentido = 1 if self.estado == BARRER else self.sentido_giro
            if self.estado == GIRAR and self.yaw_objetivo is not None and pose is not None:
                sentido = 1 if normalizar(self.yaw_objetivo - pose[2]) >= 0 else -1
            msg.angular.z = sentido * self.velocidad_de_giro()
        elif self.estado == ACERCARSE:
            gx, gy = self.objetivo_rojo
            rumbo = normalizar(math.atan2(gy - pose[1], gx - pose[0]) - pose[2])
            distancia = math.hypot(gx - pose[0], gy - pose[1])
            # Mirando un objeto rojo: siempre a la velocidad lenta, para no perder la detección.
            v_giro = self.velocidad_angular_rojo
            msg.angular.z = max(-v_giro, min(v_giro, 1.5 * rumbo))
            if abs(rumbo) < 0.35 and distancia > self.distancia_acercamiento and not obstaculo:
                msg.linear.x = min(self.velocidad_adelante, 0.5 * (distancia - self.distancia_acercamiento) + 0.05)

        self.publisher_.publish(msg)

    def debe_barrer(self, pose, rangos, rel, rmin) -> bool:
        if not (self.barridos and pose is not None and self.cobertura is not None):
            return False
        if self.ultimo_barrido is not None and math.hypot(
                pose[0] - self.ultimo_barrido[0], pose[1] - self.ultimo_barrido[1]) < self.distancia_entre_barridos:
            return False
        validos = np.isfinite(rangos) & (rangos > rmin)
        if validos.any() and rangos[validos].min() < self.espacio_para_barrer:
            return False   # girar en el lugar rozaría algo
        return self.cobertura.no_visto_alrededor(pose[0], pose[1], self.radio_barrido) > self.fraccion_para_barrer

    def nada_nuevo_adelante(self, pose) -> bool:
        if not (self.explorar_con_memoria and pose is not None and self.cobertura is not None):
            return False
        if self.ahora() - self.ultima_reevaluacion < self.periodo_reevaluar:
            return False
        self.ultima_reevaluacion = self.ahora()
        adelante = self.cobertura.no_visto_hacia(pose[0], pose[1], pose[2], math.radians(30),
                                                 self.alcance_camara)
        if adelante >= self.no_visto_minimo:
            return False
        # Adelante ya está visto: cambiar solo si en otra dirección queda bastante sin ver.
        mejor_otra = max(self.cobertura.no_visto_hacia(pose[0], pose[1], pose[2] + math.radians(a),
                                                       math.radians(30), self.alcance_camara)
                         for a in range(-150, 181, 30) if a != 0)
        return mejor_otra >= adelante + self.mejora_para_cambiar


def main(args=None):
    rclpy.init(args=args)
    nodo = Evasor()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        nodo.publisher_.publish(Twist())
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
