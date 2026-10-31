import math

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import (Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped,
                               Quaternion, TransformStamped)

from tf2_ros import TransformBroadcaster, Buffer, TransformListener


def normalizar_angulo(angulo: float) -> float:
    """Lleva un ángulo cualquiera al rango (-pi, pi]."""
    return math.atan2(math.sin(angulo), math.cos(angulo))


def yaw_a_quaternion(yaw: float) -> Quaternion:
    """Quaternion 2D (solo rotación en Z) a partir de un ángulo yaw."""
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


def invertir_pose(pose):
    """Inversa de una pose 2D (x, y, theta) como transformada rígida."""
    x, y, theta = pose
    c, s = math.cos(-theta), math.sin(-theta)
    return (-(c * x - s * y), -(s * x + c * y), -theta)


def componer_poses(pose_a, pose_b):
    """pose_a * pose_b: aplicar primero b, después a. Se usa para pasar de
    "map→base" y "odom→base" a "map→odom" (ver publicar_transformada)."""
    xa, ya, ta = pose_a
    xb, yb, tb = pose_b
    c, s = math.cos(ta), math.sin(ta)
    return (xa + c * xb - s * yb, ya + s * xb + c * yb, normalizar_angulo(ta + tb))


class Localizador(Node):

    def __init__(self):
        super().__init__('localizador')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('num_particulas', 300)
        self.declare_parameter('pose_inicial_x', 0.0)
        self.declare_parameter('pose_inicial_y', 0.0)
        self.declare_parameter('pose_inicial_theta', 0.0)
        self.declare_parameter('dispersion_inicial_xy', 0.3)
        self.declare_parameter('dispersion_inicial_theta', 0.3)
        # Ruido del modelo de movimiento odométrico mecanum. Son las seis
        # gammas de la tabla: gamma1 escala el ruido de cada eje de
        # traslación sobre sí mismo, gamma2 el acoplamiento entre los dos
        # ejes, gamma3 cuánto ruido de traslación mete el haber rotado,
        # gamma4/gamma5 cuánto ruido de rotación meten el avance y el
        # strafe, y gamma6 la rotación sobre sí misma.
        # OJO: son VARIANZAS, no desvíos (por eso los deltas van al
        # cuadrado en la fórmula). Están sin calibrar a propósito: ajustarlas
        # es parte del desafío extra.
        self.declare_parameter('gamma1', 0.0025)
        self.declare_parameter('gamma2', 0.0025)
        self.declare_parameter('gamma3', 0.0025)
        self.declare_parameter('gamma4', 0.0025)
        self.declare_parameter('gamma5', 0.0025)
        self.declare_parameter('gamma6', 0.0025)
        # Cada cuántos rayos del /scan (de los 1080 totales) se usa para
        # pesar las partículas. Bajarlo (usar más rayos) es más preciso pero
        # más lento; subirlo, al revés.
        self.declare_parameter('submuestreo_scan', 15)

        # Parametros de Escala
        self.declare_parameter('escala_avance',1.0)
        self.declare_parameter('escala_giro',1.0)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.num_particulas = self.get_parameter('num_particulas').value
        self.gamma1 = self.get_parameter('gamma1').value
        self.gamma2 = self.get_parameter('gamma2').value
        self.gamma3 = self.get_parameter('gamma3').value
        self.gamma4 = self.get_parameter('gamma4').value
        self.gamma5 = self.get_parameter('gamma5').value
        self.gamma6 = self.get_parameter('gamma6').value
        self.submuestreo_scan = self.get_parameter('submuestreo_scan').value
        self.escala_avance = self.get_parameter('escala_avance').value
        self.escala_giro = self.get_parameter('escala_giro').value

        self.get_logger().info(f'Started Localizador with: {self.escala_avance} escala_avance and {self.escala_giro} escala_giro')

        # Estado del filtro: array (num_particulas, 4) con columnas
        # (x, y, theta, peso). Arranca como una nube gaussiana angosta
        # alrededor de la pose inicial conocida (no es localización global).
        n = self.num_particulas
        x0 = np.random.normal(
            self.get_parameter('pose_inicial_x').value,
            self.get_parameter('dispersion_inicial_xy').value, n)
        y0 = np.random.normal(
            self.get_parameter('pose_inicial_y').value,
            self.get_parameter('dispersion_inicial_xy').value, n)
        theta0 = np.random.normal(
            self.get_parameter('pose_inicial_theta').value,
            self.get_parameter('dispersion_inicial_theta').value, n)
        self.particulas = np.stack([x0, y0, theta0, np.full(n, 1.0 / n)], axis=1)

        self.campo = None
        self.info_mapa = None
        self.ultimo_odom = None
        self.ultimo_scan = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        qos_mapa = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, 'likelihood_map', self.recibir_campo, qos_mapa)
        # El /scan va best effort (qos_profile_sensor_data, ver semana 03);
        # /odom y /ground_truth/odom son reliable y van con el default.
        self.create_subscription(LaserScan, 'scan', self.recibir_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, 'odom', self.recibir_odom, 10)
        self.create_subscription(Odometry, 'ground_truth/odom', self.recibir_ground_truth, 10)
        # "2D Pose Estimate" de RViz publica acá (semana 08).
        self.create_subscription(
            PoseWithCovarianceStamped, 'initialpose', self.recibir_pose_inicial, 10)

        self.pub_particulas = self.create_publisher(PoseArray, 'particlecloud', 10)
        self.pub_camino_odom = self.create_publisher(Path, 'camino_odom', 10)
        self.pub_camino_corregido = self.create_publisher(Path, 'camino_corregido', 10)
        self.pub_camino_real = self.create_publisher(Path, 'camino_real', 10)

        # frame_id=map por consistencia visual en RViz con los otros dos
        # caminos, aunque abajo se le cargan poses crudas de /odom (frame
        # odom): solo es correcto porque pose_inicial_x/y/theta arrancan en
        # (0, 0, 0), o sea que map y odom coinciden en t=0. Si cambiás esos
        # parámetros iniciales, este camino queda desalineado del real.
        self.camino_odom_msg = Path()
        self.camino_odom_msg.header.frame_id = self.map_frame
        self.camino_corregido_msg = Path()
        self.camino_corregido_msg.header.frame_id = self.map_frame
        self.camino_real_msg = Path()
        self.camino_real_msg.header.frame_id = self.map_frame

    # ---------- callbacks ----------
    # Notá que acá nos apartamos de la convención de semanas anteriores de
    # "el callback de un sensor solo guarda el dato, un timer decide": un
    # filtro de partículas es naturalmente una máquina predict/correct
    # disparada por eventos — predecir en cada /odom, corregir en cada
    # /scan — y forzar un timer aparte no lo haría más simple, solo menos
    # fiel al algoritmo real.

    def recibir_pose_inicial(self, msg: PoseWithCovarianceStamped):
        """Vuelve a sembrar la nube alrededor de la pose que marcó RViz con
        "2D Pose Estimate" (misma cuenta que el __init__)."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.yaw_de_quaternion(msg.pose.pose.orientation)
        n = self.num_particulas
        dxy = self.get_parameter('dispersion_inicial_xy').value
        dtheta = self.get_parameter('dispersion_inicial_theta').value
        x0 = np.random.normal(x, dxy, n)
        y0 = np.random.normal(y, dxy, n)
        theta0 = np.random.normal(theta, dtheta, n)
        self.particulas = np.stack([x0, y0, theta0, np.full(n, 1.0 / n)], axis=1)

    def recibir_campo(self, msg: OccupancyGrid):
        if self.campo is None:
            alto, ancho = msg.info.height, msg.info.width
            self.campo = np.array(msg.data, dtype=np.float64).reshape((alto, ancho))
            self.info_mapa = msg.info
            self.get_logger().info('Campo de verosimilitud recibido.')

    def recibir_ground_truth(self, msg: Odometry):
        """Solo para comparar visualmente en RViz -- Gazebo publica la pose
        real del robot en `/ground_truth/odom` (mismo frame que `map`, ver
        README). Nunca se usa dentro del filtro, ni existiría en el robot
        físico: es una ventaja exclusiva del simulador para verificar qué
        tan bien está corrigiendo el filtro."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.yaw_de_quaternion(msg.pose.pose.orientation)
        self.acumular_camino(self.camino_real_msg, self.pub_camino_real, x, y, theta)

    def recibir_odom(self, msg: Odometry):
        """Predicción: calcula el desplazamiento y la rotación desde la
        última odometría y se los pasa a mover_particulas()."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        theta = self.yaw_de_quaternion(msg.pose.pose.orientation)

        if self.ultimo_odom is not None:
            x0, y0, theta0 = self.ultimo_odom
            dx, dy = x - x0, y - y0
            dtheta = normalizar_angulo(theta - theta0)

            self.particulas = self.mover_particulas(self.particulas, dx, dy, dtheta, theta0)
            self.publicar_particulas()
            self.acumular_camino(self.camino_odom_msg, self.pub_camino_odom, x, y, theta)

        self.ultimo_odom = (x, y, theta)

    def recibir_scan(self, msg: LaserScan):
        """Corrección: pesa las partículas contra el scan actual, resamplea,
        estima la pose y publica la transformada map→odom."""
        self.ultimo_scan = msg
        if self.campo is None or self.ultimo_odom is None:
            return

        puntos = self.scan_a_puntos(msg)
        if puntos is None or len(puntos) == 0:
            return

        pesos = self.pesar_particulas(self.particulas, puntos)
        self.particulas = self.remuestrear(self.particulas, pesos)
        self.publicar_particulas()

        pose_estimada = self.estimar_pose(self.particulas)
        self.acumular_camino(
            self.camino_corregido_msg, self.pub_camino_corregido, *pose_estimada
        )
        self.publicar_transformada(pose_estimada)

    # ---------- plomería resuelta ----------

    def yaw_de_quaternion(self, q: Quaternion) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def scan_a_puntos(self, msg: LaserScan):
        """Convierte el LaserScan (polar, en el frame del lidar) a un array
        Nx2 de puntos (x, y) en base_frame, submuestreado. Usa tf2 para la
        transformada lidar→base — mismo patrón que ya usaste en
        detector_scan.py de semana 04, no hay que medir nada a mano."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except Exception:
            return None

        rangos = np.array(msg.ranges)
        angulos = msg.angle_min + np.arange(len(rangos)) * msg.angle_increment

        rangos = rangos[:: self.submuestreo_scan]
        angulos = angulos[:: self.submuestreo_scan]

        validos = np.isfinite(rangos) & (rangos >= msg.range_min) & (rangos <= msg.range_max)
        rangos, angulos = rangos[validos], angulos[validos]

        xs_laser = rangos * np.cos(angulos)
        ys_laser = rangos * np.sin(angulos)

        tx, ty = tf.transform.translation.x, tf.transform.translation.y
        tyaw = self.yaw_de_quaternion(tf.transform.rotation)
        c, s = math.cos(tyaw), math.sin(tyaw)

        xs = tx + c * xs_laser - s * ys_laser
        ys = ty + s * xs_laser + c * ys_laser
        return np.stack([xs, ys], axis=1)

    def estimar_pose(self, particulas):
        """Media ponderada de x/y + media circular de theta sobre el set de
        partículas ya resampleado."""
        x = float(np.mean(particulas[:, 0]))
        y = float(np.mean(particulas[:, 1]))
        theta = float(math.atan2(
            np.mean(np.sin(particulas[:, 2])), np.mean(np.cos(particulas[:, 2]))
        ))
        return (x, y, theta)

    def publicar_particulas(self):
        msg = PoseArray()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, theta, _ in self.particulas:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.orientation = yaw_a_quaternion(float(theta))
            msg.poses.append(pose)
        self.pub_particulas.publish(msg)

    def acumular_camino(self, camino_msg: Path, publisher, x, y, theta):
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = self.map_frame
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.pose.position.x = float(x)
        pose_stamped.pose.position.y = float(y)
        pose_stamped.pose.orientation = yaw_a_quaternion(float(theta))
        camino_msg.poses.append(pose_stamped)
        camino_msg.header.stamp = pose_stamped.header.stamp
        publisher.publish(camino_msg)

    def publicar_transformada(self, pose_estimada):
        """map→odom = (map→base) compuesta con la inversa de (odom→base) —
        así, RViz (y cualquier otro nodo) puede seguir usando la odometría
        cruda de siempre y obtener una pose corregida en map "gratis"
        aplicando esta transformada."""
        pose_map_odom = componer_poses(pose_estimada, invertir_pose(self.ultimo_odom))

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = pose_map_odom[0]
        t.transform.translation.y = pose_map_odom[1]
        t.transform.rotation = yaw_a_quaternion(pose_map_odom[2])
        self.tf_broadcaster.sendTransform(t)

    # ---------- el filtro de partículas ----------

    def mover_particulas(self, particulas, dx, dy, dtheta, theta0):
        """
        TODO: modelo de movimiento odométrico **mecanum**
        (`sample_motion_model_odometry`, el de la lámina de la clase),
        aplicado a las N partículas a la vez con numpy. Devolvé un array
        nuevo (mismo shape que particulas).

        La idea: el robot es holonómico, así que el desplazamiento se
        describe con tres números en el marco del propio robot —cuánto
        avanzó, cuánto se deslizó de costado, cuánto rotó— y cada uno lleva
        su ruido. No hay que inventar rotaciones que el robot no hizo.

        Pasos sugeridos:
          1. Pasar el desplazamiento (dx, dy), que viene en el frame `odom`,
             al frame que tenía el robot al empezar el paso (theta0):
               d_trans_x =  dx*cos(theta0) + dy*sin(theta0)   # adelante
               d_trans_y = -dx*sin(theta0) + dy*cos(theta0)   # de costado
               d_rot     =  dtheta                            # lo que rotó
             (`d_rot` es la rotación REAL, ya te la pasan calculada.)
          2. Calcular la varianza del ruido de cada componente. Ojo que
             `self.gamma1..6` son **varianzas**, así que los deltas van al
             cuadrado y el desvío es la raíz de la suma:
               std_x   = sqrt(gamma1*d_trans_x² + gamma2*d_trans_y² + gamma3*d_rot²)
               std_y   = sqrt(gamma2*d_trans_x² + gamma1*d_trans_y² + gamma3*d_rot²)
               std_rot = sqrt(gamma4*d_trans_x² + gamma5*d_trans_y² + gamma6*d_rot²)
             (Notá que en `std_y` gamma1 y gamma2 aparecen al revés que en
             `std_x`: gamma1 es siempre el eje propio y gamma2 el cruzado.)
          3. Restarle a cada componente una muestra de ruido gaussiano con
             ese desvío: np.random.normal(0.0, std, n) — un array de N
             muestras por componente, no un solo valor.
          4. Componer el desplazamiento ruidoso sobre la pose de cada
             partícula, rotándolo por el theta de ESA partícula:
               x'     = x + d_trans_x_hat*cos(theta) - d_trans_y_hat*sin(theta)
               y'     = y + d_trans_x_hat*sin(theta) + d_trans_y_hat*cos(theta)
               theta' = theta + d_rot_hat
             (normalizá theta' al rango (-pi, pi] con np.arctan2(sin, cos)).
          5. Devolver un array nuevo con esas x', y', theta' (y el mismo
             peso que tenían, no se toca acá).
        """
        n = len(particulas)

        # 1. Desplazamiento en el frame del robot al inicio del paso.
        c0, s0 = math.cos(theta0), math.sin(theta0)
        d_trans_x = dx * c0 + dy * s0
        d_trans_y = -dx * s0 + dy * c0
        d_rot = dtheta

        # 2. Desvío del ruido de cada componente (las gammas son varianzas).
        std_x = math.sqrt(self.gamma1 * d_trans_x ** 2 + self.gamma2 * d_trans_y ** 2
                          + self.gamma3 * d_rot ** 2)
        std_y = math.sqrt(self.gamma2 * d_trans_x ** 2 + self.gamma1 * d_trans_y ** 2
                          + self.gamma3 * d_rot ** 2)
        std_rot = math.sqrt(self.gamma4 * d_trans_x ** 2 + self.gamma5 * d_trans_y ** 2
                            + self.gamma6 * d_rot ** 2)

        # 3. Una muestra de ruido distinta por partícula.
        d_trans_x_hat = d_trans_x - np.random.normal(0.0, std_x, n)
        d_trans_y_hat = d_trans_y - np.random.normal(0.0, std_y, n)
        d_rot_hat = d_rot - np.random.normal(0.0, std_rot, n)

        # 4. Componer sobre la pose de cada partícula, con su propio theta.
        x, y, theta = particulas[:, 0], particulas[:, 1], particulas[:, 2]
        c, s = np.cos(theta), np.sin(theta)
        x_nueva = x + (d_trans_x_hat * c - d_trans_y_hat * s) * self.escala_avance
        y_nueva = y + (d_trans_x_hat * s + d_trans_y_hat * c) * self.escala_avance
        theta_nueva = theta + d_rot_hat * self.escala_giro
        theta_nueva = np.arctan2(np.sin(theta_nueva), np.cos(theta_nueva))

        # 5. Array nuevo, mismo peso.
        return np.stack([x_nueva, y_nueva, theta_nueva, particulas[:, 3]], axis=1)

    def pesar_particulas(self, particulas, puntos_scan):
        """
        TODO: el modelo de sensor. Para cada partícula, ver qué tan bien
        calzan los puntos del scan (ya en base_frame, submuestreados) contra
        el campo de verosimilitud, y devolver un array de N pesos
        normalizados (que sumen 1).

        Pasos sugeridos (todo vectorizado: pensalo como una matriz de
        N partículas × M puntos del scan):
          1. Transformar los M puntos de puntos_scan al frame map, una vez
             por cada una de las N partículas (rotar por el theta de la
             partícula, trasladar por su x/y). Te va a quedar un array
             (N, M) de coordenadas x y otro (N, M) de coordenadas y —
             usá broadcasting: particulas[:, 0][:, None] + ... con
             puntos_scan[:, 0][None, :].
          2. Convertir esas coordenadas (x, y) en metros a índices
             [fila, columna] de self.campo, usando self.info_mapa.resolution
             y self.info_mapa.origin.position.x/y (misma cuenta que hizo
             campo_verosimilitud.py al revés: índice = (coord - origen) /
             resolución).
          3. Descartar (o asignarles una probabilidad mínima, ej. 1.0) los
             índices que caen fuera de self.campo.shape.
          4. Para los índices válidos, leer self.campo[fila, columna] — es
             la probabilidad (0-100) de ese punto según esa partícula.
          5. Combinar las M probabilidades de cada partícula en un solo
             peso. Sugerencia: sumá los logaritmos (más estable
             numéricamente que multiplicar probabilidades chicas) y restale
             el máximo antes de exponenciar, para evitar underflow:
               log_pesos -= log_pesos.max()
               pesos = exp(log_pesos)
          6. Normalizar: pesos /= pesos.sum() (si la suma da 0, devolvé
             pesos uniformes 1/N en su lugar, para no dividir por cero).
        """
        n = len(particulas)

        # 1. Puntos del scan en map, una vez por partícula: arrays (N, M).
        px = particulas[:, 0][:, None]
        py = particulas[:, 1][:, None]
        c = np.cos(particulas[:, 2])[:, None]
        s = np.sin(particulas[:, 2])[:, None]
        sx = puntos_scan[:, 0][None, :]
        sy = puntos_scan[:, 1][None, :]
        xs_map = px + c * sx - s * sy
        ys_map = py + s * sx + c * sy

        # 2. Metros -> índices de la grilla (fila = y, columna = x).
        resolucion = self.info_mapa.resolution
        origen_x = self.info_mapa.origin.position.x
        origen_y = self.info_mapa.origin.position.y
        columnas = np.floor((xs_map - origen_x) / resolucion).astype(int)
        filas = np.floor((ys_map - origen_y) / resolucion).astype(int)

        # 3. Los puntos fuera del mapa reciben probabilidad mínima (1.0).
        alto, ancho = self.campo.shape
        dentro = (filas >= 0) & (filas < alto) & (columnas >= 0) & (columnas < ancho)
        probabilidades = np.full(xs_map.shape, 1.0)

        # 4. Leer el campo en los índices válidos. El piso de 1.0 evita log(0)
        # en celdas muy alejadas de toda pared.
        probabilidades[dentro] = self.campo[filas[dentro], columnas[dentro]]
        probabilidades = np.maximum(probabilidades, 1.0)

        # 5. Sumar logaritmos y restar el máximo antes de exponenciar.
        log_pesos = np.sum(np.log(probabilidades / 100.0), axis=1)
        log_pesos -= log_pesos.max()
        pesos = np.exp(log_pesos)

        # 6. Normalizar.
        suma = pesos.sum()
        if suma <= 0.0 or not np.isfinite(suma):
            return np.full(n, 1.0 / n)
        return pesos / suma

    def remuestrear(self, particulas, pesos):
        """
        TODO: resampling sistemático (de varianza baja). Dado el array de
        pesos normalizados, devolvé un array nuevo de partículas (mismo
        shape que particulas) redibujado con reemplazo, proporcional al
        peso — las partículas con más peso van a aparecer más veces, las de
        peso ínfimo capaz ninguna.

        Pasos sugeridos:
          1. Elegir un único offset aleatorio en [0, 1): np.random.uniform().
          2. Armar N "pasos" equiespaciados a partir de ese offset:
             (np.arange(N) + offset) / N — es la parte que hace que el
             muestreo sea sistemático en vez de N tiradas independientes.
          3. Calcular la suma acumulada de los pesos (np.cumsum) — fijate de
             forzar el último valor a 1.0 por errores de redondeo.
          4. Para cada paso, buscar en qué posición de esa suma acumulada
             caería (np.searchsorted) — ese índice es la partícula elegida.
          5. Devolver particulas[indices] (copia nueva), reseteando el peso
             de todas a 1/N (ya cumplieron su función al elegir a quién
             copiar).
        """
        n = len(particulas)

        # 1-2. Un solo offset aleatorio y N pasos equiespaciados.
        offset = np.random.uniform()
        pasos = (np.arange(n) + offset) / n

        # 3. Suma acumulada, con el último valor forzado a 1.0.
        acumulada = np.cumsum(pesos)
        acumulada[-1] = 1.0

        # 4. Índice de la partícula elegida para cada paso.
        indices = np.searchsorted(acumulada, pasos)

        # 5. Copia nueva con pesos uniformes.
        nuevas = particulas[indices].copy()
        nuevas[:, 3] = 1.0 / n
        return nuevas


def main(args=None):
    rclpy.init(args=args)
    nodo = Localizador()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
