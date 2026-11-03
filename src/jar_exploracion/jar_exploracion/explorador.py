import math
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import scipy.ndimage as nd

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from rclpy.impl.implementation_singleton import rclpy_implementation
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time

from nav_msgs.msg import OccupancyGrid, Path
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist, Quaternion, PoseStamped, PoseWithCovarianceStamped
from std_msgs.msg import Empty, Float32, String
from visualization_msgs.msg import Marker

from tf2_ros import Buffer, TransformListener

from jar_exploracion.mirada import (MiradaPorPasos, eje_mas_cercano, giro_seguro, normalizar,
                                    radio_de_giro)
from jar_exploracion.planificador import PlanificadorCompilado, segmento_transitable
from jar_exploracion.trazado import celdas_visibles
from jar_exploracion.seguridad import (buscar_desvio, celda_libre_cercana, escape_seguro,
                                       espacio_transitable, puntos_scan, regular_traslacion)

FRECUENCIA_HZ = 10.0
HOLGURA_MIRADA = 0.02
ESPERA_MIRADA_S = 3.0
# Sin destinos durante este tiempo, se vuelve al origen (y se reportan las
# víctimas) en lugar de quedar detenido con zonas inalcanzables.
ESPERA_SIN_DESTINOS_S = 10.0
# Hasta este desvío (rad) respecto de un eje del mapa, la huella se calcula
# alineada y el control corrige el giro: la deriva de las ruedas no cambia
# el espacio libre en cada ciclo.
TOLERANCIA_EJE = 0.1
# Holguras (m) alrededor de la huella con las que se mide el costo de A*:
# sin holgura cuesta 100 por celda; con la última, nada.
HOLGURAS_COSTO = (0.04, 0.08, 0.12, 0.16)
# Veces que una zona puede quedar sin punto de observación alcanzable antes
# de descartarla: si el que la tapa es el scan, se libera; si son paredes, no.
INTENTOS_ZONA_INALCANZABLE = 3

ESTADO_INICIALIZANDO = 'inicializando'
ESTADO_EXPLORANDO = 'explorando'
ESTADO_REGRESANDO = 'regresando_al_origen'
ESTADO_TERMINADO = 'terminado'
ESTADO_SIN_DESTINOS = 'sin_destinos_transitables'


def yaw_de_quaternion(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class Explorador(Node):
    """
    Explora con objetivos persistentes, A* y traslación validada con huella rectangular.

    La huella conserva la orientación estimada actual; los cambios de orientación
    invalidan el camino. No se ordenan rotaciones ni se supone que el robot pueda
    girar dentro de un estrechamiento. La cobertura la confirma el scan real.
    """

    def __init__(self):
        super().__init__('explorador')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('confirmar_pose_automatica', False)
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('velocidad_maxima', 0.3)
        self.declare_parameter('ganancia_atractiva', 0.6)
        self.declare_parameter('suavizado', 0.3)
        self.declare_parameter('tiempo_asentamiento_s', 3.0)
        # Las dimensiones incluyen ruedas y el desplazamiento del chasis respecto de la base.
        self.declare_parameter('margen_seguridad', 0.05)
        self.declare_parameter('timeout_scan_s', 0.5)
        self.declare_parameter('reintento_objetivo_s', 10.0)
        self.declare_parameter('timeout_sin_avance_s', 15.0)
        self.declare_parameter('largo_robot', 0.37)
        self.declare_parameter('ancho_robot', 0.24)
        self.declare_parameter('timeout_pose_s', 0.5)
        self.declare_parameter('velocidad_pasillo', 0.10)
        self.declare_parameter('horizonte_colision_s', 0.8)
        self.declare_parameter('desaceleracion_segura', 0.4)
        self.declare_parameter('subdivisiones_mapa', 2)
        self.declare_parameter('porcentaje_objetivo', 98.0)
        # Giro de 360° por pasos para que la cámara mire alrededor; 0 lo desactiva.
        self.declare_parameter('distancia_entre_miradas', 1.5)
        self.declare_parameter('pasos_mirada', 8)
        self.declare_parameter('pausa_mirada_s', 0.6)
        self.declare_parameter('velocidad_mirada', 0.6)
        # Impactos del scan a menos de esto de una pared del mapa son esa pared
        # vista con error de localización, no un obstáculo nuevo.
        self.declare_parameter('distancia_pared_conocida', 0.10)
        # Margen para obstáculos que solo ve el scan (cajas). Las paredes del mapa,
        # dibujadas más gruesas que las reales, conservan margen_seguridad.
        self.declare_parameter('margen_obstaculos_scan', 0.02)
        # Solo impactos cercanos son obstáculos temporales: lejos, unos grados de
        # error de orientación corren la pared conocida más que distancia_pared_conocida.
        self.declare_parameter('alcance_obstaculos_scan', 1.0)
        # Scans distintos que tienen que ver una celda para tratarla como obstáculo:
        # un impacto suelto (ruido, error de pose) no se repite; una caja, sí.
        self.declare_parameter('impactos_minimos_scan', 1)
        # Si el robot queda dentro del margen de una pared, se aleja despacio
        # hasta la celda libre más cercana dentro de este radio.
        self.declare_parameter('radio_escape', 0.20)
        # Reglamento: corrida de 10 min desde /map. Pasado este tiempo se reportan
        # las víctimas y se vuelve a la largada (0 lo desactiva).
        self.declare_parameter('tiempo_regreso_s', 0.0)
        # Con las N víctimas reportadas no queda nada que buscar: volver y /done.
        self.declare_parameter('regresar_con_todas', True)
        # El jurado corta la corrida tras 60 s sin moverse: antes, una mirada.
        self.declare_parameter('espera_inmovil_s', 40.0)
        self.declare_parameter('velocidad_escape', 0.06)

        self.largo_robot = self.get_parameter('largo_robot').value
        self.ancho_robot = self.get_parameter('ancho_robot').value
        self.timeout_pose_s = self.get_parameter('timeout_pose_s').value
        self.velocidad_pasillo = self.get_parameter('velocidad_pasillo').value
        self.horizonte_colision_s = self.get_parameter('horizonte_colision_s').value
        self.desaceleracion_segura = self.get_parameter('desaceleracion_segura').value
        self.subdivisiones_mapa = self.get_parameter('subdivisiones_mapa').value
        self.porcentaje_objetivo = self.get_parameter('porcentaje_objetivo').value
        if not math.isfinite(self.porcentaje_objetivo) or not 0 < self.porcentaje_objetivo <= 100:
            raise ValueError('porcentaje_objetivo debe ser mayor que 0 y como máximo 100')
        self.total_celdas_libres = 0
        self.celdas_cubiertas = 0
        if self.subdivisiones_mapa not in (1, 2, 3, 4):
            raise ValueError('subdivisiones_mapa debe estar entre 1 y 4')
        self.yaw_huella = 0.0
        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.velocidad_maxima = self.get_parameter('velocidad_maxima').value
        self.ganancia_atractiva = self.get_parameter('ganancia_atractiva').value
        # Peso del comando nuevo contra el anterior (media móvil exponencial),
        # para que el resultado de seguir el camino no salte bruscamente de
        # un waypoint a otro.
        self.suavizado = self.get_parameter('suavizado').value
        self.tiempo_asentamiento_s = self.get_parameter('tiempo_asentamiento_s').value

        self.confirmar_pose_automatica = self.get_parameter('confirmar_pose_automatica').value
        self.margen_seguridad = self.get_parameter('margen_seguridad').value
        self.timeout_scan_s = self.get_parameter('timeout_scan_s').value
        self.reintento_objetivo_s = self.get_parameter('reintento_objetivo_s').value
        self.timeout_sin_avance_s = self.get_parameter('timeout_sin_avance_s').value
        positivos = (self.largo_robot, self.ancho_robot, self.timeout_pose_s,
                     self.velocidad_maxima, self.ganancia_atractiva,
                     self.timeout_sin_avance_s,
                     self.velocidad_pasillo, self.horizonte_colision_s,
                     self.desaceleracion_segura)
        if not all(math.isfinite(v) and v > 0 for v in positivos):
            raise ValueError('Dimensiones, velocidades, ganancias y tiempos deben ser positivos')
        if not math.isfinite(self.margen_seguridad) or self.margen_seguridad < 0:
            raise ValueError('margen_seguridad debe ser finito y no negativo')
        if self.timeout_sin_avance_s <= 0:
            raise ValueError('timeout_sin_avance_s debe ser positivo')
        self.seguimiento_avance = None
        self.velocidad_ordenada = 0.0
        if self.timeout_scan_s <= 0 or self.reintento_objetivo_s <= 0:
            raise ValueError('Los tiempos de scan y reintento deben ser positivos')
        if not 0.0 < self.suavizado <= 1.0:
            raise ValueError('suavizado debe estar entre 0 (excluido) y 1')
        self.vx_anterior = 0.0
        self.vy_anterior = 0.0
        self.origen = None
        self.regreso_pendiente = False
        self.distancia_entre_miradas = self.get_parameter('distancia_entre_miradas').value
        self.distancia_pared_conocida = self.get_parameter('distancia_pared_conocida').value
        self.margen_obstaculos_scan = self.get_parameter('margen_obstaculos_scan').value
        self.alcance_obstaculos_scan = self.get_parameter('alcance_obstaculos_scan').value
        self.impactos_minimos_scan = self.get_parameter('impactos_minimos_scan').value
        self.radio_escape = self.get_parameter('radio_escape').value
        self.tiempo_regreso_s = self.get_parameter('tiempo_regreso_s').value
        self.regresar_con_todas = self.get_parameter('regresar_con_todas').value
        self.espera_inmovil_s = self.get_parameter('espera_inmovil_s').value
        self.momento_mapa = None
        self.motivo_regreso = None
        self.done_publicado = False
        self.momento_movimiento = None
        self.forzar_mirada = False
        self.velocidad_escape = self.get_parameter('velocidad_escape').value
        if self.distancia_pared_conocida < 0 or self.margen_obstaculos_scan < 0:
            raise ValueError(
                'distancia_pared_conocida y margen_obstaculos_scan no pueden ser negativos')
        self.mirada = MiradaPorPasos(
            pasos=self.get_parameter('pasos_mirada').value,
            pausa_s=self.get_parameter('pausa_mirada_s').value,
            velocidad=self.get_parameter('velocidad_mirada').value)
        if self.distancia_entre_miradas < 0 or self.mirada.pasos < 1:
            raise ValueError('distancia_entre_miradas y pasos_mirada no pueden ser negativos')
        self.radio_giro = radio_de_giro(
            self.largo_robot, self.ancho_robot, self.margen_seguridad)
        # Sin vuelta al empezar: la pose inicial ya la fija el filtro de
        # partículas, y la primera mirada llega después del primer tramo.
        self.recorrido_desde_mirada = 0.0
        # Ya girando, un impacto que entra hasta HOLGURA_MIRADA en el círculo
        # (ruido del scan en el límite) no corta la vuelta; si entra más, el
        # robot espera quieto hasta ESPERA_MIRADA_S sin perder lo girado.
        self.momento_bloqueo_mirada = None
        # Celdas cubiertas al terminar la última mirada: sin cobertura nueva
        # no hay nada nuevo que mirar y no se gira.
        self.cubiertas_en_ultima_mirada = -1
        self.momento_sin_destinos = None
        self.posicion_ultima_mirada = None
        self.pose_inicial_confirmada = False
        self.momento_confirmacion = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.libre_fino = None      # bool[alto, ancho], resolución del /map: transitable, para A*
        self.costo_fino = None      # float[alto, ancho]: costo extra por cercanía a una pared
        self.info_mapa = None       # MapMetaData del /map (resolución fina)
        self.info_mapa_fuente = None
        self.datos_mapa_fuente = None
        self.no_cubiertas = None    # bool[alto, ancho], resolución de grilla_cobertura
        self.info_grilla = None     # MapMetaData de grilla_cobertura (resolución gruesa)
        self.obstaculos_temporales = {}
        # Scans distintos en que se vio cada celda de obstaculos_temporales.
        self.conteo_impactos = {}
        # Destino del camino vigente (punto de la zona o de observación).
        self.destino_actual = None
        self.scan_integrado = None
        self.ultimo_scan = None
        self.momento_scan = None
        self.objetivos_pendientes = {}
        self.zonas_sin_destino = set()
        self.puntos_observacion = {}
        self.observaciones_fallidas = {}
        self.cache_busquedas = {}
        self.rondas_observacion = {}
        # Zonas cuya búsqueda de punto de observación terminó sin resultado.
        # A diferencia de zonas_sin_destino, no se vacía al cambiar la
        # orientación de la huella: solo con un mapa o una grilla nuevos.
        self.zonas_inobservables = set()
        self.motivo_sin_observacion = None
        self.cache_zonas_alcanzables = {}
        self.estado = ESTADO_INICIALIZANDO

        self.objetivo_celda = None  # (fila, col) al que corresponde self.camino
        self.objetivo_mision = None  # Se conserva hasta cubrir, rechazar o suspender la zona.
        self.camino = None          # lista de (fila, col), en orden, sin recorrer
        self.futuro_plan = None
        self.clave_plan = None
        self.cache_planificador = None

        qos_grilla = QoSProfile(
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, 'map', self.recibir_mapa, qos_grilla)
        self.create_subscription(
            OccupancyGrid, 'grilla_cobertura', self.recibir_grilla, qos_grilla)
        self.create_subscription(LaserScan, 'scan', self.recibir_scan, qos_profile_sensor_data)
        self.create_subscription(
            PoseWithCovarianceStamped, 'initialpose', self.recibir_pose_inicial, 10)

        self.pub_cmd_vel = self.create_publisher(Twist, 'cmd_vel', 10)
        self.pub_objetivo = self.create_publisher(Marker, 'objetivo_actual', 10)
        self.pub_camino = self.create_publisher(Path, 'camino_planificado', 10)
        self.pub_estado = self.create_publisher(String, 'estado', 10)
        # /done (reliable) termina la corrida; con el robot en la largada suma el bono.
        self.pub_done = self.create_publisher(Empty, 'done', 10)
        self.create_subscription(Empty, 'victimas_completas', self.recibir_victimas_completas, 10)
        self.pub_porcentaje = self.create_publisher(Float32, 'porcentaje_cobertura', 10)
        # Modo competencia: el robot manda solo el mismo mensaje que el
        # "2D Pose Estimate" de RViz, cuando ya está todo cargado.
        self.pub_pose_inicial = self.create_publisher(
            PoseWithCovarianceStamped, 'initialpose', 10)
        self.pose_auto_enviada = False
        self.momento_primer_scan = None

        self.ejecutor_plan = ThreadPoolExecutor(max_workers=1, thread_name_prefix='planificador')
        self.timer = self.create_timer(1.0 / FRECUENCIA_HZ, self.controlar)

    # ---------- callbacks ----------

    def recibir_mapa(self, msg: OccupancyGrid):
        alto, ancho = msg.info.height, msg.info.width
        if (msg.header.frame_id != self.map_frame or alto <= 0 or ancho <= 0
                or not math.isfinite(msg.info.resolution) or msg.info.resolution <= 0
                or len(msg.data) != alto * ancho):
            self.get_logger().error('Mapa inválido o publicado en un frame inesperado.')
            return
        datos = np.array(msg.data, dtype=np.int16).reshape((alto, ancho))
        if (self.info_mapa_fuente == msg.info
                and np.array_equal(self.datos_mapa_fuente, datos)):
            return
        self.info_mapa_fuente = deepcopy(msg.info)
        self.datos_mapa_fuente = datos.copy()
        if self.momento_mapa is None:
            # La publicación de /map larga el cronómetro oficial.
            self.momento_mapa = self.get_clock().now()
        info = deepcopy(msg.info)
        factor = self.subdivisiones_mapa
        if factor > 1:
            datos = np.repeat(np.repeat(datos, factor, axis=0), factor, axis=1)
            info.resolution /= factor
            info.width *= factor
            info.height *= factor
        self.objetivo_mision = None
        self.zonas_sin_destino.clear()
        self.puntos_observacion.clear()
        self.observaciones_fallidas.clear()
        self.cache_busquedas.clear()
        self.rondas_observacion.clear()
        self.zonas_inobservables.clear()
        self.cache_zonas_alcanzables.clear()
        ocupado = datos != 0  # pared o desconocido: no confiar

        # Distancia real (en metros) de cada celda a la pared más cercana.
        dist_a_pared = nd.distance_transform_edt(~ocupado) * info.resolution
        self.distancia_a_pared = dist_a_pared
        self.datos_mapa = datos
        self.obstaculos_temporales.clear()
        self.conteo_impactos.clear()
        self.scan_integrado = None
        self.datos_planificacion = datos.copy()

        # Huella rectangular, margen y tamaño de píxel para la orientación actual.
        self.libre_fino = espacio_transitable(
            datos, info.resolution, self.largo_robot, self.ancho_robot,
            self.margen_seguridad + 0.006,
            self.yaw_huella - yaw_de_quaternion(msg.info.origin.orientation))
        self.componentes_libres, _ = nd.label(self.libre_fino)

        self.info_mapa = info
        self.costo_fino = self.costo_por_holgura(
            datos, self.yaw_huella - yaw_de_quaternion(msg.info.origin.orientation))
        self.camino = None
        self.objetivo_celda = None
        self.seguimiento_avance = None
        self.publicar_camino([])
        self.detener()

    def espacio_libre(self, datos, yaw_mapa, extra=0.0):
        """
        Espacio de configuraciones para la huella con orientación `yaw_mapa`.

        Las celdas del mapa usan margen_seguridad; las que solo marcó el scan,
        margen_obstaculos_scan. `extra` se suma a los dos márgenes.
        """
        libres = espacio_transitable(
            self.datos_mapa, self.info_mapa.resolution, self.largo_robot, self.ancho_robot,
            self.margen_seguridad + 0.006 + extra, yaw_mapa)
        temporales = (datos != 0) & (self.datos_mapa == 0)
        if not temporales.any():
            return libres
        return libres & espacio_transitable(
            np.where(temporales, 100, 0), self.info_mapa.resolution, self.largo_robot,
            self.ancho_robot, self.margen_obstaculos_scan + 0.006 + extra, yaw_mapa)

    def costo_por_holgura(self, datos, yaw_mapa):
        """
        Costo de A* según la holgura que le queda a la huella rectangular.

        Mide cuántos centímetros extra entran alrededor del rectángulo del robot
        (no la distancia desde su centro), así el camino va por el medio de
        pasillos y pasos angostos y se arrima a un costado solo si no queda otra.
        """
        holgura = np.zeros(datos.shape)
        for extra in HOLGURAS_COSTO:
            holgura[self.espacio_libre(datos, yaw_mapa, extra)] = extra
        return 100.0 * (1.0 - holgura / HOLGURAS_COSTO[-1]) ** 2

    def actualizar_orientacion_huella(self, yaw):
        """Recalcula el espacio libre si cambia significativamente la orientación."""
        eje = eje_mas_cercano(yaw)
        if abs(normalizar(yaw - eje)) <= TOLERANCIA_EJE:
            yaw = eje
        if abs(normalizar(yaw - self.yaw_huella)) < 0.03:
            return
        self.yaw_huella = yaw
        yaw_mapa = yaw - yaw_de_quaternion(self.info_mapa.origin.orientation)
        self.libre_fino = self.espacio_libre(self.datos_planificacion, yaw_mapa)
        self.componentes_libres, _ = nd.label(self.libre_fino)
        self.costo_fino = self.costo_por_holgura(self.datos_planificacion, yaw_mapa)
        for cache in (self.cache_zonas_alcanzables, self.cache_busquedas,
                      self.puntos_observacion):
            cache.clear()
        self.zonas_sin_destino.clear()
        self.camino = None
        self.publicar_camino([], conservar_objetivo=True)
        self.detener()

    def actualizar_obstaculos(self, pose, impactos):
        """Integra impactos en el mapa de planificación durante ocho segundos."""
        ahora = self.get_clock().now().nanoseconds / 1e9
        self.obstaculos_temporales = {
            celda: instante for celda, instante in self.obstaculos_temporales.items()
            if 0 <= ahora - instante <= 8.0}
        self.conteo_impactos = {celda: self.conteo_impactos.get(celda, 0)
                                for celda in self.obstaculos_temporales}
        if self.scan_integrado is not self.ultimo_scan:
            vistas = set()
            x, y, yaw = pose
            c, s = math.cos(yaw), math.sin(yaw)
            alto, ancho = self.datos_mapa.shape
            for bx, by in impactos:
                if math.hypot(bx, by) > self.alcance_obstaculos_scan:
                    continue
                fila, col = self.celda_de(x + c * bx - s * by,
                                          y + s * bx + c * by, self.info_mapa)
                if (0 <= fila < alto and 0 <= col < ancho
                        and self.datos_mapa[fila, col] == 0
                        and self.distancia_a_pared[fila, col] > self.distancia_pared_conocida):
                    self.obstaculos_temporales[fila, col] = ahora
                    vistas.add((fila, col))
            for celda in vistas:
                self.conteo_impactos[celda] = self.conteo_impactos.get(celda, 0) + 1
            self.scan_integrado = self.ultimo_scan
        datos = self.datos_mapa.copy()
        for celda in self.obstaculos_temporales:
            if self.conteo_impactos.get(celda, 0) >= self.impactos_minimos_scan:
                datos[celda] = 100
        if np.array_equal(datos, self.datos_planificacion):
            return
        self.datos_planificacion = datos
        libres = self.espacio_libre(
            datos, self.yaw_huella - yaw_de_quaternion(self.info_mapa.origin.orientation))
        if np.array_equal(libres, self.libre_fino):
            return
        self.libre_fino = libres
        self.costo_fino = self.costo_por_holgura(
            datos, self.yaw_huella - yaw_de_quaternion(self.info_mapa.origin.orientation))
        self.componentes_libres, _ = nd.label(libres)
        # Los obstáculos del scan aparecen y vencen todo el tiempo: la búsqueda
        # de observación conserva su progreso y comprueba el alcance al usarlo.
        # Las zonas alcanzables dependen de las etiquetas, que cambian.
        self.cache_zonas_alcanzables.clear()
        if self.futuro_plan is not None:
            self.futuro_plan.cancel()
        self.futuro_plan = None
        self.clave_plan = None
        if self.camino and all(libres[celda] for celda in self.camino):
            return
        self.camino = None
        self.seguimiento_avance = None
        self.publicar_camino([], conservar_objetivo=True)
        self.detener()

    def recibir_grilla(self, msg: OccupancyGrid):
        alto, ancho = msg.info.height, msg.info.width
        if (msg.header.frame_id != self.map_frame or alto <= 0 or ancho <= 0
                or not math.isfinite(msg.info.resolution) or msg.info.resolution <= 0
                or len(msg.data) != alto * ancho):
            self.get_logger().error('Grilla de cobertura inválida o en un frame inesperado.')
            return
        datos = np.array(msg.data, dtype=np.int16).reshape((alto, ancho))
        if self.info_grilla != msg.info:
            self.objetivo_mision = None
            self.zonas_sin_destino.clear()
            self.puntos_observacion.clear()
            self.observaciones_fallidas.clear()
            self.cache_busquedas.clear()
            self.rondas_observacion.clear()
            self.zonas_inobservables.clear()
            self.cache_zonas_alcanzables.clear()
        self.no_cubiertas = (datos == 100)
        self.celdas_cubiertas = int(np.count_nonzero(datos == 0))
        self.total_celdas_libres = self.celdas_cubiertas + int(np.count_nonzero(self.no_cubiertas))
        porcentaje = self.porcentaje_cobertura()
        if porcentaje is not None:
            self.pub_porcentaje.publish(Float32(data=porcentaje))
            self.get_logger().info(
                f'Cobertura: {porcentaje:.1f}% / objetivo {self.porcentaje_objetivo:.1f}%.',
                throttle_duration_sec=10.0)
        self.info_grilla = msg.info
        if self.objetivo_celda is not None and self.estado != ESTADO_REGRESANDO:
            f, c = self.objetivo_celda
            if not (0 <= f < alto and 0 <= c < ancho) or not self.no_cubiertas[f, c]:
                self.get_logger().info(
                    f'Zona {self.objetivo_celda} cubierta; se libera el objetivo.')
                self.objetivo_celda = None
                self.seguimiento_avance = None
                self.camino = None
                self.publicar_camino([])

    def recibir_scan(self, msg: LaserScan):
        self.ultimo_scan = msg
        self.momento_scan = self.get_clock().now()
        if self.momento_primer_scan is None:
            self.momento_primer_scan = self.get_clock().now()

    def recibir_pose_inicial(self, msg: PoseWithCovarianceStamped):
        # Reinicia el asentamiento cada vez (incluso si ya se había
        # confirmado antes): si el humano corrige la pose a mitad de
        # recorrido, el robot debe volver a quedarse quieto un momento
        # para que el filtro converja de nuevo antes de seguir moviéndose.
        if msg.header.frame_id and msg.header.frame_id != self.map_frame:
            self.get_logger().warn('Pose inicial ignorada: debe estar en el frame del mapa.')
            return
        self.detener()
        self.camino = None
        self.objetivo_celda = None
        self.publicar_camino([])
        self.objetivos_pendientes.clear()
        self.objetivo_mision = None
        self.obstaculos_temporales.clear()
        self.conteo_impactos.clear()
        self.scan_integrado = None
        if self.futuro_plan is not None:
            self.futuro_plan.cancel()
        self.futuro_plan = None
        self.clave_plan = None
        self.seguimiento_avance = None
        self.zonas_sin_destino.clear()
        self.puntos_observacion.clear()
        self.observaciones_fallidas.clear()
        self.cache_busquedas.clear()
        self.rondas_observacion.clear()
        self.zonas_inobservables.clear()
        self.cache_zonas_alcanzables.clear()
        self.pose_inicial_confirmada = True
        self.momento_confirmacion = self.get_clock().now()
        self.mirada.activa = False
        self.posicion_ultima_mirada = None
        if self.estado != ESTADO_INICIALIZANDO:
            self.estado = ESTADO_INICIALIZANDO
            self.camino = None
            self.get_logger().info(
                f'Transición: * -> {ESTADO_INICIALIZANDO} (pose corregida desde RViz)')

    # ---------- geometría ----------

    def obtener_pose_robot(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return None
        if tf.header.stamp.sec or tf.header.stamp.nanosec:
            edad = (self.get_clock().now() - Time.from_msg(tf.header.stamp)).nanoseconds / 1e9
            if edad > self.timeout_pose_s or edad < -0.1:
                return None
        x = tf.transform.translation.x
        y = tf.transform.translation.y
        yaw = yaw_de_quaternion(tf.transform.rotation)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return None
        return x, y, yaw

    def celda_a_mundo(self, fila: int, col: int, info):
        yaw = yaw_de_quaternion(info.origin.orientation)
        c, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = (col + 0.5) * info.resolution, (fila + 0.5) * info.resolution
        return (info.origin.position.x + c * dx - sn * dy,
                info.origin.position.y + sn * dx + c * dy)

    def celda_de(self, x: float, y: float, info):
        yaw = yaw_de_quaternion(info.origin.orientation)
        c, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = x - info.origin.position.x, y - info.origin.position.y
        col = math.floor((c * dx + sn * dy) / info.resolution)
        fila = math.floor((-sn * dx + c * dy) / info.resolution)
        return fila, col

    def celda_fina_segura(self, x: float, y: float):
        """Devuelve None si el punto queda fuera del mapa fino."""
        fila, col = self.celda_de(x, y, self.info_mapa)
        alto, ancho = self.libre_fino.shape
        if not (0 <= fila < alto and 0 <= col < ancho):
            return None
        return fila, col

    def celda_no_cubierta_mas_cercana(self, fila_r: int, col_r: int, preferidas=None):
        """Conserva un objetivo válido y prioriza nuevas zonas alcanzables."""
        candidatas = self.no_cubiertas.copy()
        for celda in set(self.zonas_sin_destino) | self.zonas_inobservables:
            if (0 <= celda[0] < candidatas.shape[0]
                    and 0 <= celda[1] < candidatas.shape[1]):
                candidatas[celda] = False
        ahora = self.get_clock().now().nanoseconds / 1e9
        for celda, vencimiento in list(self.objetivos_pendientes.items()):
            if ahora >= vencimiento:
                del self.objetivos_pendientes[celda]
            elif (0 <= celda[0] < candidatas.shape[0]
                  and 0 <= celda[1] < candidatas.shape[1]):
                candidatas[celda] = False
        # La cercanía elige un objetivo nuevo; no debe cambiar uno todavía válido.
        if self.objetivo_mision is not None:
            f, c = self.objetivo_mision
            if (0 <= f < candidatas.shape[0] and 0 <= c < candidatas.shape[1]
                    and candidatas[f, c]):
                return self.objetivo_mision
        self.objetivo_mision = None
        directas = candidatas & preferidas if preferidas is not None else candidatas
        if preferidas is not None:
            for f, c in self.observaciones_fallidas:
                if 0 <= f < directas.shape[0] and 0 <= c < directas.shape[1]:
                    directas[f, c] = False
        if np.any(directas):
            candidatas = directas
        elif preferidas is not None:
            # Una ronda completa antes de volver a las mismas zonas difíciles.
            filas, cols = np.where(candidatas)
            if len(filas):
                rondas = np.array([self.rondas_observacion.get((int(f), int(c)), 0)
                                   for f, c in zip(filas, cols)])
                candidatas[filas[rondas > rondas.min()], cols[rondas > rondas.min()]] = False
        filas, cols = np.where(candidatas)

        if len(filas) == 0:
            return None

        dist2 = (filas - fila_r)**2 + (cols - col_r)**2

        i = np.argmin(dist2)

        self.objetivo_mision = (int(filas[i]), int(cols[i]))
        return self.objetivo_mision

    def vector_hacia(self, x_r, y_r, yaw_r, punto_mundo, distancia_restante=None):
        """Orienta la traslación al waypoint y frena según el recorrido restante."""
        # 1. Coordenadas del objetivo
        gx, gy = punto_mundo

        # 2. Vector desde el robot hasta el objetivo
        dx = gx - x_r
        dy = gy - y_r

        # 3. Distancia al objetivo
        dist = math.hypot(dx, dy)

        if dist < 1e-3:
            return (0.0, 0.0, dist)

        # 4. Calcular velocidad proporcional a la distancia
        frenado = dist if distancia_restante is None else distancia_restante
        escala = min(frenado, 1.0) * self.ganancia_atractiva

        dir_x_mundo = (dx / dist) * escala
        dir_y_mundo = (dy / dist) * escala

        # 5. Rotar el vector al frame del robot
        c = math.cos(yaw_r)
        s = math.sin(yaw_r)

        vx = c * dir_x_mundo + s * dir_y_mundo
        vy = -s * dir_x_mundo + c * dir_y_mundo

        # 6. Devolver el vector y la distancia
        return (vx, vy, dist)

    def detener(self):
        """Frena y reinicia el filtro para no reutilizar velocidades anteriores."""
        self.vx_anterior = self.vy_anterior = 0.0
        self.velocidad_ordenada = 0.0
        self.pub_cmd_vel.publish(Twist())

    def scan_vigente(self):
        if self.ultimo_scan is None or self.momento_scan is None:
            return False
        ahora = self.get_clock().now()
        edad = (ahora - self.momento_scan).nanoseconds / 1e9
        if not 0.0 <= edad <= self.timeout_scan_s:
            return False
        stamp = self.ultimo_scan.header.stamp
        if stamp.sec != 0 or stamp.nanosec != 0:
            edad = (ahora - Time.from_msg(stamp)).nanoseconds / 1e9
            if not 0.0 <= edad <= self.timeout_scan_s:
                return False
        return True

    def impactos_en_base(self):
        """Obtiene impactos contemporáneos para comprobar la huella y su barrido."""
        if not self.scan_vigente():
            return None
        try:
            tf = self.tf_buffer.lookup_transform_full(
                self.base_frame, Time(), self.ultimo_scan.header.frame_id,
                Time.from_msg(self.ultimo_scan.header.stamp), self.map_frame)
        except Exception:
            return None
        return puntos_scan(self.ultimo_scan, tf.transform.translation.x,
                           tf.transform.translation.y, yaw_de_quaternion(tf.transform.rotation))

    # ---------- planificación ----------

    def zonas_alcanzables(self, componente):
        """Proyecta las celdas seguras conectadas al robot sobre la cobertura."""
        if componente in self.cache_zonas_alcanzables:
            return self.cache_zonas_alcanzables[componente]
        mascara = np.zeros(self.no_cubiertas.shape, dtype=bool)
        if componente == 0:
            return mascara
        filas, cols = np.where(self.componentes_libres == componente)
        mapa, grilla = self.info_mapa, self.info_grilla
        yaw = yaw_de_quaternion(mapa.origin.orientation)
        dx = (cols + 0.5) * mapa.resolution
        dy = (filas + 0.5) * mapa.resolution
        x = mapa.origin.position.x + math.cos(yaw) * dx - math.sin(yaw) * dy
        y = mapa.origin.position.y + math.sin(yaw) * dx + math.cos(yaw) * dy
        yaw = yaw_de_quaternion(grilla.origin.orientation)
        dx, dy = x - grilla.origin.position.x, y - grilla.origin.position.y
        cc = np.floor((math.cos(yaw) * dx + math.sin(yaw) * dy)
                      / grilla.resolution).astype(int)
        ff = np.floor((-math.sin(yaw) * dx + math.cos(yaw) * dy)
                      / grilla.resolution).astype(int)
        validos = (ff >= 0) & (ff < mascara.shape[0]) & (cc >= 0) & (cc < mascara.shape[1])
        mascara[ff[validos], cc[validos]] = True
        self.cache_zonas_alcanzables[componente] = mascara
        return mascara

    def objetivo_sin_avance(self, objetivo, x, y, distancia_restante=None):
        """Exige progreso hacia el destino; oscilar en el sitio no cuenta como avance."""
        ahora = self.get_clock().now().nanoseconds / 1e9
        anterior = self.seguimiento_avance
        mejora = False
        if anterior is not None:
            if distancia_restante is None:
                mejora = math.hypot(x - anterior[1], y - anterior[2]) >= 0.05
            elif anterior[4] is None:
                mejora = True
            else:
                mejora = distancia_restante < anterior[4] - 0.03
        if anterior is None or anterior[0] != objetivo or mejora:
            self.seguimiento_avance = (objetivo, x, y, ahora, distancia_restante)
            return False
        return ahora - anterior[3] >= self.timeout_sin_avance_s

    def punto_seguro_en_zona(self, zona, componente=None):
        """Busca un centro fino transitable dentro de la celda de cobertura."""
        centro = self.celda_a_mundo(*zona, self.info_grilla)
        fila, col = self.celda_de(*centro, self.info_mapa)
        # El radio cubre incluso grillas rotadas respecto del mapa fino.
        radio = math.ceil(self.info_grilla.resolution / self.info_mapa.resolution) + 1
        alto, ancho = self.libre_fino.shape
        candidatos = []
        for f in range(max(0, fila - radio), min(alto, fila + radio + 1)):
            for c in range(max(0, col - radio), min(ancho, col + radio + 1)):
                if not self.libre_fino[f, c]:
                    continue
                if componente is not None and self.componentes_libres[f, c] != componente:
                    continue
                punto = self.celda_a_mundo(f, c, self.info_mapa)
                if self.celda_de(*punto, self.info_grilla) != zona:
                    continue
                distancia = (punto[0] - centro[0])**2 + (punto[1] - centro[1])**2
                # Primero la holgura de la huella, después la cercanía al centro:
                # el centro de la zona puede quedar pegado a una pared.
                costo = self.costo_fino[f, c] if self.costo_fino is not None else 0.0
                candidatos.append((costo, distancia, punto))
        return min(candidatos)[2] if candidatos else None

    def describir_celda(self, punto, celda):
        """Describe si una posición queda bloqueada por mapa o por margen."""
        posicion = f'({punto[0]:.2f}, {punto[1]:.2f}) m'
        if celda is None:
            return f'{posicion}: fuera del mapa'
        valor = self.datos_mapa[celda]
        distancia = self.distancia_a_pared[celda]
        if valor != 0:
            motivo = f'celda ocupada o desconocida (valor={valor})'
        elif not self.libre_fino[celda]:
            motivo = ('bloqueada por huella rectangular y margen; '
                      f'distancia al muro={distancia:.2f} m')
        else:
            motivo = f'transitable, distancia a obstáculos/desconocido={distancia:.2f} m'
        return f'{posicion}, celda={celda}: {motivo}'

    def punto_para_observar(self, zona, x, y):
        """
        Busca un destino alcanzable con vista a una celda libre de la zona.

        La visibilidad sale del mapa, así que se calcula una sola vez por zona
        lanzando rayos desde la zona; el alcance se comprueba en cada llamada
        porque cambia con los obstáculos del scan. Si devuelve None,
        `motivo_sin_observacion` dice si la zona no se ve desde ningún lado
        ('invisible') o si ahora no se llega a ningún punto que la vea
        ('inalcanzable').
        """
        self.motivo_sin_observacion = 'inalcanzable'
        inicio = self.celda_fina_segura(x, y)
        if inicio is None or not self.libre_fino[inicio]:
            return None
        etiquetas = self.componentes_libres
        alcanzable = etiquetas == etiquetas[inicio]
        if zona in self.puntos_observacion:
            punto = self.puntos_observacion[zona]
            celda = self.celda_fina_segura(*punto)
            if celda is not None and alcanzable[celda]:
                return punto
        alcance = self.ultimo_scan.range_max if self.ultimo_scan is not None else 0.0
        # Se usa el alcance medido del láser, sin suponer visibilidad a través de paredes.
        if not math.isfinite(alcance) or alcance <= 0:
            return None
        clave = (zona, alcance)
        if clave not in self.cache_busquedas:
            self.cache_busquedas[clave] = self.candidatos_de_observacion(zona, alcance)
        candidatos = self.cache_busquedas[clave]
        if not len(candidatos):
            self.motivo_sin_observacion = 'invisible'
            return None
        candidatos = candidatos[alcanzable[candidatos[:, 0], candidatos[:, 1]]]
        if not len(candidatos):
            return None
        fallidos = self.observaciones_fallidas.get(zona, [])
        for f, c in candidatos:
            punto = self.celda_a_mundo(int(f), int(c), self.info_mapa)
            if math.hypot(punto[0] - x, punto[1] - y) < 0.25:
                continue
            # Hay puntos lejanos que la ven: si todos fallaron antes, se descarta.
            self.motivo_sin_observacion = 'invisible'
            if any(math.hypot(punto[0] - px, punto[1] - py) < 0.25 for px, py in fallidos):
                continue
            self.puntos_observacion[zona] = punto
            return punto
        return None

    def candidatos_de_observacion(self, zona, alcance):
        """Celdas libres del mapa con vista a la zona, de la más cercana a la más lejana."""
        centro = self.celda_a_mundo(*zona, self.info_grilla)
        f0, c0 = self.celda_de(*centro, self.info_mapa)
        radio = math.ceil(self.info_grilla.resolution / self.info_mapa.resolution) + 1
        alto, ancho = self.datos_mapa.shape
        blancos = []
        for f in range(max(0, f0 - radio), min(alto, f0 + radio + 1)):
            for c in range(max(0, c0 - radio), min(ancho, c0 + radio + 1)):
                if self.datos_mapa[f, c] != 0:
                    continue
                punto = self.celda_a_mundo(f, c, self.info_mapa)
                if self.celda_de(*punto, self.info_grilla) == zona:
                    blancos.append((f, c))
        if not blancos:
            return np.zeros((0, 2), dtype=np.int64)
        # Unos pocos orígenes repartidos por la zona alcanzan para verla.
        muestras = [blancos[i] for i in np.linspace(0, len(blancos) - 1,
                                                    min(9, len(blancos))).astype(int)]
        # Más allá de la diagonal del mapa los rayos ya salieron: no hace falta seguir.
        largo_rayo = min(alcance / self.info_mapa.resolution, math.hypot(alto, ancho))
        visibles = celdas_visibles(self.datos_mapa == 0, muestras, largo_rayo)
        candidatos = np.argwhere(visibles)
        orden = np.argsort((candidatos[:, 0] - f0)**2 + (candidatos[:, 1] - c0)**2,
                           kind='stable')
        if len(self.cache_busquedas) >= 64:
            self.cache_busquedas.pop(next(iter(self.cache_busquedas)))
        return candidatos[orden]

    def zona_descartada(self, zona):
        """Sin destino conocido o sin punto de observación posible."""
        return zona in self.zonas_sin_destino or zona in self.zonas_inobservables

    def calcular_camino(self, libres, costos, inicio, objetivo):
        """Ejecuta solo cálculo numérico fuera del executor ROS y reutiliza el grafo."""
        clave = (id(libres), id(costos))
        if self.cache_planificador is None or self.cache_planificador[0] != clave:
            self.cache_planificador = (clave, PlanificadorCompilado(libres, costos))
        return self.cache_planificador[1].buscar(inicio, objetivo)

    def replanificar_si_hace_falta(self, objetivo_grueso, objetivo_mundo, x_r, y_r):
        """Calcula rutas fuera del executor ROS y conserva un camino válido."""
        cambio_de_objetivo = objetivo_grueso != self.objetivo_celda
        if cambio_de_objetivo:
            self.get_logger().info(
                f'Nuevo objetivo de cobertura: {objetivo_grueso}; '
                f'destino ({objetivo_mundo[0]:.2f}, {objetivo_mundo[1]:.2f}) m.')
        if not cambio_de_objetivo and self.camino:
            return
        self.objetivo_celda = objetivo_grueso
        celda_robot = self.celda_fina_segura(x_r, y_r)
        celda_objetivo = self.celda_fina_segura(*objetivo_mundo)
        if celda_robot is None or not self.libre_fino[celda_robot]:
            self.get_logger().warn(
                'Planificación detenida: posición estimada del robot '
                + self.describir_celda((x_r, y_r), celda_robot),
                throttle_duration_sec=10.0)
            self.camino = None
            self.publicar_camino([])
            return
        clave = (objetivo_grueso, celda_objetivo, id(self.libre_fino))
        if self.futuro_plan is None or self.clave_plan != clave:
            if self.futuro_plan is not None:
                self.futuro_plan.cancel()
            self.detener()
            self.futuro_plan = self.ejecutor_plan.submit(
                self.calcular_camino, self.libre_fino, self.costo_fino,
                celda_robot, celda_objetivo)
            self.clave_plan = clave
        if not self.futuro_plan.done():
            self.camino = None
            self.detener()
            self.publicar_camino([], conservar_objetivo=True)
            self.get_logger().info('Calculando camino; robot detenido.', throttle_duration_sec=5.0)
            return
        try:
            camino = self.futuro_plan.result()
        except Exception as error:
            self.get_logger().error(f'Falló el cálculo del camino: {error}')
            camino = None
        self.futuro_plan = None
        if camino and not segmento_transitable(self.libre_fino, celda_robot, camino[0]):
            # La localización cambió mientras se calculaba: no seguir un inicio desactualizado.
            self.camino = None
            self.detener()
            self.publicar_camino([], conservar_objetivo=True)
            return
        if camino is None:
            motivo = self.describir_celda(objetivo_mundo, celda_objetivo)
            if celda_objetivo is not None and self.libre_fino[celda_objetivo]:
                motivo += '; A* no encuentra conexión transitable con el robot'
            self.get_logger().warn(
                f'No hay camino hasta {objetivo_grueso}: objetivo {motivo}. '
                f'Robot en ({x_r:.2f}, {y_r:.2f}) m; se reintentará más adelante.',
                throttle_duration_sec=10.0)
            self.objetivos_pendientes[objetivo_grueso] = (
                self.get_clock().now().nanoseconds / 1e9 + self.reintento_objetivo_s)
            self.camino = None
            self.publicar_camino([])
            return

        self.camino = camino
        self.destino_actual = objetivo_mundo
        self.publicar_camino(camino)

    def siguiente_punto_del_camino(self, x_r, y_r):
        """Avanza por waypoints sin tomar atajos que corten esquinas."""
        if not self.camino:
            return None
        tolerancia = max(self.info_mapa.resolution * 3.0, 0.1)
        celda_robot = self.celda_fina_segura(x_r, y_r)
        while len(self.camino) > 1:
            wx, wy = self.celda_a_mundo(*self.camino[0], self.info_mapa)
            if (math.hypot(wx - x_r, wy - y_r) < tolerancia
                    and segmento_transitable(self.libre_fino, celda_robot, self.camino[1])):
                self.camino.pop(0)
            else:
                break
        if not segmento_transitable(self.libre_fino, celda_robot, self.camino[0]):
            self.camino = None
            self.publicar_camino([], conservar_objetivo=True)
            return None
        return self.celda_a_mundo(*self.camino[0], self.info_mapa)

    # ---------- visualización ----------

    def publicar_objetivo(self, objetivo_mundo):
        marcador = Marker()
        marcador.header.frame_id = self.map_frame
        marcador.header.stamp = self.get_clock().now().to_msg()
        marcador.ns = 'objetivo_actual'
        marcador.id = 0
        marcador.type = Marker.SPHERE
        marcador.action = Marker.ADD
        marcador.pose.position.x = objetivo_mundo[0]
        marcador.pose.position.y = objetivo_mundo[1]
        marcador.pose.orientation.w = 1.0
        marcador.scale.x = marcador.scale.y = marcador.scale.z = 0.25
        marcador.color.r = 1.0
        marcador.color.g = 0.6
        marcador.color.a = 1.0
        self.pub_objetivo.publish(marcador)

    def publicar_camino(self, camino, conservar_objetivo=False):
        if not camino and not conservar_objetivo:
            marcador = Marker()
            marcador.header.frame_id = self.map_frame
            marcador.ns = 'objetivo_actual'
            marcador.id = 0
            marcador.action = Marker.DELETE
            self.pub_objetivo.publish(marcador)
        msg = Path()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for fila, col in camino:
            x, y = self.celda_a_mundo(fila, col, self.info_mapa)
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)
        self.pub_camino.publish(msg)

    # ---------- máquina de estados ----------

    def holgura_de_giro(self, x, y):
        """Distancia al obstáculo más cercano, incluidos los temporales del scan."""
        fila, col = self.celda_de(x, y, self.info_mapa)
        alto, ancho = self.datos_planificacion.shape
        if not (0 <= fila < alto and 0 <= col < ancho):
            return None
        distancias = nd.distance_transform_edt(self.datos_planificacion == 0)
        return float(distancias[fila, col]) * self.info_mapa.resolution

    def mirar_alrededor(self, pose):
        """
        Avanza o inicia el giro de 360°; devuelve True si controló el robot.

        Solo se gira donde cabe el círculo que barre la huella, según mapa y
        scan; en un pasillo angosto la mirada queda pendiente.
        """
        x, y, yaw = pose
        ahora = self.get_clock().now().nanoseconds / 1e9
        if self.posicion_ultima_mirada is None:
            self.posicion_ultima_mirada = (x, y)
        if not self.mirada.activa:
            # En línea recta desde la última vuelta: sumar cada paso acumulaba
            # el temblor de la pose y el robot giraba casi sin moverse.
            self.recorrido_desde_mirada = math.dist((x, y), self.posicion_ultima_mirada)
        permitidos = ((ESTADO_EXPLORANDO, ESTADO_REGRESANDO) if self.forzar_mirada
                      else (ESTADO_EXPLORANDO,))
        if self.estado not in permitidos or self.distancia_entre_miradas == 0:
            self.mirada.activa = False
            return False
        if not self.mirada.activa and not self.forzar_mirada and (
                self.recorrido_desde_mirada < self.distancia_entre_miradas
                or self.celdas_cubiertas <= self.cubiertas_en_ultima_mirada):
            return False
        impactos = self.impactos_en_base()
        radio = self.radio_giro - (HOLGURA_MIRADA if self.mirada.activa else 0.0)
        if impactos is None or not giro_seguro(impactos, self.holgura_de_giro(x, y), radio):
            if not self.mirada.activa:
                return False
            if self.momento_bloqueo_mirada is None:
                self.momento_bloqueo_mirada = ahora
            if ahora - self.momento_bloqueo_mirada < ESPERA_MIRADA_S:
                # Quieto, sin reiniciar la vuelta: se retoma donde quedó.
                self.detener()
                self.seguimiento_avance = None
                return True
            # Sin reintentar enseguida: la próxima mirada, después de otro tramo.
            self.mirada.activa = False
            self.momento_bloqueo_mirada = None
            self.recorrido_desde_mirada = 0.0
            self.posicion_ultima_mirada = (x, y)
            self.cubiertas_en_ultima_mirada = self.celdas_cubiertas
            self.seguimiento_avance = None
            self.get_logger().warn(
                'Mirada abandonada: hay algo dentro del radio de giro; '
                'se reintenta después del próximo tramo.')
            return False
        self.momento_bloqueo_mirada = None
        if not self.mirada.activa:
            self.detener()
            self.forzar_mirada = False
            self.mirada.iniciar(yaw, ahora)
            self.get_logger().info('Mirada de 360° por pasos.')
        cmd = Twist()
        cmd.angular.z = self.mirada.actualizar(yaw, ahora)
        if not self.mirada.activa:
            self.recorrido_desde_mirada = 0.0
            self.posicion_ultima_mirada = (x, y)
            self.cubiertas_en_ultima_mirada = self.celdas_cubiertas
            mensajes = {
                'completa': 'Mirada completa.',
                'alineada_por_tiempo': 'Mirada cortada por tiempo; robot alineado con el mapa.',
                'abandonada': 'Mirada abandonada: el robot no llegó a alinearse con el mapa.',
            }
            nivel = self.get_logger().info if self.mirada.resultado == 'completa' \
                else self.get_logger().warn
            nivel(f'{mensajes.get(self.mirada.resultado, "Mirada terminada.")} '
                  f'Orientación final {math.degrees(yaw):.0f}°.')
        # El tiempo girando no cuenta para timeout_sin_avance_s.
        self.seguimiento_avance = None
        self.vx_anterior = self.vy_anterior = 0.0
        self.velocidad_ordenada = 0.0
        self.pub_cmd_vel.publish(cmd)
        return True

    def porcentaje_cobertura(self):
        """Cuenta solo celdas libres; paredes y desconocidos no forman el denominador."""
        if self.total_celdas_libres == 0:
            return None
        return 100.0 * self.celdas_cubiertas / self.total_celdas_libres

    def transicionar(self, pose):
        """Actualiza la misión sin confundir cobertura completa con falta de destinos."""
        anterior = self.estado
        if self.estado == ESTADO_SIN_DESTINOS:
            if (self.no_cubiertas is not None and self.total_celdas_libres > 0
                    and not np.any(self.no_cubiertas)):
                self.estado = ESTADO_REGRESANDO
            elif self.no_cubiertas is not None and any(
                    not self.zona_descartada((int(f), int(c)))
                    for f, c in zip(*np.where(self.no_cubiertas))):
                self.estado = ESTADO_EXPLORANDO
        if self.estado == ESTADO_INICIALIZANDO:

            if (
                pose is not None
                and self.no_cubiertas is not None
                and self.libre_fino is not None
                and self.pose_inicial_confirmada
            ):
                tiempo_transcurrido = (
                    self.get_clock().now() - self.momento_confirmacion
                ).nanoseconds / 1e9

                if tiempo_transcurrido >= self.tiempo_asentamiento_s:
                    if self.origen is None:
                        self.origen = tuple(pose[:2])
                        self.get_logger().info(
                            f'Origen guardado: ({self.origen[0]:.2f}, {self.origen[1]:.2f}) m.')
                    self.estado = (ESTADO_REGRESANDO if self.regreso_pendiente
                                   else ESTADO_EXPLORANDO)

        elif self.estado == ESTADO_EXPLORANDO:

            if (
                self.no_cubiertas is not None
                and self.total_celdas_libres > 0
                and not np.any(self.no_cubiertas)
            ):
                self.estado = ESTADO_REGRESANDO
            elif self.no_cubiertas is not None and all(
                    self.zona_descartada((int(f), int(c)))
                    for f, c in zip(*np.where(self.no_cubiertas))):
                self.estado = ESTADO_SIN_DESTINOS
                self.get_logger().warn(
                    'Quedan zonas sin cubrir, pero no se encontraron más destinos '
                    'ni puntos de observación seguros. No es cobertura completa. '
                    'Se reconsiderarán si cambia el mapa o la grilla.')

        if self.estado == ESTADO_SIN_DESTINOS:
            ahora = self.get_clock().now().nanoseconds / 1e9
            if self.momento_sin_destinos is None:
                self.momento_sin_destinos = ahora
            elif ahora - self.momento_sin_destinos >= ESPERA_SIN_DESTINOS_S:
                self.estado = ESTADO_REGRESANDO
                porcentaje = self.porcentaje_cobertura() or 0.0
                self.get_logger().warn(
                    f'Sin destinos alcanzables: regreso al origen con {porcentaje:.1f}% cubierto.')
        else:
            self.momento_sin_destinos = None
        if self.estado in (ESTADO_EXPLORANDO, ESTADO_SIN_DESTINOS):
            porcentaje = self.porcentaje_cobertura()
            if porcentaje is not None and porcentaje + 1e-9 >= self.porcentaje_objetivo:
                self.estado = ESTADO_REGRESANDO
                self.get_logger().info(
                    f'Objetivo de cobertura alcanzado: {porcentaje:.1f}% '
                    f'>= {self.porcentaje_objetivo:.1f}%.')
        if self.estado in (ESTADO_EXPLORANDO, ESTADO_SIN_DESTINOS) and self.motivo_regreso:
            self.estado = ESTADO_REGRESANDO
            self.get_logger().info(f'Regreso a la largada: {self.motivo_regreso}.')
        if self.estado == ESTADO_REGRESANDO:
            self.regreso_pendiente = True
            if anterior != ESTADO_REGRESANDO:
                if self.futuro_plan is not None:
                    self.futuro_plan.cancel()
                self.futuro_plan = None
                self.clave_plan = None
                self.camino = None
                self.objetivo_celda = None
                self.objetivo_mision = None
                self.seguimiento_avance = None
                self.publicar_camino([])
                self.detener()
            if (pose is not None and self.origen is not None
                    and math.hypot(pose[0] - self.origen[0], pose[1] - self.origen[1]) <= 0.10):
                self.estado = ESTADO_TERMINADO
        if self.estado != anterior:
            self.get_logger().info(
                f'Transición: {anterior} -> {self.estado}'
            )

    def objetivo_de_la_mision(self, fila_r, col_r, preferidas=None):
        """Elige o conserva una zona pendiente de cobertura."""
        return self.celda_no_cubierta_mas_cercana(fila_r, col_r, preferidas)

    def actuar(self, pose):
        """Planifica y ejecuta una traslación comprobada contra mapa y scan."""
        x_r, y_r, yaw_r = pose
        impactos = self.impactos_en_base()
        if impactos is None:
            self.detener()
            return
        self.actualizar_obstaculos(pose, impactos)
        if self.estado == ESTADO_REGRESANDO:
            if self.origen is None:
                self.detener()
                return
            objetivo_grueso = (-1, -1)
            objetivo_mundo = self.origen
            ahora = self.get_clock().now().nanoseconds / 1e9
            if ahora < self.objetivos_pendientes.get(objetivo_grueso, 0.0):
                self.detener()
                return
        else:
            fila_r_gruesa, col_r_gruesa = self.celda_de(x_r, y_r, self.info_grilla)

            celda_robot = self.celda_fina_segura(x_r, y_r)
            componente = (self.componentes_libres[celda_robot]
                          if celda_robot is not None else 0)
            preferidas = self.zonas_alcanzables(componente)
            objetivo_grueso = self.objetivo_de_la_mision(
                fila_r_gruesa, col_r_gruesa, preferidas)
            if objetivo_grueso is None:
                self.camino = None
                self.seguimiento_avance = None
                self.publicar_camino([])
                self.get_logger().warn(
                    'Quedan celdas sin cubrir, pero todos los objetivos están en espera '
                    'de reintento por falta de camino o de avance.',
                    throttle_duration_sec=10.0)
                self.detener()
                return
            # Con un camino vigente hacia esta zona se sigue sin volver a elegir:
            # un cambio pasajero del espacio libre no hace abandonar el objetivo.
            # actualizar_obstaculos borra el camino si deja de ser transitable.
            comprometido = (objetivo_grueso == self.objetivo_celda and bool(self.camino)
                            and self.destino_actual is not None)
            objetivo_mundo = (self.destino_actual if comprometido
                              else self.punto_seguro_en_zona(objetivo_grueso, componente))
            if objetivo_mundo is not None and not comprometido:
                celda_robot = self.celda_fina_segura(x_r, y_r)
                celda_objetivo = self.celda_fina_segura(*objetivo_mundo)
                if (celda_robot is None or not self.libre_fino[celda_robot]
                        or self.componentes_libres[celda_robot]
                        != self.componentes_libres[celda_objetivo]):
                    objetivo_mundo = None
            if objetivo_grueso in self.observaciones_fallidas and not comprometido:
                objetivo_mundo = None
            if objetivo_mundo is None:
                celda_robot = self.celda_fina_segura(x_r, y_r)
                if celda_robot is None or not self.libre_fino[celda_robot]:
                    if self.salir_del_margen(x_r, y_r, yaw_r, impactos):
                        return
                    self.get_logger().warn(
                        'No se buscan puntos de observación: posición del robot bloqueada.',
                        throttle_duration_sec=10.0)
                    self.camino = None
                    self.publicar_camino([])
                    self.detener()
                    return
                if objetivo_grueso not in self.puntos_observacion:
                    self.detener()
                objetivo_mundo = self.punto_para_observar(objetivo_grueso, x_r, y_r)
            if objetivo_mundo is None and self.motivo_sin_observacion == 'inalcanzable':
                intentos = self.rondas_observacion.get(objetivo_grueso, 0) + 1
                self.rondas_observacion[objetivo_grueso] = intentos
                if intentos >= INTENTOS_ZONA_INALCANZABLE:
                    self.motivo_sin_observacion = 'invisible'
            if objetivo_mundo is None and self.motivo_sin_observacion == 'inalcanzable':
                # Hay lugares que la ven, pero ahora no se llega (por ejemplo,
                # el scan encierra al robot): reintentar, no descartar.
                self.objetivos_pendientes[objetivo_grueso] = (
                    self.get_clock().now().nanoseconds / 1e9 + self.reintento_objetivo_s)
                celda_r = self.celda_fina_segura(x_r, y_r)
                area = (np.count_nonzero(
                    self.componentes_libres == self.componentes_libres[celda_r])
                    * self.info_mapa.resolution ** 2)
                self.get_logger().info(
                    f'Zona {objetivo_grueso}: ningún punto que la ve es alcanzable ahora '
                    f'(área alcanzable {area:.2f} m², {len(self.obstaculos_temporales)} celdas '
                    f'del scan, robot en ({x_r:.2f}, {y_r:.2f})); se reintenta más tarde.',
                    throttle_duration_sec=10.0)
                self.camino = None
                self.publicar_camino([])
                self.detener()
                return
            if objetivo_mundo is None:
                self.zonas_sin_destino.add(objetivo_grueso)
                self.zonas_inobservables.add(objetivo_grueso)
                self.get_logger().warn(
                    f'Zona {objetivo_grueso} sin punto de observación alcanzable y visible; '
                    'se excluye como destino hasta actualizar el mapa. Sigue sin cubrir.')
                self.camino = None
                self.publicar_camino([])
                self.detener()
                return
        self.publicar_objetivo(objetivo_mundo)

        self.replanificar_si_hace_falta(objetivo_grueso, objetivo_mundo, x_r, y_r)

        punto_seguimiento = self.siguiente_punto_del_camino(x_r, y_r)
        if punto_seguimiento is None:
            self.detener()
            return
        distancia_restante = math.hypot(
            punto_seguimiento[0] - x_r, punto_seguimiento[1] - y_r)
        distancia_restante += sum(
            math.hypot(a[0] - b[0], a[1] - b[1]) * self.info_mapa.resolution
            for a, b in zip(self.camino, self.camino[1:]))
        if self.objetivo_sin_avance(objetivo_grueso, x_r, y_r, distancia_restante):
            self.get_logger().warn(
                f'Objetivo {objetivo_grueso} sin avance durante '
                f'{self.timeout_sin_avance_s:.0f} s; se reintentará la planificación. '
                'Se recalculará el camino.')
            self.objetivos_pendientes[objetivo_grueso] = (
                self.get_clock().now().nanoseconds / 1e9 + self.reintento_objetivo_s)
            if self.estado != ESTADO_REGRESANDO:
                self.observaciones_fallidas.setdefault(objetivo_grueso, []).append(objetivo_mundo)
                self.puntos_observacion.pop(objetivo_grueso, None)
            self.seguimiento_avance = None
            self.camino = None
            self.publicar_camino([])
            self.detener()
            return
        vx_atr, vy_atr, _ = self.vector_hacia(
            x_r, y_r, yaw_r, punto_seguimiento, distancia_restante)
        impactos = self.impactos_en_base()
        if impactos is None:
            self.get_logger().warn(
                'Frenado: scan inválido, vencido o sin transformación a la base.',
                throttle_duration_sec=10.0)
            self.detener()
            return
        vx, vy = vx_atr, vy_atr
        magnitud = math.hypot(vx, vy)
        if magnitud > self.velocidad_maxima:
            vx *= self.velocidad_maxima / magnitud
            vy *= self.velocidad_maxima / magnitud

        vx = self.suavizado * vx + (1.0 - self.suavizado) * self.vx_anterior
        vy = self.suavizado * vy + (1.0 - self.suavizado) * self.vy_anterior
        celda_inicio = self.celda_fina_segura(x_r, y_r)
        c, sn = math.cos(yaw_r), math.sin(yaw_r)

        def mapa_seguro(dx, dy):
            fin = self.celda_fina_segura(x_r + c * dx - sn * dy, y_r + sn * dx + c * dy)
            return segmento_transitable(self.libre_fino, celda_inicio, fin)

        # Las paredes del mapa ya tienen margen_seguridad en mapa_seguro; contra
        # el scan se usa el mismo margen que el planificador da a las cajas, o
        # el robot frena en caminos que el planificador sí considera libres.
        pedida = (vx, vy)
        (vx, vy), motivo = regular_traslacion(
            pedida, impactos, self.largo_robot, self.ancho_robot,
            self.margen_obstaculos_scan, self.velocidad_maxima, mapa_seguro,
            horizonte=self.horizonte_colision_s, desaceleracion=self.desaceleracion_segura,
            velocidad_pasillo=self.velocidad_pasillo)
        if motivo == 'obstaculo_o_margen':
            # Con la pose corrida, el camino del mapa puede rozar una pared real:
            # en vez de quedarse frenado, desviarse hacia donde el scan está libre.
            desvio = buscar_desvio(
                pedida, impactos, self.largo_robot, self.ancho_robot,
                self.margen_obstaculos_scan, self.velocidad_pasillo,
                horizonte=self.horizonte_colision_s, desaceleracion=self.desaceleracion_segura)
            if desvio is not None:
                (vx, vy), grados = desvio
                self.get_logger().info(
                    f'Desvío de {grados}° para no rozar lo que ve el scan.',
                    throttle_duration_sec=5.0)
            else:
                self.get_logger().warn(
                    'Frenado preventivo: el barrido de la huella no tiene espacio libre.',
                    throttle_duration_sec=5.0)
        self.vx_anterior, self.vy_anterior = vx, vy

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        # Corrige la deriva del giro para seguir alineado con el mapa; desvíos
        # mayores (una mirada abandonada) no se corrigen girando junto a paredes.
        error = normalizar(eje_mas_cercano(yaw_r) - yaw_r)
        if abs(error) <= TOLERANCIA_EJE:
            cmd.angular.z = max(-0.2, min(0.2, 1.0 * error))
        self.velocidad_ordenada = math.hypot(vx, vy)
        self.pub_cmd_vel.publish(cmd)

    def salir_del_margen(self, x_r, y_r, yaw_r, impactos):
        """
        Aleja el robot hacia la celda libre más cercana si quedó dentro del margen.

        Sin esto el explorador se queda detenido: desde una celda no transitable
        no busca puntos de observación ni planifica. Devuelve False si no hay
        celda libre cerca o si el paso acercaría el chasis a un impacto del scan.
        """
        celda = self.celda_fina_segura(x_r, y_r)
        if celda is None:
            return False
        resolucion = self.info_mapa.resolution
        destino = celda_libre_cercana(
            self.libre_fino, celda, math.ceil(self.radio_escape / resolucion))
        if destino is None:
            return False
        # Celdas -> metros en el frame del mapa y luego en base_footprint.
        yaw_mapa = yaw_de_quaternion(self.info_mapa.origin.orientation)
        d_col, d_fila = destino[1] - celda[1], destino[0] - celda[0]
        dx_m = (math.cos(yaw_mapa) * d_col - math.sin(yaw_mapa) * d_fila) * resolucion
        dy_m = (math.sin(yaw_mapa) * d_col + math.cos(yaw_mapa) * d_fila) * resolucion
        distancia = math.hypot(dx_m, dy_m)
        if distancia < 1e-6:
            return False
        c, sn = math.cos(yaw_r), math.sin(yaw_r)
        ux, uy = (c * dx_m + sn * dy_m) / distancia, (-sn * dx_m + c * dy_m) / distancia
        if not escape_seguro(impactos, self.largo_robot, self.ancho_robot,
                             (0.02 * ux, 0.02 * uy)):
            return False
        self.get_logger().warn(
            f'Robot dentro del margen de una pared: me alejo {100 * distancia:.0f} cm '
            'hacia la celda libre más cercana.', throttle_duration_sec=5.0)
        self.camino = None
        self.publicar_camino([])
        cmd = Twist()
        cmd.linear.x = self.velocidad_escape * ux
        cmd.linear.y = self.velocidad_escape * uy
        self.vx_anterior, self.vy_anterior = cmd.linear.x, cmd.linear.y
        self.velocidad_ordenada = self.velocidad_escape
        self.pub_cmd_vel.publish(cmd)
        return True

    # ---------- control ----------

    def recibir_victimas_completas(self, _msg):
        if self.regresar_con_todas and not self.motivo_regreso:
            self.motivo_regreso = 'todas las víctimas reportadas'

    def revisar_reloj(self):
        """Vuelve a la largada al vencer tiempo_regreso_s y evita quedar inmóvil."""
        ahora = self.get_clock().now()
        if (self.tiempo_regreso_s > 0 and self.momento_mapa is not None
                and not self.motivo_regreso):
            transcurrido = (ahora - self.momento_mapa).nanoseconds / 1e9
            if transcurrido >= self.tiempo_regreso_s:
                self.motivo_regreso = f'{transcurrido:.0f} s desde la publicación de /map'
        if self.velocidad_ordenada > 0.01 or self.mirada.activa or self.momento_movimiento is None:
            self.momento_movimiento = ahora
        elif (self.estado in (ESTADO_EXPLORANDO, ESTADO_REGRESANDO) and self.espera_inmovil_s > 0
              and (ahora - self.momento_movimiento).nanoseconds / 1e9 >= self.espera_inmovil_s):
            if not self.forzar_mirada:
                self.get_logger().warn(
                    f'{self.espera_inmovil_s:.0f} s sin moverse: mirada para no quedar inmóvil.')
            self.forzar_mirada = True

    def controlar(self):
        pose = self.obtener_pose_robot()
        self.revisar_reloj()
        # Modo competencia: cuando hay mapa y el LiDAR lleva 5 s andando,
        # "hacemos el clic" de 2D Pose Estimate en el origen, una sola vez.
        if (self.confirmar_pose_automatica and not self.pose_auto_enviada
                and self.libre_fino is not None
                and self.momento_primer_scan is not None
                and (self.get_clock().now() - self.momento_primer_scan).nanoseconds >= 5e9):
            clic = PoseWithCovarianceStamped()
            clic.header.frame_id = self.map_frame
            clic.header.stamp = self.get_clock().now().to_msg()
            clic.pose.pose.orientation.w = 1.0   # mirando hacia +x
            self.pub_pose_inicial.publish(clic)
            self.pose_auto_enviada = True
            self.get_logger().info(
                'Todo cargado: mando la pose inicial (0, 0, 0) como el clic de RViz.')

        # Durante la mirada el yaw recorre una vuelta y vuelve al inicial.
        if pose is not None and self.info_mapa is not None and not self.mirada.activa:
            self.actualizar_orientacion_huella(pose[2])
        estado_anterior = self.estado
        self.transicionar(pose)
        if self.estado == ESTADO_TERMINADO and not self.done_publicado:
            self.pub_done.publish(Empty())
            self.done_publicado = True
            self.get_logger().info('En la largada: publico /done.')
        if self.estado == ESTADO_TERMINADO and estado_anterior != ESTADO_TERMINADO:
            if self.futuro_plan is not None:
                self.futuro_plan.cancel()
            self.futuro_plan = None
            self.camino = None
            self.objetivo_celda = None
            self.objetivo_mision = None
            self.publicar_camino([])
        self.pub_estado.publish(String(data=self.estado))

        if self.estado in (
                ESTADO_INICIALIZANDO, ESTADO_TERMINADO, ESTADO_SIN_DESTINOS) or pose is None:
            if pose is None and self.estado == ESTADO_EXPLORANDO:
                self.get_logger().warn(
                    'Frenado: no está disponible la transformación map -> base del robot.',
                    throttle_duration_sec=10.0)
            self.detener()
            return

        if not self.scan_vigente():
            self.get_logger().warn(
                'Frenado: no llega scan o supera timeout_scan_s.',
                throttle_duration_sec=10.0)
            self.mirada.activa = False
            self.detener()
            return
        if self.mirar_alrededor(pose):
            return
        self.actuar(pose)


def main(args=None):
    rclpy.init(args=args)
    nodo = Explorador()
    try:
        rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if nodo.context.ok():
                try:
                    nodo.pub_cmd_vel.publish(Twist())
                except rclpy_implementation.RCLError:
                    # Una señal puede apagar ROS entre la comprobación y el publish.
                    if nodo.context.ok():
                        raise
        finally:
            nodo.ejecutor_plan.shutdown(wait=False, cancel_futures=True)
            nodo.destroy_node()
            rclpy.try_shutdown()


if __name__ == '__main__':
    main()
