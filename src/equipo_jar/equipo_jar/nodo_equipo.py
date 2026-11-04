import numpy as np
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PointStamped, PoseArray
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image, LaserScan

from equipo_jar.core import Board, DetectionClusterer, normalize
from equipo_jar.detector import ProcesadorVision
from equipo_jar.localizer import MAP_QOS, node_params, seconds, yaw_of
from equipo_jar.reportes import armar_reporte

UMBRALES = ('hue_rojo_bajo_1', 'hue_rojo_alto_1', 'hue_rojo_bajo_2', 'hue_rojo_alto_2', 'saturacion_min', 'valor_min', 'mitad_caja', 'erosion_px', 'salto_max', 'rayos_min')

class NodoEquipo(Node):
    def __init__(self):
        super().__init__('nodo_equipo', automatically_declare_parameters_from_overrides=True)
        p = node_params(self) # Los valores están solo en params.yaml (sección nodo_equipo), comentados

        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N): nunca se reporta más
        self.cantidad_victimas = int(p['cantidad_victimas'])

        # Detector: umbrales HSV del rojo, tamaño de la caja y filtros de paralaje
        self.procesador_vision = ProcesadorVision(**{k: float(p[k]) for k in UMBRALES})
        self.deskew = p['deskew'] # orden temporal de los rayos, como en el localizer; 0: todos en el instante medio
        self.pared_min = p['pared_min'] # m: un rayo que pegó a esto o menos de una pared del mapa es pared, no caja (0: en una celda de pared)
        self.rayos_reporte = int(p['rayos_reporte']) # rayos rojos sumados para reportar una caja
        self.observaciones_min = int(p['observaciones_min']) # en al menos estos scans
        self.dispersion_max = p['dispersion_max'] # m: grupos más dispersos no se reportan
        self.observaciones_rojo = int(p['observaciones_rojo']) # para avisarle al navegador que ahí hay rojo (/jar/rojo)
        self.desfase_foto_max = p['desfase_foto_max'] # s: sin foto tan cerca del scan, el scan no se usa
        self.espera_max = p['espera_max'] # s: un scan espera hasta esto a que lleguen la TF de su instante y su foto; después se descarta

        self.clusterer = DetectionClusterer(p['radio_grupo'])

        self.puente_cv = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_rojo = self.create_publisher(PoseArray, '/jar/rojo', 10) # interno: dónde ve rojo sin reportar todavía (para el navegador)

        # Escuchamos
        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, MAP_QOS)
        self.create_subscription(CameraInfo, '/cam_1/color/camera_info', self.al_recibir_info_camara, qos_profile_sensor_data)
        self.create_subscription(Image, '/cam_1/color/image_raw', self.al_recibir_imagen, qos_profile_sensor_data)
        self.create_subscription(LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)

        # Estado inicial
        self.mapa = self.info_camara = None
        self.pendientes = [] # scans esperando la TF de su instante o su foto
        self.fotos = [] # las de los últimos espera_max + desfase_foto_max s (por stamp): de acá sale la más cercana a cada scan
        self.reportes = 0
        self.get_logger().info(f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    # ---------- entradas ----------
    def al_recibir_mapa(self, mapa):
        if self.mapa is not None: return

        self.mapa = Board.from_msg(mapa)
        self.get_logger().info(f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, resolución {mapa.info.resolution:.3f} m')

    def al_recibir_info_camara(self, msg):
        self.info_camara = (msg.k[0], msg.k[4], msg.k[2], msg.k[5]) # fx, fy, cx, cy

    def al_recibir_imagen(self, msg):
        t = seconds(msg.header.stamp)
        self.fotos = [f for f in self.fotos if t - seconds(f.header.stamp) < self.espera_max + self.desfase_foto_max] + [msg] # se convierte a OpenCV solo la que se usa
        self.procesar_pendientes()

    def al_recibir_scan(self, msg):
        """
        Encola el scan y procesa los que ya se pueden. La TF de su último rayo y la foto llegan después que el scan (TF no extrapola), y con el software en
        otra compu, más tarde todavía. Antes se procesaba el scan anterior al llegar el siguiente y, si faltaba algo, se perdía: con la TF 0.14 s más
        atrasada, o las fotos 0.3 s, no se reportaba nada (experimentos/robustez/RESULTADOS.md).
        """
        if self.mapa is None or self.info_camara is None: return

        self.pendientes.append(msg)
        self.procesar_pendientes()
        self.publicar_rojo()

    def procesar_pendientes(self):
        """Procesa los scans pendientes que ya tienen la TF de su instante y su foto. Descarta los que quedaron más de espera_max detrás del más nuevo."""
        if not self.pendientes: return

        ultimo, quedan = seconds(self.pendientes[-1].header.stamp), []
        for scan in self.pendientes:
            try: listo = self.procesar(scan)
            except tf2_ros.TransformException: listo = False # todavía no llegó la TF de su instante (o no hay pose: antes de localizarse)
            except Exception as e:
                self.get_logger().error(f'percepción: {e!r}', throttle_duration_sec=1.0)
                listo = True
            if not listo and ultimo - seconds(scan.header.stamp) < self.espera_max: quedan.append(scan)
        self.pendientes = quedan

    # ---------- percepción ----------
    def procesar(self, scan):
        """
        Detecciones de un scan -> map -> DetectionClusterer -> reportes. False si tiene que esperar una foto (todavía no llegó ninguna posterior al scan, y
        la más cercana puede ser esa); tf2_ros.TransformException si falta la TF de su instante.

        Usa la foto más cercana al instante medio del scan y lleva cada rayo a ese instante: le suma a su ángulo lo que giró el lidar entre que lo midió y la foto
        (en el robot real los rayos se reparten en 0.13 s; girando, foto y scan se desfasan unos grados). El avance en ese lapso (< 3 cm) no se corrige.
        """
        n, dt = len(scan.ranges), scan.time_increment
        t0 = Time.from_msg(scan.header.stamp)
        t_medio = t0 + Duration(seconds=dt * (n - 1) / 2)
        if not self.fotos or Time.from_msg(self.fotos[-1].header.stamp) < t_medio: return False # puede llegar una foto más cerca
        foto = min(self.fotos, key=lambda f: abs((Time.from_msg(f.header.stamp) - t_medio).nanoseconds))
        t_foto = Time.from_msg(foto.header.stamp)
        if abs((t_foto - t_medio).nanoseconds) * 1e-9 > self.desfase_foto_max: return True # sin foto cerca del scan

        # Giro del lidar en odom al empezar y terminar el scan y en la foto -> ángulo de cada rayo en el instante de la foto
        laser = scan.header.frame_id
        tf = lambda destino, t: self.tf_buffer.lookup_transform(destino, laser, t).transform
        a, b, c = (yaw_of(tf('odom', t).rotation) for t in (t0, t0 + Duration(seconds=dt * (n - 1)), t_foto))
        m = tf('map', t_foto) # antes del detector: sin pose todavía, el reintento no procesa la foto
        k = np.arange(n)
        fraccion = 0.5 + self.deskew * (k - (n - 1) / 2) / max(n - 1, 1) # 0 al empezar el scan, 1 al terminar
        angulos = scan.angle_min + k * scan.angle_increment + normalize(a + fraccion * normalize(b - a) - c)

        # Detector: rayos rojos -> centros de caja en el frame del lidar
        cam = self.tf_buffer.lookup_transform(foto.header.frame_id, laser, Time()).transform # montaje: fijo
        q, v = cam.rotation, cam.translation
        centros = self.procesador_vision.extraer_coordenadas_cajas(
            self.puente_cv.imgmsg_to_cv2(foto, desired_encoding='bgr8'), np.asarray(scan.ranges), angulos,
            Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix(), np.array([v.x, v.y, v.z]), *self.info_camara)
        if not centros: return True

        # A map. El rayo pegó media caja antes del centro: si ahí hay una pared del mapa, era la pared (luz cálida, paralaje)
        th = yaw_of(m.rotation)
        rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        centros = np.asarray(centros)
        golpes = centros * (1 - self.procesador_vision.mitad_caja / np.hypot(*centros.T))[:, None]
        centros, golpes = (pts @ rot.T + (m.translation.x, m.translation.y) for pts in (centros, golpes))
        centros = centros[self.mapa.distance(golpes[:, 0], golpes[:, 1]) > self.pared_min]

        # Una observación por caja y por scan (la mediana de sus rayos) con cuántos rayos tuvo: ready() pide rayos y scans
        en_scan = DetectionClusterer(self.clusterer.radius)
        for x, y in centros: en_scan.add(x, y)
        for g in en_scan.groups: self.clusterer.add(*en_scan.center(g), len(g['obs']))

        self.reportar_listas()

        return True

    def reportar_listas(self):
        """Reporta cada caja que ready() da por buena y no se reportó, mientras queden reportes. Cada reporte gasta uno de los N (§7.2)."""
        for g in self.clusterer.groups:
            if self.reportes >= self.cantidad_victimas: return
            if g['reported'] or not self.clusterer.ready(g, self.rayos_reporte, self.observaciones_min, self.dispersion_max): continue

            x, y = self.clusterer.center(g)
            self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))
            g['reported'], self.reportes = True, self.reportes + 1
            self.get_logger().info(f'víctima {self.reportes}/{self.cantidad_victimas} en ({x:.2f}, {y:.2f}): {len(g["obs"])} observaciones, '
                                   f'{sum(o[2] for o in g["obs"])} rayos rojos')

    def publicar_rojo(self):
        """
        Publica en /jar/rojo los grupos sin reportar con observaciones_rojo observaciones o más: el navegador sigue mirando un candidato con rojo cerca hasta el reporte y
        descarta en ~1 s uno sin rojo (sin esto, esperaba 3.5 s en cada señuelo).
        """
        msg = PoseArray()
        msg.header.frame_id, msg.header.stamp = 'map', self.get_clock().now().to_msg()
        for g in self.clusterer.groups:
            if g['reported'] or len(g['obs']) < self.observaciones_rojo: continue

            pose = Pose()
            pose.position.x, pose.position.y = self.clusterer.center(g)
            msg.poses.append(pose)
        self.pub_rojo.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    nodo = NodoEquipo()

    try: rclpy.spin(nodo)
    except (KeyboardInterrupt, ExternalShutdownException): pass
    except Exception:
        if rclpy.ok(): raise # con ROS ya apagado (Ctrl+C, el launch), spin puede tirar RCLError al armar la espera: no es un error
    finally:
        nodo.destroy_node()
        if rclpy.ok(): rclpy.shutdown()

if __name__ == '__main__':
    main()
